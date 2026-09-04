"""Run the hybrid planner + UI-Venus-Ground-72B agent on AndroidWorld.

Planner  : GPT-5.1 (OpenAI), Gemini 3.1 Pro (GCP), or Claude.
Grounder : inclusionAI/UI-Venus-Ground-72B served via OpenAI-compatible
           endpoint (e.g. vLLM).

Example:
    # Start grounder.
    vllm serve inclusionAI/UI-Venus-Ground-72B \\
        --host 0.0.0.0 --port 8001 \\
        --served-model-name ui-venus-gd \\
        --tensor-parallel-size 4 \\
        --limit-mm-per-prompt '{"image": 1}'

    # Run with GPT-5.1 planner.
    OPENAI_API_KEY=sk-... python benchmark/run_m3a_venus.py \\
        --planner=gpt --gpt_model_name=gpt-5.1 \\
        --grounder_url=http://127.0.0.1:8001 \\
        --grounder_model=ui-venus-gd

    # Run with Gemini 3.1 Pro planner.
    GCP_API_KEY=... python benchmark/run_m3a_venus.py \\
        --planner=gemini --gemini_model_name=gemini-3.1-pro-preview \\
        --grounder_url=http://127.0.0.1:8001 \\
        --grounder_model=ui-venus-gd
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from absl import app, flags, logging

try:
  import pysqlite3  # type: ignore
  sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except Exception:  # pylint: disable=broad-exception-caught
  pass

from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import infer as infer_module
from android_world.agents import m3a_venus
from android_world.env import env_launcher


logging.set_verbosity(logging.WARNING)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ.pop("GRPC_TRACE", None)


def _find_adb_directory() -> str:
  potential_paths = [
      "/opt/android/platform-tools/adb",
      os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
      os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
      os.path.expanduser("~/android-sdk/platform-tools/adb"),
      os.path.expanduser("~/androidsdk/platform-tools/adb"),
  ]
  for path in potential_paths:
    if os.path.isfile(path):
      return path
  raise EnvironmentError(
      "adb not found. Install Android SDK so that platform-tools/adb exists.")


# -- flags -----------------------------------------------------------------

_ADB_PATH = flags.DEFINE_string("adb_path", _find_adb_directory(), "Path to adb.")
_EMULATOR_SETUP = flags.DEFINE_boolean(
    "perform_emulator_setup", False, "Run emulator setup (one-time).")
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    "console_port", 5554, "Emulator console port.")
_GRPC_PORT = flags.DEFINE_integer(
    "grpc_port", 8554, "Emulator gRPC port.")

_SUITE_FAMILY = flags.DEFINE_enum(
    "suite_family",
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    "Suite family to run.",
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    "task_random_seed", 30, "Random seed.")
_TASKS = flags.DEFINE_list("tasks", None, "Optional subset of tasks.")
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    "n_task_combinations", 1, "Instances per task template.")
_FIXED_TASK_SEED = flags.DEFINE_boolean(
    "fixed_task_seed", False, "Use identical params across combinations.")
_CHECKPOINT_DIR = flags.DEFINE_string("checkpoint_dir", "", "Resume dir.")
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", os.path.expanduser("~/android_world_runs/m3a_venus"),
    "Output root if checkpoint_dir is empty.")
_MAX_STEPS = flags.DEFINE_integer(
    "max_steps", None, "Max steps per task (defaults to task complexity * 10).")
_WAIT_AFTER_ACTION = flags.DEFINE_float(
    "wait_after_action_seconds", 2.0, "Seconds to wait after each action.")

# Planner flags.
_PLANNER = flags.DEFINE_enum(
    "planner", "gpt", ["gpt", "gemini", "claude"],
    "Planner backend: 'gpt' (OpenAI), 'gemini' (GCP genai), or 'claude'.")
_GPT_MODEL_NAME = flags.DEFINE_string(
    "gpt_model_name", "gpt-5.1",
    "OpenAI model id for the planner (e.g. gpt-5.1).")
_GPT_BASE_URL = flags.DEFINE_string(
    "gpt_base_url", "https://api.openai.com/v1/chat/completions",
    "OpenAI-compatible chat completions URL.")
_GEMINI_MODEL_NAME = flags.DEFINE_string(
    "gemini_model_name", "gemini-3.1-pro-preview",
    "GCP genai model id for the planner (e.g. gemini-3.1-pro-preview).")
_CLAUDE_MODEL_NAME = flags.DEFINE_string(
    "claude_model_name", "claude-sonnet-4-6",
    "Claude model id for the planner.")
_CLAUDE_PROXY_BASE_URL = flags.DEFINE_string(
    "claude_proxy_base_url", "",
    "OpenAI-compatible proxy URL for Claude planner requests.")

# Grounder flags.
_GROUNDER_URL = flags.DEFINE_string(
    "grounder_url", "http://127.0.0.1:8001",
    "Base URL of OpenAI-compatible endpoint serving UI-Venus-Ground-72B.")
_GROUNDER_MODEL = flags.DEFINE_string(
    "grounder_model", "ui-venus-gd",
    "--served-model-name used when serving UI-Venus.")
_GROUNDER_ENDPOINT_FORMAT = flags.DEFINE_enum(
    "grounder_endpoint_format", "openai", ["openai", "predict"],
    "UI-Venus grounder protocol: OpenAI /v1 or CATBench /predict proxy.")
_GROUNDER_API_KEY = flags.DEFINE_string(
    "grounder_api_key", os.environ.get("UI_VENUS_API_KEY", "EMPTY"),
    "API key for the grounder endpoint (usually EMPTY for self-hosted vLLM).")


def _build_planner() -> infer_module.MultimodalLlmWrapper:
  if _PLANNER.value == "gpt":
    return infer_module.Gpt4Wrapper(
        model_name=_GPT_MODEL_NAME.value,
        base_url=_GPT_BASE_URL.value,
    )
  if _PLANNER.value == "gemini":
    return infer_module.GeminiGcpWrapper(
        model_name=_GEMINI_MODEL_NAME.value,
    )
  if _PLANNER.value == "claude":
    return infer_module.ClaudeProxyWrapper(
        model_name=_CLAUDE_MODEL_NAME.value,
        base_url=_CLAUDE_PROXY_BASE_URL.value,
    )
  raise ValueError(f"Unknown planner: {_PLANNER.value}")


def _main() -> None:
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      grpc_port=_GRPC_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
  )

  task_registry = registry.TaskRegistry()
  suite = suite_utils.create_suite(
      task_registry.get_registry(family=_SUITE_FAMILY.value),
      n_task_combinations=_N_TASK_COMBINATIONS.value,
      seed=_TASK_RANDOM_SEED.value,
      tasks=_TASKS.value,
      use_identical_params=_FIXED_TASK_SEED.value,
  )
  suite.suite_family = _SUITE_FAMILY.value

  planner = _build_planner()
  if _PLANNER.value == "gpt":
    planner_model = _GPT_MODEL_NAME.value
  elif _PLANNER.value == "gemini":
    planner_model = _GEMINI_MODEL_NAME.value
  else:
    planner_model = _CLAUDE_MODEL_NAME.value
  print(
      f"Planner: {_PLANNER.value} "
      f"(model={planner_model})"
  )

  grounder = m3a_venus.UIVenusGrounder(
      base_url=_GROUNDER_URL.value,
      model_name=_GROUNDER_MODEL.value,
      api_key=_GROUNDER_API_KEY.value,
      endpoint_format=_GROUNDER_ENDPOINT_FORMAT.value,
  )
  print(
      f"Grounder: UI-Venus @ {_GROUNDER_URL.value} "
      f"(model={_GROUNDER_MODEL.value}, "
      f"format={_GROUNDER_ENDPOINT_FORMAT.value})"
  )

  agent = m3a_venus.M3AVenusAgent(
      env,
      planner_llm=planner,
      grounder=grounder,
      wait_after_action_seconds=_WAIT_AFTER_ACTION.value,
  )
  if _MAX_STEPS.value is not None:
    agent.max_steps = _MAX_STEPS.value

  checkpoint_dir = (
      _CHECKPOINT_DIR.value
      or checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)
  )
  print(f"Running M3A+UI-Venus on {_SUITE_FAMILY.value}; writing to {checkpoint_dir}")
  suite_utils.run(
      suite, agent,
      checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
      demo_mode=False,
  )
  print(f"Finished. Results at {checkpoint_dir}.")
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == "__main__":
  app.run(main)
