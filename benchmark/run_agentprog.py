#!/usr/bin/env python3
"""Run MobileLLM/AgentProg on AndroidWorld/CATBench tasks."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence

try:
  import pysqlite3  # type: ignore
  import sys
  sys.modules["sqlite3"] = pysqlite3
except ImportError:
  pass

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import agentprog as agentprog_agent
from android_world.env import env_launcher


logging.set_verbosity(logging.WARN)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = "none"


def _first_writable_dir(candidates: list[str]) -> str:
  for path in candidates:
    expanded = os.path.expanduser(path)
    try:
      os.makedirs(expanded, exist_ok=True)
      test_path = os.path.join(expanded, ".write_test")
      with open(test_path, "w", encoding="utf-8") as handle:
        handle.write("ok")
      os.remove(test_path)
      return expanded
    except OSError:
      continue
  return "/tmp"


def _default_runs_root() -> str:
  user = os.environ.get("USER", "ttran")
  return _first_writable_dir([
      os.environ.get("CATBENCH_RUNS_DIR", ""),
      f"$HOME/android_world_runs",
      f"$HOME/{user}/android_world_runs",
      "~/catbench_runs",
      "/tmp/catbench_runs",
  ])


def _env_int(name: str, default: int) -> int:
  value = os.environ.get(name, "")
  return int(value) if value else default


def _find_adb_directory() -> str:
  adb = shutil.which("adb")
  if adb:
    return adb
  potential_paths = [
      os.path.join(os.environ.get("ANDROID_SDK_ROOT", ""), "platform-tools", "adb"),
      os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
      os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
  ]
  for path in potential_paths:
    if path and os.path.isfile(path):
      return path
  raise EnvironmentError("adb not found. Set --adb_path to the installed adb.")


_ADB_PATH = flags.DEFINE_string("adb_path", _find_adb_directory(), "Path to adb.")
_EMULATOR_SETUP = flags.DEFINE_boolean(
    "perform_emulator_setup", False, "Whether to perform emulator setup."
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    "console_port", 5554, "Android emulator console port."
)
_GRPC_PORT = flags.DEFINE_integer("grpc_port", 8554, "Android emulator gRPC port.")
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
_TASK_RANDOM_SEED = flags.DEFINE_integer("task_random_seed", 30, "Task seed.")
_TASKS = flags.DEFINE_list(
    "tasks", None, "Specific task templates to run in the suite family."
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    "n_task_combinations", 1, "Number of instances per task template."
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir", "", "Existing checkpoint dir to resume."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    os.path.join(_default_runs_root(), "agentprog"),
    "Output root if checkpoint_dir is not provided.",
)
_FIXED_TASK_SEED = flags.DEFINE_boolean(
    "fixed_task_seed", False, "Reuse identical task params for combinations."
)

_AGENTPROG_ROOT = flags.DEFINE_string(
    "agentprog_root",
    os.environ.get("AGENTPROG_ROOT", ""),
    "Path to MobileLLM/AgentProg checkout if the package is not installed.",
)
_AGENTPROG_AGENT_NAME = flags.DEFINE_enum(
    "agentprog_agent_name",
    "agentprog",
    ["agentprog"],
    "AgentProg variant. CATBench currently uses the official default variant.",
)
_EXP_NAME = flags.DEFINE_string(
    "exp_name", "catbench", "Experiment name used inside AgentProg output dirs."
)
_TOOL_SET = flags.DEFINE_enum(
    "tool_set", "mobile", ["mobile", "ai_phone"], "AgentProg tool set."
)
_MODEL = flags.DEFINE_string(
    "model",
    os.environ.get("AGENTPROG_MODEL", "gemini/gemini-2.5-pro"),
    "LiteLLM model name used by AgentProg workflow/executor calls.",
)
_API_KEY = flags.DEFINE_string(
    "api_key",
    os.environ.get("AGENTPROG_API_KEY", ""),
    "API key for AgentProg workflow/executor model. Empty uses AgentProg env defaults.",
)
_BASE_URL = flags.DEFINE_string(
    "base_url",
    os.environ.get("AGENTPROG_BASE_URL", ""),
    "Base URL for AgentProg workflow/executor model. Empty uses env defaults.",
)
_USE_AW_LOCATOR = flags.DEFINE_boolean(
    "use_aw_locator",
    False,
    "If true, use AndroidWorld locator instead of UI-TARS (matches upstream "
    "'agentprog_w_aw_env' variant).",
)
_UI_TARS_MODEL = flags.DEFINE_string(
    "ui_tars_model",
    os.environ.get("AGENTPROG_UI_TARS_MODEL", "doubao-seed-1-8-251228"),
    "UI-TARS locator model name used by AgentProg.",
)
_UI_TARS_API_KEY = flags.DEFINE_string(
    "ui_tars_api_key",
    os.environ.get("AGENTPROG_UI_TARS_API_KEY", ""),
    "UI-TARS/Ark API key. Empty uses ARK_API_KEY.",
)
_UI_TARS_BASE_URL = flags.DEFINE_string(
    "ui_tars_base_url",
    os.environ.get("AGENTPROG_UI_TARS_BASE_URL", ""),
    "UI-TARS/Ark base URL. Empty uses DOUBAO_BASE_URL.",
)
_USE_BELIEF_STATE = flags.DEFINE_boolean(
    "use_belief_state", True, "Use AgentProg belief-state mode."
)
_CACHE_MODE = flags.DEFINE_boolean(
    "cache_mode", False, "Use AgentProg cache mode."
)
_SHOW_DASHBOARD = flags.DEFINE_boolean(
    "show_dashboard", False, "Show AgentProg rich dashboard while running."
)
_AGENTPROG_MAX_RETRY_TIME = flags.DEFINE_integer(
    "agentprog_max_retry_time",
    _env_int("AGENTPROG_MAX_RETRY_TIME", 12),
    "Maximum internal AgentProg retries per sequential workflow step.",
)
_AGENTPROG_MAX_LOOP_TIME = flags.DEFINE_integer(
    "agentprog_max_loop_time",
    _env_int("AGENTPROG_MAX_LOOP_TIME", 12),
    "Maximum internal AgentProg retries per loop workflow step.",
)
_AGENTPROG_STEP_TIMEOUT_SECONDS = flags.DEFINE_integer(
    "agentprog_step_timeout_seconds",
    _env_int("AGENTPROG_STEP_TIMEOUT_SECONDS", 1200),
    "Wall-clock timeout for one AgentProg pipeline step. Non-positive disables it.",
)


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
      env=env,
  )
  suite.suite_family = _SUITE_FAMILY.value

  checkpoint_dir = (
      _CHECKPOINT_DIR.value
      or checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)
  )

  agent = agentprog_agent.AgentProg(
      env=env,
      name=_AGENTPROG_AGENT_NAME.value,
      output_path=checkpoint_dir,
      agentprog_root=_AGENTPROG_ROOT.value,
      console_port=_DEVICE_CONSOLE_PORT.value,
      grpc_port=_GRPC_PORT.value,
      exp_name=_EXP_NAME.value,
      tool_set=_TOOL_SET.value,
      model=_MODEL.value,
      api_key=_API_KEY.value,
      base_url=_BASE_URL.value,
      ui_tars_model=_UI_TARS_MODEL.value,
      ui_tars_api_key=_UI_TARS_API_KEY.value,
      ui_tars_base_url=_UI_TARS_BASE_URL.value,
      use_belief_state=_USE_BELIEF_STATE.value,
      use_aw_locator=_USE_AW_LOCATOR.value,
      cache_mode=_CACHE_MODE.value,
      show_dashboard=_SHOW_DASHBOARD.value,
      fold_dashboard=True,
      transition_pause=None,
      max_retry_time=_AGENTPROG_MAX_RETRY_TIME.value,
      max_loop_time=_AGENTPROG_MAX_LOOP_TIME.value,
      step_timeout_seconds=_AGENTPROG_STEP_TIMEOUT_SECONDS.value,
  )
  agent.get_task_name(suite)

  print(
      f"Starting AgentProg eval on {_SUITE_FAMILY.value}; "
      f"checkpoints/traces: {checkpoint_dir}",
      flush=True,
  )
  suite_utils.run(
      suite,
      agent,
      checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
      demo_mode=False,
  )
  print(f"Finished AgentProg eval. Wrote to {checkpoint_dir}.", flush=True)
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == "__main__":
  app.run(main)
