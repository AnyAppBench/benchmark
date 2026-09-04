"""Run MarsXL/UI-Voyager (4B) on AndroidWorld.

UI-Voyager is served via an OpenAI-compatible endpoint (e.g. vLLM).

Example (one-shot host):
    vllm serve MarsXL/UI-Voyager \\
        --host 0.0.0.0 --port 8000 \\
        --served-model-name UI-Voyager \\
        --limit-mm-per-prompt '{"image": 1}' \\
        --max-model-len 196608

If engine startup fails with a KV-cache memory error, reduce
`--max-model-len` further (for example to 131072).

Then:
    python benchmark/run_ui_voyager.py \\
        --endpoint_url=http://127.0.0.1:8000 \\
        --model_name=UI-Voyager \\
        --tasks=ContactsAddContact
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from absl import app, flags, logging

# Fix for SQLite FTS4 support (needed for Joplin app setup).
try:
  import pysqlite3  # type: ignore
  sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except Exception:  # pylint: disable=broad-exception-caught
  pass

from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import ui_voyager as ui_voyager_module
from android_world.env import env_launcher

try:
  from benchmark import endpoint_contract
except ModuleNotFoundError:
  import endpoint_contract


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
    "task_random_seed", 30, "Random seed for task randomness.")
_TASKS = flags.DEFINE_list(
    "tasks", None,
    "List of specific tasks to run. If None, run all tasks in suite family.")
_HYBRID_CATEGORIES = flags.DEFINE_list(
    "hybrid_categories", None,
    "Cross-app hybrid mode. Comma-separated category names from"
    " {sms,files,maps,contacts,clock}. Expands to the union of every"
    " AW-original task name AND every cross-app generated task name in"
    " those categories. Combines with --tasks (intersection if both set).")
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    "n_task_combinations", 1,
    "Number of task instances to run for each task template.")


# Maps each user-facing category name to the task-class-name prefixes that
# belong to it -- both AW-original and cross-app-generated.
_HYBRID_CATEGORY_PREFIXES: dict[str, tuple[str, ...]] = {
    "sms": ("Sms", "SimpleSms"),
    "files": ("Files",),
    "maps": ("Maps", "OsmAnd"),
    "contacts": ("Contacts",),
    "clock": ("Clock",),
}


def _resolve_hybrid_tasks(
    full_registry: dict[str, type],
    categories: list[str],
) -> list[str]:
  """Returns every registered task name matching any prefix for the categories.

  Unknown category names cause the run to abort with a clear message rather
  than silently dropping them.
  """
  unknown = [c for c in categories if c not in _HYBRID_CATEGORY_PREFIXES]
  if unknown:
    raise ValueError(
        f"Unknown hybrid category/categories: {unknown}. Valid options:"
        f" {sorted(_HYBRID_CATEGORY_PREFIXES)}"
    )
  prefixes = tuple(
      p for c in categories for p in _HYBRID_CATEGORY_PREFIXES[c]
  )
  return sorted(name for name in full_registry if name.startswith(prefixes))
_FIXED_TASK_SEED = flags.DEFINE_boolean(
    "fixed_task_seed", False,
    "Use identical params across task combinations.")

_CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir", "", "Resume checkpoint directory.")
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    os.path.expanduser("~/android_world_runs/ui_voyager"),
    "Output root if no checkpoint_dir is provided.")

_ENDPOINT_URL = flags.DEFINE_string(
    "endpoint_url", "http://127.0.0.1:8000",
    "Base URL of the OpenAI-compatible endpoint serving UI-Voyager.")
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "UI-Voyager",
    "Model name registered with the endpoint (--served-model-name).")
_API_KEY = flags.DEFINE_string(
    "api_key", os.environ.get("UI_VOYAGER_API_KEY", "EMPTY"),
    "API key for the endpoint. Usually 'EMPTY' for self-hosted vLLM.")

_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens", 512,
    "Max new tokens per step (the upstream action-sized default).")
_TEMPERATURE = flags.DEFINE_float(
    "temperature", 0.7, "Sampling temperature.")
_TOP_P = flags.DEFINE_float(
    "top_p", 0.8, "Nucleus sampling top-p.")
_MAX_STEPS = flags.DEFINE_integer(
    "max_steps", None,
    "Max steps per task. Defaults to task.complexity * 10 if unset.")
_WAIT_AFTER_ACTION = flags.DEFINE_float(
    "wait_after_action_seconds", 1.5, "Seconds to wait after each action.")
_HISTORY_LEN = flags.DEFINE_integer(
    "history_len", 30, "Number of previous actions to include in the prompt.")
_ENDPOINT_WAIT_TIMEOUT_SEC = flags.DEFINE_integer(
    "endpoint_wait_timeout_sec", 1800, "Max seconds to wait for endpoint."
)
_ENDPOINT_WAIT_POLL_SEC = flags.DEFINE_float(
    "endpoint_wait_poll_sec", 5.0, "Endpoint readiness polling interval."
)
_ENDPOINT_MIN_CONTEXT_LEN = flags.DEFINE_integer(
    "endpoint_min_context_len",
    16384,
    "Minimum integer max_model_len advertised by /v1/models.",
)
_ENDPOINT_REQUIRE_LOOPBACK = flags.DEFINE_boolean(
    "endpoint_require_loopback",
    True,
    "Require endpoint_url to use localhost or an IP loopback address.",
)


def _main() -> None:
  record = endpoint_contract.wait_for_model(
      _ENDPOINT_URL.value,
      _MODEL_NAME.value,
      max(0, _ENDPOINT_MIN_CONTEXT_LEN.value),
      timeout_sec=max(1, _ENDPOINT_WAIT_TIMEOUT_SEC.value),
      poll_sec=max(0.5, _ENDPOINT_WAIT_POLL_SEC.value),
      loopback_only=_ENDPOINT_REQUIRE_LOOPBACK.value,
      status_callback=lambda status: print(
          f"Waiting for endpoint: {status}", flush=True
      ),
  )
  print(
      f"Endpoint ready: exact model={_MODEL_NAME.value!r}; "
      f"max_model_len={record.get('max_model_len')!r}",
      flush=True,
  )
  reported_context = int(record["max_model_len"])
  if _MAX_NEW_TOKENS.value <= 0:
    raise ValueError("--max_new_tokens must be positive.")
  if _MAX_NEW_TOKENS.value >= reported_context:
    raise ValueError(
        "--max_new_tokens must leave room for UI-Voyager's text/image prompt; "
        f"received {_MAX_NEW_TOKENS.value} for context {reported_context}."
    )
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      grpc_port=_GRPC_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
  )

  task_registry = registry.TaskRegistry()
  full_registry = task_registry.get_registry(family=_SUITE_FAMILY.value)

  task_filter = _TASKS.value
  if _HYBRID_CATEGORIES.value:
    hybrid_names = _resolve_hybrid_tasks(
        full_registry, _HYBRID_CATEGORIES.value
    )
    if not hybrid_names:
      raise ValueError(
          f"No tasks matched hybrid categories {_HYBRID_CATEGORIES.value}."
      )
    if task_filter:
      task_filter = sorted(set(task_filter) & set(hybrid_names))
      if not task_filter:
        raise ValueError(
            "--tasks and --hybrid_categories produced an empty intersection."
        )
    else:
      task_filter = hybrid_names
    print(
        f"Hybrid mode: {_HYBRID_CATEGORIES.value} -> {len(task_filter)}"
        f" tasks (AW-original + cross-app)."
    )

  suite = suite_utils.create_suite(
      full_registry,
      n_task_combinations=_N_TASK_COMBINATIONS.value,
      seed=_TASK_RANDOM_SEED.value,
      tasks=task_filter,
      use_identical_params=_FIXED_TASK_SEED.value,
  )
  suite.suite_family = _SUITE_FAMILY.value

  print(
      f"Initializing UI-Voyager agent "
      f"(endpoint={_ENDPOINT_URL.value}, model={_MODEL_NAME.value})..."
  )
  agent = ui_voyager_module.UIVoyagerAgent(
      env,
      endpoint_url=_ENDPOINT_URL.value,
      model_name=_MODEL_NAME.value,
      api_key=_API_KEY.value,
      max_new_tokens=_MAX_NEW_TOKENS.value,
      temperature=_TEMPERATURE.value,
      top_p=_TOP_P.value,
      wait_after_action_seconds=_WAIT_AFTER_ACTION.value,
      history_len=_HISTORY_LEN.value,
  )
  if _MAX_STEPS.value is not None:
    agent.max_steps = _MAX_STEPS.value

  checkpoint_dir = (
      _CHECKPOINT_DIR.value
      or checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)
  )
  print(f"Running UI-Voyager on {_SUITE_FAMILY.value}; writing to {checkpoint_dir}")
  suite_utils.run(
      suite, agent,
      checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
      demo_mode=False,
  )
  print(f"Finished UI-Voyager run. Results at {checkpoint_dir}.")
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == "__main__":
  app.run(main)
