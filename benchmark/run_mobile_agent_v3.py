#!/usr/bin/env python3
"""Run Mobile-Agent-v3 on AndroidWorld/CATBench tasks via an OpenAI endpoint."""

from __future__ import annotations

import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging

try:
  import pysqlite3

  sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ImportError:
  pass

from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import infer_ma3
from android_world.agents import mobile_agent_v3
from android_world.env import env_launcher


logging.set_verbosity(logging.WARNING)


def _find_adb_directory() -> str:
  adb_in_path = shutil.which("adb")
  if adb_in_path:
    return adb_in_path
  for path in (
      os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
      os.path.expanduser("~/android-sdk/platform-tools/adb"),
      "/usr/bin/adb",
      "/usr/local/bin/adb",
      "/opt/android-sdk/platform-tools/adb",
  ):
    if os.path.isfile(path):
      return path
  raise EnvironmentError("adb not found. Pass --adb_path=/path/to/adb.")


def _normalize_openai_base_url(endpoint_url: str) -> str:
  endpoint_url = endpoint_url.rstrip("/")
  if endpoint_url.endswith("/chat/completions"):
    return endpoint_url[: -len("/chat/completions")]
  if endpoint_url.endswith("/v1"):
    return endpoint_url
  return f"{endpoint_url}/v1"


def _models_url(endpoint_url: str) -> str:
  return f"{_normalize_openai_base_url(endpoint_url).rstrip('/')}/models"


def _wait_for_endpoint(endpoint_url: str, timeout_sec: int, poll_sec: float) -> None:
  deadline = time.time() + timeout_sec
  url = _models_url(endpoint_url)
  last_error = "endpoint not checked yet"
  while time.time() < deadline:
    try:
      with urllib.request.urlopen(url, timeout=5.0) as response:
        if 200 <= response.status < 300:
          return
    except urllib.error.URLError as exc:
      last_error = str(exc)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      last_error = str(exc)
    time.sleep(max(0.5, poll_sec))
  raise RuntimeError(f"Endpoint did not become ready at {url}: {last_error}")


_ADB_PATH = flags.DEFINE_string("adb_path", _find_adb_directory(), "Path to adb.")
_EMULATOR_SETUP = flags.DEFINE_boolean(
    "perform_emulator_setup", False, "Whether to perform emulator setup."
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    "console_port", 5554, "The console port of the running Android device."
)
_GRPC_PORT = flags.DEFINE_integer(
    "grpc_port", 8554, "The gRPC port of the running Android device."
)
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
    "task_random_seed", 30, "Random seed for task randomness."
)
_TASKS = flags.DEFINE_list("tasks", None, "Specific task templates to run.")
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    "n_task_combinations", 1, "Number of task instances per template."
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir", "", "Directory to save/resume checkpoints."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    os.path.expanduser("~/android_world/runs/mobile_agent_v3"),
    "Path to save run outputs to if checkpoint_dir is not provided.",
)
_TRAJ_OUTPUT_PATH = flags.DEFINE_string(
    "traj_output_path", "", "Directory for Mobile-Agent-v3 step artifacts."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "mobile-agent-v3", "OpenAI-compatible model id."
)
_ENDPOINT_URL = flags.DEFINE_string(
    "endpoint_url", "http://127.0.0.1:8000", "OpenAI-compatible endpoint URL."
)
_BASE_URL = flags.DEFINE_string(
    "base_url", "", "Alias for --endpoint_url."
)
_API_KEY = flags.DEFINE_string("api_key", "EMPTY", "Endpoint API key.")
_FIXED_TASK_SEED = flags.DEFINE_boolean(
    "fixed_task_seed",
    False,
    "Whether to use identical task params across combinations.",
)
_MAX_STEPS = flags.DEFINE_integer(
    "max_steps", None, "Maximum steps per task, overriding complexity budget."
)
_ENDPOINT_WAIT_TIMEOUT_SEC = flags.DEFINE_integer(
    "endpoint_wait_timeout_sec", 1800, "Max seconds to wait for endpoint."
)
_ENDPOINT_WAIT_POLL_SEC = flags.DEFINE_float(
    "endpoint_wait_poll_sec", 5.0, "Polling interval while waiting."
)


def _main() -> None:
  endpoint_url = _BASE_URL.value or _ENDPOINT_URL.value
  normalized_base_url = _normalize_openai_base_url(endpoint_url)
  traj_output_path = _TRAJ_OUTPUT_PATH.value or os.path.join(
      _OUTPUT_PATH.value, "mobile_agent_v3_traces"
  )
  _wait_for_endpoint(
      endpoint_url,
      timeout_sec=_ENDPOINT_WAIT_TIMEOUT_SEC.value,
      poll_sec=_ENDPOINT_WAIT_POLL_SEC.value,
  )

  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
      grpc_port=_GRPC_PORT.value,
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

  print(f"[CONFIG] Mobile-Agent-v3 model: {_MODEL_NAME.value}")
  print(f"[CONFIG] Mobile-Agent-v3 endpoint: {normalized_base_url}")
  print(f"[CONFIG] Mobile-Agent-v3 trajectories: {traj_output_path}")

  agent = mobile_agent_v3.MobileAgentV3_M3A(
      env,
      infer_ma3.GUIOwlWrapper(
          api_key=_API_KEY.value or "EMPTY",
          base_url=normalized_base_url,
          model_name=_MODEL_NAME.value,
      ),
      name="mobile_agent_v3",
      output_path=traj_output_path,
  )
  if hasattr(agent, "get_task_name"):
    agent.get_task_name(suite)
  if _MAX_STEPS.value is not None:
    agent.max_steps = _MAX_STEPS.value

  checkpoint_dir = (
      _CHECKPOINT_DIR.value
      if _CHECKPOINT_DIR.value
      else checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)
  )
  print(f"Starting Mobile-Agent-v3 eval. Checkpoints: {checkpoint_dir}", flush=True)
  suite_utils.run(
      suite,
      agent,
      checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
      demo_mode=False,
  )
  print(f"Finished Mobile-Agent-v3 eval. Wrote checkpoints to {checkpoint_dir}.")
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == "__main__":
  app.run(main)
