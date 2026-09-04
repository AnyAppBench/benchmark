#!/usr/bin/env python3
"""Run OpenAI-compatible Python-action GUI agents on AndroidWorld tasks."""

from __future__ import annotations

import os
import shutil
import sys
import time
import urllib.error
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
from android_world.agents import openai_python_action
from android_world.env import android_world_controller
from android_world.env import env_launcher

try:
  from benchmark import endpoint_contract
except ModuleNotFoundError:
  import endpoint_contract


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


def _normalize_endpoint(endpoint_url: str) -> str:
  endpoint_url = endpoint_url.rstrip("/")
  return endpoint_url if endpoint_url.endswith("/v1") else f"{endpoint_url}/v1"


def _venus_predict_health_url(endpoint_url: str) -> str:
  endpoint_url = endpoint_url.rstrip("/")
  if endpoint_url.endswith("/predict"):
    return endpoint_url[: -len("/predict")] + "/health"
  return f"{endpoint_url}/health"


def _endpoint_health_url(endpoint_url: str, endpoint_format: str) -> str:
  if endpoint_format in {"venus_predict", "gui_proxy_predict"}:
    return _venus_predict_health_url(endpoint_url)
  return f"{_normalize_endpoint(endpoint_url)}/models"


def _wait_for_endpoint(
    endpoint_url: str,
    endpoint_format: str,
    model_name: str,
    minimum_context: int,
    require_loopback: bool,
    timeout_sec: int,
    poll_sec: float,
) -> None:
  if endpoint_format == "openai":
    record = endpoint_contract.wait_for_model(
        endpoint_url,
        model_name,
        minimum_context,
        timeout_sec=max(1, timeout_sec),
        poll_sec=max(0.5, poll_sec),
        loopback_only=require_loopback,
        status_callback=lambda status: print(
            f"Waiting for endpoint: {status}", flush=True
        ),
    )
    print(
        f"Endpoint ready: exact model={model_name!r}; "
        f"max_model_len={record.get('max_model_len')!r}",
        flush=True,
    )
    return
  if minimum_context > 0:
    raise endpoint_contract.EndpointContractError(
        "A positive --endpoint_min_context_len requires "
        "--endpoint_format=openai so /v1/models can be attested."
    )
  if require_loopback:
    endpoint_contract.require_loopback(endpoint_url)
  health_url = _endpoint_health_url(endpoint_url, endpoint_format)
  deadline = time.time() + timeout_sec
  last_error = "not checked"
  while time.time() < deadline:
    try:
      with urllib.request.urlopen(health_url, timeout=5.0) as response:
        if 200 <= response.status < 300:
          return
    except urllib.error.URLError as exc:
      last_error = str(exc)
    except Exception as exc:  # pylint: disable=broad-except
      last_error = str(exc)
    time.sleep(max(0.5, poll_sec))
  raise RuntimeError(
      f"Endpoint did not become ready at {health_url}: {last_error}"
  )


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
_INSTALL_A11Y_FORWARDING_APP = flags.DEFINE_boolean(
    "install_a11y_forwarding_app",
    True,
    (
        "Whether to install/reinstall the AndroidWorld accessibility "
        "forwarding APK before starting its service."
    ),
)
_A11Y_METHOD = flags.DEFINE_enum(
    "a11y_method",
    android_world_controller.A11yMethod.A11Y_FORWARDER_APP.value,
    [item.value for item in android_world_controller.A11yMethod],
    "UI tree backend to use for AndroidWorld observations.",
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
    os.path.expanduser("~/android_world/runs/openai_python_action"),
    "Path to save run outputs if checkpoint_dir is not provided.",
)
_ENDPOINT_URL = flags.DEFINE_string(
    "endpoint_url", "http://127.0.0.1:8000", "OpenAI-compatible endpoint URL."
)
_MODE = flags.DEFINE_enum(
    "mode",
    "endpoint",
    ["endpoint"],
    "Compatibility flag. This runner always calls an endpoint.",
)
_ENDPOINT_FORMAT = flags.DEFINE_enum(
    "endpoint_format",
    "openai",
    ["openai", "venus_predict", "gui_proxy_predict"],
    (
        "Endpoint protocol: OpenAI chat-completions, legacy UI-Venus-Navi "
        "/predict, or generic GUI proxy /predict."
    ),
)
_MODEL_NAME = flags.DEFINE_string("model_name", "", "Endpoint model id.")
_API_KEY = flags.DEFINE_string("api_key", "EMPTY", "Endpoint API key.")
_PROMPT_STYLE = flags.DEFINE_enum(
    "prompt_style",
    "ui_venus_navi",
    ["ui_venus_navi", "mobilerl_point_think", "appagent_v2_lite"],
    "Prompt/action grammar to use.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens", 2048, "Max output tokens per step."
)
_TEMPERATURE = flags.DEFINE_float("temperature", 0.0, "Sampling temperature.")
_IMAGE_MAX_PIXELS = flags.DEFINE_integer(
    "image_max_pixels",
    0,
    "If >0, resize screenshots to this many pixels before sending.",
)
_FIXED_TASK_SEED = flags.DEFINE_boolean(
    "fixed_task_seed", False, "Use identical task params across combinations."
)
_MAX_STEPS = flags.DEFINE_integer(
    "max_steps", None, "Maximum steps per task."
)
_WAIT_AFTER_ACTION = flags.DEFINE_float(
    "wait_after_action_seconds", 2.0, "Seconds to wait after each action."
)
_ENDPOINT_WAIT_TIMEOUT_SEC = flags.DEFINE_integer(
    "endpoint_wait_timeout_sec", 1800, "Max seconds to wait for endpoint."
)
_ENDPOINT_WAIT_POLL_SEC = flags.DEFINE_float(
    "endpoint_wait_poll_sec", 5.0, "Endpoint readiness polling interval."
)
_ENDPOINT_MIN_CONTEXT_LEN = flags.DEFINE_integer(
    "endpoint_min_context_len",
    0,
    "Minimum integer max_model_len advertised by /v1/models.",
)
_ENDPOINT_REQUIRE_LOOPBACK = flags.DEFINE_boolean(
    "endpoint_require_loopback",
    False,
    "Require endpoint_url to use localhost or an IP loopback address.",
)


def _main() -> None:
  if not _MODEL_NAME.value:
    raise ValueError("--model_name is required.")
  os.environ.setdefault("ANDROID_SERIAL", f"emulator-{_DEVICE_CONSOLE_PORT.value}")
  _wait_for_endpoint(
      _ENDPOINT_URL.value,
      endpoint_format=_ENDPOINT_FORMAT.value,
      model_name=_MODEL_NAME.value,
      minimum_context=max(0, _ENDPOINT_MIN_CONTEXT_LEN.value),
      require_loopback=_ENDPOINT_REQUIRE_LOOPBACK.value,
      timeout_sec=_ENDPOINT_WAIT_TIMEOUT_SEC.value,
      poll_sec=_ENDPOINT_WAIT_POLL_SEC.value,
  )
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      grpc_port=_GRPC_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
      a11y_method=android_world_controller.A11yMethod(_A11Y_METHOD.value),
      install_a11y_forwarding_app=_INSTALL_A11Y_FORWARDING_APP.value,
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

  agent = openai_python_action.OpenAIPythonActionAgent(
      env,
      endpoint_url=_ENDPOINT_URL.value,
      model_name=_MODEL_NAME.value,
      api_key=_API_KEY.value,
      endpoint_format=_ENDPOINT_FORMAT.value,
      prompt_style=_PROMPT_STYLE.value,
      max_new_tokens=_MAX_NEW_TOKENS.value,
      temperature=_TEMPERATURE.value,
      image_max_pixels=(
          _IMAGE_MAX_PIXELS.value if _IMAGE_MAX_PIXELS.value > 0 else None
      ),
      wait_after_action_seconds=_WAIT_AFTER_ACTION.value,
      output_path=os.path.join(_OUTPUT_PATH.value, "traces"),
      name=f"python_action_{_PROMPT_STYLE.value}",
  )
  if _MAX_STEPS.value is not None:
    agent.max_steps = _MAX_STEPS.value

  checkpoint_dir = (
      _CHECKPOINT_DIR.value
      or checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)
  )
  print(
      f"Running {_MODEL_NAME.value} with prompt_style={_PROMPT_STYLE.value}; "
      f"endpoint_format={_ENDPOINT_FORMAT.value}; "
      f"checkpoints={checkpoint_dir}",
      flush=True,
  )
  suite_utils.run(
      suite,
      agent,
      checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
      demo_mode=False,
  )
  print(f"Finished run. Results at {checkpoint_dir}.")
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == "__main__":
  app.run(main)
