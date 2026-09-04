"""Run GUI-Owl on Android World tasks.

This runner is adapted from the GUI-Owl path used in the sibling UI-Venus
repository, but follows CATBench's runner conventions so it can be called
directly or through run_app_generalization.py.

Examples:
  python benchmark/run_gui_owl.py --suite_family=android_world \
      --tasks=ClockStopWatchRunning --mode=endpoint \
      --endpoint_url=http://127.0.0.1:8001 \
      --model_name=mPLUG/GUI-Owl-7B

  python benchmark/run_gui_owl.py --suite_family=android_world \
      --tasks=ClockStopWatchRunning --mode=local --device=cuda:0
"""

import atexit
import os
import shutil
import subprocess
import sys
import urllib.parse
from collections.abc import Sequence


def _first_writable_dir(candidates: list[str]) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.expanduser(candidate)
        try:
            os.makedirs(path, exist_ok=True)
            probe_path = os.path.join(path, ".catbench_write_probe")
            with open(probe_path, "w", encoding="utf-8") as probe_file:
                probe_file.write("ok")
            os.remove(probe_path)
            return path
        except OSError:
            continue
    return os.path.expanduser("~/.cache")


_USER_NAME = os.environ.get("USER", "")
_DEFAULT_TMP_DIR = _first_writable_dir(
    [
        os.environ.get("TMPDIR", ""),
        os.path.join("$HOME", _USER_NAME, "tmp") if _USER_NAME else "",
        "/tmp",
    ]
)
_DEFAULT_CACHE_ROOT = _first_writable_dir(
    [
        os.environ.get("CATBENCH_HF_CACHE_ROOT", ""),
        os.path.join("$HOME", _USER_NAME, "catbench_cache") if _USER_NAME else "",
        os.path.join("/tmp", f"catbench_cache_{_USER_NAME}") if _USER_NAME else "",
        "/tmp/catbench_cache",
        os.environ.get("XDG_CACHE_HOME", ""),
        os.path.expanduser("~/.cache"),
    ]
)

os.environ["TMPDIR"] = _DEFAULT_TMP_DIR
os.environ["TEMP"] = _DEFAULT_TMP_DIR
os.environ["TMP"] = _DEFAULT_TMP_DIR

os.environ.setdefault("CATBENCH_HF_CACHE_ROOT", _DEFAULT_CACHE_ROOT)
os.environ.setdefault("XDG_CACHE_HOME", _DEFAULT_CACHE_ROOT)

_HF_HOME = os.path.join(os.environ["CATBENCH_HF_CACHE_ROOT"], "huggingface")
os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub")
)
os.environ.setdefault("HF_XET_CACHE", os.path.join(os.environ["HF_HOME"], "xet"))
os.environ.setdefault(
    "TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers")
)

os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_XET_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

from absl import app, flags, logging

try:
    import pysqlite3

    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ImportError:
    # Fall back to stdlib sqlite3 when pysqlite3 is unavailable.
    pass

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from benchmark import endpoint_contract
from android_world.agents import gui_owl
from android_world.agents import infer_ma3
from android_world.env import env_launcher

logging.set_verbosity(logging.WARNING)

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ.pop("GRPC_TRACE", None)


def _default_runs_root() -> str:
    explicit = os.environ.get("CATBENCH_RUNS_DIR")
    if explicit:
        return os.path.expanduser(explicit)

    candidates = []
    if _USER_NAME:
        candidates.append(os.path.join("$HOME", _USER_NAME, "android_world_runs"))
    candidates.append(os.path.expanduser("~/catbench_runs"))

    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except OSError:
            continue

    return os.path.expanduser("~/catbench_runs")


DEFAULT_RUNS_ROOT = _default_runs_root()
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_RUNS_ROOT, "gui_owl")


def _find_adb_directory() -> str:
    """Returns the path to adb."""
    adb_in_path = shutil.which("adb")
    if adb_in_path:
        return adb_in_path

    potential_paths = [
        os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
        os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
        os.path.expanduser("~/android-sdk/platform-tools/adb"),
        os.path.expanduser("~/androidsdk/platform-tools/adb"),
        "/usr/bin/adb",
        "/usr/local/bin/adb",
        "/opt/android-sdk/platform-tools/adb",
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path

    raise EnvironmentError(
        "adb not found. Install the Android SDK platform-tools and either:\n"
        "  - Add the platform-tools directory to PATH, or\n"
        "  - Pass --adb_path=/path/to/adb explicitly."
    )


def _normalize_openai_base_url(endpoint_url: str) -> str:
    base = endpoint_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _endpoint_models_url(endpoint_url: str) -> str:
    return endpoint_contract.models_url(endpoint_url)


def _fetch_served_models(models_url: str, timeout_sec: float) -> list[str]:
    return list(endpoint_contract.fetch_model_records(models_url, timeout_sec))


def _wait_for_endpoint_model(endpoint_url: str, model_name: str) -> None:
    models_url = _endpoint_models_url(endpoint_url)
    timeout_sec = max(1, _ENDPOINT_WAIT_TIMEOUT_SEC.value)
    poll_sec = max(0.5, _ENDPOINT_WAIT_POLL_SEC.value)
    minimum_context = max(0, _ENDPOINT_MIN_CONTEXT_LEN.value)

    print(
        f"[CONFIG] Waiting for endpoint model {model_name!r} at {models_url} "
        f"(timeout={timeout_sec}s)",
        flush=True,
    )

    record = endpoint_contract.wait_for_model(
        endpoint_url,
        model_name,
        minimum_context,
        timeout_sec=timeout_sec,
        poll_sec=poll_sec,
        loopback_only=_ENDPOINT_REQUIRE_LOOPBACK.value,
        status_callback=lambda status: print(
            f"[WAIT] Endpoint not ready: {status}", flush=True
        ),
    )
    print(
        f"[CONFIG] Endpoint ready: found exact model {model_name!r}; "
        f"max_model_len={record.get('max_model_len')!r}",
        flush=True,
    )


def _launch_local_vllm(
    model_name: str,
    device: str,
    endpoint_url: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    allowed_local_media_path: str,
    limit_mm_per_prompt: str,
    mm_processor_kwargs: str,
) -> "subprocess.Popen[bytes]":
    """Launches a local OpenAI-compatible vLLM server for GUI-Owl."""
    parsed = urllib.parse.urlparse(endpoint_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--trust-remote-code",
        "--dtype",
        _VLLM_DTYPE.value,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]

    if allowed_local_media_path:
        cmd += ["--allowed-local-media-path", allowed_local_media_path]
    if limit_mm_per_prompt:
        cmd += ["--limit-mm-per-prompt", limit_mm_per_prompt]
    if mm_processor_kwargs:
        cmd += ["--mm-processor-kwargs", mm_processor_kwargs]

    env = os.environ.copy()
    if device.startswith("cuda:"):
        gpu_idx = device.split(":", 1)[1]
        env["CUDA_VISIBLE_DEVICES"] = gpu_idx
        print(
            f"[LOCAL] Pinning vLLM to GPU {gpu_idx} "
            f"(CUDA_VISIBLE_DEVICES={gpu_idx})",
            flush=True,
        )
    elif device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd += ["--device", "cpu"]

    print(f"[LOCAL] Launching vLLM server: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, env=env)
    print(f"[LOCAL] vLLM server PID={proc.pid} - waiting for model to load...", flush=True)
    return proc


def _effective_model_name() -> str:
    return _MODEL.value or _MODEL_NAME.value


def _effective_endpoint_url() -> str:
    return _BASE_URL.value or _ENDPOINT_URL.value


def _effective_api_key() -> str:
    return _API_KEY.value or "EMPTY"


def _effective_traj_output_path() -> str:
    if _TRAJ_OUTPUT_PATH.value:
        return _TRAJ_OUTPUT_PATH.value
    return os.path.join(_OUTPUT_PATH.value, "traj")


_ADB_PATH = flags.DEFINE_string(
    "adb_path",
    _find_adb_directory(),
    "Path to adb. Set if not installed through SDK.",
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    "perform_emulator_setup",
    False,
    "Whether to perform emulator setup. This must be done once and only once "
    "before running Android World.",
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    "console_port",
    5554,
    "The console port of the running Android device.",
)
_GRPC_PORT = flags.DEFINE_integer(
    "grpc_port",
    8554,
    "The gRPC port of the running Android device.",
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
    "Suite family to run. See registry.py for more information.",
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    "task_random_seed", 30, "Random seed for task randomness."
)
_TASKS = flags.DEFINE_list(
    "tasks",
    None,
    "List of specific tasks to run in the given suite family. If None, run all tasks.",
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    "n_task_combinations",
    1,
    "Number of task instances to run for each task template.",
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir",
    "",
    "Directory to save checkpoints and resume evaluation from.",
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    DEFAULT_OUTPUT_PATH,
    "Path to save run outputs to if checkpoint_dir is not provided.",
)
_TRAJ_OUTPUT_PATH = flags.DEFINE_string(
    "traj_output_path",
    "",
    "Directory to save screenshot trajectories and action logs.",
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "mPLUG/GUI-Owl-7B",
    "Model identifier to use for GUI-Owl requests.",
)
_MODEL = flags.DEFINE_string(
    "model",
    "",
    "Optional alias for --model_name, kept for UI-Venus-style compatibility.",
)
_API_KEY = flags.DEFINE_string(
    "api_key",
    "",
    "API key for the OpenAI-compatible endpoint. Defaults to 'EMPTY'.",
)
_DEVICE = flags.DEFINE_string(
    "device",
    "cuda:3",
    "Device to load model on. Options: cuda:0, cuda:1, auto, cpu.",
)
_FIXED_TASK_SEED = flags.DEFINE_boolean(
    "fixed_task_seed",
    False,
    "Whether to use the same task seed when running multiple task combinations.",
)
_MAX_STEPS = flags.DEFINE_integer(
    "max_steps",
    None,
    "Maximum number of steps per task. If not set, uses task complexity-based budget.",
)
_MODE = flags.DEFINE_enum(
    "mode",
    "endpoint",
    ["local", "endpoint"],
    "Inference mode: local (launch vLLM) or endpoint (connect to existing server).",
)
_ENDPOINT_URL = flags.DEFINE_string(
    "endpoint_url",
    "http://127.0.0.1:8000",
    "URL of model server when using endpoint mode.",
)
_BASE_URL = flags.DEFINE_string(
    "base_url",
    "",
    "Optional alias for --endpoint_url, kept for UI-Venus-style compatibility.",
)
_ENDPOINT_WAIT_TIMEOUT_SEC = flags.DEFINE_integer(
    "endpoint_wait_timeout_sec",
    1800,
    "Max seconds to wait for endpoint readiness.",
)
_ENDPOINT_WAIT_POLL_SEC = flags.DEFINE_float(
    "endpoint_wait_poll_sec",
    5.0,
    "Polling interval in seconds while waiting for endpoint readiness.",
)
_ENDPOINT_MIN_CONTEXT_LEN = flags.DEFINE_integer(
    "endpoint_min_context_len",
    16384,
    "Minimum integer max_model_len advertised by the endpoint.",
)
_ENDPOINT_REQUIRE_LOOPBACK = flags.DEFINE_boolean(
    "endpoint_require_loopback",
    True,
    "Require endpoint_url to use localhost or an IP loopback address.",
)
_GPU_MEMORY_UTILIZATION = flags.DEFINE_float(
    "gpu_memory_utilization",
    0.75,
    "Fraction of GPU memory vLLM may use (0.0-1.0).",
)
_MAX_MODEL_LEN = flags.DEFINE_integer(
    "max_model_len",
    32768,
    "Max model length forwarded to the local vLLM server.",
)
_ALLOWED_LOCAL_MEDIA_PATH = flags.DEFINE_string(
    "allowed_local_media_path",
    "/",
    "Allowed local media path forwarded to the local vLLM server.",
)
_LIMIT_MM_PER_PROMPT = flags.DEFINE_string(
    "limit_mm_per_prompt",
    '{"image":2,"video":0}',
    "Value forwarded to vLLM --limit-mm-per-prompt when mode=local.",
)
_MM_PROCESSOR_KWARGS = flags.DEFINE_string(
    "mm_processor_kwargs",
    '{"min_pixels":3136,"max_pixels":10035200}',
    "JSON string forwarded to vLLM --mm-processor-kwargs when mode=local.",
)
_VLLM_DTYPE = flags.DEFINE_string(
    "vllm_dtype",
    "bfloat16",
    "Dtype used when launching the local vLLM server.",
)


def _main() -> None:
    model_name = _effective_model_name()
    endpoint_url = _effective_endpoint_url()
    traj_output_path = _effective_traj_output_path()
    normalized_base_url = _normalize_openai_base_url(endpoint_url)

    _vllm_proc = None
    if _MODE.value == "local":
        _vllm_proc = _launch_local_vllm(
            model_name=model_name,
            device=_DEVICE.value,
            endpoint_url=endpoint_url,
            gpu_memory_utilization=_GPU_MEMORY_UTILIZATION.value,
            max_model_len=_MAX_MODEL_LEN.value,
            allowed_local_media_path=_ALLOWED_LOCAL_MEDIA_PATH.value,
            limit_mm_per_prompt=_LIMIT_MM_PER_PROMPT.value,
            mm_processor_kwargs=_MM_PROCESSOR_KWARGS.value,
        )
        atexit.register(
            lambda: _vllm_proc.terminate()
            if _vllm_proc and _vllm_proc.poll() is None
            else None
        )

    _wait_for_endpoint_model(endpoint_url, model_name)

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

    print(f"[CONFIG] GUI-Owl model: {model_name}")
    print(f"[CONFIG] GUI-Owl endpoint: {normalized_base_url}")
    print(f"[CONFIG] GUI-Owl trajectories: {traj_output_path}")

    agent = gui_owl.GUIOwl(
        env,
        infer_ma3.GUIOwlWrapper(_effective_api_key(), normalized_base_url, model_name),
        "abs_resized",
        api_key=None,
        url=None,
        output_path=traj_output_path,
    )
    agent.name = "gui_owl"
    if hasattr(agent, "get_task_name"):
        agent.get_task_name(suite)

    if _MAX_STEPS.value is not None:
        agent.max_steps = _MAX_STEPS.value
        print(f"[CONFIG] Set agent.max_steps = {agent.max_steps}")
    else:
        print("[CONFIG] Using default task complexity-based step budgets")

    if _CHECKPOINT_DIR.value:
        checkpoint_dir = _CHECKPOINT_DIR.value
    else:
        checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)

    print(
        f"Starting eval with agent gui_owl and writing checkpoints to {checkpoint_dir}",
        flush=True,
    )
    suite_utils.run(
        suite,
        agent,
        checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
        demo_mode=False,
    )
    print(
        f"Finished running agent gui_owl on {_SUITE_FAMILY.value} family. "
        f"Wrote checkpoints to {checkpoint_dir}.",
        flush=True,
    )
    env.close()


def main(argv: Sequence[str]) -> None:
    del argv
    _main()


if __name__ == "__main__":
    app.run(main)
