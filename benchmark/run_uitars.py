import atexit
import os
import shutil
import subprocess
import sys
import inspect
import urllib.parse
from collections.abc import Sequence


def _first_writable_dir(candidates: list[str]) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.expanduser(candidate)
        try:
            os.makedirs(path, exist_ok=True)
            probe_path = os.path.join(path, '.catbench_write_probe')
            with open(probe_path, 'w', encoding='utf-8') as probe_file:
                probe_file.write('ok')
            os.remove(probe_path)
            return path
        except OSError:
            continue
    return os.path.expanduser('~/.cache')


_USER_NAME = os.environ.get('USER', '')
_default_tmp_dir = _first_writable_dir([
    os.environ.get('TMPDIR', ''),
    os.path.join('$HOME', _USER_NAME, 'tmp') if _USER_NAME else '',
    '/tmp',
])
_default_cache_root = _first_writable_dir([
    os.environ.get('CATBENCH_HF_CACHE_ROOT', ''),
    os.path.join('$HOME', _USER_NAME, 'catbench_cache') if _USER_NAME else '',
    os.environ.get('XDG_CACHE_HOME', ''),
    os.path.expanduser('~/.cache'),
])

os.environ['TMPDIR'] = _default_tmp_dir
os.environ['TEMP'] = _default_tmp_dir
os.environ['TMP'] = _default_tmp_dir

os.environ.setdefault('CATBENCH_HF_CACHE_ROOT', _default_cache_root)
os.environ.setdefault('XDG_CACHE_HOME', _default_cache_root)

_hf_home = os.path.join(os.environ['CATBENCH_HF_CACHE_ROOT'], 'huggingface')
os.environ.setdefault('HF_HOME', _hf_home)
os.environ.setdefault('HF_HUB_CACHE', os.path.join(os.environ['HF_HOME'], 'hub'))
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(os.environ['HF_HOME'], 'hub'))
os.environ.setdefault('HF_XET_CACHE', os.path.join(os.environ['HF_HOME'], 'xet'))
os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(os.environ['HF_HOME'], 'transformers'))

os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['HF_HUB_CACHE'], exist_ok=True)
os.makedirs(os.environ['HF_XET_CACHE'], exist_ok=True)
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)

from absl import app, flags, logging

try:
    import pysqlite3

    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    # Fall back to stdlib sqlite3 when pysqlite3 is unavailable.
    pass

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from benchmark.agent_import_utils import resolve_uitars_agent_class
    from benchmark import endpoint_contract
except ModuleNotFoundError:
    from agent_import_utils import resolve_uitars_agent_class
    import endpoint_contract

AndroidWorldUITARSAgent, _UITARS_AGENT_SOURCE = resolve_uitars_agent_class(project_root)
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.env import env_launcher

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ.pop('GRPC_TRACE', None)


def _find_adb_directory() -> str:
    """Returns the path to the adb binary."""
    # 1. Check PATH first — works for any system where adb is installed globally
    #    (e.g. `sudo apt install adb` puts it in /usr/bin/adb).
    adb_in_path = shutil.which('adb')
    if adb_in_path:
        return adb_in_path

    # 2. Common Android SDK locations (Linux / macOS / Windows-WSL).
    potential_paths = [
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),        # Linux default
        os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'), # macOS default
        os.path.expanduser('~/android-sdk/platform-tools/adb'),
        os.path.expanduser('~/androidsdk/platform-tools/adb'),
        '/usr/bin/adb',
        '/usr/local/bin/adb',
        '/opt/android-sdk/platform-tools/adb',
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path

    raise EnvironmentError(
        'adb not found. Install the Android SDK platform-tools and either:\n'
        '  • Add the platform-tools directory to your PATH, or\n'
        '  • Pass --adb_path=/path/to/adb explicitly.'
    )


_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device.',
)
_GRPC_PORT = flags.DEFINE_integer(
    'grpc_port',
    8554,
    'The gRPC port of the running Android device.',
)

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    'Suite family to run. See registry.py for more information.',
)

_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed', 42, 'Random seed for task randomness.'
)

_TASKS = flags.DEFINE_list(
    'tasks',
    None,
    'List of specific tasks to run in the given suite family. If None, run all tasks.',
)

_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    1,
    'Number of task instances to run for each task template.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'The directory to save checkpoints and resume evaluation from.',
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    '$HOME/anyappbench_runs/uitars',
    'The path to save results to if checkpoint_dir is not provided.',
)

_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    'ByteDance-Seed/UI-TARS-1.5-7B',
    'HuggingFace model identifier for UI-TARS.',
)

_DEVICE = flags.DEFINE_string(
    'device',
    'cuda:0',
    'Device to load model on. Options: cuda:0, cuda:1, auto (multi-GPU), cpu.',
)

_SAVE_FAILED_TASKS = flags.DEFINE_boolean(
    'save_failed_tasks',
    True,
    'Save screenshot sequences and thought/action traces for failed tasks.',
)

_FIXED_TASK_SEED = flags.DEFINE_boolean(
    'fixed_task_seed',
    False,
    'Whether to use the same task seed when running multiple task combinations.',
)

_MAX_STEPS = flags.DEFINE_integer(
    'max_steps',
    None,
    'Maximum number of steps per task. If not set, uses task complexity-based budget.',
)

_MODE = flags.DEFINE_enum(
    'mode',
    'local',
    ['local', 'endpoint'],
    'Inference mode: local (load model) or endpoint (call API server).',
)

_ENDPOINT_URL = flags.DEFINE_string(
    'endpoint_url',
    'http://127.0.0.1:8000',
    'URL of model server when using endpoint mode.',
)

_VERBOSE_TRACE = flags.DEFINE_boolean(
    'verbose_trace',
    False,
    'Whether to print per-step reasoning/action traces when backend supports it.',
)

_TRACE_HISTORY = flags.DEFINE_boolean(
    'trace_history',
    False,
    'Whether to include recent text/visual history in verbose traces.',
)

_ENDPOINT_WAIT_TIMEOUT_SEC = flags.DEFINE_integer(
    'endpoint_wait_timeout_sec',
    1800,
    'Max seconds to wait for endpoint readiness. Default 1800 s (30 min) to '
    'allow time for a 9B model to load weights and compile CUDA graphs.',
)

_ENDPOINT_WAIT_POLL_SEC = flags.DEFINE_float(
    'endpoint_wait_poll_sec',
    5.0,
    'Polling interval in seconds while waiting for endpoint readiness.',
)

_ENDPOINT_MIN_CONTEXT_LEN = flags.DEFINE_integer(
    'endpoint_min_context_len',
    0,
    'Minimum max_model_len that the OpenAI-compatible endpoint must report. '
    'Zero disables this compatibility check.',
)

_ENDPOINT_REQUIRE_LOOPBACK = flags.DEFINE_boolean(
    'endpoint_require_loopback',
    False,
    'Require endpoint_url to use localhost or an IP loopback address.',
)

_GPU_MEMORY_UTILIZATION = flags.DEFINE_float(
    'gpu_memory_utilization',
    0.75,
    'Fraction of GPU memory vLLM may use (0.0-1.0). Default 0.75 leaves '
    'headroom for other processes and avoids OOM when the GPU is partially '
    'occupied by a previous run.',
)


def _endpoint_models_url(endpoint_url: str) -> str:
    return endpoint_contract.models_url(endpoint_url)


def _fetch_served_model_records(
    models_url: str, timeout_sec: float
) -> dict[str, dict[str, object]]:
    return endpoint_contract.fetch_model_records(models_url, timeout_sec)


def _fetch_served_models(models_url: str, timeout_sec: float) -> list[str]:
    """Compatibility wrapper returning only endpoint model identifiers."""
    return list(_fetch_served_model_records(models_url, timeout_sec))


def _endpoint_context_compatible(
    record: dict[str, object], minimum_context: int
) -> bool:
    return endpoint_contract.context_compatible(
        record, minimum_context
    )


def _wait_for_endpoint_model(endpoint_url: str, model_name: str) -> None:
    models_url = _endpoint_models_url(endpoint_url)
    timeout_sec = max(1, _ENDPOINT_WAIT_TIMEOUT_SEC.value)
    poll_sec = max(0.5, _ENDPOINT_WAIT_POLL_SEC.value)
    minimum_context = max(0, _ENDPOINT_MIN_CONTEXT_LEN.value)

    print(
        f'[CONFIG] Waiting for endpoint model {model_name!r} at {models_url} '
        f'(timeout={timeout_sec}s)',
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
            f'[WAIT] Endpoint not ready: {status}', flush=True
        ),
    )
    print(
        f'[CONFIG] Endpoint ready: found exact model {model_name!r}; '
        f'max_model_len={record.get("max_model_len")!r}',
        flush=True,
    )


def _launch_local_vllm(
    model_name: str,
    device: str,
    endpoint_url: str,
    gpu_memory_utilization: float = 0.75,
) -> 'subprocess.Popen[bytes]':
    """Start a vLLM OpenAI-compatible server as a subprocess for local inference.

    MobileRL-9B is based on GLM-4.1V-9B and requires --trust-remote-code.
    The subprocess is bound to the same Python interpreter so it uses the same
    virtualenv / conda environment.

    Args:
        model_name: HuggingFace model ID (e.g. 'xuyifan/MobileRL-9B').
        device: Device string from --device flag ('cuda:0', 'cuda:1', 'auto', 'cpu').
        endpoint_url: The URL the server should listen on (host + port extracted here).
        gpu_memory_utilization: Fraction of GPU memory vLLM may use (default 0.75).
            Lower this if the GPU is partially occupied by other processes.

    Returns:
        The running subprocess handle.
    """
    parsed = urllib.parse.urlparse(endpoint_url)
    host = parsed.hostname or '127.0.0.1'
    port = parsed.port or 8000

    cmd = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', model_name,
        '--host', host,
        '--port', str(port),
        '--trust-remote-code',   # required for GLM-4.1V (MobileRL-9B base)
        '--dtype', 'bfloat16',
        '--gpu-memory-utilization', str(gpu_memory_utilization),
    ]

    env = os.environ.copy()
    if device.startswith('cuda:'):
        # Map 'cuda:N' → CUDA_VISIBLE_DEVICES=N so vLLM uses exactly that GPU.
        gpu_idx = device.split(':', 1)[1]
        env['CUDA_VISIBLE_DEVICES'] = gpu_idx
        print(f'[LOCAL] Pinning vLLM to GPU {gpu_idx} (CUDA_VISIBLE_DEVICES={gpu_idx})', flush=True)
    elif device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
        cmd += ['--device', 'cpu']
    # 'auto' or any other value: let vLLM pick GPUs itself.

    print(f'[LOCAL] Launching vLLM server: {" ".join(cmd)}', flush=True)
    proc = subprocess.Popen(cmd, env=env)
    print(f'[LOCAL] vLLM server PID={proc.pid} — waiting for model to load…', flush=True)
    return proc


def _main() -> None:
    """Runs eval suite with UI-TARS agent."""
    _vllm_proc = None
    if _MODE.value == 'local':
        # Launch a local vLLM server for the model, then wait until it is ready.
        # This is the correct interpretation of --mode=local: the model is served
        # on this machine instead of connecting to a pre-existing remote endpoint.
        _vllm_proc = _launch_local_vllm(
            _MODEL_NAME.value, _DEVICE.value, _ENDPOINT_URL.value,
            gpu_memory_utilization=_GPU_MEMORY_UTILIZATION.value,
        )
        # Ensure the server is terminated cleanly when the Python process exits.
        atexit.register(
            lambda: _vllm_proc.terminate()
            if _vllm_proc and _vllm_proc.poll() is None
            else None
        )
        _wait_for_endpoint_model(_ENDPOINT_URL.value, _MODEL_NAME.value)
    elif _MODE.value == 'endpoint':
        _wait_for_endpoint_model(_ENDPOINT_URL.value, _MODEL_NAME.value)

    env = env_launcher.load_and_setup_env(
        console_port=_DEVICE_CONSOLE_PORT.value,
        grpc_port=_GRPC_PORT.value,
        emulator_setup=_EMULATOR_SETUP.value,
        adb_path=_ADB_PATH.value,
    )

    print(f"[CONFIG] UI-TARS backend: {_UITARS_AGENT_SOURCE}")

    task_registry = registry.TaskRegistry()
    print(f"[CONFIG] Task random seed: {_TASK_RANDOM_SEED.value}")
    suite = suite_utils.create_suite(
        task_registry.get_registry(family=_SUITE_FAMILY.value),
        n_task_combinations=_N_TASK_COMBINATIONS.value,
        seed=_TASK_RANDOM_SEED.value,
        tasks=_TASKS.value,
        use_identical_params=_FIXED_TASK_SEED.value,
    )
    suite.suite_family = _SUITE_FAMILY.value

    agent_kwargs = {
        'model_name': _MODEL_NAME.value,
        'device': _DEVICE.value,
        'save_failed_tasks': _SAVE_FAILED_TASKS.value,
        'mode': _MODE.value,
        'endpoint_url': _ENDPOINT_URL.value,
    }
    init_params = set(inspect.signature(AndroidWorldUITARSAgent.__init__).parameters)
    if 'debug_verbose' in init_params:
        agent_kwargs['debug_verbose'] = _VERBOSE_TRACE.value
    if 'debug_history' in init_params:
        agent_kwargs['debug_history'] = _TRACE_HISTORY.value

    agent = AndroidWorldUITARSAgent(
        env,
        **agent_kwargs,
    )
    agent.name = 'uitars'
    
    if _MAX_STEPS.value is not None:
        agent.max_steps = _MAX_STEPS.value
        print(f"[CONFIG] Set agent.max_steps = {agent.max_steps}")
    else:
        print(f"[CONFIG] Using default task complexity-based step budgets")

    if _CHECKPOINT_DIR.value:
        checkpoint_dir = _CHECKPOINT_DIR.value
    else:
        checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)

    suite_utils.run(
        suite,
        agent,
        checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
        demo_mode=False,
    )
    env.close()


def main(argv: Sequence[str]) -> None:
    del argv
    _main()


if __name__ == '__main__':
    app.run(main)
