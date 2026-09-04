"""Run Qwen3-VL agent on Android World tasks.

Evaluation script for testing Qwen3-VL-8B-Instruct on Android World benchmark.
"""

import hashlib
import os
import shutil
import sys
from collections.abc import Sequence

def _first_writable_dir(candidates: list[str]) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.expanduser(candidate)
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except OSError:
            continue
    return '/tmp'


_USER_NAME = os.environ.get('USER', '')
user_tmp_dir = _first_writable_dir([
    os.environ.get('TMPDIR', ''),
    os.path.join('$HOME', _USER_NAME, 'tmp') if _USER_NAME else '',
    os.path.join('$HOME', _USER_NAME, 'tmp') if _USER_NAME else '',
    '/tmp',
])
os.environ['TMPDIR'] = user_tmp_dir
os.environ['TEMP'] = user_tmp_dir
os.environ['TMP'] = user_tmp_dir


def _default_runs_root() -> str:
    explicit = os.environ.get('CATBENCH_RUNS_DIR')
    if explicit:
        return os.path.expanduser(explicit)
    return _first_writable_dir([
        os.path.join('$HOME', _USER_NAME, 'android_world_runs') if _USER_NAME else '',
        os.path.join('$HOME', _USER_NAME, 'android_world_runs') if _USER_NAME else '',
        os.path.expanduser('~/catbench_runs'),
        '/tmp/catbench_runs',
    ])


DEFAULT_RUNS_ROOT = _default_runs_root()

from absl import app, flags, logging

try:
    import pysqlite3

    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from benchmark import endpoint_contract
from android_world.agents import infer_ma3
from android_world.agents import qwen3vl
from android_world.env import env_launcher

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ.pop('GRPC_TRACE', None)


def _find_adb_directory() -> str:
    """Returns the directory where adb is located."""
    adb_in_path = shutil.which('adb')
    if adb_in_path:
        return adb_in_path
    potential_paths = [
        os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
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
        'adb not found in the common Android SDK paths. Please install Android'
        " SDK and ensure adb is in one of the expected directories."
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
    'task_random_seed', 30, 'Random seed for task randomness.'
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
    os.path.join(DEFAULT_RUNS_ROOT, 'qwen3vl'),
    'The path to save results to if checkpoint_dir is not provided.',
)

_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    'Qwen/Qwen3-VL-8B-Instruct',
    'HuggingFace model identifier for Qwen3-VL.',
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

_API_KEY = flags.DEFINE_string(
    'api_key',
    'EMPTY',
    'API key for the OpenAI-compatible endpoint.',
)

_ENDPOINT_FORMAT = flags.DEFINE_enum(
    'endpoint_format',
    'openai',
    ['openai', 'predict'],
    'Endpoint API format: openai for /v1/chat/completions, predict for CATBench /predict proxies.',
)

_SRC_FORMAT = flags.DEFINE_string(
    'src_format',
    'abs_resized',
    'Coordinate source format used by the UI-Venus Qwen3-VL action parser.',
)

_MAX_NEW_TOKENS = flags.DEFINE_integer(
    'max_new_tokens',
    512,
    'Maximum number of tokens to generate.',
)

_FAILED_TASKS_DIR = flags.DEFINE_string(
    'failed_tasks_dir',
    '',
    'Directory to save debug images and traces. By default, each worker uses '
    '<output_path>/debug_qwen3vl so concurrent app workers never share files.',
)
_ENDPOINT_WAIT_TIMEOUT_SEC = flags.DEFINE_integer(
    'endpoint_wait_timeout_sec', 1800, 'Max seconds to wait for endpoint.'
)
_ENDPOINT_WAIT_POLL_SEC = flags.DEFINE_float(
    'endpoint_wait_poll_sec', 5.0, 'Endpoint readiness polling interval.'
)
_ENDPOINT_MIN_CONTEXT_LEN = flags.DEFINE_integer(
    'endpoint_min_context_len',
    16384,
    'Minimum integer max_model_len advertised by /v1/models.',
)
_ENDPOINT_REQUIRE_LOOPBACK = flags.DEFINE_boolean(
    'endpoint_require_loopback',
    True,
    'Require endpoint_url to use localhost or an IP loopback address.',
)


def _main() -> None:
    """Runs eval suite with Qwen3-VL agent."""
    if _MODE.value != 'endpoint':
        raise ValueError(
            'run_qwen3vl.py now uses the local UI-Venus AndroidWorld Qwen3-VL '
            'agent, which expects an OpenAI-compatible endpoint. Use '
            '--mode=endpoint and serve the model with vLLM/SGLang first.'
        )

    if _ENDPOINT_FORMAT.value == 'openai':
        record = endpoint_contract.wait_for_model(
            _ENDPOINT_URL.value,
            _MODEL_NAME.value,
            max(0, _ENDPOINT_MIN_CONTEXT_LEN.value),
            timeout_sec=max(1, _ENDPOINT_WAIT_TIMEOUT_SEC.value),
            poll_sec=max(0.5, _ENDPOINT_WAIT_POLL_SEC.value),
            loopback_only=_ENDPOINT_REQUIRE_LOOPBACK.value,
            status_callback=lambda status: print(
                f'Waiting for endpoint: {status}', flush=True
            ),
        )
        print(
            f'Endpoint ready: exact model={_MODEL_NAME.value!r}; '
            f'max_model_len={record.get("max_model_len")!r}',
            flush=True,
        )
    else:
        if _ENDPOINT_MIN_CONTEXT_LEN.value > 0:
            raise endpoint_contract.EndpointContractError(
                'A positive --endpoint_min_context_len requires '
                '--endpoint_format=openai so /v1/models can be attested.'
            )
        if _ENDPOINT_REQUIRE_LOOPBACK.value:
            endpoint_contract.require_loopback(_ENDPOINT_URL.value)

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

    endpoint_url = _ENDPOINT_URL.value.rstrip('/')
    if _ENDPOINT_FORMAT.value == 'openai':
        endpoint_base_url = endpoint_url
        if not endpoint_base_url.endswith('/v1'):
            endpoint_base_url = f'{endpoint_base_url}/v1'
        print(
            'Initializing Qwen3-VL agent from UI-Venus implementation '
            f'(model: {_MODEL_NAME.value}, endpoint: {endpoint_base_url})...'
        )
        wrapper = infer_ma3.Qwen3VLWrapper(
            _API_KEY.value,
            endpoint_base_url,
            _MODEL_NAME.value,
        )
    else:
        print(
            'Initializing Qwen3-VL agent from /predict endpoint '
            f'(model: {_MODEL_NAME.value}, endpoint: {endpoint_url})...'
        )
        wrapper = infer_ma3.Qwen3VLPredictWrapper(
            endpoint_url,
            max_tokens=_MAX_NEW_TOKENS.value,
        )
    
    failed_tasks_dir = _FAILED_TASKS_DIR.value
    if _SAVE_FAILED_TASKS.value and not failed_tasks_dir:
        # A matrix launches one process per app.  A global debug directory
        # makes those processes overwrite screenshot_N.png and trace.jsonl.
        # Deadline runs may also put these high-frequency, non-evaluation
        # traces on local storage so NFS is reserved for checkpoints.
        debug_root = os.environ.get('CATBENCH_QWEN_DEBUG_ROOT', '').strip()
        if debug_root:
            worker_key = hashlib.sha256(
                os.path.abspath(_OUTPUT_PATH.value).encode('utf-8')
            ).hexdigest()[:16]
            failed_tasks_dir = os.path.join(debug_root, worker_key)
        else:
            failed_tasks_dir = os.path.join(_OUTPUT_PATH.value, 'debug_qwen3vl')

    agent = qwen3vl.Qwen3_VL(
        env,
        wrapper,
        _SRC_FORMAT.value,
        api_key=_API_KEY.value,
        url=_ENDPOINT_URL.value,
        output_path=failed_tasks_dir if _SAVE_FAILED_TASKS.value else '',
    )
    agent.name = 'qwen3vl'
    if hasattr(agent, 'get_task_name'):
        agent.get_task_name(suite)
    
    if _MAX_STEPS.value is not None:
        agent.max_steps = _MAX_STEPS.value
        print(f'Using fixed max_steps: {_MAX_STEPS.value}')
    else:
        print('Using task complexity-based step budget')

    if _CHECKPOINT_DIR.value:
        checkpoint_dir = _CHECKPOINT_DIR.value
    else:
        checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)

    print(f'Starting eval with Qwen3-VL and writing to {checkpoint_dir}')
    suite_utils.run(
        suite,
        agent,
        checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
        demo_mode=False,
    )
    print(f'Finished running Qwen3-VL on {_SUITE_FAMILY.value} family. Wrote to {checkpoint_dir}.')
    env.close()


def main(argv: Sequence[str]) -> None:
    del argv
    _main()


if __name__ == '__main__':
    app.run(main)
