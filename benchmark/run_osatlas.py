"""Run OS-Atlas agent on Android World tasks.

Evaluation script for testing OS-Atlas-Pro-7B on Android World benchmark.
"""

import os
import random
import sys
from collections.abc import Sequence
from typing import Type

from absl import app, flags, logging

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from agents.osatlas.adapters.android_world import AndroidWorldOSAtlasAgent
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'Missing local adapter module: agents.osatlas.adapters.android_world. '
        'Place copied agent code under <project_root>/agents/... or install '
        'a package that provides this module.'
    ) from exc
from android_world import registry
from android_world.env import env_launcher
from android_world.task_evals import task_eval

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ.pop('GRPC_TRACE', None)


def _find_adb_directory() -> str:
    """Returns the directory where adb is located."""
    potential_paths = [
        os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
        os.path.expanduser('~/android-sdk/platform-tools/adb'),
        os.path.expanduser('~/androidsdk/platform-tools/adb'),
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

_TASK = flags.DEFINE_string(
    'task',
    None,
    'A specific task to run. If None, a random task is selected.',
)

_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    'OS-Copilot/OS-Atlas-Pro-7B',
    'HuggingFace model identifier for OS-Atlas.',
)

_DEVICE = flags.DEFINE_string(
    'device',
    'cuda:0',
    'Device to load model on. Options: cuda:0, cuda:1, auto (multi-GPU), cpu.',
)

_MAX_STEPS = flags.DEFINE_integer(
    'max_steps',
    None,
    'Maximum number of steps per task. If None, uses task.complexity * 10.',
)

_MAX_NEW_TOKENS = flags.DEFINE_integer(
    'max_new_tokens',
    256,
    'Maximum number of tokens to generate per step.',
)

_SEED = flags.DEFINE_integer(
    'seed',
    42,
    'Random seed for reproducibility. Set to None for random behavior.',
)


def _main() -> None:
    """Runs a single task with OS-Atlas agent."""
    # Set random seed for reproducibility
    if _SEED.value is not None:
        random.seed(_SEED.value)
        print(f'Random seed set to: {_SEED.value}')
    
    # Setup environment
    env = env_launcher.load_and_setup_env(
        console_port=_DEVICE_CONSOLE_PORT.value,
        grpc_port=_GRPC_PORT.value,
        emulator_setup=_EMULATOR_SETUP.value,
        adb_path=_ADB_PATH.value,
    )
    env.reset(go_home=True)
    
    # Select task
    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
    
    if _TASK.value:
        if _TASK.value not in aw_registry:
            raise ValueError(f'Task {_TASK.value} not found in registry.')
        task_type: Type[task_eval.TaskEval] = aw_registry[_TASK.value]
    else:
        task_type: Type[task_eval.TaskEval] = random.choice(
            list(aw_registry.values())
        )
    
    params = task_type.generate_random_params()
    task = task_type(params)
    task.initialize_task(env)
    
    # Initialize OS-Atlas agent
    print(f'Initializing OS-Atlas agent (model: {_MODEL_NAME.value}, device: {_DEVICE.value})...')
    agent = AndroidWorldOSAtlasAgent(
        env,
        model_name=_MODEL_NAME.value,
        device=_DEVICE.value,
        max_new_tokens=_MAX_NEW_TOKENS.value,
    )
    
    print(f'Goal: {task.goal}')
    
    # Run task
    max_steps = _MAX_STEPS.value if _MAX_STEPS.value else int(task.complexity * 10)
    is_done = False
    
    for step_num in range(max_steps):
        print(f'\n========== Step {step_num + 1}/{max_steps} ==========')
        response = agent.step(task.goal)
        
        if response.done:
            is_done = True
            break
    
    # Check success
    agent_successful = is_done and task.is_successful(env) == 1
    
    print('\n' + '='*60)
    if agent_successful:
        print(f'✓ Task Successful: {task.goal}')
    else:
        print(f'✗ Task Failed: {task.goal}')
        if not is_done:
            print(f'  Reason: Max steps ({max_steps}) reached without completion')
        else:
            print(f'  Reason: Task completed but goal not achieved')
    print('='*60)
    
    env.close()


def main(argv: Sequence[str]) -> None:
    del argv
    _main()


if __name__ == '__main__':
    app.run(main)
