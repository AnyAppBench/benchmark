# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run eval suite.

The run.py module is used to run a suite of tasks, with configurable task
combinations, environment setups, and agent configurations. You can run specific
tasks or all tasks in the suite and customize various settings using the
command-line flags.
"""

from collections.abc import Sequence
import os
import sys

# Fix for SQLite FTS4 support (needed for Joplin app setup)
import pysqlite3
sys.modules['sqlite3'] = sys.modules['pysqlite3']

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import human_agent
from android_world.agents import m3a
from android_world.agents import random_agent
from android_world.env import env_launcher
from android_world.env import interface
from agent_import_utils import resolve_uitars_agent_class

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ.pop('GRPC_TRACE', None)  # Unset: disabling trace does not use a 'none' tracer.

try:
  from android_world.agents import qwen_vlm as _qwen_vlm
  _DEFAULT_QWEN_MODEL_ID = _qwen_vlm.DEFAULT_MODEL_ID
except ModuleNotFoundError:
  _qwen_vlm = None
  _DEFAULT_QWEN_MODEL_ID = 'namhokaist/Qwen2.5-VL-7B-ContinueTrain-v2'


def _find_adb_directory() -> str:
  """Returns the directory where adb is located."""
  potential_paths = [
      '/opt/android/platform-tools/adb',
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
      " SDK and ensure adb is in one of the expected directories. If it's"
      ' already installed, point to the installed location.'
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
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
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
        # Families from the paper.
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        # Other families for more testing.
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
    'List of specific tasks to run in the given suite family. If None, run all'
    ' tasks in the suite family.',
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    1,
    'Number of task instances to run for each task template.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'The directory to save checkpoints and resume evaluation from. If the'
    ' directory contains existing checkpoint files, evaluation will resume from'
    ' the latest checkpoint. If the directory is empty or does not exist, a new'
    ' directory will be created.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    os.path.expanduser('~/android_world/runs'),
    'The path to save results to if not resuming from a checkpoint is not'
    ' provided.',
)

# Agent specific.
_AGENT_NAME = flags.DEFINE_string('agent_name', 'm3a_gpt4v', help='Agent name.')
_GPT_MODEL_NAME = flags.DEFINE_string(
    'gpt_model_name',
    'gpt-4-turbo-2024-04-09',
    'OpenAI model name used by GPT-based agents (for example gpt-5.1).',
)
_GPT_BASE_URL = flags.DEFINE_string(
    'gpt_base_url',
    'https://api.openai.com/v1/chat/completions',
    'OpenAI-compatible chat completions URL for GPT-based agents.',
)
_GEMINI_MODEL_NAME = flags.DEFINE_string(
    'gemini_model_name',
    'gemini-3.1-pro-preview',
    'Gemini model name for Gemini-based agents.',
)
_GEMINI_PROXY_BASE_URL = flags.DEFINE_string(
    'gemini_proxy_base_url',
    '',
    'OpenAI-compatible proxy URL for --agent_name=m3a_gemini_proxy.',
)
_CLAUDE_MODEL_NAME = flags.DEFINE_string(
    'claude_model_name',
    'claude-sonnet-4-6',
    'Claude model name for Claude proxy agents.',
)
_CLAUDE_PROXY_BASE_URL = flags.DEFINE_string(
    'claude_proxy_base_url',
    '',
    'OpenAI-compatible proxy URL for --agent_name=m3a_claude_proxy.',
)
_QWEN_VERBOSE_TRACE = flags.DEFINE_boolean(
  'qwen_verbose_trace',
  False,
  'Whether to print per-step Qwen agent thought/action traces for debugging.',
)
_QWEN_TRACE_HISTORY = flags.DEFINE_boolean(
  'qwen_trace_history',
  False,
  'Whether to include recent text/visual history in Qwen debug traces.',
)
_QWEN_MODEL_ID = flags.DEFINE_string(
  'qwen_model_id',
  _DEFAULT_QWEN_MODEL_ID,
  'Hugging Face model id for the qwen_vlm agent.',
)
_QWEN_ALLOW_INFEASIBLE = flags.DEFINE_boolean(
  'qwen_allow_infeasible',
  False,
  'Whether the qwen_vlm agent is allowed to terminate tasks with'
  ' infeasible(). Disabled by default for benchmark runs.',
)
_QWEN3VL_MODEL_NAME = flags.DEFINE_string(
  'qwen3vl_model_name',
  'Qwen3-VL-8B',
  'Served model name for --agent_name=qwen3vl.',
)
_QWEN3VL_BASE_URL = flags.DEFINE_string(
  'qwen3vl_base_url',
  'http://127.0.0.1:8000/v1',
  'OpenAI-compatible /v1 base URL for --agent_name=qwen3vl.',
)
_QWEN3VL_API_KEY = flags.DEFINE_string(
  'qwen3vl_api_key',
  'EMPTY',
  'API key for --agent_name=qwen3vl.',
)
_AUTODEV_PLANNER_MODEL = flags.DEFINE_string(
  'autodev_planner_model',
  os.environ.get('AUTODEV_PLANNER_MODEL', 'openai/gpt-4o'),
  'LiteLLM model name for AutoDev planner, e.g. openai/gpt-4o.',
)
_AUTODEV_EXECUTOR_MODEL = flags.DEFINE_string(
  'autodev_executor_model',
  os.environ.get('AUTODEV_EXECUTOR_MODEL', 'openai/gpt-4o'),
  'LiteLLM model name for AutoDev executor.',
)
_AUTODEV_TRANSCRIBER_MODEL = flags.DEFINE_string(
  'autodev_transcriber_model',
  os.environ.get('AUTODEV_TRANSCRIBER_MODEL', 'openai/gpt-4o'),
  'LiteLLM model name for AutoDev screen transcription.',
)
_AUTODEV_BASE_URL = flags.DEFINE_string(
  'autodev_base_url',
  os.environ.get('AUTODEV_BASE_URL', ''),
  'OpenAI-compatible /v1 base URL for AutoDev. Use a local vLLM URL to avoid '
  'cloud API calls. Leave empty for provider-native LiteLLM routing.',
)
_AUTODEV_API_KEY = flags.DEFINE_string(
  'autodev_api_key',
  os.environ.get('AUTODEV_API_KEY', ''),
  'API key for AutoDev. Empty lets the selected provider read its normal env '
  'vars; use EMPTY for local OpenAI-compatible servers.',
)
_AUTODEV_LOG_DIR = flags.DEFINE_string(
  'autodev_log_dir',
  os.environ.get('AUTODEV_LOG_DIR', ''),
  'Directory for AutoDev planner/executor traces. Defaults to output_path/autodev_logs.',
)
_AUTODEV_ENABLE_LOGGING = flags.DEFINE_boolean(
  'autodev_enable_logging',
  True,
  'Whether AutoDev should save detailed planner/executor traces.',
)
_AUTODEV_SCALE = flags.DEFINE_float(
  'autodev_scale',
  0.4,
  'Scale factor for screenshots sent to AutoDev models.',
)
_DEVICE = flags.DEFINE_string(
    'device',
    'cuda:0',
    'Device for UI-TARS agent (cuda:0, cuda:1, auto for multi-GPU, or cpu).',
)

_SAVE_FAILED_TASKS = flags.DEFINE_boolean(
    'save_failed_tasks',
    True,
    'Save screenshots and traces for failed tasks (UI-TARS, OS-Atlas).',
)

_FIXED_TASK_SEED = flags.DEFINE_boolean(
    'fixed_task_seed',
    False,
    'Whether to use the same task seed when running multiple task combinations'
    ' (n_task_combinations > 1).',
)

_MAX_STEPS = flags.DEFINE_integer(
  'max_steps',
  None,
  'Minimum maximum number of steps per task. If set, this value is compared'
  ' against the task complexity budget and the higher value is used.',
)


# MiniWoB is very lightweight and new screens/View Hierarchy load quickly.
_MINIWOB_TRANSITION_PAUSE = 0.2

# Additional guidelines for the MiniWob tasks.
_MINIWOB_ADDITIONAL_GUIDELINES = [
    (
        'This task is running in a mock app, you must stay in this app and'
        ' DO NOT use the `navigate_home` action.'
    ),
]


def _get_agent(
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
  """Gets agent."""
  print('Initializing agent...')
  agent = None
  if _AGENT_NAME.value == 'human_agent':
    agent = human_agent.HumanAgent(env)
  elif _AGENT_NAME.value == 'random_agent':
    agent = random_agent.RandomAgent(env)
  # Gemini.
  elif _AGENT_NAME.value == 'm3a_gemini_gcp':
    from android_world.agents import infer as infer_module
    agent = m3a.M3A(
        env, infer_module.GeminiGcpWrapper(model_name=_GEMINI_MODEL_NAME.value)
    )
  elif _AGENT_NAME.value == 'm3a_gemini_proxy':
    from android_world.agents import infer as infer_module
    agent = m3a.M3A(
        env,
        infer_module.GeminiProxyWrapper(
            model_name=_GEMINI_MODEL_NAME.value,
            base_url=_GEMINI_PROXY_BASE_URL.value,
        ),
    )
  elif _AGENT_NAME.value == 'm3a_claude_proxy':
    from android_world.agents import infer as infer_module
    agent = m3a.M3A(
        env,
        infer_module.ClaudeProxyWrapper(
            model_name=_CLAUDE_MODEL_NAME.value,
            base_url=_CLAUDE_PROXY_BASE_URL.value,
        ),
    )
  elif _AGENT_NAME.value == 't3a_gemini_gcp':
    try:
      from android_world.agents import t3a as t3a_module
    except ImportError as exc:
      raise ImportError(
          't3a agent requested but android_world.agents.t3a is not available in '
          'this checkout. Use --agent_name=m3a_gpt4v (or another available '
          'agent), or restore the t3a module.'
      ) from exc
    from android_world.agents import infer as infer_module
    agent = t3a_module.T3A(
        env, infer_module.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  # GPT.
  elif _AGENT_NAME.value == 't3a_gpt4':
    try:
      from android_world.agents import t3a as t3a_module
    except ImportError as exc:
      raise ImportError(
          't3a agent requested but android_world.agents.t3a is not available in '
          'this checkout. Use --agent_name=m3a_gpt4v (or another available '
          'agent), or restore the t3a module.'
      ) from exc
    from android_world.agents import infer as infer_module
    agent = t3a_module.T3A(
        env, infer_module.Gpt4Wrapper(_GPT_MODEL_NAME.value)
    )
  elif _AGENT_NAME.value == 'm3a_gpt4v':
    from android_world.agents import infer as infer_module
    agent = m3a.M3A(
        env,
        infer_module.Gpt4Wrapper(
            _GPT_MODEL_NAME.value,
            base_url=_GPT_BASE_URL.value,
        ),
    )
  elif _AGENT_NAME.value == 'qwen_vlm':
    qwen_module = _qwen_vlm
    if qwen_module is None:
      try:
        from android_world.agents import qwen_vlm as qwen_module
      except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'qwen_vlm dependencies are missing (e.g., torch/transformers). '
            'Install optional dependencies or choose another --agent_name.'
        ) from exc

    agent = qwen_module.QwenVlmAgent(
        env,
        model_id=_QWEN_MODEL_ID.value,
        debug_verbose=_QWEN_VERBOSE_TRACE.value,
        debug_history=_QWEN_TRACE_HISTORY.value,
        allow_infeasible=_QWEN_ALLOW_INFEASIBLE.value,
    )
  # SeeAct.
  elif _AGENT_NAME.value == 'seeact':
    try:
      from android_world.agents import seeact as seeact_module
    except Exception as exc:  # pylint: disable=broad-exception-caught
      raise ModuleNotFoundError(
          'seeact dependencies are unavailable in the current environment. '
          'Install/repair seeact optional dependencies (for example '
          'matplotlib/NumPy compatibility), or choose another --agent_name. '
          f'Underlying error: {exc}'
      ) from exc
    agent = seeact_module.SeeAct(env)
  # UI-TARS.
  elif _AGENT_NAME.value == 'uitars':
    AndroidWorldUITARSAgent, uitars_source = resolve_uitars_agent_class(
      project_root
    )
    print(f'Using UI-TARS backend: {uitars_source}')
    agent = AndroidWorldUITARSAgent(
        env, 
        device=_DEVICE.value,
        save_failed_tasks=_SAVE_FAILED_TASKS.value,
    )
  # OS-Atlas.
  elif _AGENT_NAME.value == 'osatlas':
    from agents.osatlas.adapters.android_world import AndroidWorldOSAtlasAgent
    agent = AndroidWorldOSAtlasAgent(
        env,
        device=_DEVICE.value,
        save_failed_tasks=_SAVE_FAILED_TASKS.value,
    )
  # Qwen3-VL.
  elif _AGENT_NAME.value == 'qwen3vl':
    from android_world.agents import infer_ma3
    from android_world.agents import qwen3vl
    base_url = _QWEN3VL_BASE_URL.value.rstrip('/')
    if not base_url.endswith('/v1'):
      base_url = f'{base_url}/v1'
    agent = qwen3vl.Qwen3_VL(
        env,
        infer_ma3.Qwen3VLWrapper(
            _QWEN3VL_API_KEY.value,
            base_url,
            _QWEN3VL_MODEL_NAME.value,
        ),
        'abs_resized',
        api_key=_QWEN3VL_API_KEY.value,
        url=base_url,
    )
  elif _AGENT_NAME.value == 'autodev_agent':
    from android_world.agents import autodev_agent
    log_dir = _AUTODEV_LOG_DIR.value or os.path.join(
        _OUTPUT_PATH.value, 'autodev_logs'
    )
    agent = autodev_agent.AutoDev(
        env,
        name='autodev_agent',
        scale=_AUTODEV_SCALE.value,
        log_dir=log_dir,
        enable_logging=_AUTODEV_ENABLE_LOGGING.value,
        planner_model=_AUTODEV_PLANNER_MODEL.value,
        executor_model=_AUTODEV_EXECUTOR_MODEL.value,
        transcriber_model=_AUTODEV_TRANSCRIBER_MODEL.value,
        api_key=_AUTODEV_API_KEY.value,
        base_url=_AUTODEV_BASE_URL.value,
    )

  if not agent:
    raise ValueError(f'Unknown agent: {_AGENT_NAME.value}')

  if (
      agent.name in ['M3A', 'T3A', 'SeeAct']
      and family
      and family.startswith('miniwob')
      and hasattr(agent, 'set_task_guidelines')
  ):
    agent.set_task_guidelines(_MINIWOB_ADDITIONAL_GUIDELINES)
  agent.name = _AGENT_NAME.value

  return agent


def _main() -> None:
  """Runs eval suite and gets rewards back."""
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      grpc_port=_GRPC_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
  )

  n_task_combinations = _N_TASK_COMBINATIONS.value
  task_registry = registry.TaskRegistry()
  suite = suite_utils.create_suite(
      task_registry.get_registry(family=_SUITE_FAMILY.value),
      n_task_combinations=n_task_combinations,
      seed=_TASK_RANDOM_SEED.value,
      tasks=_TASKS.value,
      use_identical_params=_FIXED_TASK_SEED.value,
  )
  suite.suite_family = _SUITE_FAMILY.value

  agent = _get_agent(env, _SUITE_FAMILY.value)

  if _MAX_STEPS.value is not None:
    agent.max_steps = _MAX_STEPS.value
    print(f'[CONFIG] Set agent.max_steps = {agent.max_steps}')
  else:
    print('[CONFIG] Using default task complexity-based step budgets')

  if _SUITE_FAMILY.value.startswith('miniwob'):
    # MiniWoB pages change quickly, don't need to wait for screen to stabilize.
    agent.transition_pause = _MINIWOB_TRANSITION_PAUSE
  else:
    agent.transition_pause = None

  if _CHECKPOINT_DIR.value:
    checkpoint_dir = _CHECKPOINT_DIR.value
  else:
    checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)

  print(
      f'Starting eval with agent {_AGENT_NAME.value} and writing to'
      f' {checkpoint_dir}'
  )
  suite_utils.run(
      suite,
      agent,
      checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
      demo_mode=False,
  )
  print(
      f'Finished running agent {_AGENT_NAME.value} on {_SUITE_FAMILY.value}'
      f' family. Wrote to {checkpoint_dir}.'
  )
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
