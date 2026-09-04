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

"""Utilities for evaluating automation agents."""

import collections
import copy
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import signal
import threading
import time
import traceback
from typing import Any, Callable, Type, TypeVar

from android_env import env_interface
from android_world import checkpointer as checkpointer_lib
from android_world import constants
from android_world import episode_runner
from android_world.agents import base_agent
from android_world.agents import episode_exceptions
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import runtime_health
from android_world.task_evals import task_eval
from android_world.task_evals.miniwob import miniwob_base
from fuzzywuzzy import process
import jsonschema
import numpy as np
import pandas as pd

import exact_task_params as exact_task_params_lib

try:
  import task_breakdowns as task_breakdowns_lib
except ImportError:
  task_breakdowns_lib = None

# A fixed seed to use when use identical parameters but seed is not set.
_FIXED_SEED = 123
_TASK_TEMPLATE_COLUMN = 'task_template'
_CATBENCH_INSTANCE_ID_ATTR = '_catbench_instance_id'
_CATBENCH_EXACT_GOAL_ATTR = '_catbench_exact_goal'
_CATBENCH_CLASS_RENDERED_GOAL_ATTR = '_catbench_class_rendered_goal'
_CATBENCH_INSTANCE_ID_ENV = 'CATBENCH_INSTANCE_ID'
_CATBENCH_CONDITIONS = frozenset({'c1', 'c2_g', 'c2_o'})
_TASK_PROMPT_COLUMN = 'task_prompt'
TaskEvalType = TypeVar('TaskEvalType', bound=task_eval.TaskEval)


def _effective_task_goal(task: task_eval.TaskEval) -> str:
  """Returns the one goal used by prompts, results, logs, and debug paths."""
  exact_goal = getattr(task, _CATBENCH_EXACT_GOAL_ATTR, None)
  if exact_goal is not None:
    if not isinstance(exact_goal, str) or not exact_goal:
      raise ValueError(f'Invalid exact goal attached to {task.name}.')
    return exact_goal
  return task.goal


def _class_rendered_task_goal(task: task_eval.TaskEval) -> str:
  """Returns the pre-override class-rendered goal for provenance."""
  if hasattr(task, _CATBENCH_CLASS_RENDERED_GOAL_ATTR):
    return getattr(task, _CATBENCH_CLASS_RENDERED_GOAL_ATTR)
  return task.goal


def _load_exact_task_params_override(
) -> exact_task_params_lib.ExactTaskParamsBundle | None:
  """Loads the narrowly authorized instance-0 replacement contract.

  The orchestration layer supplies a per-job projection whose task set must
  exactly equal the runner's ``--tasks`` selection.  Requiring replacement
  provenance here prevents this mechanism from being inherited by a primary
  or ordinary matrix run.
  """
  values = {
      name: os.environ.get(name, '').strip()
      for name in exact_task_params_lib.ENV_NAMES
  }
  if not any(values.values()):
    return None
  required = (
      exact_task_params_lib.ENV_FILE,
      exact_task_params_lib.ENV_SHA256,
      exact_task_params_lib.ENV_MODE,
      exact_task_params_lib.ENV_SOURCE_FILE,
      exact_task_params_lib.ENV_SOURCE_SHA256,
      exact_task_params_lib.ENV_GOAL_OVERRIDE_ENABLED,
      exact_task_params_lib.ENV_GOAL_MAPPING_SHA256,
  )
  missing = [name for name in required if not values[name]]
  if missing:
    raise ValueError(
        'Incomplete exact task-parameter override environment: missing '
        + ', '.join(missing)
    )
  source_file = values[exact_task_params_lib.ENV_SOURCE_FILE]
  source_sha256 = values[exact_task_params_lib.ENV_SOURCE_SHA256]
  if values[exact_task_params_lib.ENV_MODE] != exact_task_params_lib.MODE:
    raise ValueError(
        'Unsupported exact task-parameter override mode: '
        f"{values[exact_task_params_lib.ENV_MODE]!r}."
    )
  if values[exact_task_params_lib.ENV_GOAL_OVERRIDE_ENABLED] != '1':
    raise ValueError(
        'Exact historical task parameters require the centralized archived-'
        'goal override contract.'
    )
  replacement_contract = {
      'CATBENCH_CONDITION': 'c1',
      'CATBENCH_RELEASE_PURPOSE': 'revision_rerun_candidate',
      'CATBENCH_ARTIFACT_ROLE': 'invalid_episode_replacement_candidate',
      'CATBENCH_ANALYSIS_ELIGIBLE': '0',
  }
  mismatches = [
      f'{name}={os.environ.get(name, "")!r} (expected {expected!r})'
      for name, expected in replacement_contract.items()
      if os.environ.get(name, '') != expected
  ]
  if mismatches:
    raise ValueError(
        'Exact task-parameter overrides are restricted to analysis-ineligible '
        'C1 invalid-episode replacement candidates: ' + '; '.join(mismatches)
    )
  bundle = exact_task_params_lib.load_bundle(
      Path(values[exact_task_params_lib.ENV_FILE]),
      expected_sha256=values[exact_task_params_lib.ENV_SHA256],
      expected_mode=values[exact_task_params_lib.ENV_MODE],
  )
  if (
      values[exact_task_params_lib.ENV_GOAL_MAPPING_SHA256].lower()
      != bundle.sha256
  ):
    raise ValueError(
        'Exact archived-goal mapping SHA-256 must equal the effective '
        'per-job override-file SHA-256.'
    )
  source_path = Path(source_file).expanduser().resolve()
  expected_source_hash = source_sha256.lower()
  if source_path != bundle.source_path:
    raise ValueError(
        'Exact task-parameter source path differs between the environment and '
        f'the pinned override file: {source_path} != {bundle.source_path}'
    )
  if expected_source_hash != bundle.source_sha256:
    raise ValueError(
        'Exact task-parameter source SHA-256 differs between the environment '
        'and the pinned override file.'
    )
  return bundle


def _instantiate_exact_override_task(
    name: str,
    task_type: Type[task_eval.TaskEval],
    entry: dict[str, Any],
    env: interface.AsyncEnv | None,
) -> task_eval.TaskEval:
  """Instantiates and independently validates one exact task override."""
  params = copy.deepcopy(entry['params'])
  try:
    task_instance = _instantiate_task(task_type, params=params, env=env)
  except Exception as error:
    raise ValueError(
        f'Unable to instantiate exact task-parameter override for {name}: '
        f'{error}'
    ) from error
  if task_instance.name != name:
    raise ValueError(
        f'Exact override task-name mismatch: registry={name!r}, '
        f'instance={task_instance.name!r}.'
    )
  if task_instance.params != entry['params']:
    raise ValueError(
        f'Exact override params were modified while instantiating {name}.'
    )
  if task_instance.params.get(constants.EpisodeConstants.SEED) != entry[
      'expected_seed'
  ]:
    raise ValueError(f'Exact override seed mismatch after instantiating {name}.')
  schema = getattr(task_instance, 'schema', None)
  if not isinstance(schema, dict):
    raise ValueError(f'Task {name} does not expose an object JSON schema.')
  try:
    validator_type = jsonschema.validators.validator_for(schema)
    validator_type.check_schema(schema)
    validator_type(schema).validate(task_instance.params)
  except Exception as error:  # jsonschema has version-specific ref errors.
    raise ValueError(
        f'Exact override params fail the task schema for {name}: {error}'
    ) from error
  try:
    class_rendered_goal = task_instance.goal
  except Exception as error:
    raise ValueError(
        f'Unable to render the current class goal for exact override {name}: '
        f'{error}'
    ) from error
  setattr(
      task_instance,
      _CATBENCH_CLASS_RENDERED_GOAL_ATTR,
      class_rendered_goal,
  )
  # This is the only permitted goal substitution: the byte-pinned historical
  # C1 instruction within the K=1 invalid-episode replacement contract.
  setattr(task_instance, _CATBENCH_EXACT_GOAL_ATTR, entry['expected_goal'])
  if _effective_task_goal(task_instance) != entry['expected_goal']:
    raise ValueError(
        f'Exact effective-goal mismatch after instantiating {name}.'
    )
  setattr(task_instance, _CATBENCH_INSTANCE_ID_ATTR, 0)
  return task_instance


def validate_exact_task_params_bundle(
    task_registry: dict[str, Type[task_eval.TaskEval]],
    bundle: exact_task_params_lib.ExactTaskParamsBundle,
    task_names: list[str],
    env: interface.AsyncEnv | None = None,
) -> None:
  """Validates every exact entry against the current task implementation."""
  exact_task_params_lib.require_exact_task_names(
      bundle, task_names, registry_names=task_registry
  )
  validate_exact_task_params_entries(
      task_registry, bundle.overrides, task_names, env
  )


def validate_exact_task_params_entries(
    task_registry: dict[str, Type[task_eval.TaskEval]],
    overrides: dict[str, dict[str, Any]],
    task_names: list[str],
    env: interface.AsyncEnv | None = None,
) -> None:
  """Validates an explicitly projected exact-entry set one-to-one."""
  if len(task_names) != len(set(task_names)):
    raise ValueError('Projected exact task names contain duplicates.')
  expected = set(task_names)
  actual = set(overrides)
  if actual != expected:
    raise ValueError(
        'Projected exact task-parameter set mismatch: '
        f'missing={sorted(expected - actual)}, extra={sorted(actual - expected)}'
    )
  unknown = sorted(actual - set(task_registry))
  if unknown:
    raise ValueError(
        'Projected exact tasks are absent from the registry: '
        + ', '.join(unknown)
    )
  for name in task_names:
    _instantiate_exact_override_task(
        name, task_registry[name], overrides[name], env
    )


def _catbench_verifier_settle_attempts() -> int:
  """Number of bounded verifier reads after a CATBench agent action."""
  raw = os.environ.get('CATBENCH_VERIFIER_SETTLE_ATTEMPTS', '3')
  try:
    attempts = int(raw)
  except ValueError as error:
    raise ValueError(
        'CATBENCH_VERIFIER_SETTLE_ATTEMPTS must be an integer; '
        f'got {raw!r}.'
    ) from error
  if not 1 <= attempts <= 20:
    raise ValueError(
        'CATBENCH_VERIFIER_SETTLE_ATTEMPTS must be in [1, 20]; '
        f'got {attempts}.'
    )
  return attempts


def _catbench_verifier_settle_interval_seconds() -> float:
  """Delay between bounded CATBench verifier reads."""
  raw = os.environ.get('CATBENCH_VERIFIER_SETTLE_INTERVAL_SECONDS', '0.2')
  try:
    interval = float(raw)
  except ValueError as error:
    raise ValueError(
        'CATBENCH_VERIFIER_SETTLE_INTERVAL_SECONDS must be numeric; '
        f'got {raw!r}.'
    ) from error
  if not 0 <= interval <= 5:
    raise ValueError(
        'CATBENCH_VERIFIER_SETTLE_INTERVAL_SECONDS must be in [0, 5]; '
        f'got {interval}.'
    )
  return interval


def _read_task_success_with_settle(
    task: task_eval.TaskEval, env: interface.AsyncEnv
) -> float:
  """Reads a CATBench verifier with a bounded durable-write retry.

  Android apps commonly commit provider/SQLite state shortly after the UI
  action returns. A single immediate read can therefore turn a correct action
  into a timing-dependent failure. Generated CATBench tasks opt in through
  ``catbench_semantic_id``; upstream AndroidWorld tasks retain their historical
  single-read contract.

  Exceptions deliberately propagate. A failed native-state read is an invalid
  verifier/environment attempt, never evidence of task failure.
  """
  attempts = (
      _catbench_verifier_settle_attempts()
      if getattr(task, 'catbench_semantic_id', None)
      else 1
  )
  interval = (
      _catbench_verifier_settle_interval_seconds() if attempts > 1 else 0.0
  )
  score = 0.0
  for attempt in range(attempts):
    score = task.is_successful(env)
    if score == 1.0 or attempt == attempts - 1:
      return score
    if interval:
      time.sleep(interval)
  return score


class _CatbenchTaskTimeoutError(BaseException):
  """Raised when one AndroidWorld task exceeds the CATBench task budget."""


def _catbench_task_timeout_seconds() -> int:
  value = os.environ.get('CATBENCH_TASK_TIMEOUT_SECONDS', '')
  return int(value) if value else 0


def _run_with_catbench_task_timeout(task_name: str, fn: Callable[[], Any]) -> Any:
  timeout = _catbench_task_timeout_seconds()
  if timeout <= 0 or threading.current_thread() is not threading.main_thread():
    return fn()

  def _handle_timeout(signum, frame):  # pylint: disable=unused-argument
    raise _CatbenchTaskTimeoutError(
        f'Task {task_name} exceeded {timeout} seconds'
    )

  previous_handler = signal.getsignal(signal.SIGALRM)
  previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout)
  signal.signal(signal.SIGALRM, _handle_timeout)
  try:
    return fn()
  finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous_handler)
    if previous_timer[0] > 0:
      signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _is_catbench_timeout_control_flow(error: BaseException) -> bool:
  return error.__class__.__name__ in {
      '_CatbenchTaskTimeoutError',
      '_AgentProgPipelineTimeoutError',
  }


class Suite(dict[str, list[task_eval.TaskEval]]):
  """A suite of tasks.

  Each key is the task name as defined in registry.py and its value is a list
  of instantiated task objects. These instances differ from each other by their
  parameter initializations; i.e. each task will have different task parameters.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._suite_family = None

  @property
  def suite_family(self) -> str:
    """Getter for suite_family."""
    if self._suite_family is None:
      raise ValueError('Suite family is not set; please first set it.')
    return self._suite_family

  @suite_family.setter
  def suite_family(self, value: str):
    """Setter for suite_family."""
    self._suite_family = value


def _log_and_print(msg: str, *args: object) -> None:
  formatted = msg % args if args else msg
  logging.info(formatted)
  print(formatted)


def _lookup_task_breakdown_context(
    task: task_eval.TaskEval,
    effective_goal: str,
) -> dict[str, Any]:
  if task_breakdowns_lib is None:
    if os.environ.get('CATBENCH_TASK_BREAKDOWN_FILE'):
      raise ImportError(
          'CATBENCH_TASK_BREAKDOWN_FILE is set, but task_breakdowns.py could '
          'not be imported. Run from the benchmark environment or add '
          'benchmark/ to PYTHONPATH.'
      )
    return {
        'enabled': False,
        'found': False,
        'prompt_goal': effective_goal,
        'task_breakdown_text': '',
        'task_breakdown_metadata': {},
    }
  return task_breakdowns_lib.build_prompt_context(
      task.name,
      effective_goal,
      getattr(task, _CATBENCH_INSTANCE_ID_ATTR, None),
  )


def _get_task_breakdown_context(
    task: task_eval.TaskEval,
    *,
    strict: bool = True,
) -> dict[str, Any]:
  """Returns the breakdown prompt context for ``task``.

  ``strict=True`` (the episode path) lets lookup errors propagate so that
  ``CATBENCH_TASK_BREAKDOWN_REQUIRED=1`` fails the episode.  ``strict=False``
  (the result-annotation path) captures the same error into the returned
  context instead: annotating an already-failed episode must never re-raise
  inside the ``_run_task`` exception handler, which would abort the whole
  suite rather than record one attributed invalid episode.
  """
  effective_goal = _effective_task_goal(task)
  try:
    return _lookup_task_breakdown_context(task, effective_goal)
  except Exception as error:  # pylint: disable=broad-exception-caught
    if strict:
      raise
    logging.exception(
        'Task-breakdown lookup failed while annotating %s: %s',
        task.name,
        error,
    )
    error_text = f'{type(error).__name__}: {error}'
    return {
        'enabled': True,
        'found': False,
        'status': 'lookup_error',
        'error': error_text,
        'source_file': os.environ.get('CATBENCH_TASK_BREAKDOWN_FILE', ''),
        'prompt_goal': effective_goal,
        'task_breakdown_text': '',
        'task_breakdown_metadata': {
            'lookup_error': True,
            'error': error_text,
            'task_template': task.name,
            'instance_id': getattr(task, _CATBENCH_INSTANCE_ID_ATTR, None),
        },
    }


def _annotate_catbench_result(
    result: dict[str, Any],
    task: task_eval.TaskEval,
    episode_status: str,
    exception_attribution: episode_exceptions.ExceptionAttribution | None = None,
    exception_message: str = '',
    exception_traceback: str = '',
) -> dict[str, Any]:
  """Adds condition, semantic-instance, and release provenance to a result."""
  def _optional_int_env(name: str) -> int | None:
    raw_value = os.environ.get(name, '').strip()
    if not raw_value:
      return None
    try:
      return int(raw_value)
    except ValueError:
      # Persist the malformed launch contract as an invalid condition rather
      # than silently inventing a numeric identity.
      return None

  task_type = type(task)
  effective_goal = _effective_task_goal(task)
  class_rendered_goal = _class_rendered_task_goal(task)
  semantic_task_id = str(
      getattr(task_type, 'catbench_semantic_id', task.name)
  )
  app_display_name = getattr(
      task_type, 'catbench_app_display_name', None
  )
  if task_breakdowns_lib is not None:
    semantic_goal = task_breakdowns_lib.app_neutral_goal(
        effective_goal, app_display_name
    )
    semantic_goal_hash = task_breakdowns_lib.goal_sha256(semantic_goal)
  else:
    semantic_goal = effective_goal
    semantic_goal_hash = hashlib.sha256(
        ' '.join(effective_goal.strip().split()).encode('utf-8')
    ).hexdigest()
  params_json = json.dumps(
      task.params,
      sort_keys=True,
      separators=(',', ':'),
      ensure_ascii=False,
      default=str,
  )
  breakdown_context = _get_task_breakdown_context(task, strict=False)
  instance_id = getattr(task, _CATBENCH_INSTANCE_ID_ATTR, None)
  selected_instance_raw = os.environ.get(_CATBENCH_INSTANCE_ID_ENV, '').strip()
  selected_instance_id = _optional_int_env(_CATBENCH_INSTANCE_ID_ENV)
  declared_condition = os.environ.get('CATBENCH_CONDITION', '').strip()
  breakdown_ready = bool(
      breakdown_context.get('enabled') and breakdown_context.get('found')
  )
  condition_config_valid = True
  if declared_condition:
    condition = declared_condition
    if declared_condition not in _CATBENCH_CONDITIONS:
      condition_config_valid = False
    elif declared_condition == 'c1' and breakdown_context.get('enabled'):
      condition_config_valid = False
    elif declared_condition in {'c2_g', 'c2_o'} and not breakdown_ready:
      condition_config_valid = False
  elif breakdown_ready:
    # Legacy/development runs remain readable but cannot satisfy the strict
    # primary-release condition requirement.
    condition = 'breakdown'
  elif breakdown_context.get('enabled'):
    condition = 'breakdown_missing_or_empty'
  else:
    condition = 'baseline'
  if selected_instance_raw and selected_instance_id != instance_id:
    condition_config_valid = False

  result.update(
      {
          'catbench_condition': condition,
          'catbench_condition_config_valid': condition_config_valid,
          'catbench_episode_status': episode_status,
          'catbench_exception_stage': None,
          'catbench_exception_attribution': None,
          'catbench_exception_valid_agent_failure': False,
          'catbench_exception_declared_agent_output': False,
          'catbench_exception_type': None,
          'catbench_exception_failure_code': None,
          'catbench_exception_message': exception_message,
          'catbench_exception_traceback': exception_traceback,
          'semantic_task_id': semantic_task_id,
          'semantic_goal': semantic_goal,
          'semantic_goal_sha256': semantic_goal_hash,
          'semantic_parameter_sha256': hashlib.sha256(
              params_json.encode('utf-8')
          ).hexdigest(),
          'instance_id': instance_id,
          'selected_instance_id': selected_instance_id,
          'package_name': getattr(task_type, 'package_name', None),
          'app_display_name': app_display_name,
          'code_revision': os.environ.get('CATBENCH_CODE_REVISION', ''),
          'source_snapshot_sha256': os.environ.get(
              'CATBENCH_SOURCE_SNAPSHOT_SHA256', ''
          ),
          'release_id': os.environ.get('CATBENCH_RELEASE_ID', ''),
          'release_purpose': os.environ.get('CATBENCH_RELEASE_PURPOSE', ''),
          'artifact_role': os.environ.get('CATBENCH_ARTIFACT_ROLE', ''),
          'analysis_eligible': (
              os.environ.get('CATBENCH_ANALYSIS_ELIGIBLE', '') == '1'
          ),
          'cohort_sha256': os.environ.get('CATBENCH_COHORT_SHA256', ''),
          'episode_runtime_policy_sha256': os.environ.get(
              'CATBENCH_EPISODE_RUNTIME_POLICY_SHA256', ''
          ),
          'schedule_manifest_sha256': os.environ.get(
              'CATBENCH_SCHEDULE_MANIFEST_SHA256', ''
          ),
          'pair_id': os.environ.get('CATBENCH_PAIR_ID', ''),
          'slot_id': os.environ.get('CATBENCH_SLOT_ID', ''),
          'attempt_id': os.environ.get('CATBENCH_ATTEMPT_ID', ''),
          'attempt_index': _optional_int_env('CATBENCH_ATTEMPT_INDEX'),
          'snapshot_family_id': os.environ.get(
              'CATBENCH_SNAPSHOT_FAMILY_ID', ''
          ),
          'snapshot_clone_id': os.environ.get(
              'CATBENCH_SNAPSHOT_CLONE_ID', ''
          ),
          'app_id': os.environ.get('CATBENCH_APP_ID', ''),
          'app_version': os.environ.get('CATBENCH_APP_VERSION', ''),
          'app_version_code': os.environ.get(
              'CATBENCH_APP_VERSION_CODE', ''
          ),
          'apk_sha256': os.environ.get('CATBENCH_APK_SHA256', ''),
          'model_name': os.environ.get('CATBENCH_MODEL_NAME', ''),
          'model_revision': os.environ.get('CATBENCH_MODEL_REVISION', ''),
          'runner_config_sha256': os.environ.get(
              'CATBENCH_RUNNER_CONFIG_SHA256', ''
          ),
          'model_config_sha256': os.environ.get(
              'CATBENCH_MODEL_CONFIG_SHA256', ''
          ),
          'model_endpoint_attestation_sha256': os.environ.get(
              'CATBENCH_MODEL_ENDPOINT_ATTESTATION_SHA256', ''
          ),
          'app_pins_sha256': os.environ.get(
              'CATBENCH_APP_PINS_SHA256', ''
          ),
          'exact_task_params_override_file': os.environ.get(
              exact_task_params_lib.ENV_FILE, ''
          ),
          'exact_task_params_override_sha256': os.environ.get(
              exact_task_params_lib.ENV_SHA256, ''
          ),
          'exact_task_params_override_mode': os.environ.get(
              exact_task_params_lib.ENV_MODE, ''
          ),
          'exact_task_params_source_file': os.environ.get(
              exact_task_params_lib.ENV_SOURCE_FILE, ''
          ),
          'exact_task_params_source_sha256': os.environ.get(
              exact_task_params_lib.ENV_SOURCE_SHA256, ''
          ),
          'exact_task_params_override_enabled': bool(
              os.environ.get(exact_task_params_lib.ENV_FILE, '').strip()
          ),
          'exact_goal_override_enabled': bool(
              os.environ.get(exact_task_params_lib.ENV_FILE, '').strip()
              and hasattr(task, _CATBENCH_EXACT_GOAL_ATTR)
              and os.environ.get(
                  exact_task_params_lib.ENV_GOAL_OVERRIDE_ENABLED, ''
              ) == '1'
          ),
          'exact_goal_mapping_sha256': os.environ.get(
              exact_task_params_lib.ENV_GOAL_MAPPING_SHA256, ''
          ),
          'exact_goal_sha256': hashlib.sha256(
              effective_goal.encode('utf-8')
          ).hexdigest() if hasattr(task, _CATBENCH_EXACT_GOAL_ATTR) else '',
          'class_rendered_goal': class_rendered_goal,
          'class_rendered_goal_sha256': hashlib.sha256(
              class_rendered_goal.encode('utf-8')
          ).hexdigest(),
          'exact_goal_differs_from_class_rendered': (
              effective_goal != class_rendered_goal
          ),
          'installed_app_attestation_sha256': os.environ.get(
              'CATBENCH_INSTALLED_APP_ATTESTATION_SHA256', ''
          ),
          'plan_file_sha256': os.environ.get(
              'CATBENCH_PLAN_FILE_SHA256', ''
          ),
          'task_random_seed': _optional_int_env(
              'CATBENCH_TASK_RANDOM_SEED'
          ),
          'n_task_combinations': _optional_int_env(
              'CATBENCH_N_TASK_COMBINATIONS'
          ),
          'schedule_seed': _optional_int_env('CATBENCH_SCHEDULE_SEED'),
          'prompt_goal': breakdown_context.get(
              'prompt_goal', effective_goal
          ),
      }
  )
  if exception_attribution is not None:
    result.update(exception_attribution.as_episode_fields())
  if breakdown_context.get('enabled'):
    result['task_breakdown_text'] = breakdown_context.get(
        'task_breakdown_text', ''
    )
    result['task_breakdown_metadata'] = breakdown_context.get(
        'task_breakdown_metadata', {}
    )
  return result


def _instantiate_task(
    task: Type[task_eval.TaskEval],
    params: dict[str, Any] | None = None,
    seed: int | None = None,
    env: interface.AsyncEnv | None = None,
) -> task_eval.TaskEval:
  """Creates an instance of a task with params.

  If params is not provided, it will use random params, controlled by a seed.

  Args:
    task: The task to instantiate.
    params: Params to use.
    seed: Seed for the random number generator.
    env: The environment.

  Returns:
    An instance of a task.
  """
  task.set_device_time(env)
  if params is None:
    if seed is not None:
      random.seed(seed)
    params = task.generate_random_params()
    params[constants.EpisodeConstants.SEED] = seed
  return task(params)


def create_suite(
    task_registry: dict[str, Type[task_eval.TaskEval]],
    n_task_combinations: int = 1,
    seed: int | None = None,
    tasks: list[str] | None = None,
    use_identical_params: bool = False,
    env: interface.AsyncEnv | None = None
) -> Suite:
  """Creates task suite.

  A task suite is a set of tasks. Each task is instantiated
  `n_task_combinations` times using new parameters. For example a task suite
  could look like:

  ```python
  {
      'GoogleSearchTask': [
          GoogleSearchTask({'term': 'cute cats'}),
          GoogleSearchTask({'term': 'comfy pillows'}),
      ],
      'WifiDisable': [  # No params for WiFi task.
          WifiDisable({}),
          WifiDisable({}),
      ],
  }
  ```

  Args:
    task_registry: Maps task names to their TaskEvals.
    n_task_combinations: Number of instances to create per task. Each instance
      will have unique param combinations.
    seed: Seed for the random number generator. Setting the seed will result in
      the same sequence of params for task instantiation per each task.
    tasks: List of task types that should be in the suite. If value is `None`
      all task types and associated instances will be created.
    use_identical_params: If True, each instance of a task, for a total of
      `n_task_combinations`, will have the same params.
    env: The environment that will be run on.

      Set ``CATBENCH_INSTANCE_ID`` to a zero-based instance index to retain
      only that original instance from each task. This is intended for frozen
      episode schedules and replacement runs: parameter generation still uses
      the requested ``n_task_combinations`` schedule, and the retained task
      keeps its original instance/checkpoint identity.

  Returns:
    A mapping of task name to instances of the task.
  """

  def _get_instance_seed(name: str, task_type: type[Any], i: int) -> int:
    # Generated CATBench task classes are app-specific subclasses (for example,
    # ``ClockCreateAlarmForChrono`` and ``ClockCreateAlarmForGoogleClock``).
    # Hashing the subclass name makes nominally paired cross-app tasks sample
    # different parameters.  A generated class can therefore opt into a shared,
    # app-independent seed namespace via ``catbench_semantic_id``.  All other
    # AndroidWorld tasks retain the historical name-based behavior.
    seed_namespace = getattr(task_type, 'catbench_semantic_id', name)
    unique_seed_str = f'{seed}_{seed_namespace}_{i}'
    return int(hashlib.sha256(unique_seed_str.encode()).hexdigest(), 16) % (
        2**32
    )

  selected_instance_id = None
  raw_selected_instance_id = os.environ.get(_CATBENCH_INSTANCE_ID_ENV, '')
  if raw_selected_instance_id.strip():
    try:
      selected_instance_id = int(raw_selected_instance_id)
    except ValueError as error:
      raise ValueError(
          f'{_CATBENCH_INSTANCE_ID_ENV} must be a zero-based integer; got '
          f'{raw_selected_instance_id!r}.'
      ) from error
    if not 0 <= selected_instance_id < n_task_combinations:
      raise ValueError(
          f'{_CATBENCH_INSTANCE_ID_ENV}={selected_instance_id} is outside '
          f'the scheduled range [0, {n_task_combinations}).'
      )

  exact_override = _load_exact_task_params_override()
  if exact_override is not None:
    if n_task_combinations != 1:
      raise ValueError(
          'Exact historical task-parameter overrides require '
          'n_task_combinations=1; they must not claim membership in a K>1 '
          'generated schedule.'
      )
    if selected_instance_id != 0 or raw_selected_instance_id.strip() != '0':
      raise ValueError(
          'Exact historical task-parameter overrides require an explicit '
          'CATBENCH_INSTANCE_ID=0 selection.'
      )
    if use_identical_params:
      raise ValueError(
          'Exact task-parameter overrides cannot use use_identical_params.'
      )
    if not tasks:
      raise ValueError(
          'Exact task-parameter overrides require an explicit non-empty task '
          'selection.'
      )
    exact_task_params_lib.require_exact_task_names(
        exact_override,
        tasks,
        registry_names=task_registry,
    )
    exact_suite: dict[str, list[task_eval.TaskEval]] = {}
    for name in tasks:
      task_instance = _instantiate_exact_override_task(
          name,
          task_registry[name],
          exact_override.overrides[name],
          env,
      )
      exact_suite[name] = [task_instance]
    return Suite(sorted(exact_suite.items()))

  suite = {}
  for name, task_type in task_registry.items():
    current = []
    for i in range(n_task_combinations):
      if use_identical_params:
        instance_seed = (
            _get_instance_seed(name, task_type, 0)
            if seed is not None
            else _FIXED_SEED
        )
      elif seed is not None:
        instance_seed = _get_instance_seed(name, task_type, i)
      else:
        instance_seed = None
      task_instance = _instantiate_task(
          task_type, seed=instance_seed, env=env
      )
      setattr(task_instance, _CATBENCH_INSTANCE_ID_ATTR, i)
      current.append(task_instance)
    if selected_instance_id is not None:
      current = [current[selected_instance_id]]
    suite[name] = current
  suite = _filter_tasks(suite, task_registry, tasks)

  # Sort suite alphabetically by task name.
  return Suite(sorted(suite.items()))


def _suggest_keyword(
    typo: str, keywords: list[str], threshold: int = 80
) -> str:
  """Suggests a keyword."""
  suggestion, score = process.extractOne(typo, keywords)
  if score >= threshold:
    return f" Did you mean '{suggestion}'?"
  else:
    return ''


def _filter_tasks(
    suite: dict[str, list[task_eval.TaskEval]],
    task_registry: dict[str, Type[task_eval.TaskEval]],
    tasks: list[str] | None = None,
) -> dict[str, list[task_eval.TaskEval]]:
  """Filters a suite by specific tasks.

  Args:
    suite: The suite to retrieve tasks from.
    task_registry: The task registry the suite is from.
    tasks: The tasks to retrieve. If None, just return entire suite.

  Returns:
    A "mini-suite" of tasks from suite.

  Raises:
    ValueError: If invalid task name.
  """
  if tasks is None:
    return suite
  subset = {}

  # Validate.
  for name in tasks:
    if name not in task_registry:
      raise ValueError(
          f'Task {name} not found in the task registry.'
          + _suggest_keyword(name, list(task_registry.keys()))
      )

  # Filter.
  for name, instances in suite.items():
    if name in tasks:
      subset[name] = instances
  return subset


def _create_attributed_exception_result(
    task: task_eval.TaskEval,
    error: BaseException,
    stage: episode_exceptions.EpisodeStage,
    start_time: float,
    *,
    known_environment_or_evaluator: bool = False,
    invalid_kind: str = '',
) -> dict[str, Any]:
  """Creates a scored or invalid result from an explicitly staged error."""
  attribution = episode_exceptions.attribute_exception(
      error,
      stage,
      known_environment_or_evaluator=known_environment_or_evaluator,
  )
  traceback_text = traceback.format_exc()
  run_time = time.time() - start_time
  if attribution.valid_agent_failure:
    result = _create_agent_output_failure_result(task, run_time)
    episode_status = 'valid_failure'
  elif invalid_kind == 'uninstalled':
    result = _create_skipped_uninstalled_result(
        task.name, _effective_task_goal(task), str(error), run_time
    )
    episode_status = 'invalid_infrastructure'
  elif invalid_kind == 'environment':
    result = _create_skipped_environment_result(
        task.name, _effective_task_goal(task), str(error), run_time
    )
    episode_status = 'invalid_infrastructure'
  else:
    result = _create_failed_result(
        task.name, _effective_task_goal(task), traceback_text, run_time
    )
    episode_status = 'invalid_infrastructure'
  return _annotate_catbench_result(
      result,
      task,
      episode_status,
      exception_attribution=attribution,
      exception_message=str(error),
      exception_traceback=traceback_text,
  )


def _record_secondary_teardown_exception(
    result: dict[str, Any], error: BaseException
) -> None:
  """Preserves a teardown error without erasing an earlier primary error."""
  attribution = episode_exceptions.attribute_exception(error, 'teardown')
  result['catbench_secondary_teardown_exception'] = {
      **attribution.as_episode_fields(),
      'catbench_exception_message': str(error),
      'catbench_exception_traceback': traceback.format_exc(),
  }


def _run_task(
    task: TaskEvalType,
    run_episode: Callable[[TaskEvalType], episode_runner.EpisodeResult],
    env: interface.AsyncEnv,
    demo_mode: bool,
    agent: base_agent.EnvironmentInteractingAgent = None,
) -> dict[str, Any]:
  """Runs a task.

  Args:
    task: The task.
    run_episode: Runs the agent on the task.
    env: Environment that will be run on.
    demo_mode: Whether running in demo mode; will display success overlay if so.

  Returns:
    Episode data and associated success signals.

  Raises:
    ValueError: If step data was not as expected.
  """
  # Late import: keeps the dependency on the cross-app module optional for
  # callers that only run upstream tasks.
  try:
    from android_world.task_evals.single.app_generalization_generated import (
        _cross_app_base as _xapp_base,
    )
    _TaskAppNotInstalled = _xapp_base.TaskAppNotInstalled
    _EnvironmentNetworkError = _xapp_base._EnvironmentNetworkError
  except Exception:  # pylint: disable=broad-except
    class _TaskAppNotInstalled(Exception):
      pass
    class _EnvironmentNetworkError(Exception):
      pass

  start = time.time()
  stage: episode_exceptions.EpisodeStage = 'initialize'

  def _execute_task():
    nonlocal stage
    stage = 'initialize'
    task.initialize_task(env)
    _log_and_print(
        'Running task %s with goal "%s"',
        task.name,
        _effective_task_goal(task),
    )
    stage = 'agent'
    interaction_results = run_episode(task)
    stage = 'verifier'
    is_catbench = bool(getattr(task, 'catbench_semantic_id', None))
    if is_catbench:
      runtime_health.assert_device_runtime_healthy(env)
    task_successful = _read_task_success_with_settle(task, env)
    if is_catbench:
      # Catch a persistent ANR/crash that surfaced while the official verifier
      # was reading state; do not let it become a semantic zero.
      runtime_health.assert_device_runtime_healthy(env)
    return interaction_results, task_successful

  try:
    interaction_results, task_successful = _run_with_catbench_task_timeout(
        task.name, _execute_task
    )
  except BaseException as error:  # pylint: disable=broad-exception-caught
    is_timeout = _is_catbench_timeout_control_flow(error)
    if not isinstance(error, Exception) and not is_timeout:
      raise
    if isinstance(error, episode_exceptions.StagedEpisodeError):
      stage = error.stage
      error = error.cause
    is_uninstalled = isinstance(error, _TaskAppNotInstalled)
    is_environment = isinstance(error, _EnvironmentNetworkError)
    if is_uninstalled:
      _log_and_print(
          '%s\nSKIPPED (app not installed) %s: %s',
          '~' * 80,
          task.name,
          error,
      )
    elif is_environment:
      _log_and_print(
          '%s\nSKIPPED (environment) %s: %s',
          '~' * 80,
          task.name,
          error,
      )
    elif (
        stage == 'agent'
        and isinstance(error, episode_exceptions.DeclaredAgentOutputError)
    ):
      _log_and_print(
          '%s\nSCORED AGENT OUTPUT FAILURE %s: %s',
          '~' * 80,
          task.name,
          error,
      )
    else:
      _log_and_print('%s\nSKIPPING %s.', '~' * 80, task.name)
    logging.exception(
        'Episode exception at stage=%s for task %s: %s',
        stage,
        task.name,
        error,
    )
    traceback.print_exc()
    result = _create_attributed_exception_result(
        task,
        error,
        stage,
        start,
        known_environment_or_evaluator=(
            is_uninstalled or is_environment
        ),
        invalid_kind=(
            'uninstalled'
            if is_uninstalled
            else 'environment'
            if is_environment
            else ''
        ),
    )
  else:
    agent_successful = task_successful if interaction_results.done else 0.0
    _log_and_print(
        '%s; %s',
        'Task Successful ✅' if agent_successful > 0.5 else 'Task Failed ❌',
        f' {_effective_task_goal(task)}',
    )
    try:
      if demo_mode:
        _display_success_overlay(env.controller, agent_successful)
      result = {
          constants.EpisodeConstants.GOAL: _effective_task_goal(task),
          constants.EpisodeConstants.TASK_TEMPLATE: task.name,
          constants.EpisodeConstants.EPISODE_DATA: interaction_results.step_data,
          constants.EpisodeConstants.IS_SUCCESSFUL: agent_successful,
          constants.EpisodeConstants.RUN_TIME: time.time() - start,
          constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
          constants.EpisodeConstants.EPISODE_LENGTH: len(
              interaction_results.step_data[constants.STEP_NUMBER]
          ),
          constants.EpisodeConstants.AUX_DATA: interaction_results.aux_data,
          constants.EpisodeConstants.SCREEN_CONFIG: _get_screen_config(task),
          constants.EpisodeConstants.EXCEPTION_INFO: None,
          constants.EpisodeConstants.SEED: task.params[
              constants.EpisodeConstants.SEED
          ],
      }
      _annotate_catbench_result(
          result,
          task,
          'valid_success' if agent_successful > 0.5 else 'valid_failure',
      )
    except Exception as error:  # pylint: disable=broad-exception-caught
      logging.exception('Agent episode result was malformed: %s', error)
      result = _create_attributed_exception_result(
          task, error, 'agent', start
      )
    if (
        result.get('catbench_episode_status') != 'invalid_infrastructure'
        and agent
        and hasattr(agent, 'save_task_debug_images')
    ):
      try:
        failure_reason = None
        if agent_successful <= 0.5:
          failure_reason = (
              'Max steps reached'
              if not interaction_results.done
              else 'Task completed but goal not achieved'
          )
        agent.save_task_debug_images(
            task.name,
            agent_successful > 0.5,
            _effective_task_goal(task),
            failure_reason,
        )
      except Exception as error:  # pylint: disable=broad-exception-caught
        logging.exception('Task debug-artifact finalization failed: %s', error)
        result = _create_attributed_exception_result(
            task, error, 'teardown', start
        )

  try:
    task.tear_down(env)
  except Exception as error:  # pylint: disable=broad-exception-caught
    logging.exception('Task teardown failed for %s: %s', task.name, error)
    if result.get('catbench_episode_status') == 'invalid_infrastructure':
      _record_secondary_teardown_exception(result, error)
    else:
      result = _create_attributed_exception_result(
          task, error, 'teardown', start
      )
  return result


def _get_task_info(
    episodes: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
  """Gets task info from episodes.

  Args:
    episodes: Episodes to get info from.

  Returns:
    A tuple of completed and failed task lookup tables.
  """

  completed = collections.defaultdict(list)
  failed = collections.defaultdict(list)
  for episode in episodes:
    instance_name = (
        episode[constants.EpisodeConstants.TASK_TEMPLATE]
        + checkpointer_lib.INSTANCE_SEPARATOR
        + str(episode[constants.EpisodeConstants.INSTANCE_ID])
    )
    if episode.get(constants.EpisodeConstants.EXCEPTION_INFO) is not None:
      failed[instance_name].append(episode)
    else:
      completed[instance_name].append(episode)
  return completed, failed


def _run_task_suite(
    suite: Suite,
    run_episode: Callable[[task_eval.TaskEval], episode_runner.EpisodeResult],
    env: interface.AsyncEnv,
    checkpointer: checkpointer_lib.Checkpointer = checkpointer_lib.NullCheckpointer(),
    demo_mode: bool = False,
    agent_name: str = '',
    return_full_episode_data: bool = False,
    process_episodes_fn=None,
    check_episode_fn: Callable[[dict[str, Any]], bool] | None = None,
    agent: base_agent.EnvironmentInteractingAgent = None,
) -> list[dict[str, Any]]:
  """Runs e2e system on suite.

  Args:
    suite: The suite to run it on.
    run_episode: The e2e system. See run_suite.py for an example.
    env: The environment e2e system runs on.
    checkpointer: See docstring from `run`.
    demo_mode: Whether to display the scoreboard.
    agent_name: The name of the agent.
    return_full_episode_data: Whether to return full episode data instead of
      just metadata.
    process_episodes_fn: The function to process episode data. Usually to
      compute metrics. Deafaults to process_episodes from this file.
    check_episode_fn: The function to check episode data.

  Returns:
    Metadata for each episode, including the scripted reward.
  """
  metadata_fields = [
      constants.EpisodeConstants.GOAL,
      constants.EpisodeConstants.TASK_TEMPLATE,
      constants.EpisodeConstants.INSTANCE_ID,
      constants.EpisodeConstants.IS_SUCCESSFUL,
      constants.EpisodeConstants.EPISODE_LENGTH,
      constants.EpisodeConstants.RUN_TIME,
      constants.EpisodeConstants.EXCEPTION_INFO,
      constants.EpisodeConstants.AUX_DATA,
  ]
  completed_tasks, failed_tasks = _get_task_info(
      checkpointer.load(fields=metadata_fields)
  )
  if process_episodes_fn is None:
    process_episodes_fn = process_episodes

  if (completed_tasks or failed_tasks) and return_full_episode_data:
    raise ValueError(
        'Cannot return full episode data when resuming from a checkpoint.'
    )
  episodes_metadata: list[dict[str, Any]] = []
  full_episode_data = []
  correct, total = 0, 0
  for name, instances in suite.items():
    msg = 'Running task: ' + name
    _log_and_print(msg + '\n' + '=' * len(msg))

    for i, instance in enumerate(instances):
      # Suites can be assembled manually rather than through create_suite.
      # Attach the exact scheduled identity before any prompt lookup so C2
      # cannot select a same-goal breakdown from another K>1 instance.
      instance_id = getattr(instance, _CATBENCH_INSTANCE_ID_ATTR, i)
      if (
          isinstance(instance_id, bool)
          or not isinstance(instance_id, int)
          or instance_id < 0
      ):
        raise ValueError(
            f'Invalid scheduled instance ID {instance_id!r} for {instance.name}.'
        )
      setattr(instance, _CATBENCH_INSTANCE_ID_ATTR, instance_id)
      instance_name = (
          instance.name
          + checkpointer_lib.INSTANCE_SEPARATOR
          + str(instance_id)
      )
      # Transferring from old checkpoint.
      if instance_name in completed_tasks:
        completed_episodes: list[dict[str, Any]] = completed_tasks[
            instance_name
        ]
        episodes_metadata.extend(completed_episodes)
      if instance_name in failed_tasks:
        episodes_metadata.extend(failed_tasks[instance_name])
      already_processed = (
          instance_name in completed_tasks and instance_name not in failed_tasks
      )
      if already_processed:
        _log_and_print('Skipping already processed task %s', instance_name)
        continue

      episode = _run_task(instance, run_episode, env, demo_mode=demo_mode, agent=agent)
      if (
          episode.get(constants.EpisodeConstants.EXCEPTION_INFO) is None
          and check_episode_fn is not None
      ):
        if not check_episode_fn(episode):
          continue
      episode[constants.EpisodeConstants.AGENT_NAME] = agent_name
      episode[constants.EpisodeConstants.INSTANCE_ID] = instance_id
      checkpointer.save_episodes([episode], instance_name)

      if return_full_episode_data:
        full_episode_data.append(episode)

      episode_metadata = {k: episode[k] for k in metadata_fields}
      for optional_key in (
          'catbench_condition',
          'catbench_condition_config_valid',
          'catbench_episode_status',
          'catbench_exception_stage',
          'catbench_exception_attribution',
          'catbench_exception_valid_agent_failure',
          'catbench_exception_declared_agent_output',
          'catbench_exception_type',
          'catbench_exception_failure_code',
          'catbench_exception_message',
          'catbench_exception_traceback',
          'catbench_secondary_teardown_exception',
          'prompt_goal',
          'task_breakdown_text',
          'task_breakdown_metadata',
      ):
        if optional_key in episode:
          episode_metadata[optional_key] = episode[optional_key]
      episodes_metadata.append(episode_metadata)
      process_episodes_fn(episodes_metadata, print_summary=True)

      if episode[constants.EpisodeConstants.EXCEPTION_INFO] is not None:
        # Don't include episode in tally if execution/eval logic errored out.
        continue
      correct += episode[constants.EpisodeConstants.IS_SUCCESSFUL]
      total += 1
      if demo_mode:
        _update_scoreboard(correct, total, env.controller)
    print()

  return full_episode_data if return_full_episode_data else episodes_metadata


def run(
    suite: Suite,
    agent: base_agent.EnvironmentInteractingAgent,
    checkpointer: checkpointer_lib.Checkpointer = checkpointer_lib.NullCheckpointer(),
    demo_mode: bool = False,
    return_full_episode_data: bool = False,
    process_episodes_fn=None,
    check_episode_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
  """Create suite and runs eval suite.

  Args:
    suite: The suite of tasks to run on.
    agent: An agent that interacts on the environment.
    checkpointer: Checkpointer that loads from existing run and resumes from
      there. NOTE: It will resume from the last fully completed task template.
      Relatedly, data for a task template will not be saved until all instances
      are executed.
    demo_mode: Whether to run in demo mode, which displays a scoreboard and the
      task instruction as a notification.
    return_full_episode_data: Whether to return full episode data instead of
      just metadata.
    process_episodes_fn: The function to process episode data. Usually to
      compute metrics. Deafaults to process_episodes from this file.
    check_episode_fn: The function to check episode data.

  Returns:
    Step-by-step data from each episode.
  """

  def run_episode(task: task_eval.TaskEval) -> episode_runner.EpisodeResult:
    if demo_mode:
      _display_goal(agent.env, task)
    calculated_budget = _allocate_step_budget(task.complexity)
    if agent.max_steps:
        # Use the HIGHER value: if user explicitly sets max_steps, respect it
        # This allows running tasks with more steps than the default budget
        max_steps = max(calculated_budget, agent.max_steps)
    else:
        max_steps = calculated_budget
    print(f"[DEBUG BUDGET] Task: {task.name}, Complexity: {task.complexity}")
    print(f"[DEBUG BUDGET] Calculated: {calculated_budget}, Agent max_steps: {agent.max_steps}, Final: {max_steps}")
    prompt_goal = _get_task_breakdown_context(task).get(
        'prompt_goal', _effective_task_goal(task)
    )
    if task.name.lower().startswith('miniwob'):
      termination_fn = miniwob_base.is_episode_terminated
    else:
      termination_fn = _make_success_termination_fn(task)
    return episode_runner.run_episode(
        goal=prompt_goal,
        agent=agent,
        max_n_steps=max_steps,
        start_on_home_screen=task.start_on_home_screen,
        termination_fn=termination_fn,
        health_check_fn=(
            runtime_health.assert_device_runtime_healthy
            if getattr(task, 'catbench_semantic_id', None)
            else None
        ),
    )

  if demo_mode:
    adb_utils.send_android_intent(
        'broadcast',
        'com.example.ACTION_UPDATE_SCOREBOARD',
        agent.env.controller,
        extras={'player_name': agent.name, 'scoreboard_value': '00/00'},
    )

  results = _run_task_suite(
      suite,
      run_episode,
      agent.env,
      checkpointer=checkpointer,
      demo_mode=demo_mode,
      agent_name=agent.name,
      return_full_episode_data=return_full_episode_data,
      process_episodes_fn=process_episodes_fn,
      check_episode_fn=check_episode_fn,
      agent=agent,
  )

  return results


def _make_success_termination_fn(task: task_eval.TaskEval):
  """Ends the episode as soon as the task's goal state is reached.

  Agents frequently complete a task correctly but fail to emit the
  terminate/done action; subsequent steps then drift the device away from
  the goal state and the final-state validator scores a real success as a
  failure. Checking the validator after every step (the same mechanism AW
  uses for MiniWoB via ``termination_fn``) ends the episode at the first
  step where the goal state holds; the runner then re-runs ``is_successful``
  on that frozen state for the official score, so scoring semantics are
  unchanged — only post-completion drift is eliminated.

  Disable with CATBENCH_EARLY_STOP_ON_SUCCESS=0 to reproduce strict
  AW final-state-only behaviour.

  Environment-level errors raised by validators (e.g. the maps network
  dialog) propagate so the runner can bucket the episode as an environment
  skip rather than an agent failure.
  """
  enabled = os.environ.get(
      'CATBENCH_EARLY_STOP_ON_SUCCESS', '1'
  ).strip().lower() not in ('0', 'false', 'no')
  if not enabled:
    return None

  def _goal_state_reached(env: interface.AsyncEnv) -> bool:
    try:
      return _read_task_success_with_settle(task, env) == 1.0
    except Exception as e:  # pylint: disable=broad-except
      # episode_runner invokes this callback from inside its agent loop. Wrap
      # the cause so _run_task can recover the true verifier stage instead of
      # misattributing it to the agent. No evaluator exception is converted to
      # a behavioural failure or silently ignored.
      raise episode_exceptions.StagedEpisodeError('verifier', e) from e

  return _goal_state_reached


def _allocate_step_budget(task_complexity: float) -> int:
  """Allocates number of steps dynamically based on the complexity score.

  Args:
    task_complexity: Complexity score of the task.

  Returns:
    Allocated number of steps for the task.
  """
  if task_complexity is None:
    raise ValueError('Task complexity must be provided.')
  return int(10 * (task_complexity))


def _display_message(
    header: str, body: str, env: env_interface.AndroidEnvInterface
) -> None:
  adb_utils.send_android_intent(
      'broadcast',
      'com.example.ACTION_UPDATE_OVERLAY',
      env,
      extras={'task_type_string': header, 'goal_string': body},
  )


def _display_goal(env: interface.AsyncEnv, task: task_eval.TaskEval) -> None:
  """Displays the goal on the screen using Android World.

  Args:
    env: The environment.
    task: The current task.
  """
  adb_utils.launch_app('android world', env.controller)
  time.sleep(1.0)
  _display_message(_effective_task_goal(task), task.name, env.controller)
  time.sleep(6.0)
  adb_utils.press_home_button(env.controller)
  time.sleep(1.0)


def _get_screen_config(task: task_eval.TaskEval) -> dict[str, Any]:
  return {
      'width': task.width if hasattr(task, 'width') else 1080,
      'height': task.height if hasattr(task, 'height') else 2400,
      'orientation': (
          task.orientation if hasattr(task, 'orientation') else 'portrait'
      ),
      'config_name': (
          task.config_name if hasattr(task, 'config_name') else 'default'
      ),
  }


def _create_failed_result(
    name: str, goal: str, exception: str, run_time: float
) -> dict[str, Any]:
  """Creates empty result to use if the run fails for some reason."""
  return {
      constants.EpisodeConstants.GOAL: goal,
      constants.EpisodeConstants.TASK_TEMPLATE: name,
      constants.EpisodeConstants.EPISODE_DATA: np.nan,
      constants.EpisodeConstants.IS_SUCCESSFUL: np.nan,
      constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
      constants.EpisodeConstants.RUN_TIME: run_time,
      constants.EpisodeConstants.EPISODE_LENGTH: np.nan,
      constants.EpisodeConstants.EXCEPTION_INFO: exception,
      constants.EpisodeConstants.AUX_DATA: None,
  }


def _create_agent_output_failure_result(
    task: task_eval.TaskEval, run_time: float
) -> dict[str, Any]:
  """Creates a scored failure for a declared parse/malformed-action error.

  ``exception_info`` remains ``None`` because AndroidWorld historically uses
  that field to remove infrastructure exceptions from the denominator. The
  typed exception, message, traceback, and whitelist decision are persisted in
  the ``catbench_exception_*`` fields by ``_annotate_catbench_result``.
  """
  return {
      constants.EpisodeConstants.GOAL: _effective_task_goal(task),
      constants.EpisodeConstants.TASK_TEMPLATE: task.name,
      constants.EpisodeConstants.EPISODE_DATA: {},
      constants.EpisodeConstants.IS_SUCCESSFUL: 0.0,
      constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
      constants.EpisodeConstants.RUN_TIME: run_time,
      constants.EpisodeConstants.EPISODE_LENGTH: 0,
      constants.EpisodeConstants.EXCEPTION_INFO: None,
      constants.EpisodeConstants.AUX_DATA: None,
      constants.EpisodeConstants.SCREEN_CONFIG: _get_screen_config(task),
      constants.EpisodeConstants.SEED: task.params.get(
          constants.EpisodeConstants.SEED
      ),
  }


def _create_skipped_uninstalled_result(
    name: str, goal: str, reason: str, run_time: float,
) -> dict[str, Any]:
  """Result for tasks whose target package is not installed on the device.

  Distinct from the generic failed result so the reporter can count these
  separately. ``IS_SUCCESSFUL`` is NaN (matching ``_create_failed_result``)
  but ``EXCEPTION_INFO`` carries the prefix ``[skipped_uninstalled]`` which
  the reporter pattern-matches.
  """
  return {
      constants.EpisodeConstants.GOAL: goal,
      constants.EpisodeConstants.TASK_TEMPLATE: name,
      constants.EpisodeConstants.EPISODE_DATA: np.nan,
      constants.EpisodeConstants.IS_SUCCESSFUL: np.nan,
      constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
      constants.EpisodeConstants.RUN_TIME: run_time,
      constants.EpisodeConstants.EPISODE_LENGTH: np.nan,
      constants.EpisodeConstants.EXCEPTION_INFO: f'[skipped_uninstalled] {reason}',
      constants.EpisodeConstants.AUX_DATA: None,
  }


def _create_skipped_environment_result(
    name: str, goal: str, reason: str, run_time: float,
) -> dict[str, Any]:
  """Result for task failures proven to be environmental."""
  return {
      constants.EpisodeConstants.GOAL: goal,
      constants.EpisodeConstants.TASK_TEMPLATE: name,
      constants.EpisodeConstants.EPISODE_DATA: np.nan,
      constants.EpisodeConstants.IS_SUCCESSFUL: np.nan,
      constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
      constants.EpisodeConstants.RUN_TIME: run_time,
      constants.EpisodeConstants.EPISODE_LENGTH: np.nan,
      constants.EpisodeConstants.EXCEPTION_INFO: f'[skipped_environment] {reason}',
      constants.EpisodeConstants.AUX_DATA: None,
  }


def _display_success_overlay(
    env: env_interface.AndroidEnvInterface, success: float
) -> None:
  """Displays success overlay."""
  adb_utils.send_android_intent(
      'broadcast',
      'com.example.ACTION_UPDATE_OVERLAY',
      env,
      extras={'success_string': str(int(success))},
  )
  time.sleep(1.0)  # Let display linger.


def _update_scoreboard(
    n_correct: int, n: int, env: env_interface.AndroidEnvInterface
) -> None:
  """Updates the scoreboard."""
  percentage = (n_correct / n) * 100
  scoreboard_value = f'{n_correct}/{n} ({percentage:.1f}%)'

  adb_utils.send_android_intent(
      'broadcast',
      'com.example.ACTION_UPDATE_SCOREBOARD',
      env,
      extras={'scoreboard_value': scoreboard_value},
  )


def _extract_task_metadata() -> pd.DataFrame:
  """Extracts metadata from task_metadata.json."""
  name = 'task_metadata.json'
  filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
  df = pd.read_json(filepath)
  df.rename(columns={_TASK_TEMPLATE_COLUMN: _TASK_PROMPT_COLUMN}, inplace=True)
  df.rename(columns={'task_name': _TASK_TEMPLATE_COLUMN}, inplace=True)
  return df.set_index(_TASK_TEMPLATE_COLUMN)[
      ['difficulty', 'optimal_steps', 'tags']
  ]


def _print_results_by_tag(result_df: pd.DataFrame) -> None:
  exploded_df = result_df.explode('tags').reset_index()
  exploded_df.replace(regex={'tags': r''}, value='untagged', inplace=True)  # pytype: disable=wrong-arg-types
  return (
      exploded_df.groupby(['tags', 'difficulty'], as_index=False)
      .agg(
          num_tasks=(_TASK_TEMPLATE_COLUMN, 'count'),
          mean_success_rate=('mean_success_rate', 'mean'),
      )
      .pivot_table(
          index=['tags'],
          columns='difficulty',
          values=[
              'mean_success_rate',
          ],
      )
      .fillna('-')
      .reindex(columns=['easy', 'medium', 'hard'], level='difficulty')
  )


def process_episodes(
    episodes: list[dict[str, Any]], print_summary: bool = False
) -> pd.DataFrame:
  """Processes task suite results; i.e. the output from `run_task_suite`.

  results = run_task_suite(...)
  # Contents of results.
  results = [
    {
        'goal': 'Pause the stopwatch.',
        'task_template': 'ClockStopWatchPaused',
        'episode_data': ...,
        'is_successful': True
    },
    {
        'goal': 'Pause the stopwatch.',
        'task_template': 'ClockStopWatchPaused',
        'episode_data': ...,
        'is_successful': False
    },
    {
        'goal': 'Run the stopwatch.',
        'task_template': 'ClockStopWatchRunnin',
        'episode_data': ...,
        'is_successful': True
    },
    {
        'goal': 'Run the stopwatch.',
        'task_template': 'ClockStopWatchRunnin',
        'episode_data': ...,
        'is_successful': True
    }
  ]

  process_episodes(results)
  # Output:
  # | task_template               |   n_trials |   average_success_rate |
  # |:----------------------------|-----------:|-----------------------:|
  # | ClockStopWatchPausedVerify  |          2 |                   0.5  |
  # | ClockStopWatchRunning       |          2 |                   1    |
  # | ==========Average========== |          2 |                   0.75 |

  Args:
    episodes: Results from running `run_task_suite`.
    print_summary: Whether to print the dataframe with a summary row.

  Returns:
    A dataframe aggregating results of run.
  """

  df = pd.DataFrame(list(episodes))

  # Add exeception info for backwards compatibility.
  df = df.assign(**{
      constants.EpisodeConstants.EXCEPTION_INFO: df.get(
          constants.EpisodeConstants.EXCEPTION_INFO, np.nan
      )
  })

  result_df = df.groupby(
      constants.EpisodeConstants.TASK_TEMPLATE, dropna=True
  ).agg({
      constants.EpisodeConstants.IS_SUCCESSFUL: ['count', 'mean'],
      constants.EpisodeConstants.EPISODE_LENGTH: 'mean',
      constants.EpisodeConstants.RUN_TIME: 'sum',
      constants.EpisodeConstants.EXCEPTION_INFO: [
          ('none_count', lambda x: x.notnull().sum())
      ],
  })
  result_df = result_df.sort_index()
  result_df.columns = [
      'num_complete_trials',
      'mean_success_rate',
      'mean_episode_length',
      'total_runtime_s',
      'num_fail_trials',
  ]
  result_df['total_runtime_s'] = result_df['total_runtime_s'].map(
      lambda x: float('{:.1f}'.format(x))
  )

  # Extract metadata and merge with the results table.
  metadata_df = _extract_task_metadata()
  tagged_result_df = result_df.merge(
      metadata_df, on=[_TASK_TEMPLATE_COLUMN], how='left'
  )

  if print_summary:
    avg = result_df.mean(axis=0)
    avg.name = '========= Average ========='

    result = pd.concat([result_df, avg.to_frame().T])
    result.index.name = 'task'
    result.insert(0, 'task_num', list(range(len(result) - 1)) + [0])
    result.task_num = result.task_num.astype(int)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.width', 1000)
    _log_and_print('\n\n%s', result)  # Use lazy % formatting

    # Add a chart that shows mean success rate by tag and difficulty.
    tags_df = _print_results_by_tag(tagged_result_df)
    pd.set_option('display.precision', 2)
    _log_and_print('\n\n%s', tags_df)

  return tagged_result_df
