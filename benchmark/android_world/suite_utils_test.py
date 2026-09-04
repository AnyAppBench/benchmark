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

"""Tests for suite utils."""

import copy
import json
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from android_world import checkpointer
from android_world import constants
from android_world import episode_runner
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import episode_exceptions
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base,
)
from android_world.utils import test_utils
import dm_env
import exact_task_params
import numpy as np
import task_breakdowns


class TestCreateSuite(parameterized.TestCase):
  """Test that entire suite can be created.

  Later tests probe specific features related to the registry.
  """

  @parameterized.named_parameters(
      dict(testcase_name='android', family='information_retrieval'),
      dict(testcase_name='miniwob', family='miniwob'),
      dict(
          testcase_name='information_retrieval', family='information_retrieval'
      ),
  )
  def test_create_suite(self, family: str):
    suite_utils.create_suite(
        registry.TaskRegistry().get_registry(family), n_task_combinations=2
    )


class TestSuite(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.task_registry = registry.TaskRegistry()
    self.original_registry = copy.deepcopy(
        self.task_registry.ANDROID_TASK_REGISTRY
    )
    self.testing_registry = {
        'Task1': test_utils.FakeCurrentStateEval,
        'Task2': test_utils.FakeAdbEval,
    }
    self.seed = 42

  def test_create_entire_suite(self):
    suite_utils.create_suite(self.original_registry, n_task_combinations=2)

  def test_create_suite(self):
    n_task_combinations = 2
    suite = suite_utils.create_suite(
        self.testing_registry, n_task_combinations=n_task_combinations
    )
    self.assertLen(
        suite['Task1'],
        n_task_combinations,
        'Should create 2 instances for Task1',
    )
    self.assertLen(
        suite['Task2'],
        n_task_combinations,
        'Should create 2 instances for Task2',
    )

  def test_determinism_with_same_seed(self):
    suite1 = suite_utils.create_suite(
        self.testing_registry, n_task_combinations=2, seed=self.seed
    )
    suite2 = suite_utils.create_suite(
        self.testing_registry, n_task_combinations=2, seed=self.seed
    )

    self.assertEqual(
        suite1['Task1'][0].params,
        suite2['Task1'][0].params,
        'Task1 instance 1 params should match with the same seed',
    )
    self.assertEqual(
        suite1['Task1'][1].params,
        suite2['Task1'][1].params,
        'Task1 instance 2 params should match with the same seed',
    )
    self.assertEqual(
        suite1['Task2'][0].params,
        suite2['Task2'][0].params,
        'Task2 instance 1 params should match with the same seed',
    )
    self.assertEqual(
        suite1['Task2'][1].params,
        suite2['Task2'][1].params,
        'Task2 instance 2 params should match with the same seed',
    )

  def test_semantic_id_pairs_params_across_app_specific_classes(self):
    class TestOnlyClockAlarmForChrono(test_utils.FakeCurrentStateEval):
      catbench_semantic_id = 'CanonicalTask'

    class TestOnlyClockAlarmForGoogleClock(test_utils.FakeCurrentStateEval):
      catbench_semantic_id = 'CanonicalTask'

    paired_registry = {
        'ClockAlarmForChrono': TestOnlyClockAlarmForChrono,
        'ClockAlarmForGoogleClock': TestOnlyClockAlarmForGoogleClock,
    }
    suite = suite_utils.create_suite(
        paired_registry, n_task_combinations=3, seed=self.seed
    )

    for instance_index in range(3):
      self.assertEqual(
          suite['ClockAlarmForChrono'][instance_index].params,
          suite['ClockAlarmForGoogleClock'][instance_index].params,
      )

  def test_tasks_without_semantic_id_keep_name_seed_namespace(self):
    suite = suite_utils.create_suite(
        self.testing_registry, n_task_combinations=1, seed=self.seed
    )

    self.assertNotEqual(suite['Task1'][0].params, suite['Task2'][0].params)

  def test_instance_selector_keeps_original_params_and_identity(self):
    with mock.patch.dict(
        suite_utils.os.environ, {'CATBENCH_INSTANCE_ID': ''}
    ):
      full_suite = suite_utils.create_suite(
          self.testing_registry, n_task_combinations=3, seed=self.seed
      )
    with mock.patch.dict(
        suite_utils.os.environ, {'CATBENCH_INSTANCE_ID': '2'}
    ):
      selected_suite = suite_utils.create_suite(
          self.testing_registry, n_task_combinations=3, seed=self.seed
      )

    for task_name in self.testing_registry:
      self.assertLen(selected_suite[task_name], 1)
      selected = selected_suite[task_name][0]
      self.assertEqual(selected.params, full_suite[task_name][2].params)
      self.assertEqual(getattr(selected, '_catbench_instance_id'), 2)

  def test_instance_selector_rejects_invalid_or_out_of_range_values(self):
    for value in ('not-an-integer', '-1', '3'):
      with self.subTest(value=value):
        with mock.patch.dict(
            suite_utils.os.environ, {'CATBENCH_INSTANCE_ID': value}
        ):
          with self.assertRaisesRegex(ValueError, 'CATBENCH_INSTANCE_ID'):
            suite_utils.create_suite(
                self.testing_registry,
                n_task_combinations=3,
                seed=self.seed,
            )

  def test_exact_override_instantiates_only_pinned_instance_zero(self):
    class ExactTask(test_utils.FakeCurrentStateEval):
      template = 'Do {value}'
      schema = {
          'type': 'object',
          'properties': {
              'value': {'type': 'string'},
              'seed': {'type': 'integer'},
          },
          'required': ['value', 'seed'],
          'additionalProperties': False,
      }

    with tempfile.TemporaryDirectory() as tmpdir:
      source = Path(tmpdir) / 'canonical.json'
      source.write_text('{"audited":true}\n', encoding='utf-8')
      source_hash = exact_task_params.file_sha256(source)
      payload = {
          'schema_version': 1,
          'mode': exact_task_params.MODE,
          'source': {'file': str(source), 'sha256': source_hash},
          'overrides': {
              'ExactTask': {
                  'instance_id': 0,
                  'params': {'value': 'alpha', 'seed': 90210},
                  'expected_goal': 'Do alpha',
                  'expected_seed': 90210,
              },
          },
      }
      path = Path(tmpdir) / 'exact.json'
      path.write_text(json.dumps(payload), encoding='utf-8')
      environment = {
          exact_task_params.ENV_FILE: str(path),
          exact_task_params.ENV_SHA256: exact_task_params.file_sha256(path),
          exact_task_params.ENV_MODE: exact_task_params.MODE,
          exact_task_params.ENV_SOURCE_FILE: str(source),
          exact_task_params.ENV_SOURCE_SHA256: source_hash,
          exact_task_params.ENV_GOAL_OVERRIDE_ENABLED: '1',
          exact_task_params.ENV_GOAL_MAPPING_SHA256: (
              exact_task_params.file_sha256(path)
          ),
          'CATBENCH_INSTANCE_ID': '0',
          'CATBENCH_CONDITION': 'c1',
          'CATBENCH_RELEASE_PURPOSE': 'revision_rerun_candidate',
          'CATBENCH_ARTIFACT_ROLE': (
              'invalid_episode_replacement_candidate'
          ),
          'CATBENCH_ANALYSIS_ELIGIBLE': '0',
      }
      with mock.patch.dict(suite_utils.os.environ, environment, clear=False):
        suite = suite_utils.create_suite(
            {'ExactTask': ExactTask, 'Task2': test_utils.FakeAdbEval},
            n_task_combinations=1,
            seed=self.seed,
            tasks=['ExactTask'],
        )
        with self.assertRaisesRegex(ValueError, 'n_task_combinations=1'):
          suite_utils.create_suite(
              {'ExactTask': ExactTask},
              n_task_combinations=3,
              seed=self.seed,
              tasks=['ExactTask'],
          )

    self.assertEqual(['ExactTask'], list(suite))
    self.assertLen(suite['ExactTask'], 1)
    instance = suite['ExactTask'][0]
    self.assertEqual({'value': 'alpha', 'seed': 90210}, instance.params)
    self.assertEqual('Do alpha', instance.goal)
    self.assertEqual(0, getattr(instance, '_catbench_instance_id'))

  def test_exact_override_uses_archived_goal_but_rejects_schema_mismatch(self):
    class ExactTask(test_utils.FakeCurrentStateEval):
      template = 'Do {value}'
      schema = {
          'type': 'object',
          'properties': {
              'value': {'type': 'string'},
              'seed': {'type': 'integer'},
          },
          'required': ['value', 'seed'],
      }

    for case in ('goal', 'schema'):
      with self.subTest(case=case), tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / 'canonical.json'
        source.write_text('{"audited":true}\n', encoding='utf-8')
        source_hash = exact_task_params.file_sha256(source)
        params = {'value': 'alpha', 'seed': 7}
        expected_goal = 'Wrong goal'
        if case == 'schema':
          params['value'] = 12
          expected_goal = 'Do 12'
        payload = {
            'schema_version': 1,
            'mode': exact_task_params.MODE,
            'source': {'file': str(source), 'sha256': source_hash},
            'overrides': {
                'ExactTask': {
                    'instance_id': 0,
                    'params': params,
                    'expected_goal': expected_goal,
                    'expected_seed': 7,
                },
            },
        }
        path = Path(tmpdir) / 'exact.json'
        path.write_text(json.dumps(payload), encoding='utf-8')
        environment = {
            exact_task_params.ENV_FILE: str(path),
            exact_task_params.ENV_SHA256: exact_task_params.file_sha256(path),
            exact_task_params.ENV_MODE: exact_task_params.MODE,
            exact_task_params.ENV_SOURCE_FILE: str(source),
            exact_task_params.ENV_SOURCE_SHA256: source_hash,
            exact_task_params.ENV_GOAL_OVERRIDE_ENABLED: '1',
            exact_task_params.ENV_GOAL_MAPPING_SHA256: (
                exact_task_params.file_sha256(path)
            ),
            'CATBENCH_INSTANCE_ID': '0',
            'CATBENCH_CONDITION': 'c1',
            'CATBENCH_RELEASE_PURPOSE': 'revision_rerun_candidate',
            'CATBENCH_ARTIFACT_ROLE': (
                'invalid_episode_replacement_candidate'
            ),
            'CATBENCH_ANALYSIS_ELIGIBLE': '0',
        }
        with mock.patch.dict(
            suite_utils.os.environ, environment, clear=False
        ):
          if case == 'schema':
            with self.assertRaisesRegex(ValueError, 'fail the task schema'):
              suite_utils.create_suite(
                  {'ExactTask': ExactTask},
                  n_task_combinations=1,
                  seed=self.seed,
                  tasks=['ExactTask'],
              )
          else:
            suite = suite_utils.create_suite(
                {'ExactTask': ExactTask},
                n_task_combinations=1,
                seed=self.seed,
                tasks=['ExactTask'],
            )
            instance = suite['ExactTask'][0]
            self.assertEqual('Do alpha', instance.goal)
            self.assertEqual(
                'Wrong goal', suite_utils._effective_task_goal(instance)
            )
            result = suite_utils._annotate_catbench_result(
                {}, instance, 'valid_failure'
            )
            self.assertTrue(result['exact_goal_override_enabled'])
            self.assertEqual('Do alpha', result['class_rendered_goal'])
            self.assertTrue(
                result['exact_goal_differs_from_class_rendered']
            )
            self.assertEqual('Wrong goal', result['prompt_goal'])
            instance.initialize_task = mock.MagicMock()
            instance.is_successful = mock.MagicMock(return_value=0.0)
            instance.tear_down = mock.MagicMock()
            run_episode = mock.MagicMock(
                return_value=episode_runner.EpisodeResult(
                    done=True, step_data={'step_number': [0]}
                )
            )
            agent = mock.MagicMock()
            run_result = suite_utils._run_task(
                instance,
                run_episode,
                mock.MagicMock(),
                demo_mode=False,
                agent=agent,
            )
            self.assertEqual(
                'Wrong goal', run_result[constants.EpisodeConstants.GOAL]
            )
            self.assertEqual(
                'Wrong goal', agent.save_task_debug_images.call_args.args[2]
            )

  def test_variation_with_different_seed(self):
    suite1 = suite_utils.create_suite(
        self.testing_registry, n_task_combinations=2, seed=self.seed
    )
    suite2 = suite_utils.create_suite(
        self.testing_registry, n_task_combinations=2, seed=self.seed + 1
    )

    self.assertNotEqual(
        suite1['Task1'][0].params,
        suite2['Task1'][0].params,
        'Task1 instance 1 params should not match with different seeds',
    )
    self.assertNotEqual(
        suite1['Task2'][0].params,
        suite2['Task2'][0].params,
        'Task2 instance 1 params should not match with different seeds',
    )

  @mock.patch.object(suite_utils.random, 'seed')
  def test_no_seed_provides_randomness(self, mock_seed):
    suite_utils.create_suite(self.testing_registry, n_task_combinations=2)
    mock_seed.assert_not_called()

  def test_return_all_when_tasks_none(self):
    suite = suite_utils.Suite(
        **{
            'Task1': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
            ],
            'Task2': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'test'
    tasks = None

    result = suite_utils._filter_tasks(
        suite,
        self.task_registry.get_registry(registry.TaskRegistry.ANDROID_FAMILY),
        tasks,
    )

    self.assertEqual(
        result, suite, 'Should return the same suite when tasks is None'
    )

  def test_valid_tasks_subset(self):
    expected = [
        test_utils.FakeCurrentStateEval(
            test_utils.FakeCurrentStateEval.generate_random_params()
        )
    ]
    tasks = ['Task1']

    result = suite_utils._filter_tasks(
        {
            'Task1': expected,
            'Task2': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                )
            ],
            'Task3': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
        self.testing_registry,
        tasks,
    )

    self.assertEqual(
        result, {'Task1': expected}, 'Should return the subset of tasks'
    )

  def test_invalid_task_raises_value_error(self):
    suite = suite_utils.Suite(
        **{
            'Task1': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                )
            ],
            'Task2': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'test'
    tasks = ['Task1', 'Task3']

    with self.assertRaises(ValueError):
      suite_utils._filter_tasks(
          suite,
          self.task_registry.get_registry(registry.TaskRegistry.ANDROID_FAMILY),
          tasks,
      )


class SuiteUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.testing_registry = {
        'Task1': test_utils.FakeCurrentStateEval,
        'Task2': test_utils.FakeAdbEval,
    }

  @mock.patch.object(time, 'sleep', autospec=True)
  def test_catbench_verifier_retries_bounded_delayed_write(self, mock_sleep):
    task = mock.Mock(spec=['catbench_semantic_id', 'is_successful'])
    task.catbench_semantic_id = 'ContactsAddContact'
    task.is_successful.side_effect = [0.0, 0.0, 1.0]

    with mock.patch.object(
        suite_utils, '_catbench_verifier_settle_attempts', return_value=3
    ), mock.patch.object(
        suite_utils,
        '_catbench_verifier_settle_interval_seconds',
        return_value=0.2,
    ):
      score = suite_utils._read_task_success_with_settle(
          task, mock.MagicMock()
      )

    self.assertEqual(score, 1.0)
    self.assertEqual(task.is_successful.call_count, 3)
    self.assertEqual(mock_sleep.call_args_list, [mock.call(0.2), mock.call(0.2)])

  @mock.patch.object(time, 'sleep', autospec=True)
  def test_upstream_verifier_retains_single_read(self, mock_sleep):
    task = mock.Mock(spec=['is_successful'])
    task.is_successful.return_value = 0.0

    score = suite_utils._read_task_success_with_settle(
        task, mock.MagicMock()
    )

    self.assertEqual(score, 0.0)
    task.is_successful.assert_called_once()
    mock_sleep.assert_not_called()

  def test_catbench_verifier_read_error_propagates(self):
    task = mock.Mock(spec=['catbench_semantic_id', 'is_successful'])
    task.catbench_semantic_id = 'MapsSearchPlace'
    task.is_successful.side_effect = RuntimeError('native read failed')

    with self.assertRaisesRegex(RuntimeError, 'native read failed'):
      suite_utils._read_task_success_with_settle(task, mock.MagicMock())

  def test_result_records_episode_runtime_policy_hash(self):
    task = test_utils.FakeCurrentStateEval(
        test_utils.FakeCurrentStateEval.generate_random_params()
    )
    expected = 'a' * 64
    with mock.patch.dict(
        suite_utils.os.environ,
        {'CATBENCH_EPISODE_RUNTIME_POLICY_SHA256': expected},
        clear=False,
    ):
      result = suite_utils._annotate_catbench_result(
          {}, task, 'valid_failure'
      )
    self.assertEqual(result['episode_runtime_policy_sha256'], expected)

  @parameterized.named_parameters(
      dict(testcase_name='no_demo_mode', demo_mode=False),
      dict(testcase_name='demo_mode', demo_mode=True),
  )
  @mock.patch.object(test_utils.FakeAdbEval, 'initialize_task', autospec=True)
  @mock.patch.object(interface, 'AsyncAndroidEnv')
  @mock.patch.object(adb_utils, 'send_android_intent')
  @mock.patch.object(time, 'sleep', autospec=True)
  def test_run_adb_task_instances(
      self,
      mock_sleep,
      mock_send_android_intent,
      mock_env,
      mock_initialize_task,
      demo_mode,
  ):
    pixels = np.zeros((3, 3, 3))
    mock_env.get_state.return_value = (
        dm_env.TimeStep(
            observation={'pixels': pixels},
            reward=0,
            discount=0,
            step_type=dm_env.StepType.LAST,
        ),
        [],
    )
    mock_android_env = mock.PropertyMock(return_value=mock.MagicMock())
    mock_env.controller = mock_android_env
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.return_value = episode_runner.EpisodeResult(
        True,
        {'step_number': [0]},
    )

    result = suite_utils._run_task(
        test_utils.FakeAdbEval(test_utils.FakeAdbEval.generate_random_params()),
        mock_run_e2e,
        mock_env,
        demo_mode=demo_mode,
    )

    self.assertEqual(result['is_successful'], 1)
    self.assertIn(result['goal'], 'ADB eval')
    mock_initialize_task.assert_called_once()
    if demo_mode:
      mock_send_android_intent.assert_has_calls([
          mock.call(
              'broadcast',
              'com.example.ACTION_UPDATE_OVERLAY',
              mock_android_env,
              extras={'success_string': '1'},
          ),
      ])
      mock_sleep.assert_called()

  def test_run_miniwob_task_instances_initialize(self):
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.return_value = episode_runner.EpisodeResult(
        done=True,
        step_data={'step_number': [0]},
    )
    failing_instance = test_utils.FakeMiniWobTask(
        test_utils.FakeMiniWobTask.generate_random_params()
    )

    result = suite_utils._run_task(
        failing_instance, mock_run_e2e, mock.MagicMock(), demo_mode=False
    )

    self.assertIsNone(result[constants.EpisodeConstants.EXCEPTION_INFO])

  def test_run_adb_task_instances_initialize_fails(self):
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.return_value = episode_runner.EpisodeResult(
        done=True,
        step_data={'step_number': [0]},
    )
    failing_instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    failing_instance.initialize_task = lambda: ValueError(
        'Something went wrong'
    )

    result = suite_utils._run_task(
        failing_instance, mock_run_e2e, mock.MagicMock(), demo_mode=False
    )
    self.assertIsNotNone(result[constants.EpisodeConstants.EXCEPTION_INFO])

  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_declared_agent_parse_exception_is_scored_failure(self, mock_env):
    instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    instance.initialize_task = mock.MagicMock()
    instance.tear_down = mock.MagicMock()
    run_episode = mock.MagicMock(
        side_effect=episode_exceptions.ActionParseError('missing tool call')
    )

    result = suite_utils._run_task(
        instance, run_episode, mock_env, demo_mode=False
    )

    self.assertEqual(0.0, result[constants.EpisodeConstants.IS_SUCCESSFUL])
    self.assertIsNone(result[constants.EpisodeConstants.EXCEPTION_INFO])
    self.assertEqual('valid_failure', result['catbench_episode_status'])
    self.assertEqual('agent', result['catbench_exception_stage'])
    self.assertEqual(
        'agent_output_parse_or_malformed_action',
        result['catbench_exception_attribution'],
    )
    self.assertTrue(result['catbench_exception_valid_agent_failure'])
    self.assertTrue(result['catbench_exception_declared_agent_output'])

  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_release_analysis_role_is_preserved_in_episode(self, mock_env):
    instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    instance.initialize_task = mock.MagicMock()
    instance.is_successful = mock.MagicMock(return_value=0.0)
    instance.tear_down = mock.MagicMock()
    setattr(instance, '_catbench_instance_id', 2)
    run_episode = mock.MagicMock(return_value=episode_runner.EpisodeResult(
        done=True, step_data={'step_number': [0]}
    ))
    with mock.patch.dict(
        suite_utils.os.environ,
        {
            'CATBENCH_RELEASE_PURPOSE': (
                'g6_discard_only_end_to_end_validation'
            ),
            'CATBENCH_ARTIFACT_ROLE': (
                'discard_only_never_primary_analysis'
            ),
            'CATBENCH_ANALYSIS_ELIGIBLE': '0',
            'CATBENCH_INSTANCE_ID': '2',
            'CATBENCH_SOURCE_SNAPSHOT_SHA256': 'a' * 64,
            exact_task_params.ENV_FILE: '/tmp/effective.json',
            exact_task_params.ENV_SHA256: 'b' * 64,
            exact_task_params.ENV_MODE: exact_task_params.MODE,
            exact_task_params.ENV_SOURCE_FILE: '/tmp/source.json',
            exact_task_params.ENV_SOURCE_SHA256: 'c' * 64,
        },
    ):
      result = suite_utils._run_task(
          instance, run_episode, mock_env, demo_mode=False
      )

    self.assertEqual(
        result['release_purpose'],
        'g6_discard_only_end_to_end_validation',
    )
    self.assertEqual(
        result['artifact_role'], 'discard_only_never_primary_analysis'
    )
    self.assertFalse(result['analysis_eligible'])
    self.assertEqual(2, result['instance_id'])
    self.assertEqual(2, result['selected_instance_id'])
    self.assertEqual('a' * 64, result['source_snapshot_sha256'])
    self.assertTrue(result['exact_task_params_override_enabled'])
    self.assertEqual(
        '/tmp/effective.json', result['exact_task_params_override_file']
    )
    self.assertEqual(
        'b' * 64, result['exact_task_params_override_sha256']
    )
    self.assertEqual(
        exact_task_params.MODE, result['exact_task_params_override_mode']
    )
    self.assertEqual(
        'c' * 64, result['exact_task_params_source_sha256']
    )
    self.assertTrue(result['catbench_condition_config_valid'])

  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_required_breakdown_lookup_failure_is_recorded_not_raised(
      self, mock_env
  ):
    """C2 REQUIRED=1 with a missing plan entry must yield one invalid episode.

    The annotation path used to call the strict breakdown lookup again inside
    the ``_run_task`` exception handler, re-raising the same KeyError and
    killing the whole app-level job.
    """
    instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    instance.initialize_task = mock.MagicMock()
    instance.is_successful = mock.MagicMock(return_value=1.0)
    instance.tear_down = mock.MagicMock()
    setattr(instance, '_catbench_instance_id', 0)

    def run_episode(task):
      # Mirrors the closure in suite_utils.run(): strict lookup before step.
      suite_utils._get_task_breakdown_context(task)
      return episode_runner.EpisodeResult(
          done=True, step_data={'step_number': [0]}
      )

    task_breakdowns.clear_payload_cache()
    with tempfile.TemporaryDirectory() as tmpdir:
      plan_path = Path(tmpdir) / 'plans.json'
      plan_path.write_text(json.dumps({'breakdowns': []}), encoding='utf-8')
      with mock.patch.dict(
          suite_utils.os.environ,
          {
              'CATBENCH_CONDITION': 'c2_g',
              task_breakdowns.ENV_BREAKDOWN_FILE: str(plan_path),
              task_breakdowns.ENV_BREAKDOWN_REQUIRED: '1',
          },
      ):
        suite_utils.os.environ.pop(task_breakdowns.ENV_BREAKDOWN_MODE, None)
        result = suite_utils._run_task(
            instance, run_episode, mock_env, demo_mode=False
        )

    self.assertIsNotNone(result[constants.EpisodeConstants.EXCEPTION_INFO])
    self.assertEqual(
        'invalid_infrastructure', result['catbench_episode_status']
    )
    self.assertEqual('c2_g', result['catbench_condition'])
    self.assertFalse(result['catbench_condition_config_valid'])
    self.assertEqual('agent', result['catbench_exception_stage'])
    self.assertIn(
        'Missing task breakdown', result['catbench_exception_message']
    )
    self.assertTrue(result['task_breakdown_metadata']['lookup_error'])
    self.assertEqual(result['prompt_goal'], instance.goal)
    instance.tear_down.assert_called_once()

  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_generic_agent_exception_remains_invalid(self, mock_env):
    instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    instance.initialize_task = mock.MagicMock()
    instance.tear_down = mock.MagicMock()
    run_episode = mock.MagicMock(side_effect=ValueError('unknown agent error'))

    result = suite_utils._run_task(
        instance, run_episode, mock_env, demo_mode=False
    )

    self.assertTrue(
        np.isnan(result[constants.EpisodeConstants.IS_SUCCESSFUL])
    )
    self.assertIsNotNone(result[constants.EpisodeConstants.EXCEPTION_INFO])
    self.assertEqual(
        'invalid_infrastructure', result['catbench_episode_status']
    )
    self.assertEqual('agent', result['catbench_exception_stage'])
    self.assertEqual('unknown', result['catbench_exception_attribution'])
    self.assertFalse(result['catbench_exception_valid_agent_failure'])

  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_emulator_health_exception_is_invalid_infrastructure(self, mock_env):
    instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    instance.initialize_task = mock.MagicMock()
    instance.tear_down = mock.MagicMock()
    run_episode = mock.MagicMock(
        side_effect=episode_exceptions.EmulatorRuntimeHealthError(
            'visible Application Error dialog'
        )
    )

    result = suite_utils._run_task(
        instance, run_episode, mock_env, demo_mode=False
    )

    self.assertTrue(
        np.isnan(result[constants.EpisodeConstants.IS_SUCCESSFUL])
    )
    self.assertEqual(
        'invalid_infrastructure', result['catbench_episode_status']
    )
    self.assertEqual(
        'environment_or_evaluator',
        result['catbench_exception_attribution'],
    )
    self.assertEqual(
        'emulator_runtime_health_error',
        result['catbench_exception_failure_code'],
    )

  @mock.patch.object(
      suite_utils.runtime_health, 'assert_device_runtime_healthy'
  )
  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_final_catbench_verifier_is_bracketed_by_health_checks(
      self, mock_env, health_check
  ):
    instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    instance.catbench_semantic_id = 'TestSemanticTask'
    instance.initialize_task = mock.MagicMock()
    instance.is_successful = mock.MagicMock(return_value=1.0)
    instance.tear_down = mock.MagicMock()
    run_episode = mock.MagicMock(return_value=episode_runner.EpisodeResult(
        done=True, step_data={'step_number': [0]}
    ))

    result = suite_utils._run_task(
        instance, run_episode, mock_env, demo_mode=False
    )

    self.assertEqual('valid_success', result['catbench_episode_status'])
    self.assertEqual(
        [mock.call(mock_env), mock.call(mock_env)],
        health_check.call_args_list,
    )

  @mock.patch.object(
      suite_utils.runtime_health, 'assert_device_runtime_healthy'
  )
  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_late_health_fault_invalidates_apparent_success(
      self, mock_env, health_check
  ):
    instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    instance.catbench_semantic_id = 'TestSemanticTask'
    instance.initialize_task = mock.MagicMock()
    instance.is_successful = mock.MagicMock(return_value=1.0)
    instance.tear_down = mock.MagicMock()
    health_check.side_effect = [
        None,
        episode_exceptions.EmulatorRuntimeHealthError(
            'System UI is not responding'
        ),
    ]
    run_episode = mock.MagicMock(return_value=episode_runner.EpisodeResult(
        done=True, step_data={'step_number': [0]}
    ))

    result = suite_utils._run_task(
        instance, run_episode, mock_env, demo_mode=False
    )

    self.assertTrue(
        np.isnan(result[constants.EpisodeConstants.IS_SUCCESSFUL])
    )
    self.assertEqual(
        'invalid_infrastructure', result['catbench_episode_status']
    )
    self.assertEqual('verifier', result['catbench_exception_stage'])
    self.assertEqual(
        'emulator_runtime_health_error',
        result['catbench_exception_failure_code'],
    )

  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_verifier_exception_is_attributed_and_invalid(self, mock_env):
    instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    instance.initialize_task = mock.MagicMock()
    instance.is_successful = mock.MagicMock(
        side_effect=RuntimeError('validator read failed')
    )
    instance.tear_down = mock.MagicMock()
    run_episode = mock.MagicMock(return_value=episode_runner.EpisodeResult(
        done=True, step_data={'step_number': [0]}
    ))

    result = suite_utils._run_task(
        instance, run_episode, mock_env, demo_mode=False
    )

    self.assertEqual('verifier', result['catbench_exception_stage'])
    self.assertEqual(
        'environment_or_evaluator',
        result['catbench_exception_attribution'],
    )
    self.assertEqual(
        'invalid_infrastructure', result['catbench_episode_status']
    )

  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_teardown_exception_invalidates_otherwise_valid_episode(self, mock_env):
    instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    instance.initialize_task = mock.MagicMock()
    instance.is_successful = mock.MagicMock(return_value=1.0)
    instance.tear_down = mock.MagicMock(
        side_effect=RuntimeError('cleanup failed')
    )
    run_episode = mock.MagicMock(return_value=episode_runner.EpisodeResult(
        done=True, step_data={'step_number': [0]}
    ))

    result = suite_utils._run_task(
        instance, run_episode, mock_env, demo_mode=False
    )

    self.assertEqual('teardown', result['catbench_exception_stage'])
    self.assertEqual(
        'environment_or_evaluator',
        result['catbench_exception_attribution'],
    )
    self.assertEqual(
        'invalid_infrastructure', result['catbench_episode_status']
    )

  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_run_adb_task_instances_is_successful_fails(self, mock_env):
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.return_value = episode_runner.EpisodeResult(
        True,
        {
            'step_number': [0],
        },
    )
    failing_instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    failing_instance.is_successful = lambda: ValueError('Something went wrong')

    result = suite_utils._run_task(
        failing_instance,
        mock_run_e2e,
        mock_env,
        demo_mode=False,
    )

    self.assertIsNotNone(result[constants.EpisodeConstants.EXCEPTION_INFO])

  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_run_task_skips_environment_network_error(self, mock_env):
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.return_value = episode_runner.EpisodeResult(
        True,
        {'step_number': [0]},
    )
    failing_instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )

    def _raise_network_error(unused_env):
      raise _cross_app_base._EnvironmentNetworkError(
          'network/connectivity error dialog visible in com.google.android.apps.maps'
      )

    failing_instance.is_successful = _raise_network_error

    result = suite_utils._run_task(
        failing_instance,
        mock_run_e2e,
        mock_env,
        demo_mode=False,
    )

    self.assertTrue(
        str(result[constants.EpisodeConstants.EXCEPTION_INFO]).startswith(
            '[skipped_environment]'
        )
    )
    self.assertTrue(
        np.isnan(result[constants.EpisodeConstants.IS_SUCCESSFUL])
    )

  @mock.patch.object(suite_utils, '_run_task_suite')
  @mock.patch.object(base_agent, 'EnvironmentInteractingAgent', autospec=True)
  def test_run(
      self,
      mock_agent,
      mock_run_suite,
  ):
    mock_run_suite.return_value = [{
        'goal': 'Goal',
        'is_successful': 1.0,
        'agent_name': 'AnAgent',
    }] * 2
    mock_agent.name = 'AnAgent'
    mock_agent.env = test_utils.FakeAsyncEnv()
    n_task_combinations = 1
    tasks = ['Task1']
    suite = suite_utils.create_suite(
        self.testing_registry,
        n_task_combinations=n_task_combinations,
        tasks=tasks,
    )

    results = suite_utils.run(suite, agent=mock_agent, demo_mode=False)

    mock_run_suite.assert_called_once()
    self.assertLen(results, 2)
    for result in results:
      self.assertEqual(result[constants.EpisodeConstants.GOAL], 'Goal')
      self.assertEqual(result[constants.EpisodeConstants.IS_SUCCESSFUL], True)
      self.assertEqual(result[constants.EpisodeConstants.AGENT_NAME], 'AnAgent')


class RunTaskSuiteTest(absltest.TestCase):

  def assertTaskResults(self, results: list[dict[str, Any]]) -> None:
    """Asserts that the tasks have executed as expected.

    Args:
      results: A list of dictionaries containing the result of task execution.
    """
    self.assertEqual(results[0]['is_successful'], 0)
    self.assertIn(results[0]['goal'], 'Current state eval')
    self.assertEqual(results[1]['is_successful'], 1)
    self.assertIn(results[1]['goal'], 'Current state eval')
    self.assertEqual(results[2]['is_successful'], 1)
    self.assertIn(results[2]['goal'], 'ADB eval')

  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_run_task_suite(self, mock_env):
    mock_env.get_state.return_value = (
        dm_env.TimeStep(
            observation={'pixels': np.zeros((3, 3, 3))},
            reward=0,
            discount=0,
            step_type=dm_env.StepType.LAST,
        ),
        [],
    )
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.side_effect = [
        # for each, add task_template
        episode_runner.EpisodeResult(
            False,
            {'step_number': [0]},
        ),
        episode_runner.EpisodeResult(
            True,
            {'step_number': [0]},
        ),
        episode_runner.EpisodeResult(
            True,
            {'step_number': [0]},
        ),
    ]

    suite = suite_utils.Suite(
        **{
            'Task1': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
            ],
            'Task2': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'android'
    result = suite_utils._run_task_suite(
        suite, mock_run_e2e, mock_env, demo_mode=False
    )

    self.assertTaskResults(result)

  @mock.patch.object(suite_utils, '_run_task')
  def test_run_task_suite_preserves_scheduled_instance_identity(
      self, mock_run_task
  ):
    instance = test_utils.FakeCurrentStateEval(
        test_utils.FakeCurrentStateEval.generate_random_params()
    )
    setattr(instance, '_catbench_instance_id', 2)
    suite = suite_utils.Suite({'FakeCurrentStateEval': [instance]})
    suite.suite_family = 'android'
    episode = {
        constants.EpisodeConstants.GOAL: 'Current state eval',
        constants.EpisodeConstants.TASK_TEMPLATE: 'FakeCurrentStateEval',
        constants.EpisodeConstants.IS_SUCCESSFUL: 1.0,
        constants.EpisodeConstants.EPISODE_LENGTH: 1,
        constants.EpisodeConstants.RUN_TIME: 0.0,
        constants.EpisodeConstants.EXCEPTION_INFO: None,
        constants.EpisodeConstants.AUX_DATA: None,
    }
    mock_run_task.return_value = episode
    mock_checkpointer = mock.MagicMock()
    mock_checkpointer.load.return_value = []

    result = suite_utils._run_task_suite(
        suite,
        mock.MagicMock(),
        test_utils.FakeAsyncEnv(),
        checkpointer=mock_checkpointer,
        process_episodes_fn=mock.MagicMock(),
    )

    self.assertEqual(
        result[0][constants.EpisodeConstants.INSTANCE_ID], 2
    )
    self.assertEqual(getattr(instance, '_catbench_instance_id'), 2)
    mock_checkpointer.save_episodes.assert_called_once_with(
        [episode], 'FakeCurrentStateEval_2'
    )

  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_run_task_suite_continues_after_required_breakdown_failure(
      self, mock_env
  ):
    mock_env.get_state.return_value = (
        dm_env.TimeStep(
            observation={'pixels': np.zeros((3, 3, 3))},
            reward=0,
            discount=0,
            step_type=dm_env.StepType.LAST,
        ),
        [],
    )
    missing = test_utils.FakeCurrentStateEval(
        test_utils.FakeCurrentStateEval.generate_random_params()
    )
    planned = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    suite = suite_utils.Suite(
        **{missing.name: [missing], planned.name: [planned]}
    )
    suite.suite_family = 'android'
    seen_prompt_goals = []

    def run_episode(task):
      # Mirrors the closure in suite_utils.run(): strict lookup before step.
      prompt_goal = suite_utils._get_task_breakdown_context(task).get(
          'prompt_goal', task.goal
      )
      seen_prompt_goals.append(prompt_goal)
      return episode_runner.EpisodeResult(True, {'step_number': [0]})

    task_breakdowns.clear_payload_cache()
    with tempfile.TemporaryDirectory() as tmpdir:
      plan_path = Path(tmpdir) / 'plans.json'
      plan_path.write_text(
          json.dumps({
              'breakdowns': [{
                  'key': task_breakdowns.make_key(
                      planned.name, planned.goal, 0
                  ),
                  'task_template': planned.name,
                  'instance_id': 0,
                  'goal_sha256': task_breakdowns.goal_sha256(planned.goal),
                  'breakdown': {'steps': ['Open the app.', 'Finish.']},
              }]
          }),
          encoding='utf-8',
      )
      with mock.patch.dict(
          suite_utils.os.environ,
          {
              'CATBENCH_CONDITION': 'c2_g',
              task_breakdowns.ENV_BREAKDOWN_FILE: str(plan_path),
              task_breakdowns.ENV_BREAKDOWN_REQUIRED: '1',
          },
      ):
        suite_utils.os.environ.pop(task_breakdowns.ENV_BREAKDOWN_MODE, None)
        result = suite_utils._run_task_suite(
            suite,
            run_episode,
            mock_env,
            demo_mode=False,
            return_full_episode_data=True,
        )

    self.assertLen(result, 2)
    failed, succeeded = result
    self.assertEqual(missing.name, failed['task_template'])
    self.assertIsNotNone(failed[constants.EpisodeConstants.EXCEPTION_INFO])
    self.assertEqual('invalid_infrastructure', failed['catbench_episode_status'])
    self.assertFalse(failed['catbench_condition_config_valid'])
    self.assertEqual(planned.name, succeeded['task_template'])
    self.assertIsNone(succeeded[constants.EpisodeConstants.EXCEPTION_INFO])
    self.assertEqual(1, succeeded['is_successful'])
    self.assertTrue(succeeded['catbench_condition_config_valid'])
    self.assertLen(seen_prompt_goals, 1)
    self.assertIn(task_breakdowns.BREAKDOWN_HEADER, seen_prompt_goals[0])
    self.assertEqual(
        planned.goal,
        task_breakdowns.original_goal_from_prompt_goal(seen_prompt_goals[0]),
    )

  @mock.patch.object(time, 'sleep', autospec=True)
  @mock.patch.object(interface, 'AsyncAndroidEnv')
  @mock.patch.object(adb_utils, 'send_android_intent')
  @mock.patch.object(checkpointer, 'Checkpointer')
  def test_resume_from_middle(
      self,
      mock_checkpointer,
      unused_mock_send_android_intent,
      mock_env,
      unused_mock_sleep,
  ):
    # Simulating partially completed Task1
    mock_checkpointer.load.return_value = [
        {
            'instance_id': 0,
            'is_successful': 0.0,
            'goal': 'Current state eval',
            'task_template': 'FakeCurrentStateEval',
            'episode_length': 1,
            'run_time': 0,
        },
    ]
    mock_env.get_state.return_value = (
        dm_env.TimeStep(
            observation={'pixels': np.zeros((3, 3, 3))},
            reward=0,
            discount=0,
            step_type=dm_env.StepType.LAST,
        ),
        [],
    )
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.side_effect = [
        episode_runner.EpisodeResult(
            True,
            {'step_number': [0]},
        ),
        episode_runner.EpisodeResult(
            True,
            {'step_number': [0]},
        ),
    ]

    suite = suite_utils.Suite(
        **{
            'FakeCurrentStateEval': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
            ],
            'FakeAdbEval': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'android'
    result = suite_utils._run_task_suite(
        suite, mock_run_e2e, mock_env, mock_checkpointer
    )

    self.assertTaskResults(result)
    mock_checkpointer.load.assert_called_once()
    # Run one instance for Task1, one instance for Task2
    mock_run_e2e.assert_called()
    mock_checkpointer.save_episodes.assert_has_calls([
        mock.call(mock.ANY, 'FakeCurrentStateEval_1'),
        mock.call(mock.ANY, 'FakeAdbEval_0'),
    ])

  @mock.patch.object(time, 'sleep', autospec=True)
  @mock.patch.object(interface, 'AsyncAndroidEnv')
  @mock.patch.object(adb_utils, 'send_android_intent')
  @mock.patch.object(checkpointer, 'Checkpointer')
  def test_start_from_beginning(
      self,
      mock_checkpointer,
      unused_mock_send_android_intent,
      mock_env,
      unused_mock_sleep,
  ):
    mock_checkpointer.load.return_value = []
    mock_env.get_state.return_value = (
        dm_env.TimeStep(
            observation={'pixels': np.zeros((3, 3, 3))},
            reward=0,
            discount=0,
            step_type=dm_env.StepType.LAST,
        ),
        [],
    )
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.side_effect = [
        episode_runner.EpisodeResult(
            False,
            {'step_number': [0]},
        ),
        episode_runner.EpisodeResult(
            True,
            {'step_number': [0]},
        ),
        episode_runner.EpisodeResult(
            True,
            {'step_number': [0]},
        ),
    ]
    suite = suite_utils.Suite(
        **{
            'FakeCurrentStateEval': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
            ],
            'FakeAdbEval': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'android'

    result = suite_utils._run_task_suite(
        suite, mock_run_e2e, mock_env, mock_checkpointer
    )

    self.assertTaskResults(result)
    self.assertEqual(mock_run_e2e.call_count, 3)
    mock_checkpointer.load.assert_called_once()
    mock_checkpointer.save_episodes.assert_has_calls(
        [
            mock.call(mock.ANY, 'FakeCurrentStateEval_0'),
            mock.call(mock.ANY, 'FakeCurrentStateEval_1'),
            mock.call(mock.ANY, 'FakeAdbEval_0'),
        ],
        any_order=False,
    )

  @mock.patch.object(time, 'sleep', autospec=True)
  @mock.patch.object(interface, 'AsyncAndroidEnv')
  @mock.patch.object(adb_utils, 'send_android_intent')
  @mock.patch.object(checkpointer, 'Checkpointer')
  def test_start_from_end(
      self,
      mock_checkpointer,
      unused_mock_send_android_intent,
      mock_env,
      unused_mock_sleep,
  ):
    mock_checkpointer.load.return_value = [
        {
            'instance_id': 0,
            'is_successful': 0,
            'goal': 'Current state eval',
            'task_template': 'FakeCurrentStateEval',
        },
        {
            'instance_id': 1,
            'is_successful': 1,
            'goal': 'Current state eval',
            'task_template': 'FakeCurrentStateEval',
        },
        {
            'instance_id': 0,
            'is_successful': 1,
            'goal': 'ADB eval',
            'task_template': 'FakeAdbEval',
        },
    ]
    mock_run_e2e = mock.MagicMock()
    suite = suite_utils.Suite(
        **{
            'FakeCurrentStateEval': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
            ],
            'FakeAdbEval': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'android'

    result = suite_utils._run_task_suite(
        suite, mock_run_e2e, mock_env, mock_checkpointer
    )

    self.assertTaskResults(result)
    mock_run_e2e.assert_not_called()
    mock_checkpointer.load.assert_called_once()
    mock_checkpointer.save.assert_not_called()

  @mock.patch.object(time, 'sleep', autospec=True)
  @mock.patch.object(interface, 'AsyncAndroidEnv')
  @mock.patch.object(adb_utils, 'send_android_intent')
  @mock.patch.object(checkpointer, 'Checkpointer')
  def test_result_suite_equal_in_number(
      self,
      mock_checkpointer,
      unused_mock_send_android_intent,
      mock_env,
      unused_mock_sleep,
  ):
    mock_checkpointer.load.return_value = [
        {
            'instance_id': 0,
            'is_successful': 0,
            'goal': 'Current state eval',
            'task_template': 'FakeCurrentStateEval',
        },
        {
            'instance_id': 0,
            'is_successful': 1,
            'goal': 'ADB eval',
            'task_template': 'FakeAdbEval',
        },
    ]
    mock_run_e2e = mock.MagicMock()
    suite = suite_utils.Suite(
        **{
            'FakeCurrentStateEval': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
            ],
            'FakeAdbEval': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'android'

    result = suite_utils._run_task_suite(
        suite, mock_run_e2e, mock_env, mock_checkpointer
    )
    self.assertLen(result, 2)

    suite2 = suite_utils.Suite(
        **{
            'FakeAdbEval': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    result2 = suite_utils._run_task_suite(
        suite2, mock_run_e2e, mock_env, mock_checkpointer
    )
    self.assertLen(result2, 1)


if __name__ == '__main__':
  absltest.main()
