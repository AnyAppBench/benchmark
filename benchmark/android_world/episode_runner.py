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

"""Runs an agent on the environment."""

import dataclasses
from typing import Any, Callable, Optional
from android_world import constants
from android_world.agents import base_agent
from android_world.env import interface
import termcolor


@dataclasses.dataclass()
class EpisodeResult:
  """Represents an episode of an agent interacting with the environment.

  Attributes:
    done: Whether the agent indicated the task is complete.
    step_data: Environment and agent data for each step.
    env_reward: Reward returned by environment, if applicable.
    aux_data: Additional data from the episode which may be used for metrics.
  """

  done: bool
  step_data: dict[str, Any]
  env_reward: Optional[float] = None
  aux_data: Optional[dict[str, Any]] = None


def _extract_step_reasoning(step_data: dict[str, Any]) -> str | None:
  """Returns best-effort reasoning text from a step payload."""
  if not isinstance(step_data, dict):
    return None

  # Prefer explicit reason fields when available.
  for key in ('action_reason', 'thought', 'reason'):
    value = step_data.get(key)
    if isinstance(value, str) and value.strip():
      return value.strip()

  # Fallback to summary text if no explicit reasoning was provided.
  summary = step_data.get('summary')
  if isinstance(summary, str) and summary.strip():
    return summary.strip()

  return None


def run_episode(
    goal: str,
    agent: base_agent.EnvironmentInteractingAgent,
    max_n_steps: int = 10,
    start_on_home_screen: bool = False,
    termination_fn: Callable[[interface.AsyncEnv], float] | None = None,
    health_check_fn: Callable[[interface.AsyncEnv], None] | None = None,
    print_fn: Callable[[str], None] = print,
) -> EpisodeResult:
  """Runs an agent on goal, e.g., "turn off wifi".

  An agent will start from whatever state the provided environment is in and
  run until it determines a task is complete, if the max number of
  steps is reached, of if the termination_fn is True.

  Args:
    goal: The goal instruction for the agent.
    agent: The agent to run on the environment.
    max_n_steps: The max number of steps to allow an agent to run before ending
      an episode.
    start_on_home_screen: Whether to start episode from the home screen or just
      the current screen.
    termination_fn: If provided, a determines whether to terminate an episode.
      For example, for MiniWoB++ tasks, the episode should terminate if there is
      a nonzero reward.
    health_check_fn: If provided, validates the runtime after reset and after
      every agent action, before success/done is accepted. Exceptions propagate
      to the suite's infrastructure-attribution path.
    print_fn: A function to print log messages to the console or logger.

  Returns:
    Data collected during running agent on goal.
  """
  if max_n_steps == 0:
    if health_check_fn is not None:
      health_check_fn(agent.env)
    return EpisodeResult(done=False, step_data={})
  if termination_fn is None:
    termination_fn = lambda env: False

  agent.reset(start_on_home_screen)
  agent.set_max_steps(max_n_steps)
  if health_check_fn is not None:
    health_check_fn(agent.env)

  output = []
  for step_n in range(max_n_steps):
    result = agent.step(goal)
    if health_check_fn is not None:
      health_check_fn(agent.env)
    reasoning = _extract_step_reasoning(result.data)
    if reasoning:
      print_fn('Reasoning step {:d}: {}'.format(step_n + 1, reasoning))
    print_fn('Completed step {:d}.'.format(step_n + 1))
    assert constants.STEP_NUMBER not in result.data
    output.append(result.data | {constants.STEP_NUMBER: step_n})
    if termination_fn(agent.env):
      print_fn('Environment ends episode.')
      return EpisodeResult(
          done=True,
          step_data=_transpose_lod_to_dol(output),
      )
    elif result.done:
      print_fn('Agent indicates task is done.')
      return EpisodeResult(
          done=result.done,
          step_data=_transpose_lod_to_dol(output),
      )
  print_fn(
      termcolor.colored(
          'Agent did not indicate task is done. Reached max number of steps.',
          'red',
      )
  )
  return EpisodeResult(
      done=result.done, step_data=_transpose_lod_to_dol(output)  # pylint: disable=undefined-variable
  )


def _transpose_lod_to_dol(data: list[dict[str, Any]]) -> dict[str, list[Any]]:
  """Transposes a list of dictionaries to a dictionary of lists.

  Args:
    data: A list of dictionaries.

  Returns:
    A dictionary where each key is from the input dictionaries and each value is
    a list of values for that key.
  """
  result = {}
  for d in data:
    for key, value in d.items():
      if key not in result:
        result[key] = []
      result[key].append(value)
  return result


def transpose_dol_to_lod(data: dict[str, list[Any]]) -> list[dict[str, Any]]:
  """Converts a dictionary of lists to a list of dictionaries.

  Useful for post-processing of results; e.g., in colab.

  Args:
    data: A dictionary where each value is a list.

  Returns:
    A list of dictionaries.
  """
  return [dict(zip(data.keys(), values)) for values in zip(*data.values())]
