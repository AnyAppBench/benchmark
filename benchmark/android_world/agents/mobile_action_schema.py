"""Explicit schema boundary for model-emitted ``mobile_use`` calls.

Only violations proven from the model-owned JSON are raised as declared agent
output errors.  Callers must allow every other converter/runtime exception to
escape so CATBench records an invalid episode rather than a scored failure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from numbers import Real
from typing import Any

from android_world.agents import episode_exceptions


SUPPORTED_MOBILE_ACTIONS = frozenset({
    "click",
    "terminate",
    "answer",
    "long_press",
    "type",
    "swipe",
    "wait",
    "system_button",
    "open",
    "open_app",
})


def _validated_coordinate(value: Any, field_name: str) -> list[Real]:
  """Returns a model-owned point, reducing a bounding box to its center."""
  if (
      not isinstance(value, Sequence)
      or isinstance(value, (str, bytes))
      or len(value) not in {2, 4}
      or not all(isinstance(item, Real) for item in value)
  ):
    raise episode_exceptions.MalformedActionError(
        f"mobile_use.{field_name} must contain two coordinates or a "
        "four-number bounding box."
    )
  if len(value) == 4:
    return [(value[0] + value[2]) / 2, (value[1] + value[3]) / 2]
  return [value[0], value[1]]


def validate_mobile_agent_action(dummy_action: Any) -> dict[str, Any]:
  """Validates and normalizes one parsed model-emitted tool call."""
  if not isinstance(dummy_action, Mapping):
    raise episode_exceptions.MalformedActionError(
        "mobile_use tool call must be a JSON object."
    )
  if dummy_action.get("name") != "mobile_use":
    raise episode_exceptions.MalformedActionError(
        "Expected a mobile_use tool call."
    )
  raw_arguments = dummy_action.get("arguments")
  if not isinstance(raw_arguments, Mapping):
    raise episode_exceptions.MalformedActionError(
        "mobile_use.arguments must be a JSON object."
    )

  normalized = {"name": "mobile_use", "arguments": dict(raw_arguments)}
  arguments = normalized["arguments"]
  raw_action = arguments.get("action")
  if not isinstance(raw_action, str) or not raw_action.strip():
    raise episode_exceptions.MalformedActionError(
        "mobile_use.arguments.action must be a non-empty string."
    )
  action = raw_action.strip().lower().replace("tap", "click")
  if action not in SUPPORTED_MOBILE_ACTIONS:
    raise episode_exceptions.MalformedActionError(
        f"Unsupported mobile_use action: {raw_action!r}."
    )
  arguments["action"] = action

  if action in {"click", "long_press"}:
    arguments["coordinate"] = _validated_coordinate(
        arguments.get("coordinate"), "coordinate"
    )
  elif action == "swipe":
    has_coordinate = "coordinate" in arguments
    has_coordinate2 = "coordinate2" in arguments
    direction = arguments.get("direction")
    if has_coordinate:
      arguments["coordinate"] = _validated_coordinate(
          arguments["coordinate"], "coordinate"
      )
    if has_coordinate2:
      arguments["coordinate2"] = _validated_coordinate(
          arguments["coordinate2"], "coordinate2"
      )
    if not (has_coordinate and has_coordinate2) and direction not in {
        "up",
        "down",
        "left",
        "right",
    }:
      raise episode_exceptions.MalformedActionError(
          "swipe requires coordinate+coordinate2 or a valid direction."
      )
  elif action in {"type", "answer", "open", "open_app"}:
    if not isinstance(arguments.get("text"), str):
      raise episode_exceptions.MalformedActionError(
          f"{action} requires a string text argument."
      )
  elif action == "system_button":
    button = arguments.get("button")
    if not isinstance(button, str) or button.lower() not in {
        "back",
        "home",
        "enter",
    }:
      raise episode_exceptions.MalformedActionError(
          "system_button requires Back, Home, or Enter."
      )
  elif action == "terminate":
    status = arguments.get("status")
    if status is not None and not isinstance(status, str):
      raise episode_exceptions.MalformedActionError(
          "terminate.status must be a string when supplied."
      )
  return normalized


def parse_mobile_agent_tool_call(raw_tool_call: Any) -> dict[str, Any]:
  """Parses JSON and applies the explicit model-action schema."""
  if not isinstance(raw_tool_call, str) or not raw_tool_call.strip():
    raise episode_exceptions.ActionParseError(
        "mobile_use tool call is empty."
    )
  try:
    parsed = json.loads(raw_tool_call)
  except json.JSONDecodeError as error:
    raise episode_exceptions.ActionParseError(
        "mobile_use tool call is not valid JSON."
    ) from error
  return validate_mobile_agent_action(parsed)
