"""Tests for the explicit model-owned mobile action schema."""

from __future__ import annotations

import unittest

from android_world.agents import episode_exceptions
from android_world.agents import mobile_action_schema


class MobileActionSchemaTest(unittest.TestCase):

  def test_parses_and_normalizes_valid_call(self):
    parsed = mobile_action_schema.parse_mobile_agent_tool_call(
        '{"name":"mobile_use","arguments":'
        '{"action":"tap","coordinate":[10,20]}}'
    )

    self.assertEqual("click", parsed["arguments"]["action"])
    self.assertEqual([10, 20], parsed["arguments"]["coordinate"])

  def test_bounding_box_is_reduced_to_center(self):
    parsed = mobile_action_schema.validate_mobile_agent_action({
        "name": "mobile_use",
        "arguments": {"action": "click", "coordinate": [2, 4, 10, 20]},
    })

    self.assertEqual([6, 12], parsed["arguments"]["coordinate"])

  def test_invalid_json_is_declared_parse_error(self):
    with self.assertRaises(episode_exceptions.ActionParseError):
      mobile_action_schema.parse_mobile_agent_tool_call("{not json}")

  def test_missing_action_field_is_declared_malformed_action(self):
    with self.assertRaises(episode_exceptions.MalformedActionError):
      mobile_action_schema.validate_mobile_agent_action({
          "name": "mobile_use",
          "arguments": {},
      })

  def test_unknown_action_is_declared_malformed_action(self):
    with self.assertRaises(episode_exceptions.MalformedActionError):
      mobile_action_schema.validate_mobile_agent_action({
          "name": "mobile_use",
          "arguments": {"action": "invent_button"},
      })


if __name__ == "__main__":
  unittest.main()
