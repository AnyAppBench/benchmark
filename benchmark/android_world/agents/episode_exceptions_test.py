"""Tests for typed episode-exception attribution."""

from __future__ import annotations

import unittest

from android_world.agents import episode_exceptions


class EpisodeExceptionsTest(unittest.TestCase):

  def test_declared_parse_error_in_agent_stage_is_valid_failure(self):
    attribution = episode_exceptions.attribute_exception(
        episode_exceptions.ActionParseError("missing tool call"), "agent"
    )

    self.assertTrue(attribution.valid_agent_failure)
    self.assertTrue(attribution.declared_agent_output_exception)
    self.assertEqual(
        "agent_output_parse_or_malformed_action", attribution.attribution
    )
    self.assertEqual("action_parse_error", attribution.failure_code)

  def test_declared_malformed_action_in_agent_stage_is_valid_failure(self):
    attribution = episode_exceptions.attribute_exception(
        episode_exceptions.MalformedActionError("coordinate is absent"),
        "agent",
    )

    self.assertTrue(attribution.valid_agent_failure)
    self.assertEqual("malformed_action_error", attribution.failure_code)

  def test_generic_agent_value_error_is_unknown_and_invalid(self):
    attribution = episode_exceptions.attribute_exception(
        ValueError("could be runner code"), "agent"
    )

    self.assertFalse(attribution.valid_agent_failure)
    self.assertFalse(attribution.declared_agent_output_exception)
    self.assertEqual("unknown", attribution.attribution)

  def test_endpoint_error_is_never_a_scored_agent_failure(self):
    attribution = episode_exceptions.attribute_exception(
        episode_exceptions.ModelEndpointError("context limit"), "agent"
    )

    self.assertFalse(attribution.valid_agent_failure)
    self.assertFalse(attribution.declared_agent_output_exception)
    self.assertEqual("environment_or_evaluator", attribution.attribution)
    self.assertEqual("model_endpoint_error", attribution.failure_code)

  def test_emulator_health_error_in_agent_stage_is_environment(self):
    attribution = episode_exceptions.attribute_exception(
        episode_exceptions.EmulatorRuntimeHealthError(
            "visible crash dialog"
        ),
        "agent",
    )

    self.assertFalse(attribution.valid_agent_failure)
    self.assertEqual("environment_or_evaluator", attribution.attribution)
    self.assertEqual(
        "emulator_runtime_health_error", attribution.failure_code
    )

  def test_declared_type_outside_agent_stage_is_invalid(self):
    attribution = episode_exceptions.attribute_exception(
        episode_exceptions.ActionParseError("verifier misuse"), "verifier"
    )

    self.assertFalse(attribution.valid_agent_failure)
    self.assertTrue(attribution.declared_agent_output_exception)
    self.assertEqual("environment_or_evaluator", attribution.attribution)

  def test_known_environment_error_overrides_declared_agent_type(self):
    attribution = episode_exceptions.attribute_exception(
        episode_exceptions.ActionParseError("environment-owned response"),
        "agent",
        known_environment_or_evaluator=True,
    )

    self.assertFalse(attribution.valid_agent_failure)
    self.assertEqual("environment_or_evaluator", attribution.attribution)

  def test_all_non_agent_stages_are_invalid_infrastructure(self):
    for stage in ("initialize", "verifier", "teardown"):
      with self.subTest(stage=stage):
        attribution = episode_exceptions.attribute_exception(
            RuntimeError("stage failure"), stage
        )
        self.assertFalse(attribution.valid_agent_failure)
        self.assertEqual(
            "environment_or_evaluator", attribution.attribution
        )

  def test_episode_fields_are_explicit(self):
    fields = episode_exceptions.attribute_exception(
        episode_exceptions.ActionParseError("bad action"), "agent"
    ).as_episode_fields()

    self.assertEqual("agent", fields["catbench_exception_stage"])
    self.assertTrue(fields["catbench_exception_valid_agent_failure"])
    self.assertIn("ActionParseError", fields["catbench_exception_type"])


if __name__ == "__main__":
  unittest.main()
