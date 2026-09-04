"""Tests for CATBench task-breakdown prompt utilities."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_breakdowns


class TaskBreakdownsTest(unittest.TestCase):

  def test_revision_default_uses_three_exact_instances(self) -> None:
    self.assertEqual(3, task_breakdowns.DEFAULT_N_TASK_COMBINATIONS)

  def setUp(self) -> None:
    task_breakdowns.clear_payload_cache()

  def tearDown(self) -> None:
    task_breakdowns.clear_payload_cache()

  def _write_payload(self, payload: dict) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
    with handle:
      json.dump(payload, handle)
    return handle.name

  def test_build_prompt_context_prepends_breakdown(self) -> None:
    goal = "Send Alice the message hello."
    path = self._write_payload(
        {
            "breakdowns": [
                {
                    "key": task_breakdowns.make_key("SmsTask", goal, 0),
                    "task_template": "SmsTask",
                    "instance_id": 0,
                    "goal": goal,
                    "goal_sha256": task_breakdowns.goal_sha256(goal),
                    "generator_model": "gemini-3.1-pro-preview",
                    "breakdown": {"steps": ["Identify recipient.", "Send text."]},
                }
            ]
        }
    )

    with mock.patch.dict(
        os.environ,
        {
            task_breakdowns.ENV_BREAKDOWN_FILE: path,
            task_breakdowns.ENV_BREAKDOWN_MODE: "prepend",
        },
        clear=False,
    ):
      context = task_breakdowns.build_prompt_context("SmsTask", goal, 0)

    self.assertTrue(context["enabled"])
    self.assertTrue(context["found"])
    self.assertIn("Application-independent task breakdown", context["prompt_goal"])
    self.assertIn("Original user instruction:", context["prompt_goal"])
    self.assertIn(goal, context["prompt_goal"])
    self.assertIn("1. Identify recipient.", context["task_breakdown_text"])

  def test_build_prompt_context_original_first_keeps_target_app_salient(
      self,
  ) -> None:
    goal = "Using the Fossify Messages app, send Alice the message hello."
    path = self._write_payload(
        {
            "breakdowns": [
                {
                    "key": task_breakdowns.make_key("SmsTask", goal, 0),
                    "task_template": "SmsTask",
                    "instance_id": 0,
                    "goal_sha256": task_breakdowns.goal_sha256(goal),
                    "breakdown": {"steps": ["Open the messaging app."]},
                }
            ]
        }
    )

    with mock.patch.dict(
        os.environ,
        {
            task_breakdowns.ENV_BREAKDOWN_FILE: path,
            task_breakdowns.ENV_BREAKDOWN_MODE: "original_first",
        },
        clear=False,
    ):
      context = task_breakdowns.build_prompt_context("SmsTask", goal, 0)

    prompt_goal = context["prompt_goal"]
    self.assertLess(
        prompt_goal.index("Original user instruction"),
        prompt_goal.index("Application-independent task breakdown"),
    )
    self.assertIn("Do not substitute another app.", prompt_goal)
    self.assertIn(goal, prompt_goal)

  def test_required_missing_breakdown_raises(self) -> None:
    path = self._write_payload({"breakdowns": []})
    with mock.patch.dict(
        os.environ,
        {
            task_breakdowns.ENV_BREAKDOWN_FILE: path,
            task_breakdowns.ENV_BREAKDOWN_MODE: "prepend",
            task_breakdowns.ENV_BREAKDOWN_REQUIRED: "1",
        },
        clear=False,
    ):
      with self.assertRaises(KeyError):
        task_breakdowns.build_prompt_context(
            "SmsTask", "Send Alice hello.", 0
        )

  def test_disabled_without_file(self) -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
      context = task_breakdowns.build_prompt_context(
          "SmsTask", "Do something.", None
      )
    self.assertFalse(context["enabled"])
    self.assertEqual("Do something.", context["prompt_goal"])

  def test_compact_key_payload_resolves(self) -> None:
    """The compact `{"Task|hash": {...}}` payload shape must still match."""
    goal = "Add Bob to contacts."
    key = task_breakdowns.make_key("ContactsAddContact", goal, 0)
    path = self._write_payload(
        {
            key: {
                "instance_id": 0,
                "breakdown": {"steps": ["Pick a contacts surface."]},
            }
        }
    )
    with mock.patch.dict(
        os.environ,
        {
            task_breakdowns.ENV_BREAKDOWN_FILE: path,
            task_breakdowns.ENV_BREAKDOWN_MODE: "prepend",
        },
        clear=False,
    ):
      context = task_breakdowns.build_prompt_context(
          "ContactsAddContact", goal, 0
      )
    self.assertTrue(context["found"])
    self.assertIn("Pick a contacts surface.", context["task_breakdown_text"])

  def test_cross_template_guard_blocks_match(self) -> None:
    """Same goal across two templates must NOT swap breakdowns."""
    goal = "Open the menu."
    path = self._write_payload(
        {
            "breakdowns": [
                {
                    "key": task_breakdowns.make_key("TemplateA", goal, 0),
                    "task_template": "TemplateA",
                    "instance_id": 0,
                    "goal_sha256": task_breakdowns.goal_sha256(goal),
                    "goal": goal,
                    "breakdown": {"steps": ["A-specific step."]},
                }
            ]
        }
    )
    with mock.patch.dict(
        os.environ,
        {
            task_breakdowns.ENV_BREAKDOWN_FILE: path,
            task_breakdowns.ENV_BREAKDOWN_MODE: "prepend",
        },
        clear=False,
    ):
      context_a = task_breakdowns.build_prompt_context("TemplateA", goal, 0)
      context_b = task_breakdowns.build_prompt_context("TemplateB", goal, 0)
    self.assertTrue(context_a["found"])
    self.assertFalse(context_b["found"])

  def test_empty_steps_marks_empty_breakdown(self) -> None:
    """Entry with no usable steps must surface an empty_breakdown marker."""
    goal = "Do the thing."
    path = self._write_payload(
        {
            "breakdowns": [
                {
                    "key": task_breakdowns.make_key("T", goal, 0),
                    "task_template": "T",
                    "instance_id": 0,
                    "goal_sha256": task_breakdowns.goal_sha256(goal),
                    "breakdown": {"steps": []},
                }
            ]
        }
    )
    with mock.patch.dict(
        os.environ,
        {
            task_breakdowns.ENV_BREAKDOWN_FILE: path,
            task_breakdowns.ENV_BREAKDOWN_MODE: "prepend",
        },
        clear=False,
    ):
      context = task_breakdowns.build_prompt_context("T", goal, 0)
    self.assertFalse(context["found"])
    self.assertTrue(context["task_breakdown_metadata"].get("empty_breakdown"))

  def test_required_empty_raises(self) -> None:
    goal = "Do the thing."
    path = self._write_payload(
        {
            "breakdowns": [
                {
                    "key": task_breakdowns.make_key("T", goal, 0),
                    "task_template": "T",
                    "instance_id": 0,
                    "goal_sha256": task_breakdowns.goal_sha256(goal),
                    "breakdown": {"steps": []},
                }
            ]
        }
    )
    with mock.patch.dict(
        os.environ,
        {
            task_breakdowns.ENV_BREAKDOWN_FILE: path,
            task_breakdowns.ENV_BREAKDOWN_MODE: "prepend",
            task_breakdowns.ENV_BREAKDOWN_REQUIRED: "1",
        },
        clear=False,
    ):
      with self.assertRaises(ValueError):
        task_breakdowns.build_prompt_context("T", goal, 0)

  def test_format_breakdown_text_variants(self) -> None:
    self.assertEqual(
        "1. Step a\n2. Step b",
        task_breakdowns.format_breakdown_text(
            {"breakdown": ["Step a", "Step b"]}
        ),
    )
    self.assertEqual(
        "1. Step a",
        task_breakdowns.format_breakdown_text(
            {"breakdown": {"steps": ["Step a"]}}
        ),
    )
    self.assertEqual(
        "raw text",
        task_breakdowns.format_breakdown_text({"breakdown": "raw text"}),
    )
    self.assertEqual(
        "literal",
        task_breakdowns.format_breakdown_text({"breakdown_text": "literal"}),
    )
    self.assertEqual("", task_breakdowns.format_breakdown_text({"breakdown": {}}))

  def test_app_neutral_goal_replaces_only_target_app_identity(self) -> None:
    neutral = task_breakdowns.app_neutral_goal(
        "Using the Fossify Clock app, create an alarm for 08:30.",
        "Fossify Clock",
    )

    self.assertEqual(
        "Using the [TARGET_APP] app, create an alarm for 08:30.", neutral
    )

  def test_app_neutral_goal_fails_closed_when_app_is_absent(self) -> None:
    with self.assertRaises(ValueError):
      task_breakdowns.app_neutral_goal("Create an alarm for 08:30.", "Chrono")

  def test_semantic_plan_key_is_shared_across_app_routing(self) -> None:
    chrono_goal = task_breakdowns.app_neutral_goal(
        "Using Chrono, create an alarm for 08:30.", "Chrono"
    )
    clock_goal = task_breakdowns.app_neutral_goal(
        "Using Google Clock, create an alarm for 08:30.", "Google Clock"
    )

    self.assertEqual(
        task_breakdowns.make_semantic_plan_key("ClockCreateAlarm", 0, chrono_goal),
        task_breakdowns.make_semantic_plan_key("ClockCreateAlarm", 0, clock_goal),
    )

  def test_same_goal_instances_resolve_distinct_exact_entries(self) -> None:
    goal = "Create an alarm for 08:30."
    entries = []
    for instance_id, step in ((0, "Plan for instance zero."), (1, "Plan for instance one.")):
      entries.append(
          {
              "key": task_breakdowns.make_key(
                  "ClockCreateAlarm", goal, instance_id
              ),
              "task_template": "ClockCreateAlarm",
              "instance_id": instance_id,
              "goal": goal,
              "goal_sha256": task_breakdowns.goal_sha256(goal),
              "breakdown": {"steps": [step]},
          }
      )
    path = self._write_payload({"breakdowns": entries})
    with mock.patch.dict(
        os.environ,
        {
            task_breakdowns.ENV_BREAKDOWN_FILE: path,
            task_breakdowns.ENV_BREAKDOWN_MODE: "prepend",
        },
        clear=False,
    ):
      first = task_breakdowns.build_prompt_context(
          "ClockCreateAlarm", goal, 0
      )
      second = task_breakdowns.build_prompt_context(
          "ClockCreateAlarm", goal, 1
      )
      missing = task_breakdowns.build_prompt_context(
          "ClockCreateAlarm", goal, 2
      )

    self.assertIn("Plan for instance zero.", first["task_breakdown_text"])
    self.assertIn("Plan for instance one.", second["task_breakdown_text"])
    self.assertEqual(0, first["task_breakdown_metadata"]["instance_id"])
    self.assertEqual(1, second["task_breakdown_metadata"]["instance_id"])
    self.assertFalse(missing["found"])
    self.assertTrue(missing["task_breakdown_metadata"]["lookup_missing"])

  def test_legacy_key_without_instance_id_fails_closed(self) -> None:
    goal = "Create an alarm for 08:30."
    legacy_key = f"ClockCreateAlarm|{task_breakdowns.goal_sha256(goal)}"
    path = self._write_payload(
        {
            "breakdowns": [
                {
                    "key": legacy_key,
                    "task_template": "ClockCreateAlarm",
                    "goal": goal,
                    "goal_sha256": task_breakdowns.goal_sha256(goal),
                    "breakdown": {"steps": ["Ambiguous legacy plan."]},
                }
            ]
        }
    )
    with mock.patch.dict(
        os.environ,
        {
            task_breakdowns.ENV_BREAKDOWN_FILE: path,
            task_breakdowns.ENV_BREAKDOWN_MODE: "prepend",
        },
        clear=False,
    ):
      context = task_breakdowns.build_prompt_context(
          "ClockCreateAlarm", goal, 0
      )
    self.assertFalse(context["found"])
    self.assertTrue(context["task_breakdown_metadata"]["lookup_missing"])

  def test_duplicate_exact_instance_entries_raise(self) -> None:
    goal = "Create an alarm for 08:30."
    entry = {
        "key": task_breakdowns.make_key("ClockCreateAlarm", goal, 0),
        "task_template": "ClockCreateAlarm",
        "instance_id": 0,
        "goal_sha256": task_breakdowns.goal_sha256(goal),
        "breakdown": {"steps": ["One plan."]},
    }
    path = self._write_payload({"breakdowns": [entry, dict(entry)]})
    with mock.patch.dict(
        os.environ,
        {
            task_breakdowns.ENV_BREAKDOWN_FILE: path,
            task_breakdowns.ENV_BREAKDOWN_MODE: "prepend",
        },
        clear=False,
    ):
      with self.assertRaisesRegex(ValueError, "Duplicate task breakdown"):
        task_breakdowns.build_prompt_context("ClockCreateAlarm", goal, 0)

  def test_enabled_lookup_without_runtime_instance_id_fails_closed(self) -> None:
    goal = "Create an alarm for 08:30."
    path = self._write_payload({"breakdowns": []})
    with mock.patch.dict(
        os.environ,
        {
            task_breakdowns.ENV_BREAKDOWN_FILE: path,
            task_breakdowns.ENV_BREAKDOWN_MODE: "prepend",
        },
        clear=False,
    ):
      context = task_breakdowns.build_prompt_context(
          "ClockCreateAlarm", goal, None
      )
    self.assertFalse(context["found"])
    self.assertTrue(
        context["task_breakdown_metadata"]["missing_instance_id"]
    )

  def test_validate_runner_compatibility_reports_mismatches(self) -> None:
    path = self._write_payload(
        {
            "metadata": {
                "task_random_seed": 30,
                "n_task_combinations": 1,
                "fixed_task_seed": False,
                "suite_family": "android_world",
            },
            "breakdowns": [],
        }
    )
    issues = task_breakdowns.validate_runner_compatibility(
        runner_seed=31,
        runner_n_task_combinations=1,
        runner_fixed_task_seed=False,
        runner_suite_family="android_world",
        path=path,
    )
    self.assertEqual(1, len(issues))
    self.assertIn("task_random_seed", issues[0])

    issues = task_breakdowns.validate_runner_compatibility(
        runner_seed=30,
        runner_n_task_combinations=1,
        runner_fixed_task_seed=False,
        runner_suite_family="android_world",
        path=path,
    )
    self.assertEqual([], issues)

  def test_original_goal_round_trips_through_every_prompt_mode(self) -> None:
    goal = "Using the Clock You app, set an alarm for 7:30 AM labelled 'Gym'."
    # A breakdown that itself contains the marker text must not confuse the
    # parser: the original instruction is always recovered exactly.
    breakdown_text = (
        "1. Open the clock app.\n"
        "2. Original user instruction:\nis repeated here by the planner.\n"
        "3. Save the alarm."
    )
    for mode in ("prepend", "original_first", "original-first", "append",
                 "unknown_mode"):
      with self.subTest(mode=mode):
        prompt_goal = task_breakdowns.build_prompt_goal(
            goal, breakdown_text, mode
        )
        self.assertNotEqual(goal, prompt_goal)
        self.assertEqual(
            goal, task_breakdowns.original_goal_from_prompt_goal(prompt_goal)
        )

    # Default mode (MODE unset) must round-trip as well.
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop(task_breakdowns.ENV_BREAKDOWN_MODE, None)
      prompt_goal = task_breakdowns.build_prompt_goal(goal, breakdown_text)
    self.assertEqual(
        goal, task_breakdowns.original_goal_from_prompt_goal(prompt_goal)
    )

  def test_original_goal_from_plain_goal_is_unchanged(self) -> None:
    for goal in (
        "Send Alice the message hello.",
        "",
        "Original user instruction: not a prompt goal",
        "Application-independent task breakdown mentioned inline only.",
    ):
      with self.subTest(goal=goal):
        self.assertEqual(
            goal, task_breakdowns.original_goal_from_prompt_goal(goal)
        )

  def test_effective_mode_matches_runtime_default(self) -> None:
    path = self._write_payload({"breakdowns": []})
    with mock.patch.dict(
        os.environ, {task_breakdowns.ENV_BREAKDOWN_FILE: path}, clear=False
    ):
      os.environ.pop(task_breakdowns.ENV_BREAKDOWN_MODE, None)
      self.assertEqual("prepend", task_breakdowns.effective_mode())
      self.assertTrue(task_breakdowns.is_enabled())
      os.environ[task_breakdowns.ENV_BREAKDOWN_MODE] = "Original_First"
      self.assertEqual("original_first", task_breakdowns.effective_mode())
      os.environ[task_breakdowns.ENV_BREAKDOWN_MODE] = "off"
      self.assertEqual("off", task_breakdowns.effective_mode())
      self.assertFalse(task_breakdowns.is_enabled())
    with mock.patch.dict(
        os.environ,
        {
            task_breakdowns.ENV_BREAKDOWN_FILE: "",
            task_breakdowns.ENV_BREAKDOWN_MODE: "prepend",
        },
        clear=False,
    ):
      self.assertEqual("off", task_breakdowns.effective_mode())
      self.assertFalse(task_breakdowns.is_enabled())


if __name__ == "__main__":
  unittest.main()
