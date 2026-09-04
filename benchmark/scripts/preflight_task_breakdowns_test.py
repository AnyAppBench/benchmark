"""Focused tests for strict C2 plan-content preflight."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, BENCHMARK_ROOT):
  if str(import_root) not in sys.path:
    sys.path.insert(0, str(import_root))

import preflight_task_breakdowns as preflight
import task_breakdowns


def _entry(step: str) -> dict:
  breakdown = {"steps": [step], "notes": []}
  text = task_breakdowns.format_breakdown_text({"breakdown": breakdown})
  return {
      "breakdown": breakdown,
      "breakdown_text": text,
      "plan_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
      "validation_warnings": [],
  }


class PlanContentPreflightTest(unittest.TestCase):

  def test_goal_geographic_coordinates_are_preserved_task_values(self) -> None:
    errors = preflight._plan_content_errors(  # pylint: disable=protected-access
        _entry("Enter the geographic coordinates 47.23976, 9.5262837."),
        key="maps-plan",
        app_display_names=(),
        semantic_goal=(
            "Add a favorite marker for 47.23976, 9.5262837 in "
            "[TARGET_APP]."
        ),
    )

    self.assertFalse(any("coordinate_word" in error for error in errors))

  def test_coordinate_word_without_goal_value_remains_forbidden(self) -> None:
    errors = preflight._plan_content_errors(  # pylint: disable=protected-access
        _entry("Use screen coordinates to select the target."),
        key="unsafe-plan",
        app_display_names=(),
        semantic_goal="Add a favorite marker for Madrid in [TARGET_APP].",
    )

    self.assertTrue(any("coordinate_word" in error for error in errors))

  def test_goal_coordinate_allows_whitespace_and_terminal_punctuation(
      self,
  ) -> None:
    errors = preflight._plan_content_errors(  # pylint: disable=protected-access
        _entry(
            "Enter the geographic coordinates:\n"
            "(47.23976 ,\t9.5262837)."
        ),
        key="maps-plan",
        app_display_names=(),
        semantic_goal="Mark 47.23976, 9.5262837. in [TARGET_APP].",
    )

    self.assertFalse(any("coordinate_pair" in error for error in errors))
    self.assertFalse(any("coordinate_word" in error for error in errors))

  def test_goal_coordinate_does_not_hide_other_coordinate_leakage(self) -> None:
    breakdown = {
        "steps": [
            "Enter coordinates 47.23976, 9.5262837.",
            "Use screen coordinates 120.5, 300.25 to select it.",
        ],
        "notes": [],
    }
    text = task_breakdowns.format_breakdown_text({"breakdown": breakdown})
    entry = {
        "breakdown": breakdown,
        "breakdown_text": text,
        "plan_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "validation_warnings": [],
    }
    errors = preflight._plan_content_errors(  # pylint: disable=protected-access
        entry,
        key="unsafe-plan",
        app_display_names=(),
        semantic_goal="Mark 47.23976, 9.5262837 in [TARGET_APP].",
    )

    self.assertTrue(any("coordinate_pair" in error for error in errors))
    self.assertTrue(any("coordinate_word" in error for error in errors))

  def test_goal_coordinate_repurposed_as_pixel_location_is_forbidden(
      self,
  ) -> None:
    errors = preflight._plan_content_errors(  # pylint: disable=protected-access
        _entry("Tap position 47.23976, 9.5262837 as a pixel location."),
        key="unsafe-plan",
        app_display_names=(),
        semantic_goal="Mark 47.23976, 9.5262837 in [TARGET_APP].",
    )

    self.assertTrue(any("coordinate_pair" in error for error in errors))

  def test_invented_decimal_pair_is_forbidden(self) -> None:
    errors = preflight._plan_content_errors(  # pylint: disable=protected-access
        _entry("Use the location value 47.20001, 9.50002."),
        key="unsafe-plan",
        app_display_names=(),
        semantic_goal="Mark 47.23976, 9.5262837 in [TARGET_APP].",
    )

    self.assertTrue(any("coordinate_pair" in error for error in errors))

  def test_unattached_coordinate_word_remains_forbidden(self) -> None:
    breakdown = {
        "steps": [
            "Enter coordinates 47.23976, 9.5262837.",
            "Ensure the coordinates use the required format.",
        ],
        "notes": [],
    }
    text = task_breakdowns.format_breakdown_text({"breakdown": breakdown})
    entry = {
        "breakdown": breakdown,
        "breakdown_text": text,
        "plan_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "validation_warnings": [],
    }
    errors = preflight._plan_content_errors(  # pylint: disable=protected-access
        entry,
        key="unsafe-plan",
        app_display_names=(),
        semantic_goal="Mark 47.23976, 9.5262837 in [TARGET_APP].",
    )

    self.assertTrue(any("coordinate_word" in error for error in errors))

  def test_geographic_references_are_checked_per_occurrence(self) -> None:
    breakdown = {
        "steps": ["Enter coordinates 47.23976, 9.5262837."],
        "notes": [
            "The location marker uses coordinates.",
            "The coordinates must be entered precisely as provided.",
        ],
    }
    text = task_breakdowns.format_breakdown_text({"breakdown": breakdown})
    entry = {
        "breakdown": breakdown,
        "breakdown_text": text,
        "plan_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "validation_warnings": [],
    }
    errors = preflight._plan_content_errors(  # pylint: disable=protected-access
        entry,
        key="maps-plan",
        app_display_names=(),
        semantic_goal="Mark 47.23976, 9.5262837 in [TARGET_APP].",
    )

    self.assertFalse(any("coordinate_pair" in error for error in errors))
    self.assertFalse(any("coordinate_word" in error for error in errors))

  def test_geographic_reference_requires_preserved_goal_pair(self) -> None:
    errors = preflight._plan_content_errors(  # pylint: disable=protected-access
        _entry("Create the location marker using coordinates."),
        key="unsafe-plan",
        app_display_names=(),
        semantic_goal="Mark 47.23976, 9.5262837 in [TARGET_APP].",
    )

    self.assertTrue(any("coordinate_word" in error for error in errors))

  def test_invalid_goal_ranges_do_not_create_coordinate_exemption(self) -> None:
    errors = preflight._plan_content_errors(  # pylint: disable=protected-access
        _entry("Enter coordinates 91.5, 181.5."),
        key="unsafe-plan",
        app_display_names=(),
        semantic_goal="Mark 91.5, 181.5 in [TARGET_APP].",
    )

    self.assertTrue(any("coordinate_pair" in error for error in errors))

  def test_multiple_goal_coordinates_are_exempted_independently(self) -> None:
    breakdown = {
        "steps": [
            "Use coordinates 47.23976, 9.5262837.",
            "Then use coordinates -33.8688, 151.2093.",
        ],
        "notes": [],
    }
    text = task_breakdowns.format_breakdown_text({"breakdown": breakdown})
    entry = {
        "breakdown": breakdown,
        "breakdown_text": text,
        "plan_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "validation_warnings": [],
    }
    errors = preflight._plan_content_errors(  # pylint: disable=protected-access
        entry,
        key="maps-plan",
        app_display_names=(),
        semantic_goal=(
            "Compare 47.23976, 9.5262837 with -33.8688, 151.2093."
        ),
    )

    self.assertFalse(any("coordinate_pair" in error for error in errors))
    self.assertFalse(any("coordinate_word" in error for error in errors))

  def test_decimal_x_y_coordinate_is_forbidden(self) -> None:
    errors = preflight._plan_content_errors(  # pylint: disable=protected-access
        _entry("Use x = 120.5 and y:\t300.25."),
        key="unsafe-plan",
        app_display_names=(),
    )

    self.assertTrue(any("x_y_coordinate" in error for error in errors))


if __name__ == "__main__":
  unittest.main()
