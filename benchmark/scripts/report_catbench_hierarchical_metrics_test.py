"""Tests for strict CATBench hierarchical metrics.

The fixtures are in-memory result rows using CATBench's real first-five app
identifiers.  They do not register task classes, install apps, or touch an
emulator.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import report_catbench_hierarchical_metrics as metrics  # noqa: E402


MODEL = "GUI-Owl-7B"


def _rows_for_category(
    category: str,
    app_outcomes: dict[str, dict[str, list[int]]],
) -> list[dict[str, object]]:
  rows: list[dict[str, object]] = []
  for app_id, templates in app_outcomes.items():
    for semantic_task_id, outcomes in templates.items():
      for instance_id, success in enumerate(outcomes):
        rows.append({
            "model": MODEL,
            "category": category,
            "app_id": app_id,
            "semantic_task_id": semantic_task_id,
            "instance_id": instance_id,
            "instance_seed": f"{category}:{semantic_task_id}:{instance_id}",
            "task_random_seed": 30,
            "is_successful": bool(success),
        })
  return rows


def _analyze(rows: list[dict[str, object]], **kwargs):
  return metrics.analyze_rows(
      rows,
      bootstrap_replicates=kwargs.pop("bootstrap_replicates", 60),
      bootstrap_seed=kwargs.pop("bootstrap_seed", 19),
      **kwargs,
  )


class HierarchicalMetricsTest(unittest.TestCase):

  def test_categories_receive_equal_weight_despite_different_app_counts(self):
    rows = _rows_for_category("sms", {
        "sms_simple_sms_messenger": {"SmsSend": [1]},
        "sms_fossify_messages": {"SmsSend": [0]},
    })
    rows += _rows_for_category("clock", {
        "clock_google_clock": {"ClockCreateAlarm": [1]},
        "clock_clock": {"ClockCreateAlarm": [1]},
        "clock_chrono": {"ClockCreateAlarm": [1]},
        "clock_fossify_clock": {"ClockCreateAlarm": [1]},
    })

    report = _analyze(rows)
    overall = report["models"][MODEL]["overall"]

    self.assertEqual(overall["new_sr"], 0.5)
    # A micro-average over the four new apps would be 0.75.
    self.assertNotEqual(overall["new_sr"], 0.75)
    self.assertEqual(overall["aw_sr"], 1.0)

  def test_new_group_is_arithmetic_mean_of_app_scores(self):
    rows = _rows_for_category("sms", {
        "sms_simple_sms_messenger": {"SmsSend": [1], "SmsReply": [1]},
        "sms_fossify_messages": {"SmsSend": [0], "SmsReply": [0]},
        "sms_quik_sms": {"SmsSend": [0], "SmsReply": [0]},
        "sms_google_messages": {"SmsSend": [1], "SmsReply": [1]},
    })

    report = _analyze(rows)
    category = report["models"][MODEL]["categories"]["sms"]
    new_apps = category["groups"]["new"]["apps"]
    expected = sum(category["apps"][app]["sr"] for app in new_apps) / 3

    self.assertAlmostEqual(category["groups"]["new"]["sr"], expected)
    self.assertAlmostEqual(category["groups"]["new"]["sr"], 1 / 3)

  def test_delta_sign_is_new_minus_aw(self):
    rows = _rows_for_category("files", {
        "files_material_files": {"FilesCreateFolder": [1]},
        "files_amaze": {"FilesCreateFolder": [0]},
    })

    report = _analyze(rows)
    category = report["models"][MODEL]["categories"]["files"]

    self.assertEqual(report["delta_definition"], "New - AW")
    self.assertEqual(category["delta_new_minus_aw"], -1.0)
    self.assertEqual(
        report["models"][MODEL]["overall"]["delta_new_minus_aw"], -1.0
    )

  def test_incomplete_app_common_semantic_roster_is_rejected(self):
    rows = _rows_for_category("contacts", {
        "contacts_google_contacts": {
            "ContactsAddContact": [1],
            "ContactsDeleteContact": [0],
        },
        "contacts_fossify_contacts": {
            "ContactsAddContact": [1],
            "ContactsDeleteContact": [0],
        },
    })
    rows = [
        row
        for row in rows
        if not (
            row["app_id"] == "contacts_fossify_contacts"
            and row["semantic_task_id"] == "ContactsDeleteContact"
        )
    ]

    with self.assertRaises(metrics.MatrixValidationError) as context:
      _analyze(rows)
    issue_types = {
        issue["type"]
        for issue in context.exception.report["validation"]["issues"]
    }
    self.assertIn("app_missing_semantic_instances", issue_types)

    diagnostic = _analyze(rows, allow_incomplete=True)
    self.assertFalse(diagnostic["strictly_valid"])

  def test_paired_hierarchical_bootstrap_is_deterministic(self):
    rows = _rows_for_category("maps", {
        "maps_osmand": {
            "MapsSearchPlace": [1, 1, 0],
            "MapsGetDirections": [1, 0, 1],
        },
        "maps_organic_maps": {
            "MapsSearchPlace": [1, 0, 0],
            "MapsGetDirections": [0, 0, 1],
        },
        "maps_comaps": {
            "MapsSearchPlace": [0, 1, 0],
            "MapsGetDirections": [1, 0, 0],
        },
    })

    first = _analyze(rows, bootstrap_replicates=250, bootstrap_seed=90210)
    second = _analyze(rows, bootstrap_replicates=250, bootstrap_seed=90210)

    first_ci = first["models"][MODEL]["overall"]["bootstrap_ci"]
    second_ci = second["models"][MODEL]["overall"]["bootstrap_ci"]
    self.assertEqual(first_ci, second_ci)
    self.assertLessEqual(
        first_ci["delta_new_minus_aw"][0],
        first_ci["delta_new_minus_aw"][1],
    )

  def test_bootstrap_pairs_the_same_semantic_instances_across_apps(self):
    outcomes = {
        "MapsSearchPlace": [1, 0, 1, 0],
        "MapsGetDirections": [0, 1, 0, 1],
    }
    rows = _rows_for_category("maps", {
        "maps_osmand": outcomes,
        "maps_organic_maps": outcomes,
    })

    report = _analyze(rows, bootstrap_replicates=250, bootstrap_seed=73)
    delta_ci = report["models"][MODEL]["overall"]["bootstrap_ci"][
        "delta_new_minus_aw"
    ]

    # Independent per-app instance resampling would generally yield a
    # non-zero interval even though every paired outcome is identical.
    self.assertEqual(delta_ci, [0.0, 0.0])


if __name__ == "__main__":
  unittest.main()
