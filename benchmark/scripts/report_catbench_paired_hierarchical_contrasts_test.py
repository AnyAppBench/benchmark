"""Tests for paired CATBench C1/C2 hierarchical contrasts.

Fixtures are in-memory artifact records built with real CATBench model, app,
package, semantic-task, and generated task-class identifiers. They are unit
fixtures only: no benchmark result file is fabricated or written.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from pair_baseline_and_breakdown import Harvest  # noqa: E402
import report_catbench_paired_hierarchical_contrasts as contrasts  # noqa: E402


MODEL = "GUI-Owl-7B"
RELEASE = "catbench_acl_revision_5cat_v1"
APP_METADATA = {
    "sms_simple_sms_messenger": (
        "com.simplemobiletools.smsmessenger", "SimpleSMSMessenger"
    ),
    "sms_fossify_messages": (
        "org.fossify.messages", "FossifyMessages"
    ),
    "clock_google_clock": (
        "com.google.android.deskclock", "GoogleClock"
    ),
    "clock_clock": ("com.best.deskclock", "Clock"),
    "clock_chrono": ("com.vicolo.chrono", "Chrono"),
    "contacts_google_contacts": (
        "com.google.android.contacts", "GoogleContacts"
    ),
    "contacts_fossify_contacts": (
        "org.fossify.contacts", "FossifyContacts"
    ),
    "maps_osmand": ("net.osmand.plus", "OsmAnd"),
    "maps_organic_maps": ("app.organicmaps", "OrganicMaps"),
}


def _sha(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row(
    condition: str,
    category: str,
    app_id: str,
    semantic_task_id: str,
    instance_id: int,
    success: int,
) -> dict[str, object]:
  package_name, suffix = APP_METADATA[app_id]
  task_template = f"{semantic_task_id}For{suffix}"
  semantic_instance = f"{category}|{semantic_task_id}|{instance_id}"
  row: dict[str, object] = {
      "model": MODEL,
      "category": category,
      "app_id": app_id,
      "task_template": task_template,
      "semantic_task_id": semantic_task_id,
      "instance_id": instance_id,
      "goal_sha256": _sha(f"{semantic_instance}|{app_id}"),
      "semantic_goal_sha256": _sha(semantic_instance),
      "semantic_parameter_sha256": _sha(f"params|{semantic_instance}"),
      "instance_seed": _sha(f"seed|{semantic_instance}"),
      "task_random_seed": 30,
      "package_name": package_name,
      "app_version": "frozen-version",
      "app_version_code": "1",
      "apk_sha256": _sha(f"apk|{app_id}"),
      "code_revision": _sha("catbench-frozen-code")[:40],
      "release_id": RELEASE,
      "model_revision": "GUI-Owl-7B-frozen",
      "runner_config_sha256": _sha("frozen-runner-config"),
      "model_config_sha256": _sha("frozen-model-config"),
      "condition": condition,
      "condition_config_valid": True,
      "is_successful": bool(success),
      "pkl_path": f"/{condition}/{category}/{app_id}/{task_template}.pkl.gz",
  }
  if condition in {"c2_g", "c2_o"}:
    row.update({
        "plan_key": f"{condition}|{semantic_instance}",
        "plan_sha256": _sha(f"{condition}|plan|{semantic_instance}"),
    })
  return row


def _harvest(rows: list[dict[str, object]]) -> Harvest:
  indexed = {}
  roster = set()
  for row in rows:
    key = (
        str(row["model"]),
        str(row["category"]),
        str(row["app_id"]),
        str(row["task_template"]),
        int(row["instance_id"]),
    )
    indexed[key] = row
    roster.add((key[0], key[1], key[2]))
  return Harvest(
      rows=indexed,
      invalid_records=[],
      invalid_slots={},
      roster=roster,
  )


def _condition_harvests(
    outcomes: dict[str, dict[tuple[str, str, str, int], int]],
) -> dict[str, list[Harvest]]:
  result = {}
  for condition, condition_outcomes in outcomes.items():
    rows = [
        _row(condition, category, app_id, task, instance, success)
        for (category, app_id, task, instance), success
        in condition_outcomes.items()
    ]
    result[condition] = [_harvest(rows)]
  return result


def _analyze(harvests, **kwargs):
  return contrasts.analyze_harvests(
      harvests,
      bootstrap_replicates=kwargs.pop("bootstrap_replicates", 80),
      bootstrap_seed=kwargs.pop("bootstrap_seed", 41),
      **kwargs,
  )


class PairedHierarchicalContrastsTest(unittest.TestCase):

  def test_default_draw_count_is_ten_thousand(self):
    default = inspect.signature(contrasts.analyze_harvests).parameters[
        "bootstrap_replicates"
    ].default
    self.assertEqual(default, 10000)

  def test_macro_contrast_weights_categories_not_raw_app_cells(self):
    slots = [
        ("sms", "sms_simple_sms_messenger", "SmsSend", 0),
        ("sms", "sms_fossify_messages", "SmsSend", 0),
        ("clock", "clock_google_clock", "ClockCreateAlarm", 0),
        ("clock", "clock_clock", "ClockCreateAlarm", 0),
        ("clock", "clock_chrono", "ClockCreateAlarm", 0),
    ]
    c1 = {slot: 0 for slot in slots}
    c2_g = {slot: int(slot[0] == "sms") for slot in slots}
    c2_o = dict(c1)

    report = _analyze(_condition_harvests({
        "c1": c1,
        "c2_g": c2_g,
        "c2_o": c2_o,
    }))
    generated = report["contrasts"]["c1_vs_c2_g"]["models"][MODEL]
    human_reference = report["contrasts"]["c1_vs_c2_o"]["models"][MODEL]

    self.assertTrue(report["strictly_valid"])
    self.assertEqual(generated["overall"]["delta_c2_minus_c1"], 0.5)
    # A five-cell micro-average would assign extra weight to Clock: 2/5.
    self.assertNotEqual(generated["overall"]["delta_c2_minus_c1"], 2 / 5)
    self.assertEqual(
        human_reference["overall"]["delta_c2_minus_c1"], 0.0
    )
    self.assertEqual(
        report["contrasts"]["c1_vs_c2_g"]["raw"]["c1_fail_c2_pass"],
        2,
    )

  def test_bootstrap_is_deterministic_and_keeps_condition_pairs(self):
    slots = [
        ("maps", "maps_osmand", "MapsSearchPlace", instance)
        for instance in range(3)
    ] + [
        ("maps", "maps_organic_maps", "MapsSearchPlace", instance)
        for instance in range(3)
    ]
    patterned = {slot: slot[3] % 2 for slot in slots}
    harvests = _condition_harvests({
        "c1": patterned,
        "c2_g": patterned,
        "c2_o": patterned,
    })

    first = _analyze(harvests, bootstrap_replicates=250, bootstrap_seed=73)
    second = _analyze(harvests, bootstrap_replicates=250, bootstrap_seed=73)
    first_overall = first["contrasts"]["c1_vs_c2_g"]["models"][MODEL][
        "overall"
    ]
    second_overall = second["contrasts"]["c1_vs_c2_g"]["models"][MODEL][
        "overall"
    ]

    self.assertEqual(
        first_overall["bootstrap_ci"], second_overall["bootstrap_ci"]
    )
    self.assertEqual(
        first_overall["bootstrap_ci"]["delta_c2_minus_c1"], [0.0, 0.0]
    )
    self.assertEqual(first_overall["draw_seed"], second_overall["draw_seed"])

  def test_provenance_mismatch_is_not_treated_as_a_pair(self):
    slots = [
        ("sms", "sms_simple_sms_messenger", "SmsSend", 0),
        ("sms", "sms_fossify_messages", "SmsSend", 0),
    ]
    outcomes = {slot: 0 for slot in slots}
    harvests = _condition_harvests({
        "c1": outcomes,
        "c2_g": outcomes,
        "c2_o": outcomes,
    })
    changed = next(iter(harvests["c2_g"][0].rows.values()))
    changed["instance_seed"] = _sha("different-semantic-instance")

    with self.assertRaises(contrasts.PairedContrastValidationError):
      _analyze(harvests)
    diagnostic = _analyze(harvests, allow_incomplete=True)
    pairing = diagnostic["contrasts"]["c1_vs_c2_g"]

    self.assertFalse(diagnostic["strictly_valid"])
    self.assertEqual(pairing["exact_valid_pairs"], 1)
    self.assertEqual(
        len(pairing["pairing_audit"]["provenance_mismatches"]), 1
    )

  def test_raw_audit_counts_invalid_missing_and_replacement_attempts(self):
    slots = [
        ("contacts", "contacts_google_contacts", "ContactsAddContact", 0),
        ("contacts", "contacts_fossify_contacts", "ContactsAddContact", 0),
    ]
    complete = {slot: 0 for slot in slots}
    c2_o_valid = {slots[0]: 0}
    harvests = _condition_harvests({
        "c1": complete,
        "c2_g": complete,
        "c2_o": c2_o_valid,
    })
    missing_row = _row("c2_o", *slots[1], success=0)
    legacy_slot = (
        str(missing_row["model"]),
        str(missing_row["category"]),
        str(missing_row["app_id"]),
        str(missing_row["task_template"]),
        int(missing_row["instance_id"]),
    )
    invalid_records = [
        {
            "slot": list(legacy_slot),
            "pkl_path": f"/c2_o/retry_{attempt}.pkl.gz",
            "episode_index": 0,
            "issues": ["infrastructure_exception"],
        }
        for attempt in range(2)
    ]
    harvests["c2_o"][0].invalid_records = invalid_records
    harvests["c2_o"][0].invalid_slots = {
        legacy_slot: invalid_records
    }
    expected = {
        (MODEL, category, app_id, task, instance)
        for category, app_id, task, instance in slots
    }

    report = _analyze(
        harvests, expected_slots=expected, allow_incomplete=True
    )
    audit = report["raw_condition_audits"]["c2_o"]

    self.assertEqual(audit["scheduled_cells"], 2)
    self.assertEqual(audit["selected_valid_outcome_cells"], 1)
    self.assertEqual(audit["infrastructure_or_harvest_invalid_attempts"], 2)
    self.assertEqual(audit["invalid_unique_scheduled_cells"], 1)
    self.assertEqual(audit["missing_valid_outcome_cells"], 1)
    self.assertEqual(audit["missing_terminal_artifact_cells"], 0)
    self.assertEqual(audit["replacement_attempts"], 1)

  def test_frozen_cartesian_cohort_expands_real_identifiers(self):
    cohort = {
        "models": [MODEL],
        "n_task_combinations": 3,
        "categories": {
            "sms": {
                "app_ids": [
                    "sms_simple_sms_messenger",
                    "sms_fossify_messages",
                ],
                "semantic_task_ids": ["SmsSend"],
            }
        },
    }

    expected = contrasts.expected_slots_from_cohort(cohort)

    self.assertEqual(len(expected), 6)
    self.assertIn(
        (MODEL, "sms", "sms_fossify_messages", "SmsSend", 2), expected
    )


if __name__ == "__main__":
  unittest.main()
