"""Regression checks for the real five-category semantic-pairing audit."""

from __future__ import annotations

import unittest

import audit_cross_app_pairing as pairing_audit


class FiveCategoryPairingAuditTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    cls.enabled_report = pairing_audit.audit(
        pairing_audit.DEFAULT_CATEGORIES,
        n_task_combinations=3,
        task_random_seed=30,
        suite_family="android_world",
    )

  def test_enabled_real_roster_is_semantically_paired(self) -> None:
    report = self.enabled_report
    self.assertTrue(report["valid"])
    self.assertEqual(report["enabled_real_app_count"], 23)
    self.assertEqual(report["registered_task_class_count"], 230)
    self.assertEqual(report["semantic_template_count"], 50)
    self.assertEqual(report["scheduled_task_app_instances"], 690)
    self.assertEqual(report["semantic_instance_groups"], 150)
    self.assertEqual(report["parameter_mismatch_groups"], 0)
    self.assertEqual(report["neutral_goal_mismatch_groups"], 0)
    self.assertEqual(report["violations"], [])

  def test_strict_frozen_cohort_is_complete_and_paired(self) -> None:
    cohort = pairing_audit._strict_json(
        pairing_audit.DEFAULT_COHORT_MANIFEST.resolve()
    )
    report = pairing_audit.audit(
        pairing_audit.DEFAULT_CATEGORIES,
        n_task_combinations=3,
        task_random_seed=30,
        suite_family="android_world",
        cohort=cohort,
    )
    self.assertTrue(report["valid"])
    self.assertEqual(report["violations"], [])
    self.assertEqual(report["enabled_real_app_count"], 23)
    self.assertEqual(report["registered_task_class_count"], 230)
    self.assertEqual(report["scheduled_task_app_instances"], 690)
    self.assertEqual(report["cohort_violations"], [])
    self.assertEqual(
        report["cohort_release_id"], "catbench_acl_revision_5cat_v1"
    )


if __name__ == "__main__":
  unittest.main()
