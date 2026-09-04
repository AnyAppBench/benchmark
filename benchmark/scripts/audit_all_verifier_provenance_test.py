"""Deterministic checks for the frozen verifier provenance inventory."""

from __future__ import annotations

from absl.testing import absltest

import audit_all_verifier_provenance as audit


class AuditAllVerifierProvenanceTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.report = audit.build_audit(
        audit.BENCHMARK_ROOT / "configs/catbench_5cat_primary_cohort.json"
    )

  def test_exact_frozen_roster_is_resolved_once(self):
    rows = self.report["records"]
    self.assertLen(rows, 230)
    self.assertLen({row["task_class"] for row in rows}, 230)
    self.assertEqual(
        {"sms": 40, "files": 50, "maps": 30,
         "contacts": 50, "clock": 60},
        {
            category: sum(row["category"] == category for row in rows)
            for category in {row["category"] for row in rows}
        },
    )

  def test_semantic_and_executable_lineage_are_separate(self):
    summary = self.report["summary"]
    self.assertEqual(16, summary["androidworld_intent_adapted_semantic_templates"])
    self.assertEqual(34, summary["catbench_new_semantic_templates"])
    self.assertEqual(0, summary["exact_androidworld_task_classes_reused"])
    self.assertEqual(0, summary["exact_androidworld_verifier_methods_reused"])
    for row in self.report["records"]:
      self.assertFalse(row["exact_androidworld_task_class_reused"])
      self.assertFalse(row["exact_androidworld_verifier_method_reused"])
      self.assertIn("app_generalization_generated", row["verifier"]["source_file"])

  def test_aw_baseline_apps_still_use_generated_classes(self):
    rows = [
        row for row in self.report["records"]
        if row["is_androidworld_baseline_app"]
    ]
    self.assertLen(rows, 50)
    self.assertTrue(all(not row["exact_androidworld_task_class_reused"] for row in rows))

  def test_material_files_extract_owner_and_clock_metadata_are_consistent(self):
    material_extract = next(
        row for row in self.report["records"]
        if row["task_class"] == "FilesExtractArchiveForMaterialFiles"
    )
    self.assertEqual(
        "_FilesExtractArchiveBase",
        material_extract["verifier"]["owner_class"],
    )
    discrepancies = [
        row for row in self.report["records"]
        if row["validation_mode_metadata_discrepancy"]
    ]
    self.assertEmpty(discrepancies)
    clock_you_storage_rows = [
        row for row in self.report["records"]
        if row["app_id"] == "clock_clockyou"
        and row["semantic_task_id"] in {
            "ClockCreateAlarm",
            "ClockEditAlarm",
            "ClockEnableAlarm",
            "ClockDeleteAlarm",
            "ClockAddWorldClock",
        }
    ]
    self.assertLen(clock_you_storage_rows, 5)
    for row in clock_you_storage_rows:
      self.assertEqual(
          "Clock You Room SQLite durable state",
          row["declared_validation_mode"],
      )
      self.assertEqual(
          row["declared_validation_mode"], row["audited_validation_mode"]
      )

  def test_no_row_is_promoted_from_narrow_fixture_to_g3(self):
    self.assertFalse(any(
        row["conformance"]["g3_qualified"]
        for row in self.report["records"]
    ))

  def test_right_contact_uses_declared_provider_modes(self):
    rows = [
        row for row in self.report["records"]
        if row["app_id"] == "contacts_right_contact"
    ]
    self.assertLen(rows, 10)
    transition_latches = {
        row["semantic_task_id"] for row in rows
        if row["audited_validation_mode"]
        == "ContactsProvider durable state + exact-target transition latch"
    }
    self.assertEqual(
        {"ContactsRemoveFavoriteContact", "ContactsDeleteContact"},
        transition_latches,
    )
    self.assertFalse(any(
        "opaque" in row["audited_validation_mode"].casefold()
        or "fail-closed" in row["audited_validation_mode"].casefold()
        for row in rows
    ))
    self.assertTrue(all(
        row["declared_validation_mode"] == row["audited_validation_mode"]
        for row in rows
    ))

  def test_runtime_universes_are_not_conflated(self):
    universes = self.report["universe_distinctions"]
    self.assertEqual(
        653,
        universes["all_registered_generated_classes"]["class_count"],
    )
    profiles = universes["current_domain_profiles"]
    # All ten categories are wired, so the live profiles now cover the same
    # 520 task-application pairs as the submitted grid asserted below.
    self.assertEqual(520, profiles["task_app_reference_count"])
    self.assertEqual([], profiles["missing_submitted_categories"])
    self.assertLen(
        profiles["domains_detail"]["notes"]["apps_with_zero_implemented_tasks"],
        1,
    )
    self.assertLen(
        profiles["domains_detail"]["todo"]["apps_with_zero_implemented_tasks"],
        2,
    )
    submitted = universes["submitted_legacy_grid"]
    self.assertEqual(520, submitted["reported_task_app_combination_count"])
    self.assertFalse(submitted["same_as_frozen_replacement_cohort"])


if __name__ == "__main__":
  absltest.main()
