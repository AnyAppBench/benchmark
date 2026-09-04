"""Tests for strict frozen K=3 C1 app-level reporting."""

from __future__ import annotations

import json

from absl.testing import absltest

import build_catbench_frozen_schedule as schedule_builder
import report_c1_app_level_frozen as reporter


def _cohort() -> dict:
  return json.loads(
      schedule_builder.DEFAULT_COHORT.read_text(encoding="utf-8")
  )


def _all_failure_rows(cohort: dict) -> list[dict]:
  rows = []
  for model in cohort["models"]:
    for category, spec in cohort["categories"].items():
      for app_id in spec["app_ids"]:
        for semantic_task_id in spec["semantic_task_ids"]:
          for instance_id in range(3):
            rows.append({
                "model": model,
                "category": category,
                "app_id": app_id,
                "semantic_task_id": semantic_task_id,
                "instance_id": instance_id,
                "semantic_origin": spec["semantic_origins"][semantic_task_id],
                "success": 0,
            })
  return rows


class FrozenC1AppLevelReportTest(absltest.TestCase):

  def test_complete_k3_rows_use_frozen_origin_denominators(self):
    cohort = _cohort()
    rows = _all_failure_rows(cohort)
    rows[0]["success"] = 1

    aggregated = reporter.aggregate_rows(cohort, rows)

    sms_key = (
        cohort["models"][0],
        "sms",
        "sms_simple_sms_messenger",
    )
    self.assertEqual(
        aggregated[sms_key + (schedule_builder.SEMANTIC_ORIGIN_AW,)],
        (1, 15),
    )
    self.assertEqual(
        aggregated[sms_key + (schedule_builder.SEMANTIC_ORIGIN_NEW,)],
        (0, 15),
    )
    files_key = (cohort["models"][0], "files", "files_material_files")
    self.assertEqual(
        aggregated[files_key + (schedule_builder.SEMANTIC_ORIGIN_AW,)],
        (0, 9),
    )
    self.assertEqual(
        aggregated[files_key + (schedule_builder.SEMANTIC_ORIGIN_NEW,)],
        (0, 21),
    )

  def test_missing_instance_is_rejected_instead_of_shrinking_denominator(self):
    cohort = _cohort()
    rows = _all_failure_rows(cohort)
    rows.pop()
    with self.assertRaisesRegex(
        reporter.AppLevelReportError, "Incomplete K=3"
    ):
      reporter.aggregate_rows(cohort, rows)

  def test_duplicate_instance_is_rejected(self):
    cohort = _cohort()
    rows = _all_failure_rows(cohort)
    rows.append(dict(rows[0]))
    with self.assertRaisesRegex(
        reporter.AppLevelReportError, "Duplicate C1 result row"
    ):
      reporter.aggregate_rows(cohort, rows)

  def test_sms_send_to_contact_cannot_be_relabelled_as_aw(self):
    cohort = _cohort()
    self.assertEqual(
        cohort["categories"]["sms"]["semantic_origins"]["SmsSendToContact"],
        schedule_builder.SEMANTIC_ORIGIN_NEW,
    )
    rows = _all_failure_rows(cohort)
    target = next(
        row for row in rows
        if row["semantic_task_id"] == "SmsSendToContact"
    )
    target["semantic_origin"] = schedule_builder.SEMANTIC_ORIGIN_AW
    with self.assertRaisesRegex(
        reporter.AppLevelReportError, "Incomplete K=3"
    ):
      reporter.aggregate_rows(cohort, rows)


if __name__ == "__main__":
  absltest.main()
