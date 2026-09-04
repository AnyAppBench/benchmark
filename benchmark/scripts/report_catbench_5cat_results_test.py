"""Fail-closed tests for the legacy five-category reporter."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from absl.testing import absltest

import report_catbench_5cat_results as report


class ReportCatbenchFiveCategoryResultsTest(absltest.TestCase):

  def test_typed_infrastructure_and_unknown_outcomes_are_skipped(self):
    self.assertTrue(report._is_skipped({  # pylint: disable=protected-access
        "catbench_episode_status": "invalid_infrastructure",
        "is_successful": 0.0,
    }))
    self.assertTrue(report._is_skipped({  # pylint: disable=protected-access
        "catbench_episode_status": "future_unknown_status",
        "is_successful": 0.0,
    }))
    self.assertFalse(report._is_skipped({  # pylint: disable=protected-access
        "catbench_episode_status": "valid_failure",
        "is_successful": 0.0,
    }))

  def test_rate_never_turns_typed_infrastructure_into_failure(self):
    episodes = [
        {
            "catbench_episode_status": "invalid_infrastructure",
            "is_successful": 0.0,
        },
        {"catbench_episode_status": "valid_success", "is_successful": 1.0},
    ]
    self.assertEqual(100.0, report._rate(episodes))  # pylint: disable=protected-access

  def test_analysis_eligible_manifest_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / "manifest.json"
      path.write_text(json.dumps({
          "analysis_eligible": True,
          "jobs": [],
      }), encoding="utf-8")
      with self.assertRaisesRegex(
          report.LegacyReportError, "report_c1_app_level_frozen"
      ):
        report._jobs_from_manifest(path)  # pylint: disable=protected-access

  def test_job_level_analysis_eligibility_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / "manifest.json"
      path.write_text(json.dumps({
          "jobs": [{"analysis_eligible": True}],
      }), encoding="utf-8")
      with self.assertRaises(report.LegacyReportError):
        report._jobs_from_manifest(path)  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
