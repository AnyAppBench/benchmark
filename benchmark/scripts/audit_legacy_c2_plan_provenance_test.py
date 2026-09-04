"""Focused checks for the legacy C2 provenance audit."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import audit_legacy_c2_plan_provenance as audit_module


class StrictJsonTest(unittest.TestCase):

  def test_duplicate_keys_and_non_finite_values_are_rejected(self) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
      duplicate = Path(temporary_directory) / "duplicate.json"
      duplicate.write_text('{"x": 1, "x": 2}', encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
        audit_module._strict_json(duplicate)

      non_finite = Path(temporary_directory) / "nan.json"
      non_finite.write_text('{"x": NaN}', encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "Non-finite JSON constant"):
        audit_module._strict_json(non_finite)


@unittest.skipUnless(
    audit_module.DEFAULT_GPT_FILE.is_file()
    and audit_module.DEFAULT_GEMINI_FILE.is_file(),
    "Legacy forensic artifacts are not mounted on this host.",
)
class MountedArtifactRegressionTest(unittest.TestCase):

  def test_exact_saved_artifacts_and_comparison(self) -> None:
    report = audit_module.audit(
        audit_module.DEFAULT_GPT_FILE,
        audit_module.DEFAULT_GEMINI_FILE,
        audit_module.DEFAULT_COHORT,
    )
    self.assertEqual(
        report["gpt54"]["sha256"],
        "79090cac3c550ea0b055bcc6c2822429e10ad234edc117cc3f6f2ea6878ee179",
    )
    self.assertEqual(
        report["gemini31"]["sha256"],
        "4216d2a4faaa851abfcde79782b3c26a9c01a38ec6a8dd1ff35ad1d381651650",
    )
    self.assertEqual(report["gpt54"]["entry_count"], 250)
    self.assertEqual(report["gemini31"]["entry_count"], 242)
    self.assertEqual(
        report["gpt54"]["app_specific_raw_goal_count"], 250
    )
    self.assertEqual(report["gpt54"]["current_registered_task_entry_count"], 249)
    self.assertEqual(
        [
            row["task_template"]
            for row in report["gpt54"]["non_current_registry_entries"]
        ],
        ["SmsArchiveConversationForQUIKSMS"],
    )
    self.assertTrue(
        report["gpt54"]["grid"]["complete_k1_cartesian_grid"]
    )
    self.assertEqual(
        report["artifact_comparison"]["shared_exact_legacy_keys"], 242
    )
    self.assertEqual(report["artifact_comparison"]["gpt_only_count"], 8)
    self.assertEqual(report["artifact_comparison"]["gemini_only_count"], 0)
    self.assertEqual(
        report["frozen_cohort_comparison"]["legacy_only_app_ids"],
        ["maps_google_maps", "maps_maps_me"],
    )
    self.assertEqual(
        report["frozen_cohort_comparison"][
            "legacy_only_semantic_task_ids"
        ],
        ["SmsArchiveConversation", "SmsSendClipboard"],
    )


if __name__ == "__main__":
  unittest.main()
