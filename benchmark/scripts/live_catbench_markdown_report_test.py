"""Focused contract tests for the live CATBench report taxonomy."""

from __future__ import annotations

from absl.testing import absltest

import live_catbench_markdown_report as report


class LiveCatbenchMarkdownReportTaxonomyTest(absltest.TestCase):

  def test_sms_table_matches_frozen_send_to_contact_roster(self):
    self.assertIn("SendToContact", report.PROVIDED_TABLE_TASKS["sms"])
    self.assertNotIn("SendClipboard", report.PROVIDED_TABLE_TASKS["sms"])

  def test_send_to_contact_uses_sms_provider_validation(self):
    self.assertIn("SmsSendToContact", report.SMS_PROVIDER_TASKS)
    self.assertNotIn("SmsSendClipboard", report.SMS_PROVIDER_TASKS)
    self.assertEqual(
        report._validation_mode(
            "sms",
            "SmsSendToContactForSimpleSMSMessenger",
            "sms_simple_sms_messenger",
        ),
        "SmsProvider",
    )


if __name__ == "__main__":
  absltest.main()
