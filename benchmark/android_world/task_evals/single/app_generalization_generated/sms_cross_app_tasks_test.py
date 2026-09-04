from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from android_env.proto import adb_pb2

from android_world.env import adb_utils
from android_world.task_evals.single.app_generalization_generated import (
    sms_cross_app_tasks as sms_tasks,
)


def _adb_response(output: str) -> adb_pb2.AdbResponse:
  response = adb_pb2.AdbResponse()
  response.generic.output = output.encode("utf-8")
  return response


def _row(number: str, body: str) -> dict[str, str]:
  return {"address": number, "body": body}


class _FakeEnv:

  def __init__(self, ui_texts: tuple[str, ...] = ()):
    self.controller = mock.Mock()
    self._ui_texts = ui_texts

  def get_state(self):
    return SimpleNamespace(
        ui_elements=[
            SimpleNamespace(text=text, content_description="")
            for text in self._ui_texts
        ]
    )


def _score_with_rows(
    task_cls,
    params,
    *,
    sent=(),
    draft=(),
    all_rows=(),
):
  task = task_cls(params)
  task.initialized = True
  rows_by_uri = {
      "content://sms/sent": list(sent),
      "content://sms/draft": list(draft),
      "content://sms": list(all_rows),
  }
  with mock.patch.object(
      sms_tasks,
      "_query_sms_rows",
      side_effect=lambda unused_env, uri: rows_by_uri[uri],
  ):
    return task.is_successful(_FakeEnv())


class SmsCrossAppTasksTest(absltest.TestCase):

  def test_parse_sms_rows_preserves_body_commas(self):
    rows = sms_tasks._parse_sms_rows(
        "Row: 0 address=+1 202-555-0101, body=Hello, friend, read=1\n"
        "Row: 1 address=303-555-0102, body=Plain text, read=1"
    )

    self.assertEqual(rows[0]["address"], "+1 202-555-0101")
    self.assertEqual(rows[0]["body"], "Hello, friend")
    self.assertEqual(rows[1]["body"], "Plain text")

  def test_sent_contains_requires_expected_body(self):
    env = _FakeEnv()
    with mock.patch.object(
        adb_utils,
        "issue_generic_request",
        return_value=_adb_response(
            "202-555-0101\x1fWrong message"
        ),
    ):
      self.assertFalse(
          sms_tasks._sent_contains(env, "202-555-0101", "Correct message")
      )

  def test_sent_contains_matches_number_and_body(self):
    env = _FakeEnv()
    with mock.patch.object(
        adb_utils,
        "issue_generic_request",
        return_value=_adb_response(
            "+1 202-555-0101\x1fCorrect message"
        ),
    ):
      self.assertTrue(
          sms_tasks._sent_contains(env, "202-555-0101", "Correct message")
      )

  def test_number_match_allows_country_code_but_rejects_arbitrary_suffix(self):
    self.assertTrue(
        sms_tasks._row_matches(
            _row("+1 202-555-0101", "Correct message"),
            "202-555-0101",
            "Correct message",
        )
    )
    self.assertFalse(
        sms_tasks._row_matches(
            _row("991-202-555-0101", "Correct message"),
            "202-555-0101",
            "Correct message",
        )
    )

  def test_body_match_is_exact_in_case_and_spacing(self):
    expected = "On my way."
    self.assertTrue(
        sms_tasks._row_matches(
            _row("202-555-0101", expected), "202-555-0101", expected
        )
    )
    self.assertFalse(
        sms_tasks._row_matches(
            _row("202-555-0101", "ON MY WAY."),
            "202-555-0101",
            expected,
        )
    )
    self.assertFalse(
        sms_tasks._row_matches(
            _row("202-555-0101", "On  my way."),
            "202-555-0101",
            expected,
        )
    )

  def test_send_positive_noop_wrong_recipient_and_partial_body(self):
    params = {"number": "202-555-0101", "message": "On my way."}
    task_cls = sms_tasks.SmsSendForSimpleSMSMessenger
    cases = (
        ((), 0.0),
        ((_row("202-555-0102", "On my way."),), 0.0),
        ((_row("202-555-0101", "On my"),), 0.0),
        ((_row("+1 202-555-0101", "On my way."),), 1.0),
    )
    for sent, expected in cases:
      with self.subTest(sent=sent):
        self.assertEqual(
            _score_with_rows(task_cls, params, sent=sent), expected
        )

  def test_reply_requires_target_send_and_both_seed_threads(self):
    params = {
        "number": "202-555-0101",
        "decoy_number": "303-555-0102",
        "message": "On my way.",
    }
    target_seed = _row(params["number"], "Older message you must reply to")
    decoy_seed = _row(params["decoy_number"], "Decoy newest thread")
    target_send = _row(params["number"], params["message"])
    wrong_send = _row(params["decoy_number"], params["message"])

    self.assertEqual(
        _score_with_rows(
            sms_tasks.SmsReplyForSimpleSMSMessenger,
            params,
            sent=(target_send,),
            all_rows=(target_seed, decoy_seed, target_send),
        ),
        1.0,
    )
    for sent, all_rows in (
        ((), (target_seed, decoy_seed)),
        ((wrong_send,), (target_seed, decoy_seed, wrong_send)),
        ((target_send,), (decoy_seed, target_send)),
    ):
      with self.subTest(sent=sent, all_rows=all_rows):
        self.assertEqual(
            _score_with_rows(
                sms_tasks.SmsReplyForSimpleSMSMessenger,
                params,
                sent=sent,
                all_rows=all_rows,
            ),
            0.0,
        )

  def test_reply_seed_has_strictly_newer_decoy_timestamp(self):
    params = {
        "number": "202-555-0101",
        "decoy_number": "303-555-0102",
        "message": "On my way.",
    }
    task = sms_tasks.SmsReplyForSimpleSMSMessenger(params)
    with mock.patch.object(sms_tasks.time, "time", return_value=1234.567), \
         mock.patch.object(sms_tasks, "_seed_inbox_message") as seed:
      task._seed_state(_FakeEnv())

    self.assertEqual(seed.call_count, 2)
    older_call, newest_call = seed.call_args_list
    self.assertEqual(older_call.kwargs["date_ms"], 1233567)
    self.assertEqual(newest_call.kwargs["date_ms"], 1234567)

  def test_reply_most_recent_requires_seed_and_correct_reply(self):
    params = {
        "number": "202-555-0101",
        "message": "On my way.",
        "seed_message": "CATBench most recent message",
    }
    seed = _row(params["number"], params["seed_message"])
    reply = _row(params["number"], params["message"])
    task_cls = sms_tasks.SmsReplyMostRecentForSimpleSMSMessenger
    self.assertEqual(
        _score_with_rows(
            task_cls,
            params,
            sent=(reply,),
            all_rows=(seed, reply),
        ),
        1.0,
    )
    self.assertEqual(
        _score_with_rows(task_cls, params, sent=(reply,), all_rows=(reply,)),
        0.0,
    )
    self.assertEqual(
        _score_with_rows(task_cls, params, sent=(), all_rows=(seed,)), 0.0
    )

  def test_resend_requires_second_exact_sent_row(self):
    params = {"number": "202-555-0101", "message": "On my way."}
    exact = _row(params["number"], params["message"])
    partial = _row(params["number"], "On my")
    task_cls = sms_tasks.SmsResendForSimpleSMSMessenger
    self.assertEqual(
        _score_with_rows(task_cls, params, sent=(exact, exact)), 1.0
    )
    self.assertEqual(_score_with_rows(task_cls, params, sent=(exact,)), 0.0)
    self.assertEqual(
        _score_with_rows(task_cls, params, sent=(exact, partial)), 0.0
    )

  def test_send_to_contact_verifier_and_contact_isolation(self):
    params = {
        "contact_name": "Alex Morgan",
        "number": "202-555-0101",
        "message": "On my way.",
    }
    exact = _row(params["number"], params["message"])
    task_cls = sms_tasks.SmsSendToContactForSimpleSMSMessenger
    self.assertEqual(_score_with_rows(task_cls, params, sent=(exact,)), 1.0)
    self.assertEqual(_score_with_rows(task_cls, params, sent=()), 0.0)
    self.assertEqual(
        _score_with_rows(
            task_cls,
            params,
            sent=(_row("202-555-0102", params["message"]),),
        ),
        0.0,
    )

    task = task_cls(params)
    env = _FakeEnv()
    with mock.patch.object(
        sms_tasks.contacts_utils, "clear_contacts"
    ) as clear_contacts, mock.patch.object(
        sms_tasks, "_seed_contact"
    ) as seed_contact, mock.patch.object(
        sms_tasks._SmsTaskBase, "tear_down"
    ) as parent_teardown:
      task._seed_state(env)
      task.tear_down(env)
    self.assertEqual(clear_contacts.call_count, 2)
    seed_contact.assert_called_once_with(
        env, params["contact_name"], params["number"]
    )
    parent_teardown.assert_called_once_with(env)

  def test_send_received_address_uses_paired_source_and_preserves_seed(self):
    params = {
        "name1": "Alex Morgan",
        "number": "202-555-0101",
        "name2": "Blair Chen",
        "message": "123 Main St Girdwood, AK, 99587",
    }
    source = sms_tasks._deterministic_distinct_number(params["number"])
    self.assertEqual(source, "202-555-0102")
    self.assertFalse(sms_tasks._numbers_equivalent(source, params["number"]))
    sent = _row(params["number"], params["message"])
    source_seed = _row(source, params["message"])
    task_cls = sms_tasks.SmsSendReceivedAddressForSimpleSMSMessenger

    self.assertEqual(
        _score_with_rows(
            task_cls,
            params,
            sent=(sent,),
            all_rows=(source_seed, sent),
        ),
        1.0,
    )
    self.assertEqual(
        _score_with_rows(task_cls, params, sent=(sent,), all_rows=(sent,)),
        0.0,
    )
    self.assertEqual(
        _score_with_rows(
            task_cls, params, sent=(), all_rows=(source_seed,)
        ),
        0.0,
    )

  def test_create_draft_rejects_noop_send_and_wrong_body_send(self):
    params = {"number": "202-555-0101", "message": "On my way."}
    draft = _row(params["number"], params["message"])
    task_cls = sms_tasks.SmsCreateDraftMessageForSimpleSMSMessenger
    self.assertEqual(
        _score_with_rows(task_cls, params, draft=(draft,)), 1.0
    )
    self.assertEqual(_score_with_rows(task_cls, params), 0.0)
    for sent in (
        _row(params["number"], params["message"]),
        _row(params["number"], "Wrong body"),
    ):
      with self.subTest(sent=sent):
        self.assertEqual(
            _score_with_rows(
                task_cls, params, sent=(sent,), draft=(draft,)
            ),
            0.0,
        )

  def test_edit_draft_requires_replacement_only_and_no_sent_rows(self):
    params = {
        "number": "202-555-0101",
        "old_message": "On my way.",
        "new_message": "Running 5 minutes late.",
    }
    old = _row(params["number"], params["old_message"])
    new = _row(params["number"], params["new_message"])
    task_cls = sms_tasks.SmsEditDraftMessageForSimpleSMSMessenger
    self.assertEqual(_score_with_rows(task_cls, params, draft=(new,)), 1.0)
    self.assertEqual(_score_with_rows(task_cls, params, draft=(old,)), 0.0)
    self.assertEqual(
        _score_with_rows(task_cls, params, draft=(old, new)), 0.0
    )
    self.assertEqual(
        _score_with_rows(
            task_cls,
            params,
            sent=(_row(params["number"], "Wrong body"),),
            draft=(new,),
        ),
        0.0,
    )

  def test_delete_conversation_requires_target_absent_and_decoy_preserved(self):
    params = {
        "number": "202-555-0101",
        "decoy_number": "303-555-0102",
    }
    target = _row(params["number"], "hello")
    decoy = _row(params["decoy_number"], "CATBench decoy conversation")
    suffix_collision = _row("991-202-555-0101", "unrelated")
    task_cls = sms_tasks.SmsDeleteConversationForSimpleSMSMessenger
    self.assertEqual(
        _score_with_rows(task_cls, params, all_rows=(decoy,)), 1.0
    )
    self.assertEqual(
        _score_with_rows(task_cls, params, all_rows=(target, decoy)), 0.0
    )
    self.assertEqual(_score_with_rows(task_cls, params, all_rows=()), 0.0)
    self.assertEqual(
        _score_with_rows(
            task_cls, params, all_rows=(suffix_collision, decoy)
        ),
        1.0,
    )

  def test_forward_requires_exact_target_send_and_preserved_source(self):
    params = {
        "source_number": "202-555-0101",
        "target_number": "303-555-0102",
        "message": "On my way.",
    }
    source = _row(params["source_number"], params["message"])
    exact = _row(params["target_number"], params["message"])
    task_cls = sms_tasks.SmsForwardMessageForSimpleSMSMessenger
    self.assertEqual(
        _score_with_rows(
            task_cls, params, sent=(exact,), all_rows=(source, exact)
        ),
        1.0,
    )
    self.assertEqual(
        _score_with_rows(task_cls, params, sent=(), all_rows=(source,)), 0.0
    )
    # Sending the known body directly and deleting the seeded source is not a
    # completed forward operation.
    self.assertEqual(
        _score_with_rows(task_cls, params, sent=(exact,), all_rows=(exact,)),
        0.0,
    )
    for wrong in (
        _row(params["source_number"], params["message"]),
        _row(params["target_number"], "On my"),
    ):
      with self.subTest(wrong=wrong):
        self.assertEqual(
            _score_with_rows(
                task_cls, params, sent=(wrong,), all_rows=(source, wrong)
            ),
            0.0,
        )

  def test_archive_ignores_ui_markers_requires_durable_state(self):
    """UI text like "Moved to archive" or an empty inbox must NOT pass:
    those strings false-passed pre-agent on every target app. Only the
    telephony threads.archived flag or the app's private conversation DB
    counts."""
    params = {"number": "202-555-0101", "message": "Seeded message"}
    task = sms_tasks.SmsArchiveConversationForSimpleSMSMessenger(params)
    task.initialized = True

    with mock.patch.object(sms_tasks, "_sms_contains", return_value=True):
      with mock.patch.object(sms_tasks, "_thread_archived", return_value=False):
        with mock.patch.object(
            sms_tasks, "_app_private_archived", return_value=False
        ):
          self.assertEqual(task.is_successful(_FakeEnv()), 0.0)
          self.assertEqual(
              task.is_successful(_FakeEnv(("Moved to archive",))), 0.0
          )
          self.assertEqual(
              task.is_successful(
                  _FakeEnv(("No stored conversations have been found",))
              ),
              0.0,
          )
        with mock.patch.object(
            sms_tasks, "_app_private_archived", return_value=True
        ):
          self.assertEqual(task.is_successful(_FakeEnv()), 1.0)

  def test_archive_private_db_row_parsing(self):
    rows = "202-555-0101\tSeeded message\t1\n303-555-9999\tother\t0\n"
    with mock.patch.object(sms_tasks, "_sqlite_exec_path", return_value=rows):
      self.assertTrue(
          sms_tasks._app_private_archived(
              mock.Mock(), "org.fossify.messages", "202-555-0101", "Seeded message"
          )
      )
      self.assertFalse(
          sms_tasks._app_private_archived(
              mock.Mock(), "org.fossify.messages", "303-555-9999", "other"
          )
      )

  def test_archive_excluded_for_quik(self):
    self.assertFalse(
        hasattr(sms_tasks, "SmsArchiveConversationForQUIKSMS")
    )

  def test_archive_accepts_provider_thread_archived_flag(self):
    params = {"number": "202-555-0101", "message": "Seeded message"}
    task = sms_tasks.SmsArchiveConversationForSimpleSMSMessenger(params)
    task.initialized = True

    with mock.patch.object(sms_tasks, "_sms_contains", return_value=True):
      with mock.patch.object(sms_tasks, "_thread_archived", return_value=True):
        self.assertEqual(task.is_successful(_FakeEnv()), 1.0)

  def test_archive_rejects_deleted_conversation(self):
    params = {"number": "202-555-0101", "message": "Seeded message"}
    task = sms_tasks.SmsArchiveConversationForSimpleSMSMessenger(params)
    task.initialized = True

    with mock.patch.object(sms_tasks, "_sms_contains", return_value=False):
      self.assertEqual(task.is_successful(_FakeEnv()), 0.0)

  def test_notification_task_goal_is_conversation_mute(self):
    # The notification-settings port is not part of the curated generated
    # matrix, so synthesise a per-app class the same way the generator does.
    task_cls = type(
        "SmsOpenNotificationSettingsForSimpleSMSMessenger",
        (sms_tasks._SmsOpenNotificationSettingsBase,),
        {
            "app_names": ("com.simplemobiletools.smsmessenger",),
            "package_name": "com.simplemobiletools.smsmessenger",
            "template": sms_tasks._TEMPLATES[
                sms_tasks._SmsOpenNotificationSettingsBase
            ].format(app="Simple SMS Messenger"),
        },
    )
    params = {"number": "202-555-0101", "message": "Seeded message"}
    task = task_cls(params)

    self.assertIn("mute", task.goal.lower())
    self.assertIn("202-555-0101", task.goal)
    self.assertIn("specific conversation", task.goal.lower())

  def test_send_received_address_matches_aw_address_semantics(self):
    params = {
        "name1": "Alex Morgan",
        "number": "202-555-0101",
        "name2": "Blair Chen",
        "message": "123 Main St Girdwood, AK, 99587",
    }
    task = sms_tasks.SmsSendReceivedAddressForSimpleSMSMessenger(params)
    task.initialized = True

    with mock.patch.object(
        sms_tasks, "_sent_contains", return_value=True
    ) as sent_contains, mock.patch.object(
        sms_tasks, "_sms_contains", return_value=True
    ) as sms_contains:
      self.assertEqual(task.is_successful(_FakeEnv()), 1.0)

    sent_contains.assert_called_once_with(
        mock.ANY,
        "202-555-0101",
        "123 Main St Girdwood, AK, 99587",
    )
    sms_contains.assert_called_once_with(
        mock.ANY,
        "202-555-0102",
        "123 Main St Girdwood, AK, 99587",
    )
    self.assertIn("address of the event", task.goal.lower())
    self.assertIn("Alex Morgan", task.goal)
    self.assertIn("Blair Chen", task.goal)
    self.assertNotIn("phone number", task.goal.lower())


if __name__ == "__main__":
  absltest.main()
