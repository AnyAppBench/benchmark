"""Cross-app SMS task ports for the app-generalization suite.

SMS apps all talk to the system Telephony / SmsProvider, so the most reliable
way to validate a durable SMS task is to query the SMS content provider via
``adb`` after the agent acts, rather than parsing per-app UIs. Most tasks here
rely on that shared ground truth:

  1. ``initialize_task`` resets inbox/sent state (where possible) via adb.
  2. ``is_successful`` runs ``content query --uri content://sms`` and checks
     for the expected message body + recipient.

Tasks in this module:

  * ``SmsSend`` -- send a new SMS to a number.
  * ``SmsReply`` -- reply to a specific seeded conversation.
  * ``SmsReplyMostRecent`` -- reply to the most recent inbox thread.
  * ``SmsCreateDraftMessage`` -- create but do not send a draft.
  * ``SmsEditDraftMessage`` -- edit a seeded draft.
  * ``SmsSendClipboardContent`` -- send whatever is on the clipboard.
  * ``SmsForwardMessage`` -- forward a seeded message to another recipient.
  * ``SmsOpenConversation`` -- open a seeded conversation.
  * ``SmsDeleteConversation`` -- delete an entire seeded conversation thread.
  * ``SmsArchiveConversation`` -- legacy/debug-only archive task.
  * ``SmsOpenNotificationSettings`` -- mute a seeded conversation.
"""

from __future__ import annotations

import random
import re
import shlex
import string
import time
from typing import Any, Final

from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base as base,
)
from android_world.utils import contacts_utils


_MESSAGES: Final[tuple[str, ...]] = (
    "On my way.",
    "Running 5 minutes late.",
    "Lunch at 12?",
    "Meeting moved to 3pm.",
    "Call me when you get this.",
    "Happy birthday!",
)
_ADDRESSES: Final[tuple[str, ...]] = (
    "123 Main St Girdwood, AK, 99587",
    "6 Elm St, Birmingham, AL, 35217",
    "789 E Oak St, Phoenix AZ 85006",
    "1011 S Maple St, Little Rock, AR, 72204",
    "1415 W Cedar Ave Denver, CO, 80223",
    "968 Spruce St, Hartford, CT, 06103",
    "1819 Birch Ct, Dover, DE, 19901",
    "2021 Poplar St, Atlanta, GA, 30340",
)
_CONTACT_NAMES: Final[tuple[str, ...]] = (
    "Alex Morgan",
    "Blair Chen",
    "Casey Rivera",
    "Devon Patel",
    "Elliot Smith",
    "Finley Brown",
)
_SMS_DB_PATH: Final[str] = (
    "/data/data/com.android.providers.telephony/databases/mmssms.db"
)
_SMS_BOX_TYPES: Final[dict[str, int]] = {
    "inbox": 1,
    "sent": 2,
    "draft": 3,
}


def _random_number() -> str:
  return (
      f"{random.randint(200, 999)}-{random.randint(100, 999)}"
      f"-{random.randint(1000, 9999)}"
  )


def _random_distinct_number(*existing: str) -> str:
  existing_digits = {_normalize_number(number) for number in existing}
  number = _random_number()
  while _normalize_number(number) in existing_digits:
    number = _random_number()
  return number


def _deterministic_distinct_number(number: str) -> str:
  """Derive a stable hidden peer number from a frozen target number."""
  digits = _normalize_number(number)
  if not digits:
    raise ValueError(f"Cannot derive a peer from an empty number: {number!r}")
  replacement = str((int(digits[-1]) + 1) % 10)
  derived = digits[:-1] + replacement
  if len(derived) == 10:
    return f"{derived[:3]}-{derived[3:6]}-{derived[6:]}"
  return derived


def _random_message() -> str:
  return random.choice(_MESSAGES)


def _random_address() -> str:
  return random.choice(_ADDRESSES)


def _random_distinct_contact_names() -> tuple[str, str]:
  first, second = random.sample(_CONTACT_NAMES, 2)
  return first, second


def _random_long_message() -> str:
  # > 160 chars to force a multi-segment send.
  return "".join(random.choices(string.ascii_letters + " .", k=200))


def _adb_shell(env: interface.AsyncEnv, cmd: str) -> str:
  out = adb_utils.issue_generic_request(["shell", cmd], env.controller)
  return out.generic.output.decode("utf-8", errors="ignore") if out else ""


def _sqlite_exec_path(env: interface.AsyncEnv, db_path: str, sql: str) -> str:
  wrapped_sql = f"PRAGMA busy_timeout=5000; {sql}"
  last_exc: Exception | None = None
  for attempt in range(5):
    try:
      return _adb_shell(
          env,
          f"sqlite3 {shlex.quote(db_path)} {shlex.quote(wrapped_sql)}",
      )
    except Exception as exc:  # pylint: disable=broad-except
      last_exc = exc
      if attempt == 4:
        break
      time.sleep(0.5 * (attempt + 1))
  if last_exc is not None:
    raise last_exc
  return ""


def _sqlite_exec(env: interface.AsyncEnv, sql: str) -> str:
  return _sqlite_exec_path(env, _SMS_DB_PATH, sql)


def _sql_quote(value: str) -> str:
  return "'" + value.replace("'", "''") + "'"


def _reset_sms_provider(env: interface.AsyncEnv) -> None:
  _sqlite_exec(env, "DELETE FROM sms; DELETE FROM threads;")


def _set_default_sms(env: interface.AsyncEnv, package_name: str) -> None:
  _adb_shell(
      env,
      (
          "cmd role add-role-holder --user 0 android.app.role.SMS"
          f" {shlex.quote(package_name)} >/dev/null 2>&1 ||"
          " settings put secure sms_default_application"
          f" {shlex.quote(package_name)}"
          " >/dev/null 2>&1 || true"
      ),
  )


_SMS_QUERY_PROJECTION: Final[str] = "address:body"
_FIELD_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|, )([A-Za-z_][A-Za-z0-9_]*)="
)


def _normalize_number(value: str) -> str:
  return "".join(ch for ch in value if ch.isdigit())


def _numbers_equivalent(first: str, second: str) -> bool:
  """Match the same NANP number without accepting arbitrary suffixes.

  SmsProvider may add or remove the North-American country code, but an
  unrestricted ``endswith`` also accepts a different, longer phone number.
  Frozen CATBench SMS instances use ten-digit US-style numbers, so the only
  permitted normalization beyond punctuation is one leading ``1``.
  """
  first_digits = _normalize_number(first)
  second_digits = _normalize_number(second)
  if not first_digits or not second_digits:
    return False
  if first_digits == second_digits:
    return True
  if len(first_digits) == 10 and second_digits == "1" + first_digits:
    return True
  if len(second_digits) == 10 and first_digits == "1" + second_digits:
    return True
  return False


def _normalize_body(value: str) -> str:
  return re.sub(r"\s+", " ", value).strip().casefold()


def _canonical_sms_body(value: str) -> str:
  """Normalize transport line endings while preserving requested content."""
  return value.replace("\r\n", "\n").replace("\r", "\n")


def _parse_sms_row(row: str) -> dict[str, str]:
  """Parse one ``adb shell content query`` row without losing comma-rich body."""
  match = re.match(r"Row:\s*\d+\s*(.*)", row.strip(), flags=re.S)
  payload = match.group(1) if match else row.strip()
  fields: dict[str, str] = {}
  matches = list(_FIELD_RE.finditer(payload))
  for index, field_match in enumerate(matches):
    key = field_match.group(1)
    start = field_match.end()
    end = matches[index + 1].start() if index + 1 < len(matches) else len(payload)
    value = payload[start:end]
    if value.endswith(", "):
      value = value[:-2]
    fields[key] = value.strip()
  return fields


def _parse_sms_rows(output: str) -> list[dict[str, str]]:
  if not output.strip() or output.lstrip().startswith("No result"):
    return []
  rows = re.split(r"(?m)(?=^Row:\s*\d+)", output)
  return [
      parsed for parsed in (_parse_sms_row(row) for row in rows)
      if parsed
  ]


def _query_sms_rows(env: interface.AsyncEnv, uri: str) -> list[dict[str, str]]:
  sms_type = None
  if uri.endswith("/inbox"):
    sms_type = _SMS_BOX_TYPES["inbox"]
  elif uri.endswith("/sent"):
    sms_type = _SMS_BOX_TYPES["sent"]
  elif uri.endswith("/draft"):
    sms_type = _SMS_BOX_TYPES["draft"]
  where = f" WHERE type={sms_type}" if sms_type is not None else ""
  out = _sqlite_exec(
      env,
      "SELECT COALESCE(address,'') || char(31) || COALESCE(body,'')"
      f" FROM sms{where};",
  )
  rows = []
  for line in out.splitlines():
    if "\x1f" not in line:
      continue
    address, body = line.split("\x1f", 1)
    rows.append({"address": address, "body": body})
  return rows


def _row_matches(
    row: dict[str, str],
    number: str,
    body: str,
    *,
    body_prefix: bool = False,
) -> bool:
  if not _numbers_equivalent(row.get("address", ""), number):
    return False
  actual_body = _canonical_sms_body(row.get("body", ""))
  expected_body = _canonical_sms_body(body)
  if body_prefix:
    return actual_body.startswith(expected_body)
  return actual_body == expected_body


def _message_count(
    env: interface.AsyncEnv,
    uri: str,
    number: str,
    body: str,
    *,
    body_prefix: bool = False,
) -> int:
  return sum(
      1 for row in _query_sms_rows(env, uri)
      if _row_matches(row, number, body, body_prefix=body_prefix)
  )


def _message_contains(
    env: interface.AsyncEnv,
    uri: str,
    number: str,
    body: str,
    *,
    body_prefix: bool = False,
) -> bool:
  return _message_count(env, uri, number, body, body_prefix=body_prefix) > 0


def _content_bind(name: str, value_type: str, value: str | int) -> str:
  return f"--bind {name}:{value_type}:{shlex.quote(str(value))}"


def _content_value(value: str) -> str:
  return shlex.quote(value)


def _seed_contact(env: interface.AsyncEnv, name: str, number: str) -> None:
  """Insert a contact directly through ContactsProvider.

  UI-based contact insertion is brittle across launcher/contact-app states.
  The SMS tasks only need the contact to exist for lookup, so provider seeding
  is the stable equivalent.
  """
  last_raw_out = ""
  for attempt in range(3):
    raw_out = _adb_shell(
        env,
        (
            "content insert --uri content://com.android.contacts/raw_contacts"
            " --bind account_name:s:seed --bind account_type:s:local"
        ),
    )
    last_raw_out = raw_out
    match = re.search(r"/raw_contacts/(\d+)", raw_out)
    if not match:
      match = re.search(r"/(\d+)\s*$", raw_out.strip())
    raw_id = match.group(1) if match else ""
    if not raw_id:
      # Some platform builds return an empty string for content insert; query
      # the newest seeded row instead of treating that as a setup failure.
      query_out = _adb_shell(
          env,
          (
              "content query --uri content://com.android.contacts/raw_contacts"
              " --projection _id --where \"account_name='seed'\""
              " --sort \"_id DESC\""
          ),
      )
      ids = re.findall(r"_id=(\d+)", query_out)
      if ids:
        raw_id = ids[0]
    if raw_id:
      _adb_shell(
          env,
          (
              f"content insert --uri content://com.android.contacts/data"
              f" --bind raw_contact_id:i:{raw_id}"
              f" --bind mimetype:s:vnd.android.cursor.item/name"
              f" --bind data1:s:{_content_value(name)}"
          ),
      )
      _adb_shell(
          env,
          (
              f"content insert --uri content://com.android.contacts/data"
              f" --bind raw_contact_id:i:{raw_id}"
              f" --bind mimetype:s:vnd.android.cursor.item/phone_v2"
              f" --bind data1:s:{_content_value(number)}"
          ),
      )
      return
    time.sleep(0.5 * (attempt + 1))
  raise RuntimeError(f"Could not seed contact raw row: {last_raw_out!r}")


def _insert_sms_message(
    env: interface.AsyncEnv,
    box: str,
    number: str,
    body: str,
    *,
    date_ms: int | None = None,
) -> None:
  sms_type = _SMS_BOX_TYPES[box]
  quoted_number = _sql_quote(number)
  quoted_body = _sql_quote(body)
  recipient_id = (
      f"(SELECT _id FROM canonical_addresses WHERE address={quoted_number} "
      "ORDER BY _id DESC LIMIT 1)"
  )
  recipient_text = f"CAST({recipient_id} AS TEXT)"
  date_expr = (
      str(int(date_ms))
      if date_ms is not None
      else "(strftime('%s','now') * 1000)"
  )
  sql = " ".join((
      "INSERT INTO canonical_addresses(address)",
      f"SELECT {quoted_number}",
      "WHERE NOT EXISTS (",
      "SELECT 1 FROM canonical_addresses",
      f"WHERE address={quoted_number}",
      ");",
      "INSERT INTO threads(date,message_count,recipient_ids,snippet,read,archived)",
      f"SELECT {date_expr},0,",
      f"{recipient_text},{quoted_body},1,0",
      f"WHERE NOT EXISTS (SELECT 1 FROM threads WHERE recipient_ids={recipient_text});",
      "INSERT INTO sms(thread_id,address,date,read,status,type,body,seen)",
      "VALUES (",
      f"(SELECT _id FROM threads WHERE recipient_ids={recipient_text}",
      "ORDER BY _id DESC LIMIT 1),",
      f"{quoted_number},{date_expr},1,-1,{sms_type},",
      f"{quoted_body},1",
      ");",
  ))
  _sqlite_exec(env, sql)
  uri = f"content://sms/{box}"
  if not _message_contains(env, uri, number, body):
    raise RuntimeError(
        f"Failed to seed SMS {box} row for {number!r}: {body!r}"
    )


def _seed_inbox_message(
    env: interface.AsyncEnv,
    number: str,
    body: str,
    *,
    date_ms: int | None = None,
) -> None:
  _insert_sms_message(env, "inbox", number, body, date_ms=date_ms)


def _seed_sent_message(env: interface.AsyncEnv, number: str, body: str) -> None:
  _insert_sms_message(env, "sent", number, body)


def _seed_draft_message(env: interface.AsyncEnv, number: str, body: str) -> None:
  _insert_sms_message(env, "draft", number, body)


def _sent_contains(
    env: interface.AsyncEnv,
    number: str,
    body: str,
    *,
    body_prefix: bool = False,
) -> bool:
  return _message_contains(
      env, "content://sms/sent", number, body, body_prefix=body_prefix
  )


def _draft_contains(env: interface.AsyncEnv, number: str, body: str) -> bool:
  return _message_contains(env, "content://sms/draft", number, body)


def _sms_contains(env: interface.AsyncEnv, number: str, body: str) -> bool:
  return _message_contains(env, "content://sms", number, body)


# App-private archive flags, discovered by on-device probing per
# docs/tasks_guide.md. Fossify Messages and Simple SMS Messenger persist an
# `archived` column in their Room `conversations.db`; Google Messages keeps
# `archive_status` in `bugle_db`. QUIK uses a Realm database with no stable
# parseable artifact, so the archive task is excluded for QUIK.
_PRIVATE_CONVERSATIONS_DBS: dict[str, str] = {
    "org.fossify.messages": (
        "/data/data/org.fossify.messages/databases/conversations.db"
    ),
    "com.simplemobiletools.smsmessenger": (
        "/data/data/com.simplemobiletools.smsmessenger/databases/"
        "conversations.db"
    ),
}
_BUGLE_DB_PATH: str = (
    "/data/data/com.google.android.apps.messaging/databases/bugle_db"
)


def _app_private_archived(
    env: interface.AsyncEnv,
    package_name: str,
    number: str,
    body: str,
) -> bool:
  """Returns True iff the app's own storage marks the conversation archived."""
  expected_digits = re.sub(r"\D", "", number)[-7:]
  expected_body = _normalize_body(body)

  def row_matches(number_field: str, snippet_field: str) -> bool:
    row_digits = re.sub(r"\D", "", number_field)
    if expected_digits and expected_digits in row_digits:
      return True
    return bool(expected_body) and (
        expected_body in _normalize_body(snippet_field)
    )

  db_path = _PRIVATE_CONVERSATIONS_DBS.get(package_name)
  if db_path:
    out = _sqlite_exec_path(
        env,
        db_path,
        "SELECT phone_number || char(9) || snippet || char(9) || archived"
        " FROM conversations;",
    )
    for line in out.splitlines():
      parts = line.split("\t")
      if len(parts) == 3 and row_matches(parts[0], parts[1]):
        return parts[2].strip() == "1"
    return False

  if package_name == "com.google.android.apps.messaging":
    out = _sqlite_exec_path(
        env,
        _BUGLE_DB_PATH,
        "SELECT COALESCE(name, '') || char(9) ||"
        " COALESCE(snippet_text, '') || char(9) || archive_status"
        " FROM conversations;",
    )
    for line in out.splitlines():
      parts = line.split("\t")
      if len(parts) == 3 and row_matches(parts[0], parts[1]):
        return parts[2].strip() not in ("", "0")
    return False

  return False


def _thread_archived(env: interface.AsyncEnv, number: str, body: str) -> bool:
  quoted_number = _sql_quote(number)
  quoted_body = _sql_quote(body)
  out = _sqlite_exec(
      env,
      "SELECT 'archived=' || COALESCE(t.archived, 0)"
      " FROM threads t JOIN sms s ON s.thread_id = t._id"
      f" WHERE s.address = {quoted_number} AND s.body = {quoted_body}"
      " ORDER BY s._id DESC LIMIT 1;",
  )
  # Match the tagged value exactly: _sqlite_exec prefixes every query with
  # `PRAGMA busy_timeout=5000;`, whose "5000" echo previously made any
  # non-"0" line count as archived — a guaranteed false pass.
  return any(line.strip() == "archived=1" for line in out.splitlines())


# -----------------------------------------------------------------------------
# Base evaluators.
# -----------------------------------------------------------------------------


class _SmsTaskBase(base.PackageAppEval):
  """Shared SMS teardown."""

  clear_data_on_init = False
  clear_data_on_teardown = False

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _reset_sms_provider(env)
    _set_default_sms(env, self.package_name)
    super().initialize_task(env)
    self._seed_state(env)
    if type(self)._seed_state is not _SmsTaskBase._seed_state:
      # Subclass seeded precondition data after the app was launched; restart
      # the app so the seeded threads/contacts are actually visible to it.
      self._relaunch_app(env)

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    """Hook: seed precondition SMS/contact data. Runs after app launch and is
    always followed by an app relaunch (see initialize_task)."""

  def _relaunch_app(self, env: interface.AsyncEnv) -> None:
    """Force-stop and relaunch the target app so it re-reads SmsProvider.

    Seeding writes ``mmssms.db`` directly with sqlite3, which bypasses
    ContentObserver notifications — an already-running messenger keeps
    showing its stale (empty) conversation list and the agent never sees
    the seeded precondition data. Every seeding task must call this after
    its seeds are in place.
    """
    _adb_shell(
        env, f"am force-stop {shlex.quote(self.package_name)} || true"
    )
    deadline = time.time() + 3.0
    while time.time() < deadline:
      pid = _adb_shell(
          env, f"pidof {shlex.quote(self.package_name)} || true"
      ).strip()
      if not pid:
        break
      time.sleep(0.1)
    adb_utils.launch_app(self.package_name, env.controller)


class _SmsSendBase(_SmsTaskBase):
  """Send a new SMS; verify via sent-provider."""

  complexity = 2
  schema = {
      "type": "object",
      "properties": {
          "number": {"type": "string"},
          "message": {"type": "string"},
      },
      "required": ["number", "message"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if _sent_contains(env, self._params["number"], self._params["message"])
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"number": _random_number(), "message": _random_message()}


class _SmsSendLongBase(_SmsTaskBase):
  """Send a multi-segment SMS (> 160 chars)."""

  complexity = 2.4
  schema = _SmsSendBase.schema

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if _sent_contains(
            env,
            self._params["number"],
            self._params["message"][:20],
            body_prefix=True,
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"number": _random_number(), "message": _random_long_message()}


class _SmsReplyMostRecentBase(_SmsTaskBase):
  """Reply to the most recent thread (seeded in ``initialize_task``)."""

  complexity = 2.6
  schema = {
      "type": "object",
      "properties": {
          "number": {"type": "string"},
          "message": {"type": "string"},
          "seed_message": {"type": "string"},
      },
      "required": ["number", "message"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_inbox_message(
        env,
        self._params["number"],
        self._params.get("seed_message", "CATBench most recent message"),
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if _sent_contains(env, self._params["number"], self._params["message"])
        and _sms_contains(
            env,
            self._params["number"],
            self._params.get("seed_message", "CATBench most recent message"),
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {
        "number": _random_number(),
        "message": _random_message(),
        "seed_message": "CATBench most recent message",
    }


class _SmsSendClipboardContentBase(_SmsTaskBase):
  """Send a message whose body is on the clipboard."""

  complexity = 2.4
  schema = _SmsSendBase.schema

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _adb_shell(
        env,
        "am broadcast -a clipper.set -e text"
        f" {shlex.quote(self._params['message'])}"
        " >/dev/null 2>&1 || true",
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if _sent_contains(env, self._params["number"], self._params["message"])
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"number": _random_number(), "message": _random_message()}


class _SmsCreateDraftMessageBase(_SmsTaskBase):
  """Create a draft message without sending it."""

  complexity = 2.2
  schema = _SmsSendBase.schema

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # The provider is empty at initialization, so any sent row is collateral
    # execution and violates the explicit "Do NOT send it" instruction.  The
    # prior expected-body-only check false-passed after a wrong/partial send.
    if _query_sms_rows(env, "content://sms/sent"):
      return 0.0
    return (
        1.0
        if _draft_contains(env, self._params["number"], self._params["message"])
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"number": _random_number(), "message": _random_message()}


class _SmsEditDraftMessageBase(_SmsTaskBase):
  """Edit an existing draft message and keep it unsent."""

  complexity = 2.6
  schema = {
      "type": "object",
      "properties": {
          "number": {"type": "string"},
          "old_message": {"type": "string"},
          "new_message": {"type": "string"},
      },
      "required": ["number", "old_message", "new_message"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_draft_message(
        env, self._params["number"], self._params["old_message"]
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if _query_sms_rows(env, "content://sms/sent"):
      return 0.0
    new_present = _draft_contains(
        env, self._params["number"], self._params["new_message"]
    )
    old_present = _draft_contains(
        env, self._params["number"], self._params["old_message"]
    )
    return 1.0 if new_present and not old_present else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    old_message, new_message = random.sample(_MESSAGES, 2)
    return {
        "number": _random_number(),
        "old_message": old_message,
        "new_message": new_message,
    }


class _SmsSendToContactBase(_SmsTaskBase):
  """Send a message by looking up a contact name (not the raw number)."""

  complexity = 2.8
  schema = {
      "type": "object",
      "properties": {
          "contact_name": {"type": "string"},
          "number": {"type": "string"},
          "message": {"type": "string"},
      },
      "required": ["contact_name", "number", "message"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    # Prevent duplicate names from earlier tasks selecting a stale number.
    # This mirrors SmsSendReceivedAddress and makes every app start from the
    # same single-contact precondition.
    contacts_utils.clear_contacts(env.controller)
    _seed_contact(env, self._params["contact_name"], self._params["number"])

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if _sent_contains(env, self._params["number"], self._params["message"])
        else 0.0
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    try:
      contacts_utils.clear_contacts(env.controller)
    finally:
      super().tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    first = random.choice(("Alex", "Sam", "Jordan", "Taylor", "Casey"))
    last = random.choice(("Parker", "Lee", "Morgan", "Kim", "Singh"))
    return {
        "contact_name": f"{first} {last}",
        "number": _random_number(),
        "message": _random_message(),
    }


class _SmsForwardMessageBase(_SmsTaskBase):
  """Forward a seeded inbox message to a new recipient."""

  complexity = 3
  schema = {
      "type": "object",
      "properties": {
          "source_number": {"type": "string"},
          "target_number": {"type": "string"},
          "message": {"type": "string"},
      },
      "required": ["source_number", "target_number", "message"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_inbox_message(
        env, self._params["source_number"], self._params["message"]
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if _sent_contains(
            env, self._params["target_number"], self._params["message"]
        )
        and _sms_contains(
            env, self._params["source_number"], self._params["message"]
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    source_number = _random_number()
    return {
        "source_number": source_number,
        "target_number": _random_distinct_number(source_number),
        "message": _random_message(),
    }


class _SmsDeleteThreadBase(_SmsTaskBase):
  """Delete an entire conversation thread seeded in the inbox."""

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {
          "number": {"type": "string"},
          "decoy_number": {"type": "string"},
      },
      "required": ["number"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    self._params.setdefault("decoy_number", _random_number())
    while _numbers_equivalent(
        self._params["decoy_number"], self._params["number"]
    ):
      self._params["decoy_number"] = _random_distinct_number(
          self._params["number"]
      )
    for body in ("hello", "are you there?", "please respond"):
      _seed_inbox_message(env, self._params["number"], body)
    _seed_inbox_message(
        env,
        self._params["decoy_number"],
        "CATBench decoy conversation",
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    target_present = any(
        row for row in _query_sms_rows(env, "content://sms")
        if _numbers_equivalent(
            row.get("address", ""), self._params["number"]
        )
    )
    decoy_present = _sms_contains(
        env, self._params["decoy_number"], "CATBench decoy conversation"
    )
    return 1.0 if not target_present and decoy_present else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    number = _random_number()
    decoy_number = _random_distinct_number(number)
    return {"number": number, "decoy_number": decoy_number}


class _SmsSendNumericBase(_SmsSendBase):
  """Send a body that is purely digits (tests number-pad text entry)."""

  complexity = 1.8

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    digits = "".join(str(random.randint(0, 9)) for _ in range(20))
    return {"number": _random_number(), "message": digits}


class _SmsSendEmojiBase(_SmsSendBase):
  """Send a body containing emoji characters."""

  complexity = 2.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    emoji = random.choice(("Great work \U0001F389", "On the way \U0001F697",
                           "Coffee soon? ☕", "Lunch! \U0001F371"))
    return {"number": _random_number(), "message": emoji}


class _SmsSendUppercaseBase(_SmsSendBase):
  """Send a body in ALL CAPS."""

  complexity = 1.8

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {
        "number": _random_number(),
        "message": _random_message().upper(),
    }


class _SmsSendQuestionBase(_SmsSendBase):
  """Send a body that ends with a question mark."""

  complexity = 1.8

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    body = random.choice((
        "Are you free tonight?",
        "Did you receive the package?",
        "Should we book the table?",
        "Can you pick up milk?",
    ))
    return {"number": _random_number(), "message": body}


class _SmsSendToTwoRecipientsBase(_SmsTaskBase):
  """Send the same body to two distinct recipients (sequential or group)."""

  complexity = 2.6
  schema = {
      "type": "object",
      "properties": {
          "number_a": {"type": "string"},
          "number_b": {"type": "string"},
          "message": {"type": "string"},
      },
      "required": ["number_a", "number_b", "message"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    a = _sent_contains(env, self._params["number_a"], self._params["message"])
    b = _sent_contains(env, self._params["number_b"], self._params["message"])
    return 1.0 if a and b else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    number_a = _random_number()
    return {
        "number_a": number_a,
        "number_b": _random_distinct_number(number_a),
        "message": _random_message(),
    }


class _SmsSendMultilineBase(_SmsSendBase):
  """Send a body with explicit newline characters."""

  complexity = 2.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {
        "number": _random_number(),
        "message": "Line 1\nLine 2\nLine 3",
    }


# -----------------------------------------------------------------------------
# Cross-app target tasks (hybrid mode).
# -----------------------------------------------------------------------------


class _SmsReplyBase(_SmsTaskBase):
  """Reply to a specific seeded conversation, identified by sender number.

  Distinct from ``ReplyMostRecent``: the agent must navigate to the named
  thread (which is not necessarily the most recent) before replying.
  """

  complexity = 2.6
  schema = {
      "type": "object",
      "properties": {
          "number": {"type": "string"},
          "decoy_number": {"type": "string"},
          "message": {"type": "string"},
      },
      "required": ["number", "decoy_number", "message"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    # Seed two threads; the more recent one is from the decoy so the agent
    # must specifically pick the named sender's thread. Explicit timestamps
    # avoid both inserts landing in the same millisecond and being ordered by
    # an app-specific tie-breaker.
    newest_ms = int(time.time() * 1000)
    for sender, body, date_ms in (
        (
            self._params["number"],
            "Older message you must reply to",
            newest_ms - 1000,
        ),
        (self._params["decoy_number"], "Decoy newest thread", newest_ms),
    ):
      _seed_inbox_message(env, sender, body, date_ms=date_ms)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if _sent_contains(env, self._params["number"], self._params["message"])
        and not _sent_contains(
            env, self._params["decoy_number"], self._params["message"]
        )
        and _sms_contains(
            env, self._params["number"], "Older message you must reply to"
        )
        and _sms_contains(
            env, self._params["decoy_number"], "Decoy newest thread"
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    number = _random_number()
    return {
        "number": number,
        "decoy_number": _random_distinct_number(number),
        "message": _random_message(),
    }


class _SmsResendBase(_SmsTaskBase):
  """Re-send the most recent sent message to its original recipient.

  Two rows in content://sms/sent with the same address+body are accepted as
  evidence the message was re-sent rather than only sent once.
  """

  complexity = 2.6
  schema = {
      "type": "object",
      "properties": {
          "number": {"type": "string"},
          "message": {"type": "string"},
      },
      "required": ["number", "message"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    # Seed a previously-sent message that the agent will re-send.
    _seed_sent_message(env, self._params["number"], self._params["message"])

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    matches = _message_count(
        env,
        "content://sms/sent",
        self._params["number"],
        self._params["message"],
    )
    return 1.0 if matches >= 2 else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"number": _random_number(), "message": _random_message()}


class _SmsSendAddressBase(_SmsTaskBase):
  """Forward an address text received from one contact to another contact.

  This preserves AndroidWorld's ``SimpleSmsSendReceivedAddress`` semantics:
  a source contact sends an address, and the agent must text that address to
  the target contact.
  """

  complexity = 1.8
  schema = {
      "type": "object",
      "properties": {
          "name1": {"type": "string"},
          "number": {"type": "string"},
          "name2": {"type": "string"},
          "message": {"type": "string"},
      },
      "required": ["name1", "number", "name2", "message"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    contacts_utils.clear_contacts(env.controller)
    # Derive the hidden source number from the frozen target parameters. A
    # fresh random draw here made the nominally paired app instances start
    # from different provider states even though their visible instruction
    # and serialized params were identical.
    source_number = _deterministic_distinct_number(self._params["number"])
    _seed_contact(env, self._params["name1"], self._params["number"])
    _seed_contact(env, self._params["name2"], source_number)
    _seed_inbox_message(
        env,
        source_number,
        self._params["message"],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    source_number = _deterministic_distinct_number(self._params["number"])
    return (
        1.0
        if _sent_contains(
            env, self._params["number"], self._params["message"]
        )
        and _sms_contains(env, source_number, self._params["message"])
        else 0.0
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    try:
      contacts_utils.clear_contacts(env.controller)
    finally:
      super().tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    name1, name2 = _random_distinct_contact_names()
    return {
        "name1": name1,
        "number": _random_number(),
        "name2": name2,
        "message": _random_address(),
    }


class _SmsOpenConversationBase(_SmsTaskBase):
  """Open a seeded conversation from a known sender."""

  complexity = 1.8
  schema = {
      "type": "object",
      "properties": {
          "number": {"type": "string"},
          "message": {"type": "string"},
      },
      "required": ["number", "message"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_inbox_message(
        env, self._params["number"], self._params["message"]
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not _sms_contains(env, self._params["number"], self._params["message"]):
      return 0.0
    ui = env.get_state().ui_elements
    body_ok = base.element_text_contains(ui, (self._params["message"],))
    number_ok = base.element_text_contains(ui, (self._params["number"],))
    # The inbox list also shows sender + snippet, so body/number alone do not
    # prove the conversation was opened. Require a compose/reply affordance,
    # which only the conversation detail view exposes.
    compose_ok = base.element_text_contains(
        ui,
        (
            "type a message",
            "text message",
            "send message",
            "write a message",
            "message...",
            "type message",
            "say something",
        ),
    ) or base.element_text_contains_word(ui, ("send",))
    return 1.0 if (body_ok or number_ok) and compose_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"number": _random_number(), "message": _random_message()}


class _SmsArchiveConversationBase(_SmsTaskBase):
  """Archive a seeded conversation, validated against durable storage only.

  Earlier revisions fell back to UI heuristics (an "archive(d)" label or an
  empty inbox) which false-passed on every target app: "Archive" appears as
  standing UI text and the inbox placeholder shows while the list is still
  loading. Per docs/tasks_guide.md the validator now reads only durable
  state:

    * shared telephony ``threads.archived`` flag, and
    * the app's private archive flag, discovered by on-device probing —
      Fossify Messages / Simple SMS Messenger keep ``archived`` in their
      ``conversations.db``; Google Messages keeps ``archive_status`` in
      ``bugle_db``.

  QUIK stores archive state in a Realm database with no stable parseable
  artifact, so this task is excluded for QUIK (see ``excluded_packages``
  and app_generalization_profiles.py).
  """

  excluded_packages = ("dev.octoshrimpy.quik.fdroid",)
  complexity = 2.2
  schema = _SmsOpenConversationBase.schema

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_inbox_message(
        env, self._params["number"], self._params["message"]
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not _sms_contains(env, self._params["number"], self._params["message"]):
      return 0.0
    if _thread_archived(
        env, self._params["number"], self._params["message"]
    ):
      return 1.0
    return (
        1.0
        if _app_private_archived(
            env,
            self.package_name,
            self._params["number"],
            self._params["message"],
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"number": _random_number(), "message": _random_message()}


class _SmsOpenNotificationSettingsBase(_SmsTaskBase):
  """Mute or disable notifications for a seeded SMS conversation.

  Per-conversation notification state is app-private for the target SMS apps,
  not a shared SmsProvider column. We therefore seed a real conversation and
  only accept visible mute / notifications-off UI evidence while requiring the
  conversation row to remain present.
  """

  complexity = 1.8
  schema = _SmsOpenConversationBase.schema

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_inbox_message(
        env, self._params["number"], self._params["message"]
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not _sms_contains(env, self._params["number"], self._params["message"]):
      return 0.0
    ui = env.get_state().ui_elements
    muted = base.element_text_contains(
        ui,
        (
            "muted",
            "mute",
            "unmute",
            "notifications off",
            "notification off",
            "silent",
            "no notifications",
            "blocked notifications",
        ),
    )
    return 1.0 if muted else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"number": _random_number(), "message": _random_message()}


class _SmsScheduleMessageBase(_SmsTaskBase):
  """Schedule a future-dated send.

  Only a subset of SMS apps support scheduled sends (Fossify, QUIK, Google
  Messages do; Simple SMS Messenger does NOT). For unsupported apps we
  exclude this task on the base class so no per-app port is generated.

  Success heuristic: the message has NOT yet appeared in content://sms/sent
  but the body + recipient are visible in the app UI marked with a
  scheduled / pending indicator.
  """

  excluded_packages = ("com.simplemobiletools.smsmessenger",)
  complexity = 3
  schema = {
      "type": "object",
      "properties": {
          "number": {"type": "string"},
          "message": {"type": "string"},
          "hours_ahead": {"type": "integer"},
      },
      "required": ["number", "message", "hours_ahead"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Must NOT have actually been sent yet.
    if _sent_contains(env, self._params["number"], self._params["message"]):
      return 0.0
    ui = env.get_state().ui_elements
    # Match a real prefix of the message body, not just its first word —
    # single words like "running" appear all over unrelated UI text.
    body_needle = self._params["message"][:20].casefold()
    body_visible = any(
        body_needle in (el.text or "").casefold()
        or body_needle in (el.content_description or "").casefold()
        for el in ui
    )
    scheduled_marker = any(
        any(
            tag in (el.text or "").lower()
            for tag in ("scheduled", "pending", "later", "delayed")
        )
        for el in ui
    )
    return 1.0 if body_visible and scheduled_marker else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "number": _random_number(),
        "message": _random_message(),
        "hours_ahead": random.choice((1, 2, 4, 12, 24)),
    }


# -----------------------------------------------------------------------------
# Per-app packages and generated ports.
# -----------------------------------------------------------------------------

_SIMPLE_SMS_PACKAGE: Final[str] = "com.simplemobiletools.smsmessenger"
_FOSSIFY_MESSAGES_PACKAGE: Final[str] = "org.fossify.messages"
_QUIK_SMS_PACKAGE: Final[str] = "dev.octoshrimpy.quik.fdroid"
_GOOGLE_MESSAGES_PACKAGE: Final[str] = "com.google.android.apps.messaging"


_APP_DISPLAY_NAMES: Final[dict[str, str]] = {
    _SIMPLE_SMS_PACKAGE: "Simple SMS Messenger",
    _FOSSIFY_MESSAGES_PACKAGE: "Fossify Messages",
    _QUIK_SMS_PACKAGE: "QUIK SMS",
    _GOOGLE_MESSAGES_PACKAGE: "Messages",
}


_TEMPLATES: Final[dict[type, str]] = {
    _SmsSendBase: (
        "Using the {app} app, send an SMS to {{number}} with the message:"
        " `{{message}}`."
    ),
    _SmsSendLongBase: (
        "Using the {app} app, send a long SMS (more than 160 characters)"
        " to {{number}} with the message: `{{message}}`."
    ),
    _SmsReplyMostRecentBase: (
        "Using the {app} app, reply to the most recent SMS conversation"
        " with the message: `{{message}}`."
    ),
    _SmsSendClipboardContentBase: (
        "Using the {app} app, send an SMS to {{number}} with the message"
        " body taken from the clipboard."
    ),
    _SmsCreateDraftMessageBase: (
        "Using the {app} app, create a draft SMS to {{number}} with the"
        " message `{{message}}`. Do NOT send it."
    ),
    _SmsEditDraftMessageBase: (
        "Using the {app} app, open the draft SMS to {{number}} whose current"
        " body is `{{old_message}}`, change it to `{{new_message}}`, and do"
        " NOT send it."
    ),
    _SmsSendToContactBase: (
        "Using the {app} app, send an SMS to the contact"
        " `{{contact_name}}` with the message: `{{message}}`."
    ),
    _SmsForwardMessageBase: (
        "Using the {app} app, forward the most recent message from"
        " {{source_number}} to {{target_number}}."
    ),
    _SmsDeleteThreadBase: (
        "Using the {app} app, delete the entire SMS conversation with"
        " {{number}}. Leave other conversations intact."
    ),
    _SmsOpenConversationBase: (
        "Using the {app} app, open the conversation from {{number}} that"
        " contains the message `{{message}}`."
    ),
    _SmsArchiveConversationBase: (
        "Using the {app} app, archive the conversation from {{number}} that"
        " contains the message `{{message}}`."
    ),
    _SmsOpenNotificationSettingsBase: (
        "Using the {app} app, open the conversation from {{number}} that"
        " contains `{{message}}` and mute / disable notifications for that"
        " specific conversation."
    ),
    _SmsSendNumericBase: (
        "Using the {app} app, send an SMS to {{number}} whose body is the"
        " digit sequence: `{{message}}'."
    ),
    _SmsSendEmojiBase: (
        "Using the {app} app, send an SMS to {{number}} containing this"
        " emoji message: `{{message}}'."
    ),
    _SmsSendUppercaseBase: (
        "Using the {app} app, send an SMS to {{number}} in ALL CAPS:"
        " `{{message}}`."
    ),
    _SmsSendQuestionBase: (
        "Using the {app} app, send the question `{{message}}' as an SMS"
        " to {{number}}."
    ),
    _SmsSendToTwoRecipientsBase: (
        "Using the {app} app, send the SMS `{{message}}' to BOTH"
        " {{number_a}} and {{number_b}}."
    ),
    _SmsSendMultilineBase: (
        "Using the {app} app, send a multi-line SMS to {{number}} whose"
        " body is exactly: `{{message}}' (use newlines)."
    ),
    _SmsReplyBase: (
        "Using the {app} app, open the inbox conversation from {{number}}"
        " (NOT the most recent thread, which is from {{decoy_number}}) and"
        " reply with the message: `{{message}}`."
    ),
    _SmsResendBase: (
        "Using the {app} app, find the most recent message you sent to"
        " {{number}} and re-send the same body again so it appears twice"
        " in the conversation."
    ),
    _SmsSendAddressBase: (
        "Text the address of the event to {{name1}} that {{name2}} just"
        " sent me using the {app} app."
    ),
    _SmsScheduleMessageBase: (
        "Using the {app} app, schedule an SMS to {{number}} with the"
        " message `{{message}}` to be sent in {{hours_ahead}} hour(s)."
        " Do NOT send it now."
    ),
}


# Cross-app SMS task templates. The 10 short names below ARE the user's
# target task list for the SMS category in hybrid mode.
_BASE_SHORT_NAMES: Final[dict[type, str]] = {
    _SmsSendBase: "SmsSend",
    _SmsReplyBase: "SmsReply",
    _SmsReplyMostRecentBase: "SmsReplyMostRecent",
    _SmsResendBase: "SmsResend",
    # SendToContact replaces SendClipboard in the canonical 10: the clipboard
    # flow needs the AW clipper helper app (not on this image) and the
    # emulator clipboard is clobbered whenever the host window takes focus,
    # making it irreproducible. SendToContact seeds a named contact the agent
    # must look up, validated against the shared telephony sent box on every
    # app. Clipboard task classes stay generated below but are unscheduled.
    _SmsSendToContactBase: "SmsSendToContact",
    _SmsSendClipboardContentBase: "SmsSendClipboard",
    _SmsSendAddressBase: "SmsSendReceivedAddress",
    _SmsCreateDraftMessageBase: "SmsCreateDraftMessage",
    _SmsEditDraftMessageBase: "SmsEditDraftMessage",
    _SmsDeleteThreadBase: "SmsDeleteConversation",
    # Forward replaces Archive in the canonical 10: archive state has no
    # durable artifact on QUIK (Realm), while forwarding validates against
    # the shared telephony sent box on every app. Archive task classes stay
    # generated below for registry compatibility but are unscheduled.
    _SmsForwardMessageBase: "SmsForwardMessage",
    _SmsArchiveConversationBase: "SmsArchiveConversation",
}


_PACKAGES = (
    _SIMPLE_SMS_PACKAGE,
    _FOSSIFY_MESSAGES_PACKAGE,
    _QUIK_SMS_PACKAGE,
    _GOOGLE_MESSAGES_PACKAGE,
)


for _base_cls, _short in _BASE_SHORT_NAMES.items():
  excluded = getattr(_base_cls, "excluded_packages", ())
  for _pkg in _PACKAGES:
    if _pkg in excluded:
      continue
    _display = _APP_DISPLAY_NAMES[_pkg]
    _suffix = _display.replace(" ", "")
    _cls_name = f"{_short}For{_suffix}"
    _attrs = {
        "app_names": (_pkg,),
        "package_name": _pkg,
        "catbench_semantic_id": _short,
        "catbench_app_display_name": _display,
        "template": _TEMPLATES[_base_cls].format(app=_display),
    }
    globals()[_cls_name] = type(_cls_name, (_base_cls,), _attrs)


del _base_cls, _short, _pkg, _display, _suffix, _cls_name, _attrs
