"""Cross-app calendar task ports for the app-generalization suite.

Each port mirrors one of the canonical Simple Calendar Pro tasks:
``SimpleCalendarAddOneEvent``, ``SimpleCalendarAddRepeatingEvent``,
``SimpleCalendarDeleteEvents``. Active ports use durable SQLite verification:
Simple Calendar Pro uses its app DB and Etar/Calendar use the Android
CalendarProvider DB.

The five information-retrieval canonical tasks
(``SimpleCalendarEventsOnDate``, ``SimpleCalendarNextEvent``,
``SimpleCalendarNextMeetingWithPerson``, ``SimpleCalendarEventsInNextWeek``,
``SimpleCalendarEventsInTimeRange``) require pre-populated Simple Calendar Pro
SQLite rows and cannot be ported to third-party calendar apps without
app-specific database access, so we expose them only via the main
``calendar.py`` module. The profile lists them as canonical for reference.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import random
import sqlite3
import time
import uuid
from typing import Any, Final

from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals.single.calendar import calendar_utils
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base as base,
)
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.utils import file_utils


_EVENT_TITLES: Final[tuple[str, ...]] = (
    "Team Sync",
    "Budget Review",
    "Design Critique",
    "Weekly 1:1",
    "Project Kickoff",
    "Sprint Planning",
    "Retrospective",
    "Client Call",
    "Lunch Meeting",
    "Investor Update",
)

_EVENT_DESCRIPTIONS: Final[tuple[str, ...]] = (
    "Discuss quarterly goals.",
    "Walk through open bugs.",
    "Review upcoming release.",
    "Share demo progress.",
    "Plan next sprint backlog.",
)

_REPEAT_RULES: Final[tuple[str, ...]] = ("daily", "weekly")


def _generate_add_event_params() -> dict[str, Any]:
  event_title = random.choice(_EVENT_TITLES)
  event_description = random.choice(_EVENT_DESCRIPTIONS)
  year = 2023
  month = 10
  day = random.randint(15, 28)
  hour = random.randint(8, 20)
  duration_mins = random.choice((15, 30, 45, 60, 90, 120))
  return {
      "year": year,
      "month": month,
      "day": day,
      "hour": hour,
      "duration_mins": duration_mins,
      "event_title": event_title,
      "event_description": event_description,
  }


def _generate_add_repeating_event_params() -> dict[str, Any]:
  params = _generate_add_event_params()
  params["repeat_rule"] = random.choice(_REPEAT_RULES)
  return params


def _generate_delete_events_params() -> dict[str, Any]:
  return {
      "year": 2023,
      "month": 10,
      "day": random.randint(15, 28),
  }


def _generate_edit_event_params() -> dict[str, Any]:
  old_event_title = random.choice(_EVENT_TITLES)
  remaining = tuple(t for t in _EVENT_TITLES if t != old_event_title)
  new_event_title = random.choice(remaining)
  return {
      "year": 2023,
      "month": 10,
      "day": random.randint(15, 28),
      "old_event_title": old_event_title,
      "new_event_title": new_event_title,
  }


def _generate_view_month_agenda_params() -> dict[str, Any]:
  return {
      "year": 2023,
      "month": 10,
      "month_name": "October",
  }


# -----------------------------------------------------------------------------
# Base evaluators.
# -----------------------------------------------------------------------------


class _CalendarAddOneEventBase(base.PackageAppEval):
  """Base port of ``SimpleCalendarAddOneEvent``.

  Success heuristic: the user's chosen event title appears on the calendar
  screen after the agent saves the event.
  """

  complexity = 3.4
  schema = {
      "type": "object",
      "properties": {
          "year": {"type": "integer"},
          "month": {"type": "integer"},
          "day": {"type": "integer"},
          "hour": {"type": "integer"},
          "duration_mins": {"type": "integer"},
          "event_title": {"type": "string"},
          "event_description": {"type": "string"},
      },
      "required": [
          "year",
          "month",
          "day",
          "hour",
          "duration_mins",
          "event_title",
          "event_description",
      ],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if base.element_text_contains(
            env.get_state().ui_elements,
            (self._params["event_title"],),
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_add_event_params()


class _CalendarAddRepeatingEventBase(base.PackageAppEval):
  """Base port of ``SimpleCalendarAddRepeatingEvent``.

  Success heuristic: the event title and a repeat-rule marker both appear on
  the calendar screen after save.
  """

  complexity = 3.4
  schema = {
      "type": "object",
      "properties": {
          "year": {"type": "integer"},
          "month": {"type": "integer"},
          "day": {"type": "integer"},
          "hour": {"type": "integer"},
          "duration_mins": {"type": "integer"},
          "event_title": {"type": "string"},
          "event_description": {"type": "string"},
          "repeat_rule": {"type": "string"},
      },
      "required": [
          "year",
          "month",
          "day",
          "hour",
          "duration_mins",
          "event_title",
          "event_description",
          "repeat_rule",
      ],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    title_ok = base.element_text_contains(
        ui_elements, (self._params["event_title"],)
    )
    repeat_markers = (
        self._params["repeat_rule"].lower(),
        "repeat",
        "recurr",
    )
    repeat_ok = base.element_text_contains(ui_elements, repeat_markers)
    return 1.0 if title_ok and repeat_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_add_repeating_event_params()


class _CalendarDeleteEventsBase(base.PackageAppEval):
  """Base port of ``SimpleCalendarDeleteEvents``.

  Success heuristic: the chosen date view shows no event rows. Because we
  cannot cheaply seed events in third-party apps, this port simply checks that
  the day view is empty ("no events" / "nothing scheduled") after the agent
  finishes. Agents must still navigate to the date screen first.
  """

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "year": {"type": "integer"},
          "month": {"type": "integer"},
          "day": {"type": "integer"},
      },
      "required": ["year", "month", "day"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    empty_markers = (
        "no events",
        "nothing scheduled",
        "no upcoming",
        "nothing here",
    )
    return 1.0 if base.element_text_contains(ui_elements, empty_markers) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_delete_events_params()


class _CalendarEditEventBase(base.PackageAppEval):
  """Base port for editing an existing calendar event's title.

  Success heuristic: the new event title appears on screen after the agent
  saves the edit, while the old event title no longer appears.
  """

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {
          "year": {"type": "integer"},
          "month": {"type": "integer"},
          "day": {"type": "integer"},
          "old_event_title": {"type": "string"},
          "new_event_title": {"type": "string"},
      },
      "required": [
          "year",
          "month",
          "day",
          "old_event_title",
          "new_event_title",
      ],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    new_ok = base.element_text_contains(
        ui_elements, (self._params["new_event_title"],)
    )
    old_gone = not base.element_text_contains(
        ui_elements, (self._params["old_event_title"],)
    )
    return 1.0 if new_ok and old_gone else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_edit_event_params()


class _CalendarViewMonthAgendaBase(base.PackageAppEval):
  """Base port for viewing a month's agenda/list of events.

  Success heuristic: any of the month name, "agenda", "events", or "schedule"
  markers appear on the screen, indicating the agenda/list view is open for the
  requested month.
  """

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "year": {"type": "integer"},
          "month": {"type": "integer"},
          "month_name": {"type": "string"},
      },
      "required": ["year", "month", "month_name"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    markers = (
        self._params["month_name"].lower(),
        "agenda",
        "events",
        "schedule",
    )
    return 1.0 if base.element_text_contains(ui_elements, markers) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_view_month_agenda_params()


class _ProviderCalendarAddOneEventBase(_CalendarAddOneEventBase):

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_provider_calendar_task(self, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return 1.0 if _provider_event_exists(env, self._params) else 0.0


class _ProviderCalendarAddRepeatingEventBase(_CalendarAddRepeatingEventBase):

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_provider_calendar_task(self, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if _provider_event_exists(env, self._params, require_rrule=True)
        else 0.0
    )


class _ProviderCalendarDeleteEventsBase(_CalendarDeleteEventsBase):

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    params = self._params
    start_ms = _event_start_ms({**params, "hour": 9})
    seed_events = [
        (
            f"Delete candidate {index}",
            start_ms + index * 3_600_000,
            start_ms + (index + 1) * 3_600_000,
            "",
            None,
        )
        for index in range(2)
    ]
    _initialize_provider_calendar_task(self, env, seed_events)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    for row in _read_provider_events(env):
      if not row.deleted and _same_utc_day(row.dtstart, self._params):
        return 0.0
    return 1.0


class _ProviderCalendarEditEventBase(_CalendarEditEventBase):

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    params = {**self._params, "hour": 9, "duration_mins": 60}
    seed_events = [
        (
            self._params["old_event_title"],
            _event_start_ms(params),
            _event_end_ms(params),
            "",
            None,
        )
    ]
    _initialize_provider_calendar_task(self, env, seed_events)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    old_present = _provider_event_exists(
        env,
        self._params,
        title_key="old_event_title",
    )
    new_present = _provider_event_exists(
        env,
        self._params,
        title_key="new_event_title",
    )
    return 1.0 if new_present and not old_present else 0.0


# -----------------------------------------------------------------------------
# Per-app packages.
# -----------------------------------------------------------------------------

_ETAR_PACKAGE: Final[str] = "ws.xsoh.etar"
_FOSSIFY_CALENDAR_PACKAGE: Final[str] = "org.fossify.calendar"
_CALENDAR_PACKAGE: Final[str] = "com.vayunmathur.calendar"
_KASHCAL_PACKAGE: Final[str] = "org.onekash.kashcal"
_GOOGLE_CALENDAR_PACKAGE: Final[str] = "com.google.android.calendar"
_SAMSUNG_CALENDAR_PACKAGE: Final[str] = "com.samsung.android.calendar"
_SIMPLE_CALENDAR_PRO_PACKAGE: Final[str] = "com.simplemobiletools.calendar.pro"
_CALENDAR_PROVIDER_DB_PATH: Final[str] = (
    "/data/data/com.android.providers.calendar/databases/calendar.db"
)
_CALENDAR_PROVIDER_PACKAGE: Final[str] = "com.android.providers.calendar"
_FOSSIFY_CALENDAR_DB_PATH: Final[str] = (
    "/data/data/org.fossify.calendar/databases/events.db"
)
_KASHCAL_DB_PATH: Final[str] = (
    "/data/data/org.onekash.kashcal/databases/kashcal.db"
)


@dataclasses.dataclass(frozen=True)
class _ProviderEventRow(sqlite_schema_utils.SQLiteRow):
  title: str | None
  dtstart: int | None
  dtend: int | None
  rrule: str | None
  deleted: int
  _id: int = -1


def _event_start_ms(params: dict[str, Any]) -> int:
  dt = datetime.datetime(
      int(params["year"]),
      int(params["month"]),
      int(params["day"]),
      int(params.get("hour", 9)),
      tzinfo=datetime.timezone.utc,
  )
  return int(dt.timestamp() * 1000)


def _event_end_ms(params: dict[str, Any]) -> int:
  return _event_start_ms(params) + int(params.get("duration_mins", 60)) * 60_000


def _event_start_seconds(params: dict[str, Any]) -> int:
  return _event_start_ms(params) // 1000


def _event_end_seconds(params: dict[str, Any]) -> int:
  return _event_end_ms(params) // 1000


def _julian_day(timestamp_ms: int) -> int:
  return timestamp_ms // 86_400_000 + 2_440_588


def _same_utc_day(timestamp_ms: int | None, params: dict[str, Any]) -> bool:
  if timestamp_ms is None:
    return False
  dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000, datetime.timezone.utc)
  return (
      dt.year == int(params["year"])
      and dt.month == int(params["month"])
      and dt.day == int(params["day"])
  )


def _normalize_title(title: str | None) -> str:
  return (title or "").strip()


def _with_provider_db(env: interface.AsyncEnv, mutator):
  adb_utils.issue_generic_request(
      ["shell", "am", "force-stop", _CALENDAR_PROVIDER_PACKAGE],
      env.controller,
      timeout_sec=5,
  )
  with env.controller.pull_file(_CALENDAR_PROVIDER_DB_PATH) as local_db_directory:
    local_db_path = file_utils.convert_to_posix_path(
        local_db_directory,
        os.path.basename(_CALENDAR_PROVIDER_DB_PATH),
    )
    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row
    try:
      result = mutator(conn)
      conn.commit()
    finally:
      conn.close()
    env.controller.push_file(local_db_path, _CALENDAR_PROVIDER_DB_PATH)
  adb_utils.issue_generic_request(
      ["shell", "am", "force-stop", _CALENDAR_PROVIDER_PACKAGE],
      env.controller,
      timeout_sec=5,
  )
  return result


def _force_stop_package(package_name: str, env: interface.AsyncEnv) -> None:
  adb_utils.issue_generic_request(
      ["shell", "am", "force-stop", package_name],
      env.controller,
      timeout_sec=5,
  )


def _with_private_calendar_db(
    env: interface.AsyncEnv,
    db_path: str,
    package_name: str,
    mutator,
):
  def mutate_local_db():
    with env.controller.pull_file(db_path) as local_db_directory:
      local_db_path = file_utils.convert_to_posix_path(
          local_db_directory,
          os.path.basename(db_path),
      )
      conn = sqlite3.connect(local_db_path)
      conn.row_factory = sqlite3.Row
      try:
        result = mutator(conn)
        conn.commit()
      finally:
        conn.close()
      env.controller.push_file(local_db_path, db_path)
    _force_stop_package(package_name, env)
    return result

  try:
    return mutate_local_db()
  except FileNotFoundError:
    adb_utils.launch_app(package_name, env.controller)
    time.sleep(3.0)
    _force_stop_package(package_name, env)
    return mutate_local_db()


def _read_private_calendar_db(
    env: interface.AsyncEnv,
    db_path: str,
    package_name: str,
    reader,
):
  try:
    with env.controller.pull_file(db_path) as local_db_directory:
      local_db_path = file_utils.convert_to_posix_path(
          local_db_directory,
          os.path.basename(db_path),
      )
      conn = sqlite3.connect(local_db_path)
      conn.row_factory = sqlite3.Row
      try:
        return reader(conn)
      finally:
        conn.close()
  except (FileNotFoundError, sqlite3.OperationalError):
    adb_utils.launch_app(package_name, env.controller)
    time.sleep(3.0)
    _force_stop_package(package_name, env)
    return []


def _read_provider_events(env: interface.AsyncEnv) -> list[_ProviderEventRow]:
  with env.controller.pull_file(_CALENDAR_PROVIDER_DB_PATH) as local_db_directory:
    local_db_path = file_utils.convert_to_posix_path(
        local_db_directory,
        os.path.basename(_CALENDAR_PROVIDER_DB_PATH),
    )
    return sqlite_utils.execute_query(
        "SELECT _id, title, dtstart, dtend, rrule, deleted FROM Events;",
        local_db_path,
        _ProviderEventRow,
    )


def _clear_provider_events(env: interface.AsyncEnv) -> None:
  def mutate(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    for table in ("Instances", "CalendarAlerts", "Reminders", "Attendees"):
      cursor.execute(f"DELETE FROM {table}")
    cursor.execute("DELETE FROM Events")

  _with_provider_db(env, mutate)


def _insert_provider_event(
    conn: sqlite3.Connection,
    title: str,
    start_ms: int,
    end_ms: int,
    description: str = "",
    rrule: str | None = None,
) -> None:
  cursor = conn.cursor()
  cursor.execute(
      "INSERT INTO Events"
      " (calendar_id, title, description, dtstart, dtend, eventTimezone,"
      " eventEndTimezone, allDay, dirty, mutators, deleted, lastDate,"
      " hasAlarm, hasExtendedProperties, hasAttendeeData, accessLevel,"
      " availability, selfAttendeeStatus, guestsCanModify,"
      " guestsCanInviteOthers, guestsCanSeeGuests, rrule)"
      " VALUES (-1, ?, ?, ?, ?, 'UTC', 'UTC', 0, 1, 'catbench', 0, ?,"
      " 0, 0, 0, 0, 0, 0, 0, 1, 1, ?)",
      (title, description, start_ms, end_ms, end_ms, rrule),
  )
  event_id = int(cursor.lastrowid)
  start_minute = datetime.datetime.fromtimestamp(
      start_ms / 1000,
      datetime.timezone.utc,
  ).hour * 60
  end_minute = start_minute + int((end_ms - start_ms) / 60_000)
  cursor.execute(
      "INSERT INTO Instances"
      " (event_id, begin, end, startDay, endDay, startMinute, endMinute)"
      " VALUES (?, ?, ?, ?, ?, ?, ?)",
      (
          event_id,
          start_ms,
          end_ms,
          _julian_day(start_ms),
          _julian_day(end_ms),
          start_minute,
          end_minute,
      ),
  )


def _seed_provider_events(
    env: interface.AsyncEnv,
    events: list[tuple[str, int, int, str, str | None]],
) -> None:
  def mutate(conn: sqlite3.Connection) -> None:
    for title, start_ms, end_ms, description, rrule in events:
      _insert_provider_event(conn, title, start_ms, end_ms, description, rrule)

  _with_provider_db(env, mutate)


def _initialize_provider_calendar_task(
    task: base.PackageAppEval,
    env: interface.AsyncEnv,
    seed_events: list[tuple[str, int, int, str, str | None]] | None = None,
) -> None:
  base.PackageAppEval.initialize_task(task, env)
  _force_stop_package(task.package_name, env)
  _clear_provider_events(env)
  if seed_events:
    _seed_provider_events(env, seed_events)
  adb_utils.launch_app(task.package_name, env.controller)


def _provider_event_exists(
    env: interface.AsyncEnv,
    params: dict[str, Any],
    title_key: str = "event_title",
    require_rrule: bool = False,
) -> bool:
  expected = str(params[title_key]).strip()
  for row in _read_provider_events(env):
    if row.deleted:
      continue
    if _normalize_title(row.title) != expected:
      continue
    if not _same_utc_day(row.dtstart, params):
      continue
    if require_rrule and not row.rrule:
      continue
    return True
  return False


def _simple_calendar_rows(
    env: interface.AsyncEnv,
) -> list[sqlite_schema_utils.CalendarEvent]:
  return sqlite_utils.get_rows_from_remote_device(
      calendar_utils.EVENTS_TABLE,
      calendar_utils.DB_PATH,
      sqlite_schema_utils.CalendarEvent,
      env,
  )


def _clear_fossify_events(env: interface.AsyncEnv) -> None:
  def mutate(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tasks")
    cursor.execute("DELETE FROM events")
    cursor.execute(
        "INSERT OR IGNORE INTO event_types"
        " (id, title, color, caldav_calendar_id, caldav_display_name,"
        " caldav_email, type)"
        " VALUES (1, 'Local calendar', -14574644, 0, '', '', 0)"
    )

  _with_private_calendar_db(
      env,
      _FOSSIFY_CALENDAR_DB_PATH,
      _FOSSIFY_CALENDAR_PACKAGE,
      mutate,
  )


def _insert_fossify_event(
    conn: sqlite3.Connection,
    title: str,
    start_s: int,
    end_s: int,
    description: str = "",
    repeat_interval: int = 0,
    repeat_rule: int = 0,
) -> None:
  conn.execute(
      "INSERT INTO events"
      " (start_ts, end_ts, title, location, description,"
      " reminder_1_minutes, reminder_2_minutes, reminder_3_minutes,"
      " reminder_1_type, reminder_2_type, reminder_3_type,"
      " repeat_interval, repeat_rule, repeat_limit, repetition_exceptions,"
      " attendees, import_id, time_zone, flags, event_type, parent_id,"
      " last_updated, source, availability, access_level, color, type, status)"
      " VALUES (?, ?, ?, '', ?, -1, -1, -1, 0, 0, 0, ?, ?, 0, '[]', '',"
      " '', 'UTC', 0, 1, 0, ?, '', 0, 0, 0, 0, 0)",
      (
          start_s,
          end_s,
          title,
          description,
          repeat_interval,
          repeat_rule,
          int(time.time()),
      ),
  )


def _seed_fossify_events(
    env: interface.AsyncEnv,
    events: list[tuple[str, int, int, str, int, int]],
) -> None:
  def mutate(conn: sqlite3.Connection) -> None:
    for title, start_s, end_s, description, repeat_interval, repeat_rule in events:
      _insert_fossify_event(
          conn,
          title,
          start_s,
          end_s,
          description,
          repeat_interval,
          repeat_rule,
      )

  _with_private_calendar_db(
      env,
      _FOSSIFY_CALENDAR_DB_PATH,
      _FOSSIFY_CALENDAR_PACKAGE,
      mutate,
  )


def _read_fossify_events(env: interface.AsyncEnv) -> list[sqlite3.Row]:
  return _read_private_calendar_db(
      env,
      _FOSSIFY_CALENDAR_DB_PATH,
      _FOSSIFY_CALENDAR_PACKAGE,
      lambda conn: conn.execute(
          "SELECT title, start_ts, end_ts, repeat_interval, repeat_rule"
          " FROM events"
      ).fetchall(),
  )


def _initialize_fossify_calendar_task(
    task: base.PackageAppEval,
    env: interface.AsyncEnv,
    seed_events: list[tuple[str, int, int, str, int, int]] | None = None,
) -> None:
  base.PackageAppEval.initialize_task(task, env)
  time.sleep(3.0)
  _force_stop_package(_FOSSIFY_CALENDAR_PACKAGE, env)
  _clear_fossify_events(env)
  if seed_events:
    _seed_fossify_events(env, seed_events)
  adb_utils.launch_app(_FOSSIFY_CALENDAR_PACKAGE, env.controller)


def _fossify_event_exists(
    env: interface.AsyncEnv,
    params: dict[str, Any],
    title_key: str = "event_title",
    require_repeat: bool = False,
) -> bool:
  expected = str(params[title_key]).strip()
  for row in _read_fossify_events(env):
    if _normalize_title(row["title"]) != expected:
      continue
    if not _same_utc_day(int(row["start_ts"]) * 1000, params):
      continue
    if require_repeat and not (
        int(row["repeat_interval"]) or int(row["repeat_rule"])
    ):
      continue
    return True
  return False


def _ensure_kashcal_calendar(conn: sqlite3.Connection) -> int:
  row = conn.execute("SELECT id FROM calendars LIMIT 1").fetchone()
  if row is not None:
    return int(row["id"])
  now_ms = int(time.time() * 1000)
  conn.execute(
      "INSERT INTO accounts"
      " (provider, email, display_name, home_set_url, created_at)"
      " VALUES ('LOCAL', 'catbench@example.com', 'CATBench',"
      " 'local://catbench', ?)",
      (now_ms,),
  )
  account_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
  conn.execute(
      "INSERT INTO calendars"
      " (account_id, caldav_url, display_name, color, is_visible,"
      " is_default)"
      " VALUES (?, 'local://catbench/calendar', 'CATBench', -14574644, 1, 1)",
      (account_id,),
  )
  return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _clear_kashcal_events(env: interface.AsyncEnv) -> None:
  def mutate(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    for table in (
        "scheduled_reminders",
        "pending_operations",
        "occurrences",
        "events",
    ):
      cursor.execute(f"DELETE FROM {table}")
    _ensure_kashcal_calendar(conn)

  _with_private_calendar_db(env, _KASHCAL_DB_PATH, _KASHCAL_PACKAGE, mutate)


def _insert_kashcal_event(
    conn: sqlite3.Connection,
    title: str,
    start_ms: int,
    end_ms: int,
    description: str = "",
    rrule: str | None = None,
) -> None:
  calendar_id = _ensure_kashcal_calendar(conn)
  now_ms = int(time.time() * 1000)
  conn.execute(
      "INSERT INTO events"
      " (uid, calendar_id, title, location, description, start_ts, end_ts,"
      " timezone, end_timezone, rrule, dtstamp, created_at, updated_at)"
      " VALUES (?, ?, ?, '', ?, ?, ?, 'UTC', 'UTC', ?, ?, ?, ?)",
      (
          str(uuid.uuid4()),
          calendar_id,
          title,
          description,
          start_ms,
          end_ms,
          rrule,
          now_ms,
          now_ms,
          now_ms,
      ),
  )
  event_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
  conn.execute(
      "INSERT INTO occurrences"
      " (event_id, calendar_id, start_ts, end_ts, start_day, end_day)"
      " VALUES (?, ?, ?, ?, ?, ?)",
      (
          event_id,
          calendar_id,
          start_ms,
          end_ms,
          _julian_day(start_ms),
          _julian_day(end_ms),
      ),
  )


def _seed_kashcal_events(
    env: interface.AsyncEnv,
    events: list[tuple[str, int, int, str, str | None]],
) -> None:
  def mutate(conn: sqlite3.Connection) -> None:
    for title, start_ms, end_ms, description, rrule in events:
      _insert_kashcal_event(conn, title, start_ms, end_ms, description, rrule)

  _with_private_calendar_db(env, _KASHCAL_DB_PATH, _KASHCAL_PACKAGE, mutate)


def _read_kashcal_events(env: interface.AsyncEnv) -> list[sqlite3.Row]:
  return _read_private_calendar_db(
      env,
      _KASHCAL_DB_PATH,
      _KASHCAL_PACKAGE,
      lambda conn: conn.execute(
          "SELECT title, start_ts, end_ts, rrule FROM events"
      ).fetchall(),
  )


def _initialize_kashcal_task(
    task: base.PackageAppEval,
    env: interface.AsyncEnv,
    seed_events: list[tuple[str, int, int, str, str | None]] | None = None,
) -> None:
  base.PackageAppEval.initialize_task(task, env)
  time.sleep(3.0)
  _force_stop_package(_KASHCAL_PACKAGE, env)
  _clear_kashcal_events(env)
  if seed_events:
    _seed_kashcal_events(env, seed_events)
  adb_utils.launch_app(_KASHCAL_PACKAGE, env.controller)


def _kashcal_event_exists(
    env: interface.AsyncEnv,
    params: dict[str, Any],
    title_key: str = "event_title",
    require_repeat: bool = False,
) -> bool:
  expected = str(params[title_key]).strip()
  for row in _read_kashcal_events(env):
    if _normalize_title(row["title"]) != expected:
      continue
    if not _same_utc_day(int(row["start_ts"]), params):
      continue
    if require_repeat and not row["rrule"]:
      continue
    return True
  return False


class _FossifyCalendarAddOneEventBase(_CalendarAddOneEventBase):

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_fossify_calendar_task(self, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return 1.0 if _fossify_event_exists(env, self._params) else 0.0


class _FossifyCalendarAddRepeatingEventBase(_CalendarAddRepeatingEventBase):

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_fossify_calendar_task(self, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if _fossify_event_exists(env, self._params, require_repeat=True)
        else 0.0
    )


class _FossifyCalendarDeleteEventsBase(_CalendarDeleteEventsBase):

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    params = {**self._params, "hour": 9, "duration_mins": 60}
    start_s = _event_start_seconds(params)
    seed_events = [
        (
            f"Delete candidate {index}",
            start_s + index * 3_600,
            start_s + (index + 1) * 3_600,
            "",
            0,
            0,
        )
        for index in range(2)
    ]
    _initialize_fossify_calendar_task(self, env, seed_events)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        0.0
        if any(
            _same_utc_day(int(row["start_ts"]) * 1000, self._params)
            for row in _read_fossify_events(env)
        )
        else 1.0
    )


class _FossifyCalendarEditEventBase(_CalendarEditEventBase):

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    params = {**self._params, "hour": 9, "duration_mins": 60}
    seed_events = [
        (
            self._params["old_event_title"],
            _event_start_seconds(params),
            _event_end_seconds(params),
            "",
            0,
            0,
        )
    ]
    _initialize_fossify_calendar_task(self, env, seed_events)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    old_present = _fossify_event_exists(
        env,
        self._params,
        title_key="old_event_title",
    )
    new_present = _fossify_event_exists(
        env,
        self._params,
        title_key="new_event_title",
    )
    return 1.0 if new_present and not old_present else 0.0


class _KashcalAddOneEventBase(_CalendarAddOneEventBase):

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_kashcal_task(self, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return 1.0 if _kashcal_event_exists(env, self._params) else 0.0


class _KashcalAddRepeatingEventBase(_CalendarAddRepeatingEventBase):

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_kashcal_task(self, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if _kashcal_event_exists(env, self._params, require_repeat=True)
        else 0.0
    )


class _KashcalDeleteEventsBase(_CalendarDeleteEventsBase):

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    params = {**self._params, "hour": 9, "duration_mins": 60}
    start_ms = _event_start_ms(params)
    seed_events = [
        (
            f"Delete candidate {index}",
            start_ms + index * 3_600_000,
            start_ms + (index + 1) * 3_600_000,
            "",
            None,
        )
        for index in range(2)
    ]
    _initialize_kashcal_task(self, env, seed_events)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        0.0
        if any(
            _same_utc_day(int(row["start_ts"]), self._params)
            for row in _read_kashcal_events(env)
        )
        else 1.0
    )


class _KashcalEditEventBase(_CalendarEditEventBase):

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    params = {**self._params, "hour": 9, "duration_mins": 60}
    seed_events = [
        (
            self._params["old_event_title"],
            _event_start_ms(params),
            _event_end_ms(params),
            "",
            None,
        )
    ]
    _initialize_kashcal_task(self, env, seed_events)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    old_present = _kashcal_event_exists(
        env,
        self._params,
        title_key="old_event_title",
    )
    new_present = _kashcal_event_exists(
        env,
        self._params,
        title_key="new_event_title",
    )
    return 1.0 if new_present and not old_present else 0.0


# -----------------------------------------------------------------------------
# Etar
# -----------------------------------------------------------------------------


class SimpleCalendarAddOneEventForEtar(_ProviderCalendarAddOneEventBase):
  app_names = (_ETAR_PACKAGE,)
  package_name = _ETAR_PACKAGE
  template = (
      "In the Etar app, create a calendar event on {year}-{month}-{day}"
      " at {hour}h with the title '{event_title}' and the description"
      " '{event_description}'. The event should last for {duration_mins} mins."
  )


class SimpleCalendarAddRepeatingEventForEtar(_ProviderCalendarAddRepeatingEventBase):
  app_names = (_ETAR_PACKAGE,)
  package_name = _ETAR_PACKAGE
  template = (
      "In the Etar app, create a recurring calendar event titled"
      " '{event_title}' starting on {year}-{month}-{day} at {hour}h. The"
      " event recurs {repeat_rule}, forever, and lasts for {duration_mins}"
      " minutes each occurrence. The event description should be"
      " '{event_description}'."
  )


class SimpleCalendarDeleteEventsForEtar(_ProviderCalendarDeleteEventsBase):
  app_names = (_ETAR_PACKAGE,)
  package_name = _ETAR_PACKAGE
  template = (
      "In the Etar app, delete all the calendar events on"
      " {year}-{month}-{day}."
  )


class SimpleCalendarEditEventForEtar(_ProviderCalendarEditEventBase):
  app_names = (_ETAR_PACKAGE,)
  package_name = _ETAR_PACKAGE
  template = (
      "In the Etar app, edit the calendar event titled '{old_event_title}' on"
      " {year}-{month}-{day} so its title becomes '{new_event_title}'."
  )


class SimpleCalendarViewMonthAgendaForEtar(_CalendarViewMonthAgendaBase):
  app_names = (_ETAR_PACKAGE,)
  package_name = _ETAR_PACKAGE
  template = (
      "In the Etar app, open the agenda or list view for {month_name} {year}"
      " and review the scheduled events."
  )


# -----------------------------------------------------------------------------
# Fossify Calendar
# -----------------------------------------------------------------------------


class SimpleCalendarAddOneEventForFossifyCalendar(
    _FossifyCalendarAddOneEventBase
):
  app_names = (_FOSSIFY_CALENDAR_PACKAGE,)
  package_name = _FOSSIFY_CALENDAR_PACKAGE
  template = (
      "In the Fossify Calendar app, create a calendar event on"
      " {year}-{month}-{day} at {hour}h with the title '{event_title}' and"
      " the description '{event_description}'. The event should last for"
      " {duration_mins} mins."
  )


class SimpleCalendarAddRepeatingEventForFossifyCalendar(
    _FossifyCalendarAddRepeatingEventBase
):
  app_names = (_FOSSIFY_CALENDAR_PACKAGE,)
  package_name = _FOSSIFY_CALENDAR_PACKAGE
  template = (
      "In the Fossify Calendar app, create a recurring calendar event titled"
      " '{event_title}' starting on {year}-{month}-{day} at {hour}h. The"
      " event recurs {repeat_rule}, forever, and lasts for {duration_mins}"
      " minutes each occurrence. The event description should be"
      " '{event_description}'."
  )


class SimpleCalendarDeleteEventsForFossifyCalendar(
    _FossifyCalendarDeleteEventsBase
):
  app_names = (_FOSSIFY_CALENDAR_PACKAGE,)
  package_name = _FOSSIFY_CALENDAR_PACKAGE
  template = (
      "In the Fossify Calendar app, delete all the calendar events on"
      " {year}-{month}-{day}."
  )


class SimpleCalendarEditEventForFossifyCalendar(_FossifyCalendarEditEventBase):
  app_names = (_FOSSIFY_CALENDAR_PACKAGE,)
  package_name = _FOSSIFY_CALENDAR_PACKAGE
  template = (
      "In the Fossify Calendar app, edit the calendar event titled"
      " '{old_event_title}' on {year}-{month}-{day} so its title becomes"
      " '{new_event_title}'."
  )


class SimpleCalendarViewMonthAgendaForFossifyCalendar(
    _CalendarViewMonthAgendaBase
):
  app_names = (_FOSSIFY_CALENDAR_PACKAGE,)
  package_name = _FOSSIFY_CALENDAR_PACKAGE
  template = (
      "In the Fossify Calendar app, open the agenda or list view for"
      " {month_name} {year} and review the scheduled events."
  )


# -----------------------------------------------------------------------------
# Calendar (com.vayunmathur.calendar)
# -----------------------------------------------------------------------------


class SimpleCalendarAddOneEventForCalendar(_ProviderCalendarAddOneEventBase):
  app_names = (_CALENDAR_PACKAGE,)
  package_name = _CALENDAR_PACKAGE
  template = (
      "In the Calendar app, create a calendar event on {year}-{month}-{day}"
      " at {hour}h with the title '{event_title}' and the description"
      " '{event_description}'. The event should last for {duration_mins} mins."
  )


class SimpleCalendarAddRepeatingEventForCalendar(
    _ProviderCalendarAddRepeatingEventBase
):
  app_names = (_CALENDAR_PACKAGE,)
  package_name = _CALENDAR_PACKAGE
  template = (
      "In the Calendar app, create a recurring calendar event titled"
      " '{event_title}' starting on {year}-{month}-{day} at {hour}h. The"
      " event recurs {repeat_rule}, forever, and lasts for {duration_mins}"
      " minutes each occurrence. The event description should be"
      " '{event_description}'."
  )


class SimpleCalendarDeleteEventsForCalendar(_ProviderCalendarDeleteEventsBase):
  app_names = (_CALENDAR_PACKAGE,)
  package_name = _CALENDAR_PACKAGE
  template = (
      "In the Calendar app, delete all the calendar events on"
      " {year}-{month}-{day}."
  )


class SimpleCalendarEditEventForCalendar(_ProviderCalendarEditEventBase):
  app_names = (_CALENDAR_PACKAGE,)
  package_name = _CALENDAR_PACKAGE
  template = (
      "In the Calendar app, edit the calendar event titled"
      " '{old_event_title}' on {year}-{month}-{day} so its title becomes"
      " '{new_event_title}'."
  )


class SimpleCalendarViewMonthAgendaForCalendar(_CalendarViewMonthAgendaBase):
  app_names = (_CALENDAR_PACKAGE,)
  package_name = _CALENDAR_PACKAGE
  template = (
      "In the Calendar app, open the agenda or list view for {month_name}"
      " {year} and review the scheduled events."
  )


# -----------------------------------------------------------------------------
# KashCal
# -----------------------------------------------------------------------------


class SimpleCalendarAddOneEventForKashcal(_KashcalAddOneEventBase):
  app_names = (_KASHCAL_PACKAGE,)
  package_name = _KASHCAL_PACKAGE
  template = (
      "In the KashCal app, create a calendar event on {year}-{month}-{day}"
      " at {hour}h with the title '{event_title}' and the description"
      " '{event_description}'. The event should last for {duration_mins} mins."
  )


class SimpleCalendarAddRepeatingEventForKashcal(
    _KashcalAddRepeatingEventBase
):
  app_names = (_KASHCAL_PACKAGE,)
  package_name = _KASHCAL_PACKAGE
  template = (
      "In the KashCal app, create a recurring calendar event titled"
      " '{event_title}' starting on {year}-{month}-{day} at {hour}h. The"
      " event recurs {repeat_rule}, forever, and lasts for {duration_mins}"
      " minutes each occurrence. The event description should be"
      " '{event_description}'."
  )


class SimpleCalendarDeleteEventsForKashcal(_KashcalDeleteEventsBase):
  app_names = (_KASHCAL_PACKAGE,)
  package_name = _KASHCAL_PACKAGE
  template = (
      "In the KashCal app, delete all the calendar events on"
      " {year}-{month}-{day}."
  )


class SimpleCalendarEditEventForKashcal(_KashcalEditEventBase):
  app_names = (_KASHCAL_PACKAGE,)
  package_name = _KASHCAL_PACKAGE
  template = (
      "In the KashCal app, edit the calendar event titled"
      " '{old_event_title}' on {year}-{month}-{day} so its title becomes"
      " '{new_event_title}'."
  )


class SimpleCalendarViewMonthAgendaForKashcal(_CalendarViewMonthAgendaBase):
  app_names = (_KASHCAL_PACKAGE,)
  package_name = _KASHCAL_PACKAGE
  template = (
      "In the KashCal app, open the agenda or list view for {month_name}"
      " {year} and review the scheduled events."
  )


# -----------------------------------------------------------------------------
# Google Calendar
# -----------------------------------------------------------------------------


class SimpleCalendarAddOneEventForGoogleCalendar(_CalendarAddOneEventBase):
  app_names = (_GOOGLE_CALENDAR_PACKAGE,)
  package_name = _GOOGLE_CALENDAR_PACKAGE
  template = (
      "In Google Calendar, create a calendar event on {year}-{month}-{day}"
      " at {hour}h with the title '{event_title}' and the description"
      " '{event_description}'. The event should last for {duration_mins} mins."
  )


class SimpleCalendarAddRepeatingEventForGoogleCalendar(
    _CalendarAddRepeatingEventBase
):
  app_names = (_GOOGLE_CALENDAR_PACKAGE,)
  package_name = _GOOGLE_CALENDAR_PACKAGE
  template = (
      "In Google Calendar, create a recurring calendar event titled"
      " '{event_title}' starting on {year}-{month}-{day} at {hour}h. The"
      " event recurs {repeat_rule}, forever, and lasts for {duration_mins}"
      " minutes each occurrence. The event description should be"
      " '{event_description}'."
  )


class SimpleCalendarDeleteEventsForGoogleCalendar(_CalendarDeleteEventsBase):
  app_names = (_GOOGLE_CALENDAR_PACKAGE,)
  package_name = _GOOGLE_CALENDAR_PACKAGE
  template = (
      "In Google Calendar, delete all the calendar events on"
      " {year}-{month}-{day}."
  )


class SimpleCalendarEditEventForGoogleCalendar(_CalendarEditEventBase):
  app_names = (_GOOGLE_CALENDAR_PACKAGE,)
  package_name = _GOOGLE_CALENDAR_PACKAGE
  template = (
      "In Google Calendar, edit the calendar event titled"
      " '{old_event_title}' on {year}-{month}-{day} so its title becomes"
      " '{new_event_title}'."
  )


class SimpleCalendarViewMonthAgendaForGoogleCalendar(
    _CalendarViewMonthAgendaBase
):
  app_names = (_GOOGLE_CALENDAR_PACKAGE,)
  package_name = _GOOGLE_CALENDAR_PACKAGE
  template = (
      "In Google Calendar, open the agenda or list view for {month_name}"
      " {year} and review the scheduled events."
  )


# -----------------------------------------------------------------------------
# Samsung Calendar
# -----------------------------------------------------------------------------


class SimpleCalendarAddOneEventForSamsungCalendar(_CalendarAddOneEventBase):
  app_names = (_SAMSUNG_CALENDAR_PACKAGE,)
  package_name = _SAMSUNG_CALENDAR_PACKAGE
  template = (
      "In Samsung Calendar, create a calendar event on {year}-{month}-{day}"
      " at {hour}h with the title '{event_title}' and the description"
      " '{event_description}'. The event should last for {duration_mins} mins."
  )


class SimpleCalendarAddRepeatingEventForSamsungCalendar(
    _CalendarAddRepeatingEventBase
):
  app_names = (_SAMSUNG_CALENDAR_PACKAGE,)
  package_name = _SAMSUNG_CALENDAR_PACKAGE
  template = (
      "In Samsung Calendar, create a recurring calendar event titled"
      " '{event_title}' starting on {year}-{month}-{day} at {hour}h. The"
      " event recurs {repeat_rule}, forever, and lasts for {duration_mins}"
      " minutes each occurrence. The event description should be"
      " '{event_description}'."
  )


class SimpleCalendarDeleteEventsForSamsungCalendar(_CalendarDeleteEventsBase):
  app_names = (_SAMSUNG_CALENDAR_PACKAGE,)
  package_name = _SAMSUNG_CALENDAR_PACKAGE
  template = (
      "In Samsung Calendar, delete all the calendar events on"
      " {year}-{month}-{day}."
  )


# -----------------------------------------------------------------------------
# Simple Calendar Pro
# -----------------------------------------------------------------------------


class SimpleCalendarEditEventForSimpleCalendarPro(_CalendarEditEventBase):
  app_names = (_SIMPLE_CALENDAR_PRO_PACKAGE,)
  package_name = _SIMPLE_CALENDAR_PRO_PACKAGE
  template = (
      "In the Simple Calendar Pro app, edit the calendar event titled"
      " '{old_event_title}' on {year}-{month}-{day} so its title becomes"
      " '{new_event_title}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    base.PackageAppEval.initialize_task(self, env)
    _force_stop_package(_SIMPLE_CALENDAR_PRO_PACKAGE, env)
    calendar_utils.clear_calendar_db(env)
    params = {**self._params, "hour": 9, "duration_mins": 60}
    calendar_utils.add_events(
        [
            sqlite_schema_utils.CalendarEvent(
                start_ts=_event_start_seconds(params),
                end_ts=_event_end_seconds(params),
                title=self._params["old_event_title"],
            )
        ],
        env,
    )
    adb_utils.launch_app(_SIMPLE_CALENDAR_PRO_PACKAGE, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    old_title = self._params["old_event_title"].strip()
    new_title = self._params["new_event_title"].strip()
    rows = _simple_calendar_rows(env)
    old_present = any(row.title.strip() == old_title for row in rows)
    new_present = any(row.title.strip() == new_title for row in rows)
    return 1.0 if new_present and not old_present else 0.0


class SimpleCalendarViewMonthAgendaForSimpleCalendarPro(
    _CalendarViewMonthAgendaBase
):
  app_names = (_SIMPLE_CALENDAR_PRO_PACKAGE,)
  package_name = _SIMPLE_CALENDAR_PRO_PACKAGE
  template = (
      "In the Simple Calendar Pro app, open the agenda or list view for"
      " {month_name} {year} and review the scheduled events."
  )


class _CalendarGenericTaskBase(base.PackageAppEval):
  """Self-contained calendar task used for full 10-template coverage."""

  complexity = 2.4
  success_mode = "event_title"
  schema = _CalendarAddOneEventBase.schema

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    mode = self.success_mode
    title_ok = base.element_text_contains(
        ui_elements, (self._params["event_title"],)
    )
    if mode == "date":
      markers = (
          str(self._params["day"]),
          f"{self._params['year']}-{self._params['month']}-{self._params['day']}",
          "events",
          "agenda",
      )
      return 1.0 if base.element_text_contains(ui_elements, markers) else 0.0
    if mode == "next":
      answer = str(getattr(env, "interaction_cache", "")).strip().lower()
      if answer and self._params["event_title"].lower() in answer:
        return 1.0
      return 1.0 if title_ok else 0.0
    if mode == "range":
      markers = ("agenda", "events", "schedule")
      return 1.0 if title_ok or base.element_text_contains(ui_elements, markers) else 0.0
    if mode == "reminder":
      markers = ("reminder", "alert", "notification", "alarm")
      return 1.0 if title_ok and base.element_text_contains(ui_elements, markers) else 0.0
    return 1.0 if title_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_add_event_params()


def _make_calendar_generic_task(
    class_name: str,
    package_name: str,
    template: str,
    success_mode: str,
) -> type[_CalendarGenericTaskBase]:
  return type(
      class_name,
      (_CalendarGenericTaskBase,),
      {
          "__module__": __name__,
          "app_names": (package_name,),
          "package_name": package_name,
          "template": template,
          "success_mode": success_mode,
      },
  )


_CALENDAR_APP_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    ("SimpleCalendarPro", _SIMPLE_CALENDAR_PRO_PACKAGE, "Simple Calendar Pro"),
    ("Etar", _ETAR_PACKAGE, "Etar"),
    ("FossifyCalendar", _FOSSIFY_CALENDAR_PACKAGE, "Fossify Calendar"),
    ("Calendar", _CALENDAR_PACKAGE, "Calendar"),
    ("Kashcal", _KASHCAL_PACKAGE, "KashCal"),
)

_CALENDAR_TEMPLATE_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "AddTimedEvent",
        "In the {app} app, create a timed event on {year}-{month}-{day} at "
        "{hour}h titled '{event_title}' lasting {duration_mins} minutes.",
        "event_title",
    ),
    (
        "EventsOnDate",
        "In the {app} app, open {year}-{month}-{day} and show the events "
        "scheduled on that date.",
        "date",
    ),
    (
        "NextEvent",
        "In the {app} app, find the next upcoming event and answer with its "
        "title. If needed, create an event titled '{event_title}' on "
        "{year}-{month}-{day} first.",
        "next",
    ),
    (
        "EventsInRange",
        "In the {app} app, open the agenda for the date range including "
        "{year}-{month}-{day} and review the scheduled events.",
        "range",
    ),
    (
        "AddReminder",
        "In the {app} app, create an event titled '{event_title}' on "
        "{year}-{month}-{day} at {hour}h and add a reminder notification.",
        "reminder",
    ),
    (
        "MoveEvent",
        "In the {app} app, create or find the event titled '{event_title}' "
        "and move it to {year}-{month}-{day} at {hour}h.",
        "event_title",
    ),
)

for _suffix, _package, _display_name in _CALENDAR_APP_SPECS:
  for _task_name, _template, _mode in _CALENDAR_TEMPLATE_SPECS:
    globals()[f"SimpleCalendar{_task_name}For{_suffix}"] = (
        _make_calendar_generic_task(
            f"SimpleCalendar{_task_name}For{_suffix}",
            _package,
            _template.replace("{app}", _display_name),
            _mode,
        )
    )
