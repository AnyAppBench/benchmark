"""Cross-app clock task ports for the app-generalization suite.

The canonical Google Clock tasks in AndroidWorld are:

  - ``ClockStopWatchPausedVerify``
  - ``ClockStopWatchRunning``
  - ``ClockTimerEntry``

Those three cover only the stopwatch/timer surfaces. To probe cross-app
generalization more thoroughly, this module also adds alarm-surface tasks
(``ClockCreateAlarm``, ``ClockEditAlarm``, ``ClockEnableAlarm``,
``ClockDeleteAlarm``), timer-control tasks, stopwatch running/reset tasks,
and world-clock creation.

Where a target app has a device-validated durable-state adapter, the evaluator
follows ``docs/tasks_guide.md`` and reads the app's real storage. The currently
qualified adapter covers Fossify Clock's ``alarms.db`` and timer ``app.db``.
The pinned Clock You APK contains a Room database whose ``alarms.time`` value
is milliseconds after local midnight and whose ``timeZones`` rows persist the
selected IANA zone.  Those encodings are used for exact alarm/world-clock
checks; transient timer and stopwatch states still use explicit UI checks.
"""

from __future__ import annotations

import random
import re
import shlex
import time
from typing import Any, Final

from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import representation_utils
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base as base,
)


_FOSSIFY_ALARMS_DB: Final[str] = (
    "/data/data/org.fossify.clock/databases/alarms.db"
)
_FOSSIFY_TIMERS_DB: Final[str] = (
    "/data/data/org.fossify.clock/databases/app.db"
)
_FOSSIFY_CLOCK_PACKAGE_NAME: Final[str] = "org.fossify.clock"
_CLOCK_YOU_DB: Final[str] = (
    "/data/data/com.bnyro.clock/databases/com.bnyro.clock"
)
_CLOCK_YOU_PACKAGE_NAME: Final[str] = "com.bnyro.clock"
_SQLITE_STATUS_MARKER: Final[str] = "__CATBENCH_CLOCK_SQLITE_STATUS__="


class ClockStorageError(RuntimeError):
  """A Clock adapter could not establish the health of app storage."""


class ClockStorageReadError(ClockStorageError):
  """A Clock verifier could not distinguish a read failure from absence."""


def _adb_shell(env: interface.AsyncEnv, cmd: str) -> str:
  out = adb_utils.issue_generic_request(["shell", cmd], env.controller)
  return out.generic.output.decode("utf-8", errors="ignore") if out else ""


def _adb_shell_or_empty(env: interface.AsyncEnv, cmd: str) -> str:
  """Run a shell command whose non-zero status is expected sometimes."""
  try:
    return _adb_shell(env, cmd)
  except Exception:  # pylint: disable=broad-except
    return ""


def _su_shell(env: interface.AsyncEnv, cmd: str) -> str:
  return _adb_shell(env, f"su 0 sh -c {shlex.quote(f'{cmd} || true')}")


def _sqlite_exec(env: interface.AsyncEnv, db_path: str, sql: str) -> str:
  """Execute SQLite without collapsing command failures to empty output."""
  sqlite_cmd = (
      f"sqlite3 {shlex.quote(db_path)} {shlex.quote(sql)} 2>&1"
  )
  checked_cmd = (
      f"{sqlite_cmd}; status=$?; "
      f"printf '\\n{_SQLITE_STATUS_MARKER}%s\\n' \"$status\""
  )
  try:
    raw = _adb_shell(env, f"su 0 sh -c {shlex.quote(checked_cmd)}")
  except Exception as error:  # pylint: disable=broad-except
    raise ClockStorageError(
        f"Clock SQLite command transport failed for {db_path!r}."
    ) from error

  normalized = raw.replace("\r\n", "\n")
  marker = f"\n{_SQLITE_STATUS_MARKER}"
  if marker not in normalized:
    raise ClockStorageError(
        f"Clock SQLite command returned no status for {db_path!r}."
    )
  output, status_text = normalized.rsplit(marker, 1)
  try:
    status = int(status_text.strip())
  except ValueError as error:
    raise ClockStorageError(
        f"Clock SQLite command returned an invalid status for {db_path!r}."
    ) from error
  if status != 0:
    detail = output.strip()[-1000:]
    raise ClockStorageError(
        f"Clock SQLite command failed for {db_path!r} with status {status}: "
        f"{detail}"
    )
  return output


def _sqlite_read(env: interface.AsyncEnv, db_path: str, sql: str) -> str:
  """Run a verifier read and preserve a typed read-health failure."""
  try:
    return _sqlite_exec(env, db_path, sql)
  except ClockStorageError as error:
    raise ClockStorageReadError(str(error)) from error


def _sqlite_count(env: interface.AsyncEnv, db_path: str, sql: str) -> int:
  """Return a scalar COUNT, rejecting malformed/empty native-state output."""
  output = _sqlite_read(env, db_path, sql)
  values = [
      int(line.strip())
      for line in output.splitlines()
      if line.strip().isdigit()
  ]
  if len(values) != 1:
    raise ClockStorageReadError(
        f"Clock SQLite COUNT read for {db_path!r} returned {output!r}."
    )
  return values[0]


def _sql_quote(value: str) -> str:
  return "'" + value.replace("'", "''") + "'"


def _alarm_minutes(hour_24: int, minute: int) -> int:
  return hour_24 * 60 + minute


def _uses_fossify_storage(package_name: str) -> bool:
  return package_name == _FOSSIFY_CLOCK_PACKAGE_NAME


def _uses_clock_you_storage(package_name: str) -> bool:
  return package_name == _CLOCK_YOU_PACKAGE_NAME


def _clock_you_has_table(env: interface.AsyncEnv, table_name: str) -> bool:
  out = _sqlite_read(
      env,
      _CLOCK_YOU_DB,
      (
          "SELECT name FROM sqlite_master WHERE type='table' AND name="
          f"{_sql_quote(table_name)};"
      ),
  )
  return table_name in {line.strip() for line in out.splitlines()}


def _clock_you_database_exists(env: interface.AsyncEnv) -> bool:
  """Check pre-launch Room initialization without opening a missing database."""
  command = (
      f"if [ -f {shlex.quote(_CLOCK_YOU_DB)} ]; then printf 1; "
      "else printf 0; fi"
  )
  try:
    output = _adb_shell(env, f"su 0 sh -c {shlex.quote(command)}")
  except Exception as error:  # pylint: disable=broad-except
    raise ClockStorageReadError(
        "Clock You database-existence check failed."
    ) from error
  normalized = output.strip()
  if normalized not in {"0", "1"}:
    raise ClockStorageReadError(
        "Clock You database-existence check returned an invalid value: "
        f"{output!r}."
    )
  return normalized == "1"


def _ensure_clock_you_storage_ready(
    env: interface.AsyncEnv,
    *,
    timeout_seconds: float = 5.0,
) -> bool:
  def ready() -> bool:
    if not _clock_you_database_exists(env):
      return False
    return _clock_you_has_table(env, "alarms") and _clock_you_has_table(
        env, "timeZones"
    )

  if ready():
    return True
  adb_utils.launch_app(_CLOCK_YOU_PACKAGE_NAME, env.controller)
  deadline = time.time() + timeout_seconds
  while time.time() < deadline:
    if ready():
      return True
    time.sleep(0.5)
  return ready()


def _clock_you_alarm_time_ms(hour_24: int, minute: int) -> int:
  return (hour_24 * 3600 + minute * 60) * 1000


def _clock_you_alarm_exists(
    env: interface.AsyncEnv,
    *,
    hour_24: int,
    minute: int,
    enabled: bool | None = None,
) -> bool:
  if not _ensure_clock_you_storage_ready(env):
    raise ClockStorageReadError(
        "Clock You Room schema was not readable after launch."
    )
  where = [f"time={_clock_you_alarm_time_ms(hour_24, minute)}"]
  if enabled is not None:
    where.append(f"enabled={1 if enabled else 0}")
  return _sqlite_count(
      env,
      _CLOCK_YOU_DB,
      f"SELECT COUNT(*) FROM alarms WHERE {' AND '.join(where)};",
  ) > 0


def _clock_you_world_clock_exists(
    env: interface.AsyncEnv,
    *,
    city: str,
) -> bool:
  if not _ensure_clock_you_storage_ready(env):
    raise ClockStorageReadError(
        "Clock You Room schema was not readable after launch."
    )
  return _sqlite_count(
      env,
      _CLOCK_YOU_DB,
      (
          "SELECT COUNT(*) FROM timeZones WHERE "
          f"LOWER(zoneName)=LOWER({_sql_quote(city)});"
      ),
  ) > 0


def _delete_alarm_observation(
    *,
    alarm_exists: bool,
    seen_target_alarm: bool,
) -> tuple[bool, bool]:
  """Latch create evidence so an initially absent alarm cannot no-op-pass."""
  if alarm_exists:
    return False, True
  return seen_target_alarm, seen_target_alarm


def _fossify_has_table(
    env: interface.AsyncEnv, db_path: str, table_name: str
) -> bool:
  out = _sqlite_read(
      env,
      db_path,
      (
          "SELECT name FROM sqlite_master WHERE type='table' AND name="
          f"{_sql_quote(table_name)};"
      ),
  )
  return table_name in {line.strip() for line in out.splitlines()}


def _ensure_fossify_storage_ready(
    env: interface.AsyncEnv,
    *,
    alarms: bool = False,
    timers: bool = False,
    timeout_seconds: float = 5.0,
) -> bool:
  def ready() -> bool:
    return (
        (not alarms or _fossify_has_table(env, _FOSSIFY_ALARMS_DB, "contacts"))
        and (not timers or _fossify_has_table(env, _FOSSIFY_TIMERS_DB, "timers"))
    )

  if ready():
    return True
  adb_utils.launch_app(_FOSSIFY_CLOCK_PACKAGE_NAME, env.controller)
  deadline = time.time() + timeout_seconds
  while time.time() < deadline:
    if ready():
      return True
    time.sleep(0.5)
  return ready()


def _fossify_clear_alarms(env: interface.AsyncEnv) -> None:
  if not _ensure_fossify_storage_ready(env, alarms=True):
    return
  _sqlite_exec(env, _FOSSIFY_ALARMS_DB, "DELETE FROM contacts;")


def _fossify_insert_alarm(
    env: interface.AsyncEnv,
    *,
    hour_24: int,
    minute: int,
    enabled: bool,
    label: str,
) -> None:
  if not _ensure_fossify_storage_ready(env, alarms=True):
    return
  _sqlite_exec(
      env,
      _FOSSIFY_ALARMS_DB,
      (
          "INSERT INTO contacts"
          " (time_in_minutes, days, is_enabled, vibrate, sound_title,"
          " sound_uri, label, one_shot)"
          " VALUES"
          f" ({_alarm_minutes(hour_24, minute)}, 0, {1 if enabled else 0},"
          " 0, 'Default (Cesium)',"
          " 'content://settings/system/alarm_alert',"
          f" {_sql_quote(label)}, 0);"
      ),
  )


def _fossify_alarm_exists(
    env: interface.AsyncEnv,
    *,
    hour_24: int,
    minute: int,
    enabled: bool | None = None,
) -> bool:
  if not _ensure_fossify_storage_ready(env, alarms=True):
    raise ClockStorageReadError(
        "Fossify Clock alarms schema was not readable after launch."
    )
  where = [f"time_in_minutes={_alarm_minutes(hour_24, minute)}"]
  if enabled is not None:
    where.append(f"is_enabled={1 if enabled else 0}")
  return _sqlite_count(
      env,
      _FOSSIFY_ALARMS_DB,
      f"SELECT COUNT(*) FROM contacts WHERE {' AND '.join(where)};",
  ) > 0


def _fossify_timer_exists(
    env: interface.AsyncEnv,
    *,
    hours: int,
    minutes: int,
    seconds: int,
    state_token: str | None = None,
) -> bool:
  if not _ensure_fossify_storage_ready(env, timers=True):
    return False
  total_seconds = hours * 3600 + minutes * 60 + seconds
  where = [f"seconds={total_seconds}"]
  if state_token:
    where.append(
        f"LOWER(state) LIKE {_sql_quote('%' + state_token.casefold() + '%')}"
    )
  out = _sqlite_read(
      env,
      _FOSSIFY_TIMERS_DB,
      f"SELECT COUNT(*) FROM timers WHERE {' AND '.join(where)};",
  )
  values = [
      int(line.strip())
      for line in out.splitlines()
      if line.strip().isdigit()
  ]
  if len(values) != 1:
    raise ClockStorageReadError(
        f"Fossify Clock timer COUNT read returned {out!r}."
    )
  return values[0] > 0


def _force_stop_and_launch(package_name: str, env: interface.AsyncEnv) -> None:
  """Force-stop ``package_name`` and relaunch, polling for the stop to settle.

  The previous implementation slept a fixed 0.5s after ``am force-stop``,
  which races on slow emulators (sometimes 1-2s). Now we poll
  ``pidof`` until the package has no live process or a hard cap fires.
  """
  _adb_shell_or_empty(env, f"am force-stop {shlex.quote(package_name)} || true")
  deadline = time.time() + 3.0
  while time.time() < deadline:
    # `|| true` keeps the adb exit code 0 when no process exists; otherwise
    # android_env retries the "failed" command 3x per poll, wasting seconds.
    pidof = _adb_shell_or_empty(
        env, f"pidof {shlex.quote(package_name)} || true"
    ).strip()
    if not pidof:
      break
    time.sleep(0.1)
  adb_utils.launch_app(package_name, env.controller)


# -----------------------------------------------------------------------------
# UI-text heuristics.
# -----------------------------------------------------------------------------

_COLON_DURATION_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d.])"
)
# The leading (?=\d) rejects the all-empty match once every component group
# is optional; without it these patterns would "fullmatch" arbitrary text.
# Seconds are optional so displays like "1h 30m" / "2 hours, 15 minutes"
# (no seconds part) still parse — long timers commonly omit :00 seconds.
_SHORT_DURATION_RE: Final[re.Pattern[str]] = re.compile(
    r"(?=\d)"
    r"(?:(\d+)\s*h(?:ours?)?\s*)?"
    r"(?:(\d+)\s*m(?:in(?:ute)?s?)?\s*)?"
    r"(?:(\d+)\s*s(?:ec(?:ond)?s?)?)?",
    re.IGNORECASE,
)
_WORD_DURATION_RE: Final[re.Pattern[str]] = re.compile(
    r"(?=\d)"
    r"(?:(\d+)\s+hours?,?\s*(?:and\s*)?)?"
    r"(?:(\d+)\s+minutes?,?\s*(?:and\s*)?)?"
    r"(?:(\d+)\s+seconds?)?",
    re.IGNORECASE,
)


def _colon_timer_candidates(
    hours: int, minutes: int, seconds: int
) -> tuple[str, ...]:
  if hours:
    return (
        f"{hours}:{minutes:02d}:{seconds:02d}",
        f"{hours:02d}:{minutes:02d}:{seconds:02d}",
    )
  return (
      f"{minutes}:{seconds:02d}",
      f"{minutes:02d}:{seconds:02d}",
      f"0:{minutes:02d}:{seconds:02d}",
      f"00:{minutes:02d}:{seconds:02d}",
  )


def _duration_seconds(hours: int, minutes: int, seconds: int) -> int:
  return hours * 3600 + minutes * 60 + seconds


def _timer_text_fields(
    ui_elements: list[representation_utils.UIElement],
) -> tuple[str, ...]:
  fields: list[str] = []
  for element in ui_elements:
    if getattr(element, "package_name", None) == "com.android.systemui":
      continue
    for field in (element.text, element.content_description):
      if field:
        fields.append(field)
        # Flutter merges child semantics into one multiline node (Chrono's
        # stopwatch elapsed time is the first line of its lap-card blob), so
        # each line must also be parseable on its own.
        if "\n" in field:
          fields.extend(
              line for line in field.splitlines() if line.strip()
          )
  return tuple(fields)


def _timer_text_contains(
    ui_elements: list[representation_utils.UIElement],
    candidates: tuple[str, ...],
) -> bool:
  lowered_candidates = tuple(candidate.casefold() for candidate in candidates)
  for field in _timer_text_fields(ui_elements):
    lowered_field = field.casefold()
    if any(candidate in lowered_field for candidate in lowered_candidates):
      return True
  return False


def _control_present(
    ui_elements: list[representation_utils.UIElement],
    controls: tuple[str, ...],
) -> bool:
  patterns = tuple(
      re.compile(rf"\b{re.escape(control.casefold())}\b")
      for control in controls
  )
  for element in ui_elements:
    for field in (element.text, element.content_description):
      if not field:
        continue
      lowered = field.casefold()
      if any(pattern.search(lowered) for pattern in patterns):
        return True
  return False


def _parse_duration_text(text: str) -> int | None:
  normalized = text.strip().lower()
  if not normalized:
    return None
  for pattern in (_WORD_DURATION_RE, _SHORT_DURATION_RE):
    match = pattern.fullmatch(normalized)
    if match:
      hours, minutes, seconds = (int(part or 0) for part in match.groups())
      return _duration_seconds(hours, minutes, seconds)
  if "." in normalized:
    # Stopwatch displays with sub-second precision: "12.3" (SimpleTools /
    # Fossify seconds.tenths), "1:30.5", "0:00.00" (AOSP DeskClock).
    decimal_match = re.fullmatch(
        r"(?:(\d{1,2}):)?(\d{1,4})\.\d{1,3}", normalized
    )
    if decimal_match:
      minutes_part, seconds_part = decimal_match.groups()
      return int(minutes_part or 0) * 60 + int(seconds_part)
    return None
  match = _COLON_DURATION_RE.fullmatch(normalized)
  if not match:
    return None
  first, second, third = match.groups()
  if third is None:
    return int(first) * 60 + int(second)
  return _duration_seconds(int(first), int(second), int(third))


def _visible_timer_durations(
    ui_elements: list[representation_utils.UIElement],
) -> tuple[int, ...]:
  durations: list[int] = []
  for field in _timer_text_fields(ui_elements):
    parsed = _parse_duration_text(field)
    if parsed is not None:
      durations.append(parsed)
  return tuple(durations)


def _field_matches_exact(
    element: representation_utils.UIElement,
    candidates: tuple[str, ...],
) -> bool:
  lowered_candidates = tuple(candidate.casefold() for candidate in candidates)
  for field in (element.text, element.content_description):
    if field and field.strip().casefold() in lowered_candidates:
      return True
  return False


def _top_level_text_present(
    ui_elements: list[representation_utils.UIElement],
    candidates: tuple[str, ...],
    *,
    max_normalized_y_center: float = 0.25,
    max_pixel_y_center: int = 650,
) -> bool:
  """Returns whether an exact title/header-like text appears near the top."""
  for element in ui_elements:
    if not _field_matches_exact(element, candidates):
      continue
    bbox = element.bbox or element.bbox_pixels
    if bbox is None:
      return True
    y_center = (bbox.y_min + bbox.y_max) / 2
    if bbox.y_max <= 1:
      if y_center <= max_normalized_y_center:
        return True
    elif y_center <= max_pixel_y_center:
      return True
  return False


def _is_stopwatch_page(
    ui_elements: list[representation_utils.UIElement],
) -> bool:
  return _top_level_text_present(ui_elements, ("stopwatch",))


def _is_stopwatch_zero_visible(
    ui_elements: list[representation_utils.UIElement],
) -> bool:
  if base.element_text_contains(
      ui_elements, ("00:00.00", "0:00.00", "00:00", "0:00")
  ):
    return True
  zero_fields = 0
  for field in _timer_text_fields(ui_elements):
    stripped = field.strip()
    # "0.0" / "0.00": SimpleTools-style seconds.tenths zero display.
    if re.fullmatch(r"0{1,2}(?:\.0{1,2})?", stripped):
      zero_fields += 1
      if "." in stripped:
        return True
  return zero_fields >= 2


def _is_timer_duration_visible(
    ui_elements: list[representation_utils.UIElement],
    *,
    hours: int,
    minutes: int,
    seconds: int,
    max_elapsed_seconds: int = 0,
) -> bool:
  expected = _duration_seconds(hours, minutes, seconds)
  for actual in _visible_timer_durations(ui_elements):
    if max_elapsed_seconds <= 0 and actual == expected:
      return True
    if max_elapsed_seconds > 0 and 0 < actual <= expected:
      if expected - actual <= max_elapsed_seconds:
        return True
  return False


def _is_stopwatch_advancing(
    env: interface.AsyncEnv,
    *,
    interval_seconds: float = 2.0,
) -> bool:
  """Returns True iff a visible elapsed-time display is actively counting up.

  Compose/Flutter clock apps (e.g. Clock You) expose the stopwatch elapsed
  time but not the Start/Pause/Lap control labels, so label-based running
  detection is blind there. We take three screen snapshots and require a
  duration value that strictly increases across BOTH sample pairs by a
  plausible per-interval amount. An in-app wall clock ("15:45" parses like
  a duration and rolls over by +1/minute) can tick at most once inside the
  ~4s window, so it can never satisfy both pairs; only a genuinely running
  stopwatch can.
  """
  snapshots = []
  for index in range(3):
    if index:
      time.sleep(interval_seconds)
    snapshots.append(
        set(_visible_timer_durations(env.get_state().ui_elements))
    )
  first, middle, last = snapshots
  max_step = int(interval_seconds) + 3
  for value in middle:
    grew_from_first = any(0 < value - v0 <= max_step for v0 in first)
    grows_into_last = any(0 < v2 - value <= max_step for v2 in last)
    if grew_from_first and grows_into_last:
      return True
  return False


def _clock_you_stopwatch_state(
    ui_elements: list[representation_utils.UIElement],
) -> str:
  """Classify Clock You's label-free stopwatch from its action layout.

  Clock You 9.1 renders its elapsed digits and icon glyphs on Compose canvas
  nodes without accessibility text.  The surrounding action buttons are
  accessible and have a stable state machine: one bottom action at untouched
  zero, three while running, and two after pause.  Restricting the count to
  the stopwatch action band avoids the settings and navigation buttons.
  """
  if not _is_stopwatch_page(ui_elements):
    return "other_surface"
  action_buttons = 0
  for element in ui_elements:
    if element.package_name != _CLOCK_YOU_PACKAGE_NAME:
      continue
    if element.class_name != "android.widget.Button" or not element.is_clickable:
      continue
    bbox = element.bbox_pixels
    if bbox is None:
      continue
    if 0.70 <= bbox.y_min / 2400 <= 0.88:
      action_buttons += 1
  return {
      1: "initial_zero",
      2: "paused_nonzero",
      3: "running_nonzero",
  }.get(action_buttons, "unknown")


def _clock_you_timer_picker_duration(
    ui_elements: list[representation_utils.UIElement],
) -> tuple[int, int, int] | None:
  """Read Clock You's selected HH:MM:SS wheel row from element geometry."""
  if not _top_level_text_present(ui_elements, ("timer", "timers")):
    return None
  colons = []
  for element in ui_elements:
    if element.package_name != _CLOCK_YOU_PACKAGE_NAME or element.text != ":":
      continue
    if element.bbox_pixels is not None:
      colons.append(element.bbox_pixels)
  if len(colons) != 2:
    return None
  colons.sort(key=lambda bbox: bbox.x_min)
  selected_y = sum(
      (bbox.y_min + bbox.y_max) / 2 for bbox in colons
  ) / len(colons)
  separators = [
      (colons[0].x_min + colons[0].x_max) / 2,
      (colons[1].x_min + colons[1].x_max) / 2,
  ]
  values: list[int | None] = [None, None, None]
  distances = [float("inf"), float("inf"), float("inf")]
  for element in ui_elements:
    if element.package_name != _CLOCK_YOU_PACKAGE_NAME:
      continue
    if not element.text or not re.fullmatch(r"\d{1,2}", element.text):
      continue
    bbox = element.bbox_pixels
    if bbox is None:
      continue
    x_center = (bbox.x_min + bbox.x_max) / 2
    y_center = (bbox.y_min + bbox.y_max) / 2
    distance = abs(y_center - selected_y)
    if distance > 120:
      continue
    column = 0 if x_center < separators[0] else (
        1 if x_center < separators[1] else 2
    )
    if distance < distances[column]:
      values[column] = int(element.text)
      distances[column] = distance
  if any(value is None for value in values):
    return None
  hours, minutes, seconds = values
  assert hours is not None and minutes is not None and seconds is not None
  if hours > 23 or minutes > 59 or seconds > 59:
    return None
  return hours, minutes, seconds


def _clock_you_running_timer_surface(
    ui_elements: list[representation_utils.UIElement],
) -> bool:
  """Clock You running cards show countdown plus a projected end time."""
  if not _top_level_text_present(ui_elements, ("timer", "timers")):
    return False
  if _control_present(ui_elements, ("start",)):
    return False
  return len(_visible_timer_durations(ui_elements)) >= 2


def _timer_has_started(
    ui_elements: list[representation_utils.UIElement],
) -> bool:
  """Return whether the UI proves a timer was started or subsequently paused."""
  if _clock_you_running_timer_surface(ui_elements):
    return True
  # A paused timer also violates "Do not start".  Word-boundary matching keeps
  # the Stopwatch tab from becoming a false Stop control.
  if _control_present(ui_elements, ("pause", "resume", "stop")):
    return True
  return _control_present(ui_elements, ("cancel",)) and not _control_present(
      ui_elements, ("start",)
  )


def _is_stopwatch_running(
    ui_elements: list[representation_utils.UIElement],
) -> bool:
  """Returns True iff the stopwatch surface is in the *actively running* state.

  We require evidence that the stopwatch is past its initial zero state to
  rule out the case where the agent just navigated to the tab without
  starting the stopwatch (CL3 in the senior review).
  """
  pause_present = False
  lap_present = False
  reset_present = False
  aw_exact_pause = False
  aw_exact_lap = False
  for element in ui_elements:
    if element.content_description == "Pause":
      aw_exact_pause = True
    elif element.content_description == "Lap":
      aw_exact_lap = True
    if base.matches_any_word(element, ("pause",)):
      pause_present = True
    elif base.matches_any_word(element, ("lap",)):
      lap_present = True
    elif base.matches_any_word(element, ("reset", "restart")):
      reset_present = True
  if aw_exact_pause and aw_exact_lap:
    # AndroidWorld's reference ClockStopWatchRunning accepts exactly this
    # Google Clock accessibility-label pair. Preserve that acceptance before
    # applying stricter cross-app guards for icon/label variants.
    return True
  if not pause_present:
    return False
  if _is_stopwatch_zero_visible(ui_elements):
    # A "Pause" control may flicker into view as the stopwatch transitions,
    # but if the display still reads 00:00 the stopwatch is not running.
    return False
  if lap_present:
    return True
  return _is_stopwatch_page(ui_elements) and reset_present


def _is_stopwatch_paused(
    ui_elements: list[representation_utils.UIElement],
) -> bool:
  """Returns True iff the stopwatch was started and is now paused.

  AW's reference ``ClockStopWatchPausedVerify`` accepts the force-cleared
  00:00 zero state because its goal is a pure verify. CATBench's port makes
  the goal self-contained ("run the stopwatch, then pause it"), so the
  validator must reject the untouched zero state: an agent that merely opens
  the stopwatch tab has not done the task.

  Paused-after-run detection:
    1. We are on the stopwatch surface (header or any Stopwatch label).
    2. No live "Pause" control (that would mean still running).
    3. Evidence the stopwatch ran: a Resume/Restart/Reset control (shown
       only after a start on every clock app we ship), or a Start label
       next to a non-zero elapsed time.
  """
  stopwatch_present = False
  start_present = False
  for element in ui_elements:
    if base.matches_any_text(element, ("stopwatch",)):
      stopwatch_present = True
    if base.matches_any_word(element, ("start",)):
      start_present = True
  on_surface = stopwatch_present or _is_stopwatch_page(ui_elements)
  if not on_surface:
    return False
  if _control_present(ui_elements, ("pause",)):
    return False
  ran_evidence = _control_present(ui_elements, ("resume", "restart", "reset"))
  nonzero_elapsed = not _is_stopwatch_zero_visible(ui_elements)
  return ran_evidence or (start_present and nonzero_elapsed)


def _stopwatch_reset_observation(
    ui_elements: list[representation_utils.UIElement],
    *,
    seen_nonzero_elapsed: bool,
) -> tuple[bool, bool]:
  """Update/reset evidence and evaluate a final reset state.

  Some Compose clocks expose the elapsed display but give their play, pause,
  and trash/reset icons no accessibility labels.  Requiring a literal
  ``Reset``/``Restart`` label therefore rejects a real run -> pause -> reset
  trajectory.  The suite invokes ``is_successful`` after every primitive
  action, so the task can conservatively latch a non-zero elapsed display on
  the stopwatch surface and require a later zero, non-running final state.

  The returned latch belongs to one task instance and must be cleared during
  task initialization.  A fresh zero screen, an unrelated duration on another
  page, and a non-zero paused/running screen all remain failures.
  """
  on_surface = _is_stopwatch_page(ui_elements)
  zero_visible = _is_stopwatch_zero_visible(ui_elements)
  if on_surface and not zero_visible:
    seen_nonzero_elapsed = seen_nonzero_elapsed or any(
        duration > 0 for duration in _visible_timer_durations(ui_elements)
    )

  if not on_surface or not zero_visible:
    return False, seen_nonzero_elapsed
  if _control_present(ui_elements, ("pause",)):
    return False, seen_nonzero_elapsed

  labelled_started_evidence = base.element_text_contains_word(
      ui_elements, ("resume", "reset", "restart")
  )
  return (
      seen_nonzero_elapsed or labelled_started_evidence,
      seen_nonzero_elapsed,
  )


def _is_timer_set(
    ui_elements: list[representation_utils.UIElement],
    *,
    hours: int,
    minutes: int,
    seconds: int,
) -> bool:
  text_candidates = (
      f"{hours:02d}h {minutes:02d}m {seconds:02d}s",
      f"{hours}h {minutes}m {seconds}s",
      f"{hours} hours, {minutes} minutes, {seconds} seconds",
      f"{hours} hour, {minutes} minute, {seconds} second",
      *_colon_timer_candidates(hours, minutes, seconds),
  )
  return _timer_text_contains(
      ui_elements, text_candidates
  ) or _is_timer_duration_visible(
      ui_elements,
      hours=hours,
      minutes=minutes,
      seconds=seconds,
  )


def _is_alarm_visible(
    ui_elements: list[representation_utils.UIElement],
    *,
    hour_24: int,
    minute: int,
) -> bool:
  candidates = _alarm_time_candidates(hour_24, minute)
  # _timer_text_contains skips com.android.systemui elements so the status-bar
  # clock can never satisfy (or poison) an alarm-time check.
  return _timer_text_contains(ui_elements, candidates)


def _alarm_time_candidates(hour_24: int, minute: int) -> tuple[str, ...]:
  """Return exact display variants for one requested alarm time."""
  hour_12 = hour_24 % 12 or 12
  meridiem = "AM" if hour_24 < 12 else "PM"
  return (
      f"{hour_24:02d}:{minute:02d}",
      f"{hour_24}:{minute:02d}",
      f"{hour_12}:{minute:02d} {meridiem}",
      f"{hour_12}:{minute:02d}{meridiem}",
      f"{hour_12:02d}:{minute:02d} {meridiem}",
  )


def _element_contains_alarm_time(
    element: representation_utils.UIElement,
    *,
    hour_24: int,
    minute: int,
) -> bool:
  if element.package_name == "com.android.systemui":
    return False
  patterns = tuple(
      re.compile(rf"(?<!\d){re.escape(candidate.casefold())}(?!\d)")
      for candidate in _alarm_time_candidates(hour_24, minute)
  )
  for field in (element.text, element.content_description):
    if not field:
      continue
    lowered = field.casefold()
    if any(pattern.search(lowered) for pattern in patterns):
      return True
  return False


def _is_alarm_editor_surface(
    ui_elements: list[representation_utils.UIElement],
) -> bool:
  """Reject alarm time pickers/editors whose requested value is not saved."""
  editor_titles = ("add alarm", "edit alarm", "new alarm", "set alarm")
  if _top_level_text_present(ui_elements, editor_titles):
    return True
  for element in ui_elements:
    class_name = (element.class_name or "").casefold()
    if "timepicker" in class_name or "numberpicker" in class_name:
      return True
    if _field_matches_exact(element, ("save", "ok", "done", "cancel")):
      # These commit/dismiss controls are present while the chosen time is
      # still a draft. Saved alarm-list rows do not expose them.
      if element.is_clickable is not False:
        return True
  return False


def _saved_alarm_visible(
    ui_elements: list[representation_utils.UIElement],
    *,
    hour_24: int,
    minute: int,
) -> bool:
  """Require a requested alarm time outside a pending editor/picker."""
  return not _is_alarm_editor_surface(ui_elements) and _is_alarm_visible(
      ui_elements, hour_24=hour_24, minute=minute
  )


def _element_bbox_pair(
    first: representation_utils.UIElement,
    second: representation_utils.UIElement,
) -> tuple[
    representation_utils.BoundingBox,
    representation_utils.BoundingBox,
] | None:
  """Return two comparable pixel or normalized boxes, preferring pixels."""
  if first.bbox_pixels is not None and second.bbox_pixels is not None:
    return first.bbox_pixels, second.bbox_pixels
  if first.bbox is not None and second.bbox is not None:
    return first.bbox, second.bbox
  return None


def _same_alarm_row(
    time_element: representation_utils.UIElement,
    switch_element: representation_utils.UIElement,
) -> bool:
  """Conservatively associate a switch with an alarm time by vertical row."""
  if time_element is switch_element:
    return True
  boxes = _element_bbox_pair(time_element, switch_element)
  if boxes is None:
    return False
  time_box, switch_box = boxes
  normalized = max(time_box.y_max, switch_box.y_max) <= 1
  padding = 0.04 if normalized else 96
  return not (
      time_box.y_max + padding < switch_box.y_min
      or switch_box.y_max + padding < time_box.y_min
  )


def _target_alarm_switch_checked(
    ui_elements: list[representation_utils.UIElement],
    *,
    hour_24: int,
    minute: int,
) -> bool:
  """Require the nearest switch in the requested alarm row to be checked."""
  time_elements = [
      element
      for element in ui_elements
      if _element_contains_alarm_time(
          element, hour_24=hour_24, minute=minute
      )
  ]
  switches = []
  for element in ui_elements:
    class_name = (element.class_name or "").casefold()
    if element.is_checkable or any(
        token in class_name for token in ("switch", "checkbox", "toggle")
    ):
      switches.append(element)

  associations = [
      (time_element, switch_element)
      for time_element in time_elements
      for switch_element in switches
      if _same_alarm_row(time_element, switch_element)
  ]
  if not associations:
    return False

  def distance(pair: tuple[
      representation_utils.UIElement,
      representation_utils.UIElement,
  ]) -> float:
    time_element, switch_element = pair
    if time_element is switch_element:
      return 0.0
    boxes = _element_bbox_pair(time_element, switch_element)
    if boxes is None:
      return float("inf")
    first, second = boxes
    return abs(
        (first.y_min + first.y_max) / 2
        - (second.y_min + second.y_max) / 2
    )

  closest_distance = min(distance(pair) for pair in associations)
  closest_states = {
      switch.is_checked
      for pair in associations
      if distance(pair) == closest_distance
      for switch in (pair[1],)
  }
  return closest_states == {True}


def _world_clock_selection_surface_present(
    ui_elements: list[representation_utils.UIElement],
) -> bool:
  """Identify a city search/selection surface rather than the saved list."""
  selection_phrases = (
      "search cities",
      "search for a city",
      "search locations",
      "choose a city",
      "select a city",
      "choose location",
      "select location",
  )
  for element in ui_elements:
    class_name = (element.class_name or "").casefold()
    fields = (
        element.text,
        element.content_description,
        element.hint_text,
        element.resource_name,
        element.resource_id,
    )
    searchable_text = " ".join(field.casefold() for field in fields if field)
    if any(phrase in searchable_text for phrase in selection_phrases):
      return True
    if element.is_editable or "edittext" in class_name:
      return True
  return False


def _alarm_list_is_empty(
    ui_elements: list[representation_utils.UIElement],
) -> bool:
  """Returns True iff the alarm list is *confirmed* empty.

  The previous heuristic returned True whenever no element looked like a
  ``HH:MM`` time -- but splash screens, loading states, and most non-alarm
  surfaces also lack ``HH:MM`` text, so the validator passed in many
  unrelated contexts (CL6 in the senior review). We now require an explicit
  empty-state marker AND that we are actually on the alarm tab/page.
  """
  empty_markers = (
      "no alarms",
      "no alarm",
      "you have no alarms",
      "nothing scheduled",
      "alarm list is empty",
      "no scheduled alarms",
  )
  on_alarm_page = _top_level_text_present(ui_elements, ("alarm", "alarms"))
  if not on_alarm_page:
    return False
  if base.element_text_contains(ui_elements, empty_markers):
    return True
  # Conservative fallback: empty AND on alarm page AND no element looks like a
  # HH:MM time row anywhere in the list area.
  has_time_row = any(
      (element.text or "").count(":") == 1
      and any(ch.isdigit() for ch in (element.text or ""))
      for element in ui_elements
  )
  return not has_time_row


# -----------------------------------------------------------------------------
# Base evaluators (shared by every app port).
# -----------------------------------------------------------------------------


class _ClockStopWatchPausedVerifyBase(base.PackageAppEval):
  """Base port of ``ClockStopWatchPausedVerify``."""

  complexity = 1
  schema = {"type": "object", "properties": {}}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    if _uses_clock_you_storage(self.package_name):
      return 1.0 if _clock_you_stopwatch_state(ui) == "paused_nonzero" else 0.0
    if _is_stopwatch_paused(ui):
      return 1.0
    # Label-free fallback for apps whose stopwatch controls expose no
    # accessibility labels (e.g. Compose icon buttons): on the stopwatch
    # surface, a non-zero elapsed display that is NOT advancing means the
    # stopwatch ran and is now paused.
    on_surface = base.element_text_contains_word(ui, ("stopwatch",))
    if (
        on_surface
        and not _is_stopwatch_zero_visible(ui)
        and _visible_timer_durations(ui)
        and not _control_present(ui, ("pause",))
        and not _is_stopwatch_advancing(env)
    ):
      return 1.0
    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class _ClockStopWatchRunningBase(base.PackageAppEval):
  """Base port of ``ClockStopWatchRunning``."""

  complexity = 1
  schema = {"type": "object", "properties": {}}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    if _uses_clock_you_storage(self.package_name):
      return 1.0 if _clock_you_stopwatch_state(ui) == "running_nonzero" else 0.0
    if _is_stopwatch_running(ui):
      return 1.0
    # Label-free fallback: on the stopwatch surface, an elapsed display
    # counting up across snapshots proves the stopwatch is running even
    # when the Pause/Lap controls expose no accessibility labels.
    on_surface = base.element_text_contains_word(ui, ("stopwatch",))
    if on_surface and _is_stopwatch_advancing(env):
      return 1.0
    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class _ClockTimerEntryBase(base.PackageAppEval):
  """Create but do not start a timer at the requested duration."""

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "hours": {"type": "integer"},
          "minutes": {"type": "integer"},
          "seconds": {"type": "integer"},
      },
      "required": ["hours", "minutes", "seconds"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    if _timer_has_started(ui):
      return 0.0
    if _uses_clock_you_storage(self.package_name):
      picker_duration = _clock_you_timer_picker_duration(ui)
      if picker_duration == (
          self._params["hours"],
          self._params["minutes"],
          self._params["seconds"],
      ):
        return 1.0
    if _uses_fossify_storage(self.package_name) and _fossify_timer_exists(
        env,
        hours=self._params["hours"],
        minutes=self._params["minutes"],
        seconds=self._params["seconds"],
        state_token="idle",
    ):
      # Strong durable evidence when Fossify's Room DB has the timer row.
      # On the benchmark image app.db has no `timers` table until the app
      # itself persists one, so a storage miss must NOT be treated as
      # failure — fall through to the same UI check used for every other
      # clock app instead of false-failing a correct agent.
      return 1.0
    return (
        1.0
        if _is_timer_set(
            ui_elements=ui,
            hours=self._params["hours"],
            minutes=self._params["minutes"],
            seconds=self._params["seconds"],
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": random.randint(0, 3),
        "minutes": random.randint(0, 59),
        "seconds": random.randint(0, 59),
    }


class _ClockStartTimerBase(base.PackageAppEval):
  """Create a timer and start it."""

  complexity = 2.0
  schema = _ClockTimerEntryBase.schema

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    timer_ok = _is_timer_duration_visible(
        ui_elements=ui,
        hours=self._params["hours"],
        minutes=self._params["minutes"],
        seconds=self._params["seconds"],
        max_elapsed_seconds=120,
    )
    if _uses_clock_you_storage(self.package_name):
      return 1.0 if timer_ok and _clock_you_running_timer_surface(ui) else 0.0
    running_ok = _control_present(ui, ("pause", "cancel", "stop"))
    resume_present = _control_present(ui, ("resume",))
    return 1.0 if timer_ok and running_ok and not resume_present else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": 0,
        "minutes": random.randint(2, 30),
        "seconds": random.choice((0, 15, 30, 45)),
    }


class _ClockTimerWithLabelBase(base.PackageAppEval):
  """Create a timer with a duration AND a visible label.

  Success heuristic: the timer duration appears AND the label string appears
  as a visible text element.
  """

  complexity = 2.2
  schema = {
      "type": "object",
      "properties": {
          "hours": {"type": "integer"},
          "minutes": {"type": "integer"},
          "seconds": {"type": "integer"},
          "label": {"type": "string"},
      },
      "required": ["hours", "minutes", "seconds", "label"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    timer_ok = _is_timer_set(
        ui_elements=ui,
        hours=self._params["hours"],
        minutes=self._params["minutes"],
        seconds=self._params["seconds"],
    )
    label_ok = base.element_text_contains(ui, (self._params["label"],))
    return 1.0 if timer_ok and label_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    labels = ("Tea", "Pasta", "Workout", "Meditation", "Focus", "Bread")
    return {
        "hours": 0,
        "minutes": random.randint(1, 30),
        "seconds": random.choice((0, 15, 30, 45)),
        "label": random.choice(labels),
    }


class _ClockAddAlarmBase(base.PackageAppEval):
  """Add an alarm at the requested time.

  Success heuristic: the alarm time appears on screen in either 24h
  (``HH:MM``) or 12h (``h:MM AM/PM``) form after save.
  """

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {
          "hour": {"type": "integer"},
          "minute": {"type": "integer"},
      },
      "required": ["hour", "minute"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if _uses_fossify_storage(self.package_name):
      return (
          1.0
          if _fossify_alarm_exists(
              env,
              hour_24=self._params["hour"],
              minute=self._params["minute"],
          )
          else 0.0
      )
    if _uses_clock_you_storage(self.package_name):
      return (
          1.0
          if _clock_you_alarm_exists(
              env,
              hour_24=self._params["hour"],
              minute=self._params["minute"],
          )
          else 0.0
      )
    return 1.0 if _saved_alarm_visible(
        env.get_state().ui_elements,
        hour_24=self._params["hour"],
        minute=self._params["minute"],
    ) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hour": random.randint(5, 22),
        "minute": random.choice((0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)),
    }


class _ClockAddAlarmWithLabelBase(base.PackageAppEval):
  """Add an alarm at a specific time WITH a label.

  Success heuristic: the alarm time is visible AND the label string appears.
  """

  complexity = 3
  schema = {
      "type": "object",
      "properties": {
          "hour": {"type": "integer"},
          "minute": {"type": "integer"},
          "label": {"type": "string"},
      },
      "required": ["hour", "minute", "label"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if _uses_fossify_storage(self.package_name):
      return (
          1.0
          if _fossify_alarm_exists(
              env,
              hour_24=self._params["hour"],
              minute=self._params["minute"],
          )
          else 0.0
      )
    ui = env.get_state().ui_elements
    alarm_ok = _is_alarm_visible(
        ui_elements=ui,
        hour_24=self._params["hour"],
        minute=self._params["minute"],
    )
    label_ok = base.element_text_contains(ui, (self._params["label"],))
    return 1.0 if alarm_ok and label_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    labels = ("Wake up", "Standup", "Commute", "Medication", "Gym", "Call mom")
    return {
        "hour": random.randint(5, 22),
        "minute": random.choice((0, 15, 30, 45)),
        "label": random.choice(labels),
    }


class _ClockDeleteAllAlarmsBase(base.PackageAppEval):
  """Delete every alarm in the app.

  This task seeds ``expected_label``: a label the agent must NOT see after
  deletion. The base class relies on the per-app teardown wipe to pre-populate
  nothing; agents that fail to delete will still see the default system alarm,
  so the heuristic tolerates a fully empty list.
  """

  complexity = 2.2
  schema = {"type": "object", "properties": {}}
  clear_data_on_init = True  # start from a clean list

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    # An empty alarm list on the alarm page is the strongest success signal
    # we have; the transient deletion snackbar disappears within ~2-3s and
    # is often gone by the time we re-poll (CL7 in the senior review).
    if _alarm_list_is_empty(ui):
      return 1.0
    deletion_marker = base.element_text_contains(
        ui, ("deleted", "removed", "undo")
    )
    return 1.0 if deletion_marker and _alarm_list_is_empty(ui) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class _ClockTimerEntryShortBase(_ClockTimerEntryBase):
  """Sub-minute timer (purely seconds)."""

  complexity = 1.4

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": 0,
        "minutes": 0,
        "seconds": random.randint(10, 59),
    }


class _ClockTimerEntryLongBase(_ClockTimerEntryBase):
  """Multi-hour timer."""

  complexity = 1.6

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": random.randint(1, 3),
        "minutes": random.choice((0, 15, 30, 45)),
        "seconds": 0,
    }


class _ClockAddAlarmMorningBase(_ClockAddAlarmBase):
  """Alarm constrained to morning hours (5am-11am)."""

  complexity = 2.4

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hour": random.randint(5, 11),
        "minute": random.choice((0, 15, 30, 45)),
    }


class _ClockAddAlarmEveningBase(_ClockAddAlarmBase):
  """Alarm constrained to evening hours (18-22)."""

  complexity = 2.4

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hour": random.randint(18, 22),
        "minute": random.choice((0, 15, 30, 45)),
    }


class _ClockAddTwoAlarmsBase(base.PackageAppEval):
  """Add two distinct alarms at different times."""

  complexity = 3.2
  schema = {
      "type": "object",
      "properties": {
          "hour_a": {"type": "integer"},
          "minute_a": {"type": "integer"},
          "hour_b": {"type": "integer"},
          "minute_b": {"type": "integer"},
      },
      "required": ["hour_a", "minute_a", "hour_b", "minute_b"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    a = _is_alarm_visible(
        ui, hour_24=self._params["hour_a"], minute=self._params["minute_a"]
    )
    b = _is_alarm_visible(
        ui, hour_24=self._params["hour_b"], minute=self._params["minute_b"]
    )
    return 1.0 if a and b else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hour_a": random.randint(6, 11),
        "minute_a": random.choice((0, 15, 30, 45)),
        "hour_b": random.randint(13, 22),
        "minute_b": random.choice((0, 15, 30, 45)),
    }


class _ClockNavigateToAlarmTabBase(base.PackageAppEval):
  """Open the Alarm tab/screen of the clock app.

  The previous heuristic matched ``alarm``/``alarms`` anywhere in the UI,
  but every clock app shows ``Alarm`` as a bottom-nav tab label from every
  tab -- so the validator passed unconditionally (CL4 in the senior review).
  We now require the text to appear as a *top-level* header near the top of
  the screen, which is how an active tab is rendered across apps. We also
  require that no other tab title (Timer / Stopwatch / World Clock) is also
  visible as a top-level header, so we don't trip on labels that happen to
  appear in headers but aren't the active tab.
  """

  complexity = 1
  schema = {"type": "object", "properties": {}}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    alarm_header = _top_level_text_present(ui, ("alarm", "alarms"))
    if not alarm_header:
      return 0.0
    other_active = (
        _is_stopwatch_page(ui)
        or _top_level_text_present(ui, ("timer", "timers"))
        or _top_level_text_present(ui, ("world clock", "clock"))
    )
    return 1.0 if not other_active else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class _ClockNavigateToTimerTabBase(base.PackageAppEval):
  """Open the Timer tab/screen of the clock app.

  Same fix as ``_ClockNavigateToAlarmTabBase`` -- require ``Timer`` as a
  top-level header rather than anywhere in the UI.
  """

  complexity = 1
  schema = {"type": "object", "properties": {}}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    timer_header = _top_level_text_present(ui, ("timer", "timers"))
    if not timer_header:
      return 0.0
    other_active = (
        _is_stopwatch_page(ui)
        or _top_level_text_present(ui, ("alarm", "alarms"))
    )
    return 1.0 if not other_active else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


# -----------------------------------------------------------------------------
# Cross-app target tasks: edit/disable/delete a single seeded alarm,
# pause/resume an in-flight timer, reset stopwatch, add a world clock.
#
# These all rely on a "set up the precondition yourself first" instruction,
# so initialize_task is just the standard package launch — no adb seeding.
# -----------------------------------------------------------------------------


class _ClockEditAlarmBase(base.PackageAppEval):
  """Add an alarm at one time, then edit it to a different time.

  The agent is told both the original and target times; success requires the
  target time to be visible AND the original time to be absent (i.e. the
  alarm row was actually mutated rather than a second alarm being added).
  """

  complexity = 3
  schema = {
      "type": "object",
      "properties": {
          "old_hour": {"type": "integer"},
          "old_minute": {"type": "integer"},
          "new_hour": {"type": "integer"},
          "new_minute": {"type": "integer"},
      },
      "required": ["old_hour", "old_minute", "new_hour", "new_minute"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if _uses_fossify_storage(self.package_name):
      new_exists = _fossify_alarm_exists(
          env,
          hour_24=self._params["new_hour"],
          minute=self._params["new_minute"],
      )
      old_exists = _fossify_alarm_exists(
          env,
          hour_24=self._params["old_hour"],
          minute=self._params["old_minute"],
      )
      return 1.0 if new_exists and not old_exists else 0.0
    if _uses_clock_you_storage(self.package_name):
      new_exists = _clock_you_alarm_exists(
          env,
          hour_24=self._params["new_hour"],
          minute=self._params["new_minute"],
      )
      old_exists = _clock_you_alarm_exists(
          env,
          hour_24=self._params["old_hour"],
          minute=self._params["old_minute"],
      )
      return 1.0 if new_exists and not old_exists else 0.0
    ui = env.get_state().ui_elements
    new_visible = _saved_alarm_visible(
        ui, hour_24=self._params["new_hour"], minute=self._params["new_minute"]
    )
    old_visible = _is_alarm_visible(
        ui, hour_24=self._params["old_hour"], minute=self._params["old_minute"]
    )
    return 1.0 if new_visible and not old_visible else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    old_hour = random.randint(6, 11)
    new_hour = random.randint(13, 22)
    return {
        "old_hour": old_hour,
        "old_minute": random.choice((0, 15, 30, 45)),
        "new_hour": new_hour,
        "new_minute": random.choice((0, 15, 30, 45)),
    }


class _ClockEnableAlarmBase(base.PackageAppEval):
  """Add an alarm, then ensure its toggle is ON/enabled.

  Storage adapters require the exact enabled row. UI adapters require a
  checked switch geometrically associated with the requested alarm-time row.
  """

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {
          "hour": {"type": "integer"},
          "minute": {"type": "integer"},
      },
      "required": ["hour", "minute"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if _uses_fossify_storage(self.package_name):
      return (
          1.0
          if _fossify_alarm_exists(
              env,
              hour_24=self._params["hour"],
              minute=self._params["minute"],
              enabled=True,
          )
          else 0.0
      )
    if _uses_clock_you_storage(self.package_name):
      return (
          1.0
          if _clock_you_alarm_exists(
              env,
              hour_24=self._params["hour"],
              minute=self._params["minute"],
              enabled=True,
          )
          else 0.0
      )
    ui = env.get_state().ui_elements
    return 1.0 if _target_alarm_switch_checked(
        ui,
        hour_24=self._params["hour"],
        minute=self._params["minute"],
    ) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hour": random.randint(6, 22),
        "minute": random.choice((0, 15, 30, 45)),
    }


class _ClockDeleteAlarmBase(base.PackageAppEval):
  """Add an alarm at H:M, then delete that specific alarm.

  Success requires episode-local observation of the requested alarm followed
  by its absence. Generic deletion/snackbar text is deliberately insufficient.
  """

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {
          "hour": {"type": "integer"},
          "minute": {"type": "integer"},
      },
      "required": ["hour", "minute"],
  }

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self._catbench_seen_target_alarm = False

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if _uses_clock_you_storage(self.package_name):
      exists = _clock_you_alarm_exists(
          env,
          hour_24=self._params["hour"],
          minute=self._params["minute"],
      )
      success, seen = _delete_alarm_observation(
          alarm_exists=exists,
          seen_target_alarm=getattr(
              self, "_catbench_seen_target_alarm", False
          ),
      )
      self._catbench_seen_target_alarm = seen
      return 1.0 if success else 0.0
    if _uses_fossify_storage(self.package_name):
      exists = _fossify_alarm_exists(
          env,
          hour_24=self._params["hour"],
          minute=self._params["minute"],
      )
      success, seen = _delete_alarm_observation(
          alarm_exists=exists,
          seen_target_alarm=getattr(
              self, "_catbench_seen_target_alarm", False
          ),
      )
      self._catbench_seen_target_alarm = seen
      return 1.0 if success else 0.0
    ui = env.get_state().ui_elements
    alarm_visible = _saved_alarm_visible(
        ui, hour_24=self._params["hour"], minute=self._params["minute"]
    )
    success, seen = _delete_alarm_observation(
        alarm_exists=alarm_visible,
        seen_target_alarm=getattr(
            self, "_catbench_seen_target_alarm", False
        ),
    )
    self._catbench_seen_target_alarm = seen
    on_alarm_page = _top_level_text_present(ui, ("alarm", "alarms"))
    return 1.0 if success and on_alarm_page else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hour": random.randint(6, 22),
        "minute": random.choice((0, 15, 30, 45)),
    }


class _ClockPauseTimerBase(base.PackageAppEval):
  """Start a timer for HH:MM:SS, then pause it before it fires.

  Success heuristic: timer duration is visible AND a Resume (paused-
  state) control is visible while a Pause control is NOT. We use
  ``_control_present`` (regex word-boundary) so ``Restart`` does not
  trip the "Resume" branch (CL2 in the senior review).
  """

  complexity = 2.4
  schema = _ClockTimerEntryBase.schema

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    timer_ok = _is_timer_duration_visible(
        ui_elements=ui,
        hours=self._params["hours"],
        minutes=self._params["minutes"],
        seconds=self._params["seconds"],
        max_elapsed_seconds=120,
    )
    resume_present = _control_present(ui, ("resume",))
    pause_present = _control_present(ui, ("pause",))
    return 1.0 if timer_ok and resume_present and not pause_present else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": 0,
        "minutes": random.randint(2, 30),
        "seconds": random.choice((0, 15, 30, 45)),
    }


class _ClockResumeTimerBase(base.PackageAppEval):
  """Start, pause, then resume a timer.

  Success heuristic: a Pause control is visible (running state).
  """

  complexity = 2.6
  schema = _ClockTimerEntryBase.schema

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    timer_ok = _is_timer_duration_visible(
        ui_elements=ui,
        hours=self._params["hours"],
        minutes=self._params["minutes"],
        seconds=self._params["seconds"],
        max_elapsed_seconds=120,
    )
    pause_present = _control_present(ui, ("pause",))
    resume_present = _control_present(ui, ("resume",))
    return 1.0 if timer_ok and pause_present and not resume_present else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": 0,
        "minutes": random.randint(2, 30),
        "seconds": random.choice((0, 15, 30, 45)),
    }


class _ClockStopWatchResetBase(base.PackageAppEval):
  """Run the stopwatch, then reset it to 0.

  Success requires a zero, non-running stopwatch surface plus episode-local
  evidence that it previously ran: either a non-zero elapsed display observed
  during an earlier verifier call or a post-run Resume/Reset/Restart label.
  We deliberately reject the initial zero state where only Start is visible.
  """

  complexity = 2.2
  schema = {"type": "object", "properties": {}}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    # Episode-local evidence only.  Never carry a previous task's observed
    # stopwatch state into a new reset trial.
    self._catbench_seen_nonzero_stopwatch = False

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    if _uses_clock_you_storage(self.package_name):
      state = _clock_you_stopwatch_state(ui)
      seen_nonzero = getattr(
          self, "_catbench_seen_nonzero_stopwatch", False
      )
      if state in {"running_nonzero", "paused_nonzero"}:
        self._catbench_seen_nonzero_stopwatch = True
        return 0.0
      return 1.0 if seen_nonzero and state == "initial_zero" else 0.0
    success, seen_nonzero = _stopwatch_reset_observation(
        ui,
        seen_nonzero_elapsed=getattr(
            self, "_catbench_seen_nonzero_stopwatch", False
        ),
    )
    self._catbench_seen_nonzero_stopwatch = seen_nonzero
    return 1.0 if success else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class _ClockAddWorldClockBase(base.PackageAppEval):
  """Add a city to the world-clock surface.

  Clock You uses its persisted Room row. UI adapters require the chosen city
  outside a city-search/selection surface.
  """

  complexity = 2.2
  schema = {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if _uses_clock_you_storage(self.package_name):
      return (
          1.0
          if _clock_you_world_clock_exists(env, city=self._params["city"])
          else 0.0
      )
    ui = env.get_state().ui_elements
    if _world_clock_selection_surface_present(ui):
      return 0.0
    return (
        1.0
        if base.element_text_contains(
            ui, (self._params["city"],)
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    # Single-word IANA time-zone localities only (Asia/Singapore,
    # Europe/Berlin, ...), verified present in the bundled city data of all
    # clock apps by grepping their APKs on-device. Exclusions:
    #   - Multi-word cities ("New York"): Clock You renders tz IDs verbatim
    #     ("New_York"), making the goal unsolvable there.
    #   - Non-IANA names (Mumbai -> Asia/Kolkata): missing from tzdb pickers.
    #   - Tokyo / London / Paris: preloaded as default world clocks in
    #     Chrono, so adding them would no-op-pass the validator.
    cities = (
        "Sydney",
        "Singapore",
        "Berlin",
        "Madrid",
    )
    return {"city": random.choice(cities)}


# -----------------------------------------------------------------------------
# Per-app packages.
# -----------------------------------------------------------------------------

_CHRONO_PACKAGE: Final[str] = "com.vicolo.chrono"
_SIMPLE_CLOCK_PACKAGE: Final[str] = "com.simplemobiletools.clock"
_FOSSIFY_CLOCK_PACKAGE: Final[str] = "org.fossify.clock"
_CLOCK_YOU_PACKAGE: Final[str] = "com.bnyro.clock"
_CLOCK_PACKAGE: Final[str] = "com.best.deskclock"
_GOOGLE_CLOCK_PACKAGE: Final[str] = "com.google.android.deskclock"
_SEEDED_ALARM_PACKAGES: Final[frozenset[str]] = frozenset((
    _FOSSIFY_CLOCK_PACKAGE,
))


_APP_DISPLAY_NAMES: Final[dict[str, str]] = {
    _CHRONO_PACKAGE: "Chrono",
    _SIMPLE_CLOCK_PACKAGE: "Simple Clock",
    _FOSSIFY_CLOCK_PACKAGE: "Fossify Clock",
    _CLOCK_YOU_PACKAGE: "Clock You",
    _CLOCK_PACKAGE: "Clock",
    _GOOGLE_CLOCK_PACKAGE: "Google Clock",
}


# -----------------------------------------------------------------------------
# Per-task templates.  ``{app}`` is substituted with the app's display name.
# -----------------------------------------------------------------------------

_TEMPLATES = {
    _ClockStopWatchPausedVerifyBase: (
        "In the {app} app, run the stopwatch for a moment, then pause it."
    ),
    _ClockStopWatchRunningBase: "In the {app} app, run the stopwatch.",
    _ClockTimerEntryBase: (
        "In the {app} app, create a timer with {{hours}} hours, {{minutes}}"
        " minutes, and {{seconds}} seconds. Do not start the timer."
    ),
    _ClockStartTimerBase: (
        "In the {app} app, create a timer with {{hours}} hours, {{minutes}}"
        " minutes, and {{seconds}} seconds, then start the timer."
    ),
    _ClockTimerEntryShortBase: (
        "In the {app} app, create a short timer of {{seconds}} seconds. Do"
        " not start it."
    ),
    _ClockTimerEntryLongBase: (
        "In the {app} app, create a long timer of {{hours}} hours and"
        " {{minutes}} minutes. Do not start it."
    ),
    _ClockTimerWithLabelBase: (
        "In the {app} app, create a timer labelled `{{label}}` with"
        " {{hours}} hours, {{minutes}} minutes, and {{seconds}} seconds. Do"
        " not start the timer."
    ),
    _ClockAddAlarmBase: (
        "In the {app} app, add a new alarm at {{hour:02d}}:{{minute:02d}}."
    ),
    _ClockAddAlarmMorningBase: (
        "In the {app} app, add a morning alarm at {{hour:02d}}:{{minute:02d}}."
    ),
    _ClockAddAlarmEveningBase: (
        "In the {app} app, add an evening alarm at {{hour:02d}}:{{minute:02d}}."
    ),
    _ClockAddAlarmWithLabelBase: (
        "In the {app} app, add a new alarm at {{hour:02d}}:{{minute:02d}}"
        " with the label `{{label}}`."
    ),
    _ClockAddTwoAlarmsBase: (
        "In the {app} app, add two alarms: the first at"
        " {{hour_a:02d}}:{{minute_a:02d}} and the second at"
        " {{hour_b:02d}}:{{minute_b:02d}}."
    ),
    _ClockDeleteAllAlarmsBase: (
        "In the {app} app, delete every alarm so the alarm list is empty."
    ),
    _ClockNavigateToAlarmTabBase: (
        "In the {app} app, open the alarm screen."
    ),
    _ClockNavigateToTimerTabBase: (
        "In the {app} app, open the timer screen."
    ),
    _ClockEditAlarmBase: (
        "In the {app} app, first add an alarm at"
        " {{old_hour:02d}}:{{old_minute:02d}}, then edit it so the alarm"
        " fires at {{new_hour:02d}}:{{new_minute:02d}} instead."
    ),
    _ClockEnableAlarmBase: (
        "In the {app} app, add an alarm at {{hour:02d}}:{{minute:02d}} and"
        " make sure its toggle is ON / enabled."
    ),
    _ClockDeleteAlarmBase: (
        "In the {app} app, add an alarm at {{hour:02d}}:{{minute:02d}} and"
        " then delete that alarm."
    ),
    _ClockPauseTimerBase: (
        "In the {app} app, start a timer for {{hours}} hours, {{minutes}}"
        " minutes, and {{seconds}} seconds, then pause the timer before it"
        " finishes."
    ),
    _ClockResumeTimerBase: (
        "In the {app} app, start a timer for {{hours}} hours, {{minutes}}"
        " minutes, and {{seconds}} seconds, pause it, and then resume it."
    ),
    _ClockStopWatchResetBase: (
        "In the {app} app, run the stopwatch for a moment and then reset"
        " it to 0."
    ),
    _ClockAddWorldClockBase: (
        "In the {app} app, add `{{city}}` to your world clocks."
    ),
}


_SEEDED_ALARM_TEMPLATES: Final[dict[type, str]] = {
    _ClockEditAlarmBase: (
        "In the {app} app, edit the existing alarm at"
        " {{old_hour:02d}}:{{old_minute:02d}} so it fires at"
        " {{new_hour:02d}}:{{new_minute:02d}} instead."
    ),
    _ClockEnableAlarmBase: (
        "In the {app} app, find the existing alarm at"
        " {{hour:02d}}:{{minute:02d}} and make sure its toggle is ON /"
        " enabled."
    ),
    _ClockDeleteAlarmBase: (
        "In the {app} app, delete the existing alarm at"
        " {{hour:02d}}:{{minute:02d}}."
    ),
}


_PACKAGES = (
    _CLOCK_PACKAGE,
    _SIMPLE_CLOCK_PACKAGE,
    _GOOGLE_CLOCK_PACKAGE,
    _CLOCK_YOU_PACKAGE,
    _CHRONO_PACKAGE,
    _FOSSIFY_CLOCK_PACKAGE,
)


# -----------------------------------------------------------------------------
# Generated per-app ports.  Module-level ``globals()`` assignment keeps the
# public names discoverable by the task registry (which reflects over this
# module's attributes).
# -----------------------------------------------------------------------------


# Cross-app Clock task templates. The 10 short names below ARE the user's
# target task list for the Clock category in hybrid mode. Every base class
# is fanned out across all 6 packages in ``_PACKAGES``.
_BASE_SHORT_NAMES = {
    _ClockAddAlarmBase: "ClockCreateAlarm",
    _ClockEditAlarmBase: "ClockEditAlarm",
    _ClockEnableAlarmBase: "ClockEnableAlarm",
    _ClockDeleteAlarmBase: "ClockDeleteAlarm",
    _ClockTimerEntryBase: "ClockCreateTimer",
    _ClockStartTimerBase: "ClockStartTimer",
    _ClockStopWatchRunningBase: "ClockStopwatchRunning",
    _ClockStopWatchPausedVerifyBase: "ClockPauseStopwatch",
    _ClockStopWatchResetBase: "ClockStopwatchReset",
    _ClockAddWorldClockBase: "ClockAddWorldClock",
}


for _base_cls, _short in _BASE_SHORT_NAMES.items():
  excluded = getattr(_base_cls, "excluded_packages", ())
  for _pkg in _PACKAGES:
    if _pkg in excluded:
      continue
    _display = _APP_DISPLAY_NAMES[_pkg]
    _suffix = _display.replace(" ", "")
    _cls_name = f"{_short}For{_suffix}"
    # Use one create-first workflow for every app.  Fossify previously received
    # a pre-seeded "existing alarm" task, so it was not a matched cross-app
    # instance even when the sampled times were identical.
    _template = _TEMPLATES[_base_cls]
    _validation_mode = "UI heuristic"
    if _pkg == _FOSSIFY_CLOCK_PACKAGE and _base_cls in (
        _ClockAddAlarmBase,
        _ClockEditAlarmBase,
        _ClockEnableAlarmBase,
        _ClockDeleteAlarmBase,
    ):
      _validation_mode = "SQLite"
    elif _pkg == _FOSSIFY_CLOCK_PACKAGE and _base_cls is _ClockTimerEntryBase:
      _validation_mode = "SQLite + UI fallback"
    elif _pkg == _CLOCK_YOU_PACKAGE and _base_cls in (
        _ClockAddAlarmBase,
        _ClockEditAlarmBase,
        _ClockEnableAlarmBase,
        _ClockDeleteAlarmBase,
        _ClockAddWorldClockBase,
    ):
      _validation_mode = "Clock You Room SQLite durable state"
    _attrs = {
        "app_names": (_pkg,),
        "package_name": _pkg,
        "catbench_semantic_id": _short,
        "catbench_app_display_name": _display,
        "template": _template.format(app=_display),
        "validation_mode": _validation_mode,
    }
    globals()[_cls_name] = type(_cls_name, (_base_cls,), _attrs)


# Clean up module-level iteration variables to avoid polluting reflection.
del _base_cls, _short, _pkg, _display, _suffix, _cls_name, _template, _validation_mode, _attrs
