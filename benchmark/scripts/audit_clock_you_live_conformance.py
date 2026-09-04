#!/usr/bin/env python3
"""Run real-device positive and near-miss checks for Clock You's 10 verifiers.

This is a deterministic harness/conformance audit.  It does not call a model,
does not produce an episode score, and must never be merged into paper results.
It targets the pinned Clock You 9.1/API-33 Pixel-6 layout and exercises the
actual generated task classes against real Room and UI states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

from android_world.env import env_launcher
from android_world.task_evals.single.app_generalization_generated import (
    clock_cross_app_tasks as clock_tasks,
)


PACKAGE = "com.bnyro.clock"
DB_PATH = "/data/data/com.bnyro.clock/databases/com.bnyro.clock"
EXPECTED_VERSION_NAME = "9.1"
EXPECTED_VERSION_CODE = "19"
EXPECTED_APK_SHA256 = (
    "48ffaa45bf01da3d59002de05022706398518324f63624aeb069368f6061a79e"
)


class LiveAudit:
  """Mutable real-device fixture with fail-closed case recording."""

  def __init__(
      self,
      *,
      adb_path: str,
      adb_server_port: int,
      serial: str,
      console_port: int,
      grpc_port: int,
  ):
    self.adb_path = adb_path
    self.adb_server_port = adb_server_port
    self.serial = serial
    self.console_port = console_port
    self.grpc_port = grpc_port
    self.cases: list[dict[str, Any]] = []
    os.environ["ADB_SERVER_PORT"] = str(adb_server_port)
    os.environ["ANDROID_ADB_SERVER_PORT"] = str(adb_server_port)
    os.environ["ANDROID_SERIAL"] = serial
    self.env = env_launcher.load_and_setup_env(
        console_port=console_port,
        grpc_port=grpc_port,
        emulator_setup=False,
        freeze_datetime=False,
        adb_path=adb_path,
    )

  def close(self) -> None:
    self.env.close()

  def adb(
      self,
      *args: str,
      check: bool = True,
      text: bool = True,
  ) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        [
            self.adb_path,
            "-P",
            str(self.adb_server_port),
            "-s",
            self.serial,
            *args,
        ],
        check=check,
        capture_output=True,
        text=text,
        timeout=60,
    )

  def shell(self, *args: str, check: bool = True) -> str:
    return self.adb("shell", *args, check=check).stdout.strip()

  def sql(self, statement: str) -> str:
    command = (
        f"sqlite3 {shlex.quote(DB_PATH)} {shlex.quote(statement)}"
    )
    return self.shell(command)

  def tap(self, x: int, y: int, wait: float = 0.8) -> None:
    self.shell("input", "tap", str(x), str(y))
    time.sleep(wait)

  def tap_timer_card_action(self, action: str, wait: float = 1.0) -> None:
    """Tap Clock You's edit/delete/run-toggle card action by live geometry."""
    ui = self.env.get_state(wait_to_stabilize=True).ui_elements
    buttons = []
    for element in ui:
      if (
          element.package_name != PACKAGE
          or element.class_name != "android.widget.Button"
          or not element.is_clickable
          or element.bbox_pixels is None
      ):
        continue
      bbox = element.bbox_pixels
      x_center = (bbox.x_min + bbox.x_max) / 2
      y_center = (bbox.y_min + bbox.y_max) / 2
      if x_center >= 500 and 250 <= y_center <= 650:
        buttons.append((x_center, y_center))
    buttons.sort()
    if len(buttons) != 3:
      raise RuntimeError(
          "Clock You timer card did not expose exactly three actions: "
          f"{buttons}"
      )
    try:
      index = {"edit": 0, "delete": 1, "toggle": 2}[action]
    except KeyError as exc:
      raise ValueError(f"unknown Clock You timer action: {action}") from exc
    x_center, y_center = buttons[index]
    self.tap(round(x_center), round(y_center), wait=wait)

  def swipe(
      self,
      x1: int,
      y1: int,
      x2: int,
      y2: int,
      duration_ms: int = 400,
      wait: float = 0.8,
  ) -> None:
    self.shell(
        "input",
        "swipe",
        str(x1),
        str(y1),
        str(x2),
        str(y2),
        str(duration_ms),
    )
    time.sleep(wait)

  def launch_tab(self, tab: str) -> None:
    self.shell("am", "force-stop", PACKAGE)
    self.shell(
        "monkey",
        "-p",
        PACKAGE,
        "-c",
        "android.intent.category.LAUNCHER",
        "1",
    )
    time.sleep(1.0)
    tab_x = {"alarm": 125, "clock": 405, "timer": 680, "stopwatch": 950}
    self.tap(tab_x[tab], 2200, wait=1.0)

  def task(self, class_name: str, params: dict[str, Any]):
    task_class = getattr(clock_tasks, class_name)
    task = task_class(params)
    # This audit controls the state fixture directly.  Production lifecycle
    # behavior is covered separately; setting this flag only enables the real
    # task's fail-closed `is_successful` method.
    task.initialized = True
    return task

  def score(self, task: Any) -> float:
    return float(task.is_successful(self.env))

  def record(
      self,
      *,
      template: str,
      case: str,
      expected: float,
      actual: float,
      state: str,
  ) -> None:
    self.cases.append({
        "template": template,
        "case": case,
        "state": state,
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
    })

  def set_alarms(self, rows: list[tuple[int, int, bool]]) -> None:
    self.shell("am", "force-stop", PACKAGE)
    self.sql("DELETE FROM alarms;")
    for hour, minute, enabled in rows:
      time_ms = (hour * 3600 + minute * 60) * 1000
      self.sql(
          "INSERT INTO alarms"
          " (time,label,enabled,days,vibrate,soundName,soundUri,repeat)"
          f" VALUES ({time_ms},'',{1 if enabled else 0},"
          "'0,1,2,3,4,5,6',0,'','',0);"
      )

  def set_world_clocks(self, cities: list[str]) -> None:
    rows = {
        "Berlin": (
            "Europe/Berlin,Berlin,Germany (Deutschland)",
            "Europe/Berlin",
            "Germany (Deutschland)",
        ),
        "Sydney": (
            "Australia/Sydney,Sydney,Australia",
            "Australia/Sydney",
            "Australia",
        ),
    }
    self.shell("am", "force-stop", PACKAGE)
    self.sql("DELETE FROM timeZones;")
    for city in cities:
      key, zone_id, country = rows[city]
      self.sql(
          "INSERT INTO timeZones (key,zoneId,zoneName,countryName) VALUES "
          f"('{key}','{zone_id}','{city}','{country}');"
      )

  def verify_identity(self) -> dict[str, Any]:
    package_dump = self.shell("dumpsys", "package", PACKAGE)
    version_name = ""
    version_code = ""
    for raw_line in package_dump.splitlines():
      line = raw_line.strip()
      if line.startswith("versionName=") and not version_name:
        version_name = line.split("=", 1)[1]
      if line.startswith("versionCode=") and not version_code:
        version_code = line.split("=", 1)[1].split()[0]
    path_output = self.shell("pm", "path", PACKAGE)
    apk_path = path_output.split("package:", 1)[-1].strip()
    payload = self.adb(
        "exec-out", "cat", apk_path, text=False
    ).stdout
    apk_sha256 = hashlib.sha256(payload).hexdigest()
    identity = {
        "package_name": PACKAGE,
        "version_name": version_name,
        "version_code": version_code,
        "apk_path": apk_path,
        "apk_sha256": apk_sha256,
        "valid": (
            version_name == EXPECTED_VERSION_NAME
            and version_code == EXPECTED_VERSION_CODE
            and apk_sha256 == EXPECTED_APK_SHA256
        ),
    }
    if not identity["valid"]:
      raise RuntimeError(f"Clock You identity mismatch: {identity}")
    if self.shell("getprop", "sys.boot_completed") != "1":
      raise RuntimeError("Android device is not fully booted")
    if not self.sql(
        "SELECT COUNT(*) FROM sqlite_master WHERE name IN ('alarms','timeZones');"
    ).strip().endswith("2"):
      raise RuntimeError("Clock You Room schema is unavailable")
    return identity

  def ensure_app_storage(self) -> None:
    self.shell(
        "pm",
        "grant",
        "--user",
        "0",
        PACKAGE,
        "android.permission.POST_NOTIFICATIONS",
        check=False,
    )
    self.launch_tab("alarm")
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
      if self.shell("test", "-f", DB_PATH, check=False) == "":
        try:
          count = self.sql(
              "SELECT COUNT(*) FROM sqlite_master "
              "WHERE name IN ('alarms','timeZones');"
          )
          if count.strip().endswith("2"):
            return
        except subprocess.CalledProcessError:
          pass
      time.sleep(0.5)
    raise RuntimeError("Clock You did not initialize its Room schema")

  def audit_lifecycle(self) -> None:
    fixtures = [
        ("ClockCreateAlarm", "ClockCreateAlarmForClockYou", {"hour": 6, "minute": 45}),
        (
            "ClockEditAlarm",
            "ClockEditAlarmForClockYou",
            {
                "old_hour": 7,
                "old_minute": 15,
                "new_hour": 18,
                "new_minute": 45,
            },
        ),
        ("ClockEnableAlarm", "ClockEnableAlarmForClockYou", {"hour": 8, "minute": 30}),
        ("ClockDeleteAlarm", "ClockDeleteAlarmForClockYou", {"hour": 9, "minute": 15}),
        (
            "ClockCreateTimer",
            "ClockCreateTimerForClockYou",
            {"hours": 0, "minutes": 2, "seconds": 0},
        ),
        (
            "ClockStartTimer",
            "ClockStartTimerForClockYou",
            {"hours": 0, "minutes": 2, "seconds": 0},
        ),
        ("ClockStopwatchRunning", "ClockStopwatchRunningForClockYou", {}),
        ("ClockPauseStopwatch", "ClockPauseStopwatchForClockYou", {}),
        ("ClockStopwatchReset", "ClockStopwatchResetForClockYou", {}),
        ("ClockAddWorldClock", "ClockAddWorldClockForClockYou", {"city": "Berlin"}),
    ]
    for semantic_template, class_name, params in fixtures:
      task_class = getattr(clock_tasks, class_name)
      task = task_class(params)
      try:
        task.initialize_task(self.env)
        time.sleep(1.0)
        current_focus = self.shell("dumpsys", "window", "windows")
        if PACKAGE not in current_focus:
          raise RuntimeError(
              f"{class_name} did not launch {PACKAGE}"
          )
        package_dump = self.shell("dumpsys", "package", PACKAGE)
        if not (
            "android.permission.POST_NOTIFICATIONS: granted=true"
            in package_dump
        ):
          raise RuntimeError(
              f"{class_name} did not grant notification permission"
          )
        initial_score = self.score(task)
        self.record(
            template=semantic_template,
            case="lifecycle_fresh_reset",
            expected=0.0,
            actual=initial_score,
            state="initialize_task reset + permission grant + target launch",
        )
      finally:
        if task.initialized:
          task.tear_down(self.env)
    self.ensure_app_storage()

  def audit_alarm_templates(self) -> None:
    create = self.task(
        "ClockCreateAlarmForClockYou", {"hour": 6, "minute": 45}
    )
    self.set_alarms([])
    self.record(
        template="ClockCreateAlarm",
        case="no_op_empty",
        expected=0.0,
        actual=self.score(create),
        state="no alarms",
    )
    self.set_alarms([(6, 30, True)])
    self.record(
        template="ClockCreateAlarm",
        case="near_miss_wrong_minute",
        expected=0.0,
        actual=self.score(create),
        state="06:30 enabled",
    )
    self.set_alarms([(6, 45, True)])
    self.record(
        template="ClockCreateAlarm",
        case="positive_exact",
        expected=1.0,
        actual=self.score(create),
        state="06:45 enabled",
    )

    edit = self.task(
        "ClockEditAlarmForClockYou",
        {
            "old_hour": 7,
            "old_minute": 15,
            "new_hour": 18,
            "new_minute": 45,
        },
    )
    self.set_alarms([(7, 15, True)])
    self.record(
        template="ClockEditAlarm",
        case="no_op_old_only",
        expected=0.0,
        actual=self.score(edit),
        state="07:15 only",
    )
    self.set_alarms([(7, 15, True), (18, 45, True)])
    self.record(
        template="ClockEditAlarm",
        case="near_miss_old_and_new",
        expected=0.0,
        actual=self.score(edit),
        state="07:15 and 18:45",
    )
    self.set_alarms([(18, 45, True)])
    self.record(
        template="ClockEditAlarm",
        case="positive_replaced",
        expected=1.0,
        actual=self.score(edit),
        state="18:45 only",
    )

    enable = self.task(
        "ClockEnableAlarmForClockYou", {"hour": 8, "minute": 30}
    )
    self.set_alarms([])
    self.record(
        template="ClockEnableAlarm",
        case="no_op_empty",
        expected=0.0,
        actual=self.score(enable),
        state="no alarms",
    )
    self.set_alarms([(8, 30, False)])
    self.record(
        template="ClockEnableAlarm",
        case="near_miss_disabled",
        expected=0.0,
        actual=self.score(enable),
        state="08:30 disabled",
    )
    self.set_alarms([(8, 30, True)])
    self.record(
        template="ClockEnableAlarm",
        case="positive_enabled",
        expected=1.0,
        actual=self.score(enable),
        state="08:30 enabled",
    )

    delete = self.task(
        "ClockDeleteAlarmForClockYou", {"hour": 9, "minute": 15}
    )
    self.set_alarms([])
    self.record(
        template="ClockDeleteAlarm",
        case="no_op_initially_absent",
        expected=0.0,
        actual=self.score(delete),
        state="target never created",
    )
    self.set_alarms([(9, 15, True)])
    self.record(
        template="ClockDeleteAlarm",
        case="partial_created_not_deleted",
        expected=0.0,
        actual=self.score(delete),
        state="09:15 exists; evaluator latches creation",
    )
    self.set_alarms([])
    self.record(
        template="ClockDeleteAlarm",
        case="positive_created_then_deleted",
        expected=1.0,
        actual=self.score(delete),
        state="09:15 absent after observed creation",
    )
    wrong_delete = self.task(
        "ClockDeleteAlarmForClockYou", {"hour": 9, "minute": 15}
    )
    self.set_alarms([(9, 30, True)])
    self.record(
        template="ClockDeleteAlarm",
        case="near_miss_wrong_alarm_only",
        expected=0.0,
        actual=self.score(wrong_delete),
        state="09:30 exists; target never created",
    )

  def audit_world_clock(self) -> None:
    task = self.task("ClockAddWorldClockForClockYou", {"city": "Berlin"})
    self.set_world_clocks([])
    self.record(
        template="ClockAddWorldClock",
        case="no_op_empty",
        expected=0.0,
        actual=self.score(task),
        state="no persisted time zones",
    )
    self.set_world_clocks(["Sydney"])
    self.record(
        template="ClockAddWorldClock",
        case="near_miss_wrong_city",
        expected=0.0,
        actual=self.score(task),
        state="Sydney only",
    )
    self.set_world_clocks(["Berlin"])
    self.record(
        template="ClockAddWorldClock",
        case="positive_exact_city",
        expected=1.0,
        actual=self.score(task),
        state="Berlin persisted as Europe/Berlin",
    )

  def set_timer_minutes(self, minutes: int) -> None:
    if not 1 <= minutes <= 4:
      raise ValueError("fixture supports timer minutes 1..4")
    self.swipe(540, 1000, 540, 1000 - 210 * minutes)

  def reset_timer_picker(self) -> None:
    """Cancel any persisted timer card and normalize the picker to zero."""
    self.launch_tab("timer")
    ui = self.env.get_state(wait_to_stabilize=True).ui_elements
    picker = clock_tasks._clock_you_timer_picker_duration(ui)
    if picker is None:
      # Clock You's middle timer-card action is Cancel/Delete. Resolve its
      # center from the live tree because the card shifts vertically between
      # running and paused states.
      self.tap_timer_card_action("delete", wait=1.0)
      ui = self.env.get_state(wait_to_stabilize=True).ui_elements
      picker = clock_tasks._clock_you_timer_picker_duration(ui)
    if picker is None:
      raise RuntimeError("Clock You did not return to the timer picker")
    wheel_x = (295, 540, 785)
    for column, selected in enumerate(picker):
      remaining = selected
      while remaining:
        step = min(remaining, 3)
        self.swipe(
            wheel_x[column],
            950,
            wheel_x[column],
            950 + 210 * step,
            wait=0.5,
        )
        remaining -= step
    normalized = clock_tasks._clock_you_timer_picker_duration(
        self.env.get_state(wait_to_stabilize=True).ui_elements
    )
    if normalized != (0, 0, 0):
      raise RuntimeError(f"Timer picker reset failed: {normalized}")

  def audit_timer_templates(self) -> None:
    create = self.task(
        "ClockCreateTimerForClockYou",
        {"hours": 0, "minutes": 2, "seconds": 0},
    )
    self.reset_timer_picker()
    self.record(
        template="ClockCreateTimer",
        case="no_op_zero_picker",
        expected=0.0,
        actual=self.score(create),
        state="00:00:00 picker",
    )
    self.set_timer_minutes(3)
    self.record(
        template="ClockCreateTimer",
        case="near_miss_wrong_duration",
        expected=0.0,
        actual=self.score(create),
        state="00:03:00 picker",
    )
    self.reset_timer_picker()
    self.set_timer_minutes(2)
    self.record(
        template="ClockCreateTimer",
        case="positive_exact_picker",
        expected=1.0,
        actual=self.score(create),
        state="00:02:00 picker; not started",
    )

    start = self.task(
        "ClockStartTimerForClockYou",
        {"hours": 0, "minutes": 2, "seconds": 0},
    )
    self.reset_timer_picker()
    self.record(
        template="ClockStartTimer",
        case="no_op_zero_picker",
        expected=0.0,
        actual=self.score(start),
        state="00:00:00 picker",
    )
    self.set_timer_minutes(2)
    self.tap(965, 2010, wait=2.0)
    self.record(
        template="ClockStartTimer",
        case="positive_running",
        expected=1.0,
        actual=self.score(start),
        state="countdown plus projected end time",
    )
    self.tap_timer_card_action("toggle", wait=1.0)
    self.record(
        template="ClockStartTimer",
        case="near_miss_paused",
        expected=0.0,
        actual=self.score(start),
        state="countdown paused; projected end time absent",
    )

  def audit_stopwatch_templates(self) -> None:
    running = self.task("ClockStopwatchRunningForClockYou", {})
    self.launch_tab("stopwatch")
    self.record(
        template="ClockStopwatchRunning",
        case="no_op_initial_zero",
        expected=0.0,
        actual=self.score(running),
        state="one action button",
    )
    self.tap(540, 1950, wait=1.5)
    self.record(
        template="ClockStopwatchRunning",
        case="positive_running",
        expected=1.0,
        actual=self.score(running),
        state="three action buttons",
    )
    self.tap(540, 1950, wait=1.0)
    self.record(
        template="ClockStopwatchRunning",
        case="near_miss_paused",
        expected=0.0,
        actual=self.score(running),
        state="two action buttons",
    )

    paused = self.task("ClockPauseStopwatchForClockYou", {})
    self.launch_tab("stopwatch")
    self.record(
        template="ClockPauseStopwatch",
        case="no_op_initial_zero",
        expected=0.0,
        actual=self.score(paused),
        state="one action button",
    )
    self.tap(540, 1950, wait=1.5)
    self.record(
        template="ClockPauseStopwatch",
        case="near_miss_still_running",
        expected=0.0,
        actual=self.score(paused),
        state="three action buttons",
    )
    self.tap(540, 1950, wait=1.0)
    self.record(
        template="ClockPauseStopwatch",
        case="positive_paused",
        expected=1.0,
        actual=self.score(paused),
        state="two action buttons",
    )

    reset = self.task("ClockStopwatchResetForClockYou", {})
    self.launch_tab("stopwatch")
    self.record(
        template="ClockStopwatchReset",
        case="no_op_initial_zero",
        expected=0.0,
        actual=self.score(reset),
        state="fresh one-button zero; no run latch",
    )
    self.tap(540, 1950, wait=1.5)
    self.record(
        template="ClockStopwatchReset",
        case="partial_running_not_reset",
        expected=0.0,
        actual=self.score(reset),
        state="three action buttons; run latched",
    )
    self.tap(540, 1950, wait=1.0)
    self.record(
        template="ClockStopwatchReset",
        case="partial_paused_not_reset",
        expected=0.0,
        actual=self.score(reset),
        state="two action buttons; run remains latched",
    )
    self.tap(690, 1950, wait=1.0)
    self.record(
        template="ClockStopwatchReset",
        case="positive_run_pause_reset",
        expected=1.0,
        actual=self.score(reset),
        state="one-button zero after observed run",
    )

  def cleanup(self) -> None:
    self.set_alarms([])
    self.set_world_clocks([])
    self.shell("am", "force-stop", PACKAGE)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--adb", default=os.environ.get("ADB", "adb"))
  parser.add_argument("--adb_server_port", type=int, required=True)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--console_port", type=int, required=True)
  parser.add_argument("--grpc_port", type=int, required=True)
  parser.add_argument("--docker_image_digest", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()

  audit = LiveAudit(
      adb_path=args.adb,
      adb_server_port=args.adb_server_port,
      serial=args.serial,
      console_port=args.console_port,
      grpc_port=args.grpc_port,
  )
  try:
    audit.ensure_app_storage()
    identity = audit.verify_identity()
    audit.audit_lifecycle()
    audit.audit_alarm_templates()
    audit.audit_world_clock()
    audit.audit_timer_templates()
    audit.audit_stopwatch_templates()
    audit.cleanup()
  finally:
    audit.close()

  templates = sorted({case["template"] for case in audit.cases})
  failed = [case for case in audit.cases if not case["passed"]]
  payload = {
      "audit_type": "clock_you_real_device_verifier_conformance",
      "artifact_role": "harness_conformance_only_not_model_result",
      "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "device": {
          "serial": args.serial,
          "console_port": args.console_port,
          "grpc_port": args.grpc_port,
          "adb_server_port": args.adb_server_port,
          "android_release": audit.shell("getprop", "ro.build.version.release"),
          "api_level": audit.shell("getprop", "ro.build.version.sdk"),
          "fingerprint": audit.shell("getprop", "ro.build.fingerprint"),
          "docker_image_digest": args.docker_image_digest,
      },
      "app_identity": identity,
      "expected_template_count": 10,
      "tested_templates": templates,
      "case_count": len(audit.cases),
      "failed_case_count": len(failed),
      "all_templates_qualified": len(templates) == 10 and not failed,
      "cases": audit.cases,
  }
  output = Path(args.output).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  print(json.dumps({
      "tested_templates": len(templates),
      "case_count": len(audit.cases),
      "failed_case_count": len(failed),
      "all_templates_qualified": payload["all_templates_qualified"],
      "failed_cases": [
          f"{case['template']}::{case['case']}" for case in failed
      ],
  }, indent=2))
  return 0 if payload["all_templates_qualified"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
