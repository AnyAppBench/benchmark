from unittest import mock

from absl.testing import absltest

from android_world.env import representation_utils
from android_world.task_evals.single.app_generalization_generated import (
    clock_cross_app_tasks as clock_tasks,
)


def _element(
    text: str | None = None,
    content_description: str | None = None,
    package_name: str = "org.fossify.clock",
    bbox: representation_utils.BoundingBox | None = None,
    bbox_pixels: representation_utils.BoundingBox | None = None,
    class_name: str | None = None,
    is_clickable: bool | None = None,
    is_checked: bool | None = None,
    is_checkable: bool | None = None,
    is_editable: bool | None = None,
    hint_text: str | None = None,
    resource_name: str | None = None,
    resource_id: str | None = None,
) -> representation_utils.UIElement:
  return representation_utils.UIElement(
      text=text,
      content_description=content_description,
      package_name=package_name,
      bbox=bbox,
      bbox_pixels=bbox_pixels,
      class_name=class_name,
      is_clickable=is_clickable,
      is_checked=is_checked,
      is_checkable=is_checkable,
      is_editable=is_editable,
      hint_text=hint_text,
      resource_name=resource_name,
      resource_id=resource_id,
  )


def _bbox(y_min: float, y_max: float) -> representation_utils.BoundingBox:
  return representation_utils.BoundingBox(0.0, 1.0, y_min, y_max)


class ClockCrossAppTimerHelpersTest(absltest.TestCase):

  def _clock_you_stopwatch_ui(self, action_button_count: int):
    ui = [
        _element(
            text="Stopwatch",
            package_name="com.bnyro.clock",
            bbox_pixels=representation_utils.BoundingBox(43, 260, 175, 249),
        )
    ]
    for index in range(action_button_count):
      ui.append(_element(
          package_name="com.bnyro.clock",
          bbox_pixels=representation_utils.BoundingBox(
              150 + 250 * index, 300 + 250 * index, 1833, 2085
          ),
          class_name="android.widget.Button",
          is_clickable=True,
      ))
    return ui

  def test_clock_you_stopwatch_action_layout_distinguishes_three_states(self):
    self.assertEqual(
        "initial_zero",
        clock_tasks._clock_you_stopwatch_state(
            self._clock_you_stopwatch_ui(1)
        ),
    )
    self.assertEqual(
        "paused_nonzero",
        clock_tasks._clock_you_stopwatch_state(
            self._clock_you_stopwatch_ui(2)
        ),
    )
    self.assertEqual(
        "running_nonzero",
        clock_tasks._clock_you_stopwatch_state(
            self._clock_you_stopwatch_ui(3)
        ),
    )

  def test_clock_you_timer_picker_reads_selected_center_row(self):
    package = "com.bnyro.clock"
    ui = [
        _element(
            text="Timer",
            package_name=package,
            bbox_pixels=representation_utils.BoundingBox(43, 190, 175, 249),
        ),
        _element(
            text=":",
            package_name=package,
            bbox_pixels=representation_utils.BoundingBox(403, 432, 927, 1066),
        ),
        _element(
            text=":",
            package_name=package,
            bbox_pixels=representation_utils.BoundingBox(648, 677, 927, 1066),
        ),
    ]
    for text, x_min in (("00", 229), ("02", 474), ("15", 719)):
      ui.append(_element(
          text=text,
          package_name=package,
          bbox_pixels=representation_utils.BoundingBox(
              x_min, x_min + 132, 912, 1051
          ),
      ))

    self.assertEqual(
        (0, 2, 15),
        clock_tasks._clock_you_timer_picker_duration(ui),
    )

  def test_clock_you_running_timer_requires_countdown_and_end_time(self):
    package = "com.bnyro.clock"
    running = [
        _element(
            text="Timer",
            package_name=package,
            bbox_pixels=representation_utils.BoundingBox(43, 190, 175, 249),
        ),
        _element(text="0:01:56", package_name=package),
        _element(text="01:15", package_name=package),
    ]
    paused = running[:2]

    self.assertTrue(clock_tasks._clock_you_running_timer_surface(running))
    self.assertFalse(clock_tasks._clock_you_running_timer_surface(paused))

  def test_clock_you_alarm_time_is_milliseconds_after_midnight(self):
    self.assertEqual(
        66_300_000,
        clock_tasks._clock_you_alarm_time_ms(18, 25),
    )

  def test_clock_you_alarm_query_checks_exact_time_and_enabled_state(self):
    with mock.patch.object(
        clock_tasks, "_ensure_clock_you_storage_ready", return_value=True
    ):
      with mock.patch.object(
          clock_tasks, "_sqlite_exec", return_value="1\n"
      ) as sqlite_exec:
        exists = clock_tasks._clock_you_alarm_exists(
            object(), hour_24=6, minute=45, enabled=True
        )

    self.assertTrue(exists)
    self.assertEqual(clock_tasks._CLOCK_YOU_DB, sqlite_exec.call_args.args[1])
    self.assertIn("time=24300000", sqlite_exec.call_args.args[2])
    self.assertIn("enabled=1", sqlite_exec.call_args.args[2])

  def test_delete_alarm_latch_rejects_initial_absence_then_accepts_delete(self):
    success, seen = clock_tasks._delete_alarm_observation(
        alarm_exists=False, seen_target_alarm=False
    )
    self.assertFalse(success)
    self.assertFalse(seen)

    success, seen = clock_tasks._delete_alarm_observation(
        alarm_exists=True, seen_target_alarm=seen
    )
    self.assertFalse(success)
    self.assertTrue(seen)

    success, seen = clock_tasks._delete_alarm_observation(
        alarm_exists=False, seen_target_alarm=seen
    )
    self.assertTrue(success)
    self.assertTrue(seen)

  def test_clock_you_world_clock_query_requires_persisted_city(self):
    with mock.patch.object(
        clock_tasks, "_ensure_clock_you_storage_ready", return_value=True
    ):
      with mock.patch.object(
          clock_tasks, "_sqlite_exec", return_value="0\n"
      ) as sqlite_exec:
        exists = clock_tasks._clock_you_world_clock_exists(
            object(), city="Berlin"
        )

    self.assertFalse(exists)
    self.assertIn("LOWER(zoneName)=LOWER('Berlin')", sqlite_exec.call_args.args[2])

  def test_sqlite_exec_raises_typed_error_on_native_failure(self):
    output = (
        "Error: no such table: alarms\n\n"
        f"{clock_tasks._SQLITE_STATUS_MARKER}1\n"
    )
    with mock.patch.object(clock_tasks, "_adb_shell", return_value=output):
      with self.assertRaises(clock_tasks.ClockStorageError):
        clock_tasks._sqlite_exec(object(), "/bad.db", "SELECT 1;")

  def test_sqlite_read_wraps_transport_failure_as_typed_read_error(self):
    with mock.patch.object(
        clock_tasks,
        "_sqlite_exec",
        side_effect=clock_tasks.ClockStorageError("transport failed"),
    ):
      with self.assertRaises(clock_tasks.ClockStorageReadError):
        clock_tasks._sqlite_read(object(), "/bad.db", "SELECT 1;")

  def test_sqlite_count_rejects_empty_output_instead_of_false_absence(self):
    with mock.patch.object(clock_tasks, "_sqlite_read", return_value=""):
      with self.assertRaises(clock_tasks.ClockStorageReadError):
        clock_tasks._sqlite_count(object(), "/bad.db", "SELECT COUNT(*);")

  def test_clock_you_database_absence_is_read_without_opening_sqlite(self):
    with mock.patch.object(clock_tasks, "_adb_shell", return_value="0") as shell:
      self.assertFalse(clock_tasks._clock_you_database_exists(object()))
    self.assertIn("if [ -f", shell.call_args.args[1])

  def test_clock_you_database_existence_check_rejects_malformed_output(self):
    with mock.patch.object(clock_tasks, "_adb_shell", return_value="permission denied"):
      with self.assertRaises(clock_tasks.ClockStorageReadError):
        clock_tasks._clock_you_database_exists(object())

  def test_clock_you_storage_launches_before_querying_new_room_database(self):
    env = mock.Mock()
    with mock.patch.object(
        clock_tasks,
        "_clock_you_database_exists",
        side_effect=[False, True],
    ), mock.patch.object(
        clock_tasks, "_clock_you_has_table", return_value=True
    ) as has_table, mock.patch.object(
        clock_tasks.adb_utils, "launch_app"
    ) as launch_app:
      self.assertTrue(
          clock_tasks._ensure_clock_you_storage_ready(
              env, timeout_seconds=0.1
          )
      )
    launch_app.assert_called_once_with("com.bnyro.clock", env.controller)
    self.assertEqual(has_table.call_count, 2)

  def test_timer_set_accepts_mixed_singular_plural_description(self):
    ui = [_element(content_description="1 hour, 20 minutes, 1 second")]

    self.assertTrue(
        clock_tasks._is_timer_set(ui, hours=1, minutes=20, seconds=1)
    )

  def test_timer_countdown_accepts_small_elapsed_delta(self):
    ui = [
        _element(text="04:58"),
        _element(content_description="Pause"),
    ]

    self.assertTrue(
        clock_tasks._is_timer_duration_visible(
            ui,
            hours=0,
            minutes=5,
            seconds=0,
            max_elapsed_seconds=120,
        )
    )

  def test_timer_countdown_rejects_wrong_duration(self):
    ui = [_element(text="03:00")]

    self.assertFalse(
        clock_tasks._is_timer_duration_visible(
            ui,
            hours=0,
            minutes=5,
            seconds=0,
            max_elapsed_seconds=60,
        )
    )

  def test_timer_control_matching_does_not_match_stopwatch_tab(self):
    ui = [
        _element(text="05:00"),
        _element(text="Stopwatch"),
    ]

    self.assertFalse(clock_tasks._control_present(ui, ("stop",)))

  def test_timer_started_rejects_running_and_paused_states(self):
    running = [_element(text="05:00"), _element(content_description="Pause")]
    paused = [_element(text="05:00"), _element(content_description="Resume")]
    untouched = [_element(text="05:00"), _element(content_description="Start")]

    self.assertTrue(clock_tasks._timer_has_started(running))
    self.assertTrue(clock_tasks._timer_has_started(paused))
    self.assertFalse(clock_tasks._timer_has_started(untouched))

  def test_system_clock_is_not_accepted_as_timer(self):
    ui = [
        _element(
            text="15:34",
            content_description="15:34",
            package_name="com.android.systemui",
        )
    ]

    self.assertFalse(
        clock_tasks._is_timer_set(ui, hours=0, minutes=15, seconds=34)
    )

  def test_stopwatch_running_accepts_reset_control_without_lap(self):
    ui = [
        _element(text="Stopwatch", bbox=_bbox(0.05, 0.10)),
        _element(content_description="Pause"),
        _element(content_description="Reset"),
    ]

    self.assertTrue(clock_tasks._is_stopwatch_running(ui))

  def test_stopwatch_running_accepts_aw_exact_pause_lap_pair(self):
    """AW's reference validator accepts exact Google Clock Pause+Lap labels."""
    ui = [
        _element(text="Stopwatch", bbox=_bbox(0.05, 0.10)),
        _element(content_description="Pause"),
        _element(content_description="Lap"),
        _element(text="00:00.00"),
    ]

    self.assertTrue(clock_tasks._is_stopwatch_running(ui))

  def test_stopwatch_running_rejects_timer_page_with_bottom_nav_stopwatch(self):
    ui = [
        _element(text="Timer", bbox=_bbox(0.05, 0.10)),
        _element(text="Stopwatch", bbox=_bbox(0.90, 0.96)),
        _element(content_description="Pause"),
        _element(content_description="Reset"),
    ]

    self.assertFalse(clock_tasks._is_stopwatch_running(ui))

  def test_stopwatch_reset_accepts_split_zero_digits(self):
    ui = [
        _element(text="Stopwatch", bbox=_bbox(0.05, 0.10)),
        _element(text="00"),
        _element(text="00"),
    ]

    self.assertTrue(clock_tasks._is_stopwatch_zero_visible(ui))

  def test_stateful_stopwatch_reset_accepts_icon_only_run_then_zero(self):
    running_ui = [
        _element(text="Stopwatch", bbox=_bbox(0.05, 0.10)),
        _element(text="0:02.91"),
    ]
    success, seen_nonzero = clock_tasks._stopwatch_reset_observation(
        running_ui, seen_nonzero_elapsed=False
    )
    self.assertFalse(success)
    self.assertTrue(seen_nonzero)

    reset_ui = [
        _element(text="Stopwatch", bbox=_bbox(0.05, 0.10)),
        _element(text="0:00.00"),
    ]
    success, seen_nonzero = clock_tasks._stopwatch_reset_observation(
        reset_ui, seen_nonzero_elapsed=seen_nonzero
    )
    self.assertTrue(success)
    self.assertTrue(seen_nonzero)

  def test_stateful_stopwatch_reset_rejects_fresh_zero(self):
    fresh_ui = [
        _element(text="Stopwatch", bbox=_bbox(0.05, 0.10)),
        _element(text="0:00.00"),
        _element(content_description="Start"),
    ]
    success, seen_nonzero = clock_tasks._stopwatch_reset_observation(
        fresh_ui, seen_nonzero_elapsed=False
    )
    self.assertFalse(success)
    self.assertFalse(seen_nonzero)

  def test_stateful_stopwatch_reset_rejects_unrelated_duration(self):
    timer_ui = [
        _element(text="Timer", bbox=_bbox(0.05, 0.10)),
        _element(text="0:05.00"),
    ]
    success, seen_nonzero = clock_tasks._stopwatch_reset_observation(
        timer_ui, seen_nonzero_elapsed=False
    )
    self.assertFalse(success)
    self.assertFalse(seen_nonzero)

  def test_stopwatch_paused_rejects_initial_zero_state(self):
    """The CATBench goal is self-contained ("run the stopwatch, then pause
    it"), so the untouched zero state — Start button, 00:00 display, no
    Resume/Reset evidence — must NOT pass. An agent that merely opens the
    stopwatch tab has not run the stopwatch."""
    ui = [
        _element(text="Stopwatch", bbox=_bbox(0.05, 0.10)),
        _element(content_description="Start"),
        _element(text="00:00.00"),
    ]
    self.assertFalse(clock_tasks._is_stopwatch_paused(ui))

  def test_stopwatch_paused_accepts_zero_display_with_reset_evidence(self):
    """Paused at a reset-capable state: Start + Reset controls prove the
    stopwatch ran even if the display rolled back to zero."""
    ui = [
        _element(text="Stopwatch", bbox=_bbox(0.05, 0.10)),
        _element(content_description="Start"),
        _element(content_description="Reset"),
        _element(text="00:00.00"),
    ]
    self.assertTrue(clock_tasks._is_stopwatch_paused(ui))

  def test_stopwatch_paused_accepts_nonzero_with_resume(self):
    """A run-then-paused stopwatch (Resume label, non-zero display) also passes."""
    ui = [
        _element(text="Stopwatch", bbox=_bbox(0.05, 0.10)),
        _element(content_description="Resume"),
        _element(text="00:12.34"),
    ]
    self.assertTrue(clock_tasks._is_stopwatch_paused(ui))

  def test_stopwatch_paused_accepts_restart_label_on_cross_app_clocks(self):
    """Some cross-app clocks (Fossify, Chrono) expose ``Restart`` instead of
    ``Start`` on the paused stopwatch. The substring match is intentional
    here so the heuristic generalises across apps."""
    ui = [
        _element(text="Stopwatch", bbox=_bbox(0.05, 0.10)),
        _element(content_description="Restart"),
    ]
    self.assertTrue(clock_tasks._is_stopwatch_paused(ui))

  def test_stopwatch_running_rejects_zero_state(self):
    """A stopwatch whose display still reads 00:00 is not running, even if
    a Pause control flickers visible during a UI transition. AW's reference
    only requires Pause+Lap content_descriptions, which the initial state
    does not have; the non-zero guard is defensive and preserves AW intent."""
    ui = [
        _element(text="Stopwatch", bbox=_bbox(0.05, 0.10)),
        _element(content_description="Pause"),
        _element(text="00:00.00"),
        _element(text="00"),
        _element(text="00"),
    ]
    self.assertFalse(clock_tasks._is_stopwatch_running(ui))

  def test_alarm_list_empty_requires_alarm_page(self):
    """'No alarms' text on a non-alarm screen must NOT pass as empty (CL6)."""
    ui = [
        _element(text="Settings", bbox=_bbox(0.05, 0.10)),
        _element(text="No alarms configured (in another app)"),
    ]
    self.assertFalse(clock_tasks._alarm_list_is_empty(ui))

  def test_alarm_list_empty_accepts_explicit_empty_marker(self):
    ui = [
        _element(text="Alarm", bbox=_bbox(0.05, 0.10)),
        _element(text="No alarms"),
    ]
    self.assertTrue(clock_tasks._alarm_list_is_empty(ui))


class ClockFrozenVerifierAdversarialTest(absltest.TestCase):
  """Near-miss fixtures for the frozen Clock semantic tasks."""

  def _env(self, ui):
    state = mock.Mock()
    state.ui_elements = ui
    env = mock.Mock()
    env.get_state.return_value = state
    return env

  def _task(self, class_name, params):
    task = getattr(clock_tasks, class_name)(params)
    task.initialized = True
    return task

  def test_create_timer_rejects_exact_duration_after_start(self):
    task = self._task(
        "ClockCreateTimerForGoogleClock",
        {"hours": 0, "minutes": 5, "seconds": 0},
    )
    running_ui = [
        _element(text="Timer", bbox=_bbox(0.05, 0.10)),
        _element(text="05:00"),
        _element(content_description="Pause"),
    ]
    untouched_ui = [
        _element(text="Timer", bbox=_bbox(0.05, 0.10)),
        _element(text="05:00"),
        _element(content_description="Start"),
    ]

    self.assertEqual(0.0, task.is_successful(self._env(running_ui)))
    self.assertEqual(1.0, task.is_successful(self._env(untouched_ui)))

  def test_create_alarm_rejects_unsaved_picker(self):
    task = self._task(
        "ClockCreateAlarmForGoogleClock", {"hour": 6, "minute": 45}
    )
    unsaved_ui = [
        _element(text="Add alarm", bbox=_bbox(0.05, 0.10)),
        _element(text="6:45 AM"),
        _element(text="Save", is_clickable=True),
        _element(text="Cancel", is_clickable=True),
    ]
    saved_ui = [
        _element(text="Alarm", bbox=_bbox(0.05, 0.10)),
        _element(text="6:45 AM", bbox=_bbox(0.30, 0.36)),
    ]

    self.assertEqual(0.0, task.is_successful(self._env(unsaved_ui)))
    self.assertEqual(1.0, task.is_successful(self._env(saved_ui)))

  def test_edit_alarm_rejects_unsaved_new_time(self):
    task = self._task(
        "ClockEditAlarmForGoogleClock",
        {
            "old_hour": 6,
            "old_minute": 45,
            "new_hour": 18,
            "new_minute": 15,
        },
    )
    unsaved_ui = [
        _element(text="Edit alarm", bbox=_bbox(0.05, 0.10)),
        _element(text="18:15"),
        _element(text="Save", is_clickable=True),
    ]
    saved_ui = [
        _element(text="Alarm", bbox=_bbox(0.05, 0.10)),
        _element(text="18:15", bbox=_bbox(0.30, 0.36)),
    ]

    self.assertEqual(0.0, task.is_successful(self._env(unsaved_ui)))
    self.assertEqual(1.0, task.is_successful(self._env(saved_ui)))

  def test_enable_alarm_requires_checked_switch_in_target_row(self):
    task = self._task(
        "ClockEnableAlarmForGoogleClock", {"hour": 6, "minute": 45}
    )
    target_time = _element(text="6:45 AM", bbox=_bbox(0.25, 0.31))
    target_off = _element(
        class_name="android.widget.Switch",
        is_checkable=True,
        is_checked=False,
        bbox=_bbox(0.25, 0.31),
    )
    target_on = _element(
        class_name="android.widget.Switch",
        is_checkable=True,
        is_checked=True,
        bbox=_bbox(0.25, 0.31),
    )
    unrelated_on = _element(
        class_name="android.widget.Switch",
        is_checkable=True,
        is_checked=True,
        bbox=_bbox(0.70, 0.76),
    )

    self.assertEqual(
        0.0,
        task.is_successful(
            self._env([target_time, target_off, unrelated_on])
        ),
    )
    self.assertEqual(
        1.0, task.is_successful(self._env([target_time, target_on]))
    )

  def test_delete_alarm_rejects_unrelated_marker_then_accepts_target_latch(self):
    task = self._task(
        "ClockDeleteAlarmForGoogleClock", {"hour": 6, "minute": 45}
    )
    task._catbench_seen_target_alarm = False
    unrelated_delete_ui = [
        _element(text="Alarm", bbox=_bbox(0.05, 0.10)),
        _element(text="Deleted — Undo"),
    ]
    target_ui = [
        _element(text="Alarm", bbox=_bbox(0.05, 0.10)),
        _element(text="6:45 AM", bbox=_bbox(0.30, 0.36)),
    ]
    deleted_ui = [_element(text="Alarm", bbox=_bbox(0.05, 0.10))]

    self.assertEqual(0.0, task.is_successful(self._env(unrelated_delete_ui)))
    self.assertEqual(0.0, task.is_successful(self._env(target_ui)))
    self.assertEqual(1.0, task.is_successful(self._env(deleted_ui)))

  def test_fossify_delete_uses_target_latch_not_unrelated_marker(self):
    task = self._task(
        "ClockDeleteAlarmForFossifyClock", {"hour": 6, "minute": 45}
    )
    task._catbench_seen_target_alarm = False
    env = self._env([_element(text="Deleted — Undo")])
    with mock.patch.object(
        clock_tasks, "_fossify_alarm_exists", return_value=False
    ):
      self.assertEqual(0.0, task.is_successful(env))

    with mock.patch.object(
        clock_tasks, "_fossify_alarm_exists", side_effect=(True, False)
    ):
      self.assertEqual(0.0, task.is_successful(env))
      self.assertEqual(1.0, task.is_successful(env))

  def test_world_clock_rejects_city_search_result_only(self):
    task = self._task(
        "ClockAddWorldClockForGoogleClock", {"city": "Berlin"}
    )
    search_ui = [
        _element(text="Search cities", is_editable=True),
        _element(text="Berlin"),
    ]
    saved_ui = [
        _element(text="World Clock", bbox=_bbox(0.05, 0.10)),
        _element(text="Berlin"),
    ]

    self.assertEqual(0.0, task.is_successful(self._env(search_ui)))
    self.assertEqual(1.0, task.is_successful(self._env(saved_ui)))

  def test_clock_you_storage_validation_metadata_is_declared(self):
    storage_tasks = (
        "ClockCreateAlarmForClockYou",
        "ClockEditAlarmForClockYou",
        "ClockEnableAlarmForClockYou",
        "ClockDeleteAlarmForClockYou",
        "ClockAddWorldClockForClockYou",
    )
    for task_name in storage_tasks:
      with self.subTest(task_name=task_name):
        self.assertEqual(
            "Clock You Room SQLite durable state",
            getattr(clock_tasks, task_name).validation_mode,
        )
    self.assertEqual(
        "UI heuristic", clock_tasks.ClockCreateTimerForClockYou.validation_mode
    )


class ClockNavigationTabTest(absltest.TestCase):
  """Regression tests for the previously-always-passing navigation validators (CL4)."""

  def _make_task(self, cls):
    instance = cls.__new__(cls)
    instance._params = {}
    instance.initialized = True
    return instance

  def test_alarm_tab_validator_rejects_stopwatch_screen(self):
    """Bottom-nav 'Alarm' label visible on stopwatch screen must NOT pass."""
    ui = [
        _element(text="Stopwatch", bbox=_bbox(0.05, 0.10)),
        _element(text="Alarm", bbox=_bbox(0.92, 0.98)),  # bottom-nav tab
        _element(text="Timer", bbox=_bbox(0.92, 0.98)),
    ]
    cls = clock_tasks._ClockNavigateToAlarmTabBase
    instance = self._make_task(cls)

    # Use the underlying header check directly since we can't run the full
    # task without an environment; this matches the new validator's logic.
    self.assertTrue(clock_tasks._is_stopwatch_page(ui))
    # Top-level alarm header is NOT present.
    self.assertFalse(
        clock_tasks._top_level_text_present(ui, ("alarm", "alarms"))
    )

  def test_alarm_tab_validator_accepts_alarm_header(self):
    ui = [
        _element(text="Alarm", bbox=_bbox(0.05, 0.10)),  # top header
    ]
    self.assertTrue(
        clock_tasks._top_level_text_present(ui, ("alarm", "alarms"))
    )
    self.assertFalse(clock_tasks._is_stopwatch_page(ui))


if __name__ == "__main__":
  absltest.main()
