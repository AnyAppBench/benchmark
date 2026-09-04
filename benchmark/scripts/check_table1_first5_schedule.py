#!/usr/bin/env python3
"""Validate that the first-five runnable profiles match manuscript Table 1.

This script is intentionally strict: it checks the exact task class names that
should instantiate the Table 1 task templates for the first five categories.
It exits non-zero if a profile schedules a substitute task or if a Table 1 task
class is not registered as runnable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

from android_world import registry  # noqa: E402
from app_generalization_profiles import get_domain_profiles  # noqa: E402


@dataclass(frozen=True)
class AppSpec:
  app_id: str
  suffix: str
  aw_original: bool = False


TODO_AW_PREFIXES = (
    "TasksCompletedTasksForDate",
    "TasksDueNextWeek",
    "TasksDueOnDate",
    "TasksHighPriorityTasks",
    "TasksHighPriorityTasksDueOnDate",
    "TasksIncompleteTasksOnDate",
)

TODO_NEW_PREFIXES = (
    "TasksDueWithTime",
    "TasksRecurring",
    "TasksEditTask",
    "TasksCompleteTask",
)

NOTES_PREFIXES = (
    "NotesCreateNote",
    "NotesEditNote",
    "NotesMergeNotes",
    "NotesDeleteNote",
    "NotesSearchNote",
    "NotesShareImport",
    "NotesCreateFolder",
    "NotesAttachContent",
    "NotesCountTodoItems",
    "NotesCreateChecklist",
)

FINANCE_PREFIXES = (
    "ExpenseAddSingle",
    "ExpenseAddIncome",
    "ExpenseAddMultiple",
    "ExpenseEditExpense",
    "ExpenseDeleteSingle",
    "ExpenseDeleteDuplicates",
    "ExpenseCategorySummary",
    "ExpenseDateRangeTotal",
    "ExpenseAttachReceipt",
    "ExpenseTransferBetweenWallets",
)

MUSIC_PREFIXES = (
    "RetroCreatePlaylist",
    "RetroRenamePlaylist",
    "RetroAddToPlaylist",
    "RetroRemoveFromPlaylist",
    "RetroAddToQueue",
    "RetroReorderQueue",
    "RetroSavePlaylist",
    "RetroPlaylistDuration",
    "RetroSleepTimer",
    "RetroSearchAndPlay",
)

CALENDAR_AW_ORIGINAL = (
    "SimpleCalendarAddOneEvent",
    "SimpleCalendarAddRepeatingEvent",
    "SimpleCalendarDeleteEvents",
)

CALENDAR_PREFIXES = (
    "SimpleCalendarAddOneEvent",
    "SimpleCalendarAddTimedEvent",
    "SimpleCalendarAddRepeatingEvent",
    "SimpleCalendarDeleteEvents",
    "SimpleCalendarEventsOnDate",
    "SimpleCalendarNextEvent",
    "SimpleCalendarEventsInRange",
    "SimpleCalendarEditEvent",
    "SimpleCalendarAddReminder",
    "SimpleCalendarMoveEvent",
)

FIRST_FIVE_APPS: dict[str, tuple[AppSpec, ...]] = {
    "todo": (
        AppSpec("tasks_org", "TasksOrg", aw_original=True),
        AppSpec("cfait", "Cfait"),
        AppSpec("todo_list_pfa", "TodoListPfa"),
        AppSpec("ntodotxt", "Ntodotxt"),
        AppSpec("taskmate", "Taskmate"),
        AppSpec("grit", "Grit"),
    ),
    "notes": (
        AppSpec("joplin", "Joplin"),
        AppSpec("markor", "Markor"),
        AppSpec("notallyx", "NotallyX"),
        AppSpec("neutrinote", "NeutriNote"),
        AppSpec("notesnook", "Notesnook"),
        AppSpec("orgzly_revived", "OrgzlyRevived"),
    ),
    "finance": (
        AppSpec("finance_pro_expense", "ProExpense"),
        AppSpec("finance_oinkoin", "Oinkoin"),
        AppSpec("finance_openmoneybox", "OpenMoneyBox"),
        AppSpec("finance_my_expenses", "MyExpenses"),
        AppSpec("finance_finance_manager", "FinanceManager"),
        AppSpec("finance_sushi", "Sushi"),
    ),
    "music": (
        AppSpec("music_retro_music", "RetroMusic"),
        AppSpec("music_fossify_music", "FossifyMusic"),
        AppSpec("music_apollo", "Apollo"),
        AppSpec("music_sicmu_neo", "SicMuNeo"),
        AppSpec("music_phonograph_plus", "PhonographPlus"),
        AppSpec("music_monstermusic", "MonsterMusic"),
    ),
    "calendar": (
        AppSpec("calendar_simple_calendar_pro", "SimpleCalendarPro", aw_original=True),
        AppSpec("calendar_etar", "Etar"),
        AppSpec("calendar_fossify_calendar", "FossifyCalendar"),
        AppSpec("calendar_calendar", "Calendar"),
        AppSpec("calendar_kashcal", "Kashcal"),
    ),
}


def _todo_expected(app: AppSpec) -> tuple[str, ...]:
  if app.aw_original:
    return (
        *TODO_AW_PREFIXES,
        *(f"{prefix}For{app.suffix}" for prefix in TODO_NEW_PREFIXES),
    )
  return tuple(
      f"{prefix}For{app.suffix}" for prefix in (*TODO_AW_PREFIXES, *TODO_NEW_PREFIXES)
  )


def _notes_expected(app: AppSpec) -> tuple[str, ...]:
  return tuple(f"{prefix}For{app.suffix}" for prefix in NOTES_PREFIXES)


def _finance_expected(app: AppSpec) -> tuple[str, ...]:
  return tuple(f"{prefix}For{app.suffix}" for prefix in FINANCE_PREFIXES)


def _music_expected(app: AppSpec) -> tuple[str, ...]:
  return tuple(f"{prefix}For{app.suffix}" for prefix in MUSIC_PREFIXES)


def _calendar_expected(app: AppSpec) -> tuple[str, ...]:
  if app.aw_original:
    return tuple(
        prefix if prefix in CALENDAR_AW_ORIGINAL else f"{prefix}For{app.suffix}"
        for prefix in CALENDAR_PREFIXES
    )
  return tuple(f"{prefix}For{app.suffix}" for prefix in CALENDAR_PREFIXES)


EXPECTED_BUILDERS = {
    "todo": _todo_expected,
    "notes": _notes_expected,
    "finance": _finance_expected,
    "music": _music_expected,
    "calendar": _calendar_expected,
}


def main() -> int:
  profiles = get_domain_profiles()
  task_registry = registry.TaskRegistry().get_registry(
      registry.TaskRegistry.ANDROID_WORLD_FAMILY
  )
  runnable = set(task_registry)
  failures: list[str] = []

  for category, apps in FIRST_FIVE_APPS.items():
    profile = profiles[category]
    profile_apps = {app.app_id: app for app in profile.apps}
    for app in apps:
      expected = set(EXPECTED_BUILDERS[category](app))
      scheduled = set(profile_apps[app.app_id].implemented_tasks)
      missing_from_profile = sorted(expected - scheduled)
      extra_in_profile = sorted(scheduled - expected)
      missing_from_registry = sorted(expected - runnable)
      wrong_registry_module: list[str] = []
      if category == "todo" and not app.aw_original:
        for task_name in sorted(expected & runnable):
          task_class = task_registry[task_name]
          if task_class.__module__ != (
              "android_world.task_evals.single.app_generalization_generated."
              "todo_cross_app_tasks"
          ):
            wrong_registry_module.append(
                f"{task_name} -> {task_class.__module__}"
            )
      if (
          missing_from_profile
          or extra_in_profile
          or missing_from_registry
          or wrong_registry_module
      ):
        failures.append(f"[{category}] {app.app_id}")
        if missing_from_profile:
          failures.append("  missing from profile: " + ", ".join(missing_from_profile))
        if extra_in_profile:
          failures.append("  extra in profile: " + ", ".join(extra_in_profile))
        if missing_from_registry:
          failures.append("  not runnable in registry: " + ", ".join(missing_from_registry))
        if wrong_registry_module:
          failures.append(
              "  wrong registry module: " + "; ".join(wrong_registry_module)
          )

  if failures:
    print("Table 1 first-five schedule check: FAIL")
    print("\n".join(failures))
    return 1

  print("Table 1 first-five schedule check: PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
