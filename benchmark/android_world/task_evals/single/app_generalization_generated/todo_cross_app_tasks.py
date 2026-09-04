"""Cross-app to-do task ports for the app-generalization suite.

These ports complement the canonical Tasks/To-Do information-retrieval suite
with five UI-write templates: ``TasksCreateTask``, ``TasksEditTask``,
``TasksCompleteTask``, ``TasksDeleteTask`` and ``TasksAddTaskWithPriority``.

Most ports use a lightweight UI-text heuristic via
``base.element_text_contains`` so each app stays short. Grit overrides this
with SQLite-backed setup and validation because its task state is stored in a
normal app database.
"""

from __future__ import annotations

import dataclasses
import os
import random
import tempfile
import time
from typing import Any, Final

from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals.information_retrieval import task_app_utils
from android_world.task_evals.information_retrieval.proto import state_pb2
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base as base,
)


# -----------------------------------------------------------------------------
# Parameter pools.
# -----------------------------------------------------------------------------

_TASK_TITLES: Final[tuple[str, ...]] = (
    "Buy groceries",
    "Call dentist",
    "Pay rent",
    "Renew passport",
    "Email Sarah",
    "Submit report",
    "Pick up package",
    "Schedule oil change",
    "Book flight",
    "Replace bulb",
)

_NEW_TASK_TITLES: Final[tuple[str, ...]] = (
    "Updated grocery list",
    "Call vet instead",
    "Pay utilities",
    "Confirm passport date",
    "Reply to Sarah",
    "Resubmit report",
    "Reroute package pickup",
    "Reschedule oil change",
    "Rebook flight",
    "Replace fixture",
)

_PRIORITIES: Final[tuple[str, ...]] = ("low", "medium", "high")
_DUE_DATES: Final[tuple[str, ...]] = (
    "2023-10-24",
    "2023-10-25",
    "2023-10-26",
    "2023-10-27",
)
_DUE_TIMES: Final[tuple[str, ...]] = ("09:00", "11:30", "14:00", "17:45")
_RECURRENCES: Final[tuple[str, ...]] = ("daily", "weekly", "monthly")


# -----------------------------------------------------------------------------
# Param generators.
# -----------------------------------------------------------------------------


def _generate_create_task_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 2**31 - 1),
      "task_title": random.choice(_TASK_TITLES),
  }


def _generate_edit_task_params() -> dict[str, Any]:
  old_title = random.choice(_TASK_TITLES)
  new_title = random.choice(_NEW_TASK_TITLES)
  while new_title == old_title:
    new_title = random.choice(_NEW_TASK_TITLES)
  return {
      "seed": random.randint(0, 2**31 - 1),
      "old_title": old_title,
      "new_title": new_title,
  }


def _generate_complete_task_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 2**31 - 1),
      "task_title": random.choice(_TASK_TITLES),
  }


def _generate_delete_task_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 2**31 - 1),
      "task_title": random.choice(_TASK_TITLES),
  }


def _generate_add_task_with_priority_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 2**31 - 1),
      "task_title": random.choice(_TASK_TITLES),
      "priority": random.choice(_PRIORITIES),
  }


def _generate_scheduled_task_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 2**31 - 1),
      "task_title": random.choice(_TASK_TITLES),
      "due_date": random.choice(_DUE_DATES),
      "due_time": random.choice(_DUE_TIMES),
      "recurrence": random.choice(_RECURRENCES),
  }


# -----------------------------------------------------------------------------
# Base evaluators.
# -----------------------------------------------------------------------------


class _TasksCreateTaskBase(base.PackageAppEval):
  """Base port of ``TasksCreateTask``.

  Success heuristic: the new task title appears in the UI element list
  after the agent saves the task.
  """

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "task_title": {"type": "string"},
      },
      "required": ["task_title"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if base.element_text_contains(
            env.get_state().ui_elements,
            (self._params["task_title"],),
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_create_task_params()


class _TasksEditTaskBase(base.PackageAppEval):
  """Base port of ``TasksEditTask``.

  Success heuristic: ``new_title`` appears on screen and ``old_title`` does
  not appear, indicating the rename has propagated to the list view.
  """

  complexity = 2.0
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "old_title": {"type": "string"},
          "new_title": {"type": "string"},
      },
      "required": ["old_title", "new_title"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    new_present = base.element_text_contains(
        ui_elements, (self._params["new_title"],)
    )
    old_present = base.element_text_contains(
        ui_elements, (self._params["old_title"],)
    )
    return 1.0 if new_present and not old_present else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_edit_task_params()


class _TasksCompleteTaskBase(base.PackageAppEval):
  """Base port of ``TasksCompleteTask``.

  Success heuristic: the task title still appears on screen and a
  completion marker (``completed``/``done``/``finished``) is also visible
  somewhere in the current UI.
  """

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "task_title": {"type": "string"},
      },
      "required": ["task_title"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    title_ok = base.element_text_contains(
        ui_elements, (self._params["task_title"],)
    )
    completion_markers = (
        "completed",
        "complete",
        "done",
        "finished",
        "checked",
    )
    completion_ok = base.element_text_contains(ui_elements, completion_markers)
    return 1.0 if title_ok and completion_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_complete_task_params()


class _TasksDeleteTaskBase(base.PackageAppEval):
  """Base port of ``TasksDeleteTask``.

  Success heuristic: the task title is no longer present, and the screen
  shows some list/empty affordance such as ``no tasks``, ``empty``,
  ``add task``, or a ``+`` add button.
  """

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "task_title": {"type": "string"},
      },
      "required": ["task_title"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    title_present = base.element_text_contains(
        ui_elements, (self._params["task_title"],)
    )
    list_markers = (
        "no tasks",
        "empty",
        "nothing here",
        "add task",
        "new task",
        "tasks",
        "todo",
        "to-do",
        "to do",
        "+",
    )
    list_ok = base.element_text_contains(ui_elements, list_markers)
    return 1.0 if (not title_present) and list_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_delete_task_params()


class _TasksAddTaskWithPriorityBase(base.PackageAppEval):
  """Base port of ``TasksAddTaskWithPriority``.

  Success heuristic: the task title appears on screen and a priority marker
  (the priority word or a common high/important indicator) is also visible.
  """

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "task_title": {"type": "string"},
          "priority": {"type": "string", "enum": list(_PRIORITIES)},
      },
      "required": ["task_title", "priority"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    title_ok = base.element_text_contains(
        ui_elements, (self._params["task_title"],)
    )
    priority = self._params["priority"].lower()
    priority_markers = (
        priority,
        "priority",
        "important",
        "!",
    )
    if priority == "high":
      priority_markers = priority_markers + ("p1", "urgent")
    elif priority == "medium":
      priority_markers = priority_markers + ("p2", "normal")
    elif priority == "low":
      priority_markers = priority_markers + ("p3",)
    priority_ok = base.element_text_contains(ui_elements, priority_markers)
    return 1.0 if title_ok and priority_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_add_task_with_priority_params()


class _TasksScheduledTaskBase(base.PackageAppEval):
  """Shared schedule/search/filter task used to complete full CATBench rows."""

  complexity = 2.0
  success_mode = "due_date"
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "task_title": {"type": "string"},
          "due_date": {"type": "string"},
          "due_time": {"type": "string"},
          "recurrence": {"type": "string"},
      },
      "required": ["task_title", "due_date"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    title_ok = base.element_text_contains(
        ui_elements, (self._params["task_title"],)
    )
    mode = self.success_mode
    if mode == "due_time":
      markers = (self._params["due_date"], self._params["due_time"])
      marker_ok = base.element_text_contains(ui_elements, markers)
      return 1.0 if title_ok and marker_ok else 0.0
    if mode == "recurring":
      markers = (self._params["recurrence"], "repeat", "recurr")
      marker_ok = base.element_text_contains(ui_elements, markers)
      return 1.0 if title_ok and marker_ok else 0.0
    if mode == "search":
      return 1.0 if title_ok else 0.0
    if mode == "overdue":
      markers = ("overdue", "late", "past due")
      marker_ok = base.element_text_contains(ui_elements, markers)
      return 1.0 if title_ok and marker_ok else 0.0
    markers = (self._params["due_date"], "due")
    marker_ok = base.element_text_contains(ui_elements, markers)
    return 1.0 if title_ok and marker_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_scheduled_task_params()


class _TasksTable1AwTaskBase(_TasksScheduledTaskBase):
  """AW-inherited to-do intent port used by the Table 1 schedule."""

  success_mode = "due_date"

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    title_ok = base.element_text_contains(
        ui_elements, (self._params["task_title"],)
    )
    due_ok = base.element_text_contains(
        ui_elements, (self._params["due_date"], "due")
    )
    priority_ok = base.element_text_contains(
        ui_elements, ("high", "priority", "important", "urgent", "!")
    )
    complete_ok = base.element_text_contains(
        ui_elements, ("completed", "complete", "done", "finished", "checked")
    )
    mode = self.success_mode
    if mode == "completed_for_date":
      return 1.0 if title_ok and (due_ok or complete_ok) else 0.0
    if mode == "due_next_week":
      next_week_ok = base.element_text_contains(
          ui_elements, ("next week", self._params["due_date"])
      )
      return 1.0 if title_ok and next_week_ok else 0.0
    if mode == "high_priority":
      return 1.0 if title_ok and priority_ok else 0.0
    if mode == "high_priority_due_date":
      return 1.0 if title_ok and priority_ok and due_ok else 0.0
    if mode == "incomplete_for_date":
      return 1.0 if title_ok and due_ok and not complete_ok else 0.0
    return 1.0 if title_ok else 0.0


# -----------------------------------------------------------------------------
# Per-app packages.
# -----------------------------------------------------------------------------

_TASKS_ORG_PACKAGE: Final[str] = "org.tasks"
_CFAIT_PACKAGE: Final[str] = "com.trougnouf.cfait"
_TODO_LIST_PFA_PACKAGE: Final[str] = "org.secuso.privacyfriendlytodolist"
_NTODOTXT_PACKAGE: Final[str] = "de.tnmgl.ntodotxt"
_TASKMATE_PACKAGE: Final[str] = "com.amirsteinbeck.taskmate"
_GRIT_PACKAGE: Final[str] = "com.shub39.grit"
_PFA_DB_PATH: Final[str] = (
    "/data/data/org.secuso.privacyfriendlytodolist/databases/TodoDatabase.db"
)
_PFA_LIST_TABLE: Final[str] = "todoLists"
_PFA_TASK_TABLE: Final[str] = "todoTasks"
_PFA_SUBTASK_TABLE: Final[str] = "todoSubtasks"
_PFA_NOISE_TITLE: Final[str] = "Review pantry list"
_NTODOTXT_FILE_PATH: Final[str] = "/data/data/de.tnmgl.ntodotxt/app_flutter/todo.txt"
_NTODOTXT_NOISE_TITLE: Final[str] = "Review inbox"
_GRIT_DB_PATH: Final[str] = (
    "/data/data/com.shub39.grit/databases/task_database"
)
_GRIT_TASK_TABLE: Final[str] = "task"
_GRIT_CATEGORY_TABLE: Final[str] = "categories"
_GRIT_NOISE_TITLE: Final[str] = "Review garden notes"

_TASKS_ORG_PRIORITY_TO_IMPORTANCE: Final[dict[str, int]] = {
    "high": 0,
    "medium": 1,
    "low": 2,
}


def _now_millis() -> int:
  return 1_697_414_400_000


def _force_stop_package(package_name: str, env: interface.AsyncEnv) -> None:
  adb_utils.issue_generic_request(
      ["shell", "am", "force-stop", package_name],
      env.controller,
      timeout_sec=5,
  )


def _wait_until(predicate, timeout_sec: float = 3.0, interval_sec: float = 0.25) -> bool:
  deadline = time.time() + timeout_sec
  while True:
    if predicate():
      return True
    if time.time() >= deadline:
      return False
    time.sleep(interval_sec)


def _tasks_org_seed_task(title: str, importance: int = 2) -> state_pb2.TasksAppTask:
  task = state_pb2.TasksAppTask()
  task.title = title
  task.importance = str(importance)
  return task


def _tasks_org_seed_tasks(
    env: interface.AsyncEnv,
    tasks: tuple[state_pb2.TasksAppTask, ...],
) -> None:
  task_app_utils.clear_task_db(env)
  task_app_utils.add_tasks(
      [task_app_utils.create_task_from_proto(task) for task in tasks], env
  )
  adb_utils.launch_app(_TASKS_ORG_PACKAGE, env.controller)


def _tasks_org_active_rows(
    env: interface.AsyncEnv,
    title: str,
) -> list[sqlite_schema_utils.Task]:
  return [
      row
      for row in task_app_utils.list_rows(env)
      if row.title == title and not row.deleted
  ]


@dataclasses.dataclass(frozen=True)
class _PfaListRow(sqlite_schema_utils.SQLiteRow):
  sortOrder: int
  name: str
  id: int = -1


@dataclasses.dataclass(frozen=True)
class _PfaTaskRow(sqlite_schema_utils.SQLiteRow):
  sortOrder: int
  name: str
  description: str
  priority: int
  recurrencePattern: int
  recurrenceInterval: int
  reminderState: int
  progress: int
  creationTime: int
  isInRecycleBin: int
  listId: int | None = None
  deadline: int | None = None
  reminderTime: int | None = None
  doneTime: int | None = None
  id: int = -1


def _list_pfa_lists(env: interface.AsyncEnv) -> list[_PfaListRow]:
  return sqlite_utils.get_rows_from_remote_device(
      _PFA_LIST_TABLE,
      _PFA_DB_PATH,
      _PfaListRow,
      env,
  )


def _list_pfa_tasks(env: interface.AsyncEnv) -> list[_PfaTaskRow]:
  return sqlite_utils.get_rows_from_remote_device(
      _PFA_TASK_TABLE,
      _PFA_DB_PATH,
      _PfaTaskRow,
      env,
  )


def _clear_pfa_db(env: interface.AsyncEnv) -> None:
  _force_stop_package(_TODO_LIST_PFA_PACKAGE, env)
  sqlite_utils.delete_all_rows_from_table(
      _PFA_SUBTASK_TABLE,
      _PFA_DB_PATH,
      env,
      _TODO_LIST_PFA_PACKAGE,
  )
  sqlite_utils.delete_all_rows_from_table(
      _PFA_TASK_TABLE,
      _PFA_DB_PATH,
      env,
      _TODO_LIST_PFA_PACKAGE,
  )
  sqlite_utils.delete_all_rows_from_table(
      _PFA_LIST_TABLE,
      _PFA_DB_PATH,
      env,
      _TODO_LIST_PFA_PACKAGE,
  )


def _insert_pfa_lists(rows: list[_PfaListRow], env: interface.AsyncEnv) -> None:
  sqlite_utils.insert_rows_to_remote_db(
      rows,
      "id",
      _PFA_LIST_TABLE,
      _PFA_DB_PATH,
      _TODO_LIST_PFA_PACKAGE,
      env,
  )


def _insert_pfa_tasks(rows: list[_PfaTaskRow], env: interface.AsyncEnv) -> None:
  sqlite_utils.insert_rows_to_remote_db(
      rows,
      "id",
      _PFA_TASK_TABLE,
      _PFA_DB_PATH,
      _TODO_LIST_PFA_PACKAGE,
      env,
  )


def _ensure_pfa_list(env: interface.AsyncEnv) -> int:
  lists = _list_pfa_lists(env)
  if not lists:
    _insert_pfa_lists([_PfaListRow(sortOrder=0, name="Tasks")], env)
    lists = _list_pfa_lists(env)
  if not lists:
    raise RuntimeError("Todo List PFA list table is empty after initialization.")
  return lists[0].id


def _prepare_pfa_db(env: interface.AsyncEnv) -> int:
  _clear_pfa_db(env)
  return _ensure_pfa_list(env)


def _pfa_seed_row(title: str, list_id: int, sort_order: int = 0) -> _PfaTaskRow:
  return _PfaTaskRow(
      listId=list_id,
      sortOrder=sort_order,
      name=title,
      description="",
      priority=0,
      recurrencePattern=0,
      recurrenceInterval=0,
      reminderState=0,
      progress=0,
      creationTime=_now_millis(),
      isInRecycleBin=0,
  )


def _seed_pfa_tasks(env: interface.AsyncEnv, rows: list[_PfaTaskRow]) -> None:
  if rows:
    _insert_pfa_tasks(rows, env)
  adb_utils.launch_app(_TODO_LIST_PFA_PACKAGE, env.controller)


def _matching_pfa_rows(
    env: interface.AsyncEnv,
    title: str,
) -> list[_PfaTaskRow]:
  return [
      row
      for row in _list_pfa_tasks(env)
      if row.name == title and not row.isInRecycleBin
  ]


def _read_ntodotxt_lines(env: interface.AsyncEnv) -> list[str]:
  response = adb_utils.issue_generic_request(
      ["shell", "cat", _NTODOTXT_FILE_PATH],
      env.controller,
  )
  output = response.generic.output.decode("utf-8")
  return [line.strip() for line in output.splitlines() if line.strip()]


def _write_ntodotxt_lines(env: interface.AsyncEnv, lines: list[str]) -> None:
  _force_stop_package(_NTODOTXT_PACKAGE, env)
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", suffix=".txt", delete=False
  ) as handle:
    handle.write("\n".join(lines))
    if lines:
      handle.write("\n")
    local_path = handle.name
  try:
    env.controller.push_file(local_path, _NTODOTXT_FILE_PATH)
  finally:
    os.remove(local_path)
  adb_utils.launch_app(_NTODOTXT_PACKAGE, env.controller)


def _ntodotxt_line_for_title(title: str) -> str:
  return title


def _ntodotxt_priority_line(title: str, priority: str) -> str:
  priority_letter = {"high": "A", "medium": "B", "low": "C"}[priority]
  return f"({priority_letter}) {title}"


def _is_iso_date_token(token: str) -> bool:
  return (
      len(token) == 10
      and token[4] == "-"
      and token[7] == "-"
      and token[:4].isdigit()
      and token[5:7].isdigit()
      and token[8:].isdigit()
  )


def _ntodotxt_title_from_line(line: str) -> str:
  line = line.strip()
  if line.startswith("x "):
    line = line[2:].strip()
    parts = line.split(maxsplit=1)
    if parts and _is_iso_date_token(parts[0]):
      line = parts[1].strip() if len(parts) == 2 else ""
  if len(line) >= 4 and line[0] == "(" and line[2] == ")":
    line = line[4:].strip()
  return line


def _ntodotxt_active_titles(env: interface.AsyncEnv) -> set[str]:
  titles = set()
  for line in _read_ntodotxt_lines(env):
    if line.startswith("x "):
      continue
    title = _ntodotxt_title_from_line(line)
    if title:
      titles.add(title)
  return titles


def _ntodotxt_completed_titles(env: interface.AsyncEnv) -> set[str]:
  titles = set()
  for line in _read_ntodotxt_lines(env):
    if not line.startswith("x "):
      continue
    title = _ntodotxt_title_from_line(line)
    if title:
      titles.add(title)
  return titles


def _ntodotxt_priority_ok(
    env: interface.AsyncEnv,
    title: str,
    priority: str,
) -> bool:
  priority_letter = {"high": "A", "medium": "B", "low": "C"}[priority]
  prefix = f"({priority_letter}) "
  return any(
      line.strip().startswith(prefix) and _ntodotxt_title_from_line(line) == title
      for line in _read_ntodotxt_lines(env)
  )


@dataclasses.dataclass(frozen=True)
class _GritCategoryRow(sqlite_schema_utils.SQLiteRow):
  name: str = "Misc"
  index: int = 0
  color: str = "gray"
  id: int = -1


@dataclasses.dataclass(frozen=True)
class _GritTaskRow(sqlite_schema_utils.SQLiteRow):
  title: str
  categoryId: int = 1
  status: int = 0
  index: int = 0
  reminder: int | None = None
  id: int = -1


def _list_grit_categories(env: interface.AsyncEnv) -> list[_GritCategoryRow]:
  return sqlite_utils.get_rows_from_remote_device(
      _GRIT_CATEGORY_TABLE,
      _GRIT_DB_PATH,
      _GritCategoryRow,
      env,
  )


def _list_grit_tasks(env: interface.AsyncEnv) -> list[_GritTaskRow]:
  return sqlite_utils.get_rows_from_remote_device(
      _GRIT_TASK_TABLE,
      _GRIT_DB_PATH,
      _GritTaskRow,
      env,
  )


def _clear_grit_tasks(env: interface.AsyncEnv) -> None:
  sqlite_utils.delete_all_rows_from_table(
      _GRIT_TASK_TABLE,
      _GRIT_DB_PATH,
      env,
      _GRIT_PACKAGE,
  )


def _insert_grit_categories(
    rows: list[_GritCategoryRow],
    env: interface.AsyncEnv,
) -> None:
  sqlite_utils.insert_rows_to_remote_db(
      rows,
      "id",
      _GRIT_CATEGORY_TABLE,
      _GRIT_DB_PATH,
      _GRIT_PACKAGE,
      env,
  )


def _insert_grit_tasks(rows: list[_GritTaskRow], env: interface.AsyncEnv) -> None:
  sqlite_utils.insert_rows_to_remote_db(
      rows,
      "id",
      _GRIT_TASK_TABLE,
      _GRIT_DB_PATH,
      _GRIT_PACKAGE,
      env,
  )


def _ensure_grit_category(env: interface.AsyncEnv) -> int:
  categories = _list_grit_categories(env)
  if not categories:
    _insert_grit_categories([_GritCategoryRow()], env)
    categories = _list_grit_categories(env)
  if not categories:
    raise RuntimeError("Grit category table is empty after initialization.")
  return categories[0].id


def _prepare_grit_db(env: interface.AsyncEnv) -> int:
  _clear_grit_tasks(env)
  return _ensure_grit_category(env)


def _seed_grit_tasks(
    env: interface.AsyncEnv,
    rows: list[_GritTaskRow],
) -> None:
  if rows:
    _insert_grit_tasks(rows, env)
  adb_utils.launch_app(_GRIT_PACKAGE, env.controller)


def _matching_grit_rows(
    env: interface.AsyncEnv,
    title: str,
) -> list[_GritTaskRow]:
  return [row for row in _list_grit_tasks(env) if row.title.strip() == title]


# -----------------------------------------------------------------------------
# Tasks.org
# -----------------------------------------------------------------------------


class TasksCreateTaskForTasksOrg(_TasksCreateTaskBase):
  app_names = (_TASKS_ORG_PACKAGE,)
  package_name = _TASKS_ORG_PACKAGE
  template = (
      "In the Tasks.org app, create a new task titled '{task_title}'."
  )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if _tasks_org_active_rows(env, self._params["task_title"])
        else 0.0
    )


class TasksEditTaskForTasksOrg(_TasksEditTaskBase):
  app_names = (_TASKS_ORG_PACKAGE,)
  package_name = _TASKS_ORG_PACKAGE
  template = (
      "In the Tasks.org app, edit the task titled '{old_title}' to be"
      " '{new_title}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _tasks_org_seed_tasks(
        env, (_tasks_org_seed_task(self._params["old_title"]),)
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    old_rows = _tasks_org_active_rows(env, self._params["old_title"])
    new_rows = _tasks_org_active_rows(env, self._params["new_title"])
    return 1.0 if new_rows and not old_rows else 0.0


class TasksCompleteTaskForTasksOrg(_TasksCompleteTaskBase):
  app_names = (_TASKS_ORG_PACKAGE,)
  package_name = _TASKS_ORG_PACKAGE
  template = (
      "In the Tasks.org app, mark the task '{task_title}' as completed."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _tasks_org_seed_tasks(
        env, (_tasks_org_seed_task(self._params["task_title"]),)
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    rows = _tasks_org_active_rows(env, self._params["task_title"])
    return 1.0 if any(row.completed for row in rows) else 0.0


class TasksDeleteTaskForTasksOrg(_TasksDeleteTaskBase):
  app_names = (_TASKS_ORG_PACKAGE,)
  package_name = _TASKS_ORG_PACKAGE
  template = (
      "In the Tasks.org app, delete the task titled '{task_title}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _tasks_org_seed_tasks(
        env, (_tasks_org_seed_task(self._params["task_title"]),)
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        0.0
        if _tasks_org_active_rows(env, self._params["task_title"])
        else 1.0
    )


class TasksAddTaskWithPriorityForTasksOrg(_TasksAddTaskWithPriorityBase):
  app_names = (_TASKS_ORG_PACKAGE,)
  package_name = _TASKS_ORG_PACKAGE
  template = (
      "In the Tasks.org app, create a task titled '{task_title}' with"
      " priority {priority}."
  )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    expected_importance = _TASKS_ORG_PRIORITY_TO_IMPORTANCE[
        self._params["priority"]
    ]
    rows = _tasks_org_active_rows(env, self._params["task_title"])
    return (
        1.0
        if any(row.importance == expected_importance for row in rows)
        else 0.0
    )


# -----------------------------------------------------------------------------
# Cfait
# -----------------------------------------------------------------------------


class TasksCreateTaskForCfait(_TasksCreateTaskBase):
  app_names = (_CFAIT_PACKAGE,)
  package_name = _CFAIT_PACKAGE
  template = (
      "In the Cfait app, create a new task titled '{task_title}'."
  )


class TasksEditTaskForCfait(_TasksEditTaskBase):
  app_names = (_CFAIT_PACKAGE,)
  package_name = _CFAIT_PACKAGE
  template = (
      "In the Cfait app, edit the task titled '{old_title}' to be"
      " '{new_title}'."
  )


class TasksCompleteTaskForCfait(_TasksCompleteTaskBase):
  app_names = (_CFAIT_PACKAGE,)
  package_name = _CFAIT_PACKAGE
  template = (
      "In the Cfait app, mark the task '{task_title}' as completed."
  )


class TasksDeleteTaskForCfait(_TasksDeleteTaskBase):
  app_names = (_CFAIT_PACKAGE,)
  package_name = _CFAIT_PACKAGE
  template = (
      "In the Cfait app, delete the task titled '{task_title}'."
  )


class TasksAddTaskWithPriorityForCfait(_TasksAddTaskWithPriorityBase):
  app_names = (_CFAIT_PACKAGE,)
  package_name = _CFAIT_PACKAGE
  template = (
      "In the Cfait app, create a task titled '{task_title}' with priority"
      " {priority}."
  )


# -----------------------------------------------------------------------------
# Todo List (PFA)
# -----------------------------------------------------------------------------


class TasksCreateTaskForTodoListPfa(_TasksCreateTaskBase):
  app_names = (_TODO_LIST_PFA_PACKAGE,)
  package_name = _TODO_LIST_PFA_PACKAGE
  template = (
      "In the Todo List (PFA) app, create a new task titled '{task_title}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _prepare_pfa_db(env)
    adb_utils.launch_app(_TODO_LIST_PFA_PACKAGE, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if _wait_until(
            lambda: bool(_matching_pfa_rows(env, self._params["task_title"]))
        )
        else 0.0
    )


class TasksEditTaskForTodoListPfa(_TasksEditTaskBase):
  app_names = (_TODO_LIST_PFA_PACKAGE,)
  package_name = _TODO_LIST_PFA_PACKAGE
  template = (
      "In the Todo List (PFA) app, edit the task titled '{old_title}' to be"
      " '{new_title}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    list_id = _prepare_pfa_db(env)
    _seed_pfa_tasks(
        env,
        [
            _pfa_seed_row(self._params["old_title"], list_id, 0),
            _pfa_seed_row(_PFA_NOISE_TITLE, list_id, 1),
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    def _success() -> bool:
      old_rows = _matching_pfa_rows(env, self._params["old_title"])
      new_rows = _matching_pfa_rows(env, self._params["new_title"])
      noise_rows = _matching_pfa_rows(env, _PFA_NOISE_TITLE)
      return bool(new_rows and not old_rows and noise_rows)

    return 1.0 if _wait_until(_success) else 0.0


class TasksCompleteTaskForTodoListPfa(_TasksCompleteTaskBase):
  app_names = (_TODO_LIST_PFA_PACKAGE,)
  package_name = _TODO_LIST_PFA_PACKAGE
  template = (
      "In the Todo List (PFA) app, mark the task '{task_title}' as"
      " completed."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    list_id = _prepare_pfa_db(env)
    _seed_pfa_tasks(
        env,
        [
            _pfa_seed_row(self._params["task_title"], list_id, 0),
            _pfa_seed_row(_PFA_NOISE_TITLE, list_id, 1),
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    def _success() -> bool:
      rows = _matching_pfa_rows(env, self._params["task_title"])
      noise_rows = _matching_pfa_rows(env, _PFA_NOISE_TITLE)
      completed = any(row.progress >= 100 or row.doneTime for row in rows)
      noise_intact = any(row.progress < 100 and not row.doneTime for row in noise_rows)
      return completed and noise_intact

    return 1.0 if _wait_until(_success) else 0.0


class TasksDeleteTaskForTodoListPfa(_TasksDeleteTaskBase):
  app_names = (_TODO_LIST_PFA_PACKAGE,)
  package_name = _TODO_LIST_PFA_PACKAGE
  template = (
      "In the Todo List (PFA) app, delete the task titled '{task_title}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    list_id = _prepare_pfa_db(env)
    _seed_pfa_tasks(
        env,
        [
            _pfa_seed_row(self._params["task_title"], list_id, 0),
            _pfa_seed_row(_PFA_NOISE_TITLE, list_id, 1),
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    def _success() -> bool:
      target_rows = _matching_pfa_rows(env, self._params["task_title"])
      noise_rows = _matching_pfa_rows(env, _PFA_NOISE_TITLE)
      return bool(not target_rows and noise_rows)

    return 1.0 if _wait_until(_success) else 0.0


class TasksAddTaskWithPriorityForTodoListPfa(_TasksAddTaskWithPriorityBase):
  app_names = (_TODO_LIST_PFA_PACKAGE,)
  package_name = _TODO_LIST_PFA_PACKAGE
  template = (
      "In the Todo List (PFA) app, create a task titled '{task_title}' with"
      " priority {priority}."
  )


# -----------------------------------------------------------------------------
# ntodotxt
# -----------------------------------------------------------------------------


class TasksCreateTaskForNtodotxt(_TasksCreateTaskBase):
  app_names = (_NTODOTXT_PACKAGE,)
  package_name = _NTODOTXT_PACKAGE
  template = (
      "In the ntodotxt app, create a new task titled '{task_title}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _write_ntodotxt_lines(env, [])

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if _wait_until(
            lambda: self._params["task_title"] in _ntodotxt_active_titles(env)
        )
        else 0.0
    )


class TasksEditTaskForNtodotxt(_TasksEditTaskBase):
  app_names = (_NTODOTXT_PACKAGE,)
  package_name = _NTODOTXT_PACKAGE
  template = (
      "In the ntodotxt app, edit the task titled '{old_title}' to be"
      " '{new_title}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _write_ntodotxt_lines(
        env,
        [
            _ntodotxt_line_for_title(self._params["old_title"]),
            _ntodotxt_line_for_title(_NTODOTXT_NOISE_TITLE),
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    def _success() -> bool:
      active_titles = _ntodotxt_active_titles(env)
      return (
          self._params["new_title"] in active_titles
          and self._params["old_title"] not in active_titles
          and _NTODOTXT_NOISE_TITLE in active_titles
      )

    return 1.0 if _wait_until(_success) else 0.0


class TasksCompleteTaskForNtodotxt(_TasksCompleteTaskBase):
  app_names = (_NTODOTXT_PACKAGE,)
  package_name = _NTODOTXT_PACKAGE
  template = (
      "In the ntodotxt app, mark the task '{task_title}' as completed."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _write_ntodotxt_lines(
        env,
        [
            _ntodotxt_line_for_title(self._params["task_title"]),
            _ntodotxt_line_for_title(_NTODOTXT_NOISE_TITLE),
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    def _success() -> bool:
      completed_titles = _ntodotxt_completed_titles(env)
      active_titles = _ntodotxt_active_titles(env)
      return (
          self._params["task_title"] in completed_titles
          and _NTODOTXT_NOISE_TITLE in active_titles
      )

    return 1.0 if _wait_until(_success) else 0.0


class TasksDeleteTaskForNtodotxt(_TasksDeleteTaskBase):
  app_names = (_NTODOTXT_PACKAGE,)
  package_name = _NTODOTXT_PACKAGE
  template = (
      "In the ntodotxt app, delete the task titled '{task_title}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _write_ntodotxt_lines(
        env,
        [
            _ntodotxt_line_for_title(self._params["task_title"]),
            _ntodotxt_line_for_title(_NTODOTXT_NOISE_TITLE),
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    def _success() -> bool:
      active_titles = _ntodotxt_active_titles(env)
      return (
          self._params["task_title"] not in active_titles
          and _NTODOTXT_NOISE_TITLE in active_titles
      )

    return 1.0 if _wait_until(_success) else 0.0


class TasksAddTaskWithPriorityForNtodotxt(_TasksAddTaskWithPriorityBase):
  app_names = (_NTODOTXT_PACKAGE,)
  package_name = _NTODOTXT_PACKAGE
  template = (
      "In the ntodotxt app, create a task titled '{task_title}' with"
      " priority {priority}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _write_ntodotxt_lines(env, [])

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if _wait_until(
            lambda: _ntodotxt_priority_ok(
                env, self._params["task_title"], self._params["priority"]
            )
        )
        else 0.0
    )


# -----------------------------------------------------------------------------
# TaskMate
# -----------------------------------------------------------------------------


class TasksCreateTaskForTaskmate(_TasksCreateTaskBase):
  app_names = (_TASKMATE_PACKAGE,)
  package_name = _TASKMATE_PACKAGE
  template = (
      "In the TaskMate app, create a new task titled '{task_title}'."
  )


class TasksEditTaskForTaskmate(_TasksEditTaskBase):
  app_names = (_TASKMATE_PACKAGE,)
  package_name = _TASKMATE_PACKAGE
  template = (
      "In the TaskMate app, edit the task titled '{old_title}' to be"
      " '{new_title}'."
  )


class TasksCompleteTaskForTaskmate(_TasksCompleteTaskBase):
  app_names = (_TASKMATE_PACKAGE,)
  package_name = _TASKMATE_PACKAGE
  template = (
      "In the TaskMate app, mark the task '{task_title}' as completed."
  )


class TasksDeleteTaskForTaskmate(_TasksDeleteTaskBase):
  app_names = (_TASKMATE_PACKAGE,)
  package_name = _TASKMATE_PACKAGE
  template = (
      "In the TaskMate app, delete the task titled '{task_title}'."
  )


class TasksAddTaskWithPriorityForTaskmate(_TasksAddTaskWithPriorityBase):
  app_names = (_TASKMATE_PACKAGE,)
  package_name = _TASKMATE_PACKAGE
  template = (
      "In the TaskMate app, create a task titled '{task_title}' with"
      " priority {priority}."
  )


# -----------------------------------------------------------------------------
# Grit
# -----------------------------------------------------------------------------


class TasksCreateTaskForGrit(_TasksCreateTaskBase):
  app_names = (_GRIT_PACKAGE,)
  package_name = _GRIT_PACKAGE
  template = (
      "In the Grit app, create a new task titled '{task_title}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _prepare_grit_db(env)
    adb_utils.launch_app(_GRIT_PACKAGE, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return 1.0 if _matching_grit_rows(env, self._params["task_title"]) else 0.0


class TasksEditTaskForGrit(_TasksEditTaskBase):
  app_names = (_GRIT_PACKAGE,)
  package_name = _GRIT_PACKAGE
  template = (
      "In the Grit app, edit the task titled '{old_title}' to be"
      " '{new_title}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    category_id = _prepare_grit_db(env)
    _seed_grit_tasks(
        env,
        [
            _GritTaskRow(
                title=self._params["old_title"],
                categoryId=category_id,
                index=0,
            ),
            _GritTaskRow(
                title=_GRIT_NOISE_TITLE,
                categoryId=category_id,
                index=1,
            ),
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    old_rows = _matching_grit_rows(env, self._params["old_title"])
    new_rows = _matching_grit_rows(env, self._params["new_title"])
    noise_rows = _matching_grit_rows(env, _GRIT_NOISE_TITLE)
    return 1.0 if new_rows and not old_rows and noise_rows else 0.0


class TasksCompleteTaskForGrit(_TasksCompleteTaskBase):
  app_names = (_GRIT_PACKAGE,)
  package_name = _GRIT_PACKAGE
  template = (
      "In the Grit app, mark the task '{task_title}' as completed."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    category_id = _prepare_grit_db(env)
    _seed_grit_tasks(
        env,
        [
            _GritTaskRow(
                title=self._params["task_title"],
                categoryId=category_id,
                index=0,
            ),
            _GritTaskRow(
                title=_GRIT_NOISE_TITLE,
                categoryId=category_id,
                index=1,
            ),
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    rows = _matching_grit_rows(env, self._params["task_title"])
    noise_rows = _matching_grit_rows(env, _GRIT_NOISE_TITLE)
    completed = any(row.status != 0 for row in rows)
    noise_intact = any(row.status == 0 for row in noise_rows)
    return 1.0 if completed and noise_intact else 0.0


class TasksDeleteTaskForGrit(_TasksDeleteTaskBase):
  app_names = (_GRIT_PACKAGE,)
  package_name = _GRIT_PACKAGE
  template = (
      "In the Grit app, delete the task titled '{task_title}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    category_id = _prepare_grit_db(env)
    _seed_grit_tasks(
        env,
        [
            _GritTaskRow(
                title=self._params["task_title"],
                categoryId=category_id,
                index=0,
            ),
            _GritTaskRow(
                title=_GRIT_NOISE_TITLE,
                categoryId=category_id,
                index=1,
            ),
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    target_rows = _matching_grit_rows(env, self._params["task_title"])
    noise_rows = _matching_grit_rows(env, _GRIT_NOISE_TITLE)
    return 1.0 if not target_rows and noise_rows else 0.0


class TasksAddTaskWithPriorityForGrit(_TasksAddTaskWithPriorityBase):
  app_names = (_GRIT_PACKAGE,)
  package_name = _GRIT_PACKAGE
  template = (
      "In the Grit app, create a task titled '{task_title}' with"
      " priority {priority}."
  )


def _make_todo_scheduled_task(
    class_name: str,
    package_name: str,
    template: str,
    success_mode: str,
) -> type[_TasksScheduledTaskBase]:
  return type(
      class_name,
      (_TasksScheduledTaskBase,),
      {
          "__module__": __name__,
          "app_names": (package_name,),
          "package_name": package_name,
          "template": template,
          "success_mode": success_mode,
      },
  )


_TODO_APP_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    ("TasksOrg", _TASKS_ORG_PACKAGE, "Tasks.org"),
    ("Cfait", _CFAIT_PACKAGE, "Cfait"),
    ("TodoListPfa", _TODO_LIST_PFA_PACKAGE, "Todo List (PFA)"),
    ("Ntodotxt", _NTODOTXT_PACKAGE, "ntodotxt"),
    ("Taskmate", _TASKMATE_PACKAGE, "TaskMate"),
    ("Grit", _GRIT_PACKAGE, "Grit"),
)

_TODO_SCHEDULED_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "DueOnDate",
        "In the {app} app, create a task titled '{task_title}' due on "
        "{due_date}.",
        "due_date",
    ),
    (
        "DueWithTime",
        "In the {app} app, create a task titled '{task_title}' due on "
        "{due_date} at {due_time}.",
        "due_time",
    ),
    (
        "Recurring",
        "In the {app} app, create a {recurrence} recurring task titled "
        "'{task_title}' due on {due_date}.",
        "recurring",
    ),
    (
        "SearchTask",
        "In the {app} app, create a task titled '{task_title}', then search "
        "for that task and open it.",
        "search",
    ),
    (
        "OverdueFilter",
        "In the {app} app, create a task titled '{task_title}' with a due "
        "date before today, then open the overdue or past-due task filter.",
        "overdue",
    ),
)

for _suffix, _package, _display_name in _TODO_APP_SPECS:
  for _task_name, _template, _mode in _TODO_SCHEDULED_SPECS:
    globals()[f"Tasks{_task_name}For{_suffix}"] = _make_todo_scheduled_task(
        f"Tasks{_task_name}For{_suffix}",
        _package,
        _template.replace("{app}", _display_name),
        _mode,
    )


def _make_todo_table1_aw_task(
    class_name: str,
    package_name: str,
    template: str,
    success_mode: str,
) -> type[_TasksTable1AwTaskBase]:
  return type(
      class_name,
      (_TasksTable1AwTaskBase,),
      {
          "__module__": __name__,
          "app_names": (package_name,),
          "package_name": package_name,
          "template": template,
          "success_mode": success_mode,
      },
  )


_TODO_TABLE1_AW_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "CompletedTasksForDate",
        "In the {app} app, create a task titled '{task_title}' due on "
        "{due_date}, mark it complete, then show completed tasks for that date.",
        "completed_for_date",
    ),
    (
        "DueNextWeek",
        "In the {app} app, create a task titled '{task_title}' due next week "
        "on {due_date}, then show tasks due next week.",
        "due_next_week",
    ),
    (
        "HighPriorityTasks",
        "In the {app} app, create a high-priority task titled '{task_title}', "
        "then show high-priority tasks.",
        "high_priority",
    ),
    (
        "HighPriorityTasksDueOnDate",
        "In the {app} app, create a high-priority task titled '{task_title}' "
        "due on {due_date}, then show high-priority tasks for that date.",
        "high_priority_due_date",
    ),
    (
        "IncompleteTasksOnDate",
        "In the {app} app, create an incomplete task titled '{task_title}' due "
        "on {due_date}, then show incomplete tasks for that date.",
        "incomplete_for_date",
    ),
)

for _suffix, _package, _display_name in _TODO_APP_SPECS:
  for _task_name, _template, _mode in _TODO_TABLE1_AW_SPECS:
    globals()[f"Tasks{_task_name}For{_suffix}"] = _make_todo_table1_aw_task(
        f"Tasks{_task_name}For{_suffix}",
        _package,
        _template.replace("{app}", _display_name),
        _mode,
    )
