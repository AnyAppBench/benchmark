"""Cross-app notes task ports for the app-generalization suite.

This module fills the 10th template slot for the Notes category, joining the
nine per-app templates already scaffolded in ``notes_*_tasks.py``. Each port
asks the agent to create an organizational container (folder/notebook/label/
category/tag, depending on the app's vocabulary) within the target notes app.

Active ports use durable file or SQLite validation. Apps whose notebook/label
state is stored in an opaque or encrypted store are de-scoped in
``app_generalization_profiles.py`` until a durable verifier is available.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any, Final

from android_world.env import interface
from android_world.env import device_constants
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base as base,
)
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.utils import file_utils


_FOLDER_NAMES: Final[tuple[str, ...]] = (
    "Personal",
    "Work",
    "Recipes",
    "Travel",
    "Ideas",
    "Reading List",
    "Projects",
    "Inbox",
    "Journal",
    "Drafts",
)
_NOTE_TITLES: Final[tuple[str, ...]] = (
    "Trip Ideas",
    "Meeting Notes",
    "Recipe Draft",
    "Book List",
    "Project Plan",
    "Garden Checklist",
    "Research Summary",
    "Weekly Review",
)
_UPDATED_NOTE_TITLES: Final[tuple[str, ...]] = (
    "Updated Trip Ideas",
    "Revised Meeting Notes",
    "Recipe Final",
    "Reading List",
    "Project Plan v2",
    "Garden Plan",
    "Research Notes",
    "Monthly Review",
)
_NOTE_BODIES: Final[tuple[str, ...]] = (
    "Remember to verify the details before sharing.",
    "Draft created for the CATBench notes task.",
    "Keep this short note available for later review.",
)


def _generate_create_folder_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 2**31 - 1),
      "folder_name": random.choice(_FOLDER_NAMES),
  }


def _generate_note_params() -> dict[str, Any]:
  old_title = random.choice(_NOTE_TITLES)
  new_title = random.choice(_UPDATED_NOTE_TITLES)
  while new_title == old_title:
    new_title = random.choice(_UPDATED_NOTE_TITLES)
  return {
      "seed": random.randint(0, 2**31 - 1),
      "note_title": old_title,
      "old_title": old_title,
      "new_title": new_title,
      "merged_title": f"{old_title} Combined",
      "folder_name": random.choice(_FOLDER_NAMES),
      "note_body": random.choice(_NOTE_BODIES),
      "todo_count": random.choice((3, 4, 5)),
  }


# -----------------------------------------------------------------------------
# Base evaluators.
# -----------------------------------------------------------------------------


class _NotesCreateFolderBase(base.PackageAppEval):
  """Base port: create a named organizational container in a notes app.

  Success heuristic: the chosen container name appears in the UI element list
  after the agent finishes (e.g., shown in the sidebar/folder list).
  """

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "folder_name": {"type": "string"},
      },
      "required": ["seed", "folder_name"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if base.element_text_contains(
            env.get_state().ui_elements,
            (self._params["folder_name"],),
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_create_folder_params()


class _NotesGenericTaskBase(base.PackageAppEval):
  """Shared self-contained notes task for full CATBench coverage.

  These ports are intentionally conservative: the task instruction includes
  the setup action when a pre-existing note would otherwise be required.
  Success is checked from visible UI text or, for count tasks, the agent's
  submitted answer.
  """

  complexity = 1.8
  success_mode = "note_title"
  schema = {
      "type": "object",
      "properties": {
          "seed": {"type": "integer"},
          "note_title": {"type": "string"},
          "old_title": {"type": "string"},
          "new_title": {"type": "string"},
          "merged_title": {"type": "string"},
          "folder_name": {"type": "string"},
          "note_body": {"type": "string"},
          "todo_count": {"type": "integer"},
      },
      "required": ["note_title"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    mode = self.success_mode
    if mode == "edited":
      new_ok = base.element_text_contains(ui_elements, (self._params["new_title"],))
      old_gone = not base.element_text_contains(ui_elements, (self._params["old_title"],))
      return 1.0 if new_ok and old_gone else 0.0
    if mode == "deleted":
      title_present = base.element_text_contains(
          ui_elements, (self._params["note_title"],)
      )
      list_markers = ("notes", "all notes", "empty", "no notes", "new note", "+")
      list_ok = base.element_text_contains(ui_elements, list_markers)
      return 1.0 if not title_present and list_ok else 0.0
    if mode == "merged":
      return (
          1.0
          if base.element_text_contains(ui_elements, (self._params["merged_title"],))
          else 0.0
      )
    if mode == "folder":
      return (
          1.0
          if base.element_text_contains(ui_elements, (self._params["folder_name"],))
          else 0.0
      )
    if mode == "count":
      answer = str(getattr(env, "interaction_cache", "")).strip().lower()
      expected = str(self._params["todo_count"])
      if answer and expected in answer:
        return 1.0
      return (
          1.0
          if base.element_text_contains(ui_elements, (self._params["note_title"],))
          else 0.0
      )
    return (
        1.0
        if base.element_text_contains(ui_elements, (self._params["note_title"],))
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_note_params()


# -----------------------------------------------------------------------------
# Per-app packages.
# -----------------------------------------------------------------------------

_MARKOR_PACKAGE: Final[str] = "net.gsantner.markor"
_JOPLIN_PACKAGE: Final[str] = "net.cozic.joplin"
_NOTALLYX_PACKAGE: Final[str] = "com.philkes.notallyx"
_NOTESNOOK_PACKAGE: Final[str] = "com.streetwriters.notesnook"
_NEUTRINOTE_PACKAGE: Final[str] = "com.appmindlab.nano"
_ORGZLY_REVIVED_PACKAGE: Final[str] = "com.orgzlyrevived"
_ORGZLY_DB_PATH: Final[str] = "/data/data/com.orgzlyrevived/databases/orgzly.db"
_JOPLIN_DB_PATH: Final[str] = "/data/data/net.cozic.joplin/databases/joplin.sqlite"


@dataclasses.dataclass(frozen=True)
class _OrgzlyBookRow(sqlite_schema_utils.SQLiteRow):
  name: str
  title: str | None = None
  is_deleted: int = 0


@dataclasses.dataclass(frozen=True)
class _JoplinFolderRow(sqlite_schema_utils.SQLiteRow):
  title: str
  deleted_time: int = 0


def _list_joplin_folders(env: interface.AsyncEnv) -> list[_JoplinFolderRow]:
  try:
    with env.controller.pull_file(_JOPLIN_DB_PATH) as local_db_directory:
      local_db_path = file_utils.convert_to_posix_path(
          local_db_directory,
          "joplin.sqlite",
      )
      return sqlite_utils.execute_query(
          "SELECT title, deleted_time FROM folders;",
          local_db_path,
          _JoplinFolderRow,
      )
  except FileNotFoundError:
    return []


def _list_orgzly_books(env: interface.AsyncEnv) -> list[_OrgzlyBookRow]:
  try:
    with env.controller.pull_file(_ORGZLY_DB_PATH) as local_db_directory:
      local_db_path = file_utils.convert_to_posix_path(
          local_db_directory,
          "orgzly.db",
      )
      return sqlite_utils.execute_query(
          "SELECT name, title, is_deleted FROM books;",
          local_db_path,
          _OrgzlyBookRow,
      )
  except FileNotFoundError:
    return []


# ----------- Markor ----------


class NotesCreateFolderForMarkor(_NotesCreateFolderBase):
  app_names = (_MARKOR_PACKAGE,)
  package_name = _MARKOR_PACKAGE
  template = (
      "In the Markor app, create a folder named '{folder_name}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    expected = self._params["folder_name"].strip()
    return (
        1.0
        if file_utils.check_file_or_folder_exists(
            expected,
            device_constants.MARKOR_DATA,
            env.controller,
        )
        else 0.0
    )


# ----------- Joplin ----------


class NotesCreateFolderForJoplin(_NotesCreateFolderBase):
  app_names = (_JOPLIN_PACKAGE,)
  package_name = _JOPLIN_PACKAGE
  template = (
      "In the Joplin app, create a notebook named '{folder_name}'."
  )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    expected = self._params["folder_name"].strip()
    return (
        1.0
        if any(
            row.title.strip() == expected and not row.deleted_time
            for row in _list_joplin_folders(env)
        )
        else 0.0
    )


# ----------- NotallyX ----------


class NotesCreateFolderForNotallyX(_NotesCreateFolderBase):
  app_names = (_NOTALLYX_PACKAGE,)
  package_name = _NOTALLYX_PACKAGE
  template = (
      "In the NotallyX app, create a label named '{folder_name}'."
  )


# ----------- Notesnook ----------


class NotesCreateFolderForNotesnook(_NotesCreateFolderBase):
  app_names = (_NOTESNOOK_PACKAGE,)
  package_name = _NOTESNOOK_PACKAGE
  template = (
      "In the Notesnook app, create a notebook named '{folder_name}'."
  )


# ----------- neutriNote CE ----------


class NotesCreateFolderForNeutriNote(_NotesCreateFolderBase):
  app_names = (_NEUTRINOTE_PACKAGE,)
  package_name = _NEUTRINOTE_PACKAGE
  template = (
      "In the neutriNote CE app, create a tag named '{folder_name}'."
  )


# ----------- Orgzly Revived ----------


class NotesCreateFolderForOrgzlyRevived(_NotesCreateFolderBase):
  app_names = (_ORGZLY_REVIVED_PACKAGE,)
  package_name = _ORGZLY_REVIVED_PACKAGE
  template = (
      "In the Orgzly Revived app, create a notebook named '{folder_name}'."
  )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    expected = self._params["folder_name"].strip()
    candidates = {expected, f"{expected}.org"}
    for book in _list_orgzly_books(env):
      if book.is_deleted:
        continue
      values = {book.name.strip()}
      if book.title:
        values.add(book.title.strip())
      if candidates & values:
        return 1.0
    return 0.0


def _make_notes_task(
    class_name: str,
    package_name: str,
    template: str,
    success_mode: str,
) -> type[_NotesGenericTaskBase]:
  return type(
      class_name,
      (_NotesGenericTaskBase,),
      {
          "__module__": __name__,
          "app_names": (package_name,),
          "package_name": package_name,
          "template": template,
          "success_mode": success_mode,
      },
  )


_NOTES_APP_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    ("Markor", _MARKOR_PACKAGE, "Markor"),
    ("Joplin", _JOPLIN_PACKAGE, "Joplin"),
    ("NotallyX", _NOTALLYX_PACKAGE, "NotallyX"),
    ("NeutriNote", _NEUTRINOTE_PACKAGE, "neutriNote CE"),
    ("Notesnook", _NOTESNOOK_PACKAGE, "Notesnook"),
    ("OrgzlyRevived", _ORGZLY_REVIVED_PACKAGE, "Orgzly Revived"),
)

_NOTES_TEMPLATE_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "CreateNote",
        "In the {app} app, create a note titled '{note_title}' with the body "
        "'{note_body}'.",
        "note_title",
    ),
    (
        "CreateChecklist",
        "In the {app} app, create a checklist note titled '{note_title}' with "
        "{todo_count} unchecked todo items.",
        "note_title",
    ),
    (
        "EditNote",
        "In the {app} app, create a note titled '{old_title}', then edit it so "
        "the title becomes '{new_title}'.",
        "edited",
    ),
    (
        "MergeNotes",
        "In the {app} app, create notes titled '{old_title}' and '{note_title}', "
        "then merge or consolidate them into a note titled '{merged_title}'.",
        "merged",
    ),
    (
        "DeleteNote",
        "In the {app} app, create a note titled '{note_title}', then delete that "
        "note.",
        "deleted",
    ),
    (
        "SearchNote",
        "In the {app} app, create a note titled '{note_title}', then search for "
        "that note and open it.",
        "note_title",
    ),
    (
        "ShareImport",
        "In the {app} app, import or create a shared-text note titled "
        "'{note_title}' containing '{note_body}'.",
        "note_title",
    ),
    (
        "AttachContent",
        "In the {app} app, create a note titled '{note_title}' and attach or "
        "insert supporting content into it.",
        "note_title",
    ),
    (
        "CountTodoItems",
        "In the {app} app, create a checklist note titled '{note_title}' with "
        "{todo_count} todo items, then answer with the number of todo items.",
        "count",
    ),
)

for _suffix, _package, _display_name in _NOTES_APP_SPECS:
  for _task_name, _template, _mode in _NOTES_TEMPLATE_SPECS:
    globals()[f"Notes{_task_name}For{_suffix}"] = _make_notes_task(
        f"Notes{_task_name}For{_suffix}",
        _package,
        _template.replace("{app}", _display_name),
        _mode,
    )
