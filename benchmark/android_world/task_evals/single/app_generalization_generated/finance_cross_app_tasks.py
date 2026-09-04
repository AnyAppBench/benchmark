"""Cross-app finance/expense task ports for the app-generalization suite.

Each port mirrors one of the canonical AndroidWorld ``expense.py`` tasks
(add single, add multiple, delete single, delete multiple, delete duplicates,
add multiple from gallery, add multiple from Markor) plus three additional
cross-app templates (edit expense, add category, view monthly total).

Only app-task pairs with durable database verification should be active in
``app_generalization_profiles.py``. The generic base classes remain as
scaffolding for inactive ports; My Expenses overrides them with SQLite-backed
setup and verification.
"""

from __future__ import annotations

import dataclasses
import os
import random
import sqlite3
import time
import uuid
from typing import Any, Final

from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.env.setup_device import apps
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base as base,
)
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils


# -----------------------------------------------------------------------------
# Parameter randomization pools.
# -----------------------------------------------------------------------------

_EXPENSE_NOTES: Final[tuple[str, ...]] = (
    "Lunch at cafe",
    "Bus ticket",
    "Coffee",
    "Groceries",
    "Movie ticket",
    "Book purchase",
    "Gym membership",
    "Gas",
    "Pharmacy",
    "Streaming subscription",
)

_EXPENSE_CATEGORIES: Final[tuple[str, ...]] = (
    "Food",
    "Entertainment",
    "Transportation",
    "Housing",
    "Health Care",
    "Education",
    "Others",
)

_EXPENSE_AMOUNTS: Final[tuple[float, ...]] = (
    4.50,
    9.99,
    12.00,
    18.75,
    25.00,
    32.40,
    45.99,
    60.00,
    95.50,
    120.00,
)

_MONTHS: Final[tuple[str, ...]] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


# -----------------------------------------------------------------------------
# Param helpers.
# -----------------------------------------------------------------------------


def _generate_add_single_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "note": random.choice(_EXPENSE_NOTES),
      "amount": random.choice(_EXPENSE_AMOUNTS),
      "category": random.choice(_EXPENSE_CATEGORIES),
  }


def _generate_expense_batch(size: int = 3) -> list[dict[str, Any]]:
  notes = random.sample(_EXPENSE_NOTES, size)
  return [
      {
          "note": notes[i],
          "amount": random.choice(_EXPENSE_AMOUNTS),
          "category": random.choice(_EXPENSE_CATEGORIES),
      }
      for i in range(size)
  ]


def _generate_add_multiple_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "expenses": _generate_expense_batch(3),
  }


def _generate_delete_single_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "note_to_delete": random.choice(_EXPENSE_NOTES),
  }


def _generate_delete_multiple_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "notes_to_delete": list(random.sample(_EXPENSE_NOTES, 3)),
  }


def _generate_delete_duplicates_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "note_to_dedupe": random.choice(_EXPENSE_NOTES),
  }


def _generate_add_multiple_from_gallery_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "expenses": _generate_expense_batch(3),
  }


def _generate_add_multiple_from_markor_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "expenses": _generate_expense_batch(3),
  }


def _generate_edit_expense_params() -> dict[str, Any]:
  old_note, new_note = random.sample(_EXPENSE_NOTES, 2)
  return {
      "seed": random.randint(0, 1_000_000),
      "old_note": old_note,
      "new_note": new_note,
      "new_amount": random.choice(_EXPENSE_AMOUNTS),
  }


def _generate_add_category_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "category_name": random.choice(_EXPENSE_CATEGORIES),
  }


def _generate_view_monthly_total_params() -> dict[str, Any]:
  return {
      "seed": random.randint(0, 1_000_000),
      "month": random.choice(_MONTHS),
  }


# -----------------------------------------------------------------------------
# Base evaluators (shared by every app port).
# -----------------------------------------------------------------------------


class _ExpenseAddSingleBase(base.PackageAppEval):
  """Base port of ``ExpenseAddSingle``.

  Success heuristic: the chosen expense note appears in the UI element list
  after the agent saves the expense.
  """

  complexity = 2.0
  schema = {
      "type": "object",
      "properties": {
          "note": {"type": "string"},
          "amount": {"type": "number"},
          "category": {"type": "string"},
      },
      "required": ["note", "amount", "category"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if base.element_text_contains(
            env.get_state().ui_elements,
            (self._params["note"],),
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_add_single_params()


class _ExpenseAddMultipleBase(base.PackageAppEval):
  """Base port of ``ExpenseAddMultiple``.

  Success heuristic: at least two of the three expense notes appear in the UI
  element list after the agent saves the batch.
  """

  complexity = 3.0
  schema = {
      "type": "object",
      "properties": {
          "expenses": {
              "type": "array",
              "items": {
                  "type": "object",
                  "properties": {
                      "note": {"type": "string"},
                      "amount": {"type": "number"},
                      "category": {"type": "string"},
                  },
                  "required": ["note", "amount", "category"],
              },
          },
      },
      "required": ["expenses"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    matches = sum(
        1
        for expense in self._params["expenses"]
        if base.element_text_contains(ui_elements, (expense["note"],))
    )
    return 1.0 if matches >= 2 else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_add_multiple_params()


class _ExpenseDeleteSingleBase(base.PackageAppEval):
  """Base port of ``ExpenseDeleteSingle``.

  Success heuristic: the chosen note is no longer present in the UI element
  list after the agent deletes it.
  """

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "note_to_delete": {"type": "string"},
      },
      "required": ["note_to_delete"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        0.0
        if base.element_text_contains(
            env.get_state().ui_elements,
            (self._params["note_to_delete"],),
        )
        else 1.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_delete_single_params()


class _ExpenseDeleteMultipleBase(base.PackageAppEval):
  """Base port of ``ExpenseDeleteMultiple``.

  Success heuristic: none of the chosen notes are present in the UI element
  list after the agent finishes deleting them.
  """

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {
          "notes_to_delete": {
              "type": "array",
              "items": {"type": "string"},
          },
      },
      "required": ["notes_to_delete"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    any_present = any(
        base.element_text_contains(ui_elements, (note,))
        for note in self._params["notes_to_delete"]
    )
    return 0.0 if any_present else 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_delete_multiple_params()


class _ExpenseDeleteDuplicatesBase(base.PackageAppEval):
  """Base port of ``ExpenseDeleteDuplicates``.

  Success heuristic: the duplicated note appears at most once in the UI
  element list (the duplicate row was removed).
  """

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {
          "note_to_dedupe": {"type": "string"},
      },
      "required": ["note_to_dedupe"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    count = sum(
        1
        for el in ui_elements
        if base.matches_any_text(el, (self._params["note_to_dedupe"],))
    )
    return 1.0 if count <= 1 else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_delete_duplicates_params()


class _ExpenseAddMultipleFromGalleryBase(base.PackageAppEval):
  """Base port of ``ExpenseAddMultipleFromGallery``.

  Success heuristic: at least two of the three expense notes (sourced from a
  gallery image batch) appear in the UI element list after the agent saves.
  """

  complexity = 3.4
  schema = {
      "type": "object",
      "properties": {
          "expenses": {
              "type": "array",
              "items": {
                  "type": "object",
                  "properties": {
                      "note": {"type": "string"},
                      "amount": {"type": "number"},
                      "category": {"type": "string"},
                  },
                  "required": ["note", "amount", "category"],
              },
          },
      },
      "required": ["expenses"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    matches = sum(
        1
        for expense in self._params["expenses"]
        if base.element_text_contains(ui_elements, (expense["note"],))
    )
    return 1.0 if matches >= 2 else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_add_multiple_from_gallery_params()


class _ExpenseAddMultipleFromMarkorBase(base.PackageAppEval):
  """Base port of ``ExpenseAddMultipleFromMarkor``.

  Success heuristic: at least two of the three expense notes (sourced from a
  Markor note) appear in the UI element list after the agent saves.
  """

  complexity = 3.4
  schema = {
      "type": "object",
      "properties": {
          "expenses": {
              "type": "array",
              "items": {
                  "type": "object",
                  "properties": {
                      "note": {"type": "string"},
                      "amount": {"type": "number"},
                      "category": {"type": "string"},
                  },
                  "required": ["note", "amount", "category"],
              },
          },
      },
      "required": ["expenses"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    matches = sum(
        1
        for expense in self._params["expenses"]
        if base.element_text_contains(ui_elements, (expense["note"],))
    )
    return 1.0 if matches >= 2 else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_add_multiple_from_markor_params()


class _ExpenseEditExpenseBase(base.PackageAppEval):
  """Base port of ``ExpenseEditExpense``.

  Success heuristic: the new note appears in the UI element list AND the old
  note no longer appears after the agent saves the edit.
  """

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {
          "old_note": {"type": "string"},
          "new_note": {"type": "string"},
          "new_amount": {"type": "number"},
      },
      "required": ["old_note", "new_note", "new_amount"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    new_present = base.element_text_contains(
        ui_elements, (self._params["new_note"],)
    )
    old_present = base.element_text_contains(
        ui_elements, (self._params["old_note"],)
    )
    return 1.0 if new_present and not old_present else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_edit_expense_params()


class _ExpenseAddCategoryBase(base.PackageAppEval):
  """Base port of ``ExpenseAddCategory``.

  Success heuristic: the new category name appears in the UI element list
  after the agent creates it.
  """

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "category_name": {"type": "string"},
      },
      "required": ["category_name"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if base.element_text_contains(
            env.get_state().ui_elements,
            (self._params["category_name"],),
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_add_category_params()


class _ExpenseViewMonthlyTotalBase(base.PackageAppEval):
  """Base port of ``ExpenseViewMonthlyTotal``.

  Success heuristic: any of the markers ("total", "sum", "spent", or the
  lower-cased month name) appears in the UI element list after the agent
  navigates to the monthly total view.
  """

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "month": {"type": "string"},
      },
      "required": ["month"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    markers = ("total", "sum", "spent", self._params["month"].lower())
    return 1.0 if base.element_text_contains(ui_elements, markers) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_view_monthly_total_params()


class _ExpenseAddIncomeBase(_ExpenseAddSingleBase):
  """Base port for adding an income transaction."""

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = _generate_add_single_params()
    params["category"] = "Income"
    return params


class _ExpenseAttachReceiptBase(_ExpenseAddMultipleFromGalleryBase):
  """Base port for adding an expense with receipt evidence attached."""


class _ExpenseCategorySummaryBase(base.PackageAppEval):
  """Base port for opening a category-level spending summary."""

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "category_name": {"type": "string"},
      },
      "required": ["category_name"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    category_ok = base.element_text_contains(
        ui_elements, (self._params["category_name"],)
    )
    summary_ok = base.element_text_contains(
        ui_elements, ("summary", "category", "total", "spent", "spending")
    )
    return 1.0 if category_ok and summary_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return _generate_add_category_params()


class _ExpenseDateRangeTotalBase(_ExpenseViewMonthlyTotalBase):
  """Base port for viewing total spending over a date range."""

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    markers = ("total", "sum", "spent", "range", "from", "to")
    return (
        1.0
        if base.element_text_contains(env.get_state().ui_elements, markers)
        else 0.0
    )


class _ExpenseTransferBetweenWalletsBase(base.PackageAppEval):
  """Base port for starting a transfer between wallets/accounts."""

  complexity = 2.0
  schema = {
      "type": "object",
      "properties": {
          "amount": {"type": "number"},
      },
      "required": ["amount"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    markers = ("transfer", "wallet", "account", str(self._params["amount"]))
    return (
        1.0
        if base.element_text_contains(env.get_state().ui_elements, markers)
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "seed": random.randint(0, 1_000_000),
        "amount": random.choice(_EXPENSE_AMOUNTS),
    }


# -----------------------------------------------------------------------------
# Per-app package constants.
# -----------------------------------------------------------------------------

_OINKOIN_PACKAGE: Final[str] = "com.github.emavgl.piggybankpro"
_OPENMONEYBOX_PACKAGE: Final[str] = "com.igisw.openmoneybox"
_MYEXPENSES_PACKAGE: Final[str] = "org.totschnig.myexpenses"
_FINANCEMANAGER_PACKAGE: Final[str] = "org.secuso.privacyfriendlyfinancemanager"
_SUSHI_PACKAGE: Final[str] = "com.jerameeldelosreyes.sushi"
_PROEXPENSE_PACKAGE: Final[str] = "com.arduia.expense"
_MYEXPENSES_DB_PATH: Final[str] = (
    "/data/data/org.totschnig.myexpenses/databases/data"
)
_PROEXPENSE_DB_PATH: Final[str] = (
    "/data/data/com.arduia.expense/databases/accounting.db"
)
_PROEXPENSE_TABLE: Final[str] = "expense"
_PROEXPENSE_APP_NAME: Final[str] = "pro expense"
_MYEXPENSES_ACCOUNT_LABEL: Final[str] = "CATBench Cash"
_MYEXPENSES_DATE_MS: Final[int] = 1_697_328_000_000
_PROEXPENSE_DATE_MS: Final[int] = 1_697_328_000_000
_PROEXPENSE_CATEGORY_BY_NAME: Final[dict[str, int]] = {
    name: category_id
    for category_id, name in sqlite_schema_utils.Expense.category_id_to_name.items()
}


@dataclasses.dataclass(frozen=True)
class _MyExpensesTransaction:
  row_id: int
  comment: str
  amount: int


@dataclasses.dataclass(frozen=True)
class _ProExpenseTransaction:
  row_id: int
  name: str
  note: str
  amount: int


def _amount_to_cents(amount: float) -> int:
  return int(round(float(amount) * 100))


def _force_stop_package(package_name: str, env: interface.AsyncEnv) -> None:
  adb_utils.issue_generic_request(
      ["shell", "am", "force-stop", package_name],
      env.controller,
      timeout_sec=5,
  )


def _with_myexpenses_db(env: interface.AsyncEnv, mutator):
  with env.controller.pull_file(_MYEXPENSES_DB_PATH) as local_db_directory:
    local_db_path = file_utils.convert_to_posix_path(
        local_db_directory,
        os.path.basename(_MYEXPENSES_DB_PATH),
    )
    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row
    try:
      result = mutator(conn)
      conn.commit()
    finally:
      conn.close()
    env.controller.push_file(local_db_path, _MYEXPENSES_DB_PATH)
  _force_stop_package(_MYEXPENSES_PACKAGE, env)
  return result


def _with_proexpense_db(env: interface.AsyncEnv, mutator):
  if not sqlite_utils.table_exists(_PROEXPENSE_TABLE, _PROEXPENSE_DB_PATH, env):
    apps.ExpenseApp.setup(env)
  with env.controller.pull_file(_PROEXPENSE_DB_PATH) as local_db_directory:
    local_db_path = file_utils.convert_to_posix_path(
        local_db_directory,
        os.path.basename(_PROEXPENSE_DB_PATH),
    )
    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row
    try:
      result = mutator(conn)
      conn.commit()
    finally:
      conn.close()
    env.controller.push_file(local_db_path, _PROEXPENSE_DB_PATH)
  adb_utils.close_app(_PROEXPENSE_APP_NAME, env.controller)
  return result


def _read_myexpenses_db(env: interface.AsyncEnv, reader):
  try:
    with env.controller.pull_file(_MYEXPENSES_DB_PATH) as local_db_directory:
      local_db_path = file_utils.convert_to_posix_path(
          local_db_directory,
          os.path.basename(_MYEXPENSES_DB_PATH),
      )
      conn = sqlite3.connect(local_db_path)
      conn.row_factory = sqlite3.Row
      try:
        return reader(conn)
      finally:
        conn.close()
  except FileNotFoundError:
    return []


def _myexpenses_schema_ready(env: interface.AsyncEnv) -> bool:
  def read(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master"
        " WHERE type = 'table' AND name = 'transactions'"
    ).fetchone()
    return row is not None

  return bool(_read_myexpenses_db(env, read))


def _wait_for_myexpenses_schema(env: interface.AsyncEnv) -> None:
  deadline = time.time() + 20.0
  while time.time() < deadline:
    if _myexpenses_schema_ready(env):
      return
    adb_utils.launch_app(_MYEXPENSES_PACKAGE, env.controller)
    time.sleep(2.0)
  raise RuntimeError("Timed out waiting for My Expenses database schema.")


def _read_proexpense_db(env: interface.AsyncEnv, reader):
  try:
    if not sqlite_utils.table_exists(_PROEXPENSE_TABLE, _PROEXPENSE_DB_PATH, env):
      apps.ExpenseApp.setup(env)
    with env.controller.pull_file(_PROEXPENSE_DB_PATH) as local_db_directory:
      local_db_path = file_utils.convert_to_posix_path(
          local_db_directory,
          os.path.basename(_PROEXPENSE_DB_PATH),
      )
      conn = sqlite3.connect(local_db_path)
      conn.row_factory = sqlite3.Row
      try:
        return reader(conn)
      finally:
        conn.close()
  except (FileNotFoundError, ValueError, sqlite3.OperationalError):
    return []


def _prepare_myexpenses_db(env: interface.AsyncEnv) -> int:
  def mutate(conn: sqlite3.Connection) -> int:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions")
    cursor.execute("DELETE FROM accounts")
    cursor.execute(
        "DELETE FROM categories WHERE label NOT IN"
        " ('__SPLIT_TRANSACTION__', 'Transfer')"
    )
    cursor.execute(
        "INSERT INTO accounts"
        " (label, opening_balance, currency, type, grouping, sort_direction,"
        " flag, sealed, dynamic)"
        " VALUES (?, 0, 'USD', 1, 'NONE', 'DESC', 0, 0, 0)",
        (_MYEXPENSES_ACCOUNT_LABEL,),
    )
    return int(cursor.lastrowid)

  return _with_myexpenses_db(env, mutate)


def _prepare_proexpense_db(env: interface.AsyncEnv) -> None:
  def mutate(conn: sqlite3.Connection) -> None:
    conn.execute(f"DELETE FROM {_PROEXPENSE_TABLE}")

  _with_proexpense_db(env, mutate)


def _myexpenses_category_id(
    conn: sqlite3.Connection,
    category_name: str,
) -> int:
  cursor = conn.cursor()
  cursor.execute(
      "INSERT OR IGNORE INTO categories"
      " (label, label_normalized, parent_id, usages, type)"
      " VALUES (?, ?, NULL, 0, -1)",
      (category_name, category_name.lower()),
  )
  row = cursor.execute(
      "SELECT _id FROM categories WHERE label = ?",
      (category_name,),
  ).fetchone()
  if row is None:
    raise RuntimeError(f"Failed to create My Expenses category {category_name}.")
  return int(row["_id"])


def _proexpense_category_id(category_name: str) -> int:
  return _PROEXPENSE_CATEGORY_BY_NAME.get(category_name, 1)


def _insert_myexpenses_transactions(
    env: interface.AsyncEnv,
    account_id: int,
    expenses: list[dict[str, Any]],
) -> None:
  def mutate(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    for expense in expenses:
      cat_id = _myexpenses_category_id(conn, expense["category"])
      cursor.execute(
          "INSERT INTO transactions"
          " (comment, date, value_date, amount, cat_id, account_id,"
          " cr_status, uuid)"
          " VALUES (?, ?, ?, ?, ?, ?, 'RECONCILED', ?)",
          (
              expense["note"],
              _MYEXPENSES_DATE_MS,
              _MYEXPENSES_DATE_MS,
              -_amount_to_cents(expense["amount"]),
              cat_id,
              account_id,
              str(uuid.uuid4()),
          ),
      )

  _with_myexpenses_db(env, mutate)


def _insert_proexpense_transactions(
    env: interface.AsyncEnv,
    expenses: list[dict[str, Any]],
) -> None:
  rows = [
      sqlite_schema_utils.Expense(
          name=expense["note"],
          amount=_amount_to_cents(expense["amount"]),
          category=_proexpense_category_id(expense["category"]),
          note=expense["note"],
          created_date=_PROEXPENSE_DATE_MS,
          modified_date=_PROEXPENSE_DATE_MS,
      )
      for expense in expenses
  ]
  sqlite_utils.insert_rows_to_remote_db(
      rows,
      "expense_id",
      _PROEXPENSE_TABLE,
      _PROEXPENSE_DB_PATH,
      _PROEXPENSE_APP_NAME,
      env,
  )


def _list_myexpenses_transactions(
    env: interface.AsyncEnv,
) -> list[_MyExpensesTransaction]:
  def read(conn: sqlite3.Connection) -> list[_MyExpensesTransaction]:
    rows = conn.execute(
        "SELECT _id, comment, amount FROM transactions"
    ).fetchall()
    return [
        _MyExpensesTransaction(
            row_id=int(row["_id"]),
            comment=(row["comment"] or "").strip(),
            amount=int(row["amount"]),
        )
        for row in rows
    ]

  return _read_myexpenses_db(env, read)


def _list_proexpense_transactions(
    env: interface.AsyncEnv,
) -> list[_ProExpenseTransaction]:
  def read(conn: sqlite3.Connection) -> list[_ProExpenseTransaction]:
    rows = conn.execute(
        "SELECT expense_id, name, note, amount FROM expense"
    ).fetchall()
    return [
        _ProExpenseTransaction(
            row_id=int(row["expense_id"]),
            name=(row["name"] or "").strip(),
            note=(row["note"] or "").strip(),
            amount=int(row["amount"]),
        )
        for row in rows
    ]

  return _read_proexpense_db(env, read)


def _list_myexpenses_categories(env: interface.AsyncEnv) -> set[str]:
  def read(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row["label"]).strip()
        for row in conn.execute("SELECT label FROM categories").fetchall()
    }

  return _read_myexpenses_db(env, read)


def _expense_rows_as_text(expenses: list[dict[str, Any]]) -> str:
  lines = ["note,amount,category"]
  lines.extend(
      f"{expense['note']},{expense['amount']},{expense['category']}"
      for expense in expenses
  )
  return "\n".join(lines)


def _matching_myexpenses_rows(
    env: interface.AsyncEnv,
    note: str,
) -> list[_MyExpensesTransaction]:
  return [row for row in _list_myexpenses_transactions(env) if row.comment == note]


def _matching_proexpense_rows(
    env: interface.AsyncEnv,
    note: str,
) -> list[_ProExpenseTransaction]:
  return [
      row
      for row in _list_proexpense_transactions(env)
      if row.name == note or row.note == note
  ]


def _myexpenses_row_exists(
    env: interface.AsyncEnv,
    note: str,
    amount: float | None = None,
) -> bool:
  expected_amount = None if amount is None else _amount_to_cents(amount)
  for row in _matching_myexpenses_rows(env, note):
    if expected_amount is None or abs(row.amount) == expected_amount:
      return True
  return False


def _proexpense_row_exists(
    env: interface.AsyncEnv,
    note: str,
    amount: float | None = None,
) -> bool:
  expected_amount = None if amount is None else _amount_to_cents(amount)
  for row in _matching_proexpense_rows(env, note):
    if expected_amount is None or abs(row.amount) == expected_amount:
      return True
  return False


def _write_expenses_to_gallery(
    env: interface.AsyncEnv,
    expenses: list[dict[str, Any]],
) -> None:
  user_data_generation.clear_device_storage(env)
  user_data_generation.write_to_gallery(
      _expense_rows_as_text(expenses),
      "expenses.jpg",
      env,
  )


def _write_expenses_to_markor(
    env: interface.AsyncEnv,
    expenses: list[dict[str, Any]],
) -> None:
  file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)
  user_data_generation.write_to_markor(
      _expense_rows_as_text(expenses),
      "my_expenses.txt",
      env,
  )


def _initialize_myexpenses_task(
    task: base.PackageAppEval,
    env: interface.AsyncEnv,
    seed_expenses: list[dict[str, Any]] | None = None,
) -> None:
  base.PackageAppEval.initialize_task(task, env)
  _wait_for_myexpenses_schema(env)
  _force_stop_package(_MYEXPENSES_PACKAGE, env)
  account_id = _prepare_myexpenses_db(env)
  if seed_expenses:
    _insert_myexpenses_transactions(env, account_id, seed_expenses)
  adb_utils.launch_app(_MYEXPENSES_PACKAGE, env.controller)


def _initialize_proexpense_task(
    task: base.PackageAppEval,
    env: interface.AsyncEnv,
    seed_expenses: list[dict[str, Any]] | None = None,
) -> None:
  base.PackageAppEval.initialize_task(task, env)
  time.sleep(3.0)
  _force_stop_package(_PROEXPENSE_PACKAGE, env)
  _prepare_proexpense_db(env)
  if seed_expenses:
    _insert_proexpense_transactions(env, seed_expenses)
  adb_utils.launch_app(_PROEXPENSE_PACKAGE, env.controller)


# -----------------------------------------------------------------------------
# Oinkoin
# -----------------------------------------------------------------------------


class ExpenseAddSingleForOinkoin(_ExpenseAddSingleBase):
  app_names = (_OINKOIN_PACKAGE,)
  package_name = _OINKOIN_PACKAGE
  template = (
      "In the Oinkoin app, add a new expense for {amount} with the note"
      " '{note}' under the category '{category}'."
  )


class ExpenseAddMultipleForOinkoin(_ExpenseAddMultipleBase):
  app_names = (_OINKOIN_PACKAGE,)
  package_name = _OINKOIN_PACKAGE
  template = (
      "In the Oinkoin app, add the following expenses one by one: {expenses}."
      " Each entry should use its own note, amount, and category."
  )


class ExpenseDeleteSingleForOinkoin(_ExpenseDeleteSingleBase):
  app_names = (_OINKOIN_PACKAGE,)
  package_name = _OINKOIN_PACKAGE
  template = (
      "In the Oinkoin app, delete the expense whose note is"
      " '{note_to_delete}'."
  )


class ExpenseDeleteMultipleForOinkoin(_ExpenseDeleteMultipleBase):
  app_names = (_OINKOIN_PACKAGE,)
  package_name = _OINKOIN_PACKAGE
  template = (
      "In the Oinkoin app, delete every expense whose note is in this list:"
      " {notes_to_delete}."
  )


class ExpenseDeleteDuplicatesForOinkoin(_ExpenseDeleteDuplicatesBase):
  app_names = (_OINKOIN_PACKAGE,)
  package_name = _OINKOIN_PACKAGE
  template = (
      "In the Oinkoin app, find duplicate expenses with the note"
      " '{note_to_dedupe}' and delete the duplicates so only one remains."
  )


class ExpenseAddMultipleFromGalleryForOinkoin(
    _ExpenseAddMultipleFromGalleryBase
):
  app_names = (_OINKOIN_PACKAGE,)
  package_name = _OINKOIN_PACKAGE
  template = (
      "In the Oinkoin app, add the following expenses sourced from a receipt"
      " image in the gallery: {expenses}."
  )


class ExpenseAddMultipleFromMarkorForOinkoin(_ExpenseAddMultipleFromMarkorBase):
  app_names = (_OINKOIN_PACKAGE,)
  package_name = _OINKOIN_PACKAGE
  template = (
      "In the Oinkoin app, add the following expenses sourced from a Markor"
      " note: {expenses}."
  )


class ExpenseEditExpenseForOinkoin(_ExpenseEditExpenseBase):
  app_names = (_OINKOIN_PACKAGE,)
  package_name = _OINKOIN_PACKAGE
  template = (
      "In the Oinkoin app, edit the expense titled '{old_note}' so its note"
      " becomes '{new_note}' and its amount becomes {new_amount}."
  )


class ExpenseAddCategoryForOinkoin(_ExpenseAddCategoryBase):
  app_names = (_OINKOIN_PACKAGE,)
  package_name = _OINKOIN_PACKAGE
  template = (
      "In the Oinkoin app, create a new expense category named"
      " '{category_name}'."
  )


class ExpenseViewMonthlyTotalForOinkoin(_ExpenseViewMonthlyTotalBase):
  app_names = (_OINKOIN_PACKAGE,)
  package_name = _OINKOIN_PACKAGE
  template = (
      "In the Oinkoin app, navigate to the view that shows the total expenses"
      " for {month}."
  )


# -----------------------------------------------------------------------------
# OpenMoneyBox
# -----------------------------------------------------------------------------


class ExpenseAddSingleForOpenMoneyBox(_ExpenseAddSingleBase):
  app_names = (_OPENMONEYBOX_PACKAGE,)
  package_name = _OPENMONEYBOX_PACKAGE
  template = (
      "In the OpenMoneyBox app, add a new expense for {amount} with the note"
      " '{note}' under the category '{category}'."
  )


class ExpenseAddMultipleForOpenMoneyBox(_ExpenseAddMultipleBase):
  app_names = (_OPENMONEYBOX_PACKAGE,)
  package_name = _OPENMONEYBOX_PACKAGE
  template = (
      "In the OpenMoneyBox app, add the following expenses one by one:"
      " {expenses}. Each entry should use its own note, amount, and category."
  )


class ExpenseDeleteSingleForOpenMoneyBox(_ExpenseDeleteSingleBase):
  app_names = (_OPENMONEYBOX_PACKAGE,)
  package_name = _OPENMONEYBOX_PACKAGE
  template = (
      "In the OpenMoneyBox app, delete the expense whose note is"
      " '{note_to_delete}'."
  )


class ExpenseDeleteMultipleForOpenMoneyBox(_ExpenseDeleteMultipleBase):
  app_names = (_OPENMONEYBOX_PACKAGE,)
  package_name = _OPENMONEYBOX_PACKAGE
  template = (
      "In the OpenMoneyBox app, delete every expense whose note is in this"
      " list: {notes_to_delete}."
  )


class ExpenseDeleteDuplicatesForOpenMoneyBox(_ExpenseDeleteDuplicatesBase):
  app_names = (_OPENMONEYBOX_PACKAGE,)
  package_name = _OPENMONEYBOX_PACKAGE
  template = (
      "In the OpenMoneyBox app, find duplicate expenses with the note"
      " '{note_to_dedupe}' and delete the duplicates so only one remains."
  )


class ExpenseAddMultipleFromGalleryForOpenMoneyBox(
    _ExpenseAddMultipleFromGalleryBase
):
  app_names = (_OPENMONEYBOX_PACKAGE,)
  package_name = _OPENMONEYBOX_PACKAGE
  template = (
      "In the OpenMoneyBox app, add the following expenses sourced from a"
      " receipt image in the gallery: {expenses}."
  )


class ExpenseAddMultipleFromMarkorForOpenMoneyBox(
    _ExpenseAddMultipleFromMarkorBase
):
  app_names = (_OPENMONEYBOX_PACKAGE,)
  package_name = _OPENMONEYBOX_PACKAGE
  template = (
      "In the OpenMoneyBox app, add the following expenses sourced from a"
      " Markor note: {expenses}."
  )


class ExpenseEditExpenseForOpenMoneyBox(_ExpenseEditExpenseBase):
  app_names = (_OPENMONEYBOX_PACKAGE,)
  package_name = _OPENMONEYBOX_PACKAGE
  template = (
      "In the OpenMoneyBox app, edit the expense titled '{old_note}' so its"
      " note becomes '{new_note}' and its amount becomes {new_amount}."
  )


class ExpenseAddCategoryForOpenMoneyBox(_ExpenseAddCategoryBase):
  app_names = (_OPENMONEYBOX_PACKAGE,)
  package_name = _OPENMONEYBOX_PACKAGE
  template = (
      "In the OpenMoneyBox app, create a new expense category named"
      " '{category_name}'."
  )


class ExpenseViewMonthlyTotalForOpenMoneyBox(_ExpenseViewMonthlyTotalBase):
  app_names = (_OPENMONEYBOX_PACKAGE,)
  package_name = _OPENMONEYBOX_PACKAGE
  template = (
      "In the OpenMoneyBox app, navigate to the view that shows the total"
      " expenses for {month}."
  )


# -----------------------------------------------------------------------------
# My Expenses
# -----------------------------------------------------------------------------


class ExpenseAddSingleForMyExpenses(_ExpenseAddSingleBase):
  app_names = (_MYEXPENSES_PACKAGE,)
  package_name = _MYEXPENSES_PACKAGE
  template = (
      "In the My Expenses app, add a new expense for {amount} with the note"
      " '{note}' under the category '{category}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_myexpenses_task(self, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if _myexpenses_row_exists(
            env,
            self._params["note"],
            self._params["amount"],
        )
        else 0.0
    )


class ExpenseAddMultipleForMyExpenses(_ExpenseAddMultipleBase):
  app_names = (_MYEXPENSES_PACKAGE,)
  package_name = _MYEXPENSES_PACKAGE
  template = (
      "In the My Expenses app, add the following expenses one by one:"
      " {expenses}. Each entry should use its own note, amount, and category."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_myexpenses_task(self, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if all(
            _myexpenses_row_exists(env, expense["note"], expense["amount"])
            for expense in self._params["expenses"]
        )
        else 0.0
    )


class ExpenseDeleteSingleForMyExpenses(_ExpenseDeleteSingleBase):
  app_names = (_MYEXPENSES_PACKAGE,)
  package_name = _MYEXPENSES_PACKAGE
  template = (
      "In the My Expenses app, delete the expense whose note is"
      " '{note_to_delete}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_myexpenses_task(
        self,
        env,
        [
            {
                "note": self._params["note_to_delete"],
                "amount": 12.0,
                "category": "Food",
            },
            {"note": "Keep this expense", "amount": 4.5, "category": "Travel"},
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    target_deleted = not _matching_myexpenses_rows(
        env,
        self._params["note_to_delete"],
    )
    noise_preserved = bool(_matching_myexpenses_rows(env, "Keep this expense"))
    return 1.0 if target_deleted and noise_preserved else 0.0


class ExpenseDeleteMultipleForMyExpenses(_ExpenseDeleteMultipleBase):
  app_names = (_MYEXPENSES_PACKAGE,)
  package_name = _MYEXPENSES_PACKAGE
  template = (
      "In the My Expenses app, delete every expense whose note is in this"
      " list: {notes_to_delete}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    seed_expenses = [
        {"note": note, "amount": 12.0 + index, "category": "Food"}
        for index, note in enumerate(self._params["notes_to_delete"])
    ]
    seed_expenses.append(
        {"note": "Keep this expense", "amount": 4.5, "category": "Travel"}
    )
    _initialize_myexpenses_task(self, env, seed_expenses)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    deleted = all(
        not _matching_myexpenses_rows(env, note)
        for note in self._params["notes_to_delete"]
    )
    noise_preserved = bool(_matching_myexpenses_rows(env, "Keep this expense"))
    return 1.0 if deleted and noise_preserved else 0.0


class ExpenseDeleteDuplicatesForMyExpenses(_ExpenseDeleteDuplicatesBase):
  app_names = (_MYEXPENSES_PACKAGE,)
  package_name = _MYEXPENSES_PACKAGE
  template = (
      "In the My Expenses app, find duplicate expenses with the note"
      " '{note_to_dedupe}' and delete the duplicates so only one remains."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    note = self._params["note_to_dedupe"]
    _initialize_myexpenses_task(
        self,
        env,
        [
            {"note": note, "amount": 12.0, "category": "Food"},
            {"note": note, "amount": 12.0, "category": "Food"},
            {"note": "Keep this expense", "amount": 4.5, "category": "Travel"},
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    target_count = len(
        _matching_myexpenses_rows(env, self._params["note_to_dedupe"])
    )
    noise_preserved = bool(_matching_myexpenses_rows(env, "Keep this expense"))
    return 1.0 if target_count == 1 and noise_preserved else 0.0


class ExpenseAddMultipleFromGalleryForMyExpenses(
    _ExpenseAddMultipleFromGalleryBase
):
  app_names = (_MYEXPENSES_PACKAGE, "simple gallery pro")
  package_name = _MYEXPENSES_PACKAGE
  template = (
      "In the My Expenses app, add the following expenses sourced from a"
      " receipt image in the gallery: {expenses}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_myexpenses_task(self, env)
    _write_expenses_to_gallery(env, self._params["expenses"])
    adb_utils.launch_app(_MYEXPENSES_PACKAGE, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if all(
            _myexpenses_row_exists(env, expense["note"], expense["amount"])
            for expense in self._params["expenses"]
        )
        else 0.0
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    user_data_generation.clear_device_storage(env)


class ExpenseAddMultipleFromMarkorForMyExpenses(
    _ExpenseAddMultipleFromMarkorBase
):
  app_names = (_MYEXPENSES_PACKAGE, "markor")
  package_name = _MYEXPENSES_PACKAGE
  template = (
      "In the My Expenses app, add the following expenses sourced from a"
      " Markor note: {expenses}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_myexpenses_task(self, env)
    _write_expenses_to_markor(env, self._params["expenses"])
    adb_utils.launch_app(_MYEXPENSES_PACKAGE, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if all(
            _myexpenses_row_exists(env, expense["note"], expense["amount"])
            for expense in self._params["expenses"]
        )
        else 0.0
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)


class ExpenseEditExpenseForMyExpenses(_ExpenseEditExpenseBase):
  app_names = (_MYEXPENSES_PACKAGE,)
  package_name = _MYEXPENSES_PACKAGE
  template = (
      "In the My Expenses app, edit the expense titled '{old_note}' so its"
      " note becomes '{new_note}' and its amount becomes {new_amount}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_myexpenses_task(
        self,
        env,
        [
            {
                "note": self._params["old_note"],
                "amount": 12.0,
                "category": "Food",
            }
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    old_gone = not _matching_myexpenses_rows(env, self._params["old_note"])
    new_exists = _myexpenses_row_exists(
        env,
        self._params["new_note"],
        self._params["new_amount"],
    )
    return 1.0 if old_gone and new_exists else 0.0


class ExpenseAddCategoryForMyExpenses(_ExpenseAddCategoryBase):
  app_names = (_MYEXPENSES_PACKAGE,)
  package_name = _MYEXPENSES_PACKAGE
  template = (
      "In the My Expenses app, create a new expense category named"
      " '{category_name}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_myexpenses_task(self, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if self._params["category_name"].strip()
        in _list_myexpenses_categories(env)
        else 0.0
    )


class ExpenseViewMonthlyTotalForMyExpenses(_ExpenseViewMonthlyTotalBase):
  app_names = (_MYEXPENSES_PACKAGE,)
  package_name = _MYEXPENSES_PACKAGE
  template = (
      "In the My Expenses app, navigate to the view that shows the total"
      " expenses for {month}."
  )


# -----------------------------------------------------------------------------
# Finance Manager
# -----------------------------------------------------------------------------


class ExpenseAddSingleForFinanceManager(_ExpenseAddSingleBase):
  app_names = (_FINANCEMANAGER_PACKAGE,)
  package_name = _FINANCEMANAGER_PACKAGE
  template = (
      "In the Finance Manager app, add a new expense for {amount} with the"
      " note '{note}' under the category '{category}'."
  )


class ExpenseAddMultipleForFinanceManager(_ExpenseAddMultipleBase):
  app_names = (_FINANCEMANAGER_PACKAGE,)
  package_name = _FINANCEMANAGER_PACKAGE
  template = (
      "In the Finance Manager app, add the following expenses one by one:"
      " {expenses}. Each entry should use its own note, amount, and category."
  )


class ExpenseDeleteSingleForFinanceManager(_ExpenseDeleteSingleBase):
  app_names = (_FINANCEMANAGER_PACKAGE,)
  package_name = _FINANCEMANAGER_PACKAGE
  template = (
      "In the Finance Manager app, delete the expense whose note is"
      " '{note_to_delete}'."
  )


class ExpenseDeleteMultipleForFinanceManager(_ExpenseDeleteMultipleBase):
  app_names = (_FINANCEMANAGER_PACKAGE,)
  package_name = _FINANCEMANAGER_PACKAGE
  template = (
      "In the Finance Manager app, delete every expense whose note is in this"
      " list: {notes_to_delete}."
  )


class ExpenseDeleteDuplicatesForFinanceManager(_ExpenseDeleteDuplicatesBase):
  app_names = (_FINANCEMANAGER_PACKAGE,)
  package_name = _FINANCEMANAGER_PACKAGE
  template = (
      "In the Finance Manager app, find duplicate expenses with the note"
      " '{note_to_dedupe}' and delete the duplicates so only one remains."
  )


class ExpenseAddMultipleFromGalleryForFinanceManager(
    _ExpenseAddMultipleFromGalleryBase
):
  app_names = (_FINANCEMANAGER_PACKAGE,)
  package_name = _FINANCEMANAGER_PACKAGE
  template = (
      "In the Finance Manager app, add the following expenses sourced from a"
      " receipt image in the gallery: {expenses}."
  )


class ExpenseAddMultipleFromMarkorForFinanceManager(
    _ExpenseAddMultipleFromMarkorBase
):
  app_names = (_FINANCEMANAGER_PACKAGE,)
  package_name = _FINANCEMANAGER_PACKAGE
  template = (
      "In the Finance Manager app, add the following expenses sourced from a"
      " Markor note: {expenses}."
  )


class ExpenseEditExpenseForFinanceManager(_ExpenseEditExpenseBase):
  app_names = (_FINANCEMANAGER_PACKAGE,)
  package_name = _FINANCEMANAGER_PACKAGE
  template = (
      "In the Finance Manager app, edit the expense titled '{old_note}' so its"
      " note becomes '{new_note}' and its amount becomes {new_amount}."
  )


class ExpenseAddCategoryForFinanceManager(_ExpenseAddCategoryBase):
  app_names = (_FINANCEMANAGER_PACKAGE,)
  package_name = _FINANCEMANAGER_PACKAGE
  template = (
      "In the Finance Manager app, create a new expense category named"
      " '{category_name}'."
  )


class ExpenseViewMonthlyTotalForFinanceManager(_ExpenseViewMonthlyTotalBase):
  app_names = (_FINANCEMANAGER_PACKAGE,)
  package_name = _FINANCEMANAGER_PACKAGE
  template = (
      "In the Finance Manager app, navigate to the view that shows the total"
      " expenses for {month}."
  )


# -----------------------------------------------------------------------------
# Sushi
# -----------------------------------------------------------------------------


class ExpenseAddSingleForSushi(_ExpenseAddSingleBase):
  app_names = (_SUSHI_PACKAGE,)
  package_name = _SUSHI_PACKAGE
  template = (
      "In the Sushi app, add a new expense for {amount} with the note '{note}'"
      " under the category '{category}'."
  )


class ExpenseAddMultipleForSushi(_ExpenseAddMultipleBase):
  app_names = (_SUSHI_PACKAGE,)
  package_name = _SUSHI_PACKAGE
  template = (
      "In the Sushi app, add the following expenses one by one: {expenses}."
      " Each entry should use its own note, amount, and category."
  )


class ExpenseDeleteSingleForSushi(_ExpenseDeleteSingleBase):
  app_names = (_SUSHI_PACKAGE,)
  package_name = _SUSHI_PACKAGE
  template = (
      "In the Sushi app, delete the expense whose note is '{note_to_delete}'."
  )


class ExpenseDeleteMultipleForSushi(_ExpenseDeleteMultipleBase):
  app_names = (_SUSHI_PACKAGE,)
  package_name = _SUSHI_PACKAGE
  template = (
      "In the Sushi app, delete every expense whose note is in this list:"
      " {notes_to_delete}."
  )


class ExpenseDeleteDuplicatesForSushi(_ExpenseDeleteDuplicatesBase):
  app_names = (_SUSHI_PACKAGE,)
  package_name = _SUSHI_PACKAGE
  template = (
      "In the Sushi app, find duplicate expenses with the note"
      " '{note_to_dedupe}' and delete the duplicates so only one remains."
  )


class ExpenseAddMultipleFromGalleryForSushi(_ExpenseAddMultipleFromGalleryBase):
  app_names = (_SUSHI_PACKAGE,)
  package_name = _SUSHI_PACKAGE
  template = (
      "In the Sushi app, add the following expenses sourced from a receipt"
      " image in the gallery: {expenses}."
  )


class ExpenseAddMultipleFromMarkorForSushi(_ExpenseAddMultipleFromMarkorBase):
  app_names = (_SUSHI_PACKAGE,)
  package_name = _SUSHI_PACKAGE
  template = (
      "In the Sushi app, add the following expenses sourced from a Markor"
      " note: {expenses}."
  )


class ExpenseEditExpenseForSushi(_ExpenseEditExpenseBase):
  app_names = (_SUSHI_PACKAGE,)
  package_name = _SUSHI_PACKAGE
  template = (
      "In the Sushi app, edit the expense titled '{old_note}' so its note"
      " becomes '{new_note}' and its amount becomes {new_amount}."
  )


class ExpenseAddCategoryForSushi(_ExpenseAddCategoryBase):
  app_names = (_SUSHI_PACKAGE,)
  package_name = _SUSHI_PACKAGE
  template = (
      "In the Sushi app, create a new expense category named"
      " '{category_name}'."
  )


class ExpenseViewMonthlyTotalForSushi(_ExpenseViewMonthlyTotalBase):
  app_names = (_SUSHI_PACKAGE,)
  package_name = _SUSHI_PACKAGE
  template = (
      "In the Sushi app, navigate to the view that shows the total expenses"
      " for {month}."
  )


# -----------------------------------------------------------------------------
# Pro Expense
# -----------------------------------------------------------------------------


class ExpenseAddSingleForProExpense(_ExpenseAddSingleBase):
  app_names = (_PROEXPENSE_PACKAGE,)
  package_name = _PROEXPENSE_PACKAGE
  template = (
      "In the Pro Expense app, add a new expense for {amount} with the note"
      " '{note}' under the category '{category}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_proexpense_task(self, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if _proexpense_row_exists(
            env,
            self._params["note"],
            self._params["amount"],
        )
        else 0.0
    )


class ExpenseAddMultipleForProExpense(_ExpenseAddMultipleBase):
  app_names = (_PROEXPENSE_PACKAGE,)
  package_name = _PROEXPENSE_PACKAGE
  template = (
      "In the Pro Expense app, add the following expenses one by one:"
      " {expenses}. Each entry should use its own note, amount, and category."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_proexpense_task(self, env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if all(
            _proexpense_row_exists(env, expense["note"], expense["amount"])
            for expense in self._params["expenses"]
        )
        else 0.0
    )


class ExpenseDeleteSingleForProExpense(_ExpenseDeleteSingleBase):
  app_names = (_PROEXPENSE_PACKAGE,)
  package_name = _PROEXPENSE_PACKAGE
  template = (
      "In the Pro Expense app, delete the expense whose note is"
      " '{note_to_delete}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_proexpense_task(
        self,
        env,
        [
            {
                "note": self._params["note_to_delete"],
                "amount": 12.0,
                "category": "Food",
            },
            {"note": "Keep this expense", "amount": 4.5, "category": "Food"},
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    target_deleted = not _matching_proexpense_rows(
        env,
        self._params["note_to_delete"],
    )
    noise_preserved = bool(_matching_proexpense_rows(env, "Keep this expense"))
    return 1.0 if target_deleted and noise_preserved else 0.0


class ExpenseDeleteMultipleForProExpense(_ExpenseDeleteMultipleBase):
  app_names = (_PROEXPENSE_PACKAGE,)
  package_name = _PROEXPENSE_PACKAGE
  template = (
      "In the Pro Expense app, delete every expense whose note is in this"
      " list: {notes_to_delete}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    seed_expenses = [
        {"note": note, "amount": 12.0 + index, "category": "Food"}
        for index, note in enumerate(self._params["notes_to_delete"])
    ]
    seed_expenses.append(
        {"note": "Keep this expense", "amount": 4.5, "category": "Food"}
    )
    _initialize_proexpense_task(self, env, seed_expenses)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    deleted = all(
        not _matching_proexpense_rows(env, note)
        for note in self._params["notes_to_delete"]
    )
    noise_preserved = bool(_matching_proexpense_rows(env, "Keep this expense"))
    return 1.0 if deleted and noise_preserved else 0.0


class ExpenseDeleteDuplicatesForProExpense(_ExpenseDeleteDuplicatesBase):
  app_names = (_PROEXPENSE_PACKAGE,)
  package_name = _PROEXPENSE_PACKAGE
  template = (
      "In the Pro Expense app, find duplicate expenses with the note"
      " '{note_to_dedupe}' and delete the duplicates so only one remains."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    note = self._params["note_to_dedupe"]
    _initialize_proexpense_task(
        self,
        env,
        [
            {"note": note, "amount": 12.0, "category": "Food"},
            {"note": note, "amount": 12.0, "category": "Food"},
            {"note": "Keep this expense", "amount": 4.5, "category": "Food"},
        ],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    target_count = len(
        _matching_proexpense_rows(env, self._params["note_to_dedupe"])
    )
    noise_preserved = bool(_matching_proexpense_rows(env, "Keep this expense"))
    return 1.0 if target_count == 1 and noise_preserved else 0.0


class ExpenseAddMultipleFromGalleryForProExpense(
    _ExpenseAddMultipleFromGalleryBase
):
  app_names = (_PROEXPENSE_PACKAGE, "simple gallery pro")
  package_name = _PROEXPENSE_PACKAGE
  template = (
      "In the Pro Expense app, add the following expenses sourced from a"
      " receipt image in the gallery: {expenses}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_proexpense_task(self, env)
    _write_expenses_to_gallery(env, self._params["expenses"])
    adb_utils.launch_app(_PROEXPENSE_PACKAGE, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if all(
            _proexpense_row_exists(env, expense["note"], expense["amount"])
            for expense in self._params["expenses"]
        )
        else 0.0
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    user_data_generation.clear_device_storage(env)


class ExpenseAddMultipleFromMarkorForProExpense(
    _ExpenseAddMultipleFromMarkorBase
):
  app_names = (_PROEXPENSE_PACKAGE, "markor")
  package_name = _PROEXPENSE_PACKAGE
  template = (
      "In the Pro Expense app, add the following expenses sourced from a"
      " Markor note: {expenses}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    _initialize_proexpense_task(self, env)
    _write_expenses_to_markor(env, self._params["expenses"])
    adb_utils.launch_app(_PROEXPENSE_PACKAGE, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return (
        1.0
        if all(
            _proexpense_row_exists(env, expense["note"], expense["amount"])
            for expense in self._params["expenses"]
        )
        else 0.0
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)


class ExpenseEditExpenseForProExpense(_ExpenseEditExpenseBase):
  app_names = (_PROEXPENSE_PACKAGE,)
  package_name = _PROEXPENSE_PACKAGE
  template = (
      "In the Pro Expense app, edit the expense titled '{old_note}' so its"
      " note becomes '{new_note}' and its amount becomes {new_amount}."
  )


class ExpenseAddCategoryForProExpense(_ExpenseAddCategoryBase):
  app_names = (_PROEXPENSE_PACKAGE,)
  package_name = _PROEXPENSE_PACKAGE
  template = (
      "In the Pro Expense app, create a new expense category named"
      " '{category_name}'."
  )


class ExpenseViewMonthlyTotalForProExpense(_ExpenseViewMonthlyTotalBase):
  app_names = (_PROEXPENSE_PACKAGE,)
  package_name = _PROEXPENSE_PACKAGE
  template = (
      "In the Pro Expense app, navigate to the view that shows the total"
      " expenses for {month}."
  )


def _make_finance_table1_task(
    class_name: str,
    base_class: type[base.PackageAppEval],
    package_name: str,
    template: str,
    app_names: tuple[str, ...] | None = None,
) -> type[base.PackageAppEval]:
  return type(
      class_name,
      (base_class,),
      {
          "__module__": __name__,
          "app_names": app_names or (package_name,),
          "package_name": package_name,
          "template": template,
      },
  )


_FINANCE_TABLE1_MISSING_SPECS: Final[
    tuple[tuple[str, type[base.PackageAppEval], str], ...]
] = (
    (
        "AddIncome",
        _ExpenseAddIncomeBase,
        "In the {app} app, add an income transaction titled '{note}' for "
        "{amount} in category {category}.",
    ),
    (
        "AttachReceipt",
        _ExpenseAttachReceiptBase,
        "In the {app} app, add these expenses from a receipt image and attach "
        "or use the receipt as evidence: {expenses}.",
    ),
    (
        "CategorySummary",
        _ExpenseCategorySummaryBase,
        "In the {app} app, open the spending summary for category "
        "'{category_name}'.",
    ),
    (
        "DateRangeTotal",
        _ExpenseDateRangeTotalBase,
        "In the {app} app, show the total spending for the date range covering "
        "{month}.",
    ),
    (
        "TransferBetweenWallets",
        _ExpenseTransferBetweenWalletsBase,
        "In the {app} app, start a transfer of {amount} between two wallets or "
        "accounts and stop before final confirmation.",
    ),
)

_FINANCE_TABLE1_APPS: Final[tuple[tuple[str, str, str], ...]] = (
    ("Oinkoin", _OINKOIN_PACKAGE, "Oinkoin"),
    ("OpenMoneyBox", _OPENMONEYBOX_PACKAGE, "OpenMoneyBox"),
    ("MyExpenses", _MYEXPENSES_PACKAGE, "My Expenses"),
    ("FinanceManager", _FINANCEMANAGER_PACKAGE, "Finance Manager"),
    ("Sushi", _SUSHI_PACKAGE, "Sushi"),
    ("ProExpense", _PROEXPENSE_PACKAGE, "Pro Expense"),
)

for _suffix, _package, _display_name in _FINANCE_TABLE1_APPS:
  for _task_name, _base_class, _template in _FINANCE_TABLE1_MISSING_SPECS:
    _app_names = (
        (_package, "simple gallery pro")
        if _task_name == "AttachReceipt"
        else None
    )
    globals()[f"Expense{_task_name}For{_suffix}"] = _make_finance_table1_task(
        f"Expense{_task_name}For{_suffix}",
        _base_class,
        _package,
        _template.replace("{app}", _display_name),
        _app_names,
    )
