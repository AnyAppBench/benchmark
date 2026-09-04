"""Cross-app contacts task ports for the app-generalization suite.

Contacts are durable Android data. Google Contacts and Right Contact write
through the platform ``ContactsProvider`` in the pinned clean image. Fossify
Contacts, Simple Contacts Pro SE, and Connect You store canonical rows in
app-private SQLite databases, so the validators query those databases directly.

Tasks in this module:

  * ``ContactsAddContact`` -- create + save a new contact.
  * ``ContactsEditContact`` -- edit an existing/newly-created contact.
  * ``ContactsSearchContact`` -- search for and open a target contact.
  * ``ContactsViewContactDetails`` -- open a contact detail page.
  * ``ContactsAddFavoriteContact`` -- create a contact and mark it as a
    starred / favorite entry.
  * ``ContactsRemoveFavoriteContact`` -- remove favorite/star status.
  * ``ContactsDeleteContact`` -- delete a contact.
  * ``ContactsShareContact`` -- open the share sheet for a contact.
  * ``ContactsCallContact`` -- start a call to a contact.
  * ``ContactsMessageContact`` -- open an SMS compose flow to a contact.
"""

from __future__ import annotations

import random
import re
import shlex
import time
from typing import Any, Final
from xml.etree import ElementTree

from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base as base,
)
from android_world.utils import contacts_utils


_FIRST_NAMES: Final[tuple[str, ...]] = (
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Eva",
    "Frank",
    "Grace",
    "Hannah",
    "Ivan",
    "Jack",
)

_LAST_NAMES: Final[tuple[str, ...]] = (
    "Johnson",
    "Smith",
    "Brown",
    "Taylor",
    "Adams",
    "Wilson",
    "Lee",
    "White",
    "Harris",
    "Clark",
)

_PHONE_LABELS: Final[tuple[str, ...]] = ("Home", "Work")

_COMPANIES: Final[tuple[str, ...]] = (
    "Acme Corp",
    "Globex",
    "Initech",
    "Umbrella",
    "Hooli",
    "Soylent",
)

_STREETS: Final[tuple[str, ...]] = (
    "123 Maple Street",
    "45 Oak Avenue",
    "9 Pine Road",
    "77 Cedar Lane",
    "301 Birch Blvd",
)

# These builds do not use ContactsProvider as the source of truth in this image,
# so provider seeding would create rows the target app may not show. They use
# create-first prompts plus app SQLite checks where a stable DB exists.
_UI_FALLBACK_PACKAGES: Final[frozenset[str]] = frozenset((
    "com.bnyro.contacts",
    "com.simplemobiletools.contacts.pro",
    "org.fossify.contacts",
))
_LOCAL_SQLITE_PACKAGES: Final[frozenset[str]] = frozenset((
    "com.bnyro.contacts",
    "com.simplemobiletools.contacts.pro",
    "org.fossify.contacts",
))
_SIMPLE_LOCAL_DB_PACKAGES: Final[frozenset[str]] = frozenset((
    "com.simplemobiletools.contacts.pro",
    "org.fossify.contacts",
))
_CONNECT_YOU_PACKAGE_NAME: Final[str] = "com.bnyro.contacts"
_RIGHT_CONTACTS_PACKAGE: Final[str] = "com.goodwy.contacts"
_RIGHT_CONTACTS_PREFS: Final[str] = (
    "/data/data/com.goodwy.contacts/shared_prefs/Prefs.xml"
)
_RIGHT_CONTACTS_PREFS_ABSENT: Final[str] = (
    "__CATBENCH_RIGHT_CONTACT_PREFS_ABSENT__"
)
_LOCAL_CONTACTS_DB: Final[dict[str, str]] = {
    "org.fossify.contacts": (
        "/data/data/org.fossify.contacts/databases/local_contacts.db"
    ),
    "com.simplemobiletools.contacts.pro": (
        "/data/data/com.simplemobiletools.contacts.pro/databases/local_contacts.db"
    ),
    "com.bnyro.contacts": (
        "/data/data/com.bnyro.contacts/databases/com.bnyro.contacts"
    ),
}


def _random_phone() -> str:
  return (
      f"{random.randint(100, 999)}-{random.randint(100, 999)}"
      f"-{random.randint(1000, 9999)}"
  )


def _random_email(first: str, last: str) -> str:
  domain = random.choice(("example.com", "mail.test", "contoso.com"))
  return f"{first.lower()}.{last.lower()}@{domain}"


def _generate_add_contact_params() -> dict[str, str]:
  first = random.choice(_FIRST_NAMES)
  last = random.choice(_LAST_NAMES)
  return {
      "first": first,
      "last": last,
      "name": f"{first} {last}",
      "number": _random_phone(),
  }


def _generate_draft_params() -> dict[str, str]:
  return {
      "first": random.choice(_FIRST_NAMES),
      "last": random.choice(_LAST_NAMES),
      "phone": _random_phone(),
      "phone_label": random.choice(_PHONE_LABELS),
  }


def _generate_add_email_params() -> dict[str, str]:
  first = random.choice(_FIRST_NAMES)
  last = random.choice(_LAST_NAMES)
  return {
      "first": first,
      "last": last,
      "name": f"{first} {last}",
      "number": _random_phone(),
      "email": _random_email(first, last),
  }


def _generate_add_company_params() -> dict[str, str]:
  first = random.choice(_FIRST_NAMES)
  last = random.choice(_LAST_NAMES)
  return {
      "first": first,
      "last": last,
      "name": f"{first} {last}",
      "number": _random_phone(),
      "company": random.choice(_COMPANIES),
  }


def _generate_add_address_params() -> dict[str, str]:
  first = random.choice(_FIRST_NAMES)
  last = random.choice(_LAST_NAMES)
  return {
      "first": first,
      "last": last,
      "name": f"{first} {last}",
      "number": _random_phone(),
      "address": random.choice(_STREETS),
  }


def _generate_add_two_contacts_params() -> dict[str, str]:
  first_a, first_b = random.sample(_FIRST_NAMES, 2)
  last_a, last_b = random.sample(_LAST_NAMES, 2)
  return {
      "name_a": f"{first_a} {last_a}",
      "number_a": _random_phone(),
      "name_b": f"{first_b} {last_b}",
      "number_b": _random_phone(),
  }


def _normalize_phone(value: str) -> str:
  return "".join(ch for ch in value if ch.isdigit())


def _phone_numbers_equivalent(first: str, second: str) -> bool:
  """Match one frozen US-style number with an optional leading country code.

  ContactsProvider may persist a generated ten-digit CATBench number with a
  leading North-American country code.  Accept exactly that normalization,
  while avoiding suffix matching that would also accept a different, longer
  number.
  """
  first_digits = _normalize_phone(first)
  second_digits = _normalize_phone(second)
  if not first_digits or not second_digits:
    return False
  if first_digits == second_digits:
    return True
  if len(first_digits) == 10 and second_digits == "1" + first_digits:
    return True
  if len(second_digits) == 10 and first_digits == "1" + second_digits:
    return True
  return False


def _phone_matches(
    ui_elements, phone_digits: str  # pylint: disable=missing-type-doc
) -> bool:
  return any(
      _phone_numbers_equivalent(el.text or "", phone_digits)
      or _phone_numbers_equivalent(el.content_description or "", phone_digits)
      for el in ui_elements
  )


def _is_editable_element(element: Any) -> bool:
  class_name = (getattr(element, "class_name", None) or "").casefold()
  return bool(getattr(element, "is_editable", False)) or class_name.endswith(
      "edittext"
  )


def _editable_text_matches(ui_elements, expected: str) -> bool:
  expected = expected.strip().casefold()
  return any(
      _is_editable_element(element)
      and (element.text or "").strip().casefold() == expected
      for element in ui_elements
  )


def _editable_phone_matches(ui_elements, expected_digits: str) -> bool:
  return any(
      _is_editable_element(element)
      and _phone_numbers_equivalent(element.text or "", expected_digits)
      for element in ui_elements
  )


def _destructive_transition_observation(
    *,
    target_state_active: bool,
    seen_target_state: bool,
) -> tuple[bool, bool]:
  """Latch an exact target state before accepting its later absence.

  The suite invokes ``is_successful`` after every primitive action.  This lets
  destructive create-then-remove workflows distinguish their initial absent
  state from a completed transition without relying on a transient snackbar.
  """
  if target_state_active:
    return False, True
  return seen_target_state, seen_target_state


def _favorite_state_visible(ui_elements) -> bool:
  """Strong UI evidence that the currently visible contact is starred."""
  return base.element_text_contains(
      ui_elements,
      (
          "starred",
          "remove from favorites",
          "remove from favourites",
          "remove favorite",
          "remove favourite",
      ),
  )


def _unstarred_state_visible(ui_elements) -> bool:
  """Strong UI evidence that the currently visible contact is unstarred."""
  return base.element_text_contains(
      ui_elements,
      (
          "add to favorites",
          "add to favourites",
          "add favorite",
          "add favourite",
      ),
  )


def _favorite_removal_confirmed(ui_elements) -> bool:
  """Require transition evidence, not the initial unstarred final state."""
  return base.element_text_contains(
      ui_elements,
      (
          "removed from favorites",
          "removed from favourites",
          "favorite removed",
          "favourite removed",
          "unstarred",
      ),
  )


def _contact_deletion_confirmed(ui_elements) -> bool:
  return base.element_text_contains(
      ui_elements, ("contact deleted", "deleted contact")
  ) or base.element_text_contains_word(ui_elements, ("deleted", "undo"))


def _external_call_surface(
    ui_elements,
    contacts_package: str,
) -> bool:
  """Require an active-call marker rendered outside the Contacts app."""
  source_package = contacts_package.casefold()
  return any(
      (getattr(element, "package_name", None) or "").casefold()
      not in ("", source_package)
      and base.matches_any_word(
          element,
          ("calling", "dialing", "speaker", "mute", "hold", "end call"),
      )
      for element in ui_elements
  )


def _external_message_composer(
    ui_elements,
    contacts_package: str,
) -> bool:
  """Require the message-body editor in an app outside Contacts."""
  source_package = contacts_package.casefold()
  markers = (
      "text message",
      "type a message",
      "write a message",
      "message text",
  )
  for element in ui_elements:
    package_name = (
        getattr(element, "package_name", None) or ""
    ).casefold()
    if package_name in ("", source_package) or not _is_editable_element(
        element
    ):
      continue
    fields = (
        element.text or "",
        element.content_description or "",
        getattr(element, "hint_text", None) or "",
    )
    if any(marker in field.casefold() for marker in markers for field in fields):
      return True
  return False


def _adb_shell(env: interface.AsyncEnv, cmd: str) -> str:
  out = adb_utils.issue_generic_request(["shell", cmd], env.controller)
  return out.generic.output.decode("utf-8", errors="ignore") if out else ""


_VERIFIER_READ_RC_MARKER: Final[str] = "__CATBENCH_VERIFIER_READ_RC__"


def _adb_shell_read(env: interface.AsyncEnv, cmd: str) -> str:
  """Run a native-state read and distinguish absence from command failure."""
  wrapped = (
      f"{cmd}; _catbench_read_rc=$?; printf "
      f"'\\n{_VERIFIER_READ_RC_MARKER}%s\\n' \"$_catbench_read_rc\""
  )
  try:
    out = _adb_shell(env, wrapped)
  except Exception as exc:  # pylint: disable=broad-except
    raise base.VerifierStateReadError(
        "Contacts verifier native-state command raised an exception."
    ) from exc
  marker = re.search(
      rf"(?:^|\n){re.escape(_VERIFIER_READ_RC_MARKER)}(\d+)\r?\n?$",
      out,
  )
  if marker is None:
    raise base.VerifierStateReadError(
        "Contacts verifier native-state command returned no status marker."
    )
  if int(marker.group(1)) != 0:
    raise base.VerifierStateReadError(
        f"Contacts verifier native-state command failed with exit status "
        f"{marker.group(1)}."
    )
  return out[:marker.start()].rstrip("\r\n")


def _sqlite_exec(env: interface.AsyncEnv, db_path: str, sql: str) -> str:
  """Reads rows from an app-local database.

  These apps create their local database lazily, on the first contact they
  store, so an absent file is a legitimate empty state and must be reported as
  zero rows. Only a database that exists but cannot be read is a state-read
  failure: ``sqlite3`` exits non-zero in both cases, which would otherwise
  turn every "agent stored nothing" episode into an infrastructure error and
  drop it from the denominator.
  """
  quoted_db = shlex.quote(db_path)
  return _adb_shell_read(
      env,
      "su 0 sh -c "
      + shlex.quote(
          f"if [ ! -f {quoted_db} ]; then exit 0; fi; "
          f"sqlite3 {quoted_db} {shlex.quote(sql)} 2>/dev/null"
      ),
  )


def _assert_right_contact_provider_contract(env: interface.AsyncEnv) -> None:
  """Fail closed unless Right Contact will use public phone storage.

  Right Contact is dual-source: missing Contacts permissions or a persisted
  ``smt_private`` preference redirects new contacts into its private Room DB.
  The frozen CATBench adapter intentionally uses the clean-image default
  (empty source, Android phone storage), so silently falling onto the private
  path would invalidate every provider-backed verifier for this app.
  """
  permission_output = _adb_shell_read(
      env,
      "dumpsys package com.goodwy.contacts | grep -E "
      + shlex.quote(
          r"android\.permission\.(READ_CONTACTS|WRITE_CONTACTS): granted="
      ),
  )
  grants: dict[str, str] = {}
  for line in permission_output.splitlines():
    match = re.search(
        r"android\.permission\.(READ_CONTACTS|WRITE_CONTACTS): "
        r"granted=(true|false)\b",
        line,
    )
    if match is None:
      raise base.VerifierStateReadError(
          "Right Contact permission attestation returned an unparseable row."
      )
    prior = grants.setdefault(match.group(1), match.group(2))
    if prior != match.group(2):
      raise base.VerifierStateReadError(
          "Right Contact permission attestation returned conflicting rows."
      )
  if grants != {"READ_CONTACTS": "true", "WRITE_CONTACTS": "true"}:
    raise base.VerifierStateReadError(
        "Right Contact requires granted READ_CONTACTS and WRITE_CONTACTS "
        f"for its frozen ContactsProvider adapter; observed {grants}."
    )

  prefs_output = _adb_shell_read(
      env,
      "su 0 sh -c "
      + shlex.quote(
          f"if [ -f {shlex.quote(_RIGHT_CONTACTS_PREFS)} ]; then "
          f"cat {shlex.quote(_RIGHT_CONTACTS_PREFS)}; else "
          f"printf '{_RIGHT_CONTACTS_PREFS_ABSENT}\\n'; fi"
      ),
  )
  if prefs_output.strip() == _RIGHT_CONTACTS_PREFS_ABSENT:
    return
  try:
    root = ElementTree.fromstring(prefs_output)
  except ElementTree.ParseError as exc:
    raise base.VerifierStateReadError(
        "Right Contact preferences are malformed or only partially readable."
    ) from exc
  source_nodes = [
      node
      for node in root.findall("string")
      if node.attrib.get("name") == "last_used_contact_source"
  ]
  if len(source_nodes) > 1:
    raise base.VerifierStateReadError(
        "Right Contact preferences contain duplicate contact-source entries."
    )
  source = (source_nodes[0].text or "").strip() if source_nodes else ""
  if source:
    raise base.VerifierStateReadError(
        "Right Contact frozen adapter requires Android phone storage "
        f"(empty last_used_contact_source), observed {source!r}."
    )


def _uses_local_sqlite(package_name: str) -> bool:
  return package_name in _LOCAL_SQLITE_PACKAGES


def _local_sqlite_rows(env: interface.AsyncEnv, package_name: str) -> list[dict[str, str]]:
  db_path = _LOCAL_CONTACTS_DB.get(package_name)
  if not db_path:
    return []
  rows: list[dict[str, str]] = []
  if package_name in _SIMPLE_LOCAL_DB_PACKAGES:
    out = _sqlite_exec(
        env,
        db_path,
        (
            "SELECT COALESCE(first_name,'') || char(31) ||"
            " COALESCE(surname,'') || char(31) ||"
            " COALESCE(phone_numbers,'') || char(31) ||"
            " COALESCE(starred,0) FROM contacts;"
        ),
    )
    for line in out.splitlines():
      parts = line.split("\x1f")
      if len(parts) != 4:
        raise base.VerifierStateReadError(
            f"Contacts SQLite query for {package_name} returned an "
            "unparseable row."
        )
      first, last, phones, starred = parts
      rows.append({
          "name": f"{first} {last}".strip(),
          "phones": phones,
          "starred": starred,
      })
  elif package_name == _CONNECT_YOU_PACKAGE_NAME:
    out = _sqlite_exec(
        env,
        db_path,
        (
            "SELECT COALESCE(c.displayName,'') || char(31) ||"
            " COALESCE(c.firstName,'') || char(31) ||"
            " COALESCE(c.surName,'') || char(31) ||"
            " COALESCE(c.favorite,0) || char(31) ||"
            " COALESCE(group_concat(v.value, char(30)),'')"
            " FROM localContacts c"
            " LEFT JOIN valuableTypes v ON v.contactId = c.id"
            " GROUP BY c.id;"
        ),
    )
    for line in out.splitlines():
      parts = line.split("\x1f")
      if len(parts) != 5:
        raise base.VerifierStateReadError(
            f"Contacts SQLite query for {package_name} returned an "
            "unparseable row."
        )
      display, first, last, favorite, values = parts
      rows.append({
          "name": display or f"{first} {last}".strip(),
          "phones": values,
          "starred": favorite,
      })
  return rows


def _local_has_phone(
    env: interface.AsyncEnv,
    package_name: str,
    name: str,
    number: str,
) -> bool:
  expected_name = name.strip().casefold()
  for row in _local_sqlite_rows(env, package_name):
    if row["name"].strip().casefold() != expected_name:
      continue
    if _phone_field_contains_equivalent(row["phones"], number):
      return True
  return False


_PHONEISH_RUN_RE = re.compile(
    r"(?<![+\d])\+?\d(?:[\d .()-]*\d)?(?!\d)"
)


def _phone_field_contains_equivalent(phone_field: str, expected: str) -> bool:
  """Find an exact-equivalent CATBench number in a serialized DB field."""
  return any(
      _phone_numbers_equivalent(match.group(0), expected)
      for match in _PHONEISH_RUN_RE.finditer(phone_field)
  )


def _local_phone_present(
    env: interface.AsyncEnv,
    package_name: str,
    number: str,
) -> bool:
  """Return whether ``number`` occurs in any durable local contact row."""
  return any(
      _phone_field_contains_equivalent(row["phones"], number)
      for row in _local_sqlite_rows(env, package_name)
  )


def _local_name_present(
    env: interface.AsyncEnv,
    package_name: str,
    name: str,
) -> bool:
  expected_name = name.strip().casefold()
  return any(
      row["name"].strip().casefold() == expected_name
      for row in _local_sqlite_rows(env, package_name)
  )


def _local_is_starred(
    env: interface.AsyncEnv,
    package_name: str,
    name: str,
) -> bool:
  expected_name = name.strip().casefold()
  for row in _local_sqlite_rows(env, package_name):
    if row["name"].strip().casefold() == expected_name:
      return row["starred"].strip() == "1"
  return False


def _quote_content_value(value: str) -> str:
  return value.replace("'", "'\\''")


def _provider_data_rows(env: interface.AsyncEnv) -> str:
  out = _adb_shell_read(
      env,
      (
          "content query --uri content://com.android.contacts/data"
          " --projection display_name:data1:mimetype"
      ),
  )
  for row in out.splitlines():
    stripped = row.strip()
    if not stripped or stripped == "No result found.":
      continue
    if not stripped.startswith("Row:") or "display_name=" not in row:
      raise base.VerifierStateReadError(
          "ContactsProvider data query returned an unparseable row."
      )
  return out


def _provider_contact_rows(env: interface.AsyncEnv) -> str:
  out = _adb_shell_read(
      env,
      (
          "content query --uri content://com.android.contacts/contacts"
          " --projection display_name:starred"
      ),
  )
  for row in out.splitlines():
    stripped = row.strip()
    if not stripped or stripped == "No result found.":
      continue
    if not stripped.startswith("Row:") or (
        "display_name=" not in row or "starred=" not in row
    ):
      raise base.VerifierStateReadError(
          "ContactsProvider contacts query returned an unparseable row."
      )
  return out


def _provider_contacts(env: interface.AsyncEnv) -> list[contacts_utils.Contact]:
  try:
    return contacts_utils.list_contacts(env.controller)
  except Exception:  # pylint: disable=broad-except
    return []


_PHONE_ROW_RE = re.compile(r"display_name=([^,]*), data1=(.*)$")


def _provider_phone_pairs(env: interface.AsyncEnv) -> list[tuple[str, str]]:
  """(display_name, phone) rows from the modern ContactsContract provider.

  ``contacts_utils.list_contacts`` queries the legacy ``content://contacts``
  authority, which returns no rows on the API-33 benchmark image — every
  validator built on it could never pass. Read ContactsContract data rows
  directly instead.
  """
  out = _adb_shell_read(
      env,
      (
          "content query --uri content://com.android.contacts/data"
          " --where \"mimetype='vnd.android.cursor.item/phone_v2'\""
          " --projection display_name:data1"
      ),
  )
  pairs: list[tuple[str, str]] = []
  for row in out.splitlines():
    stripped = row.strip()
    if not stripped or stripped == "No result found.":
      continue
    if not stripped.startswith("Row:"):
      raise base.VerifierStateReadError(
          "ContactsProvider phone query returned unexpected output."
      )
    match = _PHONE_ROW_RE.search(row)
    if match is None:
      raise base.VerifierStateReadError(
          "ContactsProvider phone query returned an unparseable row."
      )
    pairs.append((match.group(1).strip(), match.group(2).strip()))
  return pairs


def _provider_has_phone(
    env: interface.AsyncEnv,
    name: str,
    number: str,
) -> bool:
  expected_name = name.strip().casefold()
  for row_name, row_number in _provider_phone_pairs(env):
    if (
        row_name.casefold() == expected_name
        and _phone_numbers_equivalent(row_number, number)
    ):
      return True
  # Legacy-provider fallback, kept for parity with AW's contacts_utils on
  # device builds where the legacy authority still answers.
  expected_number = contacts_utils.clean_phone_number(number)
  return any(
      contact.name.strip().casefold() == expected_name
      and _phone_numbers_equivalent(contact.number, expected_number)
      for contact in _provider_contacts(env)
  )


def _provider_phone_present(env: interface.AsyncEnv, number: str) -> bool:
  """Return whether ``number`` occurs under any provider display name."""
  return any(
      _phone_numbers_equivalent(row_number, number)
      for _, row_number in _provider_phone_pairs(env)
  ) or any(
      _phone_numbers_equivalent(contact.number, number)
      for contact in _provider_contacts(env)
  )


def _provider_has_all_phones(
    env: interface.AsyncEnv,
    name: str,
    numbers: tuple[str, ...],
) -> bool:
  expected_name = name.strip().casefold()
  actual_numbers = [
      row_number
      for row_name, row_number in _provider_phone_pairs(env)
      if row_name.casefold() == expected_name
  ]
  actual_numbers.extend(
      contact.number
      for contact in _provider_contacts(env)
      if contact.name.strip().casefold() == expected_name
  )
  return all(
      any(
          _phone_numbers_equivalent(expected_number, actual_number)
          for actual_number in actual_numbers
      )
      for expected_number in numbers
  )


_DISPLAY_NAME_RE = re.compile(r"display_name=([^,\n]*)")


def _provider_display_names(env: interface.AsyncEnv) -> set[str]:
  """Exact display_name values currently in ContactsProvider (casefolded)."""
  names: set[str] = set()
  for dump in (_provider_contact_rows(env), _provider_data_rows(env)):
    for match in _DISPLAY_NAME_RE.finditer(dump):
      value = match.group(1).strip()
      if value and value.casefold() != "null":
        names.add(value.casefold())
  return names


def _provider_name_present(env: interface.AsyncEnv, name: str) -> bool:
  # Exact display_name match only. A raw substring scan over the provider
  # dump ("Ed" in "Ted Baker") corrupts both the present-checks and the
  # gone-after-delete checks.
  expected_name = name.strip().casefold()
  return expected_name in _provider_display_names(env) or any(
      contact.name.strip().casefold() == expected_name
      for contact in _provider_contacts(env)
  )


def _provider_data_contains(
    env: interface.AsyncEnv,
    name: str,
    token: str,
) -> bool:
  data_rows = _provider_data_rows(env).casefold()
  return (
      name.strip().casefold() in data_rows
      and token.strip().casefold() in data_rows
  )


def _provider_is_starred(env: interface.AsyncEnv, name: str) -> bool:
  rows = _provider_contact_rows(env)
  expected_name = name.strip().casefold()
  for row in rows.splitlines():
    match = _DISPLAY_NAME_RE.search(row)
    if match is None:
      continue
    if match.group(1).strip().casefold() != expected_name:
      continue
    if re.search(r"\bstarred=1\b", row.casefold()):
      return True
  return False


class ContactSeedingError(RuntimeError):
  """Raised when a task's precondition contact could not be seeded.

  Edit/delete/search tasks are meaningless if the dummy contact does not
  exist before the agent runs; raising here makes the runner record the
  episode as an environment exception instead of silently scoring the agent
  against an impossible task.
  """


def _set_provider_starred(
    env: interface.AsyncEnv,
    name: str,
    starred: bool,
) -> None:
  safe_name = _quote_content_value(name)
  value = 1 if starred else 0
  for _ in range(3):
    _adb_shell(
        env,
        (
            "content update --uri content://com.android.contacts/contacts"
            f" --bind starred:i:{value}"
            f" --where \"display_name='{safe_name}'\""
        ),
    )
    if _provider_is_starred(env, name) == starred:
      return
    time.sleep(1.0)
  raise ContactSeedingError(
      f"Failed to set starred={starred} on seeded contact {name!r}."
  )


def _insert_provider_contact(
    env: interface.AsyncEnv,
    *,
    name: str,
    number: str,
) -> None:
  """Seed a system contact through ContactsProvider for mutation/search tasks.

  Verifies the contact actually landed in the provider and retries once;
  raises ``ContactSeedingError`` if the precondition cannot be established.
  """
  for attempt in range(2):
    raw_out = _adb_shell(
        env,
        (
            "content insert --uri content://com.android.contacts/raw_contacts"
            " --bind account_type:s:local"
            " --bind account_name:s:CATBench"
        ),
    )
    match = re.search(r"/raw_contacts/(\d+)", raw_out)
    if not match:
      match = re.search(r"/(\d+)\s*$", raw_out.strip())
    raw_id = match.group(1) if match else ""
    if not raw_id:
      # Some platform builds print nothing for a successful content insert;
      # recover the new row id by querying the newest CATBench-account row.
      query_out = _adb_shell(
          env,
          (
              "content query --uri content://com.android.contacts/raw_contacts"
              " --projection _id --where \"account_name='CATBench'\""
              " --sort \"_id DESC\""
          ),
      )
      ids = re.findall(r"_id=(\d+)", query_out)
      if ids:
        raw_id = ids[0]
    if raw_id:
      safe_name = _quote_content_value(name)
      safe_number = _quote_content_value(number)
      _adb_shell(
          env,
          (
              "content insert --uri content://com.android.contacts/data"
              f" --bind raw_contact_id:i:{raw_id}"
              " --bind mimetype:s:vnd.android.cursor.item/name"
              f" --bind data1:s:'{safe_name}'"
          ),
      )
      _adb_shell(
          env,
          (
              "content insert --uri content://com.android.contacts/data"
              f" --bind raw_contact_id:i:{raw_id}"
              " --bind mimetype:s:vnd.android.cursor.item/phone_v2"
              f" --bind data1:s:'{safe_number}'"
          ),
      )
      # ContactsProvider aggregates raw contacts asynchronously; poll briefly
      # before declaring the seed failed.
      deadline = time.time() + 5.0
      while time.time() < deadline:
        if _provider_name_present(env, name):
          return
        time.sleep(0.5)
    if attempt == 0:
      time.sleep(1.0)
  raise ContactSeedingError(
      f"Failed to seed precondition contact {name!r} ({number}) through"
      " ContactsProvider; aborting episode instead of running the agent"
      " against a missing precondition."
  )


# -----------------------------------------------------------------------------
# Base evaluators.
# -----------------------------------------------------------------------------


class _ContactsTaskBase(base.PackageAppEval):
  """Shared ContactsProvider lifecycle and UI-fallback switch.

  Portable CATBench contracts make the agent create any prerequisite contacts
  named in the instruction.  We therefore clear provider state here but never
  pre-seed a different workflow for provider-backed apps.  This keeps the
  rendered instruction and initial abstract state identical across apps.
  """

  def _uses_provider_validation(self) -> bool:
    return self.package_name not in _UI_FALLBACK_PACKAGES

  def _prepare_provider_state(self, env: interface.AsyncEnv) -> None:
    """Optional hook for provider-backed tasks that need seeded contacts."""

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    if self._uses_provider_validation():
      contacts_utils.clear_contacts(env.controller)
    super().initialize_task(env)
    if self.package_name == _RIGHT_CONTACTS_PACKAGE:
      _assert_right_contact_provider_contract(env)


class _ContactsAddContactBase(_ContactsTaskBase):
  """Base port of ``ContactsAddContact``.

  Provider-backed apps validate through ContactsProvider. App-private builds
  validate through their readable SQLite store. Opaque stores fail closed:
  visible form values alone do not prove that the contact was saved.
  """

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "first": {"type": "string"},
          "last": {"type": "string"},
          "name": {"type": "string"},
          "number": {"type": "string"},
      },
      "required": ["first", "last", "name", "number"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self._uses_provider_validation():
      return (
          1.0
          if _provider_has_phone(
              env, self._params["name"], self._params["number"]
          )
          else 0.0
      )
    if _uses_local_sqlite(self.package_name):
      return (
          1.0
          if _local_has_phone(
              env,
              self.package_name,
              self._params["name"],
              self._params["number"],
          )
          else 0.0
      )
    # Unknown opaque-storage adapters must not be rewarded from editable form
    # values alone, because those values do not prove that Save occurred.
    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_add_contact_params()


class _ContactsNewContactDraftBase(_ContactsTaskBase):
  """Fill in the new-contact form without saving."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "first": {"type": "string"},
          "last": {"type": "string"},
          "phone": {"type": "string"},
          "phone_label": {"type": "string"},
      },
      "required": ["first", "last", "phone", "phone_label"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    phone_digits = _normalize_phone(self._params["phone"])
    # Values must still be in editable form controls. A saved contact detail
    # page can show the same strings but violates the explicit "Do NOT save"
    # contract.
    first_ok = _editable_text_matches(ui, self._params["first"])
    last_ok = _editable_text_matches(ui, self._params["last"])
    label_ok = base.element_text_contains(ui, (self._params["phone_label"],))
    phone_ok = _editable_phone_matches(ui, phone_digits)
    if not (first_ok and last_ok and phone_ok and label_ok):
      return 0.0

    # Editable fields distinguish a detail page from a form, but not a saved
    # contact reopened for editing. Require durable absence of the target.
    name = f"{self._params['first']} {self._params['last']}"
    if self._uses_provider_validation():
      persisted = _provider_name_present(env, name) or _provider_phone_present(
          env, self._params["phone"]
      )
      return 0.0 if persisted else 1.0
    if _uses_local_sqlite(self.package_name):
      persisted = _local_name_present(
          env, self.package_name, name
      ) or _local_phone_present(
          env, self.package_name, self._params["phone"]
      )
      return 0.0 if persisted else 1.0

    # Unknown opaque-storage adapters cannot distinguish an unsaved draft from
    # save-then-reopen-edit and therefore fail closed.
    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_draft_params()


class _ContactsAddContactWithEmailBase(_ContactsTaskBase):
  """Create a contact whose detail view shows both phone and email.

  This legacy private template is not part of the CATBench table profile.
  """

  excluded_packages = ()

  complexity = 1.8
  schema = {
      "type": "object",
      "properties": {
          "first": {"type": "string"},
          "last": {"type": "string"},
          "name": {"type": "string"},
          "number": {"type": "string"},
          "email": {"type": "string"},
      },
      "required": ["first", "last", "name", "number", "email"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self._uses_provider_validation():
      return (
          1.0
          if _provider_has_phone(
              env, self._params["name"], self._params["number"]
          )
          and _provider_data_contains(
              env, self._params["name"], self._params["email"]
          )
          else 0.0
      )
    ui = env.get_state().ui_elements
    name_ok = base.element_text_contains(ui, (self._params["name"],))
    phone_ok = _phone_matches(ui, _normalize_phone(self._params["number"]))
    email_ok = base.element_text_contains(ui, (self._params["email"],))
    return 1.0 if name_ok and phone_ok and email_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_add_email_params()


class _ContactsAddContactWithCompanyBase(_ContactsTaskBase):
  """Create a contact whose detail view shows a company / organization."""

  excluded_packages = ()
  complexity = 1.8
  schema = {
      "type": "object",
      "properties": {
          "first": {"type": "string"},
          "last": {"type": "string"},
          "name": {"type": "string"},
          "number": {"type": "string"},
          "company": {"type": "string"},
      },
      "required": ["first", "last", "name", "number", "company"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self._uses_provider_validation():
      return (
          1.0
          if _provider_has_phone(
              env, self._params["name"], self._params["number"]
          )
          and _provider_data_contains(
              env, self._params["name"], self._params["company"]
          )
          else 0.0
      )
    ui = env.get_state().ui_elements
    name_ok = base.element_text_contains(ui, (self._params["name"],))
    phone_ok = _phone_matches(ui, _normalize_phone(self._params["number"]))
    company_ok = base.element_text_contains(ui, (self._params["company"],))
    return 1.0 if name_ok and phone_ok and company_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_add_company_params()


class _ContactsAddContactWithAddressBase(_ContactsTaskBase):
  """Create a contact whose detail view shows a postal address."""

  excluded_packages = ()
  complexity = 2
  schema = {
      "type": "object",
      "properties": {
          "first": {"type": "string"},
          "last": {"type": "string"},
          "name": {"type": "string"},
          "number": {"type": "string"},
          "address": {"type": "string"},
      },
      "required": ["first", "last", "name", "number", "address"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self._uses_provider_validation():
      street_token = self._params["address"].split(" ", 1)[1]
      return (
          1.0
          if _provider_has_phone(
              env, self._params["name"], self._params["number"]
          )
          and _provider_data_contains(
              env, self._params["name"], street_token
          )
          else 0.0
      )
    ui = env.get_state().ui_elements
    name_ok = base.element_text_contains(ui, (self._params["name"],))
    phone_ok = _phone_matches(ui, _normalize_phone(self._params["number"]))
    # Address fields are often split by line; match on the street number + name.
    street_token = self._params["address"].split(" ", 1)[1]
    addr_ok = base.element_text_contains(ui, (street_token,))
    return 1.0 if name_ok and phone_ok and addr_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_add_address_params()


class _ContactsAddFavoriteContactBase(_ContactsTaskBase):
  """Create a contact and mark it as starred / favorite.

  Provider-backed apps validate the starred bit through ContactsProvider.
  App-private builds with readable SQLite validate both durable predicates.
  Opaque stores fail closed rather than treating generic favorite UI text as
  proof of a saved, starred contact.
  """

  excluded_packages = ()
  complexity = 2.2
  schema = _ContactsAddContactBase.schema

  def _prepare_provider_state(self, env: interface.AsyncEnv) -> None:
    _insert_provider_contact(
        env, name=self._params["name"], number=self._params["number"]
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self._uses_provider_validation():
      return (
          1.0
          if _provider_has_phone(
              env, self._params["name"], self._params["number"]
          )
          and _provider_is_starred(env, self._params["name"])
          else 0.0
      )
    if _uses_local_sqlite(self.package_name):
      return (
          1.0
          if _local_has_phone(
              env,
              self.package_name,
              self._params["name"],
              self._params["number"],
          )
          and _local_is_starred(env, self.package_name, self._params["name"])
          else 0.0
      )
    # UI text cannot establish either persistence or the durable favorite bit
    # for an unknown opaque-storage adapter, so this path must not score.
    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_add_contact_params()


class _ContactsRemoveFavoriteContactBase(_ContactsTaskBase):
  """Remove favorite/star status from an existing or newly-created contact."""

  excluded_packages = ()
  complexity = 2.2
  schema = _ContactsAddContactBase.schema

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    # Episode-local evidence only: the same exact contact must first be
    # observed starred before its unstarred state can succeed.
    self._catbench_seen_target_starred = False

  def _prepare_provider_state(self, env: interface.AsyncEnv) -> None:
    _insert_provider_contact(
        env, name=self._params["name"], number=self._params["number"]
    )
    _set_provider_starred(env, self._params["name"], True)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    seen_target_starred = getattr(
        self, "_catbench_seen_target_starred", False
    )
    if self._uses_provider_validation():
      contact_ok = _provider_has_phone(
          env, self._params["name"], self._params["number"]
      )
      starred = contact_ok and _provider_is_starred(
          env, self._params["name"]
      )
      success, seen_target_starred = _destructive_transition_observation(
          target_state_active=starred,
          seen_target_state=seen_target_starred,
      )
      self._catbench_seen_target_starred = seen_target_starred
      return 1.0 if contact_ok and success else 0.0
    name_ok = base.element_text_contains(ui, (self._params["name"],))
    phone_ok = _phone_matches(ui, _normalize_phone(self._params["number"]))
    if _uses_local_sqlite(self.package_name):
      contact_ok = _local_has_phone(
          env,
          self.package_name,
          self._params["name"],
          self._params["number"],
      )
      starred = _local_is_starred(
          env, self.package_name, self._params["name"]
      )
      success, seen_target_starred = _destructive_transition_observation(
          target_state_active=contact_ok and starred,
          seen_target_state=seen_target_starred,
      )
      self._catbench_seen_target_starred = seen_target_starred
      return 1.0 if contact_ok and success else 0.0

    # Defensive fallback for any future opaque-storage adapter: latch the exact
    # visible target before accepting an explicit unstarred transition.
    target_visible = name_ok and phone_ok
    starred = target_visible and _favorite_state_visible(ui)
    success, seen_target_starred = _destructive_transition_observation(
        target_state_active=starred,
        seen_target_state=seen_target_starred,
    )
    self._catbench_seen_target_starred = seen_target_starred
    unstarred = _unstarred_state_visible(ui) or _favorite_removal_confirmed(ui)
    return 1.0 if target_visible and success and unstarred else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_add_contact_params()


class _ContactsAddTwoContactsBase(_ContactsTaskBase):
  """Create two contacts and confirm both appear in the list."""

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {
          "name_a": {"type": "string"},
          "number_a": {"type": "string"},
          "name_b": {"type": "string"},
          "number_b": {"type": "string"},
      },
      "required": ["name_a", "number_a", "name_b", "number_b"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self._uses_provider_validation():
      a_ok = _provider_has_phone(
          env, self._params["name_a"], self._params["number_a"]
      )
      b_ok = _provider_has_phone(
          env, self._params["name_b"], self._params["number_b"]
      )
      return 1.0 if a_ok and b_ok else 0.0
    ui = env.get_state().ui_elements
    a_ok = base.element_text_contains(ui, (self._params["name_a"],))
    b_ok = base.element_text_contains(ui, (self._params["name_b"],))
    return 1.0 if a_ok and b_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_add_two_contacts_params()


class _ContactsAddThreeContactsBase(_ContactsTaskBase):
  """Create three contacts; all names visible after final save."""

  complexity = 2.8
  schema = {
      "type": "object",
      "properties": {
          "name_a": {"type": "string"}, "number_a": {"type": "string"},
          "name_b": {"type": "string"}, "number_b": {"type": "string"},
          "name_c": {"type": "string"}, "number_c": {"type": "string"},
      },
      "required": [
          "name_a", "number_a", "name_b", "number_b", "name_c", "number_c"
      ],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self._uses_provider_validation():
      ok = all(
          _provider_name_present(env, self._params[key])
          for key in ("name_a", "name_b", "name_c")
      )
      return 1.0 if ok else 0.0
    ui = env.get_state().ui_elements
    return (
        1.0
        if all(
            base.element_text_contains(ui, (self._params[key],))
            for key in ("name_a", "name_b", "name_c")
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    firsts = random.sample(_FIRST_NAMES, 3)
    lasts = random.sample(_LAST_NAMES, 3)
    return {
        "name_a": f"{firsts[0]} {lasts[0]}", "number_a": _random_phone(),
        "name_b": f"{firsts[1]} {lasts[1]}", "number_b": _random_phone(),
        "name_c": f"{firsts[2]} {lasts[2]}", "number_c": _random_phone(),
    }


class _ContactsAddContactWithSecondPhoneBase(_ContactsTaskBase):
  """Create a contact with two distinct phone numbers."""

  complexity = 1.8
  schema = {
      "type": "object",
      "properties": {
          "first": {"type": "string"},
          "last": {"type": "string"},
          "name": {"type": "string"},
          "number": {"type": "string"},
          "number_alt": {"type": "string"},
      },
      "required": ["first", "last", "name", "number", "number_alt"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self._uses_provider_validation():
      return (
          1.0
          if _provider_has_all_phones(
              env,
              self._params["name"],
              (self._params["number"], self._params["number_alt"]),
          )
          else 0.0
      )
    ui = env.get_state().ui_elements
    name_ok = base.element_text_contains(ui, (self._params["name"],))
    a_ok = _phone_matches(ui, _normalize_phone(self._params["number"]))
    b_ok = _phone_matches(ui, _normalize_phone(self._params["number_alt"]))
    return 1.0 if name_ok and a_ok and b_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    base_params = _generate_add_contact_params()
    base_params["number_alt"] = _random_phone()
    return base_params


class _ContactsEditContactBase(_ContactsTaskBase):
  """Create a contact, then edit it: rename surname AND change phone.

  Provider-backed apps validate the new row through ContactsProvider and ensure
  the old name/phone are gone. Readable app-private stores use SQLite; opaque
  stores fail closed.
  """

  excluded_packages = ()
  complexity = 2.6
  schema = {
      "type": "object",
      "properties": {
          "first": {"type": "string"},
          "old_last": {"type": "string"},
          "new_last": {"type": "string"},
          "old_number": {"type": "string"},
          "new_number": {"type": "string"},
      },
      "required": [
          "first", "old_last", "new_last", "old_number", "new_number"
      ],
  }

  def _prepare_provider_state(self, env: interface.AsyncEnv) -> None:
    _insert_provider_contact(
        env,
        name=f"{self._params['first']} {self._params['old_last']}",
        number=self._params["old_number"],
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self._uses_provider_validation():
      new_name = f"{self._params['first']} {self._params['new_last']}"
      old_name = f"{self._params['first']} {self._params['old_last']}"
      new_ok = _provider_has_phone(
          env, new_name, self._params["new_number"]
      )
      old_present = (
          _provider_name_present(env, old_name)
          or _provider_phone_present(env, self._params["old_number"])
      )
      return 1.0 if new_ok and not old_present else 0.0
    if _uses_local_sqlite(self.package_name):
      new_name = f"{self._params['first']} {self._params['new_last']}"
      old_name = f"{self._params['first']} {self._params['old_last']}"
      new_ok = _local_has_phone(
          env,
          self.package_name,
          new_name,
          self._params["new_number"],
      )
      old_present = _local_name_present(
          env, self.package_name, old_name
      ) or _local_phone_present(
          env, self.package_name, self._params["old_number"]
      )
      return 1.0 if new_ok and not old_present else 0.0
    # An unknown opaque-storage adapter fails closed: visible new values can be
    # an unsaved edit form and do not prove either save.
    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    first = random.choice(_FIRST_NAMES)
    old_last, new_last = random.sample(_LAST_NAMES, 2)
    return {
        "first": first,
        "old_last": old_last,
        "new_last": new_last,
        "old_number": _random_phone(),
        "new_number": _random_phone(),
    }


class _ContactsAddPhoneNumberBase(_ContactsTaskBase):
  """Create a contact with one phone, then add a SECOND phone via edit.

  Provider-backed apps validate both phone rows through ContactsProvider.
  App-private builds fall back to the detail page UI.
  """

  excluded_packages = ()
  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {
          "first": {"type": "string"},
          "last": {"type": "string"},
          "name": {"type": "string"},
          "number": {"type": "string"},
          "number_alt": {"type": "string"},
      },
      "required": ["first", "last", "name", "number", "number_alt"],
  }

  def _prepare_provider_state(self, env: interface.AsyncEnv) -> None:
    _insert_provider_contact(
        env, name=self._params["name"], number=self._params["number"]
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self._uses_provider_validation():
      return (
          1.0
          if _provider_has_all_phones(
              env,
              self._params["name"],
              (self._params["number"], self._params["number_alt"]),
          )
          else 0.0
      )
    ui = env.get_state().ui_elements
    name_ok = base.element_text_contains(ui, (self._params["name"],))
    a_ok = _phone_matches(ui, _normalize_phone(self._params["number"]))
    b_ok = _phone_matches(ui, _normalize_phone(self._params["number_alt"]))
    return 1.0 if name_ok and a_ok and b_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    params = _generate_add_contact_params()
    params["number_alt"] = _random_phone()
    return params


class _ContactsSearchContactBase(_ContactsTaskBase):
  """Create N seeded contacts, then search and open one of them.

  Provider-backed apps validate that all three contacts exist through
  ContactsProvider, then use the UI only for the transient search/filter state.
  """

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {
          "target_name": {"type": "string"},
          "target_number": {"type": "string"},
          "decoy_name_a": {"type": "string"},
          "decoy_number_a": {"type": "string"},
          "decoy_name_b": {"type": "string"},
          "decoy_number_b": {"type": "string"},
      },
      "required": [
          "target_name",
          "target_number",
          "decoy_name_a",
          "decoy_number_a",
          "decoy_name_b",
          "decoy_number_b",
      ],
  }

  def _prepare_provider_state(self, env: interface.AsyncEnv) -> None:
    for name_key, number_key in (
        ("target_name", "target_number"),
        ("decoy_name_a", "decoy_number_a"),
        ("decoy_name_b", "decoy_number_b"),
    ):
      _insert_provider_contact(
          env, name=self._params[name_key], number=self._params[number_key]
      )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    target_visible = base.element_text_contains(ui, (self._params["target_name"],))
    decoy_a_visible = base.element_text_contains(ui, (self._params["decoy_name_a"],))
    decoy_b_visible = base.element_text_contains(ui, (self._params["decoy_name_b"],))
    # A target-only UI can be an unsaved form and does not prove that all three
    # contacts were created. Opaque stores therefore remain false here.
    storage_ok = False
    if self._uses_provider_validation():
      storage_ok = (
          _provider_has_phone(
              env, self._params["target_name"], self._params["target_number"]
          )
          and _provider_has_phone(
              env, self._params["decoy_name_a"], self._params["decoy_number_a"]
          )
          and _provider_has_phone(
              env, self._params["decoy_name_b"], self._params["decoy_number_b"]
          )
      )
    elif _uses_local_sqlite(self.package_name):
      storage_ok = all(
          _local_has_phone(
              env,
              self.package_name,
              self._params[name_key],
              self._params[number_key],
          )
          for name_key, number_key in (
              ("target_name", "target_number"),
              ("decoy_name_a", "decoy_number_a"),
              ("decoy_name_b", "decoy_number_b"),
          )
      )
    target_phone_visible = _phone_matches(
        ui, _normalize_phone(self._params["target_number"])
    )
    return (
        1.0
        if storage_ok
        and target_visible
        and target_phone_visible
        and not decoy_a_visible
        and not decoy_b_visible
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    firsts = random.sample(_FIRST_NAMES, 3)
    lasts = random.sample(_LAST_NAMES, 3)
    return {
        "target_name": f"{firsts[0]} {lasts[0]}",
        "target_number": _random_phone(),
        "decoy_name_a": f"{firsts[1]} {lasts[1]}",
        "decoy_number_a": _random_phone(),
        "decoy_name_b": f"{firsts[2]} {lasts[2]}",
        "decoy_number_b": _random_phone(),
    }


class _ContactsViewContactDetailsBase(_ContactsTaskBase):
  """Open a contact's detail page."""

  complexity = 1.8
  schema = _ContactsAddContactBase.schema

  def _prepare_provider_state(self, env: interface.AsyncEnv) -> None:
    _insert_provider_contact(
        env, name=self._params["name"], number=self._params["number"]
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    name_ok = base.element_text_contains(ui, (self._params["name"],))
    phone_ok = _phone_matches(ui, _normalize_phone(self._params["number"]))
    # Opaque-storage adapters must not accept an unsaved form merely because
    # the requested name and phone are visible.
    provider_ok = False
    if self._uses_provider_validation():
      provider_ok = _provider_has_phone(
          env, self._params["name"], self._params["number"]
      )
    elif _uses_local_sqlite(self.package_name):
      provider_ok = _local_has_phone(
          env,
          self.package_name,
          self._params["name"],
          self._params["number"],
      )
    return 1.0 if provider_ok and name_ok and phone_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_add_contact_params()


class _ContactsDeleteContactBase(_ContactsTaskBase):
  """Create a contact, then delete it.

  Provider-backed apps validate deletion through ContactsProvider. App-private
  builds fall back to checking that name/phone are absent from the UI.
  """

  complexity = 2.2
  schema = _ContactsAddContactBase.schema

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    # Do not let the create-then-delete instruction pass from its initial
    # empty state.  The latch is reset for every task instance.
    self._catbench_seen_exact_target_contact = False

  def _prepare_provider_state(self, env: interface.AsyncEnv) -> None:
    _insert_provider_contact(
        env, name=self._params["name"], number=self._params["number"]
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    seen_target = getattr(
        self, "_catbench_seen_exact_target_contact", False
    )
    if self._uses_provider_validation():
      exact_target_present = _provider_has_phone(
          env, self._params["name"], self._params["number"]
      )
      contact_present = exact_target_present or _provider_name_present(
          env, self._params["name"]
      )
      success, seen_target = _destructive_transition_observation(
          target_state_active=exact_target_present,
          seen_target_state=seen_target,
      )
      self._catbench_seen_exact_target_contact = seen_target
      return 1.0 if success and not contact_present else 0.0
    name_present = base.element_text_contains(ui, (self._params["name"],))
    phone_present = _phone_matches(
        ui, _normalize_phone(self._params["number"])
    )
    if _uses_local_sqlite(self.package_name):
      exact_target_present = _local_has_phone(
          env,
          self.package_name,
          self._params["name"],
          self._params["number"],
      )
      contact_present = exact_target_present or _local_name_present(
          env, self.package_name, self._params["name"]
      )
      success, seen_target = _destructive_transition_observation(
          target_state_active=exact_target_present,
          seen_target_state=seen_target,
      )
      self._catbench_seen_exact_target_contact = seen_target
      return 1.0 if success and not contact_present else 0.0

    target_visible = name_present and phone_present
    success, seen_target = _destructive_transition_observation(
        target_state_active=target_visible,
        seen_target_state=seen_target,
    )
    self._catbench_seen_exact_target_contact = seen_target
    return (
        1.0
        if success
        and _contact_deletion_confirmed(ui)
        and not name_present
        and not phone_present
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_add_contact_params()


class _ContactsShareContactBase(_ContactsTaskBase):
  """Open the Android share sheet for a contact."""

  complexity = 2.0
  schema = _ContactsAddContactBase.schema

  def _prepare_provider_state(self, env: interface.AsyncEnv) -> None:
    _insert_provider_contact(
        env, name=self._params["name"], number=self._params["number"]
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    share_ok = base.element_text_contains(
        ui, ("share", "nearby share", "bluetooth", "messages", "send")
    )
    contact_ok = (
        base.element_text_contains(ui, (self._params["name"],))
        or _phone_matches(ui, _normalize_phone(self._params["number"]))
    )
    return 1.0 if share_ok and contact_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_add_contact_params()


class _ContactsCallContactBase(_ContactsTaskBase):
  """Start a phone call to a contact."""

  complexity = 1.8
  schema = _ContactsAddContactBase.schema

  def _prepare_provider_state(self, env: interface.AsyncEnv) -> None:
    _insert_provider_contact(
        env, name=self._params["name"], number=self._params["number"]
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    call_ok = _external_call_surface(ui, self.package_name)
    contact_ok = (
        base.element_text_contains(ui, (self._params["name"],))
        or _phone_matches(ui, _normalize_phone(self._params["number"]))
    )
    # The external dialer proves a call transition, but without readable
    # storage it does not prove the instruction's preceding contact creation.
    storage_ok = False
    if self._uses_provider_validation():
      storage_ok = _provider_has_phone(
          env, self._params["name"], self._params["number"]
      )
    elif _uses_local_sqlite(self.package_name):
      storage_ok = _local_has_phone(
          env,
          self.package_name,
          self._params["name"],
          self._params["number"],
      )
    return 1.0 if storage_ok and call_ok and contact_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_add_contact_params()


class _ContactsMessageContactBase(_ContactsTaskBase):
  """Open an SMS compose flow to a contact."""

  complexity = 1.8
  schema = _ContactsAddContactBase.schema

  def _prepare_provider_state(self, env: interface.AsyncEnv) -> None:
    _insert_provider_contact(
        env, name=self._params["name"], number=self._params["number"]
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    message_ok = _external_message_composer(ui, self.package_name)
    contact_ok = (
        base.element_text_contains(ui, (self._params["name"],))
        or _phone_matches(ui, _normalize_phone(self._params["number"]))
    )
    # The external composer proves a messaging transition, but without
    # readable storage it does not prove the preceding contact creation.
    storage_ok = False
    if self._uses_provider_validation():
      storage_ok = _provider_has_phone(
          env, self._params["name"], self._params["number"]
      )
    elif _uses_local_sqlite(self.package_name):
      storage_ok = _local_has_phone(
          env,
          self.package_name,
          self._params["name"],
          self._params["number"],
      )
    return 1.0 if storage_ok and message_ok and contact_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_add_contact_params()


class _ContactsAddContactWithMobileLabelBase(_ContactsTaskBase):
  """Create a contact with the phone label set to ``Mobile``.

  Provider-backed apps validate the contact through ContactsProvider. The
  app-private fallback also checks the visible ``Mobile`` label.
  """

  excluded_packages = ()
  complexity = 1.6
  schema = {
      "type": "object",
      "properties": {
          "first": {"type": "string"},
          "last": {"type": "string"},
          "name": {"type": "string"},
          "number": {"type": "string"},
      },
      "required": ["first", "last", "name", "number"],
  }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self._uses_provider_validation():
      return (
          1.0
          if _provider_has_phone(
              env, self._params["name"], self._params["number"]
          )
          else 0.0
      )
    ui = env.get_state().ui_elements
    name_ok = base.element_text_contains(ui, (self._params["name"],))
    phone_ok = _phone_matches(ui, _normalize_phone(self._params["number"]))
    label_ok = base.element_text_contains(ui, ("Mobile", "Cell"))
    return 1.0 if name_ok and phone_ok and label_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return _generate_add_contact_params()


# -----------------------------------------------------------------------------
# Per-app packages.
# -----------------------------------------------------------------------------

_GOOGLE_CONTACTS_PACKAGE: Final[str] = "com.google.android.contacts"
_FOSSIFY_CONTACTS_PACKAGE: Final[str] = "org.fossify.contacts"
_CONNECT_YOU_PACKAGE: Final[str] = "com.bnyro.contacts"
_SIMPLE_CONTACTS_PRO_PACKAGE: Final[str] = "com.simplemobiletools.contacts.pro"
_APP_DISPLAY_NAMES: Final[dict[str, str]] = {
    _GOOGLE_CONTACTS_PACKAGE: "Google Contacts",
    _FOSSIFY_CONTACTS_PACKAGE: "Fossify Contacts",
    _CONNECT_YOU_PACKAGE: "Connect You",
    _SIMPLE_CONTACTS_PRO_PACKAGE: "Simple Contacts Pro SE",
    _RIGHT_CONTACTS_PACKAGE: "Right Contact",
}


def _class_suffix(display_name: str) -> str:
  return "".join(ch for ch in display_name if ch.isalnum())


_TEMPLATES: Final[dict[type, str]] = {
    _ContactsAddContactBase: (
        "In the {app} app, create a new contact for {{name}}. Their number is"
        " {{number}}."
    ),
    _ContactsNewContactDraftBase: (
        "In the {app} app, go to the new contact screen and enter: First Name:"
        " {{first}}, Last Name: {{last}}, Phone: {{phone}}, Phone Label:"
        " {{phone_label}}. Do NOT hit save."
    ),
    _ContactsAddContactWithEmailBase: (
        "In the {app} app, create a new contact for {{name}} with phone"
        " {{number}} and email {{email}}."
    ),
    _ContactsAddContactWithCompanyBase: (
        "In the {app} app, create a new contact for {{name}} at {{company}}."
        " Their number is {{number}}."
    ),
    _ContactsAddContactWithAddressBase: (
        "In the {app} app, create a new contact for {{name}} with phone"
        " {{number}} and the postal address `{{address}}`."
    ),
    _ContactsAddFavoriteContactBase: (
        "In the {app} app, create a new contact for {{name}} with phone"
        " {{number}} and mark the contact as a favorite / starred."
    ),
    _ContactsRemoveFavoriteContactBase: (
        "In the {app} app, create a new contact for {{name}} with phone"
        " {{number}}, mark it as a favorite / starred, then remove it from"
        " favorites."
    ),
    _ContactsAddTwoContactsBase: (
        "In the {app} app, create two contacts: (1) {{name_a}} with phone"
        " {{number_a}} and (2) {{name_b}} with phone {{number_b}}."
    ),
    _ContactsAddThreeContactsBase: (
        "In the {app} app, create three contacts: (1) {{name_a}} ({{number_a}}),"
        " (2) {{name_b}} ({{number_b}}), and (3) {{name_c}} ({{number_c}})."
    ),
    _ContactsAddContactWithSecondPhoneBase: (
        "In the {app} app, create a new contact for {{name}} with two"
        " phone numbers: {{number}} and {{number_alt}}."
    ),
    _ContactsAddContactWithMobileLabelBase: (
        "In the {app} app, create a new contact for {{name}} with phone"
        " {{number}} labelled `Mobile'."
    ),
    _ContactsEditContactBase: (
        "In the {app} app, create a contact for `{{first}} {{old_last}}`"
        " with phone {{old_number}}, then edit the contact to change the"
        " surname to `{{new_last}}` and the phone number to {{new_number}}."
    ),
    _ContactsAddPhoneNumberBase: (
        "In the {app} app, create a contact for {{name}} with phone"
        " {{number}}, then EDIT that same contact to add a second phone"
        " number {{number_alt}}."
    ),
    _ContactsSearchContactBase: (
        "In the {app} app, create three contacts: `{{target_name}}`"
        " ({{target_number}}), `{{decoy_name_a}}` ({{decoy_number_a}}),"
        " and `{{decoy_name_b}}` ({{decoy_number_b}}). Then use the"
        " in-app search to filter the list down to `{{target_name}}` only"
        " and open that contact's detail page."
    ),
    _ContactsViewContactDetailsBase: (
        "In the {app} app, create a contact for {{name}} with phone"
        " {{number}}, then open that contact's detail page."
    ),
    _ContactsDeleteContactBase: (
        "In the {app} app, create a contact for {{name}} with phone"
        " {{number}} and then delete that contact."
    ),
    _ContactsShareContactBase: (
        "In the {app} app, create a contact for {{name}} with phone"
        " {{number}}, then share that contact and leave the Android share"
        " sheet open."
    ),
    _ContactsCallContactBase: (
        "In the {app} app, create a contact for {{name}} with phone"
        " {{number}}, then call that contact."
    ),
    _ContactsMessageContactBase: (
        "In the {app} app, create a contact for {{name}} with phone"
        " {{number}}, then start a text message to that contact."
    ),
}


_PROVIDER_TEMPLATES: Final[dict[type, str]] = {
    _ContactsAddFavoriteContactBase: (
        "In the {app} app, find the existing contact {{name}} with phone"
        " {{number}} and mark the contact as a favorite / starred."
    ),
    _ContactsRemoveFavoriteContactBase: (
        "In the {app} app, find the existing favorite contact {{name}} with"
        " phone {{number}} and remove it from favorites / starred contacts."
    ),
    _ContactsEditContactBase: (
        "In the {app} app, edit the existing contact"
        " `{{first}} {{old_last}}` with phone {{old_number}} so the"
        " surname becomes `{{new_last}}` and the phone number becomes"
        " {{new_number}}."
    ),
    _ContactsAddPhoneNumberBase: (
        "In the {app} app, edit the existing contact {{name}} with phone"
        " {{number}} and add a second phone number {{number_alt}}."
    ),
    _ContactsSearchContactBase: (
        "In the {app} app, use the in-app search to find the existing"
        " contact `{{target_name}}` and open that contact's detail page."
        " The decoy contacts `{{decoy_name_a}}` and `{{decoy_name_b}}`"
        " should not remain visible after filtering."
    ),
    _ContactsViewContactDetailsBase: (
        "In the {app} app, find the existing contact {{name}} with phone"
        " {{number}} and open that contact's detail page."
    ),
    _ContactsDeleteContactBase: (
        "In the {app} app, delete the existing contact {{name}} with phone"
        " {{number}}."
    ),
    _ContactsShareContactBase: (
        "In the {app} app, find the existing contact {{name}} with phone"
        " {{number}}, then share that contact and leave the Android share"
        " sheet open."
    ),
    _ContactsCallContactBase: (
        "In the {app} app, find the existing contact {{name}} with phone"
        " {{number}}, then call that contact."
    ),
    _ContactsMessageContactBase: (
        "In the {app} app, find the existing contact {{name}} with phone"
        " {{number}}, then start a text message to that contact."
    ),
}


# Cross-app Contacts task templates. The 10 short names below ARE the user's
# target task list for the Contacts category in hybrid mode.
_BASE_SHORT_NAMES: Final[dict[type, str]] = {
    _ContactsAddContactBase: "ContactsAddContact",
    _ContactsNewContactDraftBase: "ContactsNewContactDraft",
    _ContactsEditContactBase: "ContactsEditContact",
    _ContactsSearchContactBase: "ContactsSearchContact",
    _ContactsViewContactDetailsBase: "ContactsViewContactDetails",
    _ContactsAddFavoriteContactBase: "ContactsAddFavoriteContact",
    _ContactsRemoveFavoriteContactBase: "ContactsRemoveFavoriteContact",
    _ContactsDeleteContactBase: "ContactsDeleteContact",
    _ContactsCallContactBase: "ContactsCallContact",
    _ContactsMessageContactBase: "ContactsMessageContact",
}


_PACKAGES = (
    _GOOGLE_CONTACTS_PACKAGE,
    _FOSSIFY_CONTACTS_PACKAGE,
    _CONNECT_YOU_PACKAGE,
    _SIMPLE_CONTACTS_PRO_PACKAGE,
    _RIGHT_CONTACTS_PACKAGE,
)


for _base_cls, _short in _BASE_SHORT_NAMES.items():
  excluded = getattr(_base_cls, "excluded_packages", ())
  for _pkg in _PACKAGES:
    if _pkg in excluded:
      continue
    _display = _APP_DISPLAY_NAMES[_pkg]
    _suffix = _class_suffix(_display)
    _cls_name = f"{_short}For{_suffix}"
    # Every app receives the same create-first workflow.  The older provider
    # templates used a seeded "existing contact" only for Google Contacts,
    # which changed both the initial state and the task across apps.
    _template_source = _TEMPLATES[_base_cls]
    _ui_terminal_semantics = {
        "ContactsSearchContact", "ContactsViewContactDetails"
    }
    _external_transition_semantics = {
        "ContactsCallContact", "ContactsMessageContact"
    }
    _destructive_transition_semantics = {
        "ContactsRemoveFavoriteContact", "ContactsDeleteContact"
    }
    if _pkg in _LOCAL_SQLITE_PACKAGES:
      if _short == "ContactsNewContactDraft":
        _validation_mode = "Editable UI + SQLite durable absence"
      elif _short in _ui_terminal_semantics:
        _validation_mode = "SQLite durable state + exact-target UI terminal state"
      elif _short in _external_transition_semantics:
        _validation_mode = "SQLite durable state + external intent surface"
      elif _short in _destructive_transition_semantics:
        _validation_mode = (
            "SQLite durable state + exact-target transition latch"
        )
      else:
        _validation_mode = "SQLite durable state"
    else:
      if _short == "ContactsNewContactDraft":
        _validation_mode = "Editable UI + ContactsProvider durable absence"
      elif _short in _ui_terminal_semantics:
        _validation_mode = (
            "ContactsProvider + exact-target UI terminal state"
        )
      elif _short in _external_transition_semantics:
        _validation_mode = "ContactsProvider + external intent surface"
      elif _short in _destructive_transition_semantics:
        _validation_mode = (
            "ContactsProvider durable state + exact-target transition latch"
        )
      else:
        _validation_mode = "ContactsProvider durable state"
    _attrs = {
        "app_names": (_pkg,),
        "package_name": _pkg,
        "catbench_semantic_id": _short,
        "catbench_app_display_name": _display,
        "template": _template_source.format(app=_display),
        "validation_mode": _validation_mode,
    }
    globals()[_cls_name] = type(_cls_name, (_base_cls,), _attrs)


del _base_cls, _short, _pkg, _display, _suffix, _cls_name, _template_source, _validation_mode, _attrs
del _ui_terminal_semantics, _external_transition_semantics, _destructive_transition_semantics
