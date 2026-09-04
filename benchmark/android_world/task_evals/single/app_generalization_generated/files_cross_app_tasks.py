"""Cross-app file-manager task ports for the app-generalization suite.

Because file-manager apps share a common mental model (navigate directory tree,
select files, invoke an action), tasks here are expressed as:

  * Seed the public storage area with a known set of files BEFORE the agent
    runs. Seeding happens via ``adb shell`` calls in ``initialize_task`` and is
    shared across every file-manager app (they all read the same file system).
  * After the agent acts, verify the on-device file state via ``adb shell``,
    NOT via UI text. This avoids brittle UI heuristics since file managers
    render identical content very differently.

Tasks in this module:

  * ``FilesCreateFolder`` -- create a new folder in a target directory.
  * ``FilesRenameFile`` -- rename a seeded file.
  * ``FilesDeleteFile`` -- delete a seeded file.
  * ``FilesMoveFile`` -- move a seeded file between two directories.
  * ``FilesSaveCopyOfFile`` -- save/copy a seeded file to a second directory.
  * ``FilesSearchFile`` -- locate a nested seeded file and rename it.
  * ``FilesCompressFiles`` -- compress seeded files into an archive.
  * ``FilesExtractArchive`` -- extract a seeded archive.
  * ``FilesViewFileInfo`` -- open file details/properties (UI-text heuristic).
  * ``FilesShareFile`` -- open the system share sheet for a seeded file
    (UI-text heuristic).
"""

from __future__ import annotations

import random
import re
import shlex
import string
import time
from typing import Any, Final
from urllib import parse as urllib_parse

from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base as base,
)


_ROOT: Final[str] = "/sdcard/CATBench"
_ZIP_B64: Final[str] = (
    "UEsDBBQAAAAIAFcQnlx1Mp9fGwAAABkAAAAPAAAAcGFja2VkX25vdGUudHh0"
    "c3YMcUrNS85QSCxKzsgsS1UoSKzMyU9M4QIAUEsBAhQDFAAAAAgAVxCeXHUy"
    "n18bAAAAGQAAAA8AAAAAAAAAAAAAAIABAAAAAHBhY2tlZF9ub3RlLnR4dFBL"
    "BQYAAAAAAQABAD0AAABIAAAAAAA="
)
_ZIP_INNER_NAME: Final[str] = "packed_note.txt"
_ZIP_INNER_CONTENT: Final[str] = "CATBench archive payload"
_ZIP_SHA256: Final[str] = (
    "1811de41fddf18dc54aae665e3929a0ce1e9e7346cf4c3dd85e000d2fcf71666"
)
_DELETE_DECOY_NAME: Final[str] = "catbench_keep.txt"
_DELETE_DECOY_CONTENT: Final[str] = "keep"
_SEARCH_TARGET_DIRECTORY: Final[str] = f"{_ROOT}/Docs/Reference/Incoming"
_SEARCH_DECOYS: Final[tuple[tuple[str, str], ...]] = tuple(
    (f"{_ROOT}/Docs/decoy_{index}.txt", f"decoy {index}")
    for index in range(5)
)


def _random_name(prefix: str, ext: str = ".txt") -> str:
  suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
  return f"{prefix}_{suffix}{ext}"


def _quote(value: str) -> str:
  """Quote one dynamic value for the Android shell."""
  return shlex.quote(value)


def _delete_target_content(file_name: str) -> str:
  return f"CATBENCH_DELETE_TARGET:{file_name}"


def _search_target_content(needle: str) -> str:
  return f"CATBENCH_LOCATE_TARGET:{needle}"


def _adb_shell(env: interface.AsyncEnv, cmd: str) -> str:
  out = adb_utils.issue_generic_request(
      ["shell", cmd], env.controller
  )
  return out.generic.output.decode("utf-8", errors="ignore") if out else ""


def _seed_file(env: interface.AsyncEnv, path: str, content: str) -> None:
  parent = path.rsplit("/", 1)[0]
  _adb_shell(env, f"mkdir -p {_quote(parent)}")
  _adb_shell(
      env,
      f"printf '%s\\n' {_quote(content)} > {_quote(path)}",
  )
  if not _path_exists(env, path):
    raise RuntimeError(f"Failed to seed precondition file {path!r}.")


def _seed_dir(env: interface.AsyncEnv, path: str) -> None:
  _adb_shell(env, f"mkdir -p {_quote(path)}")
  if not _path_exists(env, path):
    raise RuntimeError(f"Failed to seed precondition directory {path!r}.")


def _seed_zip(env: interface.AsyncEnv, path: str) -> None:
  parent = path.rsplit("/", 1)[0]
  _adb_shell(env, f"mkdir -p {_quote(parent)}")
  _adb_shell(
      env,
      f"printf '%s' {_quote(_ZIP_B64)} | base64 -d > {_quote(path)}",
  )
  if _file_sha256(env, path) != _ZIP_SHA256:
    raise RuntimeError(f"Failed to seed precondition archive {path!r}.")


def _path_exists(env: interface.AsyncEnv, path: str) -> bool:
  path_q = _quote(path)
  out = _adb_shell(env, f"[ -e {path_q} ] && echo YES || echo NO")
  return out.strip() == "YES"


def _is_directory(env: interface.AsyncEnv, path: str) -> bool:
  path_q = _quote(path)
  out = _adb_shell(env, f"[ -d {path_q} ] && echo YES || echo NO")
  return out.strip() == "YES"


def _file_has_content(
    env: interface.AsyncEnv,
    path: str,
    expected_content: str,
) -> bool:
  """Return whether ``path`` is a regular file with the seeded text.

  The fixture writer uses ``printf`` with exactly one trailing newline.
  Checking type and byte-exact content prevents placeholders, symlinks, and
  line-ending changes from passing a rename/move/copy task.
  """
  path_q = _quote(path)
  out = _adb_shell(
      env,
      f"[ -f {path_q} ] && [ ! -L {path_q} ]"
      f" && cat {path_q} 2>/dev/null || true",
  )
  return out == f"{expected_content}\n"


def _file_sha256(env: interface.AsyncEnv, path: str) -> str:
  """Return a regular file's SHA-256, or an empty string fail-closed."""
  path_q = _quote(path)
  out = _adb_shell(
      env,
      f"[ -f {path_q} ] && sha256sum {path_q} 2>/dev/null || true",
  ).strip()
  match = re.match(r"^([0-9a-fA-F]{64})(?:\s|$)", out)
  return match.group(1).lower() if match else ""


def _find_name(env: interface.AsyncEnv, name: str) -> bool:
  out = _adb_shell(
      env,
      f"find {_quote(_ROOT)} -name {_quote(name)} -print -quit",
  )
  return bool(out.strip())


def _find_files_named(env: interface.AsyncEnv, name: str) -> tuple[str, ...]:
  out = _adb_shell(
      env,
      f"find {_quote(_ROOT)} -type f -name {_quote(name)} -print",
  )
  return tuple(path.strip() for path in out.splitlines() if path.strip())


def _find_file_with_content(
    env: interface.AsyncEnv,
    name: str,
    expected_content: str,
) -> bool:
  return any(
      _file_has_content(env, path, expected_content)
      for path in _find_files_named(env, name)
  )


def _find_exactly_one_file_with_content(
    env: interface.AsyncEnv,
    name: str,
    expected_content: str,
) -> bool:
  paths = _find_files_named(env, name)
  return len(paths) == 1 and _file_has_content(
      env, paths[0], expected_content
  )


def _find_any_file_with_content(
    env: interface.AsyncEnv,
    expected_content: str,
) -> bool:
  out = _adb_shell(
      env,
      f"find {_quote(_ROOT)} -type f -print",
  )
  return any(
      _file_has_content(env, path.strip(), expected_content)
      for path in out.splitlines()
      if path.strip()
  )


def _archive_kind(archive: str) -> str:
  lower = archive.casefold()
  if lower.endswith(".zip"):
    return "zip"
  if lower.endswith((".tar", ".tar.gz", ".tgz")):
    return "tar"
  # The candidate image does not contain a pinned 7z extractor. A .7z output
  # is detected as an archive candidate by the verifier, but deliberately
  # fails closed rather than being accepted from a filename-only listing.
  return ""


def _zip_listing_members(listing: str) -> tuple[str, ...]:
  """Parse member paths from Android/Info-ZIP ``unzip -lq`` output."""
  members: list[str] = []
  for line in listing.splitlines():
    # Both Android toybox and Info-ZIP put length, date, time, and then the
    # complete member path on a data row. split(..., 3) preserves spaces in the
    # path while excluding headers, separators, and the three-field summary.
    fields = line.strip().split(None, 3)
    if len(fields) == 4 and fields[0].isdigit():
      members.append(fields[3])
  return tuple(members)


def _archive_members(
    env: interface.AsyncEnv,
    archive: str,
) -> tuple[str, ...]:
  """Return inspectable member paths, or an empty tuple fail-closed."""
  kind = _archive_kind(archive)
  archive_q = _quote(archive)
  if kind == "zip":
    listing = _adb_shell(
        env,
        f"unzip -lq {archive_q} 2>/dev/null || true",
    )
    return _zip_listing_members(listing)
  if kind == "tar":
    listing = _adb_shell(
        env,
        f"tar -tf {archive_q} 2>/dev/null || true",
    )
    return tuple(line.strip() for line in listing.splitlines() if line.strip())
  return ()


def _archive_member_content(
    env: interface.AsyncEnv,
    archive: str,
    member: str,
) -> str:
  """Read one supported archive member without extracting it to storage."""
  kind = _archive_kind(archive)
  archive_q = _quote(archive)
  member_q = _quote(member)
  if kind == "zip":
    return _adb_shell(
        env,
        f"unzip -p {archive_q} {member_q} 2>/dev/null || true",
    )
  if kind == "tar":
    return _adb_shell(
        env,
        f"tar -xOf {archive_q} {member_q} 2>/dev/null || true",
    )
  return ""


def _archive_has_exact_members(
    env: interface.AsyncEnv,
    archive: str,
    expected_contents: dict[str, str],
) -> bool:
  """Whether each expected basename occurs once with its exact seeded bytes."""
  if not _archive_kind(archive):
    return False
  members = _archive_members(env, archive)
  selected: dict[str, str] = {}
  for name in expected_contents:
    matches = tuple(
        member
        for member in members
        if not member.endswith("/")
        and member.rstrip("/").rsplit("/", 1)[-1] == name
    )
    if len(matches) != 1:
      return False
    selected[name] = matches[0]
  return all(
      _archive_member_content(env, archive, selected[name])
      == f"{expected_content}\n"
      for name, expected_content in expected_contents.items()
  )


def _is_empty_dir(env: interface.AsyncEnv, path: str) -> bool:
  out = _adb_shell(env, f"ls -A '{path}' 2>/dev/null | wc -l")
  return out.strip() == "0"


# -----------------------------------------------------------------------------
# Base evaluators.
# -----------------------------------------------------------------------------


class _FilesTaskBase(base.PackageAppEval):
  """Shared seeding + teardown for file-manager tasks."""

  clear_data_on_init = False  # preserve seeded files across app launch
  clear_data_on_teardown = False

  def _cleanup(self, env: interface.AsyncEnv) -> None:
    _adb_shell(env, f"rm -rf '{_ROOT}'")

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    # Reset the shared seed area, let the subclass seed its fixture files,
    # then refresh so the (already launched) file manager actually sees them.
    self._cleanup(env)
    self._seed_state(env)
    self._refresh_after_seeding(env)

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    """Hook: seed fixture files under ``_ROOT``. Runs after app launch and is
    always followed by a media scan + app relaunch."""

  def _refresh_after_seeding(self, env: interface.AsyncEnv) -> None:
    """Make seeded files visible to the file-manager app.

    Files written via ``adb shell`` bypass MediaStore and any in-app
    directory cache: a file manager that indexed storage at launch shows a
    stale (empty) listing. Kick the media scanner for the seed root, then
    force-stop and relaunch the app so it re-reads the filesystem.
    """
    _adb_shell(
        env,
        "(am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE"
        f" -d 'file://{_ROOT}' </dev/null >/dev/null 2>&1 &) || true",
    )
    _adb_shell(env, f"am force-stop {self.package_name} || true")
    deadline = time.time() + 3.0
    while time.time() < deadline:
      if not _adb_shell(env, f"pidof {self.package_name} || true").strip():
        break
      time.sleep(0.1)
    adb_utils.launch_app(self.package_name, env.controller)

  def tear_down(self, env: interface.AsyncEnv) -> None:
    self._cleanup(env)
    super().tear_down(env)


class _FilesDeleteFileBase(_FilesTaskBase):
  """Delete a marked file from the normalized CATBench storage tree.

  File-manager trash implementations differ. CATBench therefore normalizes
  deletion as removal of the target path and its unique marker from
  ``/sdcard/CATBench``; app-private or system trash outside that tree is not
  inspected. The exact decoy file must remain intact.
  """

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {"file_name": {"type": "string"}},
      "required": ["file_name"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    file_name = self._params["file_name"]
    _seed_file(
        env,
        f"{_ROOT}/Docs/{file_name}",
        _delete_target_content(file_name),
    )
    _seed_file(
        env,
        f"{_ROOT}/Docs/{_DELETE_DECOY_NAME}",
        _DELETE_DECOY_CONTENT,
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    file_name = self._params["file_name"]
    target_gone = not _path_exists(env, f"{_ROOT}/Docs/{file_name}")
    marker_gone = not _find_any_file_with_content(
        env, _delete_target_content(file_name)
    )
    decoy_kept = _file_has_content(
        env,
        f"{_ROOT}/Docs/{_DELETE_DECOY_NAME}",
        _DELETE_DECOY_CONTENT,
    )
    return 1.0 if target_gone and marker_gone and decoy_kept else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"file_name": _random_name("note")}


class _FilesMoveFileBase(_FilesTaskBase):
  """Move a seeded file from ``Docs/`` to ``Archive/``."""

  complexity = 2.2
  schema = {
      "type": "object",
      "properties": {"file_name": {"type": "string"}},
      "required": ["file_name"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_file(env, f"{_ROOT}/Docs/{self._params['file_name']}", "payload")
    _seed_dir(env, f"{_ROOT}/Archive")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    src = f"{_ROOT}/Docs/{self._params['file_name']}"
    dst = f"{_ROOT}/Archive/{self._params['file_name']}"
    return (
        1.0
        if _file_has_content(env, dst, "payload")
        and not _path_exists(env, src)
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"file_name": _random_name("invoice")}


class _FilesCopyFileBase(_FilesTaskBase):
  """Copy a seeded file from ``Docs/`` to ``Archive/`` (original stays)."""

  complexity = 2.0
  schema = _FilesMoveFileBase.schema

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_file(env, f"{_ROOT}/Docs/{self._params['file_name']}", "payload")
    _seed_dir(env, f"{_ROOT}/Archive")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    src = f"{_ROOT}/Docs/{self._params['file_name']}"
    dst = f"{_ROOT}/Archive/{self._params['file_name']}"
    return (
        1.0
        if _file_has_content(env, src, "payload")
        and _file_has_content(env, dst, "payload")
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"file_name": _random_name("report")}


class _FilesRenameFileBase(_FilesTaskBase):
  """Rename ``old_name`` -> ``new_name`` inside ``Docs/``."""

  complexity = 2.0
  schema = {
      "type": "object",
      "properties": {
          "old_name": {"type": "string"},
          "new_name": {"type": "string"},
      },
      "required": ["old_name", "new_name"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_file(env, f"{_ROOT}/Docs/{self._params['old_name']}", "payload")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    old = f"{_ROOT}/Docs/{self._params['old_name']}"
    new = f"{_ROOT}/Docs/{self._params['new_name']}"
    return (
        1.0
        if _file_has_content(env, new, "payload")
        and not _path_exists(env, old)
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {
        "old_name": _random_name("draft"),
        "new_name": _random_name("final"),
    }


class _FilesCreateFolderBase(_FilesTaskBase):
  """Create a new folder beneath ``CATBench/``."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {"folder_name": {"type": "string"}},
      "required": ["folder_name"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_dir(env, _ROOT)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if _is_directory(env, f"{_ROOT}/{self._params['folder_name']}")
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"folder_name": _random_name("Folder", ext="")}


class _FilesDeleteFolderBase(_FilesTaskBase):
  """Delete a seeded folder (and its contents)."""

  complexity = 1.6
  schema = {
      "type": "object",
      "properties": {"folder_name": {"type": "string"}},
      "required": ["folder_name"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_dir(env, f"{_ROOT}/{self._params['folder_name']}")
    _seed_file(
        env,
        f"{_ROOT}/{self._params['folder_name']}/inside.txt",
        "inside",
    )
    _seed_file(env, f"{_ROOT}/catbench_keep/keep.txt", "keep")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    decoy_kept = _path_exists(env, f"{_ROOT}/catbench_keep/keep.txt")
    return (
        1.0
        if decoy_kept
        and not _path_exists(env, f"{_ROOT}/{self._params['folder_name']}")
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"folder_name": _random_name("Old", ext="")}


class _FilesSaveCopyOfReceiptBase(_FilesTaskBase):
  """Copy a receipt file from ``Inbox/`` to ``Receipts/``."""

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {"file_name": {"type": "string"}},
      "required": ["file_name"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_file(
        env,
        f"{_ROOT}/Inbox/{self._params['file_name']}",
        "amount: 12.34",
    )
    _seed_dir(env, f"{_ROOT}/Receipts")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    src = f"{_ROOT}/Inbox/{self._params['file_name']}"
    dst = f"{_ROOT}/Receipts/{self._params['file_name']}"
    return (
        1.0
        if _file_has_content(env, src, "amount: 12.34")
        and _file_has_content(env, dst, "amount: 12.34")
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"file_name": _random_name("receipt", ext=".pdf")}


class _FilesDeleteAllInFolderBase(_FilesTaskBase):
  """Empty a folder that was seeded with three files."""

  complexity = 2.6
  schema = {
      "type": "object",
      "properties": {"folder_name": {"type": "string"}},
      "required": ["folder_name"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    folder = f"{_ROOT}/{self._params['folder_name']}"
    for i in range(3):
      _seed_file(env, f"{folder}/item_{i}.txt", f"content {i}")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    folder = f"{_ROOT}/{self._params['folder_name']}"
    return (
        1.0
        if _path_exists(env, folder) and _is_empty_dir(env, folder)
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"folder_name": _random_name("Cleanup", ext="")}


class _FilesCreateNestedFolderBase(_FilesTaskBase):
  """Create a nested folder ``parent/child`` (two levels deep)."""

  complexity = 1.6
  schema = {
      "type": "object",
      "properties": {
          "parent": {"type": "string"},
          "child": {"type": "string"},
      },
      "required": ["parent", "child"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_dir(env, _ROOT)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (
        1.0
        if _path_exists(
            env, f"{_ROOT}/{self._params['parent']}/{self._params['child']}"
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {
        "parent": _random_name("Outer", ext=""),
        "child": _random_name("Inner", ext=""),
    }


class _FilesSearchFileBase(_FilesTaskBase):
  """Outcome-verifiable locate-and-rename task.

  Seeded layout:
    CATBench/Docs/<5 decoy files>
    CATBench/Docs/Reference/Incoming/<needle>  (nested target)

  The task does not prescribe a UI method that durable storage cannot verify.
  Success requires exactly one ``found.txt`` with the target's unique payload,
  no original basename anywhere in CATBench, and all decoys intact.
  """

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {"needle": {"type": "string"}},
      "required": ["needle"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    for path, content in _SEARCH_DECOYS:
      _seed_file(env, path, content)
    needle = self._params["needle"]
    _seed_file(
        env,
        f"{_SEARCH_TARGET_DIRECTORY}/{needle}",
        _search_target_content(needle),
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    needle = self._params["needle"]
    found_ok = _find_exactly_one_file_with_content(
        env,
        "found.txt",
        _search_target_content(needle),
    )
    original_gone = not _find_name(env, needle)
    decoys_preserved = all(
        _file_has_content(env, path, content)
        for path, content in _SEARCH_DECOYS
    )
    return 1.0 if found_ok and original_gone and decoys_preserved else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"needle": _random_name("target")}


class _FilesCompressFilesBase(_FilesTaskBase):
  """Compress a set of seeded files into a single ZIP archive.

  Seeded layout:
    CATBench/ToCompress/file_0.txt ... file_2.txt

  Success: exactly one ZIP archive exists beneath CATBench, the three source
  files remain byte-exact, and the archive contains each exact seeded payload.
  The instruction requires ZIP because the frozen runtime has no pinned 7z
  reader and accepting a format from its filename alone is not sound.
  """

  complexity = 2.6
  schema = {"type": "object", "properties": {}}

  _SEED_NAMES: Final[tuple[str, ...]] = (
      "file_0.txt", "file_1.txt", "file_2.txt"
  )

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    for name in self._SEED_NAMES:
      _seed_file(env, f"{_ROOT}/ToCompress/{name}", name)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    out = _adb_shell(
        env,
        f"find {_quote(_ROOT)} -type f -iname '*.zip' -print",
    )
    archives = [p.strip() for p in out.splitlines() if p.strip()]
    if len(archives) != 1:
      return 0.0
    expected_contents = {name: name for name in self._SEED_NAMES}
    sources_preserved = all(
        _file_has_content(
            env,
            f"{_ROOT}/ToCompress/{name}",
            expected_content,
        )
        for name, expected_content in expected_contents.items()
    )
    if not sources_preserved:
      return 0.0
    return 1.0 if _archive_has_exact_members(
        env, archives[0], expected_contents
    ) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class _FilesExtractArchiveBase(_FilesTaskBase):
  """Extract a seeded zip archive beneath ``CATBench/Archives``."""

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {"archive_name": {"type": "string"}},
      "required": ["archive_name"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_zip(env, f"{_ROOT}/Archives/{self._params['archive_name']}")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    archive = f"{_ROOT}/Archives/{self._params['archive_name']}"
    source_preserved = _file_sha256(env, archive) == _ZIP_SHA256
    extracted_ok = _find_file_with_content(
        env, _ZIP_INNER_NAME, _ZIP_INNER_CONTENT
    )
    return 1.0 if source_preserved and extracted_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"archive_name": _random_name("bundle", ext=".zip")}


class _FilesViewFileInfoBase(_FilesTaskBase):
  """Open a seeded file's details/properties screen.

  This is a UI-text heuristic by design: opening an info/properties panel does
  not mutate shared storage, so there is no filesystem delta to validate.
  """

  complexity = 1.8
  schema = {
      "type": "object",
      "properties": {"file_name": {"type": "string"}},
      "required": ["file_name"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_file(env, f"{_ROOT}/Docs/{self._params['file_name']}", "payload")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    file_name = self._params["file_name"]
    exact_file_pattern = re.compile(
        rf"(?<![\w.-]){re.escape(file_name)}(?![\w.-])"
    )
    file_ok = any(
        field and exact_file_pattern.search(field)
        for element in ui
        for field in (element.text, element.content_description)
    )
    panel_ok = base.element_text_contains_word(
        ui, ("details", "properties", "information", "file info")
    )
    # A still-unopened context menu can contain ``Properties`` while the file
    # row remains visible behind it. Require concrete metadata from the opened
    # information surface as independent terminal-state evidence.
    metadata_marker_groups = (
        ("size",),
        ("modified", "last modified"),
        ("location", "path"),
        ("file type", "mime type"),
        ("permissions",),
        ("owner",),
        ("created",),
    )
    metadata_ok = sum(
        base.element_text_contains_word(ui, marker_group)
        for marker_group in metadata_marker_groups
    ) >= 2
    return 1.0 if file_ok and panel_ok and metadata_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"file_name": _random_name("details")}


class _FilesShareFileBase(_FilesTaskBase):
  """Open the Android share sheet for a seeded file.

  Android binary sharing sends a content URI in ``Intent.EXTRA_STREAM``.  On
  API 33, IntentResolver resolves that URI's display name into the system
  chooser's ``content_preview_filename`` field.  We therefore require the
  *exact seeded basename in that chooser-owned field*.  Merely opening any
  chooser, or combining a chooser target with the filename still visible in
  the source file manager, is not enough.

  This intentionally fails closed when an app supplies no accessible file
  preview.  Such an app needs a separately validated intent/artifact probe;
  accepting generic chooser labels would certify the wrong file.
  """

  complexity = 1.8
  schema = {
      "type": "object",
      "properties": {"file_name": {"type": "string"}},
      "required": ["file_name"],
  }

  _CHOOSER_PACKAGES = (
      "com.android.intentresolver",
      # Framework-bundled ResolverActivity builds expose package ``android``.
      "android",
  )
  _FILE_PREVIEW_RESOURCE_IDS = frozenset(("content_preview_filename",))

  @classmethod
  def _chooser_preview_matches_file(
      cls,
      ui_elements: list[Any],
      file_name: str,
  ) -> bool:
    """Whether IntentResolver's stream preview names ``file_name`` exactly."""
    if not file_name or file_name.rsplit("/", 1)[-1] != file_name:
      return False
    pattern = re.compile(
        rf"(?<![\w.-]){re.escape(file_name)}(?![\w.-])"
    )
    for element in ui_elements:
      package = (getattr(element, "package_name", None) or "").casefold()
      if package not in cls._CHOOSER_PACKAGES:
        continue
      resource = (
          getattr(element, "resource_id", None)
          or getattr(element, "resource_name", None)
          or ""
      )
      resource_basename = resource.rsplit("/", 1)[-1].casefold()
      if resource_basename not in cls._FILE_PREVIEW_RESOURCE_IDS:
        continue
      for field in (element.text, element.content_description):
        if field and pattern.search(urllib_parse.unquote(field)):
          return True
    return False

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    _seed_file(env, f"{_ROOT}/Docs/{self._params['file_name']}", "payload")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui = env.get_state().ui_elements
    return (
        1.0
        if self._chooser_preview_matches_file(ui, self._params["file_name"])
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"file_name": _random_name("share")}


class _FilesMoveFolderBase(_FilesTaskBase):
  """Move a seeded folder (with one inner file) into ``Archive/``."""

  complexity = 2.4
  schema = {
      "type": "object",
      "properties": {"folder_name": {"type": "string"}},
      "required": ["folder_name"],
  }

  def _seed_state(self, env: interface.AsyncEnv) -> None:
    folder = f"{_ROOT}/Docs/{self._params['folder_name']}"
    _seed_file(env, f"{folder}/inside.txt", "content")
    _seed_dir(env, f"{_ROOT}/Archive")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    src = f"{_ROOT}/Docs/{self._params['folder_name']}"
    dst = f"{_ROOT}/Archive/{self._params['folder_name']}/inside.txt"
    return 1.0 if _path_exists(env, dst) and not _path_exists(env, src) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"folder_name": _random_name("Project", ext="")}


# -----------------------------------------------------------------------------
# Per-app packages and generated ports.
# -----------------------------------------------------------------------------

_MATERIAL_FILES_PACKAGE: Final[str] = "me.zhanghai.android.files"
_AMAZE_PACKAGE: Final[str] = "com.amaze.filemanager"
_FOSSIFY_FM_PACKAGE: Final[str] = "org.fossify.filemanager"
_TOTAL_COMMANDER_PACKAGE: Final[str] = "com.ghisler.android.TotalCommander"
_XPLORE_PACKAGE: Final[str] = "com.lonelycatgames.Xplore"


_APP_DISPLAY_NAMES: Final[dict[str, str]] = {
    _MATERIAL_FILES_PACKAGE: "Material Files",
    _AMAZE_PACKAGE: "Amaze File Manager",
    _FOSSIFY_FM_PACKAGE: "Fossify File Manager",
    _TOTAL_COMMANDER_PACKAGE: "Total Commander",
    _XPLORE_PACKAGE: "X-plore File Manager",
}


def _class_suffix(display_name: str) -> str:
  return "".join(ch for ch in display_name if ch.isalnum())


_TEMPLATES: Final[dict[type, str]] = {
    _FilesDeleteFileBase: (
        "Using the {app} app, delete the file `{{file_name}}' from the"
        " `CATBench/Docs' folder. Leave every other file unchanged."
    ),
    _FilesMoveFileBase: (
        "Using the {app} app, move the file `{{file_name}}' from"
        " `CATBench/Docs' to `CATBench/Archive'."
    ),
    _FilesCopyFileBase: (
        "Using the {app} app, copy the file `{{file_name}}' from"
        " `CATBench/Docs' to `CATBench/Archive'. Keep the original."
    ),
    _FilesRenameFileBase: (
        "Using the {app} app, rename `CATBench/Docs/{{old_name}}' to"
        " `CATBench/Docs/{{new_name}}'."
    ),
    _FilesCreateFolderBase: (
        "Using the {app} app, create a new folder named `{{folder_name}}'"
        " inside the `CATBench' directory."
    ),
    _FilesDeleteFolderBase: (
        "Using the {app} app, delete the folder `{{folder_name}}' (inside"
        " `CATBench') including every file inside it."
    ),
    _FilesSaveCopyOfReceiptBase: (
        "Using the {app} app, copy `CATBench/Inbox/{{file_name}}' into the"
        " `CATBench/Receipts' folder."
    ),
    _FilesDeleteAllInFolderBase: (
        "Using the {app} app, remove every file inside"
        " `CATBench/{{folder_name}}' but keep the folder itself."
    ),
    _FilesCreateNestedFolderBase: (
        "Using the {app} app, create the nested directory"
        " `CATBench/{{parent}}/{{child}}' (parent then child)."
    ),
    _FilesMoveFolderBase: (
        "Using the {app} app, move the folder `CATBench/Docs/{{folder_name}}'"
        " (and its contents) into `CATBench/Archive/'."
    ),
    _FilesSearchFileBase: (
        "Using the {app} app, locate the file named `{{needle}}' somewhere"
        " beneath `CATBench/Docs' and rename it to `found.txt'. Leave every"
        " other file unchanged."
    ),
    _FilesCompressFilesBase: (
        "Using the {app} app, compress every file inside"
        " `CATBench/ToCompress/' into a single ZIP archive and"
        " save the archive somewhere beneath `CATBench/'. Keep the original"
        " files unchanged."
    ),
    _FilesExtractArchiveBase: (
        "Using the {app} app, extract the archive"
        " `CATBench/Archives/{{archive_name}}' so that"
        f" `{_ZIP_INNER_NAME}' appears somewhere beneath `CATBench/'. Leave"
        " the source archive unchanged."
    ),
    _FilesViewFileInfoBase: (
        "Using the {app} app, open the details / properties / info screen"
        " for `CATBench/Docs/{{file_name}}'."
    ),
    _FilesShareFileBase: (
        "Using the {app} app, share the file"
        " `CATBench/Docs/{{file_name}}' and leave the Android share sheet"
        " open."
    ),
}


# Cross-app File-Manager task templates. The 10 short names below ARE the
# user's target task list for the Files category in hybrid mode. Every task
# that mutates shared storage verifies via adb file presence on /sdcard.
# ``ViewFileInfo`` and ``ShareFile`` are explicitly UI-text heuristics because
# they do not leave a filesystem-visible state change.
_BASE_SHORT_NAMES: Final[dict[type, str]] = {
    _FilesCreateFolderBase: "FilesCreateFolder",
    _FilesRenameFileBase: "FilesRenameFile",
    _FilesDeleteFileBase: "FilesDeleteFile",
    _FilesMoveFileBase: "FilesMoveFile",
    _FilesSaveCopyOfReceiptBase: "FilesSaveCopyOfFile",
    _FilesSearchFileBase: "FilesSearchFile",
    _FilesCompressFilesBase: "FilesCompressFiles",
    _FilesExtractArchiveBase: "FilesExtractArchive",
    _FilesViewFileInfoBase: "FilesViewFileInfo",
    _FilesShareFileBase: "FilesShareFile",
}


_PACKAGES = (
    _MATERIAL_FILES_PACKAGE,
    _AMAZE_PACKAGE,
    _FOSSIFY_FM_PACKAGE,
    _TOTAL_COMMANDER_PACKAGE,
    _XPLORE_PACKAGE,
)


for _base_cls, _short in _BASE_SHORT_NAMES.items():
  excluded = getattr(_base_cls, "excluded_packages", ())
  for _pkg in _PACKAGES:
    if _pkg in excluded:
      continue
    _display = _APP_DISPLAY_NAMES[_pkg]
    _suffix = _class_suffix(_display)
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
