"""Deterministic conformance tests for the frozen Files task adapters."""

import base64
import hashlib
from unittest import mock

from absl.testing import absltest

from android_world.env import representation_utils
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base as cross_app_base,
)
from android_world.task_evals.single.app_generalization_generated import (
    files_cross_app_tasks as files_tasks,
)


class _InMemoryState:

  def __init__(self, ui_elements=()):
    self.ui_elements = list(ui_elements)


class _InMemoryEnv:

  def __init__(self, ui_elements=()):
    self._state = _InMemoryState(ui_elements)

  def get_state(self):
    return self._state


def _element(
    text: str | None = None,
    *,
    content_description: str | None = None,
    package_name: str = "me.zhanghai.android.files",
    resource_id: str | None = None,
) -> representation_utils.UIElement:
  return representation_utils.UIElement(
      text=text,
      content_description=content_description,
      package_name=package_name,
      resource_id=resource_id,
  )


def _score(task_cls, params, ui_elements=()) -> float:
  task = task_cls(params)
  task.initialized = True
  return task.is_successful(_InMemoryEnv(ui_elements))


class FilesFrozenVerifierConformanceTest(absltest.TestCase):

  def test_path_exists_accepts_only_exact_yes_response(self):
    with mock.patch.object(files_tasks, "_adb_shell", return_value="YES\n"):
      self.assertTrue(files_tasks._path_exists(object(), "/sdcard/CATBench"))
    with mock.patch.object(
        files_tasks, "_adb_shell", return_value="NO (diagnostic: YES unavailable)"
    ):
      self.assertFalse(files_tasks._path_exists(object(), "/sdcard/CATBench"))

  def test_initialize_resets_seed_root_before_seeding_and_refresh(self):
    events = []
    task = files_tasks.FilesRenameFileForMaterialFiles(
        {"old_name": "draft_acl.txt", "new_name": "final_acl.txt"}
    )
    with mock.patch.object(
        cross_app_base.PackageAppEval,
        "initialize_task",
        side_effect=lambda unused_env: events.append("launch"),
    ), mock.patch.object(
        task, "_cleanup", side_effect=lambda unused_env: events.append("reset")
    ), mock.patch.object(
        task, "_seed_state", side_effect=lambda unused_env: events.append("seed")
    ), mock.patch.object(
        task,
        "_refresh_after_seeding",
        side_effect=lambda unused_env: events.append("refresh"),
    ):
      task.initialize_task(object())

    self.assertEqual(events, ["launch", "reset", "seed", "refresh"])

  def test_refresh_detaches_media_scan_before_relaunch(self):
    task = files_tasks.FilesRenameFileForMaterialFiles(
        {"old_name": "draft_acl.txt", "new_name": "final_acl.txt"}
    )
    commands = []
    env = mock.Mock(controller=object())
    with mock.patch.object(
        files_tasks,
        "_adb_shell",
        side_effect=lambda unused_env, command: commands.append(command) or "",
    ), mock.patch.object(files_tasks.adb_utils, "launch_app"):
      task._refresh_after_seeding(env)

    self.assertIn("MEDIA_SCANNER_SCAN_FILE", commands[0])
    self.assertIn("</dev/null >/dev/null 2>&1 &", commands[0])
    self.assertIn("am force-stop", commands[1])

  def test_missing_file_content_is_a_normal_negative_not_adb_failure(self):
    commands = []
    with mock.patch.object(
        files_tasks,
        "_adb_shell",
        side_effect=lambda unused_env, command: commands.append(command) or "",
    ):
      self.assertFalse(
          files_tasks._file_has_content(
              object(), "/sdcard/CATBench/missing.txt", "expected"
          )
      )

    self.assertLen(commands, 1)
    self.assertIn("[ ! -L ", commands[0])
    self.assertTrue(commands[0].endswith("2>/dev/null || true"))

  def test_file_content_comparison_is_byte_exact(self):
    with mock.patch.object(files_tasks, "_adb_shell", return_value="expected\n"):
      self.assertTrue(
          files_tasks._file_has_content(
              object(), "/sdcard/CATBench/file.txt", "expected"
          )
      )
    with mock.patch.object(
        files_tasks, "_adb_shell", return_value="expected\r\n"
    ):
      self.assertFalse(
          files_tasks._file_has_content(
              object(), "/sdcard/CATBench/file.txt", "expected"
          )
      )

  def test_seed_zip_hash_constant_matches_exact_archive_bytes(self):
    archive = base64.b64decode(files_tasks._ZIP_B64, validate=True)
    self.assertEqual(
        hashlib.sha256(archive).hexdigest(), files_tasks._ZIP_SHA256
    )

  def test_archive_reader_fails_closed_for_unsupported_7z(self):
    expected = {"file_0.txt": "file_0.txt"}
    with mock.patch.object(files_tasks, "_adb_shell") as adb_shell:
      self.assertFalse(
          files_tasks._archive_has_exact_members(
              object(), "/sdcard/CATBench/output.7z", expected
          )
      )
    adb_shell.assert_not_called()

  def test_create_folder_requires_directory_not_same_named_file(self):
    params = {"folder_name": "ACLRevision"}
    with mock.patch.object(files_tasks, "_is_directory", return_value=True):
      self.assertEqual(
          _score(files_tasks.FilesCreateFolderForMaterialFiles, params), 1.0
      )
    with mock.patch.object(files_tasks, "_is_directory", return_value=False):
      self.assertEqual(
          _score(files_tasks.FilesCreateFolderForMaterialFiles, params), 0.0
      )

  def test_rename_requires_preserved_payload_and_old_path_gone(self):
    params = {"old_name": "draft_acl.txt", "new_name": "final_acl.txt"}
    with mock.patch.object(
        files_tasks, "_file_has_content", return_value=True
    ), mock.patch.object(files_tasks, "_path_exists", return_value=False):
      self.assertEqual(
          _score(files_tasks.FilesRenameFileForMaterialFiles, params), 1.0
      )
    with mock.patch.object(
        files_tasks, "_file_has_content", return_value=False
    ), mock.patch.object(files_tasks, "_path_exists", return_value=False):
      self.assertEqual(
          _score(files_tasks.FilesRenameFileForMaterialFiles, params), 0.0
      )
    with mock.patch.object(
        files_tasks, "_file_has_content", return_value=True
    ), mock.patch.object(files_tasks, "_path_exists", return_value=True):
      self.assertEqual(
          _score(files_tasks.FilesRenameFileForMaterialFiles, params), 0.0
      )

  def test_delete_requires_path_and_unique_marker_removed_from_tree(self):
    params = {"file_name": "note_acl.txt"}
    with mock.patch.object(
        files_tasks, "_path_exists", return_value=False
    ), mock.patch.object(
        files_tasks, "_find_any_file_with_content", return_value=False
    ), mock.patch.object(files_tasks, "_file_has_content", return_value=True):
      self.assertEqual(
          _score(files_tasks.FilesDeleteFileForMaterialFiles, params), 1.0
      )

    # A renamed or moved target has left its unique marker under CATBench and
    # is not normalized deletion even though the original path is absent.
    with mock.patch.object(
        files_tasks, "_path_exists", return_value=False
    ), mock.patch.object(
        files_tasks, "_find_any_file_with_content", return_value=True
    ), mock.patch.object(files_tasks, "_file_has_content", return_value=True):
      self.assertEqual(
          _score(files_tasks.FilesDeleteFileForMaterialFiles, params), 0.0
      )

    # The unchanged original is a no-op and must fail independently of the
    # marker scan.
    with mock.patch.object(
        files_tasks, "_path_exists", return_value=True
    ), mock.patch.object(
        files_tasks, "_find_any_file_with_content", return_value=True
    ), mock.patch.object(files_tasks, "_file_has_content", return_value=True):
      self.assertEqual(
          _score(files_tasks.FilesDeleteFileForMaterialFiles, params), 0.0
      )

  def test_delete_rejects_wrong_content_decoy(self):
    params = {"file_name": "note_acl.txt"}
    with mock.patch.object(
        files_tasks, "_path_exists", return_value=False
    ), mock.patch.object(
        files_tasks, "_find_any_file_with_content", return_value=False
    ), mock.patch.object(files_tasks, "_file_has_content", return_value=False):
      self.assertEqual(
          _score(files_tasks.FilesDeleteFileForMaterialFiles, params), 0.0
      )

  def test_delete_rejects_wrong_type_decoy(self):
    params = {"file_name": "note_acl.txt"}
    # `_file_has_content` starts with `[ -f ... ]`, so a directory or symlinked
    # non-regular decoy fails the same exact-file predicate as wrong bytes.
    with mock.patch.object(
        files_tasks, "_path_exists", return_value=False
    ), mock.patch.object(
        files_tasks, "_find_any_file_with_content", return_value=False
    ), mock.patch.object(files_tasks, "_file_has_content", return_value=False):
      self.assertEqual(
          _score(files_tasks.FilesDeleteFileForMaterialFiles, params), 0.0
      )

  def test_delete_seeds_target_specific_marker_and_exact_decoy(self):
    params = {"file_name": "note_acl.txt"}
    task = files_tasks.FilesDeleteFileForMaterialFiles(params)
    seeded = []
    with mock.patch.object(
        files_tasks,
        "_seed_file",
        side_effect=lambda unused_env, path, content: seeded.append(
            (path, content)
        ),
    ):
      task._seed_state(object())
    self.assertEqual(
        seeded,
        [
            (
                "/sdcard/CATBench/Docs/note_acl.txt",
                "CATBENCH_DELETE_TARGET:note_acl.txt",
            ),
            ("/sdcard/CATBench/Docs/catbench_keep.txt", "keep"),
        ],
    )

  def test_move_rejects_empty_destination_placeholder(self):
    params = {"file_name": "invoice_acl.txt"}
    with mock.patch.object(
        files_tasks, "_file_has_content", return_value=True
    ), mock.patch.object(files_tasks, "_path_exists", return_value=False):
      self.assertEqual(
          _score(files_tasks.FilesMoveFileForMaterialFiles, params), 1.0
      )
    with mock.patch.object(
        files_tasks, "_file_has_content", return_value=False
    ), mock.patch.object(files_tasks, "_path_exists", return_value=False):
      self.assertEqual(
          _score(files_tasks.FilesMoveFileForMaterialFiles, params), 0.0
      )

  def test_save_copy_requires_source_and_destination_payloads(self):
    params = {"file_name": "receipt_acl.pdf"}
    with mock.patch.object(
        files_tasks, "_file_has_content", side_effect=[True, True]
    ):
      self.assertEqual(
          _score(files_tasks.FilesSaveCopyOfFileForMaterialFiles, params), 1.0
      )
    with mock.patch.object(
        files_tasks, "_file_has_content", side_effect=[False, True]
    ):
      self.assertEqual(
          _score(files_tasks.FilesSaveCopyOfFileForMaterialFiles, params), 0.0
      )
    with mock.patch.object(
        files_tasks, "_file_has_content", side_effect=[True, False]
    ):
      self.assertEqual(
          _score(files_tasks.FilesSaveCopyOfFileForMaterialFiles, params), 0.0
      )

  def test_locate_rename_requires_exact_payload_original_gone_and_decoys(self):
    params = {"needle": "target_acl.txt"}
    with mock.patch.object(
        files_tasks, "_find_exactly_one_file_with_content", return_value=True
    ), mock.patch.object(
        files_tasks, "_find_name", return_value=False
    ), mock.patch.object(files_tasks, "_file_has_content", return_value=True):
      self.assertEqual(
          _score(files_tasks.FilesSearchFileForMaterialFiles, params), 1.0
      )

    # The original basename surviving in any nested location is not a rename.
    with mock.patch.object(
        files_tasks, "_find_exactly_one_file_with_content", return_value=True
    ), mock.patch.object(
        files_tasks, "_find_name", return_value=True
    ), mock.patch.object(files_tasks, "_file_has_content", return_value=True):
      self.assertEqual(
          _score(files_tasks.FilesSearchFileForMaterialFiles, params), 0.0
      )

    # A same-named output with wrong bytes is not the target file.
    with mock.patch.object(
        files_tasks, "_find_exactly_one_file_with_content", return_value=False
    ), mock.patch.object(
        files_tasks, "_find_name", return_value=False
    ), mock.patch.object(files_tasks, "_file_has_content", return_value=True):
      self.assertEqual(
          _score(files_tasks.FilesSearchFileForMaterialFiles, params), 0.0
      )

  def test_locate_rename_rejects_missing_or_changed_decoy(self):
    params = {"needle": "target_acl.txt"}

    def decoy_state(unused_env, path, unused_content):
      return not path.endswith("decoy_3.txt")

    with mock.patch.object(
        files_tasks, "_find_exactly_one_file_with_content", return_value=True
    ), mock.patch.object(
        files_tasks, "_find_name", return_value=False
    ), mock.patch.object(
        files_tasks, "_file_has_content", side_effect=decoy_state
    ):
      self.assertEqual(
          _score(files_tasks.FilesSearchFileForMaterialFiles, params), 0.0
      )

  def test_locate_target_is_nested_and_instruction_is_outcome_verifiable(self):
    params = {"needle": "target_acl.txt"}
    task = files_tasks.FilesSearchFileForMaterialFiles(params)
    seeded = []
    with mock.patch.object(
        files_tasks,
        "_seed_file",
        side_effect=lambda unused_env, path, content: seeded.append(
            (path, content)
        ),
    ):
      task._seed_state(object())
    target_path, target_content = seeded[-1]
    self.assertEqual(
        target_path,
        "/sdcard/CATBench/Docs/Reference/Incoming/target_acl.txt",
    )
    self.assertEqual(
        target_content, "CATBENCH_LOCATE_TARGET:target_acl.txt"
    )
    self.assertNotIn("use the in-app file search", task.template)
    self.assertIn("locate the file", task.template)
    self.assertIn("Leave every other file unchanged", task.template)

  def test_found_file_must_be_unique(self):
    with mock.patch.object(
        files_tasks,
        "_find_files_named",
        return_value=(
            "/sdcard/CATBench/Docs/found.txt",
            "/sdcard/CATBench/Docs/Elsewhere/found.txt",
        ),
    ), mock.patch.object(files_tasks, "_file_has_content", return_value=True):
      self.assertFalse(
          files_tasks._find_exactly_one_file_with_content(
              object(), "found.txt", "target marker"
          )
      )

  def test_zip_archive_members_and_payloads_are_exact(self):
    expected = {
        name: name for name in files_tasks._FilesCompressFilesBase._SEED_NAMES
    }
    listing = (
        "       11  2026-07-11 10:00 nested/file_0.txt\n"
        "       11  2026-07-11 10:00 nested/file_1.txt\n"
        "       11  2026-07-11 10:00 nested/file_2.txt\n"
        "---------                     -------\n"
        "       33                     3 files\n"
    )
    commands = []

    def zip_shell(unused_env, command):
      commands.append(command)
      if command.startswith("unzip -lq "):
        return listing
      for name in expected:
        if command.startswith("unzip -p ") and f"nested/{name}" in command:
          return f"{name}\n"
      return ""

    with mock.patch.object(files_tasks, "_adb_shell", side_effect=zip_shell):
      self.assertTrue(
          files_tasks._archive_has_exact_members(
              object(), "/sdcard/CATBench/output.zip", expected
          )
      )
    self.assertTrue(commands[0].startswith("unzip -lq "))
    self.assertLen([cmd for cmd in commands if cmd.startswith("unzip -p ")], 3)

  def test_tar_archive_members_and_payloads_are_exact(self):
    expected = {
        name: name for name in files_tasks._FilesCompressFilesBase._SEED_NAMES
    }
    listing = (
        "nested/file_0.txt\n"
        "nested/file_1.txt\n"
        "nested/file_2.txt\n"
    )
    commands = []

    def tar_shell(unused_env, command):
      commands.append(command)
      if command.startswith("tar -tf "):
        return listing
      for name in expected:
        if command.startswith("tar -xOf ") and f"nested/{name}" in command:
          return f"{name}\n"
      return ""

    with mock.patch.object(files_tasks, "_adb_shell", side_effect=tar_shell):
      self.assertTrue(
          files_tasks._archive_has_exact_members(
              object(), "/sdcard/CATBench/output.tar", expected
          )
      )
    self.assertTrue(commands[0].startswith("tar -tf "))
    self.assertLen([cmd for cmd in commands if cmd.startswith("tar -xOf ")], 3)

  def test_archive_rejects_partial_member_set(self):
    expected = {
        name: name for name in files_tasks._FilesCompressFilesBase._SEED_NAMES
    }
    with mock.patch.object(
        files_tasks,
        "_adb_shell",
        return_value="file_0.txt\nfile_1.txt\n",
    ):
      self.assertFalse(
          files_tasks._archive_has_exact_members(
              object(), "/sdcard/CATBench/partial.tar", expected
          )
      )

  def test_archive_rejects_wrong_member_content(self):
    expected = {
        name: name for name in files_tasks._FilesCompressFilesBase._SEED_NAMES
    }
    listing = (
        "file_0.txt\n"
        "file_1.txt\n"
        "file_2.txt\n"
    )

    def tar_shell(unused_env, command):
      if command.startswith("tar -tf "):
        return listing
      if command.startswith("tar -xOf ") and "file_1.txt" in command:
        return "wrong-content\n"
      for name in expected:
        if command.startswith("tar -xOf ") and name in command:
          return f"{name}\n"
      return ""

    with mock.patch.object(files_tasks, "_adb_shell", side_effect=tar_shell):
      self.assertFalse(
          files_tasks._archive_has_exact_members(
              object(), "/sdcard/CATBench/output.tar", expected
          )
      )

  def test_compress_requires_one_archive_and_preserved_sources(self):
    archive_path = "/sdcard/CATBench/output.zip\n"
    with mock.patch.object(
        files_tasks, "_adb_shell", return_value=archive_path
    ), mock.patch.object(
        files_tasks, "_file_has_content", return_value=True
    ), mock.patch.object(
        files_tasks, "_archive_has_exact_members", return_value=True
    ) as archive_reader:
      self.assertEqual(
          _score(files_tasks.FilesCompressFilesForMaterialFiles, {}), 1.0
      )
    archive_reader.assert_called_once()

    duplicate_archives = (
        "/sdcard/CATBench/partial.tar\n"
        "/sdcard/CATBench/output.zip\n"
    )
    with mock.patch.object(
        files_tasks, "_adb_shell", return_value=duplicate_archives
    ), mock.patch.object(files_tasks, "_file_has_content") as source_reader:
      self.assertEqual(
          _score(files_tasks.FilesCompressFilesForMaterialFiles, {}), 0.0
      )
    source_reader.assert_not_called()

    with mock.patch.object(
        files_tasks, "_adb_shell", return_value=archive_path
    ), mock.patch.object(
        files_tasks, "_file_has_content", side_effect=[True, False]
    ), mock.patch.object(
        files_tasks, "_archive_has_exact_members"
    ) as archive_reader:
      self.assertEqual(
          _score(files_tasks.FilesCompressFilesForMaterialFiles, {}), 0.0
      )
    archive_reader.assert_not_called()

  def test_compress_ignores_non_zip_archives(self):
    with mock.patch.object(
        files_tasks,
        "_adb_shell",
        return_value="",
    ) as adb_shell, mock.patch.object(
        files_tasks, "_file_has_content"
    ) as source_reader, mock.patch.object(
        files_tasks, "_archive_has_exact_members"
    ) as archive_reader:
      self.assertEqual(
          _score(files_tasks.FilesCompressFilesForMaterialFiles, {}), 0.0
      )
    source_reader.assert_not_called()
    archive_reader.assert_not_called()
    find_command = adb_shell.call_args.args[1]
    self.assertIn("*.zip", find_command)
    self.assertNotIn("*.7z", find_command)
    self.assertNotIn("*.tar", find_command)

    self.assertIn(
        "single ZIP archive",
        files_tasks.FilesCompressFilesForMaterialFiles.template,
    )

  def test_extract_requires_payload_and_exact_source_archive_sha(self):
    params = {"archive_name": "bundle_acl.zip"}
    with mock.patch.object(
        files_tasks, "_file_sha256", return_value=files_tasks._ZIP_SHA256
    ), mock.patch.object(
        files_tasks, "_find_file_with_content", return_value=True
    ):
      self.assertEqual(
          _score(files_tasks.FilesExtractArchiveForMaterialFiles, params), 1.0
      )
    with mock.patch.object(
        files_tasks, "_file_sha256", return_value=files_tasks._ZIP_SHA256
    ), mock.patch.object(
        files_tasks, "_find_file_with_content", return_value=False
    ):
      self.assertEqual(
          _score(files_tasks.FilesExtractArchiveForMaterialFiles, params), 0.0
      )
    with mock.patch.object(
        files_tasks, "_file_sha256", return_value="0" * 64
    ), mock.patch.object(
        files_tasks, "_find_file_with_content", return_value=True
    ):
      self.assertEqual(
          _score(files_tasks.FilesExtractArchiveForMaterialFiles, params), 0.0
      )

  def test_view_info_rejects_normal_listing_metadata_and_wrong_file(self):
    params = {"file_name": "details_acl.txt"}
    # The target row plus its unopened context-menu command is not an opened
    # information panel.
    self.assertEqual(
        _score(
            files_tasks.FilesViewFileInfoForMaterialFiles,
            params,
            [_element("details_acl.txt"), _element("Properties")],
        ),
        0.0,
    )
    # A terminal information surface binds the exact file to multiple concrete
    # metadata fields.
    self.assertEqual(
        _score(
            files_tasks.FilesViewFileInfoForMaterialFiles,
            params,
            [
                _element("details_acl.txt"),
                _element("Properties"),
                _element("Size"),
                _element("Modified"),
            ],
        ),
        1.0,
    )
    self.assertEqual(
        _score(
            files_tasks.FilesViewFileInfoForMaterialFiles,
            params,
            [_element("other_file.txt"), _element("Properties")],
        ),
        0.0,
    )
    self.assertEqual(
        _score(
            files_tasks.FilesViewFileInfoForMaterialFiles,
            params,
            [
                _element("details_acl.txt.bak"),
                _element("Properties"),
                _element("Size"),
                _element("Modified"),
            ],
        ),
        0.0,
    )

  def test_share_requires_exact_file_in_system_chooser_payload_preview(self):
    params = {"file_name": "share_acl.txt"}
    chooser_package = "com.android.intentresolver"
    preview_id = "com.android.intentresolver:id/content_preview_filename"

    # Source-app toolbar text is not a chooser.
    self.assertEqual(
        _score(
            files_tasks.FilesShareFileForMaterialFiles,
            params,
            [_element("Share")],
        ),
        0.0,
    )
    # A real chooser target alone does not say which file is being shared.
    self.assertEqual(
        _score(
            files_tasks.FilesShareFileForMaterialFiles,
            params,
            [
                _element(
                    "Messages", package_name=chooser_package
                )
            ],
        ),
        0.0,
    )
    # The requested file left visible in the source app cannot bind a chooser
    # that previews a different stream.
    self.assertEqual(
        _score(
            files_tasks.FilesShareFileForMaterialFiles,
            params,
            [
                _element("share_acl.txt"),
                _element("Messages", package_name=chooser_package),
                _element(
                    "other_file.txt",
                    package_name=chooser_package,
                    resource_id=preview_id,
                ),
            ],
        ),
        0.0,
    )
    # Substring collisions are not the seeded file identity.
    self.assertEqual(
        _score(
            files_tasks.FilesShareFileForMaterialFiles,
            params,
            [
                _element(
                    "other_share_acl.txt",
                    package_name=chooser_package,
                    resource_id=preview_id,
                )
            ],
        ),
        0.0,
    )
    self.assertEqual(
        _score(
            files_tasks.FilesShareFileForMaterialFiles,
            params,
            [
                _element(
                    "share_acl.txt",
                    package_name=chooser_package,
                    resource_id=preview_id,
                )
            ],
        ),
        1.0,
    )


if __name__ == "__main__":
  absltest.main()
