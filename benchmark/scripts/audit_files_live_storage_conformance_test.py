from __future__ import annotations

import json
import subprocess

import audit_files_live_storage_conformance as audit
import pytest


def test_discovers_exact_40_real_storage_adapters() -> None:
  classes = audit.discover_task_classes()
  assert len(classes) == 40
  assert set(classes) == {
      (semantic, package)
      for package in audit.files_tasks._PACKAGES
      for semantic in audit.SCOPED_SEMANTICS
  }
  assert all(
      task_class.__name__.startswith(semantic)
      for (semantic, _), task_class in classes.items()
  )


def test_scope_partitions_frozen_ten_semantics() -> None:
  assert len(audit.SCOPED_SEMANTICS) == 8
  assert len(audit.EXCLUDED_SEMANTICS) == 2
  assert not (set(audit.SCOPED_SEMANTICS) & set(audit.EXCLUDED_SEMANTICS))
  assert set(audit.FIXED_PARAMS) == set(audit.SCOPED_SEMANTICS)
  assert set(audit.MUTATIONS) == set(audit.SCOPED_SEMANTICS)
  assert set(audit.CASE_SPECS) == set(audit.SCOPED_SEMANTICS)
  assert sum(1 + len(rows) for rows in audit.CASE_SPECS.values()) == 37
  assert 5 * sum(1 + len(rows) for rows in audit.CASE_SPECS.values()) == 185


def test_focus_observation_uses_api33_window_and_activity_fields() -> None:
  live = object.__new__(audit.LiveAudit)

  def shell(command: str, **unused_kwargs) -> str:
    if command.startswith("dumpsys window |"):
      return (
          "mCurrentFocus=Window{1 u0 "
          "me.zhanghai.android.files/.filelist.FileListActivity}"
      )
    if command.startswith("dumpsys activity activities |"):
      return (
          "topResumedActivity=ActivityRecord{2 u0 "
          "me.zhanghai.android.files/.filelist.FileListActivity}"
      )
    if command.startswith("pidof "):
      return "1234"
    raise AssertionError(command)

  live.shell = shell
  observation = live.focus_observation("me.zhanghai.android.files")
  assert observation["package_focused"] is True
  assert observation["package_process_running"] is True
  assert observation["pidof"] == ["1234"]


def test_wait_for_focus_records_delayed_success_without_fixed_sleep() -> None:
  live = object.__new__(audit.LiveAudit)
  observations = iter((
      {
          "package_focused": False,
          "package_process_running": True,
      },
      {
          "package_focused": True,
          "package_process_running": True,
      },
  ))
  live.focus_observation = lambda *args, **kwargs: next(observations)
  result = live.wait_for_focus(
      "me.zhanghai.android.files", timeout_seconds=1, poll_seconds=0
  )
  assert result["valid"] is True
  assert result["poll_count"] == 2
  assert result["elapsed_seconds"] <= 1


def test_runtime_bindings_reject_unbound_cli_labels(tmp_path) -> None:
  cohort_path = tmp_path / "cohort.json"
  request_path = tmp_path / "request.json"
  receipt_path = tmp_path / "receipt.json"
  attestation_path = tmp_path / "apps.json"
  cohort_path.write_text('{"release_id":"release"}\n', encoding="utf-8")
  cohort_sha256 = audit._sha256(cohort_path)
  request = {
      "operation": "clone_activate",
      "analysis_eligible": False,
      "cohort_sha256": cohort_sha256,
      "base_snapshot_id": "base",
      "base_snapshot_sha256": "a" * 64,
      "snapshot_clone_id": "clone",
      "device_serial": "emulator-5576",
  }
  request_path.write_text(json.dumps(request), encoding="utf-8")
  receipt = {
      "operation": "clone_activate",
      "success": True,
      "request_sha256": audit._sha256(request_path),
      "active_snapshot_clone_id": "clone",
      "active_snapshot_sha256": "a" * 64,
      "base_snapshot_id": "base",
      "base_snapshot_sha256": "a" * 64,
      "device_serial": "emulator-5576",
      "emulator_image": "image@sha256:" + "b" * 64,
      "active_worker_name": "catbench-docker-emu-0",
      "pool_manager_sha256": audit.EXPECTED_POOL_MANAGER_SHA256,
      "emulator_start_script_sha256": audit.EXPECTED_START_SCRIPT_SHA256,
  }
  receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
  app_rows = [
      {
          "category": "files",
          "app_id": app_id,
          "package_name": package,
          "version_code": "1",
          "version_name": "1.0",
          "installed_apk_sha256": ["c" * 64],
          "valid": True,
      }
      for app_id, package in zip(
          audit.FILES_APP_IDS, audit.files_tasks._PACKAGES, strict=True
      )
  ]
  app_rows.extend({
      "category": "other",
      "app_id": f"other_{index}",
      "package_name": f"org.example.other{index}",
      "valid": True,
  } for index in range(18))
  attestation_path.write_text(
      json.dumps({
          "valid": True,
          "evidence_type": "catbench_live_device_app_provision_and_attestation",
          "mode": "attest",
          "release_id": "release",
          "cohort_sha256": cohort_sha256,
          "expected_apps": 23,
          "valid_apps": 23,
          "invalid_apps": 0,
          "device": {
              "serial": "emulator-5576",
              "adb_server_port": 5051,
              "api_level": "33",
              "build_fingerprint": audit.EXPECTED_BUILD_FINGERPRINT,
              "boot_id": "boot-id",
          },
          "apps": app_rows,
      }),
      encoding="utf-8",
  )

  with pytest.raises(ValueError, match="request clone id mismatch"):
    audit.validate_runtime_bindings(
        cohort_path=cohort_path,
        clone_request_path=request_path,
        clone_receipt_path=receipt_path,
        app_attestation_path=attestation_path,
        docker_image_digest="image@sha256:" + "b" * 64,
        base_snapshot_id="base",
        base_snapshot_sha256="a" * 64,
        snapshot_clone_id="different-clone",
        serial="emulator-5576",
        worker_index=0,
        console_port=5576,
        grpc_port=8576,
        adb_server_port=5051,
        first_console_port=5576,
        first_grpc_port=8576,
        first_adb_server_port=5051,
    )

  evidence, rows = audit.validate_runtime_bindings(
      cohort_path=cohort_path,
      clone_request_path=request_path,
      clone_receipt_path=receipt_path,
      app_attestation_path=attestation_path,
      docker_image_digest="image@sha256:" + "b" * 64,
      base_snapshot_id="base",
      base_snapshot_sha256="a" * 64,
      snapshot_clone_id="clone",
      serial="emulator-5576",
      worker_index=0,
      console_port=5576,
      grpc_port=8576,
      adb_server_port=5051,
      first_console_port=5576,
      first_grpc_port=8576,
      first_adb_server_port=5051,
  )
  assert evidence["clone_activate_receipt"]["sha256"] == audit._sha256(
      receipt_path
  )
  assert len(rows) == 5


def test_active_worker_container_binds_volume_image_clone_and_ports(
    tmp_path, monkeypatch
) -> None:
  receipt_path = tmp_path / "receipt.json"
  receipt_path.write_text(json.dumps({
      "active_worker_name": "catbench-docker-emu-0",
      "active_volume_name": "catbench-episode-avd-r3",
      "emulator_image_id": "sha256:" + "a" * 64,
  }), encoding="utf-8")
  record = {
      "Id": "container-id",
      "Created": "created",
      "Name": "/catbench-docker-emu-0",
      "Image": "sha256:" + "a" * 64,
      "State": {"Running": True, "StartedAt": "started"},
      "Mounts": [{
          "Type": "volume",
          "Name": "catbench-episode-avd-r3",
          "Destination": "/root/.android",
          "RW": True,
      }],
      "HostConfig": {"NetworkMode": "host"},
      "Config": {
          "Image": "image@sha256:" + "b" * 64,
          "Labels": {
              "catbench.avd_volume": "catbench-episode-avd-r3",
              "catbench.snapshot_clone_id": "clone-r3",
              "catbench.worker_index": "0",
              "catbench.image_id": "sha256:" + "a" * 64,
              "catbench.launcher_sha256": audit.EXPECTED_START_SCRIPT_SHA256,
          },
          "Env": [
              "ANDROID_SERIAL=emulator-5576",
              "ANDROID_ADB_SERVER_PORT=5051",
              "ADB_SERVER_PORT=5051",
              "EMULATOR_CONSOLE_PORT=5576",
              "EMULATOR_GRPC_PORT=8576",
          ],
      },
  }
  monkeypatch.setattr(
      audit.subprocess,
      "run",
      lambda *args, **kwargs: subprocess.CompletedProcess(
          args=args[0], returncode=0, stdout=json.dumps([record]), stderr=""
      ),
  )
  binding = audit.validate_active_worker_container(
      docker_bin="docker",
      clone_receipt_path=receipt_path,
      docker_image_digest="image@sha256:" + "b" * 64,
      snapshot_clone_id="clone-r3",
      worker_index=0,
      serial="emulator-5576",
      console_port=5576,
      grpc_port=8576,
      adb_server_port=5051,
  )
  assert binding["active_volume_name"] == "catbench-episode-avd-r3"
  assert binding["snapshot_clone_id"] == "clone-r3"


def test_live_device_identity_requires_fresh_attested_boot() -> None:
  observation = {
      "serial": "emulator-5576",
      "boot_id": "fresh-boot",
      "build_fingerprint": audit.EXPECTED_BUILD_FINGERPRINT,
      "api_level": "33",
      "root_adb_uid": "0",
      "boot_completed": "1",
  }
  audit.validate_live_device_observation(
      observation,
      attested_device={"boot_id": "fresh-boot"},
      serial="emulator-5576",
  )
  with pytest.raises(ValueError, match="boot id mismatch"):
    audit.validate_live_device_observation(
        observation,
        attested_device={"boot_id": "stale-boot"},
        serial="emulator-5576",
    )


def test_live_split_apk_hashes_must_match_pin_source() -> None:
  live = object.__new__(audit.LiveAudit)
  responses = {
      "pm path org.example.files": (
          "package:/data/app/org.example.files/base.apk\n"
          "package:/data/app/org.example.files/split.apk"
      ),
      "sha256sum /data/app/org.example.files/base.apk": "a" * 64 + "  base.apk",
      "sha256sum /data/app/org.example.files/split.apk": "b" * 64 + "  split.apk",
  }
  live.shell = lambda command, **unused_kwargs: responses[command]
  rows = live.attest_installed_files_apps([{
      "app_id": "files_example",
      "package_name": "org.example.files",
      "version_code": "1",
      "version_name": "1.0",
      "installed_apk_sha256": ["b" * 64, "a" * 64],
  }])
  assert rows[0]["active_apk_sha256"] == ["a" * 64, "b" * 64]
