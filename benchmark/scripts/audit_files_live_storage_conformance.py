#!/usr/bin/env python3
"""Audit Files' eight durable verifiers against real Android storage.

This deterministic audit initializes the actual per-app task classes on a live
device, checks the initial state, and independently resets and injects several
wrong/partial filesystem states plus exact postconditions. It never calls a
model, does not execute app UI actions, and does not use a gold primitive-action
trajectory or a fresh snapshot per adapter. Consequently it is narrow shared-
predicate fixture evidence for 40 durable task-app adapters, not full Files-
category G3 qualification. ``FilesViewFileInfo`` and ``FilesShareFile`` remain
explicitly out of scope pending live app-action/UI evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

from android_world.env import env_launcher  # noqa: E402
from android_world.task_evals.single.app_generalization_generated import (  # noqa: E402
    files_cross_app_tasks as files_tasks,
)


SCOPED_SEMANTICS = (
    "FilesCreateFolder",
    "FilesRenameFile",
    "FilesDeleteFile",
    "FilesMoveFile",
    "FilesSaveCopyOfFile",
    "FilesSearchFile",
    "FilesCompressFiles",
    "FilesExtractArchive",
)
EXCLUDED_SEMANTICS = ("FilesViewFileInfo", "FilesShareFile")
FILES_APP_IDS = (
    "files_material_files",
    "files_amaze",
    "files_fossify_file_manager",
    "files_total_commander",
    "files_x_plore_file_manager",
)
EXPECTED_BUILD_FINGERPRINT = (
    "google/sdk_gphone64_x86_64/emu64x:13/TE1A.240213.009/12342917:"
    "userdebug/dev-keys"
)
EXPECTED_POOL_MANAGER_SHA256 = (
    "1efe2b6d864d4584703020132c28d69e58a941353c71fc67e52b463208de5411"
)
EXPECTED_START_SCRIPT_SHA256 = (
    "09173b2eb6e2e9929ddbc1981005492f74487973c8809d0f91dbf870dce0ef12"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FIXED_PARAMS: dict[str, dict[str, Any]] = {
    "FilesCreateFolder": {"folder_name": "fixture_folder"},
    "FilesRenameFile": {
        "old_name": "fixture_old.txt",
        "new_name": "fixture_new.txt",
    },
    "FilesDeleteFile": {"file_name": "fixture_delete.txt"},
    "FilesMoveFile": {"file_name": "fixture_move.txt"},
    "FilesSaveCopyOfFile": {"file_name": "fixture_receipt.pdf"},
    "FilesSearchFile": {"needle": "fixture_needle.txt"},
    "FilesCompressFiles": {},
    "FilesExtractArchive": {"archive_name": "fixture_bundle.zip"},
}

_COMPLETE_ZIP_B64 = (
    "UEsDBBQAAAAAAAAAIVwbSmdmCwAAAAsAAAAKAAAAZmlsZV8wLnR4dGZpbGVf"
    "MC50eHQKUEsDBBQAAAAAAAAAIVy+mTutCwAAAAsAAAAKAAAAZmlsZV8xLnR4"
    "dGZpbGVfMS50eHQKUEsDBBQAAAAAAAAAIVwQ668rCwAAAAsAAAAKAAAAZmls"
    "ZV8yLnR4dGZpbGVfMi50eHQKUEsBAhQDFAAAAAAAAAAhXBtKZ2YLAAAACwAA"
    "AAoAAAAAAAAAAAAAAKSBAAAAAGZpbGVfMC50eHRQSwECFAMUAAAAAAAAACFc"
    "vpk7rQsAAAALAAAACgAAAAAAAAAAAAAApIEzAAAAZmlsZV8xLnR4dFBLAQIU"
    "AxQAAAAAAAAAIVwQ668rCwAAAAsAAAAKAAAAAAAAAAAAAACkgWYAAABmaWxl"
    "XzIudHh0UEsFBgAAAAADAAMAqAAAAJkAAAAAAA=="
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
  def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f"Duplicate JSON key {key!r} in {path}")
      result[key] = value
    return result

  def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} in {path}")

  payload = json.loads(
      path.read_text(encoding="utf-8"),
      object_pairs_hook=reject_duplicates,
      parse_constant=reject_constant,
  )
  if not isinstance(payload, dict):
    raise ValueError(f"Expected JSON object in {path}")
  return payload


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def validate_runtime_bindings(
    *,
    cohort_path: Path,
    clone_request_path: Path,
    clone_receipt_path: Path,
    app_attestation_path: Path,
    docker_image_digest: str,
    base_snapshot_id: str,
    base_snapshot_sha256: str,
    snapshot_clone_id: str,
    serial: str,
    worker_index: int,
    console_port: int,
    grpc_port: int,
    adb_server_port: int,
    first_console_port: int,
    first_grpc_port: int,
    first_adb_server_port: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  """Fail closed if CLI labels are not supported by immutable evidence."""
  cohort_sha256 = _sha256(cohort_path)
  request = _strict_json(clone_request_path)
  receipt = _strict_json(clone_receipt_path)
  app_attestation = _strict_json(app_attestation_path)

  expected_serial = f"emulator-{first_console_port + 2 * worker_index}"
  _require(serial == expected_serial, "serial/worker/console mapping mismatch")
  _require(
      console_port == first_console_port + 2 * worker_index,
      "console/worker mapping mismatch",
  )
  _require(
      grpc_port == first_grpc_port + worker_index,
      "gRPC/worker mapping mismatch",
  )
  _require(
      adb_server_port == first_adb_server_port + worker_index,
      "ADB-server/worker mapping mismatch",
  )

  _require(request.get("operation") == "clone_activate", "wrong clone request")
  _require(request.get("analysis_eligible") is False, "clone must be ineligible")
  _require(request.get("cohort_sha256") == cohort_sha256, "request cohort mismatch")
  _require(
      request.get("base_snapshot_id") == base_snapshot_id,
      "request base id mismatch",
  )
  _require(
      request.get("base_snapshot_sha256") == base_snapshot_sha256,
      "request base hash mismatch",
  )
  _require(
      request.get("snapshot_clone_id") == snapshot_clone_id,
      "request clone id mismatch",
  )
  _require(request.get("device_serial") == serial, "request serial mismatch")

  _require(receipt.get("operation") == "clone_activate", "wrong clone receipt")
  _require(receipt.get("success") is True, "clone receipt is not successful")
  _require(
      receipt.get("request_sha256") == _sha256(clone_request_path),
      "clone receipt request hash mismatch",
  )
  _require(
      receipt.get("active_snapshot_clone_id") == snapshot_clone_id,
      "receipt clone id mismatch",
  )
  _require(
      receipt.get("active_snapshot_sha256") == base_snapshot_sha256,
      "receipt activated bytes mismatch",
  )
  _require(
      receipt.get("base_snapshot_id") == base_snapshot_id,
      "receipt base id mismatch",
  )
  _require(
      receipt.get("base_snapshot_sha256") == base_snapshot_sha256,
      "receipt base hash mismatch",
  )
  _require(receipt.get("device_serial") == serial, "receipt serial mismatch")
  _require(
      receipt.get("active_worker_name") == f"catbench-docker-emu-{worker_index}",
      "receipt worker mismatch",
  )
  _require(
      receipt.get("emulator_image") == docker_image_digest,
      "receipt emulator image mismatch",
  )
  _require(
      receipt.get("pool_manager_sha256") == EXPECTED_POOL_MANAGER_SHA256,
      "receipt pool manager mismatch",
  )
  _require(
      receipt.get("emulator_start_script_sha256")
      == EXPECTED_START_SCRIPT_SHA256,
      "receipt emulator start script mismatch",
  )

  _require(app_attestation.get("valid") is True, "app attestation is invalid")
  _require(
      app_attestation.get("evidence_type")
      == "catbench_live_device_app_provision_and_attestation"
      and app_attestation.get("mode") == "attest",
      "wrong app attestation type or mode",
  )
  _require(
      app_attestation.get("cohort_sha256") == cohort_sha256,
      "app attestation cohort mismatch",
  )
  cohort = _strict_json(cohort_path)
  _require(
      app_attestation.get("release_id") == cohort.get("release_id"),
      "app attestation release mismatch",
  )
  _require(
      app_attestation.get("expected_apps") == 23
      and app_attestation.get("valid_apps") == 23
      and app_attestation.get("invalid_apps") == 0,
      "app attestation full-roster count mismatch",
  )
  attested_device = app_attestation.get("device")
  _require(isinstance(attested_device, dict), "missing attested device")
  _require(attested_device.get("serial") == serial, "attested serial mismatch")
  _require(
      attested_device.get("adb_server_port") == adb_server_port,
      "attested ADB server mismatch",
  )
  _require(attested_device.get("api_level") == "33", "attested API mismatch")
  _require(
      attested_device.get("build_fingerprint") == EXPECTED_BUILD_FINGERPRINT,
      "attested build fingerprint mismatch",
  )
  _require(
      isinstance(attested_device.get("boot_id"), str)
      and bool(attested_device["boot_id"].strip()),
      "attested boot id is missing",
  )
  all_apps = app_attestation.get("apps")
  _require(
      isinstance(all_apps, list)
      and len(all_apps) == 23
      and all(isinstance(row, dict) and row.get("valid") is True for row in all_apps),
      "app attestation row count/validity mismatch",
  )
  _require(
      len({row.get("app_id") for row in all_apps}) == 23
      and len({row.get("package_name") for row in all_apps}) == 23,
      "app attestation contains duplicate app or package rows",
  )
  files_apps = [
      row
      for row in all_apps
      if isinstance(row, dict) and row.get("category") == "files"
  ]
  expected_pairs = set(zip(FILES_APP_IDS, files_tasks._PACKAGES, strict=True))
  _require(
      len(files_apps) == len(expected_pairs)
      and {(row.get("app_id"), row.get("package_name")) for row in files_apps}
      == expected_pairs,
      "app attestation Files app/package roster mismatch",
  )
  _require(all(row.get("valid") is True for row in files_apps), "invalid Files app")
  for row in files_apps:
    _require(
        isinstance(row.get("version_code"), str)
        and bool(row["version_code"]),
        "invalid Files app version_code",
    )
    _require(
        isinstance(row.get("version_name"), str)
        and bool(row["version_name"]),
        "invalid Files app version_name",
    )
    hashes = row.get("installed_apk_sha256")
    _require(
        isinstance(hashes, list)
        and bool(hashes)
        and all(
            isinstance(value, str) and SHA256_RE.fullmatch(value)
            for value in hashes
        ),
        "invalid Files active APK hashes",
    )
  evidence = {
      "clone_request": {
          "path": str(clone_request_path),
          "sha256": _sha256(clone_request_path),
      },
      "clone_activate_receipt": {
          "path": str(clone_receipt_path),
          "sha256": _sha256(clone_receipt_path),
      },
      "installed_app_attestation": {
          "path": str(app_attestation_path),
          "sha256": _sha256(app_attestation_path),
          "approval_status": app_attestation.get("approval_status"),
          "claim": app_attestation.get("claim"),
      },
  }
  return evidence, files_apps


def validate_active_worker_container(
    *,
    docker_bin: str,
    clone_receipt_path: Path,
    docker_image_digest: str,
    snapshot_clone_id: str,
    worker_index: int,
    serial: str,
    console_port: int,
    grpc_port: int,
    adb_server_port: int,
) -> dict[str, Any]:
  """Bind the receipt to the container and AVD volume active right now."""
  receipt = _strict_json(clone_receipt_path)
  worker_name = str(receipt["active_worker_name"])
  try:
    completed = subprocess.run(
        [docker_bin, "inspect", worker_name],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
  except (OSError, subprocess.TimeoutExpired) as exc:
    raise ValueError(f"Docker worker inspection failed: {exc}") from exc
  _require(completed.returncode == 0, "Docker worker inspection failed")
  try:
    records = json.loads(completed.stdout)
  except json.JSONDecodeError as exc:
    raise ValueError("Docker worker inspection returned invalid JSON") from exc
  _require(
      isinstance(records, list)
      and len(records) == 1
      and isinstance(records[0], dict),
      "Docker worker inspection shape mismatch",
  )
  record = records[0]
  state = record.get("State", {})
  config = record.get("Config", {})
  host_config = record.get("HostConfig", {})
  labels = config.get("Labels", {})
  _require(state.get("Running") is True, "receipt worker is not running")
  _require(record.get("Name") == f"/{worker_name}", "active worker name mismatch")
  _require(
      record.get("Image") == receipt.get("emulator_image_id"),
      "active image id mismatch",
  )
  _require(config.get("Image") == docker_image_digest, "active image digest mismatch")
  _require(host_config.get("NetworkMode") == "host", "worker network mode mismatch")
  expected_volume = receipt.get("active_volume_name")
  volume_mounts = [
      mount
      for mount in record.get("Mounts", [])
      if isinstance(mount, dict) and mount.get("Type") == "volume"
  ]
  _require(
      len(volume_mounts) == 1
      and volume_mounts[0].get("Name") == expected_volume
      and volume_mounts[0].get("Destination") == "/root/.android"
      and volume_mounts[0].get("RW") is True,
      "active AVD volume mount mismatch",
  )
  _require(
      labels.get("catbench.avd_volume") == expected_volume,
      "volume label mismatch",
  )
  _require(
      labels.get("catbench.snapshot_clone_id") == snapshot_clone_id,
      "active clone label mismatch",
  )
  _require(
      labels.get("catbench.worker_index") == str(worker_index),
      "worker label mismatch",
  )
  _require(
      labels.get("catbench.image_id") == receipt.get("emulator_image_id"),
      "image label mismatch",
  )
  _require(
      labels.get("catbench.launcher_sha256") == EXPECTED_START_SCRIPT_SHA256,
      "launcher label mismatch",
  )
  environment = {}
  for entry in config.get("Env", []):
    if isinstance(entry, str) and "=" in entry:
      key, value = entry.split("=", 1)
      environment[key] = value
  expected_environment = {
      "ANDROID_SERIAL": serial,
      "ANDROID_ADB_SERVER_PORT": str(adb_server_port),
      "ADB_SERVER_PORT": str(adb_server_port),
      "EMULATOR_CONSOLE_PORT": str(console_port),
      "EMULATOR_GRPC_PORT": str(grpc_port),
  }
  _require(
      all(
          environment.get(key) == value
          for key, value in expected_environment.items()
      ),
      "active worker port/serial environment mismatch",
  )
  return {
      "container_id": record.get("Id"),
      "container_created_at": record.get("Created"),
      "container_started_at": state.get("StartedAt"),
      "worker_name": worker_name,
      "running": True,
      "image_digest": config.get("Image"),
      "image_id": record.get("Image"),
      "active_volume_name": expected_volume,
      "snapshot_clone_id": labels.get("catbench.snapshot_clone_id"),
      "worker_index": worker_index,
      "network_mode": host_config.get("NetworkMode"),
      "bound_environment": expected_environment,
  }


def discover_task_classes() -> dict[tuple[str, str], type]:
  discovered: dict[tuple[str, str], type] = {}
  for value in vars(files_tasks).values():
    if not isinstance(value, type):
      continue
    semantic = getattr(value, "catbench_semantic_id", "")
    package = getattr(value, "package_name", "")
    if semantic not in SCOPED_SEMANTICS or package not in files_tasks._PACKAGES:
      continue
    key = (str(semantic), str(package))
    if key in discovered:
      raise RuntimeError(f"Duplicate Files task class for {key}")
    discovered[key] = value
  expected = {
      (semantic, package)
      for package in files_tasks._PACKAGES
      for semantic in SCOPED_SEMANTICS
  }
  if set(discovered) != expected:
    raise RuntimeError(
        "Files storage task roster mismatch: "
        f"missing={sorted(expected - set(discovered))}, "
        f"extra={sorted(set(discovered) - expected)}"
    )
  return discovered


class LiveAudit:
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
    self.shell("input keyevent KEYCODE_WAKEUP")
    self.shell("wm dismiss-keyguard || true")

  def close(self) -> None:
    self.env.close()

  def adb(
      self, *args: str, timeout_seconds: float = 60
  ) -> subprocess.CompletedProcess[str]:
    try:
      return subprocess.run(
          [
              self.adb_path,
              "-P",
              str(self.adb_server_port),
              "-s",
              self.serial,
              *args,
          ],
          check=False,
          capture_output=True,
          text=True,
          timeout=timeout_seconds,
      )
    except (OSError, subprocess.TimeoutExpired) as exc:
      raise RuntimeError(f"adb command did not complete: {args[:1]}: {exc}") from exc

  def shell(self, command: str, *, timeout_seconds: float = 60) -> str:
    completed = self.adb("shell", command, timeout_seconds=timeout_seconds)
    if completed.returncode != 0:
      raise RuntimeError(
          f"adb shell failed ({completed.returncode}): {command}\n"
          f"{completed.stderr}"
      )
    return completed.stdout.strip()

  def score(self, task: Any) -> float:
    return float(task.is_successful(self.env))

  def focus_observation(
      self, package: str, *, command_timeout_seconds: float = 2.0
  ) -> dict[str, Any]:
    """Collect API-33-compatible focus and process evidence for ``package``."""
    window_lines = self.shell(
        "dumpsys window | "
        "grep -E 'mCurrentFocus|mFocusedApp' 2>/dev/null || true",
        timeout_seconds=command_timeout_seconds,
    )
    activity_lines = self.shell(
        "dumpsys activity activities | "
        "grep -E 'topResumedActivity|ResumedActivity' 2>/dev/null | "
        "head -n 6 || true",
        timeout_seconds=command_timeout_seconds,
    )
    pidof = self.shell(
        f"pidof {shlex.quote(package)} || true",
        timeout_seconds=command_timeout_seconds,
    )
    focus_text = f"{window_lines}\n{activity_lines}"
    return {
        "package_name": package,
        "window_focus_lines": window_lines.splitlines(),
        "activity_focus_lines": activity_lines.splitlines(),
        "pidof": pidof.split(),
        "package_focused": package in focus_text,
        "package_process_running": bool(pidof.strip()),
    }

  def wait_for_focus(
      self,
      package: str,
      *,
      timeout_seconds: float = 6.0,
      poll_seconds: float = 0.25,
  ) -> dict[str, Any]:
    """Poll focus instead of converting ordinary emulator latency to failure."""
    started = time.monotonic()
    deadline = started + timeout_seconds
    polls = 0
    observation: dict[str, Any] = {}
    errors: list[str] = []
    while True:
      polls += 1
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        break
      try:
        observation = self.focus_observation(
            package,
            command_timeout_seconds=max(0.25, min(1.5, remaining / 3)),
        )
      except RuntimeError as exc:
        errors.append(str(exc))
        observation = {
            "package_name": package,
            "window_focus_lines": [],
            "activity_focus_lines": [],
            "pidof": [],
            "package_focused": False,
            "package_process_running": False,
        }
      if (
          observation["package_focused"]
          and observation["package_process_running"]
      ):
        break
      if time.monotonic() >= deadline:
        break
      time.sleep(poll_seconds)
    observation["poll_count"] = polls
    observation["timeout_seconds"] = timeout_seconds
    observation["elapsed_seconds"] = round(time.monotonic() - started, 6)
    observation["poll_errors"] = errors
    observation["valid"] = bool(
        observation["package_focused"]
        and observation["package_process_running"]
    )
    return observation

  def device_observation(self) -> dict[str, Any]:
    """Identity fields that must bind this ADB endpoint to the fresh boot."""
    return {
        "serial": self.adb("get-serialno").stdout.strip(),
        "boot_id": self.shell("cat /proc/sys/kernel/random/boot_id"),
        "build_fingerprint": self.shell("getprop ro.build.fingerprint"),
        "api_level": self.shell("getprop ro.build.version.sdk"),
        "android_release": self.shell("getprop ro.build.version.release"),
        "root_adb_uid": self.shell("id -u"),
        "boot_completed": self.shell("getprop sys.boot_completed"),
    }

  def attest_installed_files_apps(
      self, pinned_rows: list[dict[str, Any]]
  ) -> list[dict[str, Any]]:
    """Rehash every active APK on this clone against the pinned attestation."""
    results: list[dict[str, Any]] = []
    for row in pinned_rows:
      package = str(row["package_name"])
      pm_output = self.shell(f"pm path {shlex.quote(package)}")
      paths = sorted(
          line.removeprefix("package:").strip()
          for line in pm_output.splitlines()
          if line.startswith("package:") and line.removeprefix("package:").strip()
      )
      hashes = []
      for path in paths:
        output = self.shell(f"sha256sum {shlex.quote(path)}")
        fields = output.split()
        if not fields:
          raise RuntimeError(f"Empty APK hash output for {package}: {path}")
        hashes.append(fields[0])
      expected = sorted(str(value) for value in row["installed_apk_sha256"])
      actual = sorted(hashes)
      if actual != expected:
        raise RuntimeError(
            f"Live APK hash mismatch for {package}: expected={expected}, "
            f"actual={actual}"
        )
      results.append({
          "app_id": row["app_id"],
          "package_name": package,
          "version_code": row["version_code"],
          "version_name": row["version_name"],
          "active_apk_paths": paths,
          "active_apk_sha256": actual,
          "matches_pinned_attestation": True,
      })
    return sorted(results, key=lambda item: item["package_name"])


def validate_live_device_observation(
    observation: dict[str, Any],
    *,
    attested_device: dict[str, Any],
    serial: str,
) -> None:
  """Require the current ADB endpoint to be the freshly attested root boot."""
  _require(observation.get("serial") == serial, "live ADB serial mismatch")
  _require(observation.get("api_level") == "33", "live API level mismatch")
  _require(
      observation.get("build_fingerprint") == EXPECTED_BUILD_FINGERPRINT,
      "live build fingerprint mismatch",
  )
  _require(observation.get("root_adb_uid") == "0", "live ADB is not root")
  _require(observation.get("boot_completed") == "1", "live boot is incomplete")
  _require(
      observation.get("boot_id") == attested_device.get("boot_id"),
      "fresh app attestation boot id mismatch",
  )


Mutation = Callable[[LiveAudit, Any, str], str]


def _create_folder(audit: LiveAudit, task: Any, phase: str) -> str:
  target = f"{files_tasks._ROOT}/{task.params['folder_name']}"
  if phase == "wrong_type":
    files_tasks._seed_file(audit.env, target, "wrong-type")
    return "target basename exists as a regular file, not a directory"
  if phase == "wrong_name":
    audit.shell(f"mkdir -p {shlex.quote(target + '_wrong')}")
    return "a directory exists at the wrong sibling name"
  audit.shell(f"rm -f '{target}' && mkdir -p '{target}'")
  return "target exists as an exact directory"


def _rename_file(audit: LiveAudit, task: Any, phase: str) -> str:
  old = f"{files_tasks._ROOT}/Docs/{task.params['old_name']}"
  new = f"{files_tasks._ROOT}/Docs/{task.params['new_name']}"
  if phase == "wrong_payload":
    audit.shell(f"rm -f '{old}'")
    files_tasks._seed_file(audit.env, new, "wrong-content")
    return "old name absent but new regular file has wrong payload"
  if phase == "old_retained":
    audit.shell(f"cp '{old}' '{new}'")
    return "new exact copy exists but the old path remains"
  audit.shell(f"mv '{old}' '{new}'")
  return "old absent and new file preserves exact payload"


def _delete_file(audit: LiveAudit, task: Any, phase: str) -> str:
  target = f"{files_tasks._ROOT}/Docs/{task.params['file_name']}"
  decoy = f"{files_tasks._ROOT}/Docs/catbench_keep.txt"
  if phase == "wrong_decoy_deleted":
    audit.shell(f"rm -f '{decoy}'")
    return "wrong decoy deleted while exact target remains"
  if phase == "target_renamed":
    audit.shell(f"mv '{target}' '{target}.renamed'")
    return "target marker was renamed under CATBench instead of deleted"
  if phase == "decoy_wrong_payload":
    audit.shell(f"rm -f '{target}'")
    files_tasks._seed_file(audit.env, decoy, "wrong-content")
    return "target removed but protected decoy payload was changed"
  audit.shell(f"rm -f '{target}'")
  return "target marker removed from CATBench while exact decoy remains"


def _move_file(audit: LiveAudit, task: Any, phase: str) -> str:
  source = f"{files_tasks._ROOT}/Docs/{task.params['file_name']}"
  target = f"{files_tasks._ROOT}/Archive/{task.params['file_name']}"
  if phase == "copy_only":
    audit.shell(f"cp '{source}' '{target}'")
    return "correct payload copied but source still exists"
  audit.shell(f"mv '{source}' '{target}'")
  return "correct payload moved and source absent"


def _save_copy(audit: LiveAudit, task: Any, phase: str) -> str:
  source = f"{files_tasks._ROOT}/Inbox/{task.params['file_name']}"
  target = f"{files_tasks._ROOT}/Receipts/{task.params['file_name']}"
  if phase == "wrong_payload":
    files_tasks._seed_file(audit.env, target, "wrong-content")
    return "target path exists with wrong receipt payload"
  if phase == "source_missing":
    audit.shell(f"cp '{source}' '{target}' && rm -f '{source}'")
    return "exact destination exists but the required source was removed"
  audit.shell(f"cp '{source}' '{target}'")
  return "source and copied target preserve exact receipt payload"


def _search_file(audit: LiveAudit, task: Any, phase: str) -> str:
  needle = task.params["needle"]
  source = audit.shell(
      f"find '{files_tasks._ROOT}/Docs' -type f -name "
      f"{shlex.quote(needle)} -print -quit"
  ).strip()
  if not source:
    raise RuntimeError(f"Search fixture source is missing: {needle}")
  found = f"{source.rsplit('/', 1)[0]}/found.txt"
  if phase == "wrong_payload":
    audit.shell(f"rm -f '{source}'")
    files_tasks._seed_file(audit.env, found, "wrong-content")
    return "needle absent and found.txt exists with wrong payload"
  audit.shell(f"mv '{source}' '{found}'")
  if phase == "decoy_missing":
    decoy = audit.shell(
        f"find '{files_tasks._ROOT}/Docs' -type f -name decoy_0.txt "
        "-print -quit"
    ).strip()
    if not decoy:
      raise RuntimeError("Search fixture decoy_0.txt is missing")
    audit.shell(f"rm -f '{decoy}'")
    return "target was renamed correctly but a protected decoy was removed"
  return "needle renamed to found.txt with exact payload"


def _compress_files(audit: LiveAudit, task: Any, phase: str) -> str:
  del task
  root = files_tasks._ROOT
  source = f"{root}/ToCompress"
  if phase == "partial_archive":
    audit.shell(
        f"cd '{source}' && "
        f"tar -cf '{root}/partial.tar' 'file_0.txt'"
    )
    return "archive contains only one of the three required files"
  if phase == "wrong_member_payloads":
    wrong = f"{root}/WrongArchiveMembers"
    audit.shell(f"mkdir -p '{wrong}'")
    for name in files_tasks._FilesCompressFilesBase._SEED_NAMES:
      files_tasks._seed_file(audit.env, f"{wrong}/{name}", "wrong-content")
    audit.shell(
        f"cd '{wrong}' && tar -cf '{root}/wrong-content.tar' "
        "'file_0.txt' 'file_1.txt' 'file_2.txt'"
    )
    audit.shell(f"rm -rf '{wrong}'")
    return "archive has all required names but wrong member payloads"
  if phase == "unsupported_7z":
    audit.shell(f"printf '%s' 'not-a-validated-7z' > '{root}/output.7z'")
    return "one unsupported 7z candidate exists and must fail closed"
  if phase in ("source_missing", "duplicate_archives", "positive_tar"):
    audit.shell(
        f"cd '{source}' && tar -cf '{root}/complete.tar' "
        "'file_0.txt' 'file_1.txt' 'file_2.txt'"
    )
    if phase == "source_missing":
      audit.shell(f"rm -f '{source}/file_1.txt'")
      return "exact archive exists but one original source file is missing"
    if phase == "duplicate_archives":
      audit.shell(
          f"printf '%s' '{_COMPLETE_ZIP_B64}' | base64 -d > "
          f"'{root}/second.zip'"
      )
      return "two otherwise valid candidate archives exist"
    return "one inspectable tar contains every exact source payload"
  if phase == "positive_zip":
    audit.shell(
        f"printf '%s' '{_COMPLETE_ZIP_B64}' | base64 -d > "
        f"'{root}/complete.zip'"
    )
    return "one inspectable zip contains every exact source payload"
  raise ValueError(f"Unknown compress fixture phase: {phase}")


def _extract_archive(audit: LiveAudit, task: Any, phase: str) -> str:
  source = f"{files_tasks._ROOT}/Archives/{task.params['archive_name']}"
  target = f"{files_tasks._ROOT}/Extracted/{files_tasks._ZIP_INNER_NAME}"
  if phase == "wrong_payload":
    files_tasks._seed_file(audit.env, target, "wrong-content")
    return "expected extracted basename exists with wrong payload"
  if phase == "source_missing":
    files_tasks._seed_file(audit.env, target, files_tasks._ZIP_INNER_CONTENT)
    audit.shell(f"rm -f '{source}'")
    return "exact extracted file exists but the source archive was removed"
  if phase == "wrong_name":
    files_tasks._seed_file(
        audit.env,
        f"{files_tasks._ROOT}/Extracted/wrong_name.txt",
        files_tasks._ZIP_INNER_CONTENT,
    )
    return "exact payload was extracted under the wrong basename"
  files_tasks._seed_file(audit.env, target, files_tasks._ZIP_INNER_CONTENT)
  return "expected extracted basename and exact payload exist"


MUTATIONS: dict[str, Mutation] = {
    "FilesCreateFolder": _create_folder,
    "FilesRenameFile": _rename_file,
    "FilesDeleteFile": _delete_file,
    "FilesMoveFile": _move_file,
    "FilesSaveCopyOfFile": _save_copy,
    "FilesSearchFile": _search_file,
    "FilesCompressFiles": _compress_files,
    "FilesExtractArchive": _extract_archive,
}

# Every case below is independently restored to the initializer's exact storage
# fixture before mutation. This avoids the r2 defect where a partial archive
# survived into the row labelled as an exact positive.
CASE_SPECS: dict[str, tuple[tuple[str, str, float], ...]] = {
    "FilesCreateFolder": (
        ("negative_wrong_type", "wrong_type", 0.0),
        ("negative_wrong_name", "wrong_name", 0.0),
        ("positive_exact_directory", "positive", 1.0),
    ),
    "FilesRenameFile": (
        ("negative_wrong_payload", "wrong_payload", 0.0),
        ("negative_old_retained", "old_retained", 0.0),
        ("positive_exact_rename", "positive", 1.0),
    ),
    "FilesDeleteFile": (
        ("negative_wrong_decoy_deleted", "wrong_decoy_deleted", 0.0),
        ("negative_target_renamed", "target_renamed", 0.0),
        ("negative_decoy_wrong_payload", "decoy_wrong_payload", 0.0),
        ("positive_exact_deletion", "positive", 1.0),
    ),
    "FilesMoveFile": (
        ("negative_copy_only", "copy_only", 0.0),
        ("positive_exact_move", "positive", 1.0),
    ),
    "FilesSaveCopyOfFile": (
        ("negative_wrong_payload", "wrong_payload", 0.0),
        ("negative_source_missing", "source_missing", 0.0),
        ("positive_exact_copy", "positive", 1.0),
    ),
    "FilesSearchFile": (
        ("negative_wrong_payload", "wrong_payload", 0.0),
        ("negative_decoy_missing", "decoy_missing", 0.0),
        ("positive_exact_locate_and_rename", "positive", 1.0),
    ),
    "FilesCompressFiles": (
        ("negative_partial_archive", "partial_archive", 0.0),
        ("negative_wrong_member_payloads", "wrong_member_payloads", 0.0),
        ("negative_unsupported_7z", "unsupported_7z", 0.0),
        ("negative_source_missing", "source_missing", 0.0),
        ("negative_duplicate_archives", "duplicate_archives", 0.0),
        ("positive_exact_tar", "positive_tar", 1.0),
        ("positive_exact_zip", "positive_zip", 1.0),
    ),
    "FilesExtractArchive": (
        ("negative_wrong_payload", "wrong_payload", 0.0),
        ("negative_source_missing", "source_missing", 0.0),
        ("negative_wrong_name", "wrong_name", 0.0),
        ("positive_exact_extract", "positive", 1.0),
    ),
}


def _reset_storage_fixture(task: Any, audit: LiveAudit) -> None:
  """Restore a case fixture without pretending this is snapshot reset/replay."""
  task._cleanup(audit.env)  # pylint: disable=protected-access
  task._seed_state(audit.env)  # pylint: disable=protected-access


def run_audit(audit: LiveAudit) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  classes = discover_task_classes()
  cases: list[dict[str, Any]] = []
  adapters: list[dict[str, Any]] = []
  for package in files_tasks._PACKAGES:
    display_name = files_tasks._APP_DISPLAY_NAMES[package]
    for semantic in SCOPED_SEMANTICS:
      task_class = classes[(semantic, package)]
      task = task_class(dict(FIXED_PARAMS[semantic]))
      adapter_cases: list[dict[str, Any]] = []
      error = ""
      launch_ok = False
      launch_evidence: dict[str, Any] = {}
      try:
        task.initialize_task(audit.env)
        launch_evidence = audit.wait_for_focus(package)
        launch_ok = bool(launch_evidence["valid"])

        initial = audit.score(task)
        adapter_cases.append({
            "case": "initial_no_op",
            "state": "task initializer's exact seeded precondition",
            "expected": 0.0,
            "actual": initial,
            "passed": initial == 0.0,
        })

        for case_name, phase, expected in CASE_SPECS[semantic]:
          _reset_storage_fixture(task, audit)
          case_state = MUTATIONS[semantic](audit, task, phase)
          actual = audit.score(task)
          adapter_cases.append({
              "case": case_name,
              "state": case_state,
              "expected": expected,
              "actual": actual,
              "passed": actual == expected,
          })
      except Exception as exc:  # pylint: disable=broad-except
        error = f"{type(exc).__name__}: {exc}"
      finally:
        if task.initialized:
          try:
            task.tear_down(audit.env)
          except Exception as exc:  # pylint: disable=broad-except
            error = error or f"tear_down {type(exc).__name__}: {exc}"
      for case in adapter_cases:
        cases.append({
            "semantic_task_id": semantic,
            "app_display_name": display_name,
            "package_name": package,
            **case,
        })
      adapters.append({
          "semantic_task_id": semantic,
          "app_display_name": display_name,
          "package_name": package,
          "task_class": task_class.__name__,
          "params": FIXED_PARAMS[semantic],
          "launch_focus_valid": launch_ok,
          "launch_evidence": launch_evidence,
          "case_count": len(adapter_cases),
          "cases_passed": sum(bool(case["passed"]) for case in adapter_cases),
          "error": error,
          "passed": (
              launch_ok
              and not error
              and len(adapter_cases) == 1 + len(CASE_SPECS[semantic])
              and all(bool(case["passed"]) for case in adapter_cases)
          ),
      })
      print(
          json.dumps({
              "progress": f"{len(adapters)}/40",
              "package_name": package,
              "semantic_task_id": semantic,
              "launch_focus_valid": launch_ok,
              "cases_passed": adapters[-1]["cases_passed"],
              "case_count": adapters[-1]["case_count"],
              "passed": adapters[-1]["passed"],
              "error": error,
          }),
          flush=True,
      )
  files_tasks._adb_shell(audit.env, f"rm -rf '{files_tasks._ROOT}'")
  return adapters, cases


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--adb", default=os.environ.get("ADB", "adb"))
  parser.add_argument("--adb_server_port", type=int, required=True)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--console_port", type=int, required=True)
  parser.add_argument("--grpc_port", type=int, required=True)
  parser.add_argument("--worker_index", type=int, required=True)
  parser.add_argument("--first_console_port", type=int, default=5576)
  parser.add_argument("--first_grpc_port", type=int, default=8576)
  parser.add_argument("--first_adb_server_port", type=int, default=5051)
  parser.add_argument("--docker_bin", default="docker")
  parser.add_argument("--docker_image_digest", required=True)
  parser.add_argument("--base_snapshot_id", required=True)
  parser.add_argument("--base_snapshot_sha256", required=True)
  parser.add_argument("--snapshot_clone_id", required=True)
  parser.add_argument("--cohort_manifest", required=True)
  parser.add_argument("--clone_request", required=True)
  parser.add_argument("--clone_receipt", required=True)
  parser.add_argument("--installed_app_attestation", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()

  output_input = Path(args.output).expanduser()
  if output_input.is_symlink():
    raise FileExistsError(
        f"Refusing symlink audit output path: {output_input}"
    )
  output = output_input.resolve()
  if output.exists():
    raise FileExistsError(f"Refusing to overwrite existing audit output: {output}")

  cohort_path = Path(args.cohort_manifest).expanduser().resolve()
  cohort = _strict_json(cohort_path)
  files_spec = cohort.get("categories", {}).get("files", {})
  if files_spec.get("app_ids") != list(FILES_APP_IDS):
    raise ValueError("Frozen Files app roster differs from live audit roster")
  if set(files_spec.get("semantic_task_ids", [])) != (
      set(SCOPED_SEMANTICS) | set(EXCLUDED_SEMANTICS)
  ):
    raise ValueError("Frozen Files semantic roster differs from live audit roster")

  clone_request_path = Path(args.clone_request).expanduser().resolve()
  clone_receipt_path = Path(args.clone_receipt).expanduser().resolve()
  app_attestation_path = Path(args.installed_app_attestation).expanduser().resolve()
  bound_evidence, pinned_files_apps = validate_runtime_bindings(
      cohort_path=cohort_path,
      clone_request_path=clone_request_path,
      clone_receipt_path=clone_receipt_path,
      app_attestation_path=app_attestation_path,
      docker_image_digest=args.docker_image_digest,
      base_snapshot_id=args.base_snapshot_id,
      base_snapshot_sha256=args.base_snapshot_sha256,
      snapshot_clone_id=args.snapshot_clone_id,
      serial=args.serial,
      worker_index=args.worker_index,
      console_port=args.console_port,
      grpc_port=args.grpc_port,
      adb_server_port=args.adb_server_port,
      first_console_port=args.first_console_port,
      first_grpc_port=args.first_grpc_port,
      first_adb_server_port=args.first_adb_server_port,
  )
  active_worker_binding = validate_active_worker_container(
      docker_bin=args.docker_bin,
      clone_receipt_path=clone_receipt_path,
      docker_image_digest=args.docker_image_digest,
      snapshot_clone_id=args.snapshot_clone_id,
      worker_index=args.worker_index,
      serial=args.serial,
      console_port=args.console_port,
      grpc_port=args.grpc_port,
      adb_server_port=args.adb_server_port,
  )

  audit = LiveAudit(
      adb_path=args.adb,
      adb_server_port=args.adb_server_port,
      serial=args.serial,
      console_port=args.console_port,
      grpc_port=args.grpc_port,
  )
  try:
    device = audit.device_observation()
    validate_live_device_observation(
        device,
        attested_device=_strict_json(app_attestation_path)["device"],
        serial=args.serial,
    )
    device.update({
        "console_port": args.console_port,
        "grpc_port": args.grpc_port,
        "adb_server_port": args.adb_server_port,
        "worker_index": args.worker_index,
    })
    live_files_app_attestation = audit.attest_installed_files_apps(
        pinned_files_apps
    )
    adapters, cases = run_audit(audit)
  finally:
    audit.close()

  all_scoped_passed = (
      len(adapters) == 40 and all(bool(adapter["passed"]) for adapter in adapters)
  )
  payload = {
      "schema_version": 1,
      "audit_type": "catbench_files_live_storage_fixture_conformance",
      "artifact_role": "verifier_lifecycle_fixture_only_not_full_g3_or_model_result",
      "analysis_eligible": False,
      "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "release_id": cohort.get("release_id"),
      "cohort_manifest": {
          "path": str(cohort_path),
          "sha256": _sha256(cohort_path),
      },
      "runtime": {
          "docker_image_digest": args.docker_image_digest,
          "base_snapshot_id": args.base_snapshot_id,
          "base_snapshot_sha256": args.base_snapshot_sha256,
          "snapshot_clone_id": args.snapshot_clone_id,
          "device": device,
          "bound_evidence": bound_evidence,
          "active_worker_binding": active_worker_binding,
          "live_files_app_attestation": live_files_app_attestation,
      },
      "source": {
          "script_path": str(Path(__file__).resolve()),
          "script_sha256": _sha256(Path(__file__).resolve()),
          "files_task_module_path": str(Path(files_tasks.__file__).resolve()),
          "files_task_module_sha256": _sha256(Path(files_tasks.__file__).resolve()),
      },
      "scope": {
          "app_count": len(files_tasks._PACKAGES),
          "scoped_semantic_task_ids": list(SCOPED_SEMANTICS),
          "scoped_adapter_count": 40,
          "excluded_semantic_task_ids": list(EXCLUDED_SEMANTICS),
          "excluded_adapter_count": 10,
          "case_policy": {
              semantic: ["initial_no_op"]
              + [case_name for case_name, _, _ in CASE_SPECS[semantic]]
              for semantic in SCOPED_SEMANTICS
          },
          "case_isolation": "direct_cleanup_and_reseed_before_every_mutation",
          "uses_direct_filesystem_fixture_injection": True,
          "uses_human_primitive_action_trajectories": False,
          "uses_app_ui_actions": False,
          "uses_frozen_snapshot_per_adapter": False,
          "shared_storage_predicate_across_apps": True,
          "live_archive_formats_tested": ["tar", "zip"],
          "live_unsupported_archive_fail_closed_formats_tested": ["7z"],
          "native_state_evidence_recorded": False,
      },
      "adapters": adapters,
      "cases": cases,
      "scoped_adapters_passed": sum(bool(row["passed"]) for row in adapters),
      "scoped_adapters_failed": sum(not bool(row["passed"]) for row in adapters),
      "case_count": len(cases),
      "failed_case_count": sum(not bool(row["passed"]) for row in cases),
      "all_scoped_adapters_passed": all_scoped_passed,
      "full_files_category_qualified": False,
      "execution_claims": {
          "benchmark_episode_executed": False,
          "model_endpoint_called": False,
          "agent_action_generated": False,
          "human_gold_trajectory_recorded": False,
          "frozen_snapshot_reset_replay_recorded": False,
          "app_ui_reachability_established": False,
          "raw_native_state_trace_recorded": False,
          "files_view_info_qualified": False,
          "files_share_payload_qualified": False,
      },
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
  print(json.dumps({
      "scoped_adapters": len(adapters),
      "scoped_adapters_passed": payload["scoped_adapters_passed"],
      "case_count": len(cases),
      "failed_case_count": payload["failed_case_count"],
      "all_scoped_adapters_passed": all_scoped_passed,
      "full_files_category_qualified": False,
      "output": str(output),
  }, indent=2))
  return 0 if all_scoped_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
