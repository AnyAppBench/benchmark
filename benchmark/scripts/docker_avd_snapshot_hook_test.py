#!/usr/bin/env python3
"""Focused tests for the Docker AVD snapshot-clone hook.

The in-memory classes below are control-plane test doubles for Docker's volume
and container API only. They never stand in for a CATBench application, task,
model call, episode, verifier outcome, or reported result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import docker_avd_snapshot_hook as hook


BASE_SHA = "b" * 64
COHORT_SHA = "c" * 64


def _request(operation: str) -> dict[str, object]:
  return {
      "schema_version": 1,
      "operation": operation,
      "release_id": "catbench_acl_revision_5cat_v1",
      "release_purpose": "primary_five_category_analysis",
      "artifact_role": "primary_analysis_candidate",
      "analysis_eligible": True,
      "cohort_sha256": COHORT_SHA,
      "pair_id": "pair_8a7afd1163698be48eed9a98",
      "slot_id": "pair_8a7afd1163698be48eed9a98:c1",
      "attempt_id": "pair_8a7afd1163698be48eed9a98:c1:attempt:r0",
      "attempt_index": 0,
      "snapshot_family_id": (
          "catbench_acl_revision_5cat_v1:snapshot:"
          "pair_8a7afd1163698be48eed9a98"
      ),
      "snapshot_clone_id": (
          "catbench_acl_revision_5cat_v1:snapshot:"
          "pair_8a7afd1163698be48eed9a98:c1:r0"
      ),
      "model": "MAI-UI-8B",
      "category": "contacts",
      "app_id": "contacts_fossify_contacts",
      "semantic_task_id": "ContactsRemoveFavoriteContact",
      "instance_id": 0,
      "condition": "c1",
      "is_replacement": False,
      "base_snapshot_id": "catbench-api33-frozen-base-v1",
      "base_snapshot_sha256": BASE_SHA,
      "device_serial": "emulator-5576",
  }


def _base_labels(request: dict[str, object]) -> dict[str, str]:
  return {
      "catbench.snapshot.role": "frozen-base",
      "catbench.snapshot.sealed": "true",
      "catbench.snapshot.id": str(request["base_snapshot_id"]),
      "catbench.snapshot.sha256": str(request["base_snapshot_sha256"]),
      "catbench.release_id": str(request["release_id"]),
      "catbench.cohort_sha256": str(request["cohort_sha256"]),
  }


class _InMemoryDockerState:

  def __init__(self, request: dict[str, object]):
    self.volumes: dict[str, dict[str, object]] = {
        "catbench-frozen-base": {"Labels": _base_labels(request)},
    }
    self.fingerprints = {"catbench-frozen-base": BASE_SHA}
    self.worker: dict[str, object] | None = None
    self.base_users: list[str] = []
    self.actions: list[tuple[str, str]] = []


class _InMemoryRuntime:

  def __init__(self, config: hook.HookConfig, state: _InMemoryDockerState):
    self.config = config
    self.state = state

  def image_id(self, reference: str | None = None) -> str:
    del reference
    return "sha256:" + "d" * 64

  def volume(self, name: str):
    return self.state.volumes.get(name)

  def containers_using_volume(self, name: str) -> list[str]:
    if name == self.config.base_volume:
      return list(self.state.base_users)
    if self.state.worker is not None:
      mounts = self.state.worker.get("Mounts", [])
      if any(mount.get("Name") == name for mount in mounts):
        return ["worker-container-id"]
    return []

  def create_volume(self, name: str, labels):
    if name in self.state.volumes:
      raise AssertionError("test attempted duplicate volume creation")
    self.state.volumes[name] = {"Labels": dict(labels)}
    self.state.fingerprints[name] = "0" * 64

  def remove_volume(self, name: str):
    if self.containers_using_volume(name):
      raise AssertionError("test attempted to remove an attached volume")
    del self.state.volumes[name]
    del self.state.fingerprints[name]

  def copy_volume(self, source: str, target: str):
    self.state.fingerprints[target] = self.state.fingerprints[source]

  def fingerprint_volume(self, name: str) -> str:
    return self.state.fingerprints[name]

  def manager(self, action: str, volume: str):
    self.state.actions.append((action, volume))
    if action == "activate-volume":
      labels = self.state.volumes[volume]["Labels"]
      self.state.worker = {
          "Mounts": [{"Destination": "/root/.android", "Name": volume}],
          "Config": {
              "Labels": {
                  "catbench.pool": self.config.name_prefix,
                  "catbench.worker_index": str(self.config.worker_index),
                  "catbench.avd_volume": volume,
                  "catbench.launcher_sha256": (
                      self.config.start_script_sha256
                  ),
                  "catbench.base_launcher_sha256": (
                      self.config.base_start_script_sha256
                  ),
                  "catbench.emulator_memory_mb": str(
                      self.config.emulator_memory_mb
                  ),
                  "catbench.emulator_cores": str(
                      self.config.emulator_cores
                  ),
                  "catbench.snapshot_clone_id": labels[
                      "catbench.snapshot.clone_id"
                  ],
              },
              "Env": [
                  f"ANDROID_SERIAL={self.config.expected_serial}",
                  f"ADB_SERVER_PORT={self.config.expected_adb_server_port}",
                  f"CATBENCH_EMULATOR_MEMORY_MB={self.config.emulator_memory_mb}",
                  f"CATBENCH_EMULATOR_CORES={self.config.emulator_cores}",
              ],
          },
          "State": {"Running": True},
      }
    elif action == "deactivate-volume":
      if self.state.worker is None:
        raise AssertionError("test worker was already absent")
      self.state.worker = None
    else:
      raise AssertionError(f"unexpected manager action: {action}")

  def worker(self):
    return self.state.worker


class DockerAvdSnapshotHookTest(unittest.TestCase):

  def _env(self, root: Path) -> dict[str, str]:
    return {
        "CATBENCH_DOCKER_BASE_VOLUME": "catbench-frozen-base",
        "CATBENCH_DOCKER_WORKER_INDEX": "0",
        "CATBENCH_DOCKER_NUM_EMULATORS": "2",
        "CATBENCH_DOCKER_FIRST_CONSOLE_PORT": "5576",
        "CATBENCH_DOCKER_FIRST_GRPC_PORT": "8576",
        "CATBENCH_DOCKER_FIRST_ADB_SERVER_PORT": "5041",
        "ANDROID_ADB_SERVER_PORT": "5041",
        "CATBENCH_DOCKER_COMMAND_TIMEOUT": "60",
        "CATBENCH_DOCKER_EMULATOR_TIMEOUT": "40",
        "CATBENCH_DOCKER_LOCK_DIR": str(root / "locks"),
        "CATBENCH_DOCKER_POOL_MANAGER": str(hook.DEFAULT_MANAGER),
        "CATBENCH_DOCKER_START_SCRIPT": str(
            hook.DEFAULT_START_SCRIPT
        ),
        "CATBENCH_DOCKER_BASE_START_SCRIPT": str(
            hook.DEFAULT_BASE_START_SCRIPT
        ),
    }

  def _write_request(self, root: Path, operation: str) -> Path:
    path = root / f"{operation}.json"
    path.write_text(json.dumps(_request(operation), indent=2) + "\n")
    return path

  def test_fingerprint_is_deterministic_and_content_sensitive(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      (root / "nested").mkdir()
      target = root / "nested" / "userdata-qemu.img"
      target.write_bytes(b"first-state")
      os.chmod(target, 0o640)
      (root / "config-link").symlink_to("nested/userdata-qemu.img")
      first = hook.fingerprint_root(root)
      self.assertRegex(first, r"^[0-9a-f]{64}$")
      self.assertEqual(first, hook.fingerprint_root(root))
      target.write_bytes(b"second-state")
      self.assertNotEqual(first, hook.fingerprint_root(root))

  def test_fingerprint_rejects_special_files(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      fifo = Path(tmpdir) / "unsupported.fifo"
      os.mkfifo(fifo)
      with self.assertRaisesRegex(hook.HookError, "unsupported special file"):
        hook.fingerprint_root(Path(tmpdir))

  def test_request_validation_rejects_unknown_and_wrongly_typed_fields(self):
    request = _request("clone_activate")
    request["unexpected"] = "value"
    with self.assertRaisesRegex(hook.HookError, "key set mismatch"):
      hook.validate_request(request)
    request = _request("clone_activate")
    request["analysis_eligible"] = 1
    with self.assertRaisesRegex(hook.HookError, "analysis_eligible"):
      hook.validate_request(request)
    request = _request("clone_activate")
    request["model"] = "UI Voyager-4B"
    self.assertEqual(hook.validate_request(request)["model"], "UI Voyager-4B")

  def test_strict_request_loader_rejects_duplicate_json_keys(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      request = Path(tmpdir) / "request.json"
      request.write_text('{"schema_version":1,"schema_version":1}\n')
      with self.assertRaisesRegex(hook.HookError, "duplicate JSON key"):
        hook._strict_json(request)  # pylint: disable=protected-access

  def test_config_binds_request_serial_and_parent_adb_server(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      env = self._env(root)
      request = _request("clone_activate")
      env["ANDROID_ADB_SERVER_PORT"] = "5037"
      with self.assertRaisesRegex(hook.HookError, "ANDROID_ADB_SERVER_PORT"):
        hook.HookConfig.from_env(request, env)
      env = self._env(root)
      request["device_serial"] = "emulator-5578"
      with self.assertRaisesRegex(hook.HookError, "configured worker requires"):
        hook.HookConfig.from_env(request, env)

  def test_config_rejects_mixed_helper_and_emulator_images(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      env = self._env(Path(tmpdir))
      env['CATBENCH_DOCKER_HELPER_IMAGE'] = 'helper@sha256:' + 'a' * 64
      env['CATBENCH_DOCKER_EMULATOR_IMAGE'] = 'emulator@sha256:' + 'b' * 64

      with self.assertRaisesRegex(
          hook.HookError, 'must be the same immutable digest'
      ):
        hook.HookConfig.from_env(_request('clone_activate'), env)

  def test_config_rejects_unpinned_or_symlinked_manager(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      env = self._env(root)
      env["CATBENCH_DOCKER_POOL_MANAGER"] = "/bin/true"
      with self.assertRaisesRegex(hook.HookError, "hook-pinned revision"):
        hook.HookConfig.from_env(_request("clone_activate"), env)
      manager_link = root / "manager-link"
      manager_link.symlink_to(hook.DEFAULT_MANAGER)
      env["CATBENCH_DOCKER_POOL_MANAGER"] = str(manager_link)
      with self.assertRaisesRegex(hook.HookError, "may not be a symlink"):
        hook.HookConfig.from_env(_request("clone_activate"), env)

  def test_config_rejects_emulator_resource_drift(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      for name, value in (
          ("CATBENCH_EMULATOR_MEMORY_MB", "2048"),
          ("CATBENCH_EMULATOR_CORES", "1"),
      ):
        with self.subTest(name=name):
          env = self._env(root)
          env[name] = value
          with self.assertRaisesRegex(hook.HookError, "hook-pinned"):
            hook.HookConfig.from_env(_request("clone_activate"), env)

  def test_clone_activate_and_release_are_fresh_and_destructive(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      request = _request("clone_activate")
      state = _InMemoryDockerState(request)
      factory = lambda config: _InMemoryRuntime(config, state)
      activate_request = self._write_request(root, "clone_activate")
      activate_receipt_path = root / "activate-receipt.json"
      activate = hook.run_hook(
          activate_request,
          activate_receipt_path,
          environ=self._env(root),
          runtime_factory=factory,
      )
      clone_volume = activate["active_volume_name"]
      self.assertEqual(activate["active_snapshot_sha256"], BASE_SHA)
      self.assertEqual(activate["pool_manager_sha256"], hook.DEFAULT_MANAGER_SHA256)
      self.assertEqual(
          activate["emulator_start_script_sha256"],
          hook.DEFAULT_START_SCRIPT_SHA256,
      )
      self.assertEqual(
          activate["emulator_base_start_script_sha256"],
          hook.DEFAULT_BASE_START_SCRIPT_SHA256,
      )
      self.assertEqual(
          activate["emulator_memory_mb"],
          hook.DEFAULT_EMULATOR_MEMORY_MB,
      )
      self.assertEqual(
          activate["emulator_cores"], hook.DEFAULT_EMULATOR_CORES
      )
      self.assertEqual(
          activate["active_snapshot_clone_id"], request["snapshot_clone_id"]
      )
      self.assertIn(clone_volume, state.volumes)
      self.assertIsNotNone(state.worker)
      self.assertEqual(state.actions, [("activate-volume", clone_volume)])

      release_request = self._write_request(root, "release")
      release_receipt_path = root / "release-receipt.json"
      release = hook.run_hook(
          release_request,
          release_receipt_path,
          environ=self._env(root),
          runtime_factory=factory,
      )
      self.assertEqual(release["released_snapshot_sha256"], BASE_SHA)
      self.assertTrue(release["released_volume_deleted"])
      self.assertNotIn(clone_volume, state.volumes)
      self.assertIsNone(state.worker)
      self.assertEqual(
          state.actions,
          [("activate-volume", clone_volume), ("deactivate-volume", clone_volume)],
      )
      self.assertEqual(json.loads(activate_receipt_path.read_text()), activate)
      self.assertEqual(json.loads(release_receipt_path.read_text()), release)

  def test_clone_activation_fails_closed_if_base_is_attached(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      request = _request("clone_activate")
      state = _InMemoryDockerState(request)
      state.base_users = ["unexpected-container"]
      factory = lambda config: _InMemoryRuntime(config, state)
      with self.assertRaisesRegex(hook.HookError, "base volume is attached"):
        hook.run_hook(
            self._write_request(root, "clone_activate"),
            root / "receipt.json",
            environ=self._env(root),
            runtime_factory=factory,
        )
      self.assertFalse((root / "receipt.json").exists())
      self.assertEqual(set(state.volumes), {"catbench-frozen-base"})

  def test_clone_activation_refuses_stale_clone_reuse(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      request = _request("clone_activate")
      state = _InMemoryDockerState(request)
      config = hook.HookConfig.from_env(request, self._env(root))
      clone_name = hook._clone_volume_name(  # pylint: disable=protected-access
          config, request
      )
      state.volumes[clone_name] = {"Labels": {}}
      state.fingerprints[clone_name] = BASE_SHA
      factory = lambda value: _InMemoryRuntime(value, state)
      with self.assertRaisesRegex(hook.HookError, "refusing reuse"):
        hook.run_hook(
            self._write_request(root, "clone_activate"),
            root / "receipt.json",
            environ=self._env(root),
            runtime_factory=factory,
        )

  def test_receipt_writer_never_replaces_existing_file(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "receipt.json"
      path.write_text("operator-owned\n")
      with self.assertRaisesRegex(hook.HookError, "refusing to replace"):
        hook._write_json_exclusive(  # pylint: disable=protected-access
            path, {"success": True}
        )
      self.assertEqual(path.read_text(), "operator-owned\n")

  def test_seal_base_copies_offline_source_and_marks_evidence_observational(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      state = _InMemoryDockerState(_request("clone_activate"))
      state.volumes = {"offline-source-avd": {"Labels": {}}}
      state.fingerprints = {"offline-source-avd": BASE_SHA}
      factory = lambda config: _InMemoryRuntime(config, state)
      evidence_path = root / "seal-evidence.json"
      evidence = hook.seal_base(
          source_volume="offline-source-avd",
          base_volume="sealed-base-avd",
          snapshot_id="sealed-base-v1",
          release_id="catbench_acl_revision_5cat_v1",
          cohort_sha256=COHORT_SHA,
          evidence_path=evidence_path,
          environ=self._env(root),
          runtime_factory=factory,
      )
      self.assertEqual(evidence["snapshot_sha256"], BASE_SHA)
      self.assertEqual(
          evidence["approval_status"], "observational_not_release_approval"
      )
      labels = state.volumes["sealed-base-avd"]["Labels"]
      self.assertEqual(labels["catbench.snapshot.role"], "frozen-base")
      self.assertEqual(labels["catbench.snapshot.sha256"], BASE_SHA)
      self.assertEqual(json.loads(evidence_path.read_text()), evidence)


if __name__ == "__main__":
  unittest.main()
