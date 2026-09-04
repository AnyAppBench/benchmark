#!/usr/bin/env python3
"""Fail-closed Docker AVD volume clone hook for frozen CATBench episodes.

The normal invocation is the two-argument contract used by
``consume_catbench_frozen_schedule.py``::

  docker_avd_snapshot_hook.py --request REQUEST.json --receipt RECEIPT.json

``clone_activate`` materializes a new Docker volume from an unattached,
content-attested base volume and boots exactly one pool worker from that clone.
``release`` stops and removes that worker, fingerprints its final volume, and
deletes the clone.  The base volume is mounted read-only and fingerprinted on
every operation.  Existing episode clones are never reused.

The separate ``--seal-base`` operator mode copies an offline provisioned AVD
volume into a newly labelled base volume and writes observational evidence.
It does not approve a release or manufacture the other fields required by the
consumer's base-snapshot manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_MANAGER = SCRIPT_DIR / "manage_catbench_docker_pool.sh"
DEFAULT_START_SCRIPT = (
    SCRIPT_DIR.parent / "docker_setup" / "start_catbench_emu_headless.sh"
)
DEFAULT_BASE_START_SCRIPT = (
    SCRIPT_DIR.parent / "docker_setup" / "start_emu_headless.sh"
)
DEFAULT_IMAGE = (
    "android_world@sha256:"
    "6d8b2c148aebd3a1fe626768efe22c01a7a62cdbd2cbbe7d3f973adc57c7dd2f"
)
DEFAULT_MANAGER_SHA256 = (
    "c2e1a96cf324a35d652dc0076705b4a52009aaeef42f95613ca1c4d92a730b10"
)
DEFAULT_START_SCRIPT_SHA256 = (
    "4f3f25558f6ce8ba324e138cccdfa4a556f9142d6b48653d60ec3eda2e0161ac"
)
DEFAULT_BASE_START_SCRIPT_SHA256 = (
    "49dd7da2aa3d429f93171219cc4b2bdf9c437ff327c7121334c728b40ba08b40"
)
DEFAULT_EMULATOR_MEMORY_MB = 4096
DEFAULT_EMULATOR_CORES = 2
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DOCKER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/-]{0,499}$")
SERIAL_RE = re.compile(r"^emulator-([0-9]{4,5})$")
FINGERPRINT_SCHEMA = b"catbench-docker-avd-tree-v1\0"
HOOK_REQUEST_KEYS = frozenset({
    "schema_version",
    "operation",
    "release_id",
    "release_purpose",
    "artifact_role",
    "analysis_eligible",
    "cohort_sha256",
    "pair_id",
    "slot_id",
    "attempt_id",
    "attempt_index",
    "snapshot_family_id",
    "snapshot_clone_id",
    "model",
    "category",
    "app_id",
    "semantic_task_id",
    "instance_id",
    "condition",
    "is_replacement",
    "base_snapshot_id",
    "base_snapshot_sha256",
    "device_serial",
})


class HookError(RuntimeError):
  """An integrity or infrastructure failure that must block the episode."""


def _sha256_path(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
  return json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode("utf-8")


def _utc_now() -> str:
  return dt.datetime.now(dt.UTC).isoformat()


def _strict_json(path: Path) -> tuple[dict[str, Any], str]:
  if path.is_symlink() or not path.is_file():
    raise HookError(f"request must be a regular non-symlink file: {path}")
  raw = path.read_bytes()

  def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise HookError(f"duplicate JSON key in request: {key}")
      result[key] = value
    return result

  def reject_constant(value: str) -> None:
    raise HookError(f"non-finite JSON constant in request: {value}")

  try:
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise HookError(f"invalid request JSON: {exc}") from exc
  if not isinstance(payload, dict):
    raise HookError("request JSON must be an object")
  return payload, hashlib.sha256(raw).hexdigest()


def _require_safe_id(payload: Mapping[str, Any], field: str) -> str:
  value = payload.get(field)
  if not isinstance(value, str) or not SAFE_ID_RE.fullmatch(value):
    raise HookError(f"request has invalid {field}")
  return value


def validate_request(payload: Mapping[str, Any]) -> dict[str, Any]:
  observed = frozenset(payload)
  if observed != HOOK_REQUEST_KEYS:
    missing = sorted(HOOK_REQUEST_KEYS - observed)
    extra = sorted(observed - HOOK_REQUEST_KEYS)
    raise HookError(f"request key set mismatch; missing={missing}, extra={extra}")
  if payload.get("schema_version") != 1:
    raise HookError("request schema_version must be 1")
  if payload.get("operation") not in ("clone_activate", "release"):
    raise HookError("request operation must be clone_activate or release")
  for field in (
      "release_id",
      "pair_id",
      "slot_id",
      "attempt_id",
      "snapshot_family_id",
      "snapshot_clone_id",
      "category",
      "app_id",
      "semantic_task_id",
      "base_snapshot_id",
  ):
    _require_safe_id(payload, field)
  if not SHA256_RE.fullmatch(str(payload.get("cohort_sha256") or "")):
    raise HookError("request has invalid cohort_sha256")
  if not SHA256_RE.fullmatch(str(payload.get("base_snapshot_sha256") or "")):
    raise HookError("request has invalid base_snapshot_sha256")
  if not SERIAL_RE.fullmatch(str(payload.get("device_serial") or "")):
    raise HookError("request has invalid device_serial")
  for field in ("release_purpose", "artifact_role", "condition", "model"):
    value = payload.get(field)
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 500
        or any(ord(character) < 32 for character in value)
    ):
      raise HookError(f"request has invalid {field}")
  for field in ("attempt_index", "instance_id"):
    if isinstance(payload.get(field), bool) or not isinstance(payload.get(field), int):
      raise HookError(f"request has invalid {field}")
    if int(payload[field]) < 0:
      raise HookError(f"request has negative {field}")
  for field in ("analysis_eligible", "is_replacement"):
    if not isinstance(payload.get(field), bool):
      raise HookError(f"request has invalid {field}")
  return dict(payload)


def _framed_update(digest: Any, value: bytes) -> None:
  digest.update(len(value).to_bytes(8, "big"))
  digest.update(value)


def fingerprint_root(root: Path) -> str:
  """Hash a volume tree without following links or relying on tar metadata."""
  if root.is_symlink() or not root.is_dir():
    raise HookError(f"fingerprint root must be a real directory: {root}")
  digest = hashlib.sha256(FINGERPRINT_SCHEMA)

  def visit(directory: Path, relative: Path) -> None:
    try:
      entries = sorted(os.scandir(directory), key=lambda item: os.fsencode(item.name))
    except OSError as exc:
      raise HookError(f"cannot enumerate snapshot tree: {directory}: {exc}") from exc
    for entry in entries:
      path = directory / entry.name
      rel = relative / entry.name
      try:
        before = entry.stat(follow_symlinks=False)
      except OSError as exc:
        raise HookError(f"cannot stat snapshot entry: {rel}: {exc}") from exc
      mode = before.st_mode
      if stat.S_ISDIR(mode):
        kind = "directory"
      elif stat.S_ISREG(mode):
        kind = "regular"
      elif stat.S_ISLNK(mode):
        kind = "symlink"
      else:
        raise HookError(f"unsupported special file in snapshot: {rel}")
      metadata: dict[str, Any] = {
          "path_hex": os.fsencode(str(rel)).hex(),
          "kind": kind,
          "mode": stat.S_IMODE(mode),
          "uid": before.st_uid,
          "gid": before.st_gid,
      }
      try:
        xattrs = []
        for name in sorted(os.listxattr(path, follow_symlinks=False)):
          xattrs.append({
              "name_hex": os.fsencode(name).hex(),
              "value_hex": os.getxattr(
                  path, name, follow_symlinks=False
              ).hex(),
          })
        metadata["xattrs"] = xattrs
      except OSError as exc:
        raise HookError(f"cannot read snapshot xattrs: {rel}: {exc}") from exc
      if kind == "regular":
        file_digest = hashlib.sha256()
        try:
          with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
              file_digest.update(chunk)
          after = path.stat(follow_symlinks=False)
        except OSError as exc:
          raise HookError(f"cannot hash snapshot file: {rel}: {exc}") from exc
        if (
            before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
          raise HookError(f"snapshot file changed while hashing: {rel}")
        metadata["size"] = before.st_size
        metadata["content_sha256"] = file_digest.hexdigest()
      elif kind == "symlink":
        try:
          metadata["target_hex"] = os.fsencode(os.readlink(path)).hex()
        except OSError as exc:
          raise HookError(f"cannot read snapshot symlink: {rel}: {exc}") from exc
      _framed_update(digest, _canonical_json(metadata))
      if kind == "directory":
        visit(path, rel)

  visit(root, Path())
  return digest.hexdigest()


def _required_env(name: str, environ: Mapping[str, str]) -> str:
  value = str(environ.get(name) or "").strip()
  if not value:
    raise HookError(f"required environment variable is missing: {name}")
  return value


def _env_integer(
    name: str,
    environ: Mapping[str, str],
    *,
    default: int | None = None,
    minimum: int = 0,
    maximum: int = 65535,
) -> int:
  raw = environ.get(name)
  if raw is None and default is not None:
    return default
  try:
    value = int(str(raw))
  except (TypeError, ValueError) as exc:
    raise HookError(f"{name} must be an integer") from exc
  if not minimum <= value <= maximum:
    raise HookError(f"{name} must be in [{minimum}, {maximum}]")
  return value


@dataclasses.dataclass(frozen=True)
class HookConfig:
  base_volume: str
  worker_index: int
  num_emulators: int
  first_console_port: int
  first_grpc_port: int
  first_adb_server_port: int
  name_prefix: str
  baseline_volume_prefix: str
  clone_volume_prefix: str
  helper_image: str
  emulator_image: str
  manager: Path
  manager_sha256: str
  start_script: Path
  start_script_sha256: str
  base_start_script: Path
  base_start_script_sha256: str
  emulator_memory_mb: int
  emulator_cores: int
  adb: str
  docker_bin: str
  emulator_timeout: int
  command_timeout: int
  lock_dir: Path

  @property
  def expected_serial(self) -> str:
    return f"emulator-{self.first_console_port + 2 * self.worker_index}"

  @property
  def expected_adb_server_port(self) -> int:
    return self.first_adb_server_port + self.worker_index

  @property
  def worker_name(self) -> str:
    return f"{self.name_prefix}-{self.worker_index}"

  @classmethod
  def from_env(
      cls, request: Mapping[str, Any], environ: Mapping[str, str]
  ) -> "HookConfig":
    worker_index = _env_integer(
        "CATBENCH_DOCKER_WORKER_INDEX", environ, minimum=0, maximum=1024
    )
    num_emulators = _env_integer(
        "CATBENCH_DOCKER_NUM_EMULATORS",
        environ,
        default=max(2, worker_index + 1),
        minimum=1,
        maximum=1024,
    )
    if worker_index >= num_emulators:
      raise HookError("worker index is outside CATBENCH_DOCKER_NUM_EMULATORS")
    first_console = _env_integer(
        "CATBENCH_DOCKER_FIRST_CONSOLE_PORT",
        environ,
        default=5576,
        minimum=1024,
    )
    first_grpc = _env_integer(
        "CATBENCH_DOCKER_FIRST_GRPC_PORT",
        environ,
        default=8576,
        minimum=1024,
    )
    first_adb = _env_integer(
        "CATBENCH_DOCKER_FIRST_ADB_SERVER_PORT",
        environ,
        default=5041,
        minimum=1024,
    )
    expected_serial = f"emulator-{first_console + 2 * worker_index}"
    if request["device_serial"] != expected_serial:
      raise HookError(
          f"request device_serial={request['device_serial']!r}; "
          f"configured worker requires {expected_serial!r}"
      )
    expected_adb = first_adb + worker_index
    parent_adb = environ.get("ANDROID_ADB_SERVER_PORT")
    if parent_adb != str(expected_adb):
      raise HookError(
          "ANDROID_ADB_SERVER_PORT must name this worker's isolated ADB "
          f"server ({expected_adb}) before the consumer starts"
      )
    helper_image = environ.get("CATBENCH_DOCKER_HELPER_IMAGE", DEFAULT_IMAGE)
    emulator_image = environ.get("CATBENCH_DOCKER_EMULATOR_IMAGE", helper_image)
    for name, value in (
        ("CATBENCH_DOCKER_HELPER_IMAGE", helper_image),
        ("CATBENCH_DOCKER_EMULATOR_IMAGE", emulator_image),
    ):
      if not re.search(r"@sha256:[0-9a-f]{64}$", value):
        raise HookError(f"{name} must be pinned by sha256 digest")
    if helper_image != emulator_image:
      raise HookError(
          "CATBENCH_DOCKER_HELPER_IMAGE and "
          "CATBENCH_DOCKER_EMULATOR_IMAGE must be the same immutable digest"
      )
    manager_input = Path(
        environ.get("CATBENCH_DOCKER_POOL_MANAGER", str(DEFAULT_MANAGER))
    ).expanduser()
    start_script_input = Path(
        environ.get(
            "CATBENCH_DOCKER_START_SCRIPT",
            str(DEFAULT_START_SCRIPT),
        )
    ).expanduser()
    base_start_script_input = Path(
        environ.get(
            "CATBENCH_DOCKER_BASE_START_SCRIPT",
            str(DEFAULT_BASE_START_SCRIPT),
        )
    ).expanduser()
    for name, path in (
        ("pool manager", manager_input),
        ("start script", start_script_input),
        ("base start script", base_start_script_input),
    ):
      if path.is_symlink():
        raise HookError(f"{name} may not be a symlink: {path}")
    manager = manager_input.resolve()
    start_script = start_script_input.resolve()
    base_start_script = base_start_script_input.resolve()
    for name, path in (
        ("pool manager", manager),
        ("start script", start_script),
        ("base start script", base_start_script),
    ):
      if not path.is_file():
        raise HookError(f"{name} must be a regular non-symlink file: {path}")
    for name, path in (
        ("pool manager", manager),
        ("start script", start_script),
        ("base start script", base_start_script),
    ):
      if not os.access(path, os.X_OK):
        raise HookError(f"{name} is not executable: {path}")
    manager_sha256 = _sha256_path(manager)
    start_script_sha256 = _sha256_path(start_script)
    base_start_script_sha256 = _sha256_path(base_start_script)
    if manager_sha256 != DEFAULT_MANAGER_SHA256:
      raise HookError(
          "pool manager does not match the hook-pinned revision: "
          f"{manager_sha256}"
      )
    if start_script_sha256 != DEFAULT_START_SCRIPT_SHA256:
      raise HookError(
          "emulator start script does not match the hook-pinned revision: "
          f"{start_script_sha256}"
      )
    if base_start_script_sha256 != DEFAULT_BASE_START_SCRIPT_SHA256:
      raise HookError(
          "emulator base start script does not match the hook-pinned revision: "
          f"{base_start_script_sha256}"
      )
    emulator_memory_mb = _env_integer(
        "CATBENCH_EMULATOR_MEMORY_MB",
        environ,
        default=DEFAULT_EMULATOR_MEMORY_MB,
        minimum=1,
    )
    emulator_cores = _env_integer(
        "CATBENCH_EMULATOR_CORES",
        environ,
        default=DEFAULT_EMULATOR_CORES,
        minimum=1,
    )
    if emulator_memory_mb != DEFAULT_EMULATOR_MEMORY_MB:
      raise HookError(
          "CATBENCH_EMULATOR_MEMORY_MB does not match the hook-pinned "
          f"value {DEFAULT_EMULATOR_MEMORY_MB}"
      )
    if emulator_cores != DEFAULT_EMULATOR_CORES:
      raise HookError(
          "CATBENCH_EMULATOR_CORES does not match the hook-pinned "
          f"value {DEFAULT_EMULATOR_CORES}"
      )
    base_volume = _required_env("CATBENCH_DOCKER_BASE_VOLUME", environ)
    name_prefix = environ.get("CATBENCH_DOCKER_NAME_PREFIX", "catbench-docker-emu")
    baseline_prefix = environ.get(
        "CATBENCH_DOCKER_BASELINE_VOLUME_PREFIX", "catbench-docker-avd"
    )
    clone_prefix = environ.get(
        "CATBENCH_DOCKER_CLONE_VOLUME_PREFIX", "catbench-episode-avd"
    )
    for name, value in (
        ("base volume", base_volume),
        ("name prefix", name_prefix),
        ("baseline volume prefix", baseline_prefix),
        ("clone volume prefix", clone_prefix),
    ):
      if not DOCKER_NAME_RE.fullmatch(value):
        raise HookError(f"invalid Docker {name}: {value!r}")
    command_timeout = _env_integer(
        "CATBENCH_DOCKER_COMMAND_TIMEOUT",
        environ,
        default=240,
        minimum=60,
        maximum=270,
    )
    emulator_timeout = _env_integer(
        "CATBENCH_DOCKER_EMULATOR_TIMEOUT",
        environ,
        default=210,
        minimum=30,
        maximum=command_timeout - 10,
    )
    lock_dir = Path(
        environ.get("CATBENCH_DOCKER_LOCK_DIR", "/tmp/catbench-docker-snapshot-locks")
    ).expanduser().resolve()
    return cls(
        base_volume=base_volume,
        worker_index=worker_index,
        num_emulators=num_emulators,
        first_console_port=first_console,
        first_grpc_port=first_grpc,
        first_adb_server_port=first_adb,
        name_prefix=name_prefix,
        baseline_volume_prefix=baseline_prefix,
        clone_volume_prefix=clone_prefix,
        helper_image=helper_image,
        emulator_image=emulator_image,
        manager=manager,
        manager_sha256=manager_sha256,
        start_script=start_script,
        start_script_sha256=start_script_sha256,
        base_start_script=base_start_script,
        base_start_script_sha256=base_start_script_sha256,
        emulator_memory_mb=emulator_memory_mb,
        emulator_cores=emulator_cores,
        adb=(
            environ.get("CATBENCH_ADB")
            or shutil.which("adb")
            or str(Path.home() / "Android" / "Sdk" / "platform-tools" / "adb")
        ),
        docker_bin=environ.get("CATBENCH_DOCKER_BIN", "docker"),
        emulator_timeout=emulator_timeout,
        command_timeout=command_timeout,
        lock_dir=lock_dir,
    )


class DockerRuntime:
  """Small checked subprocess boundary used by the hook and its tests."""

  def __init__(self, config: HookConfig):
    self.config = config

  def run(
      self,
      args: Sequence[str],
      *,
      env: Mapping[str, str] | None = None,
      timeout: int | None = None,
  ) -> subprocess.CompletedProcess[str]:
    try:
      result = subprocess.run(
          list(args),
          check=False,
          capture_output=True,
          text=True,
          env=dict(env) if env is not None else None,
          timeout=timeout or self.config.command_timeout,
      )
    except (OSError, subprocess.TimeoutExpired) as exc:
      raise HookError(f"command could not complete: {args[0]}: {exc}") from exc
    if result.returncode != 0:
      stderr = (result.stderr or "").strip()[-2000:]
      raise HookError(
          f"command failed ({result.returncode}): {args[0]}: {stderr}"
      )
    return result

  def image_id(self, reference: str | None = None) -> str:
    image = reference or self.config.helper_image
    result = self.run([
        self.config.docker_bin,
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        image,
    ])
    image_id = result.stdout.strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
      raise HookError("helper image inspection returned an invalid image ID")
    return image_id

  def volume(self, name: str) -> dict[str, Any] | None:
    try:
      result = subprocess.run(
          [self.config.docker_bin, "volume", "inspect", name],
          check=False,
          capture_output=True,
          text=True,
          timeout=self.config.command_timeout,
      )
    except (OSError, subprocess.TimeoutExpired) as exc:
      raise HookError(f"Docker volume inspection failed: {exc}") from exc
    if result.returncode != 0:
      return None
    try:
      payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
      raise HookError("Docker returned invalid volume inspection JSON") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
      raise HookError("Docker returned an unexpected volume inspection shape")
    return payload[0]

  def containers_using_volume(self, name: str) -> list[str]:
    result = self.run([
        self.config.docker_bin,
        "ps",
        "-aq",
        "--filter",
        f"volume={name}",
    ])
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

  def create_volume(self, name: str, labels: Mapping[str, str]) -> None:
    args = [self.config.docker_bin, "volume", "create"]
    for key, value in sorted(labels.items()):
      args.extend(["--label", f"{key}={value}"])
    args.append(name)
    result = self.run(args)
    if result.stdout.strip() != name:
      raise HookError("Docker did not echo the requested created volume")

  def remove_volume(self, name: str) -> None:
    result = self.run([self.config.docker_bin, "volume", "rm", name])
    if result.stdout.strip() != name:
      raise HookError("Docker did not echo the requested removed volume")

  def copy_volume(self, source: str, target: str) -> None:
    self.run([
        self.config.docker_bin,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--mount",
        f"type=volume,src={source},dst=/source,readonly",
        "--mount",
        f"type=volume,src={target},dst=/target",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--entrypoint",
        "/bin/bash",
        self.config.helper_image,
        "-c",
        (
            "set -euo pipefail; "
            "[[ -z \"$(find /target -mindepth 1 -print -quit)\" ]]; "
            "cp -a --reflink=auto --sparse=always /source/. /target/; sync"
        ),
    ])

  def fingerprint_volume(self, name: str) -> str:
    result = self.run([
        self.config.docker_bin,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--mount",
        f"type=volume,src={name},dst=/snapshot,readonly",
        "--mount",
        f"type=bind,src={SCRIPT_PATH},dst=/opt/catbench/snapshot_hook.py,readonly",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--entrypoint",
        "python3",
        self.config.helper_image,
        "/opt/catbench/snapshot_hook.py",
        "--fingerprint-root",
        "/snapshot",
    ])
    value = result.stdout.strip()
    if not SHA256_RE.fullmatch(value):
      raise HookError("volume fingerprint helper returned an invalid digest")
    return value

  def manager(self, action: str, volume: str) -> None:
    if _sha256_path(self.config.manager) != self.config.manager_sha256:
      raise HookError("pool manager changed after hook configuration")
    if _sha256_path(self.config.start_script) != self.config.start_script_sha256:
      raise HookError("emulator start script changed after hook configuration")
    if (
        _sha256_path(self.config.base_start_script)
        != self.config.base_start_script_sha256
    ):
      raise HookError(
          "emulator base start script changed after hook configuration"
      )
    env = os.environ.copy()
    env.update({
        "NUM_EMULATORS": str(self.config.num_emulators),
        "FIRST_CONSOLE_PORT": str(self.config.first_console_port),
        "FIRST_GRPC_PORT": str(self.config.first_grpc_port),
        "FIRST_ADB_SERVER_PORT": str(self.config.first_adb_server_port),
        "NAME_PREFIX": self.config.name_prefix,
        "VOLUME_PREFIX": self.config.baseline_volume_prefix,
        "IMAGE": self.config.emulator_image,
        "START_SCRIPT": str(self.config.start_script),
        "BASE_START_SCRIPT": str(self.config.base_start_script),
        "CATBENCH_EMULATOR_MEMORY_MB": str(self.config.emulator_memory_mb),
        "CATBENCH_EMULATOR_CORES": str(self.config.emulator_cores),
        "ADB": self.config.adb,
        "EMULATOR_TIMEOUT": str(self.config.emulator_timeout),
        "DOCKER": self.config.docker_bin,
        "ADB_SERVER_PORT": str(self.config.expected_adb_server_port),
        "ANDROID_ADB_SERVER_PORT": str(self.config.expected_adb_server_port),
    })
    self.run(
        [
            str(self.config.manager),
            action,
            str(self.config.worker_index),
            volume,
        ],
        env=env,
    )

  def worker(self) -> dict[str, Any] | None:
    try:
      result = subprocess.run(
          [self.config.docker_bin, "container", "inspect", self.config.worker_name],
          check=False,
          capture_output=True,
          text=True,
          timeout=self.config.command_timeout,
      )
    except (OSError, subprocess.TimeoutExpired) as exc:
      raise HookError(f"Docker worker inspection failed: {exc}") from exc
    if result.returncode != 0:
      return None
    try:
      payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
      raise HookError("Docker returned invalid worker inspection JSON") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
      raise HookError("Docker returned an unexpected worker inspection shape")
    return payload[0]


def _labels(volume: Mapping[str, Any]) -> Mapping[str, str]:
  labels = volume.get("Labels")
  if not isinstance(labels, dict):
    raise HookError("Docker volume has no label map")
  return labels


def _clone_volume_name(config: HookConfig, request: Mapping[str, Any]) -> str:
  suffix = hashlib.sha256(request["snapshot_clone_id"].encode("utf-8")).hexdigest()
  name = f"{config.clone_volume_prefix}-{suffix}"
  if not DOCKER_NAME_RE.fullmatch(name):
    raise HookError("derived clone volume name is invalid")
  return name


def _base_labels(request: Mapping[str, Any]) -> dict[str, str]:
  return {
      "catbench.snapshot.role": "frozen-base",
      "catbench.snapshot.sealed": "true",
      "catbench.snapshot.id": str(request["base_snapshot_id"]),
      "catbench.snapshot.sha256": str(request["base_snapshot_sha256"]),
      "catbench.release_id": str(request["release_id"]),
      "catbench.cohort_sha256": str(request["cohort_sha256"]),
  }


def _clone_labels(
    config: HookConfig, request: Mapping[str, Any]
) -> dict[str, str]:
  return {
      "catbench.snapshot.role": "episode-clone",
      "catbench.snapshot.clone_id": str(request["snapshot_clone_id"]),
      "catbench.snapshot.family_id": str(request["snapshot_family_id"]),
      "catbench.snapshot.parent_id": str(request["base_snapshot_id"]),
      "catbench.snapshot.parent_sha256": str(request["base_snapshot_sha256"]),
      "catbench.snapshot.worker_index": str(config.worker_index),
      "catbench.attempt_id": str(request["attempt_id"]),
      "catbench.release_id": str(request["release_id"]),
      "catbench.cohort_sha256": str(request["cohort_sha256"]),
  }


def _require_label_subset(
    actual: Mapping[str, str], expected: Mapping[str, str], subject: str
) -> None:
  for key, value in expected.items():
    if actual.get(key) != value:
      raise HookError(
          f"{subject} label {key}={actual.get(key)!r}; expected {value!r}"
      )


def _verify_base(
    runtime: DockerRuntime,
    config: HookConfig,
    request: Mapping[str, Any],
) -> str:
  volume = runtime.volume(config.base_volume)
  if volume is None:
    raise HookError(f"base volume is missing: {config.base_volume}")
  _require_label_subset(_labels(volume), _base_labels(request), "base volume")
  users = runtime.containers_using_volume(config.base_volume)
  if users:
    raise HookError(f"base volume is attached to containers: {users}")
  observed = runtime.fingerprint_volume(config.base_volume)
  if observed != request["base_snapshot_sha256"]:
    raise HookError(
        f"base volume fingerprint mismatch: {observed} != "
        f"{request['base_snapshot_sha256']}"
    )
  return observed


def _worker_avd_mount(worker: Mapping[str, Any]) -> str:
  mounts = worker.get("Mounts")
  if not isinstance(mounts, list):
    raise HookError("worker inspection has no mount list")
  matches = [
      mount.get("Name")
      for mount in mounts
      if isinstance(mount, dict) and mount.get("Destination") == "/root/.android"
  ]
  if len(matches) != 1 or not isinstance(matches[0], str):
    raise HookError("worker does not have exactly one named AVD volume")
  return matches[0]


def _verify_worker(
    runtime: DockerRuntime,
    config: HookConfig,
    request: Mapping[str, Any],
    clone_volume: str,
    *,
    require_running: bool,
) -> None:
  worker = runtime.worker()
  if worker is None:
    raise HookError(f"worker container is missing: {config.worker_name}")
  if _worker_avd_mount(worker) != clone_volume:
    raise HookError("worker is not mounted from the requested episode clone")
  container_config = worker.get("Config")
  if not isinstance(container_config, dict):
    raise HookError("worker inspection lacks Config")
  labels = container_config.get("Labels")
  if not isinstance(labels, dict):
    raise HookError("worker inspection lacks labels")
  expected_labels = {
      "catbench.pool": config.name_prefix,
      "catbench.worker_index": str(config.worker_index),
      "catbench.avd_volume": clone_volume,
      "catbench.snapshot_clone_id": str(request["snapshot_clone_id"]),
      "catbench.launcher_sha256": config.start_script_sha256,
      "catbench.base_launcher_sha256": config.base_start_script_sha256,
      "catbench.emulator_memory_mb": str(config.emulator_memory_mb),
      "catbench.emulator_cores": str(config.emulator_cores),
  }
  _require_label_subset(labels, expected_labels, "worker")
  env_rows = container_config.get("Env")
  if not isinstance(env_rows, list):
    raise HookError("worker inspection lacks environment")
  env = {
      row.split("=", 1)[0]: row.split("=", 1)[1]
      for row in env_rows
      if isinstance(row, str) and "=" in row
  }
  if env.get("ANDROID_SERIAL") != request["device_serial"]:
    raise HookError("worker ANDROID_SERIAL does not match the request")
  if env.get("ADB_SERVER_PORT") != str(config.expected_adb_server_port):
    raise HookError("worker ADB server port does not match hook configuration")
  if env.get("CATBENCH_EMULATOR_MEMORY_MB") != str(config.emulator_memory_mb):
    raise HookError("worker emulator memory does not match hook configuration")
  if env.get("CATBENCH_EMULATOR_CORES") != str(config.emulator_cores):
    raise HookError("worker emulator cores do not match hook configuration")
  state = worker.get("State")
  if require_running and (
      not isinstance(state, dict) or state.get("Running") is not True
  ):
    raise HookError("activated worker is not running")


def _receipt_base(
    request: Mapping[str, Any],
    request_sha256: str,
    config: HookConfig,
    *,
    helper_image_id: str,
    emulator_image_id: str,
) -> dict[str, Any]:
  return {
      "schema_version": 1,
      "operation": request["operation"],
      "success": True,
      "request_sha256": request_sha256,
      "release_id": request["release_id"],
      "pair_id": request["pair_id"],
      "attempt_id": request["attempt_id"],
      "snapshot_family_id": request["snapshot_family_id"],
      "snapshot_clone_id": request["snapshot_clone_id"],
      "base_snapshot_id": request["base_snapshot_id"],
      "base_snapshot_sha256": request["base_snapshot_sha256"],
      "parent_snapshot_id": request["base_snapshot_id"],
      "parent_snapshot_sha256": request["base_snapshot_sha256"],
      "clone_generation": 1,
      "device_serial": request["device_serial"],
      "hook_revision": f"sha256_{_sha256_path(SCRIPT_PATH)}",
      "pool_manager_sha256": config.manager_sha256,
      "emulator_start_script_sha256": config.start_script_sha256,
      "emulator_base_start_script_sha256": (
          config.base_start_script_sha256
      ),
      "emulator_memory_mb": config.emulator_memory_mb,
      "emulator_cores": config.emulator_cores,
      "helper_image": config.helper_image,
      "helper_image_id": helper_image_id,
      "emulator_image": config.emulator_image,
      "emulator_image_id": emulator_image_id,
      "completed_at": _utc_now(),
  }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
  if path.exists() or path.is_symlink():
    raise HookError(f"refusing to replace existing output: {path}")
  parent = path.parent
  if parent.is_symlink() or not parent.is_dir():
    raise HookError(f"output parent must be a real directory: {parent}")
  data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
  fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
  tmp = Path(tmp_name)
  try:
    with os.fdopen(fd, "wb") as handle:
      handle.write(data)
      handle.flush()
      os.fsync(handle.fileno())
    os.link(tmp, path)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
      os.fsync(parent_fd)
    finally:
      os.close(parent_fd)
  finally:
    tmp.unlink(missing_ok=True)


def _cleanup_failed_clone(
    runtime: DockerRuntime, config: HookConfig, clone_volume: str
) -> list[str]:
  errors: list[str] = []
  try:
    worker = runtime.worker()
    if worker is not None and _worker_avd_mount(worker) == clone_volume:
      runtime.manager("deactivate-volume", clone_volume)
  except Exception as exc:  # cleanup errors are reported with the primary failure
    errors.append(f"deactivate:{type(exc).__name__}:{exc}")
  try:
    if runtime.volume(clone_volume) is not None:
      users = runtime.containers_using_volume(clone_volume)
      if users:
        errors.append(f"volume_still_attached:{users}")
      else:
        runtime.remove_volume(clone_volume)
  except Exception as exc:  # cleanup errors are reported with the primary failure
    errors.append(f"volume_remove:{type(exc).__name__}:{exc}")
  return errors


def clone_activate(
    runtime: DockerRuntime,
    config: HookConfig,
    request: Mapping[str, Any],
    request_sha256: str,
) -> dict[str, Any]:
  helper_image_id = runtime.image_id(config.helper_image)
  emulator_image_id = runtime.image_id(config.emulator_image)
  _verify_base(runtime, config, request)
  clone_volume = _clone_volume_name(config, request)
  if runtime.volume(clone_volume) is not None:
    raise HookError(f"episode clone already exists; refusing reuse: {clone_volume}")
  runtime.create_volume(clone_volume, _clone_labels(config, request))
  try:
    runtime.copy_volume(config.base_volume, clone_volume)
    # A second base hash closes the copy/hash TOCTOU window.
    _verify_base(runtime, config, request)
    clone_sha256 = runtime.fingerprint_volume(clone_volume)
    if clone_sha256 != request["base_snapshot_sha256"]:
      raise HookError("materialized clone does not equal the attested base")
    runtime.manager("activate-volume", clone_volume)
    _verify_worker(
        runtime,
        config,
        request,
        clone_volume,
        require_running=True,
    )
  except Exception as exc:
    cleanup_errors = _cleanup_failed_clone(runtime, config, clone_volume)
    suffix = f"; cleanup_errors={cleanup_errors}" if cleanup_errors else ""
    raise HookError(f"clone activation failed: {exc}{suffix}") from exc
  receipt = _receipt_base(
      request,
      request_sha256,
      config,
      helper_image_id=helper_image_id,
      emulator_image_id=emulator_image_id,
  )
  receipt.update({
      "active_snapshot_clone_id": request["snapshot_clone_id"],
      "active_snapshot_sha256": clone_sha256,
      "active_volume_name": clone_volume,
      "active_worker_name": config.worker_name,
      "fingerprint_phase": "after_materialization_before_emulator_boot",
  })
  return receipt


def release_clone(
    runtime: DockerRuntime,
    config: HookConfig,
    request: Mapping[str, Any],
    request_sha256: str,
) -> dict[str, Any]:
  helper_image_id = runtime.image_id(config.helper_image)
  emulator_image_id = runtime.image_id(config.emulator_image)
  _verify_base(runtime, config, request)
  clone_volume = _clone_volume_name(config, request)
  volume = runtime.volume(clone_volume)
  if volume is None:
    raise HookError(f"episode clone is missing: {clone_volume}")
  _require_label_subset(
      _labels(volume), _clone_labels(config, request), "episode clone"
  )
  _verify_worker(
      runtime,
      config,
      request,
      clone_volume,
      require_running=False,
  )
  runtime.manager("deactivate-volume", clone_volume)
  if runtime.worker() is not None:
    raise HookError("worker container still exists after deactivation")
  users = runtime.containers_using_volume(clone_volume)
  if users:
    raise HookError(f"episode clone remains attached after deactivation: {users}")
  released_sha256 = runtime.fingerprint_volume(clone_volume)
  runtime.remove_volume(clone_volume)
  if runtime.volume(clone_volume) is not None:
    raise HookError("episode clone still exists after Docker volume removal")
  receipt = _receipt_base(
      request,
      request_sha256,
      config,
      helper_image_id=helper_image_id,
      emulator_image_id=emulator_image_id,
  )
  receipt.update({
      "released_snapshot_clone_id": request["snapshot_clone_id"],
      "released_snapshot_sha256": released_sha256,
      "released_volume_name": clone_volume,
      "released_volume_deleted": True,
      "fingerprint_phase": "after_emulator_stop_before_volume_deletion",
  })
  return receipt


def run_hook(
    request_path: Path,
    receipt_path: Path,
    *,
    environ: Mapping[str, str] | None = None,
    runtime_factory: Any = DockerRuntime,
) -> dict[str, Any]:
  payload, request_sha256 = _strict_json(request_path)
  request = validate_request(payload)
  config = HookConfig.from_env(
      request, os.environ if environ is None else environ
  )
  config.lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
  base_lock_key = hashlib.sha256(config.base_volume.encode("utf-8")).hexdigest()
  base_lock_path = config.lock_dir / f"base-{base_lock_key}.lock"
  worker_lock_path = config.lock_dir / f"worker-{config.worker_index}.lock"
  # All workers share the base lock while verifying/materializing.  The worker
  # lock additionally prevents concurrent consumers from replacing one device.
  # The fixed acquisition order prevents cross-worker deadlocks.
  with base_lock_path.open("a+b") as base_lock, worker_lock_path.open(
      "a+b"
  ) as worker_lock:
    fcntl.flock(base_lock.fileno(), fcntl.LOCK_EX)
    fcntl.flock(worker_lock.fileno(), fcntl.LOCK_EX)
    runtime = runtime_factory(config)
    if request["operation"] == "clone_activate":
      receipt = clone_activate(runtime, config, request, request_sha256)
    else:
      receipt = release_clone(runtime, config, request, request_sha256)
    _write_json_exclusive(receipt_path, receipt)
    return receipt


def seal_base(
    *,
    source_volume: str,
    base_volume: str,
    snapshot_id: str,
    release_id: str,
    cohort_sha256: str,
    evidence_path: Path,
    environ: Mapping[str, str] | None = None,
    runtime_factory: Any = DockerRuntime,
) -> dict[str, Any]:
  environ_dict = dict(os.environ if environ is None else environ)
  # Reuse the checked Docker configuration without pretending this is an
  # episode.  The synthetic request is never emitted as schedule evidence.
  worker_index = int(environ_dict.get("CATBENCH_DOCKER_WORKER_INDEX", "0"))
  first_console = int(environ_dict.get("CATBENCH_DOCKER_FIRST_CONSOLE_PORT", "5576"))
  environ_dict["CATBENCH_DOCKER_WORKER_INDEX"] = str(worker_index)
  environ_dict["ANDROID_ADB_SERVER_PORT"] = str(
      int(environ_dict.get("CATBENCH_DOCKER_FIRST_ADB_SERVER_PORT", "5041"))
      + worker_index
  )
  environ_dict["CATBENCH_DOCKER_BASE_VOLUME"] = base_volume
  synthetic = {
      "device_serial": f"emulator-{first_console + 2 * worker_index}",
  }
  config = HookConfig.from_env(synthetic, environ_dict)
  for subject, value in (
      ("source volume", source_volume),
      ("base volume", base_volume),
  ):
    if not DOCKER_NAME_RE.fullmatch(value):
      raise HookError(f"invalid Docker {subject}: {value!r}")
  if not SAFE_ID_RE.fullmatch(snapshot_id) or not SAFE_ID_RE.fullmatch(release_id):
    raise HookError("snapshot_id and release_id must be safe non-empty IDs")
  if not SHA256_RE.fullmatch(cohort_sha256):
    raise HookError("cohort_sha256 must be a lowercase SHA-256")
  runtime = runtime_factory(config)
  runtime.image_id(config.helper_image)
  if runtime.volume(source_volume) is None:
    raise HookError(f"source volume is missing: {source_volume}")
  users = runtime.containers_using_volume(source_volume)
  if users:
    raise HookError(f"source volume must be offline and unattached: {users}")
  if runtime.volume(base_volume) is not None:
    raise HookError(f"base volume already exists: {base_volume}")
  source_before = runtime.fingerprint_volume(source_volume)
  labels = {
      "catbench.snapshot.role": "frozen-base",
      "catbench.snapshot.sealed": "true",
      "catbench.snapshot.id": snapshot_id,
      "catbench.snapshot.sha256": source_before,
      "catbench.release_id": release_id,
      "catbench.cohort_sha256": cohort_sha256,
  }
  runtime.create_volume(base_volume, labels)
  try:
    runtime.copy_volume(source_volume, base_volume)
    source_after = runtime.fingerprint_volume(source_volume)
    base_sha256 = runtime.fingerprint_volume(base_volume)
    if source_before != source_after or base_sha256 != source_before:
      raise HookError("source changed during sealing or base copy differs")
  except Exception:
    if not runtime.containers_using_volume(base_volume):
      runtime.remove_volume(base_volume)
    raise
  evidence = {
      "schema_version": 1,
      "evidence_type": "catbench_docker_avd_base_volume_seal",
      "approval_status": "observational_not_release_approval",
      "snapshot_id": snapshot_id,
      "snapshot_sha256": base_sha256,
      "release_id": release_id,
      "cohort_sha256": cohort_sha256,
      "source_volume": source_volume,
      "base_volume": base_volume,
      "base_volume_labels": labels,
      "helper_image": config.helper_image,
      "helper_image_id": runtime.image_id(config.helper_image),
      "hook_revision": f"sha256_{_sha256_path(SCRIPT_PATH)}",
      "sealed_at": _utc_now(),
  }
  _write_json_exclusive(evidence_path, evidence)
  return evidence


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--request", type=Path)
  parser.add_argument("--receipt", type=Path)
  parser.add_argument("--fingerprint-root", type=Path)
  parser.add_argument("--seal-base", action="store_true")
  parser.add_argument("--source-volume")
  parser.add_argument("--base-volume")
  parser.add_argument("--snapshot-id")
  parser.add_argument("--release-id")
  parser.add_argument("--cohort-sha256")
  parser.add_argument("--evidence", type=Path)
  args = parser.parse_args(argv)
  modes = sum((
      args.fingerprint_root is not None,
      args.seal_base,
      args.request is not None or args.receipt is not None,
  ))
  if modes != 1:
    parser.error("select exactly one of hook, --fingerprint-root, or --seal-base")
  if args.request is not None or args.receipt is not None:
    if args.request is None or args.receipt is None:
      parser.error("hook mode requires both --request and --receipt")
  if args.seal_base and not all((
      args.source_volume,
      args.base_volume,
      args.snapshot_id,
      args.release_id,
      args.cohort_sha256,
      args.evidence,
  )):
    parser.error(
        "--seal-base requires --source-volume, --base-volume, --snapshot-id, "
        "--release-id, --cohort-sha256, and --evidence"
    )
  return args


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv if argv is not None else sys.argv[1:])
  try:
    if args.fingerprint_root is not None:
      print(fingerprint_root(args.fingerprint_root))
    elif args.seal_base:
      seal_base(
          source_volume=args.source_volume,
          base_volume=args.base_volume,
          snapshot_id=args.snapshot_id,
          release_id=args.release_id,
          cohort_sha256=args.cohort_sha256,
          evidence_path=args.evidence.resolve(),
      )
    else:
      run_hook(args.request.resolve(), args.receipt.resolve())
  except HookError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 2
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
