#!/usr/bin/env python3
"""Hash and provenance-audit the frozen cohort's local model snapshots.

This utility is deliberately read-only and initiates no artifact fetch or model
request. The supplied cache may itself be a pre-mounted network filesystem; its
mount provenance is recorded rather than hidden. The audit proves that a
complete set of checkpoint bytes exists under the configured Hugging Face
revision directory; it does *not* prove that an inference endpoint loaded or
served those bytes. The latter requires a separate endpoint attestation tied to
the release and is enforced by ``consume_catbench_frozen_schedule.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
WEIGHT_SUFFIXES = (".safetensors", ".bin")


class AuditError(RuntimeError):
  """A frozen artifact is absent, ambiguous, mutable, or provenance-invalid."""


def _sha256_path(path: Path) -> tuple[str, int]:
  before = path.stat(follow_symlinks=True)
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  after = path.stat(follow_symlinks=True)
  if (
      before.st_dev != after.st_dev
      or before.st_ino != after.st_ino
      or before.st_size != after.st_size
      or before.st_mtime_ns != after.st_mtime_ns
      or before.st_ctime_ns != after.st_ctime_ns
  ):
    raise AuditError(f"artifact changed while hashing: {path}")
  return digest.hexdigest(), before.st_size


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
  return json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode("utf-8")


def _filesystem_provenance(path: Path) -> dict[str, Any]:
  findmnt = shutil.which("findmnt")
  if findmnt is None:
    return {
        "detection_status": "findmnt_unavailable",
        "path": str(path),
        "network_backed": "unknown",
    }
  completed = subprocess.run(
      [
          findmnt,
          "--json",
          "--target",
          str(path),
          "--output",
          "TARGET,SOURCE,FSTYPE,OPTIONS",
      ],
      check=False,
      capture_output=True,
      text=True,
      timeout=10,
  )
  if completed.returncode != 0:
    return {
        "detection_status": "findmnt_failed",
        "path": str(path),
        "network_backed": "unknown",
        "returncode": completed.returncode,
    }
  try:
    payload = json.loads(completed.stdout)
    filesystems = payload["filesystems"]
    filesystem = filesystems[0]
  except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
    raise AuditError(f"cannot parse findmnt output for {path}: {exc}") from exc
  fs_type = str(filesystem.get("fstype") or "")
  network_types = {
      "9p", "afs", "ceph", "cifs", "fuse.sshfs", "glusterfs", "nfs",
      "nfs4", "smb3", "sshfs",
  }
  options = str(filesystem.get("options") or "")
  return {
      "detection_status": "detected",
      "path": str(path),
      "mount_target": str(filesystem.get("target") or ""),
      "mount_source": str(filesystem.get("source") or ""),
      "filesystem_type": fs_type,
      "mount_options": options,
      "read_only": "ro" in options.split(","),
      "network_backed": fs_type.casefold() in network_types,
  }


def _strict_json(path: Path, *, validated_artifact_symlink: bool = False) -> Any:
  if (path.is_symlink() and not validated_artifact_symlink) or not path.is_file():
    raise AuditError(f"JSON input must be a regular non-symlink file: {path}")

  def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise AuditError(f"duplicate JSON key {key!r} in {path}")
      result[key] = value
    return result

  def reject_constant(value: str) -> None:
    raise AuditError(f"non-finite JSON constant {value!r} in {path}")

  try:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise AuditError(f"invalid JSON input {path}: {exc}") from exc


def _is_relative_to(path: Path, parent: Path) -> bool:
  try:
    path.relative_to(parent)
    return True
  except ValueError:
    return False


def _safe_weight_name(value: Any) -> str:
  if not isinstance(value, str) or not value:
    raise AuditError("weight index contains an invalid shard name")
  pure = PurePosixPath(value)
  if pure.is_absolute() or ".." in pure.parts or value != pure.as_posix():
    raise AuditError(f"weight index contains an unsafe shard path: {value!r}")
  return value


def _expected_weight_files(snapshot: Path) -> set[str]:
  index_path = snapshot / "model.safetensors.index.json"
  if index_path.exists():
    # The caller first validates every snapshot symlink as contained in this
    # repository's content-addressed blobs directory.
    index = _strict_json(index_path, validated_artifact_symlink=True)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
      raise AuditError(f"invalid or empty weight_map: {index_path}")
    expected = {_safe_weight_name(value) for value in weight_map.values()}
  else:
    safetensors = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*.safetensors")
        if path.is_file()
    }
    expected = safetensors or {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("pytorch_model*.bin")
        if path.is_file()
    }
  if not expected:
    raise AuditError(f"snapshot has no model weight files: {snapshot}")
  discovered = {
      path.relative_to(snapshot).as_posix()
      for path in snapshot.rglob("*")
      if path.is_file()
      and (
          path.name.endswith(".safetensors")
          or path.name.startswith("pytorch_model") and path.name.endswith(".bin")
      )
  }
  if discovered != expected:
    raise AuditError(
        f"weight index/file mismatch for {snapshot}: "
        f"missing={sorted(expected - discovered)}, extra={sorted(discovered - expected)}"
    )
  return expected


def _snapshot_entries(snapshot: Path, repository_root: Path) -> list[dict[str, Any]]:
  blobs_root = (repository_root / "blobs").resolve()
  entries: list[dict[str, Any]] = []
  for path in sorted(snapshot.rglob("*"), key=lambda item: item.relative_to(snapshot).as_posix()):
    relative = path.relative_to(snapshot).as_posix()
    if path.is_dir() and not path.is_symlink():
      continue
    if path.is_symlink():
      resolved = path.resolve(strict=True)
      if not resolved.is_file() or not _is_relative_to(resolved, blobs_root):
        raise AuditError(
            f"snapshot symlink must resolve to this repository's blobs: {path}"
        )
      entries.append({
          "relative_path": relative,
          "resolved_path": resolved,
          "storage": "huggingface_blob_symlink",
          "blob_name": resolved.name,
      })
    elif path.is_file():
      resolved = path.resolve(strict=True)
      if not _is_relative_to(resolved, snapshot.resolve()):
        raise AuditError(f"snapshot file escapes revision directory: {path}")
      entries.append({
          "relative_path": relative,
          "resolved_path": resolved,
          "storage": "snapshot_regular_file",
          "blob_name": "",
      })
    else:
      raise AuditError(f"unsupported snapshot entry: {path}")
  if not entries:
    raise AuditError(f"empty model snapshot: {snapshot}")
  return entries


def _hash_entries(
    entries: list[dict[str, Any]], workers: int
) -> list[dict[str, Any]]:
  unique_paths = sorted({entry["resolved_path"] for entry in entries})
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {executor.submit(_sha256_path, path): path for path in unique_paths}
    by_path: dict[Path, tuple[str, int]] = {}
    for future in concurrent.futures.as_completed(futures):
      path = futures[future]
      by_path[path] = future.result()
  output = []
  for entry in entries:
    digest, size = by_path[entry["resolved_path"]]
    output.append({
        "relative_path": entry["relative_path"],
        "storage": entry["storage"],
        "blob_name": entry["blob_name"],
        "size_bytes": size,
        "sha256": digest,
    })
  return output


def _audit_one_model(
    model: dict[str, Any], cache_root: Path, workers: int
) -> dict[str, Any]:
  name = model.get("name")
  repository = model.get("repository")
  revision = model.get("revision")
  if not isinstance(name, str) or not name:
    raise AuditError("model config contains an empty model name")
  if (
      not isinstance(repository, str)
      or repository.count("/") != 1
      or any(part in ("", ".", "..") for part in repository.split("/"))
  ):
    raise AuditError(f"model {name} lacks a safe explicit repository")
  if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
    raise AuditError(f"model {name} lacks an immutable commit revision")
  repository_root = cache_root / f"models--{repository.replace('/', '--')}"
  snapshot = repository_root / "snapshots" / revision
  if snapshot.is_symlink() or not snapshot.is_dir():
    raise AuditError(f"model {name} snapshot is missing or symlinked: {snapshot}")
  entries = _snapshot_entries(snapshot, repository_root)
  expected_weights = _expected_weight_files(snapshot)
  records = _hash_entries(entries, workers)
  by_relative = {record["relative_path"]: record for record in records}
  missing = expected_weights - set(by_relative)
  if missing:
    raise AuditError(f"model {name} has missing hashed weights: {sorted(missing)}")
  weight_records = [by_relative[path] for path in sorted(expected_weights)]
  for record in weight_records:
    blob_name = record["blob_name"]
    if record["storage"] == "huggingface_blob_symlink":
      if not SHA256_RE.fullmatch(blob_name) or blob_name != record["sha256"]:
        raise AuditError(
            f"model {name} weight blob name/hash mismatch: {record['relative_path']}"
        )
  portable_records = [
      {
          "relative_path": record["relative_path"],
          "storage": record["storage"],
          "blob_name": record["blob_name"],
          "size_bytes": record["size_bytes"],
          "sha256": record["sha256"],
      }
      for record in records
  ]
  return {
      "name": name,
      "repository": repository,
      "revision": revision,
      "snapshot_path": str(snapshot),
      "snapshot_entry_count": len(portable_records),
      "snapshot_total_bytes": sum(record["size_bytes"] for record in portable_records),
      "weight_file_count": len(weight_records),
      "weight_total_bytes": sum(record["size_bytes"] for record in weight_records),
      "weight_files": weight_records,
      "snapshot_manifest_sha256": _sha256_bytes(_canonical_json(portable_records)),
      "files": portable_records,
      "valid": True,
  }


def audit_models(
    *,
    cohort_path: Path,
    model_config_path: Path,
    cache_root: Path,
    workers: int,
) -> dict[str, Any]:
  cohort = _strict_json(cohort_path)
  config = _strict_json(model_config_path)
  if not isinstance(cohort, dict) or not isinstance(cohort.get("models"), list):
    raise AuditError("cohort models must be a list")
  configured = config.get("models") if isinstance(config, dict) else None
  if not isinstance(configured, list):
    raise AuditError("model config models must be a list")
  by_name: dict[str, dict[str, Any]] = {}
  for row in configured:
    if not isinstance(row, dict) or not isinstance(row.get("name"), str):
      raise AuditError("malformed model config row")
    if row["name"] in by_name:
      raise AuditError(f"duplicate configured model: {row['name']}")
    by_name[row["name"]] = row
  names = cohort["models"]
  if len(names) != len(set(names)) or any(not isinstance(name, str) for name in names):
    raise AuditError("cohort model roster is duplicated or malformed")
  missing = [name for name in names if name not in by_name]
  if missing:
    raise AuditError(f"cohort models missing from config: {missing}")
  cache_root = cache_root.resolve(strict=True)
  models = []
  for index, name in enumerate(names, start=1):
    print(
        f"HASHING_MODEL {index}/{len(names)} name={name}",
        file=sys.stderr,
        flush=True,
    )
    model = _audit_one_model(by_name[name], cache_root, workers)
    models.append(model)
    print(
        f"HASHED_MODEL {index}/{len(names)} name={name} "
        f"weight_bytes={model['weight_total_bytes']}",
        file=sys.stderr,
        flush=True,
    )
  return {
      "schema_version": 1,
      "evidence_type": "catbench_local_pinned_model_artifact_audit",
      "artifact_role": "checkpoint_bytes_identity_only_not_served_endpoint_evidence",
      "approval_status": "observational_not_model_endpoint_attestation",
      "analysis_eligible": False,
      "created_at": dt.datetime.now(dt.UTC).isoformat(),
      "cohort_path": str(cohort_path.resolve()),
      "cohort_sha256": _sha256_path(cohort_path)[0],
      "model_config_path": str(model_config_path.resolve()),
      "model_config_sha256": _sha256_path(model_config_path)[0],
      "collection_tool_sha256": _sha256_path(Path(__file__).resolve())[0],
      "cache_root": str(cache_root),
      "cache_storage": _filesystem_provenance(cache_root),
      "external_artifact_fetch_performed": False,
      "inference_endpoint_contacted": False,
      "model_count": len(models),
      "models": models,
      "valid_models": sum(bool(model["valid"]) for model in models),
      "invalid_models": sum(not bool(model["valid"]) for model in models),
      "valid": all(bool(model["valid"]) for model in models),
      "claim": (
          "Local checkpoint bytes exist at each configured repository/revision "
          "and match their content-addressed weight blobs. Bytes were read from "
          "the already-mounted cache whose filesystem provenance is recorded. "
          "No artifact fetch or inference request was initiated; this report "
          "cannot establish which weights an endpoint serves."
      ),
  }


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  try:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
  except FileExistsError as exc:
    raise AuditError(f"refusing to overwrite output: {path}") from exc
  with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cohort", required=True, type=Path)
  parser.add_argument("--model_config", required=True, type=Path)
  parser.add_argument("--cache_root", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--workers", type=int, default=4)
  return parser


def main(argv: Iterable[str] | None = None) -> int:
  args = _build_parser().parse_args(argv)
  if args.workers < 1 or args.workers > 32:
    print("FAIL: --workers must be in [1, 32]", file=sys.stderr)
    return 2
  try:
    payload = audit_models(
        cohort_path=args.cohort.resolve(),
        model_config_path=args.model_config.resolve(),
        cache_root=args.cache_root.resolve(),
        workers=args.workers,
    )
    _write_exclusive(args.output.resolve(), payload)
  except (AuditError, OSError, ValueError) as exc:
    print(f"FAIL: {exc}", file=sys.stderr)
    return 2
  print(json.dumps({
      "model_count": payload["model_count"],
      "valid_models": payload["valid_models"],
      "invalid_models": payload["invalid_models"],
      "valid": payload["valid"],
      "output": str(args.output.resolve()),
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
