"""Fail-closed contract for exact historical CATBench task parameters.

This module deliberately contains no parameter-generation logic.  It only
loads an audited JSON mapping, verifies its byte identity, and validates the
small interchange schema used by invalid-episode replacement runs.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
MODE = "historical_invalid_episode_replacement_instance0"
ENV_FILE = "CATBENCH_EXACT_TASK_PARAMS_FILE"
ENV_SHA256 = "CATBENCH_EXACT_TASK_PARAMS_SHA256"
ENV_MODE = "CATBENCH_EXACT_TASK_PARAMS_MODE"
ENV_SOURCE_FILE = "CATBENCH_EXACT_TASK_PARAMS_SOURCE_FILE"
ENV_SOURCE_SHA256 = "CATBENCH_EXACT_TASK_PARAMS_SOURCE_SHA256"
ENV_GOAL_OVERRIDE_ENABLED = "CATBENCH_EXACT_GOAL_OVERRIDE_ENABLED"
ENV_GOAL_MAPPING_SHA256 = "CATBENCH_EXACT_GOAL_MAPPING_SHA256"
ENV_NAMES = (
    ENV_FILE,
    ENV_SHA256,
    ENV_MODE,
    ENV_SOURCE_FILE,
    ENV_SOURCE_SHA256,
    ENV_GOAL_OVERRIDE_ENABLED,
    ENV_GOAL_MAPPING_SHA256,
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ROOT_FIELDS = frozenset({"schema_version", "mode", "source", "overrides"})
_SOURCE_FIELDS = frozenset({"file", "sha256"})
_ENTRY_FIELDS = frozenset(
    {"instance_id", "params", "expected_goal", "expected_seed"}
)


@dataclass(frozen=True)
class ExactTaskParamsBundle:
  """Validated immutable view of an exact-parameter JSON file."""

  path: Path
  sha256: str
  mode: str
  source_path: Path
  source_sha256: str
  overrides: dict[str, dict[str, Any]]


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  value: dict[str, Any] = {}
  for key, item in pairs:
    if key in value:
      raise ValueError(f"Duplicate JSON object key: {key!r}")
    value[key] = item
  return value


def _reject_non_finite(value: str) -> None:
  raise ValueError(f"Non-finite JSON number is forbidden: {value}")


def _read_json_object(path: Path) -> dict[str, Any]:
  try:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite,
    )
  except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
    raise ValueError(
        f"Unable to read exact task-parameter JSON {path}: {error}"
    ) from error
  if not isinstance(payload, dict):
    raise ValueError("Exact task-parameter JSON root must be an object.")
  return payload


def load_bundle(
    path: Path,
    *,
    expected_sha256: str,
    expected_mode: str = MODE,
) -> ExactTaskParamsBundle:
  """Loads a byte-pinned override file and validates its closed schema."""
  resolved = path.expanduser().resolve()
  expected_hash = str(expected_sha256 or "").strip().lower()
  if not _SHA256_RE.fullmatch(expected_hash):
    raise ValueError(
        "An exact task-parameter file requires an explicit 64-hex SHA-256."
    )
  if not resolved.is_file():
    raise ValueError(f"Exact task-parameter file does not exist: {resolved}")
  actual_hash = file_sha256(resolved)
  if actual_hash != expected_hash:
    raise ValueError(
        "Exact task-parameter file SHA-256 mismatch: "
        f"expected={expected_hash} actual={actual_hash} file={resolved}"
    )

  payload = _read_json_object(resolved)
  fields = set(payload)
  if fields != _ROOT_FIELDS:
    raise ValueError(
        "Exact task-parameter root fields mismatch: "
        f"missing={sorted(_ROOT_FIELDS - fields)}, "
        f"extra={sorted(fields - _ROOT_FIELDS)}"
    )
  version = payload["schema_version"]
  if isinstance(version, bool) or version != SCHEMA_VERSION:
    raise ValueError(
        f"Unsupported exact task-parameter schema_version={version!r}; "
        f"expected {SCHEMA_VERSION}."
    )
  mode = payload["mode"]
  if mode != expected_mode or mode != MODE:
    raise ValueError(
        f"Exact task-parameter mode={mode!r}; expected {MODE!r}."
    )
  source = payload["source"]
  if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
    source_fields = set(source) if isinstance(source, dict) else set()
    raise ValueError(
        "Exact task-parameter source fields mismatch: "
        f"missing={sorted(_SOURCE_FIELDS - source_fields)}, "
        f"extra={sorted(source_fields - _SOURCE_FIELDS)}"
    )
  raw_source_path = source["file"]
  if not isinstance(raw_source_path, str) or not raw_source_path.strip():
    raise ValueError("Exact task-parameter source.file must be a path string.")
  unresolved_source_path = Path(raw_source_path).expanduser()
  if not unresolved_source_path.is_absolute():
    raise ValueError("Exact task-parameter source.file must be absolute.")
  source_path = unresolved_source_path.resolve()
  source_sha256 = str(source["sha256"] or "").strip().lower()
  if not _SHA256_RE.fullmatch(source_sha256):
    raise ValueError("Exact task-parameter source.sha256 must be 64 hex digits.")
  if not source_path.is_file():
    raise ValueError(
        f"Exact task-parameter canonical source does not exist: {source_path}"
    )
  actual_source_sha256 = file_sha256(source_path)
  if actual_source_sha256 != source_sha256:
    raise ValueError(
        "Exact task-parameter canonical source SHA-256 mismatch: "
        f"expected={source_sha256} actual={actual_source_sha256}"
    )
  raw_overrides = payload["overrides"]
  if not isinstance(raw_overrides, dict) or not raw_overrides:
    raise ValueError("Exact task-parameter overrides must be a non-empty object.")

  overrides: dict[str, dict[str, Any]] = {}
  for task_name, entry in raw_overrides.items():
    if not isinstance(task_name, str) or not task_name.strip():
      raise ValueError(f"Invalid override task name: {task_name!r}")
    if task_name != task_name.strip():
      raise ValueError(f"Override task name contains outer whitespace: {task_name!r}")
    if not isinstance(entry, dict):
      raise ValueError(f"Override for {task_name} must be an object.")
    entry_fields = set(entry)
    if entry_fields != _ENTRY_FIELDS:
      raise ValueError(
          f"Override fields mismatch for {task_name}: "
          f"missing={sorted(_ENTRY_FIELDS - entry_fields)}, "
          f"extra={sorted(entry_fields - _ENTRY_FIELDS)}"
      )
    instance_id = entry["instance_id"]
    if isinstance(instance_id, bool) or instance_id != 0:
      raise ValueError(
          f"Override for {task_name} must have instance_id=0; "
          f"got {instance_id!r}."
      )
    params = entry["params"]
    if not isinstance(params, dict):
      raise ValueError(f"Override params for {task_name} must be an object.")
    expected_goal = entry["expected_goal"]
    if not isinstance(expected_goal, str) or not expected_goal:
      raise ValueError(
          f"Override expected_goal for {task_name} must be a non-empty string."
      )
    expected_seed = entry["expected_seed"]
    if isinstance(expected_seed, bool) or not isinstance(expected_seed, int):
      raise ValueError(
          f"Override expected_seed for {task_name} must be an integer."
      )
    actual_seed = params.get("seed")
    if (
        isinstance(actual_seed, bool)
        or not isinstance(actual_seed, int)
        or actual_seed != expected_seed
    ):
      raise ValueError(
          f"Override seed mismatch for {task_name}: "
          f"params.seed={actual_seed!r}, expected_seed={expected_seed!r}."
      )
    overrides[task_name] = entry

  return ExactTaskParamsBundle(
      path=resolved,
      sha256=actual_hash,
      mode=mode,
      source_path=source_path,
      source_sha256=source_sha256,
      overrides=overrides,
  )


def require_exact_task_names(
    bundle: ExactTaskParamsBundle,
    expected_tasks: Iterable[str],
    *,
    registry_names: Iterable[str] | None = None,
) -> None:
  """Requires a one-to-one override mapping for the requested task classes."""
  expected_list = list(expected_tasks)
  if len(expected_list) != len(set(expected_list)):
    raise ValueError("Expected override task names contain duplicates.")
  expected = set(expected_list)
  actual = set(bundle.overrides)
  if actual != expected:
    raise ValueError(
        "Exact task-parameter task set mismatch: "
        f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
    )
  if registry_names is not None:
    registry = set(registry_names)
    unknown = sorted(actual - registry)
    if unknown:
      raise ValueError(
          "Exact task-parameter file names tasks absent from the registry: "
          + ", ".join(unknown)
      )


def projected_payload(
    bundle: ExactTaskParamsBundle, task_names: Iterable[str]
) -> dict[str, Any]:
  """Returns a closed-schema subset without modifying any override data."""
  names = list(task_names)
  missing = sorted(set(names) - set(bundle.overrides))
  if missing:
    raise ValueError(f"Cannot project missing exact overrides: {missing}")
  return {
      "schema_version": SCHEMA_VERSION,
      "mode": bundle.mode,
      "source": {
          "file": str(bundle.source_path),
          "sha256": bundle.source_sha256,
      },
      "overrides": {name: bundle.overrides[name] for name in names},
  }
