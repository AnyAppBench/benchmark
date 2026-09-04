#!/usr/bin/env python3
"""Consume the frozen CATBench schedule one real episode at a time.

This is the fail-closed execution counterpart to
``build_catbench_frozen_schedule.py``.  It deliberately has no app, task,
model, condition, slot, or retry filter.  It accepts only the exact primary
release or the separate exact G6 discard-only release, validates the complete
bundle, and advances append-only state in the published schedule order.

Every attempt uses:

* one registered real task class resolved from ``(category, app_id,
  semantic_task_id)``;
* the original K=3 suite and ``CATBENCH_INSTANCE_ID`` selector, so an episode
  retains its frozen parameter seed and instance identity;
* a condition-specific, externally attested snapshot clone; and
* a conservative result contract parsed from exactly one checkpoint artifact.

Only ``invalid_infrastructure`` can authorize a replacement.  A replacement
always contains the complete C1/C2-G/C2-O triplet, is capped at two rounds,
and never consults reward, screenshots, or judge labels.  Analysis selection
is similarly round-pure: r0, r1, or r2 is selected only when all three members
of that exact round are valid.  Otherwise the pair remains pending or is
recorded as exhausted-invalid.

The checked-in primary cohort is blocked at the schedule-eligibility gate
until all 230 frozen adapters have independently approved G3 evidence.
Execution has the same gate plus the external plan, snapshot, app, and endpoint
attestations. The G6 path is not a primary subset override: it has another
release identity and every emitted artifact is marked ineligible for primary
analysis.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import datetime as dt
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import shlex
import shutil
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Protocol, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_ROOT.parent
for import_root in (BENCHMARK_ROOT, SCRIPT_DIR, REPO_ROOT):
  if str(import_root) not in sys.path:
    sys.path.insert(0, str(import_root))

import build_catbench_frozen_schedule as schedule_builder
import catbench_primary_cohort
import generate_task_breakdowns as breakdown_generator
from android_world import registry


CONDITIONS = schedule_builder.PRIMARY_CONDITIONS
VALID_TERMINAL_STATUSES = frozenset({
    "valid_success",
    "valid_failure",
    "invalid_infrastructure",
})
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
SNAPSHOT_HOOK_TIMEOUT_SECONDS = 300
EPISODE_RUNNER_TIMEOUT_SECONDS = 3600
SCHEDULE_MANIFEST_FILE = schedule_builder.MANIFEST_FILE
SCHEDULE_FILE = schedule_builder.SCHEDULE_FILE
LEDGER_SEED_FILE = schedule_builder.LEDGER_FILE
LEDGER_SCHEMA_FILE = schedule_builder.LEDGER_SCHEMA_FILE

CONSUMER_MANIFEST_FILE = "consumer_manifest.json"
JOURNAL_FILE = "attempt_journal.jsonl"
RUNTIME_LEDGER_FILE = "replacement_ledger_runtime.jsonl"
SELECTION_FILE = "selected_triplets.jsonl"
STATE_COMMIT_FILE = "state_commit.json"
LOCK_FILE = ".consumer.lock"
ATTEMPTS_DIR = "attempts"

DEFAULT_MODEL_CONFIG = BENCHMARK_ROOT / "configs" / "catbench_5cat_models.json"
DEFAULT_PINS = BENCHMARK_ROOT / "configs" / "app_versions_pinned.csv"
DEFAULT_APPS = BENCHMARK_ROOT / "app_generalization_apps.csv"

FROZEN_DOCKER_IMAGE = (
    "android_world@sha256:"
    "6d8b2c148aebd3a1fe626768efe22c01a7a62cdbd2cbbe7d3f973adc57c7dd2f"
)
VERIFIER_CONFORMANCE_POLICY = (
    "all_frozen_task_app_adapters_with_primitive_action_positive_reset_replay_"
    "and_six_negative_controls_v1"
)
VERIFIER_CONFORMANCE_ARTIFACT_ROLE = (
    "verifier_conformance_gate_only_not_model_result"
)
VERIFIER_CONFORMANCE_CASES = (
    "primitive_action_positive",
    "reset_replay",
    "no_op",
    "wrong_entity",
    "wrong_value",
    "partial",
    "stale",
    "unrelated",
)
PRIMARY_VERIFIER_ADAPTER_COUNT = 230

# These flags define the frozen episode identity or its only admissible output
# location.  Model-specific arguments are deliberately appended to the common
# runner command, so accepting a second spelling of any of these flags would
# let an otherwise valid model config replace the schedule's task, seed,
# checkpoint, or emulator endpoint.
RESERVED_RUNNER_FLAGS = frozenset({
    "checkpoint_dir",
    "console_port",
    "fixed_task_seed",
    "grpc_port",
    "n_task_combinations",
    "output_path",
    "suite_family",
    "task_random_seed",
    "tasks",
})
RUNNER_META_FLAGS = frozenset({"flagfile", "fromenv", "tryfromenv"})


class ConsumerError(RuntimeError):
  """Base class for a launch-blocking schedule-consumer error."""


class ArtifactValidationError(ConsumerError):
  """Raised when an immutable input or result artifact is inconsistent."""


class InfrastructureInvalid(ConsumerError):
  """Raised for a machine-detectable invalid episode attempt."""


def _canonical_json(value: Any) -> str:
  return json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  )


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _strict_json_loads(raw: str, source: str) -> Any:
  def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise ArtifactValidationError(
            f"Duplicate JSON key {key!r} in {source}"
        )
      result[key] = value
    return result

  def reject_constant(value: str) -> None:
    raise ArtifactValidationError(
        f"Non-finite JSON constant {value!r} in {source}"
    )

  try:
    return json.loads(
        raw,
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )
  except json.JSONDecodeError as exc:
    raise ArtifactValidationError(f"Invalid JSON in {source}: {exc}") from exc


def _read_json(path: Path) -> Any:
  if path.is_symlink() or not path.is_file():
    raise ArtifactValidationError(f"JSON input must be a regular non-symlink: {path}")
  try:
    return _strict_json_loads(path.read_text(encoding="utf-8"), str(path))
  except OSError as exc:
    raise ArtifactValidationError(f"Unable to read JSON {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  try:
    with path.open("r", encoding="utf-8") as handle:
      for line_number, raw_line in enumerate(handle, 1):
        if not raw_line.strip():
          raise ArtifactValidationError(
              f"Blank JSONL line in {path} at line {line_number}"
          )
        value = _strict_json_loads(raw_line, f"{path}:{line_number}")
        if not isinstance(value, dict):
          raise ArtifactValidationError(
              f"JSONL record is not an object in {path}:{line_number}"
          )
        rows.append(value)
  except OSError as exc:
    raise ArtifactValidationError(f"Unable to read JSONL {path}: {exc}") from exc
  return rows


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
  return "".join(_canonical_json(dict(row)) + "\n" for row in rows).encode(
      "utf-8"
  )


def _atomic_write(path: Path, data: bytes) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(path.name + ".tmp")
  try:
    with tmp.open("xb") as handle:
      handle.write(data)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(tmp, path)
  finally:
    try:
      tmp.unlink()
    except FileNotFoundError:
      pass


def _utc_now() -> str:
  return dt.datetime.now(dt.UTC).isoformat()


def _paths_overlap(first: Path, second: Path) -> bool:
  """Returns whether either resolved path contains the other."""
  first = first.resolve()
  second = second.resolve()
  return first == second or first in second.parents or second in first.parents


def _validate_prewrite_locations(
    output_root: Path, immutable_inputs: Iterable[Path]
) -> None:
  """Reject state paths that can mutate or shadow release/source inputs.

  This check is intentionally usable before ``output_root`` exists.  Primary
  state belongs outside the source checkout: otherwise creating the lock file
  would itself dirty the source revision that the consumer just attested.
  Immutable inputs also may not live below the mutable state root.
  """
  resolved_output = output_root.resolve()
  if _paths_overlap(resolved_output, REPO_ROOT):
    raise ArtifactValidationError(
        "Frozen-release output_root must be disjoint from the source repository"
    )
  for raw_path in immutable_inputs:
    resolved_input = raw_path.resolve()
    if resolved_input == resolved_output or resolved_output in resolved_input.parents:
      raise ArtifactValidationError(
          "Immutable release input may not be stored under mutable output_root: "
          f"{resolved_input}"
      )


def _validate_frozen_model_args(name: str, args: Sequence[str]) -> None:
  """Require self-contained model flags that cannot override frozen flags."""
  for arg in args:
    if not arg.startswith("--") or arg == "--" or "=" not in arg:
      raise ArtifactValidationError(
          f"Frozen model {name!r} argument must be --name=value: {arg!r}"
      )
    raw_name = arg[2:].split("=", 1)[0]
    normalized = raw_name.replace("-", "_")
    # Abseil accepts --noflag for boolean flags.  Treat that spelling as an
    # override too, even though the current frozen bool is already false.
    positive_name = normalized[2:] if normalized.startswith("no") else normalized
    if normalized in RUNNER_META_FLAGS or positive_name in RUNNER_META_FLAGS:
      raise ArtifactValidationError(
          f"Frozen model {name!r} may not inject runner meta-flag --{raw_name}"
      )
    if normalized in RESERVED_RUNNER_FLAGS or positive_name in RESERVED_RUNNER_FLAGS:
      raise ArtifactValidationError(
          f"Frozen model {name!r} may not override --{raw_name}"
      )


@dataclasses.dataclass(frozen=True)
class FrozenBundle:
  cohort: dict[str, Any]
  cohort_sha256: str
  schedule_manifest: dict[str, Any]
  schedule_manifest_sha256: str
  schedule: tuple[dict[str, Any], ...]
  ledger_seed: tuple[dict[str, Any], ...]
  ledger_schema_sha256: str


def load_and_validate_bundle(
    schedule_dir: Path, cohort_path: Path
) -> FrozenBundle:
  """Recompile and byte-compare the complete frozen schedule bundle."""
  strict_cohort = _read_json(cohort_path)
  cohort, cohort_sha256 = schedule_builder.load_frozen_cohort(cohort_path)
  if strict_cohort != cohort:
    raise ArtifactValidationError("Strict cohort parse differs from schedule builder")
  expected_schedule, expected_ledger, expected_manifest = (
      schedule_builder.compile_frozen_schedule(cohort, cohort_sha256)
  )

  manifest_path = schedule_dir / SCHEDULE_MANIFEST_FILE
  schedule_path = schedule_dir / SCHEDULE_FILE
  ledger_path = schedule_dir / LEDGER_SEED_FILE
  schema_path = schedule_dir / LEDGER_SCHEMA_FILE
  for path in (manifest_path, schedule_path, ledger_path, schema_path):
    if path.is_symlink() or not path.is_file():
      raise ArtifactValidationError(f"Required frozen schedule artifact missing: {path}")
  bundle_names = {path.name for path in schedule_dir.iterdir()}
  expected_names = {
      SCHEDULE_MANIFEST_FILE,
      SCHEDULE_FILE,
      LEDGER_SEED_FILE,
      LEDGER_SCHEMA_FILE,
  }
  if bundle_names != expected_names:
    raise ArtifactValidationError(
        "Frozen schedule directory must contain exactly the four compiler "
        f"artifacts; found {sorted(bundle_names)}"
    )

  manifest = _read_json(manifest_path)
  if not isinstance(manifest, dict):
    raise ArtifactValidationError("schedule_manifest.json must be an object")
  schedule = _read_jsonl(schedule_path)
  ledger = _read_jsonl(ledger_path)
  if _canonical_json(schedule) != _canonical_json(expected_schedule):
    raise ArtifactValidationError(
        "episode_schedule.jsonl differs from deterministic recompilation"
    )
  if _canonical_json(ledger) != _canonical_json(expected_ledger):
    raise ArtifactValidationError(
        "replacement_ledger_seed.jsonl differs from deterministic recompilation"
    )
  # Timestamps are intentionally absent from the compiler manifest, so exact
  # comparison is both stable and stronger than field-by-field checking.
  if _canonical_json(manifest) != _canonical_json(expected_manifest):
    raise ArtifactValidationError(
        "schedule_manifest.json differs from deterministic recompilation"
    )

  output_specs = manifest.get("outputs") or {}
  for filename, path in (
      (SCHEDULE_FILE, schedule_path),
      (LEDGER_SEED_FILE, ledger_path),
      (LEDGER_SCHEMA_FILE, schema_path),
  ):
    expected_hash = str((output_specs.get(filename) or {}).get("sha256") or "")
    actual_hash = _sha256_path(path)
    if expected_hash != actual_hash:
      raise ArtifactValidationError(
          f"Frozen schedule hash mismatch for {filename}: "
          f"manifest={expected_hash!r} actual={actual_hash!r}"
      )
  if manifest.get("selective_rerun_permitted") is not False:
    raise ArtifactValidationError("Schedule must prohibit selective reruns")
  release_policy = schedule_builder.release_policy(str(cohort["release_id"]))
  for field, expected in {
      **release_policy,
      "primary_reporter_acceptance_permitted": release_policy[
          "analysis_eligible"
      ],
  }.items():
    if manifest.get(field) != expected:
      raise ArtifactValidationError(
          f"Schedule manifest {field}={manifest.get(field)!r}; expected "
          f"{expected!r}"
      )
  replacement_policy = manifest.get("replacement_policy") or {}
  expected_policy = {
      "max_replacement_rounds": 2,
      "selection_unit": "full_condition_triplet",
      "trigger": "invalid_infrastructure_only",
      "replacement_rounds_initially_scheduled": 0,
  }
  if replacement_policy != expected_policy:
    raise ArtifactValidationError(
        f"Unexpected replacement policy: {replacement_policy!r}"
    )
  runtime_policy = cohort.get("episode_runtime_policy")
  runtime_policy_sha256 = schedule_builder.episode_runtime_policy_sha256(
      runtime_policy
  )
  if manifest.get("episode_runtime_policy") != runtime_policy:
    raise ArtifactValidationError(
        "Schedule manifest episode runtime policy differs from the cohort"
    )
  if manifest.get("episode_runtime_policy_sha256") != runtime_policy_sha256:
    raise ArtifactValidationError(
        "Schedule manifest episode runtime policy hash is invalid"
    )
  if {
      str(row.get("episode_runtime_policy_sha256") or "")
      for row in schedule
  } != {runtime_policy_sha256}:
    raise ArtifactValidationError(
        "Schedule rows do not bind the exact episode runtime policy"
    )
  return FrozenBundle(
      cohort=cohort,
      cohort_sha256=cohort_sha256,
      schedule_manifest=manifest,
      schedule_manifest_sha256=_sha256_path(manifest_path),
      schedule=tuple(schedule),
      ledger_seed=tuple(ledger),
      ledger_schema_sha256=_sha256_path(schema_path),
  )


@dataclasses.dataclass(frozen=True)
class RealTask:
  category: str
  app_id: str
  semantic_task_id: str
  task_template: str
  package_name: str


def resolve_real_tasks(cohort: Mapping[str, Any]) -> dict[tuple[str, str, str], RealTask]:
  task_registry = registry.TaskRegistry().get_registry(
      family=str(cohort["suite_family"])
  )
  names, identities = catbench_primary_cohort.frozen_task_names(
      cohort, task_registry
  )
  resolved: dict[tuple[str, str, str], RealTask] = {}
  for task_template in names:
    category, app_id = identities[task_template]
    task_type = task_registry[task_template]
    semantic_task_id = str(getattr(task_type, "catbench_semantic_id", ""))
    package_name = str(getattr(task_type, "package_name", "") or "")
    key = (category, app_id, semantic_task_id)
    if key in resolved:
      raise ArtifactValidationError(f"Duplicate real task resolution: {key}")
    if not package_name:
      raise ArtifactValidationError(f"Real task lacks package_name: {task_template}")
    resolved[key] = RealTask(
        category=category,
        app_id=app_id,
        semantic_task_id=semantic_task_id,
        task_template=task_template,
        package_name=package_name,
    )
  expected = int(cohort["expected"]["task_app_count"])
  if len(resolved) != expected:
    raise ArtifactValidationError(
        f"Resolved {len(resolved)} real task/app classes; expected {expected}"
    )
  return resolved


def load_app_pins(path: Path, cohort: Mapping[str, Any]) -> dict[str, dict[str, str]]:
  rows: dict[str, dict[str, str]] = {}
  if path.is_symlink() or not path.is_file():
    raise ArtifactValidationError(f"App pins must be a regular non-symlink: {path}")
  try:
    with path.open("r", encoding="utf-8", newline="") as handle:
      reader = csv.DictReader(handle)
      expected_fields = [
          "category",
          "app_id",
          "package_name",
          "version_name",
          "version_code",
          "apk_sha256",
      ]
      if reader.fieldnames != expected_fields:
        raise ArtifactValidationError(
            f"App pin columns must be exactly {expected_fields}; got {reader.fieldnames}"
        )
      for row in reader:
        app_id = str(row.get("app_id") or "")
        if not app_id or app_id in rows:
          raise ArtifactValidationError(f"Invalid/duplicate app pin: {app_id!r}")
        normalized = {str(key): str(value or "") for key, value in row.items()}
        for required in (
            "category",
            "package_name",
            "version_name",
            "version_code",
            "apk_sha256",
        ):
          if not normalized.get(required):
            raise ArtifactValidationError(
                f"App pin {app_id!r} lacks {required}"
            )
        if not HEX_SHA256.fullmatch(normalized["apk_sha256"]):
          raise ArtifactValidationError(f"Invalid APK SHA-256 for {app_id}")
        rows[app_id] = normalized
  except OSError as exc:
    raise ArtifactValidationError(f"Unable to read app pins {path}: {exc}") from exc
  frozen_ids = {
      str(app_id)
      for spec in cohort["categories"].values()
      for app_id in spec["app_ids"]
  }
  missing = sorted(frozen_ids - set(rows))
  if missing:
    raise ArtifactValidationError(f"Frozen app pins missing: {missing}")
  return {app_id: rows[app_id] for app_id in sorted(frozen_ids)}


def _expand_env(value: str) -> str:
  pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")

  def replace(match: re.Match[str]) -> str:
    current = os.environ.get(match.group(1))
    return current if current else (match.group(3) or "")

  expanded = pattern.sub(replace, value)
  if "${" in expanded:
    raise ArtifactValidationError(f"Unresolved model argument: {value!r}")
  return expanded


def _resolve_runner(raw_path: str) -> Path:
  candidate = Path(_expand_env(raw_path)).expanduser()
  candidates = [candidate] if candidate.is_absolute() else [
      REPO_ROOT / candidate,
      BENCHMARK_ROOT / candidate,
  ]
  for path in candidates:
    if path.is_file():
      return path.resolve()
  raise ArtifactValidationError(f"Model runner not found: {raw_path}")


@dataclasses.dataclass(frozen=True)
class FrozenModel:
  name: str
  revision: str
  runner: Path
  runner_sha256: str
  args: tuple[str, ...]


def load_models(path: Path, cohort: Mapping[str, Any]) -> dict[str, FrozenModel]:
  payload = _read_json(path)
  raw_models = payload.get("models") if isinstance(payload, dict) else None
  if not isinstance(raw_models, list):
    raise ArtifactValidationError("Model config must contain a models list")
  by_name: dict[str, dict[str, Any]] = {}
  for model in raw_models:
    if not isinstance(model, dict):
      continue
    name = str(model.get("name") or "")
    if name in by_name:
      raise ArtifactValidationError(f"Duplicate model config: {name!r}")
    by_name[name] = model
  resolved: dict[str, FrozenModel] = {}
  for name in cohort["models"]:
    model = by_name.get(str(name))
    if model is None:
      raise ArtifactValidationError(f"Frozen model missing from config: {name}")
    revision = str(model.get("revision") or "")
    if not revision or not HEX_REVISION.fullmatch(revision):
      raise ArtifactValidationError(
          f"Frozen model {name!r} lacks a 40-64 hex immutable revision"
      )
    args = tuple(_expand_env(str(arg)) for arg in model.get("args", []))
    _validate_frozen_model_args(str(name), args)
    resolved[str(name)] = FrozenModel(
        name=str(name),
        revision=revision,
        runner=_resolve_runner(str(model.get("runner_script") or "")),
        runner_sha256="",  # filled from the resolved immutable path below
        args=args,
    )
    resolved_model = resolved[str(name)]
    resolved[str(name)] = dataclasses.replace(
        resolved_model,
        runner_sha256=_sha256_path(resolved_model.runner),
    )
  return resolved


def _model_launch_config_sha256(model: FrozenModel) -> str:
  return _sha256_bytes(_canonical_json({
      "name": model.name,
      "revision": model.revision,
      "runner_sha256": model.runner_sha256,
      "args": list(model.args),
  }).encode("utf-8"))


def validate_model_endpoint_attestations(
    path: Path,
    *,
    bundle: FrozenBundle,
    models: Mapping[str, FrozenModel],
    model_config_sha256: str,
) -> dict[str, Any]:
  """Require exact served-weights attestations for all five frozen models."""
  payload = _read_json(path)
  if not isinstance(payload, dict):
    raise ArtifactValidationError("Model endpoint attestation must be an object")
  expected_header = {
      "schema_version": 1,
      "release_id": bundle.cohort["release_id"],
      "cohort_sha256": bundle.cohort_sha256,
      "model_config_sha256": model_config_sha256,
      "attestation_policy": "exact_served_weights_for_frozen_five_model_roster",
  }
  for field, expected in expected_header.items():
    if payload.get(field) != expected:
      raise ArtifactValidationError(
          f"Model attestation {field}={payload.get(field)!r}; expected {expected!r}"
      )
  rows = payload.get("models")
  if not isinstance(rows, list):
    raise ArtifactValidationError("Model attestation models must be a list")
  by_name: dict[str, dict[str, Any]] = {}
  for row in rows:
    if not isinstance(row, dict):
      raise ArtifactValidationError("Malformed model attestation row")
    name = str(row.get("name") or "")
    if not name or name in by_name:
      raise ArtifactValidationError(f"Duplicate/empty model attestation: {name!r}")
    by_name[name] = row
  if list(by_name) != list(bundle.cohort["models"]):
    raise ArtifactValidationError(
        "Model endpoint attestation roster/order differs from frozen cohort"
    )
  for name, model in models.items():
    row = by_name[name]
    exact = {
        "revision": model.revision,
        "runner_sha256": model.runner_sha256,
        "launch_config_sha256": _model_launch_config_sha256(model),
    }
    for field, expected in exact.items():
      if row.get(field) != expected:
        raise ArtifactValidationError(
            f"Model {name} attestation {field} mismatch"
        )
    for field in (
        "weights_sha256",
        "endpoint_identity_sha256",
        "server_software_sha256",
        "attestation_evidence_sha256",
    ):
      if not HEX_SHA256.fullmatch(str(row.get(field) or "")):
        raise ArtifactValidationError(
            f"Model {name} attestation lacks valid {field}"
        )
    for field in ("served_model_id", "attestor_id", "attested_at"):
      if not str(row.get(field) or "").strip():
        raise ArtifactValidationError(
            f"Model {name} attestation lacks {field}"
        )
  return payload


def _source_revision_clean() -> str:
  revision = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=REPO_ROOT,
      check=True,
      capture_output=True,
      text=True,
  ).stdout.strip()
  dirty = subprocess.run(
      ["git", "status", "--porcelain"],
      cwd=REPO_ROOT,
      check=True,
      capture_output=True,
      text=True,
  ).stdout.strip()
  if dirty:
    raise ArtifactValidationError(
        "Frozen schedule execution requires a clean source worktree"
    )
  if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
    raise ArtifactValidationError(f"Unexpected source revision: {revision!r}")
  return revision


def validate_base_snapshot_manifest(
    path: Path, *, bundle: FrozenBundle, pins_sha256: str
) -> dict[str, Any]:
  payload = _read_json(path)
  if not isinstance(payload, dict):
    raise ArtifactValidationError("Base snapshot manifest must be an object")
  expected_scalars = {
      "schema_version": 1,
      "release_id": bundle.cohort["release_id"],
      "cohort_sha256": bundle.cohort_sha256,
      "app_pins_sha256": pins_sha256,
  }
  for field, expected in expected_scalars.items():
    if payload.get(field) != expected:
      raise ArtifactValidationError(
          f"Base snapshot {field}={payload.get(field)!r}; expected {expected!r}"
      )
  if not str(payload.get("snapshot_id") or ""):
    raise ArtifactValidationError("Base snapshot manifest lacks snapshot_id")
  for field in (
      "snapshot_sha256",
      "emulator_binary_sha256",
      "system_image_sha256",
      "avd_config_sha256",
  ):
    if not HEX_SHA256.fullmatch(str(payload.get(field) or "")):
      raise ArtifactValidationError(f"Base snapshot manifest lacks valid {field}")
  return payload


def _require_exact_object_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
  observed = frozenset(value)
  if observed != expected:
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    raise ArtifactValidationError(
        f"{label} key set mismatch; missing={missing}, extra={extra}"
    )


def _require_evidence_identity(value: Any, label: str) -> str:
  if (
      not isinstance(value, str)
      or not value.strip()
      or len(value) > 500
      or any(ord(character) < 32 for character in value)
  ):
    raise ArtifactValidationError(f"{label} must be a non-empty identity")
  return value


def _validate_verifier_evidence_root_location(
    evidence_root: Path, output_root: Path
) -> Path:
  """Require immutable evidence outside source and mutable consumer state."""
  if evidence_root.is_symlink() or not evidence_root.is_dir():
    raise ArtifactValidationError(
        "Verifier conformance evidence root must be a real non-symlink directory"
    )
  resolved = evidence_root.resolve()
  if _paths_overlap(resolved, REPO_ROOT):
    raise ArtifactValidationError(
        "Verifier conformance evidence root must be disjoint from the source "
        "repository"
    )
  if _paths_overlap(resolved, output_root.resolve()):
    raise ArtifactValidationError(
        "Verifier conformance evidence root must be disjoint from output_root"
    )
  return resolved


def _safe_evidence_relative_path(value: Any, label: str) -> PurePosixPath:
  if (
      not isinstance(value, str)
      or not value
      or len(value) > 1000
      or "\\" in value
      or any(ord(character) < 32 for character in value)
  ):
    raise ArtifactValidationError(
        f"{label} must be a safe non-empty relative path"
    )
  relative = PurePosixPath(value)
  if (
      relative.is_absolute()
      or not relative.parts
      or any(part in ("", ".", "..") for part in relative.parts)
      or relative.as_posix() != value
  ):
    raise ArtifactValidationError(f"{label} is not a canonical relative path")
  return relative


def _validate_evidence_file(
    *,
    evidence_root: Path,
    relative_value: Any,
    expected_sha256: Any,
    label: str,
    used_paths: set[str],
    inventory: list[dict[str, Any]],
) -> None:
  relative = _safe_evidence_relative_path(relative_value, label)
  relative_string = relative.as_posix()
  if relative_string in used_paths:
    raise ArtifactValidationError(
        f"Reused verifier conformance evidence path: {relative_string!r}"
    )
  used_paths.add(relative_string)
  if not HEX_SHA256.fullmatch(str(expected_sha256 or "")):
    raise ArtifactValidationError(f"{label} lacks a valid SHA-256")

  candidate = evidence_root.joinpath(*relative.parts)
  cursor = evidence_root
  for part in relative.parts:
    cursor = cursor / part
    if cursor.is_symlink():
      raise ArtifactValidationError(
          f"{label} traverses a symlink: {relative_string}"
      )
  try:
    resolved_candidate = candidate.resolve(strict=True)
  except (OSError, RuntimeError) as exc:
    raise ArtifactValidationError(
        f"{label} is missing or unreadable: {relative_string}"
    ) from exc
  if evidence_root not in resolved_candidate.parents:
    raise ArtifactValidationError(
        f"{label} escapes the evidence root: {relative_string}"
    )

  flags = (
      os.O_RDONLY
      | getattr(os, "O_NOFOLLOW", 0)
      | getattr(os, "O_NONBLOCK", 0)
  )
  try:
    descriptor = os.open(candidate, flags)
  except OSError as exc:
    raise ArtifactValidationError(
        f"{label} is not an openable non-symlink file: {relative_string}"
    ) from exc
  try:
    try:
      before = os.fstat(descriptor)
      if not stat.S_ISREG(before.st_mode):
        raise ArtifactValidationError(
            f"{label} is not a regular file: {relative_string}"
        )
      if before.st_size <= 0:
        raise ArtifactValidationError(
            f"{label} is empty: {relative_string}"
        )
      digest = hashlib.sha256()
      while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
          break
        digest.update(chunk)
      after = os.fstat(descriptor)
    except OSError as exc:
      raise ArtifactValidationError(
          f"{label} could not be read safely: {relative_string}"
      ) from exc
  finally:
    os.close(descriptor)
  before_identity = (
      before.st_dev,
      before.st_ino,
      before.st_size,
      before.st_mtime_ns,
      before.st_ctime_ns,
  )
  after_identity = (
      after.st_dev,
      after.st_ino,
      after.st_size,
      after.st_mtime_ns,
      after.st_ctime_ns,
  )
  if before_identity != after_identity:
    raise ArtifactValidationError(
        f"{label} changed while hashing: {relative_string}"
    )
  actual_sha256 = digest.hexdigest()
  if actual_sha256 != expected_sha256:
    raise ArtifactValidationError(
        f"{label} SHA-256 mismatch for {relative_string}"
    )
  inventory.append({
      "path": relative_string,
      "size_bytes": before.st_size,
      "sha256": actual_sha256,
  })


def _evidence_inventory_sha256(inventory: Sequence[Mapping[str, Any]]) -> str:
  canonical_inventory = {
      "schema_version": 1,
      "files": sorted(
          (dict(row) for row in inventory), key=lambda row: str(row["path"])
      ),
  }
  return _sha256_bytes(
      _canonical_json(canonical_inventory).encode("utf-8")
  )


def _expected_verifier_adapter_keys(
    bundle: FrozenBundle,
) -> tuple[tuple[str, str, str], ...]:
  keys: list[tuple[str, str, str]] = []
  categories = bundle.cohort.get("categories")
  if not isinstance(categories, dict) or not categories:
    raise ArtifactValidationError(
        "Frozen cohort lacks categories for verifier conformance"
    )
  for raw_category, raw_spec in categories.items():
    category = str(raw_category)
    if not isinstance(raw_spec, dict):
      raise ArtifactValidationError(
          f"Frozen cohort category {category!r} is malformed"
      )
    app_ids = raw_spec.get("app_ids")
    semantic_task_ids = raw_spec.get("semantic_task_ids")
    if (
        not isinstance(app_ids, list)
        or not app_ids
        or not isinstance(semantic_task_ids, list)
        or not semantic_task_ids
    ):
      raise ArtifactValidationError(
          f"Frozen cohort category {category!r} lacks an adapter roster"
      )
    keys.extend(
        (category, str(app_id), str(semantic_task_id))
        for app_id in app_ids
        for semantic_task_id in semantic_task_ids
    )
  if len(keys) != len(set(keys)):
    raise ArtifactValidationError(
        "Frozen cohort verifier adapter roster contains duplicates"
    )
  expected = bundle.cohort.get("expected")
  expected_count = (
      expected.get("task_app_count") if isinstance(expected, dict) else None
  )
  if expected_count != len(keys):
    raise ArtifactValidationError(
        "Frozen cohort verifier adapter count differs from task_app_count"
    )
  if (
      bundle.cohort.get("release_id") == schedule_builder.PRIMARY_RELEASE_ID
      and len(keys) != PRIMARY_VERIFIER_ADAPTER_COUNT
  ):
    raise ArtifactValidationError(
        "Primary verifier conformance roster must contain exactly "
        f"{PRIMARY_VERIFIER_ADAPTER_COUNT} task-app adapters"
    )
  return tuple(keys)


def validate_verifier_conformance_manifest(
    path: Path,
    *,
    bundle: FrozenBundle,
    pins: Mapping[str, Mapping[str, str]],
    base_snapshot: Mapping[str, Any],
    evidence_root: Path,
) -> tuple[dict[str, Any], str]:
  """Validate external positive/replay/near-miss evidence for every adapter.

  This gate validates evidence supplied by an external qualification process;
  it never generates, repairs, or approves that evidence.  The manifest is
  release- and base-bound, is excluded from paper outcomes, and must cover the
  complete frozen task-app adapter roster before either preflight or execution
  can continue.
  """
  if evidence_root.is_symlink() or not evidence_root.is_dir():
    raise ArtifactValidationError(
        "Verifier conformance evidence root must be a real non-symlink directory"
    )
  evidence_root = evidence_root.resolve()
  payload = _read_json(path)
  if not isinstance(payload, dict):
    raise ArtifactValidationError(
        "Verifier conformance manifest must be an object"
    )
  header_keys = frozenset({
      "schema_version",
      "release_id",
      "cohort_sha256",
      "base_snapshot_id",
      "base_snapshot_sha256",
      "docker_image",
      "qualification_policy",
      "artifact_role",
      "analysis_eligible",
      "approved",
      "attestor_id",
      "attested_at",
      "approver_id",
      "approved_at",
      "collection_tool_path",
      "collection_tool_sha256",
      "manifest_evidence_path",
      "manifest_evidence_sha256",
      "approval_evidence_path",
      "approval_evidence_sha256",
      "records",
  })
  _require_exact_object_keys(
      payload, header_keys, "Verifier conformance manifest"
  )
  expected_header = {
      "schema_version": 1,
      "release_id": bundle.cohort["release_id"],
      "cohort_sha256": bundle.cohort_sha256,
      "base_snapshot_id": base_snapshot["snapshot_id"],
      "base_snapshot_sha256": base_snapshot["snapshot_sha256"],
      "docker_image": FROZEN_DOCKER_IMAGE,
      "qualification_policy": VERIFIER_CONFORMANCE_POLICY,
      "artifact_role": VERIFIER_CONFORMANCE_ARTIFACT_ROLE,
      "analysis_eligible": False,
  }
  for field, expected_value in expected_header.items():
    if payload.get(field) != expected_value:
      raise ArtifactValidationError(
          f"Verifier conformance {field}={payload.get(field)!r}; "
          f"expected {expected_value!r}"
      )
  if payload.get("approved") is not True:
    raise ArtifactValidationError(
        "Verifier conformance manifest is unapproved"
    )
  for field in ("attestor_id", "attested_at", "approver_id", "approved_at"):
    _require_evidence_identity(
        payload.get(field), f"Verifier conformance {field}"
    )
  used_evidence_paths: set[str] = set()
  evidence_inventory: list[dict[str, Any]] = []
  for path_field, hash_field in (
      ("collection_tool_path", "collection_tool_sha256"),
      ("manifest_evidence_path", "manifest_evidence_sha256"),
      ("approval_evidence_path", "approval_evidence_sha256"),
  ):
    _validate_evidence_file(
        evidence_root=evidence_root,
        relative_value=payload.get(path_field),
        expected_sha256=payload.get(hash_field),
        label=f"Verifier conformance {path_field}",
        used_paths=used_evidence_paths,
        inventory=evidence_inventory,
    )

  expected_keys = _expected_verifier_adapter_keys(bundle)
  rows = payload.get("records")
  if not isinstance(rows, list):
    raise ArtifactValidationError(
        "Verifier conformance records must be a list"
    )
  if len(rows) != len(expected_keys):
    raise ArtifactValidationError(
        "Verifier conformance record count mismatch; "
        f"expected {len(expected_keys)}, observed {len(rows)}"
    )
  row_keys = frozenset({
      "category",
      "app_id",
      "semantic_task_id",
      "package_name",
      "version_name",
      "version_code",
      "apk_sha256",
      "approved",
      "skipped",
      "overwritten",
      "app_identity_evidence_path",
      "app_identity_evidence_sha256",
      "adapter_evidence_path",
      "adapter_evidence_sha256",
      "evidence",
  })
  case_keys = frozenset({
      "case",
      "expected_success",
      "observed_success",
      "passed",
      "skipped",
      "overwritten",
      "evidence_id",
      "action_trace_path",
      "action_trace_sha256",
      "state_evidence_path",
      "state_evidence_sha256",
  })
  observed_keys: list[tuple[str, str, str]] = []
  observed_key_set: set[tuple[str, str, str]] = set()
  evidence_ids: set[str] = set()
  positive_cases = frozenset({"primitive_action_positive", "reset_replay"})
  for index, row in enumerate(rows):
    if not isinstance(row, dict):
      raise ArtifactValidationError(
          f"Verifier conformance record {index} is malformed"
      )
    label = f"Verifier conformance record {index}"
    _require_exact_object_keys(row, row_keys, label)
    key = (
        str(row.get("category") or ""),
        str(row.get("app_id") or ""),
        str(row.get("semantic_task_id") or ""),
    )
    if not all(key):
      raise ArtifactValidationError(f"{label} has an empty adapter identity")
    if key in observed_key_set:
      raise ArtifactValidationError(
          f"Duplicate verifier conformance adapter record: {key}"
      )
    observed_key_set.add(key)
    observed_keys.append(key)
    app_id = key[1]
    pin = pins.get(app_id)
    if pin is None:
      raise ArtifactValidationError(
          f"Verifier conformance record references unknown app: {app_id}"
      )
    expected_identity = {
        "category": pin["category"],
        "package_name": pin["package_name"],
        "version_name": pin["version_name"],
        "version_code": pin["version_code"],
        "apk_sha256": pin["apk_sha256"],
    }
    for field, expected_value in expected_identity.items():
      if row.get(field) != expected_value:
        raise ArtifactValidationError(
            f"Verifier conformance {key} {field} mismatch"
        )
    if row.get("approved") is not True:
      raise ArtifactValidationError(
          f"Verifier conformance adapter is unapproved: {key}"
      )
    if row.get("skipped") is not False:
      raise ArtifactValidationError(
          f"Verifier conformance adapter was skipped: {key}"
      )
    if row.get("overwritten") is not False:
      raise ArtifactValidationError(
          f"Verifier conformance adapter evidence was overwritten: {key}"
      )
    for path_field, hash_field in (
        ("app_identity_evidence_path", "app_identity_evidence_sha256"),
        ("adapter_evidence_path", "adapter_evidence_sha256"),
    ):
      _validate_evidence_file(
          evidence_root=evidence_root,
          relative_value=row.get(path_field),
          expected_sha256=row.get(hash_field),
          label=f"Verifier conformance {key} {path_field}",
          used_paths=used_evidence_paths,
          inventory=evidence_inventory,
      )

    evidence = row.get("evidence")
    if not isinstance(evidence, dict):
      raise ArtifactValidationError(
          f"Verifier conformance {key} evidence must be an object"
      )
    _require_exact_object_keys(
        evidence,
        frozenset(VERIFIER_CONFORMANCE_CASES),
        f"Verifier conformance {key} evidence",
    )
    for case_name in VERIFIER_CONFORMANCE_CASES:
      case = evidence[case_name]
      if not isinstance(case, dict):
        raise ArtifactValidationError(
            f"Verifier conformance {key} case {case_name} is malformed"
        )
      case_label = f"Verifier conformance {key} case {case_name}"
      _require_exact_object_keys(case, case_keys, case_label)
      if case.get("case") != case_name:
        raise ArtifactValidationError(f"{case_label} identity mismatch")
      expected_success = case_name in positive_cases
      if case.get("expected_success") is not expected_success:
        raise ArtifactValidationError(
            f"{case_label} has wrong expected_success"
        )
      if case.get("observed_success") is not expected_success:
        raise ArtifactValidationError(
            f"{case_label} has wrong observed_success"
        )
      if case.get("passed") is not True:
        raise ArtifactValidationError(f"{case_label} did not pass")
      if case.get("skipped") is not False:
        raise ArtifactValidationError(f"{case_label} was skipped")
      if case.get("overwritten") is not False:
        raise ArtifactValidationError(f"{case_label} was overwritten")
      evidence_id = _require_evidence_identity(
          case.get("evidence_id"), f"{case_label} evidence_id"
      )
      if evidence_id in evidence_ids:
        raise ArtifactValidationError(
            f"Duplicate verifier conformance evidence_id: {evidence_id!r}"
        )
      evidence_ids.add(evidence_id)
      for path_field, hash_field in (
          ("action_trace_path", "action_trace_sha256"),
          ("state_evidence_path", "state_evidence_sha256"),
      ):
        _validate_evidence_file(
            evidence_root=evidence_root,
            relative_value=case.get(path_field),
            expected_sha256=case.get(hash_field),
            label=f"{case_label} {path_field}",
            used_paths=used_evidence_paths,
            inventory=evidence_inventory,
        )
  if tuple(observed_keys) != expected_keys:
    raise ArtifactValidationError(
        "Verifier conformance adapter roster/order differs from frozen cohort"
    )
  return payload, _evidence_inventory_sha256(evidence_inventory)


def validate_installed_app_attestation(
    path: Path,
    *,
    bundle: FrozenBundle,
    pins: Mapping[str, Mapping[str, str]],
    pins_sha256: str,
    base_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
  """Validate an approved signer/byte attestation for the frozen base image.

  The checked-in signer audit is intentionally observational and cannot pass
  this gate.  Production needs a separate release artifact that binds the
  exact installed APK bytes and fully verified signing certificates to the
  immutable base snapshot.  Defining the gate here does not invent or approve
  any signer pin: all 23 real-app rows must be supplied and approved outside
  the consumer before a launch is possible.
  """
  payload = _read_json(path)
  if not isinstance(payload, dict):
    raise ArtifactValidationError("Installed app attestation must be an object")
  approval_status = (
      "approved_for_primary_release"
      if bundle.cohort["release_id"] == schedule_builder.PRIMARY_RELEASE_ID
      else "approved_for_discard_only_g6_dry_run"
  )
  expected_header = {
      "schema_version": 1,
      "release_id": bundle.cohort["release_id"],
      "cohort_sha256": bundle.cohort_sha256,
      "app_pins_sha256": pins_sha256,
      "base_snapshot_id": base_snapshot["snapshot_id"],
      "base_snapshot_sha256": base_snapshot["snapshot_sha256"],
      "attestation_scope": "on_device_frozen_base_snapshot",
      "attestation_policy": (
          "exact_installed_apk_bytes_and_fully_verified_signers_for_frozen_roster"
      ),
      "approval_status": approval_status,
  }
  for field, expected in expected_header.items():
    if payload.get(field) != expected:
      raise ArtifactValidationError(
          f"Installed app attestation {field}={payload.get(field)!r}; "
          f"expected {expected!r}"
      )
  for field in ("attestor_id", "attested_at", "approver_id", "approved_at"):
    if not str(payload.get(field) or "").strip():
      raise ArtifactValidationError(
          f"Installed app attestation lacks {field}"
      )
  for field in (
      "attestation_evidence_sha256",
      "approval_evidence_sha256",
      "collection_tool_sha256",
  ):
    if not HEX_SHA256.fullmatch(str(payload.get(field) or "")):
      raise ArtifactValidationError(
          f"Installed app attestation lacks valid {field}"
      )

  expected_app_ids = [
      str(app_id)
      for category in bundle.cohort["categories"].values()
      for app_id in category["app_ids"]
  ]
  rows = payload.get("apps")
  if not isinstance(rows, list):
    raise ArtifactValidationError("Installed app attestation apps must be a list")
  observed_ids = [
      str(row.get("app_id") or "") if isinstance(row, dict) else ""
      for row in rows
  ]
  if observed_ids != expected_app_ids:
    raise ArtifactValidationError(
        "Installed app attestation roster/order differs from frozen cohort"
    )
  for row in rows:
    assert isinstance(row, dict)  # established by exact roster check above
    app_id = str(row["app_id"])
    pin = pins[app_id]
    expected_identity = {
        "category": pin["category"],
        "package_name": pin["package_name"],
        "version_name": pin["version_name"],
        "version_code": pin["version_code"],
        "pinned_artifact_sha256": pin["apk_sha256"],
        "signature_verification_status": "fully_cryptographically_verified",
    }
    for field, expected in expected_identity.items():
      if row.get(field) != expected:
        raise ArtifactValidationError(
            f"Installed app attestation {app_id} {field} mismatch"
        )
    for field in (
        "installed_apk_sha256",
        "signer_leaf_certificate_sha256",
    ):
      values = row.get(field)
      if (
          not isinstance(values, list)
          or not values
          or values != sorted(set(map(str, values)))
          or any(not HEX_SHA256.fullmatch(str(value)) for value in values)
      ):
        raise ArtifactValidationError(
            f"Installed app attestation {app_id} has invalid {field}"
        )
    if pin["apk_sha256"] not in row["installed_apk_sha256"]:
      raise ArtifactValidationError(
          f"Installed app attestation {app_id} does not contain the exact "
          "pinned APK bytes"
      )
    for field in (
        "installed_bytes_evidence_sha256",
        "signature_verification_evidence_sha256",
        "verification_tool_sha256",
    ):
      if not HEX_SHA256.fullmatch(str(row.get(field) or "")):
        raise ArtifactValidationError(
            f"Installed app attestation {app_id} lacks valid {field}"
        )
  return payload


def validate_c2_g_attempt_audit(
    audit_path: Path,
    *,
    breakdown_path: Path,
    bundle: FrozenBundle,
    expected_plan_count: int,
) -> dict[str, Any]:
  """Bind every accepted C2-G plan to its append-only external-call audit."""
  try:
    records = breakdown_generator._read_audit_records(  # pylint: disable=protected-access
        audit_path
    )
  except (OSError, ValueError) as exc:
    raise ArtifactValidationError(f"Invalid C2-G attempt audit: {exc}") from exc
  if not records or records[0].get("record_type") != "attempt_audit_header":
    raise ArtifactValidationError("C2-G attempt audit lacks its immutable header")
  binding = records[0].get("binding")
  if not isinstance(binding, dict):
    raise ArtifactValidationError("C2-G attempt audit header binding is malformed")
  try:
    accepted, attempts = breakdown_generator._validate_audit_records(  # pylint: disable=protected-access
        records, binding
    )
  except ValueError as exc:
    raise ArtifactValidationError(f"Invalid C2-G attempt audit chain: {exc}") from exc
  unresolved = set(attempts) - set(accepted)
  if unresolved or records[-1].get("attempt_phase") == "started":
    raise ArtifactValidationError(
        "C2-G attempt audit has unresolved external request(s): "
        f"{sorted(unresolved)}"
    )

  breakdown = _read_json(breakdown_path)
  if not isinstance(breakdown, dict):
    raise ArtifactValidationError("C2-G plan file must be an object")
  metadata = breakdown.get("metadata")
  entries = breakdown.get("breakdowns")
  if not isinstance(metadata, dict) or not isinstance(entries, list):
    raise ArtifactValidationError("C2-G plan metadata/breakdowns are malformed")
  provider = str(metadata.get("generator_provider") or "").strip()
  model = str(metadata.get("generator_model") or "").strip()
  prompt_sha256 = str(metadata.get("prompt_sha256") or "")
  generator_config = binding.get("generator_config")
  if not isinstance(generator_config, dict):
    raise ArtifactValidationError("C2-G audit lacks generator_config")
  computed_config_sha = _sha256_bytes(
      _canonical_json(generator_config).encode("utf-8")
  )
  expected_binding = {
      "provider": provider,
      "model": model,
      "prompt_sha256": prompt_sha256,
      "cohort_release_id": bundle.cohort["release_id"],
      "cohort_manifest_sha256": bundle.cohort_sha256,
      "generator_config_sha256": computed_config_sha,
  }
  for field, expected in expected_binding.items():
    if binding.get(field) != expected:
      raise ArtifactValidationError(
          f"C2-G attempt-audit binding {field} mismatch"
      )
  expected_entry_count = len(entries)
  expected_config = {
      "provider": provider,
      "model": model,
      "prompt_sha256": prompt_sha256,
      "generator_source_sha256": _sha256_path(
          Path(breakdown_generator.__file__).resolve()
      ),
      "cohort_release_id": bundle.cohort["release_id"],
      "cohort_manifest_sha256": bundle.cohort_sha256,
      "strict_forbidden_check": True,
      "plan_reuse_policy": "one_plan_per_semantic_instance_across_apps",
      "task_entry_count": expected_entry_count,
      "semantic_plan_count": expected_plan_count,
  }
  for field, expected in expected_config.items():
    if generator_config.get(field) != expected:
      raise ArtifactValidationError(
          f"C2-G generator_config {field}={generator_config.get(field)!r}; "
          f"expected {expected!r}"
      )
  if float(generator_config.get("temperature", float("nan"))) != 0.0:
    raise ArtifactValidationError("Frozen C2-G generation temperature must be zero")
  for field in (
      "provider_endpoint_sha256",
      "task_set_sha256",
      "generator_config_sha256",
  ):
    value = (
        binding.get(field)
        if field == "generator_config_sha256"
        else generator_config.get(field)
    )
    if not HEX_SHA256.fullmatch(str(value or "")):
      raise ArtifactValidationError(f"C2-G attempt audit lacks valid {field}")
  if binding.get("task_set_sha256") != generator_config.get("task_set_sha256"):
    raise ArtifactValidationError("C2-G task-set hash differs within audit binding")
  seed_config = generator_config.get("seed_config")
  if not isinstance(seed_config, dict) or binding.get("seed_config_sha256") != (
      _sha256_bytes(_canonical_json(seed_config).encode("utf-8"))
  ):
    raise ArtifactValidationError("C2-G seed configuration hash mismatch")

  expected_by_plan: dict[str, dict[str, Any]] = {}
  for index, entry in enumerate(entries):
    if not isinstance(entry, dict):
      raise ArtifactValidationError(f"Malformed C2-G plan entry {index}")
    plan_key = str(entry.get("plan_key") or "")
    identity = {
        "plan_key": plan_key,
        "semantic_task_id": entry.get("semantic_task_id"),
        "instance_id": entry.get("instance_id"),
        "semantic_goal": entry.get("semantic_goal"),
        "semantic_goal_sha256": entry.get("semantic_goal_sha256"),
        "semantic_parameter_sha256": entry.get("semantic_parameter_sha256"),
    }
    material = {
        "identity": identity,
        "breakdown": entry.get("breakdown"),
        "plan_sha256": entry.get("plan_sha256"),
    }
    prior = expected_by_plan.setdefault(plan_key, material)
    if prior != material:
      raise ArtifactValidationError(
          f"C2-G semantic plan differs across app entries: {plan_key}"
      )
  if len(expected_by_plan) != expected_plan_count:
    raise ArtifactValidationError(
        f"C2-G has {len(expected_by_plan)} audited semantic plans; "
        f"expected {expected_plan_count}"
    )
  if set(accepted) != set(expected_by_plan) or set(attempts) != set(expected_by_plan):
    raise ArtifactValidationError(
        "C2-G attempt audit does not cover exactly the frozen semantic plan keys"
    )
  for plan_key, expected in expected_by_plan.items():
    record = accepted[plan_key]
    if record.get("plan_identity") != expected["identity"]:
      raise ArtifactValidationError(
          f"C2-G audit identity mismatch for {plan_key}"
      )
    outcome = record.get("outcome") or {}
    if outcome.get("final_accepted_plan") != expected["breakdown"]:
      raise ArtifactValidationError(
          f"C2-G accepted audit plan differs from frozen output: {plan_key}"
      )
    if outcome.get("final_accepted_plan_sha256") != expected["plan_sha256"]:
      raise ArtifactValidationError(
          f"C2-G accepted audit plan hash differs from output: {plan_key}"
      )
  for record in records[1:]:
    request = record.get("request")
    if not isinstance(request, dict):
      raise ArtifactValidationError("C2-G audit attempt has malformed request")
    if request.get("provider") != provider or request.get("model") != model:
      raise ArtifactValidationError("C2-G audit request provider/model drift")
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or request.get("prompt_sha256") != (
        _sha256_bytes(prompt.encode("utf-8"))
    ):
      raise ArtifactValidationError("C2-G audit request prompt hash mismatch")

  audit_metadata = metadata.get("attempt_audit")
  if not isinstance(audit_metadata, dict):
    raise ArtifactValidationError("C2-G output lacks attempt-audit metadata")
  expected_audit_metadata = {
      "schema_version": breakdown_generator.ATTEMPT_AUDIT_SCHEMA_VERSION,
      "header_sha256": records[0]["record_sha256"],
      "tail_sha256": records[-1]["record_sha256"],
      "record_count": len(records),
      "generator_config_sha256": computed_config_sha,
      "security_policy": "credentials_and_authorization_headers_excluded",
  }
  for field, expected in expected_audit_metadata.items():
    if audit_metadata.get(field) != expected:
      raise ArtifactValidationError(
          f"C2-G output attempt-audit metadata {field} mismatch"
      )
  return {
      "record_count": len(records),
      "semantic_plan_count": len(accepted),
      "audit_sha256": _sha256_path(audit_path),
      "header_sha256": records[0]["record_sha256"],
      "tail_sha256": records[-1]["record_sha256"],
  }


def _validate_two_person_approval(
    approval_path: Path,
    *,
    breakdown_path: Path,
    condition: str,
    release_id: str,
    expected_plan_count: int,
    c2_g_attempt_audit_path: Path | None = None,
) -> None:
  payload = _read_json(approval_path)
  if not isinstance(payload, dict):
    raise ArtifactValidationError(f"Approval manifest must be an object: {approval_path}")
  if "review_worksheet" in payload or "template_claim" in payload:
    raise ArtifactValidationError(
        f"{condition} pending review worksheet is not a release approval manifest"
    )
  expected = {
      "schema_version": 1,
      "release_id": release_id,
      "condition": condition,
      "breakdown_sha256": _sha256_path(breakdown_path),
      "approval_policy": "independent_two_person_complete_plan_set",
      "approval_status": (
          "approved_for_primary_release"
          if release_id == schedule_builder.PRIMARY_RELEASE_ID
          else "approved_for_discard_only_g6_dry_run"
      ),
  }
  if condition == "c2_g":
    if c2_g_attempt_audit_path is None:
      raise ArtifactValidationError("C2-G approval lacks its attempt-audit path")
    expected["attempt_audit_sha256"] = _sha256_path(c2_g_attempt_audit_path)
    expected["generated_plan_edit_policy"] = "accepted_generator_output_unedited"
  elif condition == "c2_o":
    expected["authoring_policy"] = "two_human_authors_app_neutral"
  for field, value in expected.items():
    if payload.get(field) != value:
      raise ArtifactValidationError(
          f"Approval {condition} {field}={payload.get(field)!r}; expected {value!r}"
      )
  breakdown = _read_json(breakdown_path)
  expected_plan_keys = {
      str(entry.get("plan_key"))
      for entry in breakdown.get("breakdowns", [])
      if isinstance(entry, dict) and entry.get("plan_key")
  }
  if len(expected_plan_keys) != expected_plan_count:
    raise ArtifactValidationError(
        f"{condition} breakdown has {len(expected_plan_keys)} semantic plans; "
        f"expected {expected_plan_count}"
    )
  ordered_plan_keys = sorted(expected_plan_keys)
  approved_keys = payload.get("approved_plan_keys")
  if approved_keys != ordered_plan_keys:
    raise ArtifactValidationError(
        f"{condition} approval does not cover the exact "
        f"sorted {expected_plan_count} semantic plan keys"
    )
  if payload.get("approved_plan_count") != expected_plan_count:
    raise ArtifactValidationError(f"{condition} approved_plan_count mismatch")
  if not str(payload.get("approved_at") or "").strip():
    raise ArtifactValidationError(f"{condition} approval lacks approved_at")
  if not HEX_SHA256.fullmatch(
      str(payload.get("approval_evidence_sha256") or "")
  ):
    raise ArtifactValidationError(
        f"{condition} approval lacks approval_evidence_sha256"
    )
  reviewers = payload.get("reviewers")
  if not isinstance(reviewers, list) or len(reviewers) != 2:
    raise ArtifactValidationError(f"{condition} requires exactly two reviewers")
  reviewer_ids: set[str] = set()
  for reviewer in reviewers:
    if not isinstance(reviewer, dict):
      raise ArtifactValidationError(f"Malformed {condition} reviewer entry")
    reviewer_id = str(reviewer.get("reviewer_id") or "").strip()
    if not reviewer_id or reviewer.get("decision") != "approved":
      raise ArtifactValidationError(f"Unapproved/missing {condition} reviewer")
    if reviewer.get("reviewed_independently") is not True:
      raise ArtifactValidationError(
          f"{condition} reviewer did not attest independent review"
      )
    if reviewer.get("reviewed_breakdown_sha256") != expected["breakdown_sha256"]:
      raise ArtifactValidationError(
          f"{condition} reviewer approved a different breakdown hash"
      )
    if reviewer.get("reviewed_plan_keys") != ordered_plan_keys:
      raise ArtifactValidationError(
          f"{condition} reviewer did not cover every sorted plan key"
      )
    if not HEX_SHA256.fullmatch(str(reviewer.get("review_sha256") or "")):
      raise ArtifactValidationError(f"{condition} reviewer lacks review_sha256")
    plan_reviews = reviewer.get("plan_reviews")
    if not isinstance(plan_reviews, list) or len(plan_reviews) != expected_plan_count:
      raise ArtifactValidationError(
          f"{condition} reviewer lacks {expected_plan_count} per-plan reviews"
      )
    reviewed_keys: list[str] = []
    for review in plan_reviews:
      if not isinstance(review, dict):
        raise ArtifactValidationError(
            f"Malformed {condition} per-plan reviewer entry"
        )
      reviewed_keys.append(str(review.get("plan_key") or ""))
      if review.get("decision") != "approved":
        raise ArtifactValidationError(
            f"{condition} reviewer has an unapproved per-plan decision"
        )
      for dimension in (
          "correctness",
          "completeness",
          "semantic_parameter_preservation",
          "app_independence",
      ):
        if review.get(dimension) is not True:
          raise ArtifactValidationError(
              f"{condition} reviewer did not approve {dimension} per plan"
          )
      if not HEX_SHA256.fullmatch(
          str(review.get("review_evidence_sha256") or "")
      ):
        raise ArtifactValidationError(
            f"{condition} per-plan review lacks evidence SHA-256"
        )
    if reviewed_keys != ordered_plan_keys:
      raise ArtifactValidationError(
          f"{condition} per-plan reviews do not match sorted frozen plan keys"
      )
    reviewer_ids.add(reviewer_id)
  if len(reviewer_ids) != 2:
    raise ArtifactValidationError(f"{condition} reviewers must be distinct")


def _validate_plan_role(path: Path, condition: str) -> None:
  payload = _read_json(path)
  if not isinstance(payload, dict):
    raise ArtifactValidationError(f"{condition} plan file must be an object")
  metadata = payload.get("metadata")
  if not isinstance(metadata, dict):
    raise ArtifactValidationError(f"{condition} plan file lacks metadata")
  if metadata.get("condition") != "application_independent_breakdown_prepend":
    raise ArtifactValidationError(f"{condition} plan condition metadata mismatch")
  provider = str(metadata.get("generator_provider") or "").strip().lower()
  if condition == "c2_g":
    if not provider or provider == "human":
      raise ArtifactValidationError(
          "C2-G requires a named non-human frozen planner provider"
      )
    if not str(metadata.get("generator_model") or "").strip():
      raise ArtifactValidationError("C2-G lacks pinned generator_model")
    if not HEX_SHA256.fullmatch(str(metadata.get("prompt_sha256") or "")):
      raise ArtifactValidationError("C2-G lacks a valid prompt_sha256")
    if not isinstance(metadata.get("attempt_audit"), dict):
      raise ArtifactValidationError("C2-G lacks append-only attempt-audit metadata")
  elif condition == "c2_o":
    if provider != "human":
      raise ArtifactValidationError("C2-O generator_provider must be 'human'")
    if metadata.get("authoring_policy") != "two_human_authors_app_neutral":
      raise ArtifactValidationError(
          "C2-O authoring_policy must be two_human_authors_app_neutral"
      )
    if metadata.get("author_input_policy") != (
        "app_neutral_goal_only_no_app_or_ui_observation"
    ):
      raise ArtifactValidationError("C2-O author input policy is not app-neutral")
    authors = metadata.get("authors")
    if not isinstance(authors, list) or len(authors) != 2:
      raise ArtifactValidationError("C2-O requires exactly two named authors")
  else:
    raise ValueError(f"Unexpected plan condition: {condition}")


def run_plan_preflight(
    *,
    python_bin: str,
    cohort_path: Path,
    breakdown_path: Path,
    condition: str,
    report_path: Path,
) -> None:
  command = [
      python_bin,
      str(SCRIPT_DIR / "preflight_task_breakdowns.py"),
      "--breakdown_file",
      str(breakdown_path),
      "--condition",
      condition,
      "--cohort_manifest",
      str(cohort_path),
      "--n_task_combinations",
      "3",
      "--task_random_seed",
      "30",
      "--fail_on_warnings",
      "--report_json",
      str(report_path),
  ]
  env = os.environ.copy()
  env.pop("CATBENCH_INSTANCE_ID", None)
  result = subprocess.run(
      command,
      cwd=REPO_ROOT,
      env=env,
      check=False,
      capture_output=True,
      text=True,
  )
  report_path.with_suffix(report_path.suffix + ".stdout").write_text(
      (result.stdout or "") + (result.stderr or ""), encoding="utf-8"
  )
  if result.returncode != 0:
    raise ArtifactValidationError(
        f"Plan preflight failed for {breakdown_path}; see {report_path}.stdout"
    )


@dataclasses.dataclass(frozen=True)
class AttemptSpec:
  release_id: str
  cohort_sha256: str
  schedule_seed: int
  suite_family: str
  task_random_seed: int
  n_task_combinations: int
  episode_runtime_policy_sha256: str
  pair_id: str
  model: str
  category: str
  app_id: str
  semantic_task_id: str
  instance_id: int
  condition: str
  slot_id: str
  attempt_id: str
  attempt_index: int
  snapshot_family_id: str
  snapshot_clone_id: str
  is_replacement: bool
  block_order: int
  within_block_order: int

  @classmethod
  def from_row(cls, row: Mapping[str, Any]) -> "AttemptSpec":
    fields = {field.name for field in dataclasses.fields(cls)}
    try:
      return cls(**{name: row[name] for name in fields})
    except (KeyError, TypeError) as exc:
      raise ArtifactValidationError(f"Malformed attempt row: {row}") from exc

  def paired_key(self) -> dict[str, Any]:
    return {
        "model": self.model,
        "category": self.category,
        "app_id": self.app_id,
        "semantic_task_id": self.semantic_task_id,
        "instance_id": self.instance_id,
    }


def _replacement_attempt(
    initial_by_condition: Mapping[str, AttemptSpec],
    authorized: Mapping[str, Any],
    round_index: int,
) -> AttemptSpec:
  condition = str(authorized["condition"])
  base = initial_by_condition[condition]
  identity = authorized["identity"]
  return dataclasses.replace(
      base,
      slot_id=str(identity["slot_id"]),
      attempt_id=str(identity["attempt_id"]),
      attempt_index=round_index,
      snapshot_clone_id=str(identity["snapshot_clone_id"]),
      is_replacement=True,
  )


@dataclasses.dataclass(frozen=True)
class AttemptOutcome:
  status: str
  artifact_path: str
  result_contract_path: str
  reason_code: str
  is_successful: float | None = None
  artifact_sha256: str = ""
  snapshot_prepare_receipt: str = ""
  snapshot_release_receipt: str = ""

  def __post_init__(self) -> None:
    if self.status not in VALID_TERMINAL_STATUSES:
      raise ValueError(f"Invalid terminal status: {self.status}")


class EpisodeExecutor(Protocol):
  def execute(self, attempt: AttemptSpec) -> AttemptOutcome:
    """Execute exactly one already-authorized attempt."""


class Journal:
  """Append-only, hash-chained attempt event journal."""

  def __init__(self, path: Path):
    self.path = path
    self.events = _read_jsonl(path) if path.exists() else []
    self._validate()

  def _validate(self) -> None:
    previous = ""
    started: set[str] = set()
    finished: set[str] = set()
    for index, event in enumerate(self.events):
      if event.get("sequence") != index:
        raise ArtifactValidationError("Attempt journal sequence is not contiguous")
      if event.get("previous_event_sha256") != previous:
        raise ArtifactValidationError("Attempt journal hash chain is broken")
      claimed_hash = str(event.get("event_sha256") or "")
      body = dict(event)
      body.pop("event_sha256", None)
      actual_hash = _sha256_bytes(_canonical_json(body).encode("utf-8"))
      if claimed_hash != actual_hash:
        raise ArtifactValidationError("Attempt journal event hash mismatch")
      previous = claimed_hash
      attempt_id = str(event.get("attempt_id") or "")
      if event.get("event") == "started":
        if attempt_id in started:
          raise ArtifactValidationError(f"Attempt started twice: {attempt_id}")
        started.add(attempt_id)
      elif event.get("event") == "finished":
        if attempt_id not in started or attempt_id in finished:
          raise ArtifactValidationError(f"Invalid attempt finish event: {attempt_id}")
        if event.get("status") not in VALID_TERMINAL_STATUSES:
          raise ArtifactValidationError(f"Invalid journal status: {event.get('status')}")
        finished.add(attempt_id)
      else:
        raise ArtifactValidationError(f"Unknown journal event: {event.get('event')}")

  @property
  def finished(self) -> dict[str, dict[str, Any]]:
    return {
        str(event["attempt_id"]): event
        for event in self.events
        if event["event"] == "finished"
    }

  @property
  def unresolved_started(self) -> list[dict[str, Any]]:
    finished = set(self.finished)
    return [
        event
        for event in self.events
        if event["event"] == "started" and event["attempt_id"] not in finished
    ]

  def append(self, event: dict[str, Any]) -> dict[str, Any]:
    body = {
        **event,
        "sequence": len(self.events),
        "previous_event_sha256": (
            str(self.events[-1]["event_sha256"]) if self.events else ""
        ),
    }
    body["event_sha256"] = _sha256_bytes(
        _canonical_json(body).encode("utf-8")
    )
    self.path.parent.mkdir(parents=True, exist_ok=True)
    with self.path.open("ab") as handle:
      handle.write((_canonical_json(body) + "\n").encode("utf-8"))
      handle.flush()
      os.fsync(handle.fileno())
    self.events.append(body)
    return body


class ReplacementState:
  """Pure round-level retry/selection state over the frozen seed ledger."""

  def __init__(
      self,
      schedule: Sequence[Mapping[str, Any]],
      ledger_seed: Sequence[Mapping[str, Any]],
      finished: Mapping[str, Mapping[str, Any]],
  ):
    self.initial_by_pair: dict[str, list[AttemptSpec]] = {}
    for row in schedule:
      attempt = AttemptSpec.from_row(row)
      self.initial_by_pair.setdefault(attempt.pair_id, []).append(attempt)
    for attempts in self.initial_by_pair.values():
      attempts.sort(key=lambda item: item.within_block_order)
    self.ledger = {str(row["pair_id"]): copy.deepcopy(dict(row)) for row in ledger_seed}
    self.finished = dict(finished)
    if set(self.initial_by_pair) != set(self.ledger):
      raise ArtifactValidationError("Schedule/ledger pair identifiers differ")
    self.pair_order = sorted(
        self.initial_by_pair,
        key=lambda pair_id: self.initial_by_pair[pair_id][0].block_order,
    )
    self._rebuild_replacement_rounds()

  @staticmethod
  def _round_is_complete(
      attempts: Sequence[AttemptSpec], finished: Mapping[str, Mapping[str, Any]]
  ) -> bool:
    return all(item.attempt_id in finished for item in attempts)

  @staticmethod
  def _round_has_invalid(
      attempts: Sequence[AttemptSpec], finished: Mapping[str, Mapping[str, Any]]
  ) -> bool:
    if not ReplacementState._round_is_complete(attempts, finished):
      raise ArtifactValidationError("Cannot classify an incomplete condition triplet")
    return any(
        finished[item.attempt_id]["status"] == "invalid_infrastructure"
        for item in attempts
    )

  def _authorized_round_attempts(
      self, pair_id: str, round_index: int
  ) -> list[AttemptSpec]:
    ledger = self.ledger[pair_id]
    authorized = ledger["authorized_replacement_rounds"][round_index - 1]
    if int(authorized["round_index"]) != round_index:
      raise ArtifactValidationError("Replacement authorization round mismatch")
    initial = self.initial_by_pair[pair_id]
    by_condition = {item.condition: item for item in initial}
    order = [item.condition for item in initial]
    return [
        _replacement_attempt(
            by_condition,
            {
                "condition": condition,
                "identity": authorized["condition_attempts"][condition],
            },
            round_index,
        )
        for condition in order
    ]

  def _rebuild_replacement_rounds(self) -> None:
    """Derive every authorized replacement solely from terminal statuses."""
    for pair_id in self.pair_order:
      ledger = self.ledger[pair_id]
      ledger["replacement_rounds"] = []
      prior_attempts = self.initial_by_pair[pair_id]
      for round_index in range(1, schedule_builder.MAX_REPLACEMENT_ROUNDS + 1):
        if not self._round_is_complete(prior_attempts, self.finished):
          break
        if not self._round_has_invalid(prior_attempts, self.finished):
          break
        replacement_attempts = self._authorized_round_attempts(pair_id, round_index)
        invalid_ids = [
            attempt.attempt_id
            for attempt in prior_attempts
            if self.finished[attempt.attempt_id]["status"]
            == "invalid_infrastructure"
        ]
        condition_attempts: dict[str, dict[str, Any]] = {}
        for attempt in replacement_attempts:
          finished = self.finished.get(attempt.attempt_id)
          condition_attempts[attempt.condition] = {
              "slot_id": attempt.slot_id,
              "attempt_id": attempt.attempt_id,
              "snapshot_clone_id": attempt.snapshot_clone_id,
              "status": finished["status"] if finished else "missing",
              "artifact_path": str(finished.get("artifact_path") or "") if finished else "",
          }
        ledger["replacement_rounds"].append({
            "round_index": round_index,
            "trigger": "invalid_infrastructure",
            "trigger_attempt_ids": invalid_ids,
            "decision_basis": (
                "automated_terminal_status_only: one or more complete prior-"
                "round attempts had catbench_episode_status="
                "invalid_infrastructure; rewards, screenshots, and judge labels "
                "were not inspected"
            ),
            "decided_at": max(
                str(self.finished[attempt.attempt_id]["recorded_at"])
                for attempt in prior_attempts
            ),
            "condition_attempts": condition_attempts,
        })
        prior_attempts = replacement_attempts

  def next_attempt(self) -> AttemptSpec | None:
    # Execute complete epochs. Every one of the 10,350 r0 slots is consumed
    # before any r1 replacement, and every eligible r1 triplet is complete
    # before any r2 replacement. Within each epoch, preserve original block
    # order and its stored condition permutation.
    for pair_id in self.pair_order:
      initial = self.initial_by_pair[pair_id]
      for attempt in initial:
        if attempt.attempt_id not in self.finished:
          return attempt
    for round_index in range(1, schedule_builder.MAX_REPLACEMENT_ROUNDS + 1):
      for pair_id in self.pair_order:
        initial = self.initial_by_pair[pair_id]
        if round_index > 1 and not self._round_has_invalid(
            initial, self.finished
        ):
          # This pair was never eligible for r1, so it cannot be eligible for
          # a later replacement epoch either.
          continue
        prior = (
            initial
            if round_index == 1
            else self._authorized_round_attempts(pair_id, round_index - 1)
        )
        if not self._round_is_complete(prior, self.finished):
          raise ArtifactValidationError(
              "Prior replacement epoch is incomplete before next epoch"
          )
        if not self._round_has_invalid(prior, self.finished):
          continue
        replacement = self._authorized_round_attempts(pair_id, round_index)
        for attempt in replacement:
          if attempt.attempt_id not in self.finished:
            return attempt
    return None

  def selections(self) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_id in self.pair_order:
      initial = self.initial_by_pair[pair_id]
      rounds: list[list[AttemptSpec]] = [initial]
      rounds.extend(
          self._authorized_round_attempts(pair_id, index)
          for index in range(1, schedule_builder.MAX_REPLACEMENT_ROUNDS + 1)
      )
      selected_index: int | None = None
      pending = False
      attempted_ids: list[str] = []
      for round_index, attempts in enumerate(rounds):
        attempted_ids.extend(
            attempt.attempt_id
            for attempt in attempts
            if attempt.attempt_id in self.finished
        )
        if not self._round_is_complete(attempts, self.finished):
          pending = True
          break
        if not self._round_has_invalid(attempts, self.finished):
          selected_index = round_index
          break
      if selected_index is not None:
        selected = rounds[selected_index]
        selection_status = "selected_complete_triplet"
        selected_attempts = {
            attempt.condition: attempt.attempt_id for attempt in selected
        }
      elif pending:
        selection_status = "pending"
        selected_attempts = {}
      else:
        selection_status = "exhausted_invalid"
        selected_attempts = {}
      rows.append({
          "release_id": initial[0].release_id,
          **schedule_builder.release_policy(initial[0].release_id),
          "pair_id": pair_id,
          "paired_key": initial[0].paired_key(),
          "selection_unit": "full_condition_triplet",
          "selection_status": selection_status,
          "selected_round": selected_index,
          "selected_attempt_ids": selected_attempts,
          "all_finished_attempt_ids": attempted_ids,
          "selection_basis": (
              "first complete round with no invalid_infrastructure member; "
              "no cross-round condition mixing"
          ),
      })
    return rows


def _runtime_state_bytes(state: ReplacementState) -> tuple[bytes, bytes]:
  ledgers = [state.ledger[pair_id] for pair_id in state.pair_order]
  return _jsonl_bytes(ledgers), _jsonl_bytes(state.selections())


def write_runtime_state(
    state_dir: Path, state: ReplacementState, journal: Journal
) -> None:
  runtime_bytes, selection_bytes = _runtime_state_bytes(state)
  _atomic_write(state_dir / RUNTIME_LEDGER_FILE, runtime_bytes)
  _atomic_write(state_dir / SELECTION_FILE, selection_bytes)
  journal_head = (
      str(journal.events[-1]["event_sha256"]) if journal.events else ""
  )
  commit = {
      "schema_version": 1,
      "journal_event_count": len(journal.events),
      "journal_head_sha256": journal_head,
      "runtime_ledger_sha256": _sha256_bytes(runtime_bytes),
      "selection_sha256": _sha256_bytes(selection_bytes),
  }
  # Written last: this is the commit marker for the two derived files.
  _atomic_write(
      state_dir / STATE_COMMIT_FILE,
      (json.dumps(commit, indent=2, sort_keys=True) + "\n").encode("utf-8"),
  )


def _attempt_provenance(attempt: AttemptSpec) -> dict[str, Any]:
  return {
      "release_id": attempt.release_id,
      **schedule_builder.release_policy(attempt.release_id),
      "cohort_sha256": attempt.cohort_sha256,
      "episode_runtime_policy_sha256": (
          attempt.episode_runtime_policy_sha256
      ),
      "pair_id": attempt.pair_id,
      "slot_id": attempt.slot_id,
      "attempt_id": attempt.attempt_id,
      "attempt_index": attempt.attempt_index,
      "snapshot_family_id": attempt.snapshot_family_id,
      "snapshot_clone_id": attempt.snapshot_clone_id,
      "model": attempt.model,
      "category": attempt.category,
      "app_id": attempt.app_id,
      "semantic_task_id": attempt.semantic_task_id,
      "instance_id": attempt.instance_id,
      "condition": attempt.condition,
      "is_replacement": attempt.is_replacement,
  }


class ScheduleConsumer:
  """Stateful sequential consumer with injectable real-execution boundary."""

  def __init__(
      self,
      bundle: FrozenBundle,
      state_dir: Path,
      executor: EpisodeExecutor,
  ):
    self.bundle = bundle
    self.state_dir = state_dir
    self.executor = executor
    self.journal = Journal(state_dir / JOURNAL_FILE)
    self._validate_or_replay_derived_state()

  def _validate_or_replay_derived_state(self) -> None:
    """Validate a committed replay, or repair one interrupted atomic epoch."""
    commit_path = self.state_dir / STATE_COMMIT_FILE
    runtime_path = self.state_dir / RUNTIME_LEDGER_FILE
    selection_path = self.state_dir / SELECTION_FILE
    if not commit_path.exists():
      if self.journal.events:
        raise ArtifactValidationError(
            "Attempt journal exists without a derived-state commit marker"
        )
      write_runtime_state(self.state_dir, self._state(), self.journal)
      return
    commit = _read_json(commit_path)
    if not isinstance(commit, dict) or commit.get("schema_version") != 1:
      raise ArtifactValidationError("Malformed state_commit.json")
    try:
      committed_count = int(commit["journal_event_count"])
    except (KeyError, TypeError, ValueError) as exc:
      raise ArtifactValidationError("Invalid committed journal count") from exc
    if not 0 <= committed_count <= len(self.journal.events):
      raise ArtifactValidationError("Committed journal count is outside replay")
    committed_head = str(commit.get("journal_head_sha256") or "")
    replay_head = (
        str(self.journal.events[committed_count - 1]["event_sha256"])
        if committed_count
        else ""
    )
    if committed_head != replay_head:
      raise ArtifactValidationError("State commit does not name a journal prefix")
    current_count = len(self.journal.events)
    if committed_count == current_count:
      expected_runtime, expected_selection = _runtime_state_bytes(self._state())
      for path, hash_field in (
          (runtime_path, "runtime_ledger_sha256"),
          (selection_path, "selection_sha256"),
      ):
        if path.is_symlink() or not path.is_file():
          raise ArtifactValidationError(f"Committed derived state missing: {path}")
        if _sha256_path(path) != commit.get(hash_field):
          raise ArtifactValidationError(
              f"Committed derived state hash mismatch: {path.name}"
          )
      if runtime_path.read_bytes() != expected_runtime:
        raise ArtifactValidationError(
            "Runtime replacement ledger does not equal journal replay"
        )
      if selection_path.read_bytes() != expected_selection:
        raise ArtifactValidationError(
            "Selected triplets do not equal journal replay"
        )
      return
    # The commit is intentionally written after the derived files. A normal
    # crash can leave only the just-started event, or a started+finished pair,
    # beyond the last commit. Anything larger is not a consumer write pattern.
    if current_count - committed_count > 2:
      raise ArtifactValidationError(
          "Derived state is more than one attempt behind the journal"
      )
    write_runtime_state(self.state_dir, self._state(), self.journal)

  def _state(self) -> ReplacementState:
    return ReplacementState(
        self.bundle.schedule, self.bundle.ledger_seed, self.journal.finished
    )

  def recover_interrupted_attempt(self) -> bool:
    unresolved = self.journal.unresolved_started
    if not unresolved:
      return False
    if len(unresolved) != 1:
      raise ArtifactValidationError(
          f"Expected at most one interrupted attempt; found {len(unresolved)}"
      )
    started = unresolved[0]
    self.journal.append({
        "event": "finished",
        "recorded_at": _utc_now(),
        "attempt_id": started["attempt_id"],
        "pair_id": started["pair_id"],
        "slot_id": started["slot_id"],
        "snapshot_clone_id": started["snapshot_clone_id"],
        "attempt_index": started["attempt_index"],
        "condition": started["condition"],
        "status": "invalid_infrastructure",
        "reason_code": "consumer_interrupted_before_terminal_contract",
        "artifact_path": "",
        "result_contract_path": "",
        "artifact_sha256": "",
        "snapshot_prepare_receipt": "",
        "snapshot_release_receipt": "",
    })
    write_runtime_state(self.state_dir, self._state(), self.journal)
    return True

  def run_until_complete(self, *, halt_after_invalid: bool = True) -> int:
    # An interrupted subprocess is a consumed infrastructure-invalid attempt.
    # Stop immediately after recording it so an operator can repair the device
    # before the remaining condition members are attempted.
    if self.recover_interrupted_attempt():
      return 3
    while True:
      state = self._state()
      write_runtime_state(self.state_dir, state, self.journal)
      attempt = state.next_attempt()
      if attempt is None:
        return 0
      provenance = _attempt_provenance(attempt)
      self.journal.append({
          "event": "started",
          "recorded_at": _utc_now(),
          **provenance,
      })
      try:
        outcome = self.executor.execute(attempt)
      except Exception as exc:  # last-resort boundary: never infer a valid result
        outcome = AttemptOutcome(
            status="invalid_infrastructure",
            artifact_path="",
            result_contract_path="",
            reason_code=f"executor_exception:{type(exc).__module__}.{type(exc).__qualname__}",
        )
      self.journal.append({
          "event": "finished",
          "recorded_at": _utc_now(),
          **provenance,
          **dataclasses.asdict(outcome),
      })
      write_runtime_state(self.state_dir, self._state(), self.journal)
      if halt_after_invalid and outcome.status == "invalid_infrastructure":
        return 3


def _load_pickle_episode(path: Path) -> dict[str, Any]:
  # The checkpoint is produced locally by the just-launched trusted runner.
  # Pickle is never accepted from a network or user-provided artifact here.
  if path.is_symlink() or not path.is_file():
    raise InfrastructureInvalid("checkpoint_not_regular_file")
  with path.open("rb") as handle:
    compressed = handle.read()
  with gzip.open(io.BytesIO(compressed), "rb") as handle:
    payload = pickle.load(handle)
  if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
    raise InfrastructureInvalid("checkpoint_not_exactly_one_episode")
  return payload[0]


def _load_exact_checkpoint(
    checkpoint_dir: Path, expected_checkpoint: Path
) -> dict[str, Any]:
  checkpoint_files = sorted(checkpoint_dir.glob("*.pkl.gz"))
  if checkpoint_files != [expected_checkpoint]:
    raise InfrastructureInvalid("checkpoint_file_set_mismatch")
  return _load_pickle_episode(expected_checkpoint)


def _parse_package_version(output: str) -> tuple[str, str]:
  code = re.search(r"\bversionCode=(\d+)\b", output)
  name = re.search(r"^\s*versionName=(.+?)\s*$", output, re.MULTILINE)
  return (name.group(1) if name else "", code.group(1) if code else "")


@dataclasses.dataclass(frozen=True)
class ExecutionContext:
  bundle: FrozenBundle
  real_tasks: Mapping[tuple[str, str, str], RealTask]
  models: Mapping[str, FrozenModel]
  app_pins: Mapping[str, Mapping[str, str]]
  model_config_sha256: str
  model_endpoint_attestation_sha256: str
  app_pins_sha256: str
  installed_app_attestation_sha256: str
  episode_runtime_policy: Mapping[str, Any]
  episode_runtime_environment: Mapping[str, str]
  episode_runtime_policy_sha256: str
  source_revision: str
  c2_g_breakdown: Path
  c2_o_breakdown: Path
  c2_g_sha256: str
  c2_o_sha256: str
  base_snapshot: Mapping[str, Any]
  base_snapshot_manifest_sha256: str
  snapshot_hook: Path
  snapshot_hook_sha256: str
  python_bin: str
  adb_path: str
  device_serial: str
  console_port: int
  grpc_port: int
  output_root: Path


class SubprocessEpisodeExecutor:
  """Production boundary for snapshot hooks, one runner, and one contract."""

  def __init__(self, context: ExecutionContext):
    self.context = context
    cohort_policy = context.bundle.cohort.get("episode_runtime_policy")
    if context.episode_runtime_policy != cohort_policy:
      raise ArtifactValidationError(
          "Execution context runtime policy differs from the frozen cohort"
      )
    if not isinstance(cohort_policy, Mapping):
      raise ArtifactValidationError("Frozen episode runtime policy is malformed")
    policy_environment = cohort_policy.get("environment")
    if (
        not isinstance(policy_environment, Mapping)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in policy_environment.items()
        )
    ):
      raise ArtifactValidationError(
          "Frozen episode runtime policy environment must contain strings"
      )
    if dict(context.episode_runtime_environment) != dict(policy_environment):
      raise ArtifactValidationError(
          "Execution context runtime environment differs from frozen policy"
      )
    expected_policy_sha256 = schedule_builder.episode_runtime_policy_sha256(
        cohort_policy
    )
    if context.episode_runtime_policy_sha256 != expected_policy_sha256:
      raise ArtifactValidationError(
          "Execution context runtime policy hash is invalid"
      )
    if (
        context.bundle.schedule_manifest.get("episode_runtime_policy")
        != cohort_policy
        or context.bundle.schedule_manifest.get(
            "episode_runtime_policy_sha256"
        ) != expected_policy_sha256
    ):
      raise ArtifactValidationError(
          "Execution context runtime policy is not bound by schedule manifest"
      )

  def _attempt_dir(self, attempt: AttemptSpec) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", attempt.attempt_id)
    return self.context.output_root / ATTEMPTS_DIR / safe_id

  def _hook(
      self, operation: str, attempt: AttemptSpec, attempt_dir: Path
  ) -> Path:
    if _sha256_path(self.context.snapshot_hook) != self.context.snapshot_hook_sha256:
      raise InfrastructureInvalid("snapshot_hook_changed_after_preflight")
    request = {
        "schema_version": 1,
        "operation": operation,
        **_attempt_provenance(attempt),
        "base_snapshot_id": self.context.base_snapshot["snapshot_id"],
        "base_snapshot_sha256": self.context.base_snapshot["snapshot_sha256"],
        "device_serial": self.context.device_serial,
    }
    request_path = attempt_dir / f"snapshot_{operation}_request.json"
    receipt_path = attempt_dir / f"snapshot_{operation}_receipt.json"
    _atomic_write(
        request_path,
        (json.dumps(request, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    try:
      completed = subprocess.run(
          [
              str(self.context.snapshot_hook),
              "--request",
              str(request_path),
              "--receipt",
              str(receipt_path),
          ],
          cwd=REPO_ROOT,
          check=False,
          capture_output=True,
          text=True,
          timeout=SNAPSHOT_HOOK_TIMEOUT_SECONDS,
      )
    except subprocess.TimeoutExpired as exc:
      raise InfrastructureInvalid(f"snapshot_{operation}_hook_timeout") from exc
    (attempt_dir / f"snapshot_{operation}.stdout").write_text(
        completed.stdout or "", encoding="utf-8"
    )
    (attempt_dir / f"snapshot_{operation}.stderr").write_text(
        completed.stderr or "", encoding="utf-8"
    )
    if completed.returncode != 0 or not receipt_path.is_file():
      raise InfrastructureInvalid(f"snapshot_{operation}_hook_failed")
    receipt = _read_json(receipt_path)
    required_echo = {
        "schema_version": 1,
        "operation": operation,
        "success": True,
        "request_sha256": _sha256_path(request_path),
        "release_id": attempt.release_id,
        "pair_id": attempt.pair_id,
        "attempt_id": attempt.attempt_id,
        "snapshot_family_id": attempt.snapshot_family_id,
        "snapshot_clone_id": attempt.snapshot_clone_id,
        "base_snapshot_id": self.context.base_snapshot["snapshot_id"],
        "base_snapshot_sha256": self.context.base_snapshot["snapshot_sha256"],
        "parent_snapshot_id": self.context.base_snapshot["snapshot_id"],
        "parent_snapshot_sha256": self.context.base_snapshot["snapshot_sha256"],
        "clone_generation": 1,
        "device_serial": self.context.device_serial,
    }
    if not isinstance(receipt, dict):
      raise InfrastructureInvalid(f"snapshot_{operation}_receipt_not_object")
    for field, expected in required_echo.items():
      if receipt.get(field) != expected:
        raise InfrastructureInvalid(
            f"snapshot_{operation}_receipt_mismatch:{field}"
        )
    if not re.fullmatch(r"[0-9A-Za-z._-]{1,200}", str(receipt.get("hook_revision") or "")):
      raise InfrastructureInvalid(f"snapshot_{operation}_missing_hook_revision")
    attestation_field = (
        "active_snapshot_sha256" if operation == "clone_activate" else "released_snapshot_sha256"
    )
    if not HEX_SHA256.fullmatch(str(receipt.get(attestation_field) or "")):
      raise InfrastructureInvalid(
          f"snapshot_{operation}_missing_{attestation_field}"
      )
    clone_identity_field = (
        "active_snapshot_clone_id"
        if operation == "clone_activate"
        else "released_snapshot_clone_id"
    )
    if receipt.get(clone_identity_field) != attempt.snapshot_clone_id:
      raise InfrastructureInvalid(
          f"snapshot_{operation}_mismatch:{clone_identity_field}"
      )
    return receipt_path

  def _device_pin_preflight(self, task: RealTask, pin: Mapping[str, str]) -> None:
    completed = subprocess.run(
        [
            self.context.adb_path,
            "-s",
            self.context.device_serial,
            "shell",
            "dumpsys",
            "package",
            task.package_name,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    actual_name, actual_code = _parse_package_version(completed.stdout or "")
    if (
        completed.returncode != 0
        or actual_name != pin["version_name"]
        or actual_code != pin["version_code"]
    ):
      raise InfrastructureInvalid("device_package_pin_mismatch")

  def _runner_config_sha256(self, model: FrozenModel) -> str:
    try:
      current_revision = _source_revision_clean()
    except (ArtifactValidationError, OSError, subprocess.SubprocessError) as exc:
      raise InfrastructureInvalid("source_worktree_changed_after_preflight") from exc
    if current_revision != self.context.source_revision:
      raise InfrastructureInvalid("source_revision_changed_after_preflight")
    if _sha256_path(model.runner) != model.runner_sha256:
      raise InfrastructureInvalid("model_runner_changed_after_preflight")
    payload = {
        "runner_path": str(model.runner),
        "runner_sha256": model.runner_sha256,
        "model": model.name,
        "model_revision": model.revision,
        "model_args": list(model.args),
        "suite_family": self.context.bundle.cohort["suite_family"],
        "n_task_combinations": 3,
        "task_random_seed": 30,
        "fixed_task_seed": False,
        "early_stop_on_success": (
            self.context.episode_runtime_environment[
                "CATBENCH_EARLY_STOP_ON_SUCCESS"
            ] == "1"
        ),
        "episode_runtime_policy": self.context.episode_runtime_policy,
        "episode_runtime_policy_sha256": (
            self.context.episode_runtime_policy_sha256
        ),
        "episode_runner_timeout_seconds": EPISODE_RUNNER_TIMEOUT_SECONDS,
        "model_endpoint_attestation_sha256": (
            self.context.model_endpoint_attestation_sha256
        ),
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))

  def _episode_env(
      self,
      attempt: AttemptSpec,
      task: RealTask,
      model: FrozenModel,
      pin: Mapping[str, str],
      runner_config_sha256: str,
  ) -> dict[str, str]:
    if (
        attempt.episode_runtime_policy_sha256
        != self.context.episode_runtime_policy_sha256
    ):
      raise InfrastructureInvalid("attempt_runtime_policy_hash_mismatch")
    env = os.environ.copy()
    release_policy = schedule_builder.release_policy(attempt.release_id)
    for key in (
        "CATBENCH_TASK_BREAKDOWN_FILE",
        "CATBENCH_TASK_BREAKDOWN_MODE",
        "CATBENCH_TASK_BREAKDOWN_REQUIRED",
        "CATBENCH_PLAN_FILE_SHA256",
    ):
      env.pop(key, None)
    # Never inherit behavior-changing CATBench controls from the shell or
    # env_file.  The exact values below are part of the hashed cohort release.
    for key in self.context.episode_runtime_environment:
      env.pop(key, None)
    # The evaluated runner must import the attested checkout and its selected
    # interpreter environment, not an operator-controlled shadow module tree.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env.update(self.context.episode_runtime_environment)
    env.update({
        "CATBENCH_INSTANCE_ID": str(attempt.instance_id),
        "CATBENCH_CONDITION": attempt.condition,
        "CATBENCH_RELEASE_ID": attempt.release_id,
        "CATBENCH_RELEASE_PURPOSE": str(release_policy["release_purpose"]),
        "CATBENCH_ARTIFACT_ROLE": str(release_policy["artifact_role"]),
        "CATBENCH_ANALYSIS_ELIGIBLE": (
            "1" if release_policy["analysis_eligible"] else "0"
        ),
        "CATBENCH_CODE_REVISION": self.context.source_revision,
        "CATBENCH_COHORT_SHA256": attempt.cohort_sha256,
        "CATBENCH_EPISODE_RUNTIME_POLICY_SHA256": (
            self.context.episode_runtime_policy_sha256
        ),
        "CATBENCH_SCHEDULE_MANIFEST_SHA256": (
            self.context.bundle.schedule_manifest_sha256
        ),
        "CATBENCH_PAIR_ID": attempt.pair_id,
        "CATBENCH_SLOT_ID": attempt.slot_id,
        "CATBENCH_ATTEMPT_ID": attempt.attempt_id,
        "CATBENCH_ATTEMPT_INDEX": str(attempt.attempt_index),
        "CATBENCH_SNAPSHOT_FAMILY_ID": attempt.snapshot_family_id,
        "CATBENCH_SNAPSHOT_CLONE_ID": attempt.snapshot_clone_id,
        "CATBENCH_APP_ID": attempt.app_id,
        "CATBENCH_MODEL_NAME": model.name,
        "CATBENCH_MODEL_REVISION": model.revision,
        "CATBENCH_RUNNER_CONFIG_SHA256": runner_config_sha256,
        "CATBENCH_MODEL_CONFIG_SHA256": self.context.model_config_sha256,
        "CATBENCH_MODEL_ENDPOINT_ATTESTATION_SHA256": (
            self.context.model_endpoint_attestation_sha256
        ),
        "CATBENCH_APP_PINS_SHA256": self.context.app_pins_sha256,
        "CATBENCH_INSTALLED_APP_ATTESTATION_SHA256": (
            self.context.installed_app_attestation_sha256
        ),
        "CATBENCH_TASK_RANDOM_SEED": str(attempt.task_random_seed),
        "CATBENCH_N_TASK_COMBINATIONS": str(attempt.n_task_combinations),
        "CATBENCH_SCHEDULE_SEED": str(attempt.schedule_seed),
        "CATBENCH_APP_VERSION": pin["version_name"],
        "CATBENCH_APP_VERSION_CODE": pin["version_code"],
        "CATBENCH_APK_SHA256": pin["apk_sha256"],
        "CATBENCH_PLAN_FILE_SHA256": "",
        "ANDROID_SERIAL": self.context.device_serial,
    })
    if attempt.condition == "c2_g":
      env.update({
          "CATBENCH_TASK_BREAKDOWN_FILE": str(self.context.c2_g_breakdown),
          "CATBENCH_TASK_BREAKDOWN_MODE": "prepend",
          "CATBENCH_TASK_BREAKDOWN_REQUIRED": "1",
          "CATBENCH_PLAN_FILE_SHA256": self.context.c2_g_sha256,
      })
    elif attempt.condition == "c2_o":
      env.update({
          "CATBENCH_TASK_BREAKDOWN_FILE": str(self.context.c2_o_breakdown),
          "CATBENCH_TASK_BREAKDOWN_MODE": "prepend",
          "CATBENCH_TASK_BREAKDOWN_REQUIRED": "1",
          "CATBENCH_PLAN_FILE_SHA256": self.context.c2_o_sha256,
      })
    return env

  def _validate_episode(
      self,
      episode: Mapping[str, Any],
      attempt: AttemptSpec,
      task: RealTask,
      model: FrozenModel,
      pin: Mapping[str, str],
      runner_config_sha256: str,
  ) -> tuple[str, float | None]:
    release_policy = schedule_builder.release_policy(attempt.release_id)
    required = {
        "task_template": task.task_template,
        "instance_id": attempt.instance_id,
        "semantic_task_id": attempt.semantic_task_id,
        "catbench_condition": attempt.condition,
        "catbench_condition_config_valid": True,
        "release_id": attempt.release_id,
        "release_purpose": release_policy["release_purpose"],
        "artifact_role": release_policy["artifact_role"],
        "analysis_eligible": release_policy["analysis_eligible"],
        "cohort_sha256": attempt.cohort_sha256,
        "episode_runtime_policy_sha256": (
            self.context.episode_runtime_policy_sha256
        ),
        "schedule_manifest_sha256": (
            self.context.bundle.schedule_manifest_sha256
        ),
        "code_revision": self.context.source_revision,
        "package_name": task.package_name,
        "app_id": attempt.app_id,
        "model_name": model.name,
        "model_revision": model.revision,
        "runner_config_sha256": runner_config_sha256,
        "model_config_sha256": self.context.model_config_sha256,
        "model_endpoint_attestation_sha256": (
            self.context.model_endpoint_attestation_sha256
        ),
        "app_pins_sha256": self.context.app_pins_sha256,
        "installed_app_attestation_sha256": (
            self.context.installed_app_attestation_sha256
        ),
        "pair_id": attempt.pair_id,
        "slot_id": attempt.slot_id,
        "attempt_id": attempt.attempt_id,
        "attempt_index": attempt.attempt_index,
        "snapshot_family_id": attempt.snapshot_family_id,
        "snapshot_clone_id": attempt.snapshot_clone_id,
        "app_version": pin["version_name"],
        "app_version_code": pin["version_code"],
        "apk_sha256": pin["apk_sha256"],
        "task_random_seed": attempt.task_random_seed,
        "n_task_combinations": attempt.n_task_combinations,
        "schedule_seed": attempt.schedule_seed,
        "plan_file_sha256": (
            ""
            if attempt.condition == "c1"
            else self.context.c2_g_sha256
            if attempt.condition == "c2_g"
            else self.context.c2_o_sha256
        ),
    }
    for field, expected in required.items():
      observed = episode.get(field)
      if type(observed) is not type(expected) or observed != expected:
        raise InfrastructureInvalid(f"episode_provenance_mismatch:{field}")
    status = str(episode.get("catbench_episode_status") or "")
    if status not in VALID_TERMINAL_STATUSES:
      raise InfrastructureInvalid("episode_terminal_status_missing_or_invalid")
    semantic_goal_sha256 = str(episode.get("semantic_goal_sha256") or "")
    semantic_parameter_sha256 = str(
        episode.get("semantic_parameter_sha256") or ""
    )
    if not HEX_SHA256.fullmatch(semantic_goal_sha256):
      raise InfrastructureInvalid("episode_invalid_semantic_goal_sha256")
    if not HEX_SHA256.fullmatch(semantic_parameter_sha256):
      raise InfrastructureInvalid("episode_invalid_semantic_parameter_sha256")
    if attempt.condition == "c1":
      if episode.get("task_breakdown_metadata") or episode.get("task_breakdown_text"):
        raise InfrastructureInvalid("c1_contains_task_breakdown")
    else:
      metadata = episode.get("task_breakdown_metadata")
      if not isinstance(metadata, dict):
        raise InfrastructureInvalid("c2_missing_task_breakdown_metadata")
      expected_metadata = {
          "task_template": task.task_template,
          "instance_id": attempt.instance_id,
          "semantic_task_id": attempt.semantic_task_id,
          "semantic_goal_sha256": semantic_goal_sha256,
          "plan_key": (
              f"{attempt.semantic_task_id}|instance={attempt.instance_id}|"
              f"{semantic_goal_sha256}"
          ),
          "condition": "application_independent_breakdown_prepend",
      }
      for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
          raise InfrastructureInvalid(f"c2_metadata_mismatch:{field}")
      breakdown_text = episode.get("task_breakdown_text")
      if not isinstance(breakdown_text, str) or not breakdown_text.strip():
        raise InfrastructureInvalid("c2_missing_task_breakdown_text")
      plan_sha256 = str(metadata.get("plan_sha256") or "")
      if (
          not HEX_SHA256.fullmatch(plan_sha256)
          or _sha256_bytes(breakdown_text.encode("utf-8")) != plan_sha256
      ):
        raise InfrastructureInvalid("c2_plan_sha256_mismatch")
    reward = episode.get("is_successful")
    if status == "valid_success" and (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or float(reward) <= 0.5
    ):
      raise InfrastructureInvalid("valid_success_reward_contract_mismatch")
    if status == "valid_failure" and (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or float(reward) > 0.5
    ):
      raise InfrastructureInvalid("valid_failure_reward_contract_mismatch")
    if status.startswith("valid_") and episode.get("exception_info") is not None:
      raise InfrastructureInvalid("valid_episode_has_exception_info")
    return status, float(reward) if isinstance(reward, (int, float)) else None

  def execute(self, attempt: AttemptSpec) -> AttemptOutcome:
    attempt_dir = self._attempt_dir(attempt)
    if attempt_dir.exists():
      raise InfrastructureInvalid("attempt_directory_already_exists")
    attempt_dir.mkdir(parents=True)
    result_contract_path = attempt_dir / "result_contract.json"
    prepare_receipt = ""
    release_receipt = ""
    artifact_path = ""
    artifact_sha256 = ""
    reason_code = ""
    status = "invalid_infrastructure"
    is_successful: float | None = None
    try:
      task = self.context.real_tasks[
          (attempt.category, attempt.app_id, attempt.semantic_task_id)
      ]
      model = self.context.models[attempt.model]
      pin = self.context.app_pins[attempt.app_id]
      if task.package_name != pin["package_name"] or pin["category"] != attempt.category:
        raise InfrastructureInvalid("task_app_pin_identity_mismatch")
      prepare_receipt = str(self._hook("clone_activate", attempt, attempt_dir))
      self._device_pin_preflight(task, pin)
      runner_config_sha256 = self._runner_config_sha256(model)
      checkpoint_dir = attempt_dir / "checkpoint"
      runner_output = attempt_dir / "runner_output"
      command = [
          self.context.python_bin,
          str(model.runner),
          f"--suite_family={attempt.suite_family}",
          f"--tasks={task.task_template}",
          "--n_task_combinations=3",
          "--task_random_seed=30",
          "--fixed_task_seed=false",
          f"--checkpoint_dir={checkpoint_dir}",
          f"--output_path={runner_output}",
          f"--console_port={self.context.console_port}",
          f"--grpc_port={self.context.grpc_port}",
          *model.args,
      ]
      env = self._episode_env(attempt, task, model, pin, runner_config_sha256)
      provenance_env_keys = {
          "ANDROID_SERIAL",
          "CATBENCH_INSTANCE_ID",
          "CATBENCH_CONDITION",
          "CATBENCH_RELEASE_ID",
          "CATBENCH_RELEASE_PURPOSE",
          "CATBENCH_ARTIFACT_ROLE",
          "CATBENCH_ANALYSIS_ELIGIBLE",
          "CATBENCH_CODE_REVISION",
          "CATBENCH_COHORT_SHA256",
          "CATBENCH_EPISODE_RUNTIME_POLICY_SHA256",
          "CATBENCH_SCHEDULE_MANIFEST_SHA256",
          "CATBENCH_PAIR_ID",
          "CATBENCH_SLOT_ID",
          "CATBENCH_ATTEMPT_ID",
          "CATBENCH_ATTEMPT_INDEX",
          "CATBENCH_SNAPSHOT_FAMILY_ID",
          "CATBENCH_SNAPSHOT_CLONE_ID",
          "CATBENCH_APP_ID",
          "CATBENCH_MODEL_NAME",
          "CATBENCH_MODEL_REVISION",
          "CATBENCH_RUNNER_CONFIG_SHA256",
          "CATBENCH_MODEL_CONFIG_SHA256",
          "CATBENCH_MODEL_ENDPOINT_ATTESTATION_SHA256",
          "CATBENCH_APP_PINS_SHA256",
          "CATBENCH_INSTALLED_APP_ATTESTATION_SHA256",
          "CATBENCH_TASK_RANDOM_SEED",
          "CATBENCH_N_TASK_COMBINATIONS",
          "CATBENCH_SCHEDULE_SEED",
          "CATBENCH_APP_VERSION",
          "CATBENCH_APP_VERSION_CODE",
          "CATBENCH_APK_SHA256",
          "CATBENCH_PLAN_FILE_SHA256",
          "CATBENCH_TASK_BREAKDOWN_FILE",
          "CATBENCH_TASK_BREAKDOWN_MODE",
          "CATBENCH_TASK_BREAKDOWN_REQUIRED",
          *self.context.episode_runtime_environment,
      }
      command_contract = {
          "command": command,
          "command_sha256": _sha256_bytes(_canonical_json(command).encode("utf-8")),
          "episode_runner_timeout_seconds": EPISODE_RUNNER_TIMEOUT_SECONDS,
          "snapshot_hook_timeout_seconds": SNAPSHOT_HOOK_TIMEOUT_SECONDS,
          "episode_runtime_policy": self.context.episode_runtime_policy,
          "episode_runtime_policy_sha256": (
              self.context.episode_runtime_policy_sha256
          ),
          "environment": {
              key: env[key]
              for key in sorted(provenance_env_keys)
              if key in env
          },
      }
      _atomic_write(
          attempt_dir / "runner_contract.json",
          (json.dumps(command_contract, indent=2, sort_keys=True) + "\n").encode(
              "utf-8"
          ),
      )
      with (attempt_dir / "runner.stdout").open("wb") as stdout_handle, (
          attempt_dir / "runner.stderr"
      ).open("wb") as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
            timeout=EPISODE_RUNNER_TIMEOUT_SECONDS,
        )
      if completed.returncode != 0:
        raise InfrastructureInvalid(f"runner_exit_code:{completed.returncode}")
      expected_checkpoint = checkpoint_dir / f"{task.task_template}_{attempt.instance_id}.pkl.gz"
      episode = _load_exact_checkpoint(checkpoint_dir, expected_checkpoint)
      status, is_successful = self._validate_episode(
          episode, attempt, task, model, pin, runner_config_sha256
      )
      artifact_path = str(expected_checkpoint)
      artifact_sha256 = _sha256_path(expected_checkpoint)
      reason_code = "episode_contract_validated"
    except (
        InfrastructureInvalid,
        ArtifactValidationError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
      reason_code = str(exc) or type(exc).__name__
      status = "invalid_infrastructure"
    finally:
      try:
        release_receipt = str(self._hook("release", attempt, attempt_dir))
      except (
          InfrastructureInvalid,
          ArtifactValidationError,
          OSError,
          subprocess.SubprocessError,
      ) as exc:
        status = "invalid_infrastructure"
        cleanup_reason = f"snapshot_release_failed:{exc}"
        reason_code = (
            f"{reason_code};secondary_{cleanup_reason}"
            if reason_code
            else cleanup_reason
        )
    contract = {
        "schema_version": 1,
        **_attempt_provenance(attempt),
        "status": status,
        "reason_code": reason_code,
        "is_successful": is_successful,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "snapshot_prepare_receipt": prepare_receipt,
        "snapshot_release_receipt": release_receipt,
        "recorded_at": _utc_now(),
    }
    _atomic_write(
        result_contract_path,
        (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return AttemptOutcome(
        status=status,
        artifact_path=artifact_path,
        result_contract_path=str(result_contract_path),
        reason_code=reason_code,
        is_successful=is_successful,
        artifact_sha256=artifact_sha256,
        snapshot_prepare_receipt=prepare_receipt,
        snapshot_release_receipt=release_receipt,
    )


def _load_env_file(path: Path) -> None:
  if not path.exists():
    return
  for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    tokens = shlex.split(line, comments=True, posix=True)
    if tokens and tokens[0] == "export":
      tokens = tokens[1:]
    for token in tokens:
      if "=" not in token:
        continue
      key, value = token.split("=", 1)
      if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        os.environ.setdefault(key, _expand_env(value))


def _validate_local_app_artifacts(
    *, cohort_path: Path, pins_path: Path, apps_path: Path, artifact_root: Path
) -> dict[str, Any]:
  from audit_pinned_app_artifacts import audit

  report = audit(cohort_path, pins_path, apps_path, artifact_root)
  if report.get("valid") is not True:
    invalid = [row["app_id"] for row in report["apps"] if not row["valid"]]
    raise ArtifactValidationError(
        f"Pinned local APK/XAPK preflight failed for real apps: {invalid}"
    )
  return report


def _initialize_or_validate_consumer_manifest(
    state_dir: Path, manifest: dict[str, Any]
) -> None:
  path = state_dir / CONSUMER_MANIFEST_FILE
  if path.exists():
    existing = _read_json(path)
    # created_at is historical; all immutable execution inputs must match.
    comparable_existing = dict(existing)
    comparable_new = dict(manifest)
    comparable_existing.pop("created_at", None)
    comparable_new.pop("created_at", None)
    if comparable_existing != comparable_new:
      raise ArtifactValidationError(
          "Consumer state was created from different immutable inputs"
      )
    return
  if any(state_dir.iterdir()):
    raise ArtifactValidationError(
        "Non-empty state directory lacks consumer_manifest.json"
    )
  _atomic_write(
      path,
      (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
  )


def _validate_state_root_before_preflight(state_dir: Path) -> None:
  """Reject user-selected subsets or unrelated preexisting output content."""
  children = {path.name: path for path in state_dir.iterdir()}
  manifest_present = CONSUMER_MANIFEST_FILE in children
  if not manifest_present:
    unexpected = sorted(set(children) - {LOCK_FILE})
    if unexpected:
      raise ArtifactValidationError(
          "Fresh output_root contains preexisting content and no immutable "
          f"consumer manifest: {unexpected}"
      )
    return
  allowed = {
      LOCK_FILE,
      CONSUMER_MANIFEST_FILE,
      "preflight",
      ATTEMPTS_DIR,
      JOURNAL_FILE,
      RUNTIME_LEDGER_FILE,
      SELECTION_FILE,
      STATE_COMMIT_FILE,
  }
  unexpected = sorted(set(children) - allowed)
  if unexpected:
    raise ArtifactValidationError(
        f"Consumer output_root contains unrecognized content: {unexpected}"
    )
  for name, path in children.items():
    if path.is_symlink():
      raise ArtifactValidationError(f"Consumer state entry is a symlink: {name}")


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--schedule_dir", required=True)
  parser.add_argument("--cohort_manifest", required=True)
  parser.add_argument("--c2_g_breakdown_file", required=True)
  parser.add_argument("--c2_g_attempt_audit", required=True)
  parser.add_argument("--c2_o_breakdown_file", required=True)
  parser.add_argument("--c2_g_approval_manifest", required=True)
  parser.add_argument("--c2_o_approval_manifest", required=True)
  parser.add_argument("--base_snapshot_manifest", required=True)
  parser.add_argument("--model_endpoint_attestation_manifest", required=True)
  parser.add_argument("--installed_app_attestation_manifest", required=True)
  parser.add_argument("--verifier_conformance_manifest", required=True)
  parser.add_argument("--verifier_conformance_evidence_root", required=True)
  parser.add_argument("--snapshot_hook", required=True)
  parser.add_argument("--output_root", required=True)
  parser.add_argument("--model_config", default=str(DEFAULT_MODEL_CONFIG))
  parser.add_argument("--app_pins", default=str(DEFAULT_PINS))
  parser.add_argument("--app_inventory", default=str(DEFAULT_APPS))
  parser.add_argument("--app_artifact_root", required=True)
  parser.add_argument("--env_file", default=str(BENCHMARK_ROOT / "configs" / "catbench.env"))
  parser.add_argument("--python_bin", default=sys.executable)
  parser.add_argument("--adb_path", default=shutil.which("adb") or "adb")
  parser.add_argument("--device_serial", required=True)
  parser.add_argument("--console_port", type=int, required=True)
  parser.add_argument("--grpc_port", type=int, required=True)
  parser.add_argument(
      "--preflight_only",
      action="store_true",
      help="Run every immutable-input gate but do not call snapshot/model runners.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = _build_parser().parse_args(argv)
  schedule_dir = Path(args.schedule_dir).expanduser().resolve()
  cohort_path = Path(args.cohort_manifest).expanduser().resolve()
  output_root = Path(args.output_root).expanduser().resolve()
  model_path = Path(args.model_config).expanduser().resolve()
  pins_path = Path(args.app_pins).expanduser().resolve()
  apps_path = Path(args.app_inventory).expanduser().resolve()
  c2_g_path = Path(args.c2_g_breakdown_file).expanduser().resolve()
  c2_g_attempt_audit_path = Path(
      args.c2_g_attempt_audit
  ).expanduser().resolve()
  c2_o_path = Path(args.c2_o_breakdown_file).expanduser().resolve()
  c2_g_approval_path = Path(
      args.c2_g_approval_manifest
  ).expanduser().resolve()
  c2_o_approval_path = Path(
      args.c2_o_approval_manifest
  ).expanduser().resolve()
  base_snapshot_path = Path(args.base_snapshot_manifest).expanduser().resolve()
  model_attestation_path = Path(
      args.model_endpoint_attestation_manifest
  ).expanduser().resolve()
  installed_app_attestation_path = Path(
      args.installed_app_attestation_manifest
  ).expanduser().resolve()
  verifier_conformance_path = Path(
      args.verifier_conformance_manifest
  ).expanduser().resolve()
  verifier_conformance_evidence_root_input = Path(
      args.verifier_conformance_evidence_root
  ).expanduser()
  snapshot_hook = Path(args.snapshot_hook).expanduser().resolve()
  env_path = Path(args.env_file).expanduser().resolve()
  artifact_root = Path(args.app_artifact_root).expanduser().resolve()
  lock_handle: Any | None = None
  lock_acquired = False
  try:
    # This is deliberately the first effectful preflight.  In particular, it
    # precedes mkdir/open of output_root, so the consumer cannot dirty its own
    # source checkout before claiming that the revision was clean.
    source_revision = _source_revision_clean()
    verifier_conformance_evidence_root = (
        _validate_verifier_evidence_root_location(
            verifier_conformance_evidence_root_input, output_root
        )
    )
    _validate_prewrite_locations(
        output_root,
        (
            schedule_dir,
            cohort_path,
            model_path,
            pins_path,
            apps_path,
            c2_g_path,
            c2_g_attempt_audit_path,
            c2_o_path,
            c2_g_approval_path,
            c2_o_approval_path,
            base_snapshot_path,
            model_attestation_path,
            installed_app_attestation_path,
            verifier_conformance_path,
            verifier_conformance_evidence_root,
            snapshot_hook,
            env_path,
            artifact_root,
        ),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_root / LOCK_FILE).open("a+b")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_acquired = True
    _validate_state_root_before_preflight(output_root)
    _load_env_file(env_path)
    bundle = load_and_validate_bundle(schedule_dir, cohort_path)
    if c2_g_path == c2_o_path:
      raise ArtifactValidationError("C2-G and C2-O must be separate frozen files")
    _validate_plan_role(c2_g_path, "c2_g")
    _validate_plan_role(c2_o_path, "c2_o")
    if _sha256_path(c2_g_path) == _sha256_path(c2_o_path):
      raise ArtifactValidationError("C2-G and C2-O plan artifacts are identical")
    if not snapshot_hook.is_file() or not os.access(snapshot_hook, os.X_OK):
      raise ArtifactValidationError(
          f"Snapshot hook is missing or non-executable: {snapshot_hook}"
      )
    real_tasks = resolve_real_tasks(bundle.cohort)
    models = load_models(model_path, bundle.cohort)
    _validate_prewrite_locations(
        output_root, (task_model.runner for task_model in models.values())
    )
    validate_model_endpoint_attestations(
        model_attestation_path,
        bundle=bundle,
        models=models,
        model_config_sha256=_sha256_path(model_path),
    )
    app_pins = load_app_pins(pins_path, bundle.cohort)
    for key, task in real_tasks.items():
      pin = app_pins[task.app_id]
      if pin["category"] != task.category or pin["package_name"] != task.package_name:
        raise ArtifactValidationError(f"Real task/app pin mismatch: {key}")
    base_snapshot = validate_base_snapshot_manifest(
        base_snapshot_path,
        bundle=bundle,
        pins_sha256=_sha256_path(pins_path),
    )
    validate_installed_app_attestation(
        installed_app_attestation_path,
        bundle=bundle,
        pins=app_pins,
        pins_sha256=_sha256_path(pins_path),
        base_snapshot=base_snapshot,
    )
    _, verifier_conformance_evidence_inventory_sha256 = (
        validate_verifier_conformance_manifest(
            verifier_conformance_path,
            bundle=bundle,
            pins=app_pins,
            base_snapshot=base_snapshot,
            evidence_root=verifier_conformance_evidence_root,
        )
    )
    artifact_report = _validate_local_app_artifacts(
        cohort_path=cohort_path,
        pins_path=pins_path,
        apps_path=apps_path,
        artifact_root=artifact_root,
    )
    preflight_dir = output_root / "preflight"
    preflight_dir.mkdir(exist_ok=True)
    _atomic_write(
        preflight_dir / "app_artifact_preflight.json",
        (json.dumps(artifact_report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    expected_plan_count = len({
        (str(row["semantic_task_id"]), int(row["instance_id"]))
        for row in bundle.schedule
    })
    c2_g_audit_report = validate_c2_g_attempt_audit(
        c2_g_attempt_audit_path,
        breakdown_path=c2_g_path,
        bundle=bundle,
        expected_plan_count=expected_plan_count,
    )
    _atomic_write(
        preflight_dir / "c2_g_attempt_audit_preflight.json",
        (json.dumps(c2_g_audit_report, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    for condition, breakdown, approval in (
        ("c2_g", c2_g_path, c2_g_approval_path),
        ("c2_o", c2_o_path, c2_o_approval_path),
    ):
      run_plan_preflight(
          python_bin=args.python_bin,
          cohort_path=cohort_path,
          breakdown_path=breakdown,
          condition=condition,
          report_path=preflight_dir / f"{condition}_plan_preflight.json",
      )
      _validate_two_person_approval(
          approval,
          breakdown_path=breakdown,
          condition=condition,
          release_id=str(bundle.cohort["release_id"]),
          expected_plan_count=expected_plan_count,
          c2_g_attempt_audit_path=(
              c2_g_attempt_audit_path if condition == "c2_g" else None
          ),
      )
    consumer_manifest = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "release_id": bundle.cohort["release_id"],
        **schedule_builder.release_policy(str(bundle.cohort["release_id"])),
        "primary_reporter_acceptance_permitted": bundle.schedule_manifest[
            "primary_reporter_acceptance_permitted"
        ],
        "cohort_sha256": bundle.cohort_sha256,
        "episode_runtime_policy": bundle.cohort["episode_runtime_policy"],
        "episode_runtime_policy_sha256": bundle.schedule_manifest[
            "episode_runtime_policy_sha256"
        ],
        "schedule_manifest_sha256": bundle.schedule_manifest_sha256,
        "ledger_schema_sha256": bundle.ledger_schema_sha256,
        "model_config_sha256": _sha256_path(model_path),
        "model_endpoint_attestation_sha256": _sha256_path(
            model_attestation_path
        ),
        "app_pins_sha256": _sha256_path(pins_path),
        "installed_app_attestation_sha256": _sha256_path(
            installed_app_attestation_path
        ),
        "verifier_conformance_manifest_sha256": _sha256_path(
            verifier_conformance_path
        ),
        "verifier_conformance_evidence_inventory_sha256": (
            verifier_conformance_evidence_inventory_sha256
        ),
        "app_inventory_sha256": _sha256_path(apps_path),
        "c2_g_breakdown_sha256": _sha256_path(c2_g_path),
        "c2_g_attempt_audit_sha256": _sha256_path(c2_g_attempt_audit_path),
        "c2_o_breakdown_sha256": _sha256_path(c2_o_path),
        "c2_g_approval_sha256": _sha256_path(
            c2_g_approval_path
        ),
        "c2_o_approval_sha256": _sha256_path(
            c2_o_approval_path
        ),
        "base_snapshot_manifest_sha256": _sha256_path(base_snapshot_path),
        "snapshot_hook_sha256": _sha256_path(snapshot_hook),
        "source_revision": source_revision,
        "device_serial": args.device_serial,
        "console_port": args.console_port,
        "grpc_port": args.grpc_port,
        "episode_slot_count": len(bundle.schedule),
        "paired_block_count": len(bundle.ledger_seed),
        "selective_filters": False,
        "automatic_state_recovery": True,
        "halt_after_each_infrastructure_invalid": True,
        "snapshot_hook_timeout_seconds": SNAPSHOT_HOOK_TIMEOUT_SECONDS,
        "episode_runner_timeout_seconds": EPISODE_RUNNER_TIMEOUT_SECONDS,
    }
    # The lock file and preflight reports are expected. Only the immutable
    # consumer manifest governs whether an existing state can be continued.
    manifest_path = output_root / CONSUMER_MANIFEST_FILE
    if manifest_path.exists():
      _initialize_or_validate_consumer_manifest(output_root, consumer_manifest)
    else:
      _atomic_write(
          manifest_path,
          (json.dumps(consumer_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
      )
    if args.preflight_only:
      print("OK: complete frozen schedule consumer preflight passed; no episode launched.")
      return 0
    context = ExecutionContext(
        bundle=bundle,
        real_tasks=real_tasks,
        models=models,
        app_pins=app_pins,
        model_config_sha256=_sha256_path(model_path),
        model_endpoint_attestation_sha256=_sha256_path(
            model_attestation_path
        ),
        app_pins_sha256=_sha256_path(pins_path),
        installed_app_attestation_sha256=_sha256_path(
            installed_app_attestation_path
        ),
        episode_runtime_policy=copy.deepcopy(
            bundle.cohort["episode_runtime_policy"]
        ),
        episode_runtime_environment={
            str(key): str(value)
            for key, value in bundle.cohort["episode_runtime_policy"][
                "environment"
            ].items()
        },
        episode_runtime_policy_sha256=bundle.schedule_manifest[
            "episode_runtime_policy_sha256"
        ],
        source_revision=source_revision,
        c2_g_breakdown=c2_g_path,
        c2_o_breakdown=c2_o_path,
        c2_g_sha256=_sha256_path(c2_g_path),
        c2_o_sha256=_sha256_path(c2_o_path),
        base_snapshot=base_snapshot,
        base_snapshot_manifest_sha256=_sha256_path(base_snapshot_path),
        snapshot_hook=snapshot_hook,
        snapshot_hook_sha256=_sha256_path(snapshot_hook),
        python_bin=args.python_bin,
        adb_path=args.adb_path,
        device_serial=args.device_serial,
        console_port=args.console_port,
        grpc_port=args.grpc_port,
        output_root=output_root,
    )
    consumer = ScheduleConsumer(
        bundle, output_root, SubprocessEpisodeExecutor(context)
    )
    return consumer.run_until_complete(halt_after_invalid=True)
  except BlockingIOError:
    print(
        "FAIL: another frozen schedule consumer holds the output lock",
        file=sys.stderr,
    )
    return 2
  except (ConsumerError, OSError, subprocess.SubprocessError, ValueError) as exc:
    print(f"FAIL: {exc}", file=sys.stderr)
    return 2
  finally:
    if lock_handle is not None:
      if lock_acquired:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
      lock_handle.close()


if __name__ == "__main__":
  raise SystemExit(main())
