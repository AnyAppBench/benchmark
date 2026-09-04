#!/usr/bin/env python3
"""Strictly pair CATBench baseline and plan-assistance episodes.

The report is deliberately conservative.  An episode is eligible for paired
analysis only when its condition and run provenance are explicit, it has no
runner/evaluator exception, and its unique experimental slot occurs exactly
once.  Baseline and treatment records are then paired only when the complete
goal hash, seeds, semantic task identity, app package/version, and code
revision agree.

The treatment is an intervention that supplies an external task breakdown.
Accordingly, baseline-fail -> treatment-pass is called *planning-responsive*;
failure under both conditions is only *residual under plan assistance*.  The
latter is not evidence of grounding by itself.

Outputs (under ``--out_dir``):

* ``paired_summary.json``: provenance audit and paired statistics.
* ``paired_summary.md``: compact human-readable report.
* ``paired_per_task.jsonl``: one comparison record per experimental slot.

The process exits with status 2 after writing the reports if any strict
validity check fails (invalid episodes, duplicate slots, missing counterparts,
provenance mismatches, roster mismatches, or inconsistent treatment plans).

For the frozen primary release, ordinary matrix manifests are not accepted.
The reader verifies the consumer's committed ``selected_triplets.jsonl``,
hash-chained journal, exact recompiled schedule, result contracts, and artifact
hashes, then admits only one complete C1/C2-G/C2-O round per semantic cell.
Infrastructure-invalid prior rounds remain in the audit but not the estimator.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import decimal
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

# This reader installs compatibility shims for historical AndroidWorld pickle
# module paths.  Do not replace it with a direct pickle.load call.
from classify_catbench_failures import _read_pkl_gz  # noqa: E402
import build_catbench_frozen_schedule as schedule_builder  # noqa: E402


SlotKey = tuple[str, str, str, str, int]
RosterKey = tuple[str, str, str]
PRIMARY_CONDITIONS = ("c1", "c2_g", "c2_o")
PRIMARY_RELEASE_ID = "catbench_acl_revision_5cat_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PAIR_ID = re.compile(r"^pair_[0-9a-f]{24}$")


class SelectedTripletValidationError(ValueError):
  """Raised when committed whole-triplet selection evidence is inconsistent."""


@dataclasses.dataclass
class Harvest:
  """Eligible rows plus invalid artifacts from one experimental condition."""

  rows: dict[SlotKey, dict[str, Any]]
  invalid_records: list[dict[str, Any]]
  invalid_slots: dict[SlotKey, list[dict[str, Any]]]
  roster: set[RosterKey] = dataclasses.field(default_factory=set)
  duplicate_roster: set[RosterKey] = dataclasses.field(default_factory=set)
  selection_audit: dict[str, Any] | None = None

  def summary(self) -> dict[str, Any]:
    return {
        "n_eligible": len(self.rows),
        "n_invalid_records": len(self.invalid_records),
        "n_invalid_slots": len(self.invalid_slots),
        "n_roster_cells": len(self.roster),
        "duplicate_roster": [list(key) for key in sorted(self.duplicate_roster)],
        "selection_audit": self.selection_audit,
        "invalid_records": self.invalid_records,
    }


def _jsonable_scalar(value: Any) -> str | int | float | bool | None:
  if value is None or isinstance(value, (str, int, bool)):
    return value
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if hasattr(value, "item"):
    try:
      return _jsonable_scalar(value.item())
    except (TypeError, ValueError):
      pass
  return str(value)


def _present(value: Any) -> bool:
  return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _first(*values: Any) -> Any:
  for value in values:
    if _present(value):
      return value
  return None


def _mapping(value: Any) -> dict[str, Any]:
  return value if isinstance(value, dict) else {}


def _sha256_text(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_int(value: Any) -> int | None:
  if value is None or isinstance(value, bool):
    return None
  try:
    parsed = int(value)
  except (TypeError, ValueError, OverflowError):
    return None
  return parsed


def _parse_cli_arg(command: Any, flag: str) -> Any:
  if not isinstance(command, list):
    return None
  prefix = f"--{flag}="
  for index, item in enumerate(command):
    text = str(item)
    if text.startswith(prefix):
      return text[len(prefix):]
    if text == f"--{flag}" and index + 1 < len(command):
      return command[index + 1]
  return None


def _manifest_app_metadata(manifest: dict[str, Any], app_id: str) -> dict[str, Any]:
  for key in ("app_provenance", "apps"):
    value = manifest.get(key)
    if isinstance(value, dict):
      candidate = value.get(app_id)
      if isinstance(candidate, dict):
        return candidate
    elif isinstance(value, list):
      for candidate in value:
        if isinstance(candidate, dict) and candidate.get("app_id") == app_id:
          return candidate
  return {}


def _condition_metadata(
    episode: dict[str, Any],
    job: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
  """Extract explicit identity/provenance without inventing defaults."""
  episode_provenance = _mapping(episode.get("provenance"))
  job_provenance = _mapping(job.get("provenance"))
  manifest_provenance = _mapping(manifest.get("provenance"))
  breakdown = _mapping(episode.get("task_breakdown_metadata"))
  matrix_args = _mapping(manifest.get("matrix_args"))
  app_id = str(_first(job.get("app_id"), episode.get("app_id")) or "")
  app_metadata = _manifest_app_metadata(manifest, app_id)

  task_random_seed = _first(
      episode.get("task_random_seed"),
      episode_provenance.get("task_random_seed"),
      job.get("task_random_seed"),
      job_provenance.get("task_random_seed"),
      _parse_cli_arg(job.get("command"), "task_random_seed"),
      matrix_args.get("task_random_seed"),
  )
  return {
      "condition": episode.get("catbench_condition"),
      "condition_config_valid": episode.get(
          "catbench_condition_config_valid"
      ),
      "episode_status": episode.get("catbench_episode_status"),
      "exception_stage": episode.get("catbench_exception_stage"),
      "exception_attribution": episode.get(
          "catbench_exception_attribution"
      ),
      "exception_valid_agent_failure": episode.get(
          "catbench_exception_valid_agent_failure"
      ),
      "exception_declared_agent_output": episode.get(
          "catbench_exception_declared_agent_output"
      ),
      "exception_type": episode.get("catbench_exception_type"),
      "exception_failure_code": episode.get(
          "catbench_exception_failure_code"
      ),
      "semantic_task_id": _first(
          episode.get("semantic_task_id"),
          episode_provenance.get("semantic_task_id"),
          breakdown.get("semantic_task_id"),
      ),
      "semantic_goal_sha256": _first(
          episode.get("semantic_goal_sha256"),
          episode_provenance.get("semantic_goal_sha256"),
          breakdown.get("semantic_goal_sha256"),
      ),
      "semantic_parameter_sha256": _first(
          episode.get("semantic_parameter_sha256"),
          episode_provenance.get("semantic_parameter_sha256"),
          breakdown.get("semantic_parameter_sha256"),
      ),
      "plan_key": _first(
          episode.get("plan_key"),
          episode_provenance.get("plan_key"),
          breakdown.get("plan_key"),
      ),
      "plan_sha256": _first(
          episode.get("plan_sha256"),
          episode_provenance.get("plan_sha256"),
          breakdown.get("plan_sha256"),
      ),
      "task_random_seed": _jsonable_scalar(task_random_seed),
      "instance_seed": _jsonable_scalar(_first(
          episode.get("instance_seed"),
          episode.get("seed"),
          episode_provenance.get("instance_seed"),
      )),
      "app_name": _first(
          job.get("app_name"), episode.get("app_name"), app_metadata.get("display_name")
      ),
      "package_name": _first(
          episode.get("package_name"),
          episode_provenance.get("package_name"),
          job.get("package_name"),
          job_provenance.get("package_name"),
          app_metadata.get("package_name"),
      ),
      "app_version": _first(
          episode.get("app_version"),
          episode.get("version_name"),
          episode_provenance.get("app_version"),
          episode_provenance.get("version_name"),
          job.get("app_version"),
          job.get("version_name"),
          job_provenance.get("app_version"),
          job_provenance.get("version_name"),
          app_metadata.get("app_version"),
          app_metadata.get("version_name"),
      ),
      "app_version_code": _jsonable_scalar(_first(
          episode.get("app_version_code"),
          episode.get("version_code"),
          episode_provenance.get("app_version_code"),
          episode_provenance.get("version_code"),
          job.get("app_version_code"),
          job.get("version_code"),
          job_provenance.get("app_version_code"),
          job_provenance.get("version_code"),
          app_metadata.get("app_version_code"),
          app_metadata.get("version_code"),
      )),
      "apk_sha256": _first(
          episode.get("apk_sha256"),
          episode_provenance.get("apk_sha256"),
          job.get("apk_sha256"),
          job_provenance.get("apk_sha256"),
          app_metadata.get("apk_sha256"),
      ),
      "code_revision": _first(
          episode.get("code_revision"),
          episode.get("git_revision"),
          episode_provenance.get("code_revision"),
          episode_provenance.get("git_revision"),
          job.get("code_revision"),
          job.get("git_revision"),
          job_provenance.get("code_revision"),
          job_provenance.get("git_revision"),
          manifest.get("code_revision"),
          manifest.get("git_revision"),
          manifest_provenance.get("code_revision"),
          manifest_provenance.get("git_revision"),
      ),
      "release_id": _first(
          episode.get("release_id"),
          episode_provenance.get("release_id"),
          job.get("release_id"),
          job_provenance.get("release_id"),
          manifest.get("release_id"),
          manifest_provenance.get("release_id"),
      ),
      "model_revision": _first(
          episode.get("model_revision"),
          episode_provenance.get("model_revision"),
          job.get("model_revision"),
          job_provenance.get("model_revision"),
      ),
      "runner_config_sha256": _first(
          episode.get("runner_config_sha256"),
          episode_provenance.get("runner_config_sha256"),
          job.get("runner_config_sha256"),
          job_provenance.get("runner_config_sha256"),
      ),
      "model_config_sha256": _first(
          episode.get("model_config_sha256"),
          episode_provenance.get("model_config_sha256"),
          job.get("model_config_sha256"),
          job_provenance.get("model_config_sha256"),
          manifest.get("model_config_sha256"),
          manifest_provenance.get("model_config_sha256"),
      ),
      "cohort_sha256": _first(
          episode.get("cohort_sha256"),
          episode_provenance.get("cohort_sha256"),
          manifest.get("cohort_sha256"),
      ),
      "schedule_manifest_sha256": _first(
          episode.get("schedule_manifest_sha256"),
          episode_provenance.get("schedule_manifest_sha256"),
          manifest.get("schedule_manifest_sha256"),
      ),
      "pair_id": _first(
          episode.get("pair_id"), episode_provenance.get("pair_id")
      ),
      "slot_id": _first(
          episode.get("slot_id"), episode_provenance.get("slot_id")
      ),
      "attempt_id": _first(
          episode.get("attempt_id"), episode_provenance.get("attempt_id")
      ),
      "attempt_index": _jsonable_scalar(_first(
          episode.get("attempt_index"),
          episode_provenance.get("attempt_index"),
      )),
      "snapshot_family_id": _first(
          episode.get("snapshot_family_id"),
          episode_provenance.get("snapshot_family_id"),
      ),
      "snapshot_clone_id": _first(
          episode.get("snapshot_clone_id"),
          episode_provenance.get("snapshot_clone_id"),
      ),
      "model_endpoint_attestation_sha256": _first(
          episode.get("model_endpoint_attestation_sha256"),
          episode_provenance.get("model_endpoint_attestation_sha256"),
          manifest.get("model_endpoint_attestation_sha256"),
      ),
      "app_pins_sha256": _first(
          episode.get("app_pins_sha256"),
          episode_provenance.get("app_pins_sha256"),
          manifest.get("app_pins_sha256"),
      ),
      "installed_app_attestation_sha256": _first(
          episode.get("installed_app_attestation_sha256"),
          episode_provenance.get("installed_app_attestation_sha256"),
          manifest.get("installed_app_attestation_sha256"),
      ),
      "schedule_seed": _jsonable_scalar(_first(
          episode.get("schedule_seed"),
          episode_provenance.get("schedule_seed"),
      )),
      "n_task_combinations": _jsonable_scalar(_first(
          episode.get("n_task_combinations"),
          episode_provenance.get("n_task_combinations"),
      )),
      "plan_file_sha256": _first(
          episode.get("plan_file_sha256"),
          episode_provenance.get("plan_file_sha256"),
      ),
  }


def _normalize_item(
    item: dict[str, Any], expected_condition: str
) -> tuple[SlotKey | None, dict[str, Any], list[str]]:
  episode = _mapping(item.get("episode"))
  job = _mapping(item.get("job"))
  manifest = _mapping(item.get("manifest"))
  pkl_path = str(item.get("pkl_path") or "")
  episode_index = _parse_int(item.get("episode_index")) or 0
  issues: list[str] = []

  model = str(_first(job.get("model_name"), episode.get("agent_name")) or "")
  category = str(_first(job.get("category"), episode.get("category")) or "")
  app_id = str(_first(job.get("app_id"), episode.get("app_id")) or "")
  task_template = str(
      _first(episode.get("task_template"), episode.get("name")) or ""
  )
  instance_id = _parse_int(episode.get("instance_id"))
  for field, value in (
      ("model", model),
      ("category", category),
      ("app_id", app_id),
      ("task_template", task_template),
      ("instance_id", instance_id),
  ):
    if not _present(value):
      issues.append(f"missing_identity:{field}")
  slot: SlotKey | None = None
  if model and category and app_id and task_template and instance_id is not None:
    slot = (model, category, app_id, task_template, instance_id)

  goal = str(episode.get("goal") or "")
  goal_hash = _sha256_text(goal) if goal else ""
  if not goal:
    issues.append("missing_goal")
  recorded_goal_hash = _first(
      episode.get("goal_sha256"),
      _mapping(episode.get("task_breakdown_metadata")).get("goal_sha256"),
  )
  if recorded_goal_hash and recorded_goal_hash != goal_hash:
    issues.append("recorded_goal_sha256_mismatch")

  provenance = _condition_metadata(episode, job, manifest)
  if provenance["condition"] != expected_condition:
    actual = provenance["condition"] if provenance["condition"] is not None else "<missing>"
    issues.append(f"condition:{actual!s}!=expected:{expected_condition}")
  if provenance.get("condition_config_valid") is False:
    issues.append("invalid_condition_configuration")

  exception_info = str(
      _first(episode.get("exception_info"), episode.get("EXCEPTION_INFO")) or ""
  ).strip()
  episode_status = provenance.get("episode_status")
  if episode_status not in {
      "valid_success",
      "valid_failure",
      "invalid_infrastructure",
  }:
    issues.append("missing_or_invalid_episode_status")
  elif episode_status == "invalid_infrastructure":
    issues.append("infrastructure_exception")
  elif exception_info:
    issues.append("exception_in_valid_episode")

  exception_stage = provenance.get("exception_stage")
  typed_exception_fields_present = any(
      _present(provenance.get(field))
      for field in (
          "exception_stage",
          "exception_attribution",
          "exception_type",
          "exception_failure_code",
      )
  ) or bool(
      provenance.get("exception_valid_agent_failure")
      or provenance.get("exception_declared_agent_output")
  )
  if exception_stage is not None and episode_status != "invalid_infrastructure":
    declared_agent_failure_is_exact = (
        episode_status == "valid_failure"
        and exception_stage == "agent"
        and provenance.get("exception_attribution")
        == "agent_output_parse_or_malformed_action"
        and provenance.get("exception_valid_agent_failure") is True
        and provenance.get("exception_declared_agent_output") is True
        and provenance.get("exception_failure_code")
        in {"action_parse_error", "malformed_action_error"}
        and _present(provenance.get("exception_type"))
        and not exception_info
    )
    if not declared_agent_failure_is_exact:
      issues.append("invalid_typed_agent_exception_attribution")
  elif (
      episode_status in {"valid_success", "valid_failure"}
      and typed_exception_fields_present
  ):
    issues.append("orphaned_typed_exception_attribution")

  raw_success = episode.get("is_successful")
  try:
    score = float(raw_success)
  except (TypeError, ValueError, OverflowError):
    score = math.nan
  if not math.isfinite(score):
    issues.append("missing_or_nonfinite_success")
  elif episode_status == "valid_success" and score < 0.5:
    issues.append("episode_status_score_mismatch")
  elif episode_status == "valid_failure" and score >= 0.5:
    issues.append("episode_status_score_mismatch")

  for field in (
      "semantic_task_id",
      "instance_seed",
      "package_name",
      "app_version",
      "code_revision",
  ):
    if not _present(provenance[field]):
      issues.append(f"missing_provenance:{field}")
  if expected_condition in {"breakdown", "c2_g", "c2_o"} and not _present(
      provenance["plan_key"]
  ):
    issues.append("missing_provenance:plan_key")

  row = {
      "model": model,
      "category": category,
      "app_id": app_id,
      "task_template": task_template,
      "instance_id": instance_id,
      "goal": goal,
      "goal_sha256": goal_hash,
      "is_successful": bool(score >= 0.5) if math.isfinite(score) else None,
      "exception_info": exception_info,
      "pkl_path": pkl_path,
      "episode_index": episode_index,
      **{key: _jsonable_scalar(value) for key, value in provenance.items()},
  }
  return slot, row, issues


def _invalid_record(
    slot: SlotKey | None, row: dict[str, Any], issues: Iterable[str]
) -> dict[str, Any]:
  return {
      "slot": list(slot) if slot is not None else None,
      "pkl_path": row.get("pkl_path", ""),
      "episode_index": row.get("episode_index"),
      "issues": sorted(set(issues)),
      "condition": row.get("condition"),
      "exception_info": row.get("exception_info", ""),
  }


def _harvest_items(
    items: Iterable[dict[str, Any]], expected_condition: str
) -> Harvest:
  """Harvest in-memory orchestration fixtures or on-disk episode items."""
  if expected_condition not in {"baseline", "breakdown", "c1", "c2_g", "c2_o"}:
    raise ValueError(
        "expected_condition must be baseline, breakdown, c1, c2_g, or c2_o"
    )

  grouped: dict[SlotKey, list[tuple[dict[str, Any], list[str]]]] = (
      collections.defaultdict(list)
  )
  unkeyed: list[tuple[dict[str, Any], list[str]]] = []
  for item in items:
    slot, row, issues = _normalize_item(item, expected_condition)
    if slot is None:
      unkeyed.append((row, issues))
    else:
      grouped[slot].append((row, issues))

  rows: dict[SlotKey, dict[str, Any]] = {}
  invalid_records: list[dict[str, Any]] = []
  invalid_slots: dict[SlotKey, list[dict[str, Any]]] = {}
  for row, issues in unkeyed:
    invalid_records.append(_invalid_record(None, row, issues))
  for slot, candidates in sorted(grouped.items()):
    if len(candidates) != 1:
      invalid_slots[slot] = []
      for row, issues in candidates:
        invalid = _invalid_record(slot, row, [*issues, "duplicate_experimental_slot"])
        invalid_records.append(invalid)
        invalid_slots[slot].append(invalid)
      continue
    row, issues = candidates[0]
    if issues:
      invalid = _invalid_record(slot, row, issues)
      invalid_records.append(invalid)
      invalid_slots[slot] = [invalid]
    else:
      rows[slot] = row
  return Harvest(rows=rows, invalid_records=invalid_records, invalid_slots=invalid_slots)


def _load_manifest(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError(f"Manifest must contain a JSON object: {path}")
  return payload


def _manifest_roster(payload: dict[str, Any]) -> tuple[set[RosterKey], set[RosterKey]]:
  counts: collections.Counter[RosterKey] = collections.Counter()
  for job in payload.get("jobs", []):
    if not isinstance(job, dict):
      continue
    key = (
        str(job.get("model_name") or ""),
        str(job.get("category") or ""),
        str(job.get("app_id") or ""),
    )
    if all(key):
      counts[key] += 1
  return set(counts), {key for key, count in counts.items() if count > 1}


def _harvest(manifest_path: Path, expected_condition: str) -> Harvest:
  payload = _load_manifest(manifest_path)
  roster, duplicate_roster = _manifest_roster(payload)
  items: list[dict[str, Any]] = []
  external_invalid: list[dict[str, Any]] = []
  for job in payload.get("jobs", []):
    if not isinstance(job, dict):
      continue
    output_path = Path(str(job.get("output_path") or "")).expanduser()
    job_label = [
        str(job.get("model_name") or ""),
        str(job.get("category") or ""),
        str(job.get("app_id") or ""),
    ]
    if not output_path.exists():
      external_invalid.append({
          "slot": None,
          "job": job_label,
          "pkl_path": "",
          "episode_index": None,
          "issues": ["missing_output_path"],
          "condition": None,
          "exception_info": "",
      })
      continue
    pickle_paths = sorted(output_path.rglob("*.pkl.gz"))
    if not pickle_paths:
      external_invalid.append({
          "slot": None,
          "job": job_label,
          "pkl_path": "",
          "episode_index": None,
          "issues": ["no_episode_pickles"],
          "condition": None,
          "exception_info": "",
      })
    for pkl_path in pickle_paths:
      try:
        loaded = _read_pkl_gz(pkl_path)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        external_invalid.append({
            "slot": None,
            "job": job_label,
            "pkl_path": str(pkl_path),
            "episode_index": None,
            "issues": ["unreadable_episode_pickle"],
            "detail": str(exc),
            "condition": None,
            "exception_info": "",
        })
        continue
      episodes = loaded if isinstance(loaded, list) else [loaded]
      for episode_index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
          external_invalid.append({
              "slot": None,
              "job": job_label,
              "pkl_path": str(pkl_path),
              "episode_index": episode_index,
              "issues": ["episode_is_not_mapping"],
              "condition": None,
              "exception_info": "",
          })
          continue
        items.append({
            "episode": episode,
            "job": job,
            "manifest": payload,
            "pkl_path": str(pkl_path),
            "episode_index": episode_index,
        })

  harvest = _harvest_items(items, expected_condition)
  harvest.invalid_records.extend(external_invalid)
  harvest.roster = roster
  harvest.duplicate_roster = duplicate_roster
  return harvest


def _canonical_json(value: Any) -> str:
  return json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  )


def _strict_json_loads(raw: str, source: str) -> Any:
  def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
      if key in value:
        raise SelectedTripletValidationError(
            f"Duplicate JSON key {key!r} in {source}"
        )
      value[key] = item
    return value

  def reject_constant(value: str) -> None:
    raise SelectedTripletValidationError(
        f"Non-finite JSON constant {value!r} in {source}"
    )

  try:
    return json.loads(
        raw,
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )
  except json.JSONDecodeError as exc:
    raise SelectedTripletValidationError(
        f"Invalid JSON in {source}: {exc}"
    ) from exc


def _regular_file_bytes(path: Path) -> bytes:
  if path.is_symlink() or not path.is_file():
    raise SelectedTripletValidationError(
        f"Selection evidence must be a regular non-symlink file: {path}"
    )
  try:
    return path.read_bytes()
  except OSError as exc:
    raise SelectedTripletValidationError(
        f"Unable to read selection evidence {path}: {exc}"
    ) from exc


def _strict_json_file(path: Path) -> dict[str, Any]:
  raw = _regular_file_bytes(path)
  try:
    text = raw.decode("utf-8")
  except UnicodeDecodeError as exc:
    raise SelectedTripletValidationError(
        f"Selection JSON is not UTF-8: {path}"
    ) from exc
  value = _strict_json_loads(text, str(path))
  if not isinstance(value, dict):
    raise SelectedTripletValidationError(
        f"Selection JSON must contain an object: {path}"
    )
  return value


def _strict_jsonl_file(path: Path) -> tuple[list[dict[str, Any]], bytes]:
  raw = _regular_file_bytes(path)
  try:
    text = raw.decode("utf-8")
  except UnicodeDecodeError as exc:
    raise SelectedTripletValidationError(
        f"Selection JSONL is not UTF-8: {path}"
    ) from exc
  if not text or not text.endswith("\n"):
    raise SelectedTripletValidationError(
        f"Selection JSONL must be nonempty and newline-terminated: {path}"
    )
  rows: list[dict[str, Any]] = []
  for line_number, line in enumerate(text.splitlines(), 1):
    if not line:
      raise SelectedTripletValidationError(
          f"Blank JSONL record in {path}:{line_number}"
      )
    value = _strict_json_loads(line, f"{path}:{line_number}")
    if not isinstance(value, dict):
      raise SelectedTripletValidationError(
          f"JSONL record must be an object in {path}:{line_number}"
      )
    rows.append(value)
  canonical = "".join(_canonical_json(row) + "\n" for row in rows).encode(
      "utf-8"
  )
  if raw != canonical:
    raise SelectedTripletValidationError(
        f"Selection JSONL is not in the consumer's canonical encoding: {path}"
    )
  return rows, raw


def _cohort_semantic_keys(
    cohort: Mapping[str, Any],
) -> set[tuple[str, str, str, str, int]]:
  models = cohort.get("models")
  categories = cohort.get("categories")
  combinations = cohort.get("n_task_combinations")
  if (
      not isinstance(models, list)
      or not isinstance(categories, Mapping)
      or isinstance(combinations, bool)
  ):
    raise SelectedTripletValidationError(
        "Primary cohort is missing models/categories/n_task_combinations"
    )
  try:
    n_instances = int(combinations)
  except (TypeError, ValueError, OverflowError) as exc:
    raise SelectedTripletValidationError(
        "Primary cohort n_task_combinations is invalid"
    ) from exc
  expected: set[tuple[str, str, str, str, int]] = set()
  for model in models:
    for category, spec in categories.items():
      if not isinstance(spec, Mapping):
        raise SelectedTripletValidationError(
            f"Invalid primary cohort category: {category!r}"
        )
      for app_id in spec.get("app_ids", []):
        for semantic_task_id in spec.get("semantic_task_ids", []):
          for instance_id in range(n_instances):
            key = (
                str(model),
                str(category),
                str(app_id),
                str(semantic_task_id),
                instance_id,
            )
            if key in expected:
              raise SelectedTripletValidationError(
                  f"Duplicate primary cohort semantic key: {key!r}"
              )
            expected.add(key)
  if not expected:
    raise SelectedTripletValidationError("Primary cohort expands to zero cells")
  return expected


def _paired_key_dict(
    key: tuple[str, str, str, str, int]
) -> dict[str, Any]:
  return {
      "model": key[0],
      "category": key[1],
      "app_id": key[2],
      "semantic_task_id": key[3],
      "instance_id": key[4],
  }


def _expected_pair_id(release_id: str, paired_key: Mapping[str, Any]) -> str:
  digest = hashlib.sha256(
      _canonical_json({
          "release_id": release_id,
          "paired_key": dict(paired_key),
      }).encode("utf-8")
  ).hexdigest()
  return f"pair_{digest[:24]}"


def _expected_attempt_identity(
    release_id: str,
    pair_id: str,
    condition: str,
    round_index: int,
) -> dict[str, Any]:
  round_label = f"r{round_index}"
  slot_id = f"{pair_id}:{condition}"
  if round_index:
    slot_id = f"{slot_id}:replacement:{round_label}"
  return {
      "slot_id": slot_id,
      "attempt_id": f"{pair_id}:{condition}:attempt:{round_label}",
      "attempt_index": round_index,
      "snapshot_family_id": f"{release_id}:snapshot:{pair_id}",
      "snapshot_clone_id": (
          f"{release_id}:snapshot:{pair_id}:{condition}:{round_label}"
      ),
      "is_replacement": bool(round_index),
  }


def _validate_selected_triplet_state(
    selection_rows: Iterable[Mapping[str, Any]],
    journal_rows: Iterable[Mapping[str, Any]],
    consumer_manifest: Mapping[str, Any],
    cohort: Mapping[str, Any],
    cohort_sha256: str,
    *,
    expected_schedule: Iterable[Mapping[str, Any]] | None = None,
    allow_exhausted: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
  """Validate committed round-pure selection using in-memory evidence.

  This is intentionally a pure orchestration validator. It does not create
  tasks, applications, or result artifacts; the on-disk wrapper below loads
  already-produced selected episode artifacts only after this gate passes.
  Strict point-estimate callers leave ``allow_exhausted`` false.  The separate
  attrition-bounds reporter may set it true, in which case a pair is admitted
  without selected outcomes only after all three complete rounds are proven to
  contain an infrastructure-invalid member.
  """
  release_id = str(cohort.get("release_id") or "")
  if release_id != PRIMARY_RELEASE_ID:
    raise SelectedTripletValidationError(
        "Whole-triplet ingestion is reserved for the frozen primary release"
    )
  if list(cohort.get("conditions") or []) != list(PRIMARY_CONDITIONS):
    raise SelectedTripletValidationError(
        "Primary cohort conditions must be exactly c1/c2_g/c2_o"
    )
  if not _SHA256.fullmatch(cohort_sha256):
    raise SelectedTripletValidationError("Primary cohort SHA-256 is invalid")

  expected_keys = _cohort_semantic_keys(cohort)
  expected_by_pair: dict[str, tuple[str, str, str, str, int]] = {}
  for key in expected_keys:
    pair_id = _expected_pair_id(release_id, _paired_key_dict(key))
    if pair_id in expected_by_pair:
      raise SelectedTripletValidationError(f"Pair ID collision: {pair_id}")
    expected_by_pair[pair_id] = key

  manifest_requirements = {
      "schema_version": 1,
      "release_id": release_id,
      "release_purpose": "primary_five_category_analysis",
      "analysis_eligible": True,
      "artifact_role": "primary_analysis_candidate",
      "primary_reporter_acceptance_permitted": True,
      "cohort_sha256": cohort_sha256,
      "episode_slot_count": len(expected_keys) * len(PRIMARY_CONDITIONS),
      "paired_block_count": len(expected_keys),
      "selective_filters": False,
  }
  for field, expected in manifest_requirements.items():
    if consumer_manifest.get(field) != expected:
      raise SelectedTripletValidationError(
          f"Consumer manifest mismatch for {field}: "
          f"expected {expected!r}, got {consumer_manifest.get(field)!r}"
      )
  for field in (
      "schedule_manifest_sha256",
      "ledger_schema_sha256",
      "model_config_sha256",
      "model_endpoint_attestation_sha256",
      "app_pins_sha256",
      "installed_app_attestation_sha256",
      "c2_g_breakdown_sha256",
      "c2_o_breakdown_sha256",
      "base_snapshot_manifest_sha256",
      "snapshot_hook_sha256",
  ):
    if not _SHA256.fullmatch(str(consumer_manifest.get(field) or "")):
      raise SelectedTripletValidationError(
          f"Consumer manifest is missing exact SHA-256 provenance: {field}"
      )
  if not str(consumer_manifest.get("source_revision") or ""):
    raise SelectedTripletValidationError(
        "Consumer manifest is missing source_revision"
    )

  starts: dict[str, dict[str, Any]] = {}
  finishes: dict[str, dict[str, Any]] = {}
  previous_hash = ""
  identity_fields = (
      "release_id",
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
  )
  materialized_journal = [dict(row) for row in journal_rows]
  for sequence, event in enumerate(materialized_journal):
    if event.get("sequence") != sequence:
      raise SelectedTripletValidationError(
          "Attempt journal sequence is not contiguous"
      )
    if event.get("previous_event_sha256") != previous_hash:
      raise SelectedTripletValidationError("Attempt journal hash chain is broken")
    claimed_hash = str(event.get("event_sha256") or "")
    body = dict(event)
    body.pop("event_sha256", None)
    actual_hash = hashlib.sha256(
        _canonical_json(body).encode("utf-8")
    ).hexdigest()
    if claimed_hash != actual_hash:
      raise SelectedTripletValidationError(
          "Attempt journal event SHA-256 mismatch"
      )
    previous_hash = claimed_hash
    attempt_id = str(event.get("attempt_id") or "")
    event_type = event.get("event")
    if event_type == "started":
      if attempt_id in starts:
        raise SelectedTripletValidationError(
            f"Attempt has multiple started events: {attempt_id}"
        )
      starts[attempt_id] = event
    elif event_type == "finished":
      if attempt_id not in starts or attempt_id in finishes:
        raise SelectedTripletValidationError(
            f"Attempt finish is unpaired or duplicated: {attempt_id}"
        )
      if event.get("status") not in {
          "valid_success", "valid_failure", "invalid_infrastructure"
      }:
        raise SelectedTripletValidationError(
            f"Attempt has invalid terminal status: {attempt_id}"
        )
      for field in identity_fields:
        if event.get(field) != starts[attempt_id].get(field):
          raise SelectedTripletValidationError(
              f"Started/finished attempt provenance differs for "
              f"{attempt_id}:{field}"
          )
      finishes[attempt_id] = event
    else:
      raise SelectedTripletValidationError(
          f"Unknown attempt journal event: {event_type!r}"
      )
  if set(starts) != set(finishes):
    unresolved = sorted(set(starts) - set(finishes))
    raise SelectedTripletValidationError(
        f"Attempt journal has unresolved starts: {unresolved[:10]}"
    )
  if len(materialized_journal) % 2:
    raise SelectedTripletValidationError(
        "Completed attempt journal must contain started/finished pairs"
    )
  for index in range(0, len(materialized_journal), 2):
    started_event = materialized_journal[index]
    finished_event = materialized_journal[index + 1]
    if (
        started_event.get("event") != "started"
        or finished_event.get("event") != "finished"
        or started_event.get("attempt_id") != finished_event.get("attempt_id")
    ):
      raise SelectedTripletValidationError(
          "Attempt journal is not a sequential started/finished execution log"
      )

  selections: dict[str, dict[str, Any]] = {}
  selection_order: list[str] = []
  exact_selection_fields = {
      "release_id",
      "pair_id",
      "paired_key",
      "selection_unit",
      "selection_status",
      "selected_round",
      "selected_attempt_ids",
      "all_finished_attempt_ids",
      "selection_basis",
  }
  for raw_selection in selection_rows:
    selection = dict(raw_selection)
    if set(selection) != exact_selection_fields:
      raise SelectedTripletValidationError(
          "Selected-triplet record fields differ from schema v1"
      )
    pair_id = str(selection.get("pair_id") or "")
    if not _PAIR_ID.fullmatch(pair_id) or pair_id in selections:
      raise SelectedTripletValidationError(
          f"Invalid or duplicate selected pair_id: {pair_id!r}"
      )
    selections[pair_id] = selection
    selection_order.append(pair_id)
  if set(selections) != set(expected_by_pair):
    raise SelectedTripletValidationError(
        "Selected-triplet pairs do not equal the frozen primary cohort"
    )
  journal_pair_ids = {str(event.get("pair_id") or "") for event in finishes.values()}
  if journal_pair_ids != set(expected_by_pair):
    raise SelectedTripletValidationError(
        "Journal pair IDs do not equal selected frozen primary pairs"
    )

  schedule_rows = (
      [dict(row) for row in expected_schedule]
      if expected_schedule is not None else None
  )
  schedule_pair_order: list[str] | None = None
  condition_order_by_pair: dict[str, list[str]] = {}
  if schedule_rows is not None:
    if len(schedule_rows) != len(expected_keys) * len(PRIMARY_CONDITIONS):
      raise SelectedTripletValidationError(
          "Recompiled primary schedule has an unexpected slot count"
      )
    schedule_pair_order = []
    schedule_attempt_ids: list[str] = []
    for global_order, row in enumerate(schedule_rows):
      if row.get("global_order") != global_order or row.get("attempt_index") != 0:
        raise SelectedTripletValidationError(
            "Recompiled primary schedule order/round identity is malformed"
        )
      pair_id = str(row.get("pair_id") or "")
      key = expected_by_pair.get(pair_id)
      if key is None or row.get("release_id") != release_id:
        raise SelectedTripletValidationError(
            "Recompiled schedule contains an unknown primary pair"
        )
      for field, expected in _paired_key_dict(key).items():
        if row.get(field) != expected:
          raise SelectedTripletValidationError(
              f"Recompiled schedule paired-key mismatch: {pair_id}:{field}"
          )
      condition = str(row.get("condition") or "")
      expected_identity = _expected_attempt_identity(
          release_id, pair_id, condition, 0
      )
      for field, expected in expected_identity.items():
        if row.get(field) != expected:
          raise SelectedTripletValidationError(
              f"Recompiled schedule attempt mismatch: {pair_id}:{field}"
          )
      if pair_id not in condition_order_by_pair:
        schedule_pair_order.append(pair_id)
        condition_order_by_pair[pair_id] = []
      condition_order_by_pair[pair_id].append(condition)
      schedule_attempt_ids.append(str(row["attempt_id"]))
    if any(
        set(order) != set(PRIMARY_CONDITIONS) or len(order) != 3
        for order in condition_order_by_pair.values()
    ):
      raise SelectedTripletValidationError(
          "Recompiled schedule does not contain exact condition triplets"
      )
    if selection_order != schedule_pair_order:
      raise SelectedTripletValidationError(
          "Selected-triplet records are not in frozen block order"
      )

  selected_by_condition: dict[str, list[dict[str, Any]]] = {
      condition: [] for condition in PRIMARY_CONDITIONS
  }
  selected_round_counts: collections.Counter[int] = collections.Counter()
  exhausted_pair_ids: list[str] = []
  terminal_round_by_pair: dict[str, int] = {}
  invalid_prior_attempts = 0
  for pair_id, key in sorted(expected_by_pair.items()):
    selection = selections[pair_id]
    paired_key = _paired_key_dict(key)
    if selection.get("release_id") != release_id:
      raise SelectedTripletValidationError(
          f"Selection release mismatch for {pair_id}"
      )
    if selection.get("paired_key") != paired_key:
      raise SelectedTripletValidationError(
          f"Selection paired_key mismatch for {pair_id}"
      )
    if selection.get("selection_unit") != "full_condition_triplet":
      raise SelectedTripletValidationError(
          f"Selection unit is not a whole triplet for {pair_id}"
      )
    selection_status = selection.get("selection_status")
    if selection_status not in {
        "selected_complete_triplet", "exhausted_invalid"
    }:
      raise SelectedTripletValidationError(
          f"Primary pair is not reportable: {pair_id} has "
          f"{selection_status!r}"
      )
    if selection_status == "exhausted_invalid" and not allow_exhausted:
      raise SelectedTripletValidationError(
          f"Primary pair is not reportable: {pair_id} has "
          "'exhausted_invalid'; use the dedicated attrition-bounds path"
      )
    if selection.get("selection_basis") != (
        "first complete round with no invalid_infrastructure member; "
        "no cross-round condition mixing"
    ):
      raise SelectedTripletValidationError(
          f"Selection basis mismatch for {pair_id}"
      )
    selected_round = selection.get("selected_round")
    if selection_status == "selected_complete_triplet":
      if (
          isinstance(selected_round, bool)
          or not isinstance(selected_round, int)
          or selected_round not in (0, 1, 2)
      ):
        raise SelectedTripletValidationError(
            f"Invalid selected round for {pair_id}: {selected_round!r}"
        )
      final_round = selected_round
      selected_round_counts[selected_round] += 1
    else:
      if selected_round is not None or selection.get("selected_attempt_ids") != {}:
        raise SelectedTripletValidationError(
            f"Exhausted pair has selected outcomes: {pair_id}"
        )
      final_round = 2
      exhausted_pair_ids.append(pair_id)
    terminal_round_by_pair[pair_id] = final_round

    pair_finishes = {
        attempt_id: event
        for attempt_id, event in finishes.items()
        if event.get("pair_id") == pair_id
    }
    expected_finished_ids: set[str] = set()
    for round_index in range(final_round + 1):
      round_events: list[dict[str, Any]] = []
      for condition in PRIMARY_CONDITIONS:
        expected_identity = _expected_attempt_identity(
            release_id, pair_id, condition, round_index
        )
        attempt_id = str(expected_identity["attempt_id"])
        expected_finished_ids.add(attempt_id)
        event = pair_finishes.get(attempt_id)
        if event is None:
          raise SelectedTripletValidationError(
              f"Incomplete condition triplet for {pair_id} round {round_index}"
          )
        expected_provenance = {
            "release_id": release_id,
            "cohort_sha256": cohort_sha256,
            "pair_id": pair_id,
            "model": key[0],
            "category": key[1],
            "app_id": key[2],
            "semantic_task_id": key[3],
            "instance_id": key[4],
            "condition": condition,
            **expected_identity,
        }
        for field, expected in expected_provenance.items():
          if event.get(field) != expected:
            raise SelectedTripletValidationError(
                f"Attempt provenance mismatch for {attempt_id}:{field}"
            )
        round_events.append(event)
      statuses = [event["status"] for event in round_events]
      is_nonselected_round = (
          selection_status == "exhausted_invalid"
          or round_index < final_round
      )
      if is_nonselected_round:
        if "invalid_infrastructure" not in statuses:
          reason = (
              "Replacement round lacks infrastructure trigger"
              if selection_status == "selected_complete_triplet"
              else "Exhausted round lacks infrastructure invalidation"
          )
          raise SelectedTripletValidationError(
              f"{reason} for {pair_id}"
          )
        invalid_prior_attempts += statuses.count("invalid_infrastructure")
      elif any(status == "invalid_infrastructure" for status in statuses):
        raise SelectedTripletValidationError(
            f"Selected round contains infrastructure-invalid member: {pair_id}"
        )

    if set(pair_finishes) != expected_finished_ids:
      raise SelectedTripletValidationError(
          f"Attempt history extends beyond or skips the terminal round: {pair_id}"
      )
    all_finished = selection.get("all_finished_attempt_ids")
    if (
        not isinstance(all_finished, list)
        or len(all_finished) != len(set(all_finished))
        or set(all_finished) != expected_finished_ids
    ):
      raise SelectedTripletValidationError(
          f"all_finished_attempt_ids mismatch for {pair_id}"
      )
    if selection_status == "exhausted_invalid":
      continue
    selected_attempt_ids = selection.get("selected_attempt_ids")
    expected_selected_ids = {
        condition: _expected_attempt_identity(
            release_id, pair_id, condition, selected_round
        )["attempt_id"]
        for condition in PRIMARY_CONDITIONS
    }
    if selected_attempt_ids != expected_selected_ids:
      raise SelectedTripletValidationError(
          f"Selected condition attempt IDs mismatch for {pair_id}"
      )
    for condition, attempt_id in expected_selected_ids.items():
      selected_by_condition[condition].append({
          "finish": pair_finishes[str(attempt_id)],
          "selection": selection,
          "paired_key": paired_key,
      })

  audit = {
      "mode": "committed_whole_triplet_selection",
      "release_id": release_id,
      "scheduled_pairs": len(expected_by_pair),
      "selected_pairs": sum(selected_round_counts.values()),
      "exhausted_invalid_pairs": len(exhausted_pair_ids),
      "exhausted_pair_ids": sorted(exhausted_pair_ids),
      "selected_attempts_per_condition": sum(selected_round_counts.values()),
      "journal_finished_attempts": len(finishes),
      "infrastructure_invalid_prior_attempts": invalid_prior_attempts,
      "pairs_selected_from_replacement_round": sum(
          count for round_index, count in selected_round_counts.items()
          if round_index > 0
      ),
      "selected_round_counts": {
          str(round_index): selected_round_counts[round_index]
          for round_index in (0, 1, 2)
      },
      "round_pure": True,
      "primary_point_estimate_permitted": not exhausted_pair_ids,
      "attrition_bounds_required": bool(exhausted_pair_ids),
  }
  if schedule_pair_order is not None:
    expected_start_order: list[str] = []
    for round_index in (0, 1, 2):
      for pair_id in schedule_pair_order:
        terminal_round = terminal_round_by_pair[pair_id]
        if terminal_round < round_index:
          continue
        for condition in condition_order_by_pair[pair_id]:
          expected_start_order.append(str(_expected_attempt_identity(
              release_id, pair_id, condition, round_index
          )["attempt_id"]))
    actual_start_order = [
        str(event["attempt_id"])
        for event in materialized_journal
        if event.get("event") == "started"
    ]
    if actual_start_order != expected_start_order:
      raise SelectedTripletValidationError(
          "Attempt journal violates frozen block order or full replacement epochs"
      )
    audit["frozen_schedule_order_validated"] = True
  return selected_by_condition, audit


def _require_child_file(path_value: Any, root: Path, label: str) -> Path:
  path = Path(str(path_value or ""))
  if not path.is_absolute():
    raise SelectedTripletValidationError(f"{label} path is not absolute: {path}")
  try:
    resolved = path.resolve(strict=True)
    root_resolved = root.resolve(strict=True)
  except OSError as exc:
    raise SelectedTripletValidationError(
        f"Unable to resolve {label} path: {path}"
    ) from exc
  if root_resolved not in resolved.parents:
    raise SelectedTripletValidationError(
        f"{label} path escapes the consumer state root: {path}"
    )
  if path.is_symlink() or not resolved.is_file():
    raise SelectedTripletValidationError(
        f"{label} must be a regular non-symlink file: {path}"
    )
  return resolved


def _harvest_selected_triplets(
    selected_triplets_path: Path,
    cohort_path: Path,
    *,
    allow_exhausted: bool = False,
) -> tuple[dict[str, Harvest], dict[str, Any]]:
  """Load committed primary selections with fail-closed state validation.

  ``allow_exhausted`` is reserved for the attrition-bounds reporter.  Ordinary
  primary point-estimate callers must keep the default and therefore reject
  any triplet without a complete selected round.
  """
  selection_input = selected_triplets_path.expanduser()
  if selection_input.is_symlink():
    raise SelectedTripletValidationError(
        f"Primary selection input must not be a symlink: {selection_input}"
    )
  selection_path = selection_input.absolute()
  if selection_path.name != "selected_triplets.jsonl":
    raise SelectedTripletValidationError(
        "Primary selection input must be named selected_triplets.jsonl"
    )
  state_root = selection_path.parent
  cohort_input = cohort_path.expanduser()
  if cohort_input.is_symlink():
    raise SelectedTripletValidationError(
        f"Primary cohort input must not be a symlink: {cohort_input}"
    )
  cohort_path = cohort_input.absolute()
  cohort_bytes = _regular_file_bytes(cohort_path)
  cohort = _strict_json_loads(cohort_bytes.decode("utf-8"), str(cohort_path))
  if not isinstance(cohort, dict):
    raise SelectedTripletValidationError("Primary cohort must be a JSON object")
  cohort_sha256 = hashlib.sha256(cohort_bytes).hexdigest()
  try:
    compiled_cohort, compiled_cohort_sha256 = (
        schedule_builder.load_primary_cohort(cohort_path)
    )
    expected_schedule, _, expected_schedule_manifest = (
        schedule_builder.compile_frozen_schedule(
            compiled_cohort, compiled_cohort_sha256
        )
    )
  except (schedule_builder.ScheduleBuildError, OSError, ValueError) as exc:
    raise SelectedTripletValidationError(
        f"Primary cohort cannot reproduce the exact frozen schedule: {exc}"
    ) from exc
  if compiled_cohort != cohort or compiled_cohort_sha256 != cohort_sha256:
    raise SelectedTripletValidationError(
        "Strict primary cohort parse differs from schedule recompilation"
    )

  selection_rows, selection_bytes = _strict_jsonl_file(selection_path)
  journal_path = state_root / "attempt_journal.jsonl"
  runtime_path = state_root / "replacement_ledger_runtime.jsonl"
  commit_path = state_root / "state_commit.json"
  consumer_manifest_path = state_root / "consumer_manifest.json"
  journal_rows, journal_bytes = _strict_jsonl_file(journal_path)
  runtime_bytes = _regular_file_bytes(runtime_path)
  commit = _strict_json_file(commit_path)
  consumer_manifest = _strict_json_file(consumer_manifest_path)
  expected_schedule_manifest_bytes = (
      json.dumps(
          expected_schedule_manifest,
          indent=2,
          sort_keys=True,
          ensure_ascii=False,
      )
      + "\n"
  ).encode("utf-8")
  if consumer_manifest.get("schedule_manifest_sha256") != hashlib.sha256(
      expected_schedule_manifest_bytes
  ).hexdigest():
    raise SelectedTripletValidationError(
        "Consumer manifest does not attest the recompiled primary schedule"
    )
  expected_ledger_schema_bytes = (
      json.dumps(
          schedule_builder.replacement_ledger_schema(),
          indent=2,
          sort_keys=True,
          ensure_ascii=False,
      )
      + "\n"
  ).encode("utf-8")
  if consumer_manifest.get("ledger_schema_sha256") != hashlib.sha256(
      expected_ledger_schema_bytes
  ).hexdigest():
    raise SelectedTripletValidationError(
        "Consumer manifest does not attest the exact replacement schema"
    )
  if commit.get("schema_version") != 1:
    raise SelectedTripletValidationError("Unsupported state commit schema")
  expected_commit = {
      "journal_event_count": len(journal_rows),
      "journal_head_sha256": (
          str(journal_rows[-1].get("event_sha256") or "")
          if journal_rows else ""
      ),
      "runtime_ledger_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
      "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
  }
  for field, expected in expected_commit.items():
    if commit.get(field) != expected:
      raise SelectedTripletValidationError(
          f"Committed selection state mismatch for {field}"
      )

  selected, audit = _validate_selected_triplet_state(
      selection_rows,
      journal_rows,
      consumer_manifest,
      cohort,
      cohort_sha256,
      expected_schedule=expected_schedule,
      allow_exhausted=allow_exhausted,
  )
  items: dict[str, list[dict[str, Any]]] = {
      condition: [] for condition in PRIMARY_CONDITIONS
  }
  contract_sha256s: list[str] = []
  for condition in PRIMARY_CONDITIONS:
    for selected_attempt in selected[condition]:
      finish = selected_attempt["finish"]
      attempt_id = str(finish["attempt_id"])
      safe_attempt_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", attempt_id)
      attempt_root = state_root / "attempts" / safe_attempt_id
      contract_path = _require_child_file(
          finish.get("result_contract_path"), state_root, "result contract"
      )
      if contract_path != (attempt_root / "result_contract.json").resolve():
        raise SelectedTripletValidationError(
            f"Result contract is outside its exact attempt directory: {attempt_id}"
        )
      contract = _strict_json_file(contract_path)
      contract_sha256s.append(hashlib.sha256(
          _regular_file_bytes(contract_path)
      ).hexdigest())
      if contract.get("schema_version") != 1:
        raise SelectedTripletValidationError(
            f"Unsupported result contract schema: {attempt_id}"
        )
      for field in (
          "release_id", "cohort_sha256", "pair_id", "slot_id", "attempt_id",
          "attempt_index", "snapshot_family_id", "snapshot_clone_id", "model",
          "category", "app_id", "semantic_task_id", "instance_id", "condition",
          "is_replacement", "status", "artifact_path", "artifact_sha256",
          "snapshot_prepare_receipt", "snapshot_release_receipt",
      ):
        if contract.get(field) != finish.get(field):
          raise SelectedTripletValidationError(
              f"Journal/result-contract mismatch for {attempt_id}:{field}"
          )
      if contract.get("reason_code") != "episode_contract_validated":
        raise SelectedTripletValidationError(
            f"Selected attempt lacks validated episode contract: {attempt_id}"
        )
      success = contract.get("is_successful")
      expected_success = 1.0 if contract["status"] == "valid_success" else 0.0
      if (
          isinstance(success, bool)
          or not isinstance(success, (int, float))
          or float(success) != expected_success
      ):
        raise SelectedTripletValidationError(
            f"Selected contract has a non-binary or status-inconsistent score: "
            f"{attempt_id}"
        )
      for receipt_field, expected_name in (
          ("snapshot_prepare_receipt", "snapshot_clone_activate_receipt.json"),
          ("snapshot_release_receipt", "snapshot_release_receipt.json"),
      ):
        receipt_path = _require_child_file(
            contract.get(receipt_field), state_root, receipt_field
        )
        if receipt_path != (attempt_root / expected_name).resolve():
          raise SelectedTripletValidationError(
              f"{receipt_field} is outside its exact attempt directory: "
              f"{attempt_id}"
          )
      artifact_path = _require_child_file(
          finish.get("artifact_path"), state_root, "selected episode artifact"
      )
      checkpoint_root = (attempt_root / "checkpoint").resolve()
      if checkpoint_root not in artifact_path.parents:
        raise SelectedTripletValidationError(
            f"Selected artifact is outside its exact checkpoint: {attempt_id}"
        )
      artifact_sha256 = hashlib.sha256(
          _regular_file_bytes(artifact_path)
      ).hexdigest()
      if artifact_sha256 != finish.get("artifact_sha256"):
        raise SelectedTripletValidationError(
            f"Selected artifact SHA-256 mismatch: {attempt_id}"
        )
      try:
        payload = _read_pkl_gz(artifact_path)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        raise SelectedTripletValidationError(
            f"Unable to read selected episode artifact {artifact_path}: {exc}"
        ) from exc
      if (
          not isinstance(payload, list)
          or len(payload) != 1
          or not isinstance(payload[0], dict)
      ):
        raise SelectedTripletValidationError(
            f"Selected checkpoint is not exactly one episode: {attempt_id}"
        )
      episode = payload[0]
      expected_artifact_name = (
          f"{episode.get('task_template')}_{finish['instance_id']}.pkl.gz"
      )
      if artifact_path.name != expected_artifact_name:
        raise SelectedTripletValidationError(
            f"Selected checkpoint filename mismatch: {attempt_id}"
        )
      checkpoint_files = sorted(
          path.resolve() for path in checkpoint_root.glob("*.pkl.gz")
      )
      if checkpoint_files != [artifact_path]:
        raise SelectedTripletValidationError(
            f"Selected checkpoint directory contains an unexpected file set: "
            f"{attempt_id}"
        )
      required_episode = {
          "release_id": finish["release_id"],
          "cohort_sha256": finish["cohort_sha256"],
          "pair_id": finish["pair_id"],
          "slot_id": finish["slot_id"],
          "attempt_id": finish["attempt_id"],
          "attempt_index": finish["attempt_index"],
          "snapshot_family_id": finish["snapshot_family_id"],
          "snapshot_clone_id": finish["snapshot_clone_id"],
          "model_name": finish["model"],
          "app_id": finish["app_id"],
          "semantic_task_id": finish["semantic_task_id"],
          "instance_id": finish["instance_id"],
          "catbench_condition": finish["condition"],
          "catbench_condition_config_valid": True,
          "catbench_episode_status": finish["status"],
          "code_revision": consumer_manifest["source_revision"],
          "schedule_manifest_sha256": consumer_manifest[
              "schedule_manifest_sha256"
          ],
          "model_config_sha256": consumer_manifest["model_config_sha256"],
          "model_endpoint_attestation_sha256": consumer_manifest[
              "model_endpoint_attestation_sha256"
          ],
          "app_pins_sha256": consumer_manifest["app_pins_sha256"],
          "installed_app_attestation_sha256": consumer_manifest[
              "installed_app_attestation_sha256"
          ],
          "task_random_seed": cohort["task_random_seed"],
          "n_task_combinations": cohort["n_task_combinations"],
          "schedule_seed": cohort["schedule_seed"],
          "plan_file_sha256": (
              ""
              if condition == "c1"
              else consumer_manifest[f"{condition}_breakdown_sha256"]
          ),
      }
      for field, expected in required_episode.items():
        if episode.get(field) != expected:
          raise SelectedTripletValidationError(
              f"Episode/selection provenance mismatch for {attempt_id}:{field}"
          )
      if episode.get("is_successful") != contract.get("is_successful"):
        raise SelectedTripletValidationError(
            f"Episode/result-contract score mismatch: {attempt_id}"
        )
      item = {
          "episode": episode,
          "job": {
              "model_name": finish["model"],
              "category": finish["category"],
              "app_id": finish["app_id"],
              "package_name": episode.get("package_name"),
              "app_version": episode.get("app_version"),
              "app_version_code": episode.get("app_version_code"),
              "apk_sha256": episode.get("apk_sha256"),
          },
          "manifest": {
              **consumer_manifest,
              "code_revision": consumer_manifest.get("source_revision"),
          },
          "pkl_path": str(artifact_path),
          "episode_index": 0,
      }
      items[condition].append(item)

  roster = {
      (key[0], key[1], key[2]) for key in _cohort_semantic_keys(cohort)
  }
  harvests: dict[str, Harvest] = {}
  for condition in PRIMARY_CONDITIONS:
    harvest = _harvest_items(items[condition], condition)
    if harvest.invalid_records:
      raise SelectedTripletValidationError(
          f"Selected {condition} episodes failed strict harvest: "
          f"{harvest.invalid_records[:3]}"
      )
    harvest.roster = set(roster)
    harvest.selection_audit = {**audit, "condition": condition}
    harvests[condition] = harvest
  audit = {
      **audit,
      "selection_path": str(selection_path),
      "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
      "journal_sha256": hashlib.sha256(journal_bytes).hexdigest(),
      "runtime_ledger_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
      "selected_result_contract_set_sha256": hashlib.sha256(
          "\n".join(sorted(contract_sha256s)).encode("utf-8")
      ).hexdigest(),
  }
  for condition in PRIMARY_CONDITIONS:
    harvests[condition].selection_audit = {**audit, "condition": condition}
  return harvests, audit


def _label_from_manifest(path: Path, fallback_index: int) -> str:
  try:
    payload = _load_manifest(path)
  except Exception:  # pylint: disable=broad-exception-caught
    return f"breakdown_{fallback_index}"
  args = _mapping(payload.get("matrix_args"))
  breakdown_file = args.get("CATBENCH_TASK_BREAKDOWN_FILE") or ""
  return Path(str(breakdown_file)).stem if breakdown_file else path.stem


def _pair_mismatch_reasons(
    baseline_row: dict[str, Any], treatment_row: dict[str, Any]
) -> list[str]:
  reasons: list[str] = []
  # Required fields are checked during harvest.  Equality here prevents two
  # individually valid but scientifically different trials from being paired.
  for field in (
      "goal_sha256",
      "semantic_goal_sha256",
      "semantic_parameter_sha256",
      "instance_seed",
      "semantic_task_id",
      "package_name",
      "app_version",
      "code_revision",
      "release_id",
      "cohort_sha256",
      "schedule_manifest_sha256",
      "model_revision",
      "runner_config_sha256",
      "model_config_sha256",
      "model_endpoint_attestation_sha256",
      "app_pins_sha256",
      "installed_app_attestation_sha256",
      "pair_id",
      "attempt_index",
      "snapshot_family_id",
      "schedule_seed",
      "n_task_combinations",
  ):
    if baseline_row.get(field) != treatment_row.get(field):
      reasons.append(f"mismatched:{field}")
  # These fields are compared whenever either run recorded them; an omitted
  # counterpart is itself a provenance mismatch.
  for field in ("task_random_seed", "app_version_code", "apk_sha256"):
    if _present(baseline_row.get(field)) or _present(treatment_row.get(field)):
      if baseline_row.get(field) != treatment_row.get(field):
        reasons.append(f"mismatched_or_missing:{field}")
  if treatment_row.get("condition") not in {"breakdown", "c2_g", "c2_o"}:
    reasons.append("treatment_not_plan_assistance")
  if baseline_row.get("condition") not in {"baseline", "c1"}:
    reasons.append("baseline_not_direct_execution")
  if not _present(treatment_row.get("plan_key")):
    reasons.append("treatment_missing_plan_key")
  return reasons


def _mcnemar_exact_p(planning_responsive: int, treatment_regressions: int) -> str:
  """Two-sided exact binomial McNemar p-value, without SciPy.

  A decimal string is returned so very small exact-test probabilities do not
  silently underflow to floating-point zero.
  """
  discordant = planning_responsive + treatment_regressions
  if discordant == 0:
    return "1"
  tail = min(planning_responsive, treatment_regressions)
  numerator = 2 * sum(math.comb(discordant, k) for k in range(tail + 1))
  denominator = 2 ** discordant
  if numerator >= denominator:
    return "1"
  with decimal.localcontext() as context:
    context.prec = 17
    value = decimal.Decimal(numerator) / decimal.Decimal(denominator)
  return format(value, ".16g")


def _paired_metrics(
    baseline: dict[SlotKey, dict[str, Any]],
    treatment: dict[SlotKey, dict[str, Any]],
) -> dict[str, Any]:
  shared_candidates = sorted(baseline.keys() & treatment.keys())
  baseline_only = sorted(baseline.keys() - treatment.keys())
  treatment_only = sorted(treatment.keys() - baseline.keys())
  mismatches: list[dict[str, Any]] = []
  paired: list[SlotKey] = []
  for slot in shared_candidates:
    reasons = _pair_mismatch_reasons(baseline[slot], treatment[slot])
    if reasons:
      mismatches.append({
          "slot": list(slot),
          "reasons": reasons,
          "baseline_goal_sha256": baseline[slot].get("goal_sha256"),
          "treatment_goal_sha256": treatment[slot].get("goal_sha256"),
          "baseline_instance_seed": baseline[slot].get("instance_seed"),
          "treatment_instance_seed": treatment[slot].get("instance_seed"),
      })
    else:
      paired.append(slot)

  planning_responsive = 0
  treatment_regressions = 0
  residual_under_plan_assistance = 0
  both_successful = 0
  baseline_correct = 0
  treatment_correct = 0
  planning_responsive_keys: list[SlotKey] = []
  treatment_regression_keys: list[SlotKey] = []
  residual_keys: list[SlotKey] = []
  for slot in paired:
    baseline_success = bool(baseline[slot]["is_successful"])
    treatment_success = bool(treatment[slot]["is_successful"])
    baseline_correct += int(baseline_success)
    treatment_correct += int(treatment_success)
    if not baseline_success and treatment_success:
      planning_responsive += 1
      planning_responsive_keys.append(slot)
    elif baseline_success and not treatment_success:
      treatment_regressions += 1
      treatment_regression_keys.append(slot)
    elif not baseline_success and not treatment_success:
      residual_under_plan_assistance += 1
      residual_keys.append(slot)
    else:
      both_successful += 1

  n_paired = len(paired)
  return {
      "n_shared_candidates": len(shared_candidates),
      "n_paired": n_paired,
      "n_pair_provenance_mismatches": len(mismatches),
      "pair_provenance_mismatches": mismatches,
      "baseline_only": len(baseline_only),
      "treatment_only": len(treatment_only),
      "baseline_sr": baseline_correct / n_paired if n_paired else None,
      "treatment_sr": treatment_correct / n_paired if n_paired else None,
      "delta_sr": (
          (treatment_correct - baseline_correct) / n_paired if n_paired else None
      ),
      "planning_responsive": planning_responsive,
      "treatment_regressions": treatment_regressions,
      "residual_under_plan_assistance": residual_under_plan_assistance,
      "both_successful": both_successful,
      "mcnemar_exact_p_two_sided": _mcnemar_exact_p(
          planning_responsive, treatment_regressions
      ),
      "baseline_only_keys": [list(key) for key in baseline_only],
      "treatment_only_keys": [list(key) for key in treatment_only],
      "planning_responsive_keys": [list(key) for key in planning_responsive_keys],
      "treatment_regression_keys": [list(key) for key in treatment_regression_keys],
      "residual_under_plan_assistance_keys": [list(key) for key in residual_keys],
      "paired_keys": [list(key) for key in paired],
  }


def _group_by(
    paired_keys: list[SlotKey],
    baseline: dict[SlotKey, dict[str, Any]],
    treatment: dict[SlotKey, dict[str, Any]],
    by: int,
) -> dict[str, dict[str, Any]]:
  buckets: dict[str, list[SlotKey]] = collections.defaultdict(list)
  for key in paired_keys:
    buckets[key[by]].append(key)
  return {
      bucket: _paired_metrics(
          {key: baseline[key] for key in keys},
          {key: treatment[key] for key in keys},
      )
      for bucket, keys in buckets.items()
  }


def _roster_validation(baseline: Harvest, treatment: Harvest) -> dict[str, Any]:
  baseline_only = sorted(baseline.roster - treatment.roster)
  treatment_only = sorted(treatment.roster - baseline.roster)
  valid = not (
      baseline_only
      or treatment_only
      or baseline.duplicate_roster
      or treatment.duplicate_roster
  )
  return {
      "valid": valid,
      "baseline_only_roster": [list(key) for key in baseline_only],
      "treatment_only_roster": [list(key) for key in treatment_only],
      "baseline_duplicate_roster": [
          list(key) for key in sorted(baseline.duplicate_roster)
      ],
      "treatment_duplicate_roster": [
          list(key) for key in sorted(treatment.duplicate_roster)
      ],
  }


def _primary_harvest_validation(
    harvest: Harvest,
    cohort: dict[str, Any],
    expected_condition: str,
) -> dict[str, Any]:
  """Validates one condition against the full frozen primary Cartesian set."""
  expected: set[tuple[str, str, str, str, int]] = set()
  for model in cohort["models"]:
    for category, spec in cohort["categories"].items():
      for app_id in spec["app_ids"]:
        for semantic_task_id in spec["semantic_task_ids"]:
          for instance_id in range(int(cohort["n_task_combinations"])):
            expected.add((model, category, app_id, semantic_task_id, instance_id))
  actual = {
      (
          str(row.get("model")),
          str(row.get("category")),
          str(row.get("app_id")),
          str(row.get("semantic_task_id")),
          int(row.get("instance_id")),
      )
      for row in harvest.rows.values()
      if row.get("instance_id") is not None
  }
  missing = sorted(expected - actual)
  extra = sorted(actual - expected)
  provenance_issues: list[dict[str, Any]] = []
  required_provenance = (
      "semantic_goal_sha256",
      "semantic_parameter_sha256",
      "instance_seed",
      "package_name",
      "app_version",
      "app_version_code",
      "apk_sha256",
      "code_revision",
      "release_id",
      "model_revision",
      "runner_config_sha256",
      "model_config_sha256",
      "cohort_sha256",
      "schedule_manifest_sha256",
      "pair_id",
      "slot_id",
      "attempt_id",
      "attempt_index",
      "snapshot_family_id",
      "snapshot_clone_id",
      "model_endpoint_attestation_sha256",
      "app_pins_sha256",
      "installed_app_attestation_sha256",
      "schedule_seed",
      "n_task_combinations",
  )
  for slot, row in sorted(harvest.rows.items()):
    reasons = [
        f"missing:{field}"
        for field in required_provenance
        if not _present(row.get(field))
    ]
    if row.get("condition") != expected_condition:
      reasons.append("condition_mismatch")
    if row.get("condition_config_valid") is not True:
      reasons.append("condition_configuration_not_explicitly_valid")
    if row.get("release_id") != cohort.get("release_id"):
      reasons.append("release_id_mismatch")
    if reasons:
      provenance_issues.append({"slot": list(slot), "reasons": reasons})
  return {
      "valid": not missing and not extra and not provenance_issues,
      "expected_rows": len(expected),
      "actual_rows": len(actual),
      "missing_count": len(missing),
      "extra_count": len(extra),
      "missing": [list(key) for key in missing],
      "extra": [list(key) for key in extra],
      "provenance_issues": provenance_issues,
  }


def _plan_reuse_validation(treatment: Harvest) -> dict[str, Any]:
  """Verify one plan identity per semantic task instance across target apps."""
  grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
  for row in treatment.rows.values():
    key = (
        row["category"],
        row["semantic_task_id"],
        row["instance_id"],
    )
    grouped[key].append(row)

  issues: list[dict[str, Any]] = []
  for key, rows in sorted(
      grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
  ):
    if len({row["app_id"] for row in rows}) < 2:
      continue
    plan_keys = {row.get("plan_key") for row in rows}
    instance_seeds = {row.get("instance_seed") for row in rows}
    task_random_seeds = {row.get("task_random_seed") for row in rows}
    semantic_goal_hashes = {
        row.get("semantic_goal_sha256")
        for row in rows
        if _present(row.get("semantic_goal_sha256"))
    }
    plan_hashes = {
        row.get("plan_sha256")
        for row in rows
        if _present(row.get("plan_sha256"))
    }
    reasons: list[str] = []
    if len(plan_keys) != 1:
      reasons.append("different_plan_keys_across_apps")
    if len(instance_seeds) != 1:
      reasons.append("different_instance_seeds_across_apps")
    if len(task_random_seeds) != 1:
      reasons.append("different_task_random_seeds_across_apps")
    if len(semantic_goal_hashes) > 1:
      reasons.append("different_semantic_goal_hashes_across_apps")
    if len(plan_hashes) > 1:
      reasons.append("different_plan_hashes_across_apps")
    if reasons:
      issues.append({
          "semantic_instance": list(key),
          "apps": sorted(row["app_id"] for row in rows),
          "reasons": reasons,
      })
  return {"valid": not issues, "issues": issues}


def _interpretation(baseline_success: bool, treatment_success: bool) -> str:
  if not baseline_success and treatment_success:
    return "planning_responsive"
  if baseline_success and not treatment_success:
    return "treatment_regression"
  if not baseline_success and not treatment_success:
    return "residual_under_plan_assistance"
  return "both_successful"


def _per_task_rows(
    label: str,
    baseline: Harvest,
    treatment: Harvest,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  all_slots = sorted(baseline.rows.keys() | treatment.rows.keys())
  for slot in all_slots:
    baseline_row = baseline.rows.get(slot)
    treatment_row = treatment.rows.get(slot)
    if baseline_row is None:
      status = "treatment_only"
      reasons: list[str] = []
    elif treatment_row is None:
      status = "baseline_only"
      reasons = []
    else:
      reasons = _pair_mismatch_reasons(baseline_row, treatment_row)
      status = "provenance_mismatch" if reasons else "paired"
    record: dict[str, Any] = {
        "treatment_label": label,
        "slot": list(slot),
        "status": status,
        "mismatch_reasons": reasons,
        "baseline_success": baseline_row.get("is_successful") if baseline_row else None,
        "treatment_success": treatment_row.get("is_successful") if treatment_row else None,
        "baseline_provenance": baseline_row,
        "treatment_provenance": treatment_row,
    }
    if status == "paired":
      record["interpretation"] = _interpretation(
          bool(baseline_row["is_successful"]),
          bool(treatment_row["is_successful"]),
      )
    rows.append(record)
  return rows


def _fmt_rate(value: float | None) -> str:
  return "NA" if value is None else f"{value:.3f}"


def _write_markdown(out_path: Path, report: dict[str, Any]) -> None:
  lines = [
      "# Baseline vs. plan-assistance paired summary",
      "",
      "C1-fail/C2-pass is reported as **planning-responsive**. Failure in both ",
      "conditions is **residual under plan assistance**, not a grounding label.",
      "",
  ]
  for label, run in report["runs"].items():
    overall = run["overall"]
    baseline_audit = run["baseline_harvest"]
    treatment_audit = run["treatment_harvest"]
    lines.extend([
        f"## Treatment: `{label}`",
        "",
        f"- Strict validity: **{'PASS' if run['strictly_valid'] else 'FAIL'}**",
        f"- Eligible paired: {overall['n_paired']}; provenance mismatches: "
        f"{overall['n_pair_provenance_mismatches']}",
        f"- Missing counterparts: baseline-only {overall['baseline_only']}; "
        f"treatment-only {overall['treatment_only']}",
        f"- Invalid episodes/artifacts: baseline "
        f"{baseline_audit['n_invalid_records']}; treatment "
        f"{treatment_audit['n_invalid_records']}",
        f"- Baseline SR: {_fmt_rate(overall['baseline_sr'])}; treatment SR: "
        f"{_fmt_rate(overall['treatment_sr'])}; delta: "
        f"{_fmt_rate(overall['delta_sr'])}",
        f"- Planning-responsive: {overall['planning_responsive']}; treatment "
        f"regressions: {overall['treatment_regressions']}; residual under plan "
        f"assistance: {overall['residual_under_plan_assistance']}; both "
        f"successful: {overall['both_successful']}",
        f"- Exact two-sided McNemar p: "
        f"{overall['mcnemar_exact_p_two_sided']}",
        "",
        "### By model",
        "",
        "| Model | n | base SR | treatment SR | delta | plan-responsive | "
        "regressions | residual | exact p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model in sorted(run["by_model"]):
      metrics = run["by_model"][model]
      lines.append(
          f"| {model} | {metrics['n_paired']} | "
          f"{_fmt_rate(metrics['baseline_sr'])} | "
          f"{_fmt_rate(metrics['treatment_sr'])} | "
          f"{_fmt_rate(metrics['delta_sr'])} | "
          f"{metrics['planning_responsive']} | "
          f"{metrics['treatment_regressions']} | "
          f"{metrics['residual_under_plan_assistance']} | "
          f"{metrics['mcnemar_exact_p_two_sided']} |"
      )
    lines.extend([
        "",
        "### By category",
        "",
        "| Category | n | base SR | treatment SR | delta | plan-responsive | "
        "regressions | residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for category in sorted(run["by_category"]):
      metrics = run["by_category"][category]
      lines.append(
          f"| {category} | {metrics['n_paired']} | "
          f"{_fmt_rate(metrics['baseline_sr'])} | "
          f"{_fmt_rate(metrics['treatment_sr'])} | "
          f"{_fmt_rate(metrics['delta_sr'])} | "
          f"{metrics['planning_responsive']} | "
          f"{metrics['treatment_regressions']} | "
          f"{metrics['residual_under_plan_assistance']} |"
      )
    lines.append("")
  out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--baseline_manifest")
  parser.add_argument(
      "--baseline_condition",
      default=None,
      choices=("baseline", "c1"),
      help=(
          "Explicit condition value required in the direct-execution run; "
          "defaults to legacy 'baseline' outside primary mode."
      ),
  )
  parser.add_argument(
      "--breakdown_manifest",
      action="append",
      default=[],
      help="Manifest of a breakdown run; repeat for multiple generators.",
  )
  parser.add_argument(
      "--label", action="append", default=[],
      help="Optional label per --breakdown_manifest.",
  )
  parser.add_argument(
      "--treatment_condition",
      action="append",
      default=[],
      choices=("breakdown", "c2_g", "c2_o"),
      help=(
          "Explicit condition per --breakdown_manifest. Repeat in the same "
          "order; defaults to legacy 'breakdown'."
      ),
  )
  parser.add_argument("--out_dir", required=True)
  parser.add_argument(
      "--primary_cohort_manifest",
      default="",
      help=(
          "Optional frozen cohort JSON. When set, identically incomplete "
          "condition manifests and missing release provenance fail."
      ),
  )
  parser.add_argument(
      "--selected_triplets",
      default="",
      help=(
          "Committed schedule-consumer selected_triplets.jsonl. Required "
          "with --primary_cohort_manifest; ordinary matrix manifests are "
          "accepted only in non-primary legacy/development mode."
      ),
  )
  args = parser.parse_args()

  primary_cohort = None
  primary_cohort_path: Path | None = None
  if args.primary_cohort_manifest:
    primary_cohort_path = Path(
        args.primary_cohort_manifest
    ).expanduser().absolute()
    primary_cohort = json.loads(
        primary_cohort_path.read_text(encoding="utf-8")
    )

  selection_audit = None
  if primary_cohort is not None:
    if not args.selected_triplets:
      parser.error(
          "--selected_triplets is required for primary-cohort reporting"
      )
    if (
        args.baseline_manifest
        or args.breakdown_manifest
        or args.label
        or args.treatment_condition
    ):
      parser.error(
          "Primary reporting ingests only the committed selected triplets; "
          "ordinary matrix manifests and treatment overrides are "
          "legacy/development inputs"
      )
    if args.baseline_condition not in (None, "c1"):
      parser.error("Primary direct-execution condition is fixed to c1")
    assert primary_cohort_path is not None
    try:
      selected_harvests, selection_audit = _harvest_selected_triplets(
          Path(args.selected_triplets), primary_cohort_path
      )
    except SelectedTripletValidationError as exc:
      parser.error(f"invalid committed triplet selection: {exc}")
    baseline = selected_harvests["c1"]
    baseline_path = Path(args.selected_triplets).expanduser().resolve()
    treatment_inputs = [
        ("c2_g", baseline_path, selected_harvests["c2_g"], "c2_g"),
        ("c2_o", baseline_path, selected_harvests["c2_o"], "c2_o"),
    ]
  else:
    if args.selected_triplets:
      parser.error(
          "--selected_triplets requires --primary_cohort_manifest"
      )
    if not args.baseline_manifest or not args.breakdown_manifest:
      parser.error(
          "Legacy/development mode requires --baseline_manifest and at "
          "least one --breakdown_manifest"
      )
    baseline_path = Path(args.baseline_manifest).expanduser().resolve()
    baseline_condition = args.baseline_condition or "baseline"
    baseline = _harvest(
        baseline_path, expected_condition=baseline_condition
    )
    treatment_inputs = []
    for index, raw_path in enumerate(args.breakdown_manifest):
      path = Path(raw_path).expanduser().resolve()
      label = (
          args.label[index]
          if index < len(args.label)
          else _label_from_manifest(path, index)
      )
      treatment_condition = (
          args.treatment_condition[index]
          if index < len(args.treatment_condition)
          else "breakdown"
      )
      treatment_inputs.append(
          (label, path, _harvest(path, treatment_condition), treatment_condition)
      )

  out_dir = Path(args.out_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  baseline_primary_validation = (
      _primary_harvest_validation(
          baseline,
          primary_cohort,
          "c1" if primary_cohort is not None else baseline_condition,
      )
      if primary_cohort is not None
      else None
  )
  print(
      f"baseline: {len(baseline.rows)} eligible, "
      f"{len(baseline.invalid_records)} invalid from {baseline_path}",
      flush=True,
  )

  runs_report: dict[str, Any] = {}
  per_task_rows: list[dict[str, Any]] = []
  all_runs_valid = True
  used_labels: set[str] = set()
  for label, path, treatment, treatment_condition in treatment_inputs:
    if label in used_labels:
      raise ValueError(f"Duplicate treatment label: {label}")
    used_labels.add(label)
    print(
        f"breakdown[{label}]: {len(treatment.rows)} eligible, "
        f"{len(treatment.invalid_records)} invalid from {path}",
        flush=True,
    )

    overall = _paired_metrics(baseline.rows, treatment.rows)
    paired_keys = [tuple(key) for key in overall["paired_keys"]]
    by_model = _group_by(paired_keys, baseline.rows, treatment.rows, by=0)
    by_category = _group_by(paired_keys, baseline.rows, treatment.rows, by=1)
    roster = _roster_validation(baseline, treatment)
    plan_reuse = _plan_reuse_validation(treatment)
    treatment_primary_validation = (
        _primary_harvest_validation(
            treatment, primary_cohort, treatment_condition
        )
        if primary_cohort is not None
        else None
    )
    strictly_valid = bool(
        overall["n_paired"]
        and not baseline.invalid_records
        and not treatment.invalid_records
        and not overall["n_pair_provenance_mismatches"]
        and not overall["baseline_only"]
        and not overall["treatment_only"]
        and roster["valid"]
        and plan_reuse["valid"]
        and (
            baseline_primary_validation is None
            or baseline_primary_validation["valid"]
        )
        and (
            treatment_primary_validation is None
            or treatment_primary_validation["valid"]
        )
    )
    all_runs_valid = all_runs_valid and strictly_valid
    runs_report[label] = {
        "baseline_manifest": str(baseline_path),
        "treatment_manifest": str(path),
        "strictly_valid": strictly_valid,
        "baseline_harvest": baseline.summary(),
        "treatment_harvest": treatment.summary(),
        "roster_validation": roster,
        "plan_reuse_validation": plan_reuse,
        "baseline_primary_validation": baseline_primary_validation,
        "treatment_primary_validation": treatment_primary_validation,
        "overall": overall,
        "by_model": by_model,
        "by_category": by_category,
    }
    per_task_rows.extend(_per_task_rows(label, baseline, treatment))

  report = {
      "interpretation": {
          "baseline_fail_treatment_pass": "planning_responsive",
          "baseline_fail_treatment_fail": "residual_under_plan_assistance",
          "warning": (
              "Residual failure under plan assistance is compatible with an "
              "incomplete plan, grounding, action execution, or recovery; it "
              "must not be labeled grounding from C1/C2 outcomes alone."
          ),
      },
      "all_runs_strictly_valid": all_runs_valid,
      "selection_audit": selection_audit,
      "inference_role": (
          "descriptive_pairing_audit_primary_inference_is_hierarchical"
          if primary_cohort is not None else "legacy_or_development"
      ),
      "runs": runs_report,
  }
  out_dir.joinpath("paired_summary.json").write_text(
      json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
  )
  _write_markdown(out_dir / "paired_summary.md", report)
  with out_dir.joinpath("paired_per_task.jsonl").open("w", encoding="utf-8") as handle:
    for row in per_task_rows:
      handle.write(json.dumps(row, ensure_ascii=False) + "\n")
  print(f"Wrote {out_dir / 'paired_summary.md'}")
  if not all_runs_valid:
    print("Strict pairing validity failed; see paired_summary.json.", file=sys.stderr)
    return 2
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
