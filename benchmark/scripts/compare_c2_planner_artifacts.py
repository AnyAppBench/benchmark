#!/usr/bin/env python3
"""Fail-closed structural comparison of Gemini and Qwen C2 plan artifacts.

This utility establishes that two generated breakdown files are comparable as
planner variants.  It verifies the frozen task roster, seed and prompt binding,
semantic identities, warning-free plans, and cross-app plan reuse.  It reports
whether plan text is identical or different, but deliberately makes no claim
about plan quality.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

import task_breakdowns


EXPECTED_ENTRY_COUNT = 690
EXPECTED_PLAN_COUNT = 150
EXPECTED_SEMANTIC_TASK_COUNT = 50
EXPECTED_INSTANCES_PER_SEMANTIC_TASK = frozenset({0, 1, 2})
EXPECTED_PLAN_REUSE_MULTIPLICITIES = {3: 30, 4: 30, 5: 60, 6: 30}
EXPECTED_CATEGORIES = ["sms", "files", "maps", "contacts", "clock"]
EXPECTED_GEMINI_MODEL_IDENTITY = "gemini-3.1-pro-preview"
EXPECTED_QWEN_MODEL_IDENTITY = "Qwen/Qwen3-VL-30B-A3B-Instruct"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ComparisonError(ValueError):
  """Raised when artifacts are not valid members of one paired comparison."""

  def __init__(self, errors: list[str]):
    self.errors = errors
    super().__init__("; ".join(errors))


def _canonical_json(value: Any) -> bytes:
  return json.dumps(
      value,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
      allow_nan=False,
  ).encode("utf-8")


def _json_sha256(value: Any) -> str:
  return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
  if path.is_symlink() or not path.is_file():
    raise ComparisonError([f"Input must be a regular non-symlink file: {path}"])

  def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f"duplicate JSON key {key!r}")
      result[key] = value
    return result

  try:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {item!r}")
        ),
    )
  except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
    raise ComparisonError([f"Invalid JSON in {path}: {exc}"]) from exc
  if not isinstance(value, dict):
    raise ComparisonError([f"Artifact root must be a JSON object: {path}"])
  return value


def _is_int(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool)


def _required_string(
    value: Any,
    *,
    location: str,
    errors: list[str],
) -> str:
  if not isinstance(value, str) or not value or value != value.strip():
    errors.append(f"{location} must be a non-empty stripped string")
    return ""
  return value


def _required_sha(
    value: Any,
    *,
    location: str,
    errors: list[str],
) -> str:
  if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
    errors.append(f"{location} must be a lowercase SHA-256")
    return ""
  return value


def _validate_artifact(path: Path, label: str) -> dict[str, Any]:
  payload = _strict_json(path)
  errors: list[str] = []
  metadata = payload.get("metadata")
  if not isinstance(metadata, dict):
    raise ComparisonError([f"{label}.metadata must be a JSON object"])

  provider = _required_string(
      metadata.get("generator_provider"),
      location=f"{label}.metadata.generator_provider",
      errors=errors,
  )
  model = _required_string(
      metadata.get("generator_model"),
      location=f"{label}.metadata.generator_model",
      errors=errors,
  )
  model_identity = _required_string(
      metadata.get("generator_model_identity"),
      location=f"{label}.metadata.generator_model_identity",
      errors=errors,
  )
  if provider.casefold() == "human":
    errors.append(f"{label} is human-authored, not a generated planner artifact")
  prompt_sha256 = _required_sha(
      metadata.get("prompt_sha256"),
      location=f"{label}.metadata.prompt_sha256",
      errors=errors,
  )
  cohort_release_id = _required_string(
      metadata.get("cohort_release_id"),
      location=f"{label}.metadata.cohort_release_id",
      errors=errors,
  )
  cohort_manifest_sha256 = _required_sha(
      metadata.get("cohort_manifest_sha256"),
      location=f"{label}.metadata.cohort_manifest_sha256",
      errors=errors,
  )
  suite_family = _required_string(
      metadata.get("suite_family"),
      location=f"{label}.metadata.suite_family",
      errors=errors,
  )

  if metadata.get("expected_entry_count") != EXPECTED_ENTRY_COUNT:
    errors.append(
        f"{label}.metadata.expected_entry_count must be {EXPECTED_ENTRY_COUNT}"
    )
  if metadata.get("expected_semantic_plan_count") != EXPECTED_PLAN_COUNT:
    errors.append(
        f"{label}.metadata.expected_semantic_plan_count must be "
        f"{EXPECTED_PLAN_COUNT}"
    )
  if metadata.get("n_task_combinations") != 3:
    errors.append(f"{label}.metadata.n_task_combinations must be 3")
  if metadata.get("task_random_seed") != 30:
    errors.append(f"{label}.metadata.task_random_seed must be 30")
  if metadata.get("fixed_task_seed") is not False:
    errors.append(f"{label}.metadata.fixed_task_seed must be false")
  if metadata.get("semantic_pairing_version") != 2:
    errors.append(f"{label}.metadata.semantic_pairing_version must be 2")
  if metadata.get("plan_reuse_policy") != (
      "one_plan_per_semantic_instance_across_apps"
  ):
    errors.append(f"{label}.metadata.plan_reuse_policy is invalid")
  if metadata.get("planner_input_app_identity") != "replaced_with_[TARGET_APP]":
    errors.append(f"{label}.metadata.planner_input_app_identity is invalid")
  categories = metadata.get("categories")
  if categories != EXPECTED_CATEGORIES:
    errors.append(
        f"{label}.metadata.categories must equal {EXPECTED_CATEGORIES!r}"
    )
  if not isinstance(metadata.get("tasks"), list):
    errors.append(f"{label}.metadata.tasks must be a JSON list")
  expected_generation_policy = {
      "temperature": 0.0,
      "max_retry": 3,
      "timeout_sec": 120.0,
      "sleep_seconds": 0.0,
      "validation_retry": 3,
      "strict_forbidden_check": True,
      "response_contract": (
          "provider_json_mode_then_common_schema_and_forbidden_detail_"
          "validation"
      ),
      "selection_policy": "first_accepted_machine_valid_plan_no_best_of_n",
  }
  generation_policy = metadata.get("generation_policy")
  if generation_policy != expected_generation_policy:
    errors.append(
        f"{label}.metadata.generation_policy must equal the frozen paired "
        "planner policy"
    )

  attempt_audit = metadata.get("attempt_audit")
  if not isinstance(attempt_audit, dict):
    errors.append(f"{label}.metadata.attempt_audit must be present")
  else:
    for field in ("header_sha256", "tail_sha256", "generator_config_sha256"):
      _required_sha(
          attempt_audit.get(field),
          location=f"{label}.metadata.attempt_audit.{field}",
          errors=errors,
      )
    record_count = attempt_audit.get("record_count")
    if not _is_int(record_count) or record_count < EXPECTED_PLAN_COUNT + 1:
      errors.append(
          f"{label}.metadata.attempt_audit.record_count must be at least "
          f"{EXPECTED_PLAN_COUNT + 1}"
      )

  entries = payload.get("breakdowns")
  if not isinstance(entries, list):
    raise ComparisonError(errors + [f"{label}.breakdowns must be a JSON list"])
  if len(entries) != EXPECTED_ENTRY_COUNT:
    errors.append(
        f"{label} has {len(entries)} exact entries; expected "
        f"{EXPECTED_ENTRY_COUNT}"
    )

  by_exact_key: dict[str, dict[str, Any]] = {}
  by_plan_key: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  for index, entry in enumerate(entries):
    location = f"{label}.breakdowns[{index}]"
    if not isinstance(entry, dict):
      errors.append(f"{location} must be a JSON object")
      continue
    exact_key = _required_string(
        entry.get("key"), location=f"{location}.key", errors=errors
    )
    plan_key = _required_string(
        entry.get("plan_key"), location=f"{location}.plan_key", errors=errors
    )
    task_template = _required_string(
        entry.get("task_template"),
        location=f"{location}.task_template",
        errors=errors,
    )
    semantic_task_id = _required_string(
        entry.get("semantic_task_id"),
        location=f"{location}.semantic_task_id",
        errors=errors,
    )
    goal = _required_string(
        entry.get("goal"), location=f"{location}.goal", errors=errors
    )
    semantic_goal = _required_string(
        entry.get("semantic_goal"),
        location=f"{location}.semantic_goal",
        errors=errors,
    )
    instance_id = entry.get("instance_id")
    if not _is_int(instance_id) or instance_id < 0:
      errors.append(f"{location}.instance_id must be a non-negative integer")
      instance_id = -1
    goal_sha256 = _required_sha(
        entry.get("goal_sha256"),
        location=f"{location}.goal_sha256",
        errors=errors,
    )
    semantic_goal_sha256 = _required_sha(
        entry.get("semantic_goal_sha256"),
        location=f"{location}.semantic_goal_sha256",
        errors=errors,
    )
    semantic_parameter_sha256 = _required_sha(
        entry.get("semantic_parameter_sha256"),
        location=f"{location}.semantic_parameter_sha256",
        errors=errors,
    )
    plan_sha256 = _required_sha(
        entry.get("plan_sha256"),
        location=f"{location}.plan_sha256",
        errors=errors,
    )

    if goal and goal_sha256 and task_breakdowns.goal_sha256(goal) != goal_sha256:
      errors.append(f"{location}.goal_sha256 does not hash the normalized goal")
    if semantic_goal and semantic_goal_sha256 and (
        task_breakdowns.goal_sha256(semantic_goal) != semantic_goal_sha256
    ):
      errors.append(
          f"{location}.semantic_goal_sha256 does not hash semantic_goal"
      )
    if task_template and goal and instance_id >= 0:
      expected_exact_key = task_breakdowns.make_key(
          task_template, goal, instance_id
      )
      if exact_key != expected_exact_key:
        errors.append(f"{location}.key does not match its exact task identity")
    if semantic_task_id and semantic_goal and instance_id >= 0:
      expected_plan_key = task_breakdowns.make_semantic_plan_key(
          semantic_task_id, instance_id, semantic_goal
      )
      if plan_key != expected_plan_key:
        errors.append(f"{location}.plan_key does not match semantic identity")

    if entry.get("generator_provider") != provider:
      errors.append(f"{location}.generator_provider differs from metadata")
    if entry.get("generator_model") != model:
      errors.append(f"{location}.generator_model differs from metadata")
    if entry.get("generator_model_identity") != model_identity:
      errors.append(
          f"{location}.generator_model_identity differs from metadata"
      )
    warnings = entry.get("validation_warnings")
    if not isinstance(warnings, list):
      errors.append(f"{location}.validation_warnings must be a JSON list")
    elif warnings:
      errors.append(f"{location}.validation_warnings is non-empty: {warnings!r}")

    breakdown = entry.get("breakdown")
    if not isinstance(breakdown, dict):
      errors.append(f"{location}.breakdown must be a JSON object")
    breakdown_text = entry.get("breakdown_text")
    if not isinstance(breakdown_text, str) or not breakdown_text.strip():
      errors.append(f"{location}.breakdown_text must be non-empty text")
      breakdown_text = ""
    if isinstance(breakdown, dict):
      computed_text = task_breakdowns.format_breakdown_text(
          {"breakdown": breakdown}
      )
      if breakdown_text != computed_text:
        errors.append(f"{location}.breakdown_text differs from breakdown JSON")
    if breakdown_text and plan_sha256 and (
        hashlib.sha256(breakdown_text.encode("utf-8")).hexdigest()
        != plan_sha256
    ):
      errors.append(f"{location}.plan_sha256 does not hash breakdown_text")

    if exact_key:
      if exact_key in by_exact_key:
        errors.append(f"{label} contains duplicate exact key {exact_key!r}")
      else:
        by_exact_key[exact_key] = entry
    if plan_key:
      by_plan_key[plan_key].append(entry)

  if len(by_plan_key) != EXPECTED_PLAN_COUNT:
    errors.append(
        f"{label} has {len(by_plan_key)} unique plan keys; expected "
        f"{EXPECTED_PLAN_COUNT}"
    )

  semantic_task_instances: dict[str, set[int]] = collections.defaultdict(set)
  reuse_multiplicities: collections.Counter[int] = collections.Counter()
  plan_material: dict[str, dict[str, Any]] = {}
  for plan_key, grouped_entries in sorted(by_plan_key.items()):
    first = grouped_entries[0]
    identity_fields = (
        "semantic_task_id",
        "instance_id",
        "semantic_goal",
        "semantic_goal_sha256",
        "semantic_parameter_sha256",
    )
    first_identity = {field: first.get(field) for field in identity_fields}
    first_text = first.get("breakdown_text")
    first_breakdown = first.get("breakdown")
    first_plan_sha = first.get("plan_sha256")
    for entry in grouped_entries[1:]:
      identity = {field: entry.get(field) for field in identity_fields}
      if identity != first_identity:
        errors.append(
            f"{label} plan {plan_key!r} has different semantic identities "
            "across apps"
        )
        break
      if (
          entry.get("breakdown_text") != first_text
          or entry.get("breakdown") != first_breakdown
          or entry.get("plan_sha256") != first_plan_sha
      ):
        errors.append(
            f"{label} plan {plan_key!r} is not reused byte-identically "
            "across apps"
        )
        break
    semantic_task_id = first.get("semantic_task_id")
    instance_id = first.get("instance_id")
    if isinstance(semantic_task_id, str) and _is_int(instance_id):
      semantic_task_instances[semantic_task_id].add(instance_id)
    reuse_multiplicities[len(grouped_entries)] += 1
    plan_material[plan_key] = {
        "identity": first_identity,
        "breakdown_text": first_text,
        "plan_sha256": first_plan_sha,
    }

  if len(semantic_task_instances) != EXPECTED_SEMANTIC_TASK_COUNT:
    errors.append(
        f"{label} has {len(semantic_task_instances)} semantic tasks; expected "
        f"{EXPECTED_SEMANTIC_TASK_COUNT}"
    )
  for semantic_task_id, instance_ids in sorted(semantic_task_instances.items()):
    if instance_ids != EXPECTED_INSTANCES_PER_SEMANTIC_TASK:
      errors.append(
          f"{label} semantic task {semantic_task_id!r} has instance IDs "
          f"{sorted(instance_ids)!r}; expected [0, 1, 2]"
      )
  if dict(sorted(reuse_multiplicities.items())) != (
      EXPECTED_PLAN_REUSE_MULTIPLICITIES
  ):
    errors.append(
        f"{label} plan reuse multiplicities are "
        f"{dict(sorted(reuse_multiplicities.items()))!r}; expected "
        f"{EXPECTED_PLAN_REUSE_MULTIPLICITIES!r}"
    )

  task_set = []
  for exact_key, entry in sorted(by_exact_key.items()):
    task_set.append({
        "key": exact_key,
        "task_template": entry.get("task_template"),
        "instance_id": entry.get("instance_id"),
        "goal": entry.get("goal"),
        "goal_sha256": entry.get("goal_sha256"),
        "semantic_task_id": entry.get("semantic_task_id"),
        "semantic_goal": entry.get("semantic_goal"),
        "semantic_goal_sha256": entry.get("semantic_goal_sha256"),
        "semantic_parameter_sha256": entry.get("semantic_parameter_sha256"),
        "plan_key": entry.get("plan_key"),
        "app_display_name": entry.get("app_display_name"),
    })

  if errors:
    raise ComparisonError(errors)
  return {
      "path": str(path.resolve()),
      "artifact_sha256": _file_sha256(path),
      "provider": provider,
      "model": model,
      "model_identity": model_identity,
      "model_identity_sha256": hashlib.sha256(
          model_identity.encode("utf-8")
      ).hexdigest(),
      "prompt_sha256": prompt_sha256,
      "cohort_release_id": cohort_release_id,
      "cohort_manifest_sha256": cohort_manifest_sha256,
      "suite_family": suite_family,
      "categories": categories,
      "tasks": metadata.get("tasks"),
      "seed_config": {
          "n_task_combinations": metadata.get("n_task_combinations"),
          "task_random_seed": metadata.get("task_random_seed"),
          "fixed_task_seed": metadata.get("fixed_task_seed"),
      },
      "generation_policy": generation_policy,
      "entry_count": len(entries),
      "semantic_plan_count": len(by_plan_key),
      "semantic_task_count": len(semantic_task_instances),
      "task_set": task_set,
      "task_set_sha256": _json_sha256(task_set),
      "exact_keys": set(by_exact_key),
      "plan_material": plan_material,
      "plan_reuse_multiplicities": dict(sorted(reuse_multiplicities.items())),
  }


def compare_artifacts(gemini_path: Path, qwen_path: Path) -> dict[str, Any]:
  gemini = _validate_artifact(gemini_path, "gemini")
  qwen = _validate_artifact(qwen_path, "qwen")
  errors: list[str] = []

  if gemini["provider"].casefold() != "gemini":
    errors.append("Gemini artifact generator_provider must be 'gemini'")
  if qwen["provider"].casefold() != "qwen":
    errors.append("Qwen artifact generator_provider must be 'qwen'")
  if gemini["model_identity"] != EXPECTED_GEMINI_MODEL_IDENTITY:
    errors.append(
        "Gemini model identity must be "
        f"{EXPECTED_GEMINI_MODEL_IDENTITY!r}"
    )
  if qwen["model_identity"] != EXPECTED_QWEN_MODEL_IDENTITY:
    errors.append(
        "Qwen model identity must be "
        f"{EXPECTED_QWEN_MODEL_IDENTITY!r}"
    )
  if gemini["provider"] == qwen["provider"]:
    errors.append("Gemini and Qwen generator providers must be distinct")
  if gemini["model"] == qwen["model"]:
    errors.append("Gemini and Qwen generator models must be distinct")
  if gemini["model_identity"] == qwen["model_identity"]:
    errors.append("Gemini and Qwen underlying model identities must be distinct")
  shared_fields = (
      "cohort_release_id",
      "cohort_manifest_sha256",
      "suite_family",
      "categories",
      "tasks",
      "seed_config",
      "generation_policy",
      "prompt_sha256",
      "task_set_sha256",
  )
  for field in shared_fields:
    if gemini[field] != qwen[field]:
      errors.append(f"Artifact comparison mismatch: {field}")
  if gemini["exact_keys"] != qwen["exact_keys"]:
    errors.append("Artifact comparison mismatch: exact entry key roster")

  gemini_plans = gemini["plan_material"]
  qwen_plans = qwen["plan_material"]
  if set(gemini_plans) != set(qwen_plans):
    errors.append("Artifact comparison mismatch: semantic plan key roster")
  else:
    for plan_key in sorted(gemini_plans):
      if gemini_plans[plan_key]["identity"] != qwen_plans[plan_key]["identity"]:
        errors.append(
            f"Artifact comparison semantic identity mismatch: {plan_key}"
        )

  if errors:
    raise ComparisonError(errors)

  identical_plan_keys = []
  different_plan_keys = []
  for plan_key in sorted(gemini_plans):
    if (
        gemini_plans[plan_key]["breakdown_text"]
        == qwen_plans[plan_key]["breakdown_text"]
    ):
      identical_plan_keys.append(plan_key)
    else:
      different_plan_keys.append(plan_key)

  def _public_artifact(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "path",
            "artifact_sha256",
            "provider",
            "model",
            "model_identity",
            "model_identity_sha256",
            "entry_count",
            "semantic_plan_count",
            "semantic_task_count",
            "plan_reuse_multiplicities",
        )
    }

  return {
      "schema_version": 1,
      "status": "pass",
      "comparison_scope": (
          "structural_comparability_only_no_plan_quality_judgment"
      ),
      "quality_assessment_performed": False,
      "shared_frozen_configuration": {
          "cohort_release_id": gemini["cohort_release_id"],
          "cohort_manifest_sha256": gemini["cohort_manifest_sha256"],
          "suite_family": gemini["suite_family"],
          "categories": gemini["categories"],
          "tasks": gemini["tasks"],
          "seed_config": gemini["seed_config"],
          "generation_policy": gemini["generation_policy"],
          "prompt_sha256": gemini["prompt_sha256"],
          "task_set_sha256": gemini["task_set_sha256"],
          "exact_entry_count": EXPECTED_ENTRY_COUNT,
          "semantic_plan_count": EXPECTED_PLAN_COUNT,
      },
      "artifacts": {
          "gemini": _public_artifact(gemini),
          "qwen": _public_artifact(qwen),
      },
      "plan_text_comparison": {
          "total": EXPECTED_PLAN_COUNT,
          "identical_count": len(identical_plan_keys),
          "different_count": len(different_plan_keys),
          "identical_plan_keys": identical_plan_keys,
          "different_plan_keys": different_plan_keys,
      },
  }


def _write_report(path: Path, report: dict[str, Any]) -> None:
  path = path.expanduser()
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
      json.dumps(report, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  temporary.replace(path)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--gemini_breakdown_file", required=True)
  parser.add_argument("--qwen_breakdown_file", required=True)
  parser.add_argument(
      "--report_json",
      default="",
      help="Optional path for the same machine-readable report printed to stdout.",
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  try:
    report = compare_artifacts(
        Path(args.gemini_breakdown_file).expanduser(),
        Path(args.qwen_breakdown_file).expanduser(),
    )
  except ComparisonError as exc:
    report = {
        "schema_version": 1,
        "status": "fail",
        "errors": exc.errors,
        "quality_assessment_performed": False,
    }
    if args.report_json:
      _write_report(Path(args.report_json), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 2
  if args.report_json:
    _write_report(Path(args.report_json), report)
  print(json.dumps(report, indent=2, ensure_ascii=False))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
