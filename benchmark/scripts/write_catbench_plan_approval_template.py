#!/usr/bin/env python3
"""Write a deliberately non-approving C2 human-review worksheet.

This utility hashes an existing frozen plan file and enumerates its semantic
plan keys.  It never fills reviewer identities, decisions, evidence hashes, or
approval status.  Consequently its output is useful as a worksheet but is
intentionally rejected by the frozen schedule consumer until two real humans
complete independent per-plan reviews and construct a release approval
manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import preflight_task_breakdowns as plan_preflight


REVIEW_DIMENSIONS = (
    "correctness",
    "completeness",
    "semantic_parameter_preservation",
    "app_independence",
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
  if path.is_symlink() or not path.is_file():
    raise ValueError(f"Input must be a regular non-symlink file: {path}")

  def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f"Duplicate JSON key {key!r} in {path}")
      result[key] = value
    return result

  value = json.loads(
      path.read_text(encoding="utf-8"),
      object_pairs_hook=_object,
      parse_constant=lambda item: (_ for _ in ()).throw(
          ValueError(f"Non-finite JSON constant {item!r} in {path}")
      ),
  )
  if not isinstance(value, dict):
    raise ValueError(f"Plan root must be a JSON object: {path}")
  return value


def _expected_cohort_roster(
    cohort_manifest_path: Path,
) -> tuple[set[str], set[str]]:
  cohort = _strict_json(cohort_manifest_path)
  categories = cohort.get("categories")
  if not isinstance(categories, dict) or not categories:
    raise ValueError("Cohort manifest must contain a non-empty categories object")
  args = argparse.Namespace(
      suite_family=str(cohort.get("suite_family") or ""),
      cohort_manifest=str(cohort_manifest_path),
      tasks="",
      categories=",".join(str(category) for category in categories),
      n_task_combinations=int(cohort.get("n_task_combinations", -1)),
      task_random_seed=int(cohort.get("task_random_seed", -1)),
      fixed_task_seed=False,
  )
  scheduled = plan_preflight._enumerate_scheduled_tasks(  # pylint: disable=protected-access
      args
  )
  exact_keys = [str(item["key"]) for item in scheduled]
  if len(exact_keys) != len(set(exact_keys)):
    raise ValueError("Frozen cohort enumeration contains duplicate exact keys")
  return set(exact_keys), {str(item["plan_key"]) for item in scheduled}


def build_template(
    *,
    breakdown_path: Path,
    cohort_manifest_path: Path,
    condition: str,
    release_id: str,
    expected_entry_count: int,
    expected_plan_count: int,
    expected_exact_instance_keys: set[str],
    expected_plan_keys: set[str],
    c2_g_attempt_audit: Path | None,
) -> dict[str, Any]:
  if condition not in {"c2_g", "c2_o"}:
    raise ValueError(f"Unsupported condition: {condition!r}")
  if not release_id or release_id != release_id.strip():
    raise ValueError("release_id must be a non-empty stripped string")
  if expected_entry_count < 1 or expected_plan_count < 1:
    raise ValueError("Expected entry and semantic-plan counts must be positive")
  cohort = _strict_json(cohort_manifest_path)
  if cohort.get("release_id") != release_id:
    raise ValueError("Cohort manifest release_id does not match release_id")
  cohort_sha256 = _sha256(cohort_manifest_path)
  payload = _strict_json(breakdown_path)
  metadata = payload.get("metadata")
  if not isinstance(metadata, dict):
    raise ValueError("Plan file must contain a metadata object")
  provider = str(metadata.get("generator_provider") or "").strip().lower()
  if condition == "c2_g" and (not provider or provider == "human"):
    raise ValueError("C2-G plan metadata must name a non-human generator")
  if condition == "c2_o" and provider != "human":
    raise ValueError("C2-O plan metadata generator_provider must be 'human'")
  if metadata.get("cohort_release_id") != release_id:
    raise ValueError("Plan metadata cohort_release_id does not match release_id")
  if metadata.get("cohort_manifest_sha256") != cohort_sha256:
    raise ValueError(
        "Plan metadata cohort_manifest_sha256 does not match the supplied cohort"
    )
  entries = payload.get("breakdowns")
  if not isinstance(entries, list) or not entries:
    raise ValueError("Plan file must contain a non-empty breakdowns list")
  plan_hashes: dict[str, str] = {}
  exact_instance_keys: set[str] = set()
  for index, entry in enumerate(entries):
    if not isinstance(entry, dict):
      raise ValueError(f"breakdowns[{index}] must be a JSON object")
    plan_key = entry.get("plan_key")
    if (
        not isinstance(plan_key, str)
        or not plan_key
        or plan_key != plan_key.strip()
    ):
      raise ValueError(f"breakdowns[{index}] has an invalid plan_key")
    exact_key = entry.get("key")
    if (
        not isinstance(exact_key, str)
        or not exact_key
        or exact_key != exact_key.strip()
    ):
      raise ValueError(f"breakdowns[{index}] has an invalid exact-instance key")
    if exact_key in exact_instance_keys:
      raise ValueError(f"Duplicate exact-instance breakdown key: {exact_key}")
    exact_instance_keys.add(exact_key)
    plan_sha256 = entry.get("plan_sha256")
    if not isinstance(plan_sha256, str) or not HEX_SHA256.fullmatch(plan_sha256):
      raise ValueError(f"breakdowns[{index}] has an invalid plan_sha256")
    prior_hash = plan_hashes.setdefault(plan_key, plan_sha256)
    if prior_hash != plan_sha256:
      raise ValueError(
          f"Entries sharing plan_key {plan_key!r} have different plan hashes"
      )
  plan_keys = sorted(plan_hashes)
  if len(entries) != expected_entry_count:
    raise ValueError(
        f"Plan file has {len(entries)} entries; expected {expected_entry_count}"
    )
  if len(plan_keys) != expected_plan_count:
    raise ValueError(
        f"Plan file has {len(plan_keys)} semantic plans; expected "
        f"{expected_plan_count}"
    )
  if exact_instance_keys != expected_exact_instance_keys:
    missing = sorted(expected_exact_instance_keys - exact_instance_keys)
    extras = sorted(exact_instance_keys - expected_exact_instance_keys)
    raise ValueError(
        "Plan exact-instance roster differs from the frozen cohort "
        f"(missing={missing[:3]}, extras={extras[:3]})"
    )
  if set(plan_keys) != expected_plan_keys:
    missing = sorted(expected_plan_keys - set(plan_keys))
    extras = sorted(set(plan_keys) - expected_plan_keys)
    raise ValueError(
        "Plan semantic-key roster differs from the frozen cohort "
        f"(missing={missing[:3]}, extras={extras[:3]})"
    )
  if metadata.get("expected_entry_count") != expected_entry_count:
    raise ValueError(
        "metadata.expected_entry_count does not match the breakdown roster"
    )
  if metadata.get("expected_semantic_plan_count") != expected_plan_count:
    raise ValueError(
        "metadata.expected_semantic_plan_count does not match the required "
        "semantic-plan roster"
    )
  template: dict[str, Any] = {
      "schema_version": 1,
      "release_id": release_id,
      "condition": condition,
      "cohort_manifest_sha256": cohort_sha256,
      "breakdown_sha256": _sha256(breakdown_path),
      "approval_policy": "independent_two_person_complete_plan_set",
      "approval_status": "pending_human_review",
      "required_entry_count": expected_entry_count,
      "required_plan_count": len(plan_keys),
      "approved_plan_count": 0,
      "required_plan_keys": plan_keys,
      "approved_plan_keys": [],
      "review_dimensions": list(REVIEW_DIMENSIONS),
      "reviewers": [],
      "review_worksheet": [
          {
              "plan_key": plan_key,
              "reviewer_1": None,
              "reviewer_2": None,
          }
          for plan_key in plan_keys
      ],
      "template_claim": (
          "worksheet_only_no_review_no_approval_no_rollout_authorization"
      ),
  }
  if condition == "c2_g":
    if c2_g_attempt_audit is None:
      raise ValueError("C2-G template requires --c2_g_attempt_audit")
    if c2_g_attempt_audit.is_symlink() or not c2_g_attempt_audit.is_file():
      raise ValueError("C2-G attempt audit must be a regular non-symlink file")
    template.update({
        "attempt_audit_sha256": _sha256(c2_g_attempt_audit),
        "generated_plan_edit_policy": "accepted_generator_output_unedited",
    })
  else:
    template["authoring_policy"] = "two_human_authors_app_neutral"
  return template


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--breakdown_file", required=True)
  parser.add_argument("--cohort_manifest", required=True)
  parser.add_argument("--condition", choices=("c2_g", "c2_o"), required=True)
  parser.add_argument("--release_id", required=True)
  parser.add_argument("--expected_entry_count", type=int, required=True)
  parser.add_argument("--expected_plan_count", type=int, required=True)
  parser.add_argument("--c2_g_attempt_audit", default="")
  parser.add_argument("--output", required=True)
  args = parser.parse_args()

  breakdown_path = Path(args.breakdown_file).expanduser().resolve()
  cohort_manifest_path = Path(args.cohort_manifest).expanduser().resolve()
  audit_path = (
      Path(args.c2_g_attempt_audit).expanduser().resolve()
      if args.c2_g_attempt_audit
      else None
  )
  output_path = Path(args.output).expanduser().resolve()
  if output_path.exists():
    raise FileExistsError(f"Refusing to overwrite review worksheet: {output_path}")
  expected_exact_keys, expected_plan_keys = _expected_cohort_roster(
      cohort_manifest_path
  )
  template = build_template(
      breakdown_path=breakdown_path,
      cohort_manifest_path=cohort_manifest_path,
      condition=args.condition,
      release_id=args.release_id,
      expected_entry_count=args.expected_entry_count,
      expected_plan_count=args.expected_plan_count,
      expected_exact_instance_keys=expected_exact_keys,
      expected_plan_keys=expected_plan_keys,
      c2_g_attempt_audit=audit_path,
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(template, indent=2, ensure_ascii=False) + "\n")
  print(
      "WROTE PENDING WORKSHEET ONLY; this file is not approval and the "
      f"schedule consumer will reject it: {output_path}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
