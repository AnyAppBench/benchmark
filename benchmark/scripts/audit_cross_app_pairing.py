#!/usr/bin/env python3
"""Audit that CATBench semantic instances are identical across real apps.

This launch gate instantiates the scheduled task roster without starting an
agent or emulator.  It groups app-specific task classes by their canonical
``catbench_semantic_id`` and verifies identical sampled parameters, identical
app-neutral goals, and a rendered goal that differs only in the declared app
slot.  The audit is outcome-independent and must pass before C1 or C2 runs.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_ROOT.parent
for path in (BENCHMARK_ROOT, REPO_ROOT):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  import pysqlite3

  sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ModuleNotFoundError:
  pass

import task_breakdowns
from android_world import registry
from android_world import suite_utils
from app_generalization_profiles import get_domain_profiles


DEFAULT_CATEGORIES = ("sms", "files", "maps", "contacts", "clock")
DEFAULT_COHORT_MANIFEST = (
    BENCHMARK_ROOT / "configs" / "catbench_5cat_primary_cohort.json"
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
  def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} in {path}")

  def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f"Duplicate JSON key {key!r} in {path}")
      result[key] = value
    return result

  payload = json.loads(
      path.read_text(encoding="utf-8"),
      parse_constant=reject_constant,
      object_pairs_hook=reject_duplicates,
  )
  if not isinstance(payload, dict):
    raise ValueError(f"Expected a JSON object in {path}")
  return payload


def _audit_provenance(
    categories: tuple[str, ...],
    cohort_path: Path | None,
) -> dict[str, Any]:
  source_paths = [
      Path(__file__).resolve(),
      BENCHMARK_ROOT / "app_generalization_profiles.py",
      BENCHMARK_ROOT / "task_breakdowns.py",
      BENCHMARK_ROOT / "android_world" / "registry.py",
      BENCHMARK_ROOT / "android_world" / "suite_utils.py",
  ]
  generated_root = (
      BENCHMARK_ROOT
      / "android_world"
      / "task_evals"
      / "single"
      / "app_generalization_generated"
  )
  source_paths.extend(
      generated_root / f"{category}_cross_app_tasks.py"
      for category in categories
  )
  if cohort_path is not None:
    source_paths.append(cohort_path)
  return {
      "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
      "python_version": platform.python_version(),
      "repository_root": str(REPO_ROOT.resolve()),
      "script_path": str(Path(__file__).resolve()),
      "script_sha256": _sha256(Path(__file__).resolve()),
      "cohort_manifest": (
          {
              "path": str(cohort_path),
              "sha256": _sha256(cohort_path),
          }
          if cohort_path is not None
          else None
      ),
      "source_files": [
          {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
          for path in source_paths
      ],
  }


def _parse_csv(value: str) -> tuple[str, ...]:
  return tuple(item.strip() for item in value.split(",") if item.strip())


def _scheduled_task_names(categories: tuple[str, ...]) -> list[str]:
  profiles = get_domain_profiles()
  unknown = sorted(set(categories) - set(profiles))
  if unknown:
    raise ValueError(f"Unknown categories: {unknown}")
  names: list[str] = []
  seen: set[str] = set()
  for category in categories:
    for app in profiles[category].apps:
      for task_name in app.implemented_tasks:
        if task_name not in seen:
          seen.add(task_name)
          names.append(task_name)
  return names


def _json_params(params: Any) -> str:
  return json.dumps(
      params,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
      default=str,
  )


def audit(
    categories: tuple[str, ...],
    n_task_combinations: int,
    task_random_seed: int,
    suite_family: str,
    cohort: dict[str, Any] | None = None,
) -> dict[str, Any]:
  task_names = _scheduled_task_names(categories)
  task_registry = registry.TaskRegistry().get_registry(family=suite_family)
  missing = sorted(set(task_names) - set(task_registry))
  if missing:
    raise ValueError(f"Scheduled classes absent from registry: {missing}")
  suite = suite_utils.create_suite(
      task_registry,
      n_task_combinations=n_task_combinations,
      seed=task_random_seed,
      tasks=task_names,
      use_identical_params=False,
  )

  rows: list[dict[str, Any]] = []
  groups: dict[tuple[str, int], list[dict[str, Any]]] = (
      collections.defaultdict(list)
  )
  for task_template, instances in suite.items():
    for instance_id, task in enumerate(instances):
      task_type = type(task)
      semantic_task_id = str(
          getattr(task_type, "catbench_semantic_id", "")
      )
      app_display_name = str(
          getattr(task_type, "catbench_app_display_name", "")
      )
      semantic_goal = task_breakdowns.app_neutral_goal(
          task.goal, app_display_name
      )
      row = {
          "task_template": task_template,
          "semantic_task_id": semantic_task_id,
          "instance_id": instance_id,
          "app_display_name": app_display_name,
          "package_name": getattr(task_type, "package_name", None),
          "goal": task.goal,
          "semantic_goal": semantic_goal,
          "params": task.params,
          "params_json": _json_params(task.params),
      }
      rows.append(row)
      groups[(semantic_task_id, instance_id)].append(row)

  violations: list[dict[str, Any]] = []
  group_summaries: list[dict[str, Any]] = []
  for key, members in sorted(groups.items()):
    reasons: list[str] = []
    if not key[0]:
      reasons.append("missing_semantic_task_id")
    params = {member["params_json"] for member in members}
    neutral_goals = {member["semantic_goal"] for member in members}
    packages = {member["package_name"] for member in members}
    rendered_goal_errors = []
    for member in members:
      expected_goal = member["semantic_goal"].replace(
          task_breakdowns.TARGET_APP_PLACEHOLDER,
          member["app_display_name"],
          1,
      )
      if task_breakdowns.normalize_goal(expected_goal) != (
          task_breakdowns.normalize_goal(member["goal"])
      ):
        rendered_goal_errors.append(member["task_template"])
    if len(params) != 1:
      reasons.append("parameters_differ_across_apps")
    if len(neutral_goals) != 1:
      reasons.append("neutral_goals_differ_across_apps")
    if len(packages) != len(members):
      reasons.append("duplicate_package_in_semantic_group")
    if rendered_goal_errors:
      reasons.append("rendered_goal_differs_beyond_app_slot")
    summary = {
        "semantic_task_id": key[0],
        "instance_id": key[1],
        "app_count": len(members),
        "apps": sorted(member["app_display_name"] for member in members),
        "semantic_goal": members[0]["semantic_goal"],
        "params": members[0]["params"],
        "valid": not reasons,
    }
    group_summaries.append(summary)
    if reasons:
      violations.append(
          {
              **summary,
              "reasons": reasons,
              "rendered_goal_error_tasks": rendered_goal_errors,
          }
      )

  cohort_violations: list[dict[str, Any]] = []
  if cohort is not None:
    expected_categories = tuple(cohort.get("categories", {}))
    if categories != expected_categories:
      cohort_violations.append({
          "reason": "categories_do_not_match_frozen_cohort",
          "expected": list(expected_categories),
          "actual": list(categories),
      })
    if n_task_combinations != cohort.get("n_task_combinations"):
      cohort_violations.append({
          "reason": "n_task_combinations_mismatch",
          "expected": cohort.get("n_task_combinations"),
          "actual": n_task_combinations,
      })
    if task_random_seed != cohort.get("task_random_seed"):
      cohort_violations.append({
          "reason": "task_random_seed_mismatch",
          "expected": cohort.get("task_random_seed"),
          "actual": task_random_seed,
      })

    profiles = get_domain_profiles()
    for category, spec in cohort.get("categories", {}).items():
      profile_apps = {app.app_id: app for app in profiles[category].apps}
      expected_apps = list(spec.get("app_ids", []))
      actual_apps = [
          app.app_id for app in profiles[category].apps if app.implemented_tasks
      ]
      if actual_apps != expected_apps:
        cohort_violations.append({
            "reason": "enabled_app_roster_mismatch",
            "category": category,
            "expected": expected_apps,
            "actual": actual_apps,
        })
      expected_semantics = set(spec.get("semantic_task_ids", []))
      for app_id in expected_apps:
        app = profile_apps.get(app_id)
        if app is None:
          cohort_violations.append({
              "reason": "frozen_app_missing_from_profile",
              "category": category,
              "app_id": app_id,
          })
          continue
        actual_semantics = {
            str(getattr(task_registry[name], "catbench_semantic_id", ""))
            for name in app.implemented_tasks
            if name in task_registry
        }
        if actual_semantics != expected_semantics:
          cohort_violations.append({
              "reason": "app_semantic_task_roster_mismatch",
              "category": category,
              "app_id": app_id,
              "expected": sorted(expected_semantics),
              "actual": sorted(actual_semantics),
          })

    expected = cohort.get("expected", {})
    expected_rows = expected.get("task_app_count")
    if expected_rows is not None:
      expected_rows *= n_task_combinations
      if len(rows) != expected_rows:
        cohort_violations.append({
            "reason": "scheduled_task_app_instance_count_mismatch",
            "expected": expected_rows,
            "actual": len(rows),
        })
    expected_groups = expected.get("semantic_template_count")
    if expected_groups is not None:
      expected_groups *= n_task_combinations
      if len(groups) != expected_groups:
        cohort_violations.append({
            "reason": "semantic_instance_group_count_mismatch",
            "expected": expected_groups,
            "actual": len(groups),
        })

  return {
      "categories": list(categories),
      "suite_family": suite_family,
      "task_random_seed": task_random_seed,
      "n_task_combinations": n_task_combinations,
      "registered_task_class_count": len(task_names),
      "enabled_real_app_count": len({
          row["package_name"] for row in rows if row["package_name"]
      }),
      "semantic_template_count": len({key[0] for key in groups}),
      "scheduled_task_app_instances": len(rows),
      "semantic_instance_groups": len(groups),
      "parameter_mismatch_groups": sum(
          "parameters_differ_across_apps" in violation["reasons"]
          for violation in violations
      ),
      "neutral_goal_mismatch_groups": sum(
          "neutral_goals_differ_across_apps" in violation["reasons"]
          for violation in violations
      ),
      "violation_reason_counts": dict(sorted(collections.Counter(
          reason
          for violation in violations
          for reason in violation["reasons"]
      ).items())),
      "violations": violations,
      "cohort_release_id": (cohort or {}).get("release_id"),
      "cohort_violations": cohort_violations,
      "groups": group_summaries,
      "valid": not violations and not cohort_violations,
  }


def _markdown(report: dict[str, Any]) -> str:
  lines = [
      "# CATBench Cross-App Semantic Pairing Audit",
      "",
      f"- Valid: **{report['valid']}**",
      f"- Categories: {', '.join(report['categories'])}",
      f"- Semantic instance groups: {report['semantic_instance_groups']}",
      f"- Scheduled task–app instances: "
      f"{report['scheduled_task_app_instances']}",
      f"- Violations: {len(report['violations'])}",
      f"- Frozen-cohort violations: {len(report.get('cohort_violations', []))}",
      "",
      "| Semantic task | Instance | Apps | Neutral goal | Status |",
      "|---|---:|---:|---|---|",
  ]
  for group in report["groups"]:
    neutral_goal = str(group["semantic_goal"]).replace("|", "\\|")
    lines.append(
        f"| {group['semantic_task_id']} | {group['instance_id']} | "
        f"{group['app_count']} | {neutral_goal} | "
        f"{'PASS' if group['valid'] else 'FAIL'} |"
    )
  return "\n".join(lines) + "\n"


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--categories", default=",".join(DEFAULT_CATEGORIES)
  )
  parser.add_argument("--n_task_combinations", type=int, default=3)
  parser.add_argument("--task_random_seed", type=int, default=30)
  parser.add_argument(
      "--suite_family", default=registry.TaskRegistry.ANDROID_WORLD_FAMILY
  )
  parser.add_argument("--report_json", default="")
  parser.add_argument("--report_md", default="")
  parser.add_argument(
      "--cohort_manifest",
      default="",
      help=(
          "Optional frozen primary-cohort manifest. When supplied, the audit "
          "also fails on any missing/extra app, task, instance, or seed."
      ),
  )
  args = parser.parse_args()

  cohort = None
  cohort_path = None
  if args.cohort_manifest:
    cohort_path = Path(args.cohort_manifest).expanduser().resolve()
    cohort = _strict_json(cohort_path)

  categories = _parse_csv(args.categories)
  report = audit(
      categories,
      args.n_task_combinations,
      args.task_random_seed,
      args.suite_family,
      cohort,
  )
  report["audit_provenance"] = _audit_provenance(
      categories, cohort_path
  )
  encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
  if args.report_json:
    Path(args.report_json).expanduser().write_text(encoded, encoding="utf-8")
  if args.report_md:
    Path(args.report_md).expanduser().write_text(
        _markdown(report), encoding="utf-8"
    )
  print(encoded, end="")
  return 0 if report["valid"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
