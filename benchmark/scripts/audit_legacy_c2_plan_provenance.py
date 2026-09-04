#!/usr/bin/env python3
"""Forensically audit the two saved legacy CATBench C2 plan files.

This script is read-only with respect to the source artifacts.  It checks the
internal entry identities and hashes, resolves task/app identities against the
current real CATBench registry and profiles, compares the two files, and
contrasts their old K=1/25-app universe with the frozen revision cohort.  It
does not treat current source code as an attestation of the historical code
that contacted either planner.
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
from typing import Any, Mapping


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

from android_world import registry  # noqa: E402
from app_generalization_profiles import get_domain_profiles  # noqa: E402


DEFAULT_GPT_FILE = Path(
    "$HOME/anyappbench_plans/gpt54_seed30_5cat.json"
)
DEFAULT_GEMINI_FILE = Path(
    "$HOME/anyappbench_plans/gemini31_seed30_5cat.json"
)
DEFAULT_COHORT = (
    BENCHMARK_ROOT / "configs" / "catbench_5cat_primary_cohort.json"
)
DEFAULT_GPT_LOGS = (
    Path("$HOME/anyappbench_plans/gpt54_resume_probe_111.log"),
    Path(
        "$HOME/anyappbench_plans/"
        "gpt54_resume_probe_111_after_validator_fix.log"
    ),
    Path("$HOME/anyappbench_plans/gpt54_resume2_full.log"),
    Path("$HOME/anyappbench_plans/gpt54_resume3_full.log"),
)
DEFAULT_GEMINI_LOGS = (
    Path("$HOME/anyappbench_plans/gemini31_seed30_5cat.log"),
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
  """Loads standards-compliant JSON and rejects duplicate object keys."""

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


def _normalized_goal(goal: str) -> str:
  return " ".join(str(goal).strip().split())


def _goal_sha256(goal: str) -> str:
  return hashlib.sha256(_normalized_goal(goal).encode("utf-8")).hexdigest()


def _count(values: list[str]) -> dict[str, int]:
  return dict(sorted(collections.Counter(values).items()))


def _profile_indexes(
    categories: tuple[str, ...],
) -> tuple[dict[str, tuple[str, Any]], dict[str, tuple[str, str, Any]]]:
  profiles = get_domain_profiles()
  by_package: dict[str, tuple[str, Any]] = {}
  by_app_id: dict[str, tuple[str, str, Any]] = {}
  for category in categories:
    if category not in profiles:
      raise ValueError(f"Unknown category in artifact metadata: {category}")
    for app in profiles[category].apps:
      if app.package_name:
        if app.package_name in by_package:
          raise ValueError(f"Duplicate profile package: {app.package_name}")
        by_package[str(app.package_name)] = (category, app)
      by_app_id[app.app_id] = (category, app.app_id, app)
  return by_package, by_app_id


def _display_in_goal(display_name: str, goal: str) -> bool:
  display = display_name.strip().rstrip("~").strip().casefold()
  return bool(display) and display in goal.casefold()


def _resolve_entry_identity(
    entry: Mapping[str, Any],
    task_registry: Mapping[str, type[Any]],
    by_package: Mapping[str, tuple[str, Any]],
    by_app_id: Mapping[str, tuple[str, str, Any]],
) -> dict[str, Any]:
  task_template = str(entry.get("task_template") or "")
  goal = str(entry.get("goal") or "")
  task_type = task_registry.get(task_template)
  if task_type is not None:
    package_name = str(getattr(task_type, "package_name", "") or "")
    profile_identity = by_package.get(package_name)
    if profile_identity is None:
      return {
          "resolved": False,
          "resolution": "registered_class_package_absent_from_profiles",
          "task_template": task_template,
          "package_name": package_name,
      }
    category, app = profile_identity
    display_name = str(
        getattr(task_type, "catbench_app_display_name", app.display_name)
    )
    return {
        "resolved": True,
        "resolution": "current_registered_task_class",
        "task_template": task_template,
        "semantic_task_id": str(
            getattr(task_type, "catbench_semantic_id", "")
        ),
        "category": category,
        "app_id": app.app_id,
        "display_name": display_name,
        "package_name": package_name,
        "goal_mentions_resolved_app": _display_in_goal(display_name, goal),
    }

  # The saved universe contains one historical class that has since been
  # removed from the live registry: SmsArchiveConversationForQUIKSMS.  Resolve
  # an unregistered row only when its raw goal names exactly one real app in
  # the five-category profiles.  The report preserves that this is a fallback,
  # not current registry evidence.
  candidates = []
  for category, app_id, app in by_app_id.values():
    if _display_in_goal(str(app.display_name), goal):
      candidates.append((category, app_id, app))
  if len(candidates) != 1:
    return {
        "resolved": False,
        "resolution": "unregistered_and_no_unique_profile_goal_match",
        "task_template": task_template,
        "candidate_app_ids": sorted(app_id for _, app_id, _ in candidates),
    }
  category, app_id, app = candidates[0]
  semantic_task_id = task_template.rsplit("For", 1)[0]
  return {
      "resolved": True,
      "resolution": "unregistered_historical_template_unique_goal_profile",
      "task_template": task_template,
      "semantic_task_id": semantic_task_id,
      "category": category,
      "app_id": app_id,
      "display_name": str(app.display_name),
      "package_name": str(app.package_name or ""),
      "goal_mentions_resolved_app": True,
  }


def _grid_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
  resolved = [row for row in rows if row["identity"].get("resolved")]
  category_apps: dict[str, set[str]] = collections.defaultdict(set)
  category_semantics: dict[str, set[str]] = collections.defaultdict(set)
  cell_counts: collections.Counter[tuple[str, str, str, int]] = (
      collections.Counter()
  )
  for row in resolved:
    identity = row["identity"]
    category = str(identity["category"])
    app_id = str(identity["app_id"])
    semantic = str(identity["semantic_task_id"])
    instance_id = int(row["instance_id"])
    category_apps[category].add(app_id)
    category_semantics[category].add(semantic)
    cell_counts[(category, app_id, semantic, instance_id)] += 1

  expected = {
      (category, app_id, semantic, 0)
      for category in category_apps
      for app_id in category_apps[category]
      for semantic in category_semantics[category]
  }
  observed = set(cell_counts)
  duplicate_cells = sorted(
      [list(key) + [count] for key, count in cell_counts.items() if count != 1]
  )
  return {
      "category_app_counts": {
          category: len(apps)
          for category, apps in sorted(category_apps.items())
      },
      "category_semantic_template_counts": {
          category: len(semantics)
          for category, semantics in sorted(category_semantics.items())
      },
      "app_ids_by_category": {
          category: sorted(apps)
          for category, apps in sorted(category_apps.items())
      },
      "semantic_task_ids_by_category": {
          category: sorted(semantics)
          for category, semantics in sorted(category_semantics.items())
      },
      "expected_k1_cells_from_observed_roster": len(expected),
      "observed_cells": len(observed),
      "missing_cells": [list(cell) for cell in sorted(expected - observed)],
      "unexpected_cells": [list(cell) for cell in sorted(observed - expected)],
      "duplicate_or_nonunit_cells": duplicate_cells,
      "complete_k1_cartesian_grid": (
          len(resolved) == len(rows)
          and observed == expected
          and not duplicate_cells
      ),
  }


def _summarize_artifact(
    path: Path,
    task_registry: Mapping[str, type[Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
  payload = _strict_json(path)
  metadata = payload.get("metadata")
  raw_entries = payload.get("breakdowns")
  if not isinstance(metadata, dict):
    raise ValueError(f"Artifact metadata is not an object: {path}")
  if not isinstance(raw_entries, list):
    raise ValueError(f"Artifact breakdowns is not a list: {path}")
  categories = tuple(str(item) for item in metadata.get("categories", []))
  by_package, by_app_id = _profile_indexes(categories)

  rows: list[dict[str, Any]] = []
  malformed_entries: list[int] = []
  for index, entry in enumerate(raw_entries):
    if not isinstance(entry, dict):
      malformed_entries.append(index)
      continue
    task_template = str(entry.get("task_template") or "")
    goal = str(entry.get("goal") or "")
    goal_hash = str(entry.get("goal_sha256") or "")
    try:
      instance_id = int(entry.get("instance_id", -1))
    except (TypeError, ValueError, OverflowError):
      instance_id = -1
    identity = _resolve_entry_identity(
        entry, task_registry, by_package, by_app_id
    )
    rows.append({
        "index": index,
        "task_template": task_template,
        "goal": goal,
        "goal_sha256": goal_hash,
        "key": str(entry.get("key") or ""),
        "instance_id": instance_id,
        "generator_provider": str(entry.get("generator_provider") or ""),
        "generator_model": str(entry.get("generator_model") or ""),
        "breakdown_text": str(entry.get("breakdown_text") or "").strip(),
        "validation_warning_count": len(entry.get("validation_warnings") or []),
        "repair_attempts": int(entry.get("repair_attempts") or 0),
        "has_semantic_task_id": bool(entry.get("semantic_task_id")),
        "has_semantic_goal": bool(entry.get("semantic_goal")),
        "has_plan_key": bool(entry.get("plan_key")),
        "has_plan_sha256": bool(entry.get("plan_sha256")),
        "identity": identity,
    })

  key_counts = collections.Counter(row["key"] for row in rows)
  template_counts = collections.Counter(row["task_template"] for row in rows)
  hash_counts = collections.Counter(row["goal_sha256"] for row in rows)
  recomputed_hash_mismatches = [
      row["task_template"]
      for row in rows
      if _goal_sha256(row["goal"]) != row["goal_sha256"]
  ]
  legacy_key_mismatches = [
      row["task_template"]
      for row in rows
      if row["key"]
      != f"{row['task_template']}|{row['goal_sha256']}"
  ]
  unresolved = [
      row["identity"] for row in rows if not row["identity"].get("resolved")
  ]
  app_goal_violations = [
      row["task_template"]
      for row in rows
      if not row["identity"].get("goal_mentions_resolved_app")
  ]
  resolution_counts = _count([
      str(row["identity"].get("resolution") or "") for row in rows
  ])
  non_registry_entries = [
      {
          "task_template": row["task_template"],
          "goal": row["goal"],
          "goal_sha256": row["goal_sha256"],
          "resolution": row["identity"].get("resolution"),
          "category": row["identity"].get("category"),
          "app_id": row["identity"].get("app_id"),
          "semantic_task_id": row["identity"].get("semantic_task_id"),
      }
      for row in rows
      if row["identity"].get("resolution")
      != "current_registered_task_class"
  ]
  grid = _grid_report(rows)
  entry_index = {row["key"]: row for row in rows}
  summary = {
      "path": str(path),
      "sha256": _sha256(path),
      "size_bytes": path.stat().st_size,
      "metadata": metadata,
      "entry_count": len(rows),
      "malformed_entry_indexes": malformed_entries,
      "distinct_task_templates": len(template_counts),
      "distinct_goal_sha256": len(hash_counts),
      "distinct_legacy_keys": len(key_counts),
      "duplicate_task_templates": sorted(
          key for key, count in template_counts.items() if count > 1
      ),
      "duplicate_goal_sha256": sorted(
          key for key, count in hash_counts.items() if count > 1
      ),
      "duplicate_legacy_keys": sorted(
          key for key, count in key_counts.items() if count > 1
      ),
      "recomputed_goal_sha256_mismatches": recomputed_hash_mismatches,
      "legacy_key_mismatches": legacy_key_mismatches,
      "instance_id_counts": _count([str(row["instance_id"]) for row in rows]),
      "entry_generator_provider_counts": _count([
          row["generator_provider"] for row in rows
      ]),
      "entry_generator_model_counts": _count([
          row["generator_model"] for row in rows
      ]),
      "nonempty_breakdown_count": sum(
          bool(row["breakdown_text"]) for row in rows
      ),
      "entries_with_validation_warnings": sum(
          row["validation_warning_count"] > 0 for row in rows
      ),
      "entries_with_repair_attempts": sum(
          row["repair_attempts"] > 0 for row in rows
      ),
      "semantic_pairing_field_counts": {
          "semantic_task_id": sum(row["has_semantic_task_id"] for row in rows),
          "semantic_goal": sum(row["has_semantic_goal"] for row in rows),
          "plan_key": sum(row["has_plan_key"] for row in rows),
          "plan_sha256": sum(row["has_plan_sha256"] for row in rows),
      },
      "identity_resolution_counts": resolution_counts,
      "current_registered_task_entry_count": resolution_counts.get(
          "current_registered_task_class", 0
      ),
      "non_current_registry_entries": non_registry_entries,
      "unresolved_identities": unresolved,
      "app_specific_raw_goal_count": len(rows) - len(app_goal_violations),
      "app_goal_identity_violations": app_goal_violations,
      "grid": grid,
  }
  summary["internal_identity_integrity_valid"] = all((
      not malformed_entries,
      len(rows) == len(raw_entries),
      len(template_counts) == len(rows),
      len(hash_counts) == len(rows),
      len(key_counts) == len(rows),
      not recomputed_hash_mismatches,
      not legacy_key_mismatches,
      not unresolved,
      not app_goal_violations,
      summary["nonempty_breakdown_count"] == len(rows),
  ))
  return summary, entry_index


def _log_evidence(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
  reports = []
  for path in paths:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
      reports.append({"path": str(resolved), "exists": False})
      continue
    lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    headers = [line for line in lines if line.startswith("[generator]")]
    write_lines = [line for line in lines if line.startswith("Wrote ")]
    reports.append({
        "path": str(resolved),
        "exists": True,
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
        "generator_headers": headers,
        "write_summaries": write_lines,
    })
  return reports


def _comparison(
    gpt_index: Mapping[str, dict[str, Any]],
    gemini_index: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
  gpt_keys = set(gpt_index)
  gemini_keys = set(gemini_index)
  shared = gpt_keys & gemini_keys
  shared_identity_mismatches = []
  for key in sorted(shared):
    gpt = gpt_index[key]
    gemini = gemini_index[key]
    fields = ("task_template", "goal", "goal_sha256", "instance_id")
    differing = [field for field in fields if gpt[field] != gemini[field]]
    if differing:
      shared_identity_mismatches.append({"key": key, "fields": differing})

  def compact(row: Mapping[str, Any]) -> dict[str, Any]:
    identity = row["identity"]
    return {
        "task_template": row["task_template"],
        "instance_id": row["instance_id"],
        "goal": row["goal"],
        "goal_sha256": row["goal_sha256"],
        "category": identity.get("category"),
        "app_id": identity.get("app_id"),
        "semantic_task_id": identity.get("semantic_task_id"),
    }

  return {
      "shared_exact_legacy_keys": len(shared),
      "shared_identity_mismatches": shared_identity_mismatches,
      "gpt_only_count": len(gpt_keys - gemini_keys),
      "gpt_only_entries": [
          compact(gpt_index[key]) for key in sorted(gpt_keys - gemini_keys)
      ],
      "gemini_only_count": len(gemini_keys - gpt_keys),
      "gemini_only_entries": [
          compact(gemini_index[key])
          for key in sorted(gemini_keys - gpt_keys)
      ],
      "gemini_is_exact_identity_subset_of_gpt": (
          gemini_keys < gpt_keys and not shared_identity_mismatches
      ),
  }


def _cohort_comparison(
    cohort_path: Path,
    gpt_summary: Mapping[str, Any],
) -> dict[str, Any]:
  cohort = _strict_json(cohort_path)
  legacy_grid = gpt_summary["grid"]
  legacy_apps = {
      app_id
      for apps in legacy_grid["app_ids_by_category"].values()
      for app_id in apps
  }
  frozen_apps = {
      app_id
      for spec in cohort["categories"].values()
      for app_id in spec["app_ids"]
  }
  legacy_semantics = {
      semantic
      for semantics in legacy_grid["semantic_task_ids_by_category"].values()
      for semantic in semantics
  }
  frozen_semantics = {
      semantic
      for spec in cohort["categories"].values()
      for semantic in spec["semantic_task_ids"]
  }
  old_only_apps = sorted(legacy_apps - frozen_apps)
  frozen_only_apps = sorted(frozen_apps - legacy_apps)
  old_only_semantics = sorted(legacy_semantics - frozen_semantics)
  frozen_only_semantics = sorted(frozen_semantics - legacy_semantics)
  return {
      "cohort_manifest": str(cohort_path),
      "cohort_manifest_sha256": _sha256(cohort_path),
      "cohort_release_id": cohort.get("release_id"),
      "legacy_metadata_n_task_combinations": gpt_summary["metadata"].get(
          "n_task_combinations"
      ),
      "frozen_n_task_combinations": cohort.get("n_task_combinations"),
      "legacy_app_count": len(legacy_apps),
      "frozen_app_count": len(frozen_apps),
      "legacy_entry_count": gpt_summary["entry_count"],
      "frozen_task_app_count": cohort["expected"].get("task_app_count"),
      "frozen_task_app_instance_count": cohort["expected"].get(
          "instances_per_model_condition"
      ),
      "legacy_only_app_ids": old_only_apps,
      "frozen_only_app_ids": frozen_only_apps,
      "legacy_only_semantic_task_ids": old_only_semantics,
      "frozen_only_semantic_task_ids": frozen_only_semantics,
      "legacy_roster_is_frozen_plus_two_excluded_map_apps": (
          old_only_apps == ["maps_google_maps", "maps_maps_me"]
          and not frozen_only_apps
      ),
      "semantic_roster_changed_in_two_sms_templates": (
          old_only_semantics
          == ["SmsArchiveConversation", "SmsSendClipboard"]
          and frozen_only_semantics
          == ["SmsForwardMessage", "SmsSendToContact"]
      ),
      "same_schedule_as_frozen_cohort": (
          not old_only_apps
          and not frozen_only_apps
          and not old_only_semantics
          and not frozen_only_semantics
          and gpt_summary["metadata"].get("n_task_combinations")
          == cohort.get("n_task_combinations")
      ),
  }


def audit(
    gpt_path: Path,
    gemini_path: Path,
    cohort_path: Path,
    gpt_logs: tuple[Path, ...] = DEFAULT_GPT_LOGS,
    gemini_logs: tuple[Path, ...] = DEFAULT_GEMINI_LOGS,
) -> dict[str, Any]:
  task_registry = registry.TaskRegistry().get_registry(
      family=registry.TaskRegistry.ANDROID_WORLD_FAMILY
  )
  gpt_summary, gpt_index = _summarize_artifact(gpt_path, task_registry)
  gemini_summary, gemini_index = _summarize_artifact(
      gemini_path, task_registry
  )
  comparison = _comparison(gpt_index, gemini_index)
  cohort_comparison = _cohort_comparison(cohort_path, gpt_summary)
  required_pairing_fields = (
      "semantic_task_id", "semantic_goal", "plan_key", "plan_sha256"
  )
  legacy_pairing_fields_complete = all(
      summary["semantic_pairing_field_counts"][field]
      == summary["entry_count"]
      for summary in (gpt_summary, gemini_summary)
      for field in required_pairing_fields
  )

  source_paths = [
      Path(__file__).resolve(),
      BENCHMARK_ROOT / "scripts" / "generate_task_breakdowns.py",
      BENCHMARK_ROOT / "app_generalization_profiles.py",
      BENCHMARK_ROOT / "android_world" / "registry.py",
      BENCHMARK_ROOT / "android_world" / "suite_utils.py",
      BENCHMARK_ROOT / "task_breakdowns.py",
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
      for category in ("sms", "files", "maps", "contacts", "clock")
  )
  report = {
      "audit_type": (
          "forensic_legacy_c2_plan_provenance_not_revised_c2_evidence"
      ),
      "audit_provenance": {
          "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
          "python_version": platform.python_version(),
          "repository_root": str(REPO_ROOT.resolve()),
          "script_path": str(Path(__file__).resolve()),
          "script_sha256": _sha256(Path(__file__).resolve()),
          "source_files": [
              {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
              for path in source_paths
          ],
          "source_code_scope": (
              "Current working-tree reference only; not an attestation of the "
              "historical generator source used in May 2026."
          ),
      },
      "gpt54": gpt_summary,
      "gemini31": gemini_summary,
      "artifact_comparison": comparison,
      "frozen_cohort_comparison": cohort_comparison,
      "generation_log_evidence": {
          "gpt54": _log_evidence(gpt_logs),
          "gemini31": _log_evidence(gemini_logs),
      },
      "historical_generation_attestation": {
          "fully_attested": False,
          "available": [
              "artifact bytes and SHA-256",
              "provider/model metadata",
              "prompt-template SHA-256 metadata",
              "suite family, categories, K, seed, and fixed-seed metadata",
              "strict-validator and final-count statements in saved logs",
          ],
          "not_available_in_artifact_or_logs": [
              "historical generator script SHA-256",
              "historical git commit",
              "complete CLI argv",
              "raw planner requests and responses",
              "provider request IDs",
          ],
      },
      "checked_conclusions": {
          "gpt54_complete_old_25_app_k1_grid": (
              gpt_summary["entry_count"] == 250
              and gpt_summary["grid"]["complete_k1_cartesian_grid"]
              and gpt_summary["internal_identity_integrity_valid"]
          ),
          "gpt54_has_250_distinct_app_specific_raw_goals": (
              gpt_summary["distinct_goal_sha256"] == 250
              and gpt_summary["app_specific_raw_goal_count"] == 250
          ),
          "legacy_files_implement_semantic_plan_reuse": (
              legacy_pairing_fields_complete
          ),
          "legacy_files_valid_for_frozen_revised_c2": (
              legacy_pairing_fields_complete
              and cohort_comparison["same_schedule_as_frozen_cohort"]
          ),
          "gemini_has_242_entries_and_8_gpt_identities_missing": (
              gemini_summary["entry_count"] == 242
              and comparison["gpt_only_count"] == 8
              and comparison["gemini_only_count"] == 0
          ),
      },
  }
  return report


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--gpt_file", default=str(DEFAULT_GPT_FILE))
  parser.add_argument("--gemini_file", default=str(DEFAULT_GEMINI_FILE))
  parser.add_argument("--cohort_manifest", default=str(DEFAULT_COHORT))
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  report = audit(
      Path(args.gpt_file).expanduser().resolve(),
      Path(args.gemini_file).expanduser().resolve(),
      Path(args.cohort_manifest).expanduser().resolve(),
  )
  output = Path(args.output).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(
      json.dumps(report, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  print(json.dumps(report["checked_conclusions"], indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
