#!/usr/bin/env python3
"""Shared frozen-cohort helpers for CATBench release preflights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from app_generalization_profiles import get_domain_profiles


def load(path: str | Path) -> dict[str, Any]:
  cohort_path = Path(path).expanduser().resolve()
  payload = json.loads(cohort_path.read_text(encoding="utf-8"))
  required = {
      "release_id",
      "suite_family",
      "task_random_seed",
      "n_task_combinations",
      "categories",
      "expected",
  }
  missing = sorted(required - set(payload))
  if missing:
    raise ValueError(f"Frozen cohort missing fields: {missing}")
  payload["_path"] = str(cohort_path)
  return payload


def validate_schedule_args(
    cohort: Mapping[str, Any],
    *,
    suite_family: str,
    categories: tuple[str, ...],
    n_task_combinations: int,
    task_random_seed: int,
    fixed_task_seed: bool,
) -> list[str]:
  issues: list[str] = []
  if suite_family != cohort["suite_family"]:
    issues.append(
        f"suite_family={suite_family!r}, expected {cohort['suite_family']!r}"
    )
  expected_categories = tuple(cohort["categories"])
  if categories != expected_categories:
    issues.append(
        f"categories={list(categories)!r}, expected {list(expected_categories)!r}"
    )
  if n_task_combinations != int(cohort["n_task_combinations"]):
    issues.append(
        f"n_task_combinations={n_task_combinations}, expected "
        f"{cohort['n_task_combinations']}"
    )
  if task_random_seed != int(cohort["task_random_seed"]):
    issues.append(
        f"task_random_seed={task_random_seed}, expected "
        f"{cohort['task_random_seed']}"
    )
  if fixed_task_seed:
    issues.append("fixed_task_seed must be false for the K=3 primary cohort")
  return issues


def frozen_task_names(
    cohort: Mapping[str, Any],
    task_registry: Mapping[str, type[Any]],
) -> tuple[list[str], dict[str, tuple[str, str]]]:
  """Resolves every frozen real app/semantic task to one registered class.

  This intentionally resolves classes listed in the frozen cohort even when a
  profile has temporarily descheduled an app. It permits plan preparation and
  cohort auditing, but does not make that app eligible for model evaluation.
  """
  profiles = get_domain_profiles()
  names: list[str] = []
  identities: dict[str, tuple[str, str]] = {}
  for category, spec in cohort["categories"].items():
    apps = {app.app_id: app for app in profiles[category].apps}
    for app_id in spec["app_ids"]:
      app = apps.get(app_id)
      if app is None:
        raise ValueError(f"Frozen app missing from profile: {category}/{app_id}")
      for semantic_task_id in spec["semantic_task_ids"]:
        candidates = []
        for task_name, task_type in task_registry.items():
          if str(getattr(task_type, "catbench_semantic_id", "")) != str(
              semantic_task_id
          ):
            continue
          package_name = str(getattr(task_type, "package_name", "") or "")
          if package_name != str(app.package_name or ""):
            continue
          candidates.append(task_name)
        if len(candidates) != 1:
          raise ValueError(
              f"Expected one real task class for {category}/{app_id}/"
              f"{semantic_task_id}; found {sorted(candidates)}"
          )
        task_name = candidates[0]
        names.append(task_name)
        identities[task_name] = (category, app_id)
  expected_count = int(cohort["expected"]["task_app_count"])
  if len(names) != expected_count or len(set(names)) != expected_count:
    raise ValueError(
        f"Frozen class roster has {len(names)} entries/"
        f"{len(set(names))} unique; expected {expected_count}"
    )
  return names, identities
