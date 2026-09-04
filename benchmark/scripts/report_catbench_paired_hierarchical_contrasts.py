#!/usr/bin/env python3
"""Strict paired hierarchical contrasts for CATBench C1/C2-G/C2-O.

This reporter compares direct execution (C1) with generated-plan assistance
(C2-G) and human-reference-plan assistance (C2-O, whose historical identifier
does not imply a perfect oracle).  A pair is eligible only when its
model, category, app, semantic template, parameter-instance index, and frozen
run provenance match exactly.  It never pairs rows by file order.
For the frozen primary release, all three conditions are loaded from one
committed whole-triplet selection state, so replacement histories cannot
create duplicate cells or mix conditions from different rounds.

Point estimates and bootstrap draws use the same hierarchy:

* instances within semantic templates;
* semantic templates within apps;
* apps within categories; and
* categories within models.

All bootstrap sampling preserves the C1/C2 outcome pair.  The signed contrast
is always C2 - C1.  The default is 10,000 deterministic percentile-bootstrap
draws.  A frozen cohort manifest is optional, but when supplied it defines the
scheduled Cartesian set rather than allowing the observed artifacts to define
their own denominator.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from pair_baseline_and_breakdown import (  # noqa: E402
    Harvest,
    SelectedTripletValidationError,
    SlotKey,
    _harvest,
    _harvest_selected_triplets,
    _mcnemar_exact_p,
    _pair_mismatch_reasons,
    _plan_reuse_validation,
    _primary_harvest_validation,
)
from report_catbench_hierarchical_metrics import (  # noqa: E402
    DEFAULT_AW_APPS,
    _ci,
    _matrix_issues,
    _merge_harvests,
    _normalize_rows,
)


CONDITIONS = ("c1", "c2_g", "c2_o")
CONTRASTS = (("c1", "c2_g"), ("c1", "c2_o"))
SemanticKey = tuple[str, str, str, str, int]


class PairedContrastValidationError(ValueError):
  """Raised when a report is ineligible for strict primary use."""

  def __init__(self, report: dict[str, Any]):
    super().__init__("CATBench paired hierarchical contrast gate failed")
    self.report = report


def _semantic_key(row: Mapping[str, Any]) -> SemanticKey:
  return (
      str(row["model"]),
      str(row["category"]),
      str(row["app_id"]),
      str(row["semantic_task_id"]),
      int(row["instance_id"]),
  )


def _semantic_index(
    harvest: Harvest,
) -> tuple[dict[SemanticKey, dict[str, Any]], list[dict[str, Any]]]:
  grouped: dict[SemanticKey, list[dict[str, Any]]] = collections.defaultdict(list)
  issues: list[dict[str, Any]] = []
  for row in harvest.rows.values():
    try:
      grouped[_semantic_key(row)].append(row)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
      issues.append({
          "type": "invalid_semantic_identity_in_valid_harvest_row",
          "pkl_path": row.get("pkl_path", ""),
          "detail": str(exc),
      })
  index: dict[SemanticKey, dict[str, Any]] = {}
  for key, candidates in sorted(grouped.items()):
    if len(candidates) != 1:
      issues.append({
          "type": "duplicate_semantic_cell",
          "semantic_key": list(key),
          "count": len(candidates),
      })
    else:
      index[key] = candidates[0]
  return index, issues


def expected_slots_from_cohort(cohort: Mapping[str, Any]) -> set[SemanticKey]:
  """Expand either the frozen Cartesian cohort or an explicit slot list."""
  if isinstance(cohort.get("slots"), list):
    expected: set[SemanticKey] = set()
    for raw_slot in cohort["slots"]:
      if isinstance(raw_slot, Mapping):
        key = (
            str(raw_slot["model"]),
            str(raw_slot["category"]),
            str(raw_slot["app_id"]),
            str(raw_slot["semantic_task_id"]),
            int(raw_slot["instance_id"]),
        )
      elif isinstance(raw_slot, (list, tuple)) and len(raw_slot) == 5:
        key = (
            str(raw_slot[0]),
            str(raw_slot[1]),
            str(raw_slot[2]),
            str(raw_slot[3]),
            int(raw_slot[4]),
        )
      else:
        raise ValueError(f"Invalid frozen cohort slot: {raw_slot!r}")
      if key in expected:
        raise ValueError(f"Duplicate frozen cohort slot: {key!r}")
      expected.add(key)
    if not expected:
      raise ValueError("Frozen cohort slots must not be empty")
    return expected

  models = cohort.get("models")
  categories = cohort.get("categories")
  combinations = cohort.get("n_task_combinations")
  if (
      not isinstance(models, list)
      or not isinstance(categories, Mapping)
      or combinations is None
  ):
    raise ValueError(
        "Frozen cohort needs slots[] or models/categories/n_task_combinations"
    )
  n_instances = int(combinations)
  if n_instances < 1:
    raise ValueError("n_task_combinations must be positive")
  expected = set()
  for model in models:
    for category, spec in categories.items():
      if not isinstance(spec, Mapping):
        raise ValueError(f"Invalid category specification: {category!r}")
      for app_id in spec.get("app_ids", []):
        for semantic_task_id in spec.get("semantic_task_ids", []):
          for instance_id in range(n_instances):
            expected.add((
                str(model),
                str(category),
                str(app_id),
                str(semantic_task_id),
                instance_id,
            ))
  if not expected:
    raise ValueError("Frozen Cartesian cohort expands to zero slots")
  return expected


def _schedule_sha256(expected: Iterable[SemanticKey]) -> str:
  payload = json.dumps(
      [list(key) for key in sorted(expected)],
      ensure_ascii=False,
      separators=(",", ":"),
  )
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record_is_episode_attempt(record: Mapping[str, Any]) -> bool:
  if record.get("slot") is not None:
    return True
  return bool(record.get("pkl_path")) and record.get("episode_index") is not None


def _invalid_record_semantic_key(
    record: Mapping[str, Any],
    legacy_to_semantic: Mapping[SlotKey, SemanticKey],
) -> SemanticKey | None:
  if all(
      record.get(field) is not None
      for field in (
          "model",
          "category",
          "app_id",
          "semantic_task_id",
          "instance_id",
      )
  ):
    try:
      return (
          str(record["model"]),
          str(record["category"]),
          str(record["app_id"]),
          str(record["semantic_task_id"]),
          int(record["instance_id"]),
      )
    except (TypeError, ValueError, OverflowError):
      return None
  slot = record.get("slot")
  if isinstance(slot, (list, tuple)) and len(slot) == 5:
    legacy = (
        str(slot[0]),
        str(slot[1]),
        str(slot[2]),
        str(slot[3]),
        int(slot[4]),
    )
    return legacy_to_semantic.get(legacy)
  return None


def _raw_condition_counts(
    raw_harvests: Sequence[Harvest],
    selected_index: Mapping[SemanticKey, Mapping[str, Any]],
    expected: set[SemanticKey],
    legacy_to_semantic: Mapping[SlotKey, SemanticKey],
) -> dict[str, Any]:
  attempts_by_recorded_slot: collections.Counter[SlotKey] = collections.Counter()
  semantic_attempted: set[SemanticKey] = set()
  invalid_semantic_cells: set[SemanticKey] = set()
  invalid_recorded_slots: set[SlotKey] = set()
  invalid_attempts = 0
  invalid_attempts_linked_to_scheduled_cells = 0
  unlinked_invalid_attempts = 0
  non_episode_artifact_errors = 0
  valid_attempts_observed = 0

  for harvest in raw_harvests:
    for legacy_slot, row in harvest.rows.items():
      attempts_by_recorded_slot[legacy_slot] += 1
      valid_attempts_observed += 1
      try:
        semantic_attempted.add(_semantic_key(row))
      except (KeyError, TypeError, ValueError, OverflowError):
        pass
    for record in harvest.invalid_records:
      if not _record_is_episode_attempt(record):
        non_episode_artifact_errors += 1
        continue
      invalid_attempts += 1
      slot = record.get("slot")
      if isinstance(slot, (list, tuple)) and len(slot) == 5:
        legacy_slot = (
            str(slot[0]),
            str(slot[1]),
            str(slot[2]),
            str(slot[3]),
            int(slot[4]),
        )
        attempts_by_recorded_slot[legacy_slot] += 1
        invalid_recorded_slots.add(legacy_slot)
      semantic_key = _invalid_record_semantic_key(
          record, legacy_to_semantic
      )
      if semantic_key is None:
        unlinked_invalid_attempts += 1
      else:
        semantic_attempted.add(semantic_key)
        invalid_semantic_cells.add(semantic_key)
        if semantic_key in expected:
          invalid_attempts_linked_to_scheduled_cells += 1

  selected_keys = set(selected_index)
  selected_rows = [
      row for key, row in selected_index.items() if key in expected
  ]
  valid_successes = sum(bool(row["is_successful"]) for row in selected_rows)
  replacements = sum(
      max(0, attempt_count - 1)
      for attempt_count in attempts_by_recorded_slot.values()
  )
  return {
      "scheduled_cells": len(expected),
      "selected_valid_outcome_cells": len(selected_keys & expected),
      "valid_successes": valid_successes,
      "valid_failures": len(selected_rows) - valid_successes,
      "valid_attempts_observed": valid_attempts_observed,
      "infrastructure_or_harvest_invalid_attempts": invalid_attempts,
      "invalid_attempts_linked_to_scheduled_cells": (
          invalid_attempts_linked_to_scheduled_cells
      ),
      "invalid_unique_recorded_cells": len(invalid_recorded_slots),
      "invalid_unique_scheduled_cells": len(invalid_semantic_cells & expected),
      "missing_valid_outcome_cells": len(expected - selected_keys),
      "missing_terminal_artifact_cells": len(expected - semantic_attempted),
      "extra_valid_outcome_cells": len(selected_keys - expected),
      "replacement_attempts": replacements,
      "unlinked_invalid_attempts": unlinked_invalid_attempts,
      "non_episode_artifact_errors": non_episode_artifact_errors,
      "replacement_definition": (
          "episode attempts beyond the first artifact for a recorded "
          "model/category/app/task-template/instance slot"
      ),
  }


def _transition_counts(
    pairs: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
  pass_pass = 0
  pass_fail = 0
  fail_pass = 0
  fail_fail = 0
  for c1_row, c2_row in pairs:
    c1_success = bool(c1_row["is_successful"])
    c2_success = bool(c2_row["is_successful"])
    if c1_success and c2_success:
      pass_pass += 1
    elif c1_success:
      pass_fail += 1
    elif c2_success:
      fail_pass += 1
    else:
      fail_fail += 1
  total = pass_pass + pass_fail + fail_pass + fail_fail
  return {
      "exact_valid_pairs": total,
      "pass_pass": pass_pass,
      "c1_pass_c2_fail": pass_fail,
      "c1_fail_c2_pass": fail_pass,
      "fail_fail": fail_fail,
      "plan_harmed_rate": pass_fail / total if total else None,
      "planning_responsive_rate": fail_pass / total if total else None,
      "mcnemar_exact_p_two_sided": _mcnemar_exact_p(fail_pass, pass_fail),
  }


def _hierarchical_point(
    pair_index: Mapping[
        SemanticKey, tuple[Mapping[str, Any], Mapping[str, Any]]
    ],
) -> dict[str, Any]:
  by_model: dict[str, list[SemanticKey]] = collections.defaultdict(list)
  for key in pair_index:
    by_model[key[0]].append(key)
  output: dict[str, Any] = {}
  for model, model_keys in sorted(by_model.items()):
    by_category: dict[str, list[SemanticKey]] = collections.defaultdict(list)
    for key in model_keys:
      by_category[key[1]].append(key)
    category_reports: dict[str, Any] = {}
    for category, category_keys in sorted(by_category.items()):
      by_app: dict[str, list[SemanticKey]] = collections.defaultdict(list)
      for key in category_keys:
        by_app[key[2]].append(key)
      app_reports: dict[str, Any] = {}
      for app_id, app_keys in sorted(by_app.items()):
        by_template: dict[str, list[SemanticKey]] = collections.defaultdict(list)
        for key in app_keys:
          by_template[key[3]].append(key)
        c1_template_rates: list[float] = []
        c2_template_rates: list[float] = []
        for template_keys in by_template.values():
          c1_template_rates.append(statistics.fmean(
              bool(pair_index[key][0]["is_successful"])
              for key in template_keys
          ))
          c2_template_rates.append(statistics.fmean(
              bool(pair_index[key][1]["is_successful"])
              for key in template_keys
          ))
        c1_sr = statistics.fmean(c1_template_rates)
        c2_sr = statistics.fmean(c2_template_rates)
        app_pairs = [pair_index[key] for key in app_keys]
        app_reports[app_id] = {
            "c1_sr": c1_sr,
            "c2_sr": c2_sr,
            "delta_c2_minus_c1": c2_sr - c1_sr,
            "semantic_templates": len(by_template),
            "raw": _transition_counts(app_pairs),
        }
      c1_sr = statistics.fmean(
          report["c1_sr"] for report in app_reports.values()
      )
      c2_sr = statistics.fmean(
          report["c2_sr"] for report in app_reports.values()
      )
      category_reports[category] = {
          "c1_sr": c1_sr,
          "c2_sr": c2_sr,
          "delta_c2_minus_c1": c2_sr - c1_sr,
          "apps": app_reports,
          "raw": _transition_counts(
              pair_index[key] for key in category_keys
          ),
      }
    overall_c1 = statistics.fmean(
        report["c1_sr"] for report in category_reports.values()
    )
    overall_c2 = statistics.fmean(
        report["c2_sr"] for report in category_reports.values()
    )
    output[model] = {
        "categories": category_reports,
        "overall": {
            "c1_sr": overall_c1,
            "c2_sr": overall_c2,
            "delta_c2_minus_c1": overall_c2 - overall_c1,
            "categories": len(category_reports),
            "raw": _transition_counts(
                pair_index[key] for key in model_keys
            ),
        },
    }
  return output


def _nested_pairs_for_model(
    pair_index: Mapping[
        SemanticKey, tuple[Mapping[str, Any], Mapping[str, Any]]
    ],
    model: str,
) -> dict[str, dict[str, dict[str, list[tuple[int, int, int]]]]]:
  nested: dict[
      str, dict[str, dict[str, list[tuple[int, int, int]]]]
  ] = collections.defaultdict(
      lambda: collections.defaultdict(lambda: collections.defaultdict(list))
  )
  for key, (c1_row, c2_row) in pair_index.items():
    if key[0] != model:
      continue
    nested[key[1]][key[2]][key[3]].append((
        key[4],
        int(bool(c1_row["is_successful"])),
        int(bool(c2_row["is_successful"])),
    ))
  return {
      category: {
          app: {template: list(values) for template, values in templates.items()}
          for app, templates in apps.items()
      }
      for category, apps in nested.items()
  }


def _draw_app(
    templates: Mapping[str, Sequence[tuple[int, int, int]]],
    rng: random.Random,
) -> tuple[float, float, float]:
  names = sorted(templates)
  sampled_names = [rng.choice(names) for _ in names]
  c1_template_rates: list[float] = []
  c2_template_rates: list[float] = []
  for name in sampled_names:
    instances = sorted(templates[name])
    sampled = [rng.choice(instances) for _ in instances]
    c1_template_rates.append(statistics.fmean(item[1] for item in sampled))
    c2_template_rates.append(statistics.fmean(item[2] for item in sampled))
  c1_sr = statistics.fmean(c1_template_rates)
  c2_sr = statistics.fmean(c2_template_rates)
  return c1_sr, c2_sr, c2_sr - c1_sr


def _draw_category(
    apps: Mapping[str, Mapping[str, Sequence[tuple[int, int, int]]]],
    rng: random.Random,
) -> tuple[float, float, float]:
  app_ids = sorted(apps)
  sampled_apps = [rng.choice(app_ids) for _ in app_ids]
  app_draws = [_draw_app(apps[app_id], rng) for app_id in sampled_apps]
  c1_sr = statistics.fmean(draw[0] for draw in app_draws)
  c2_sr = statistics.fmean(draw[1] for draw in app_draws)
  return c1_sr, c2_sr, c2_sr - c1_sr


def _stable_draw_seed(base_seed: int, model: str, contrast: str) -> int:
  digest = hashlib.sha256(
      f"{base_seed}|{model}|{contrast}".encode("utf-8")
  ).digest()
  return int.from_bytes(digest[:8], "big")


def _bootstrap_model_contrast(
    nested: Mapping[
        str, Mapping[str, Mapping[str, Sequence[tuple[int, int, int]]]]
    ],
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]]]:
  categories = sorted(nested)
  if not categories:
    raise ValueError("Cannot bootstrap an empty exact-pair cohort")
  rng = random.Random(seed)
  metric_names = ("c1_sr", "c2_sr", "delta_c2_minus_c1")
  overall_draws = {metric: [] for metric in metric_names}
  category_draws = {
      category: {metric: [] for metric in metric_names}
      for category in categories
  }
  for _ in range(replicates):
    for category in categories:
      draw = _draw_category(nested[category], rng)
      for metric, value in zip(metric_names, draw):
        category_draws[category][metric].append(value)

    sampled_categories = [rng.choice(categories) for _ in categories]
    draws = [_draw_category(nested[category], rng) for category in sampled_categories]
    for metric_index, metric in enumerate(metric_names):
      overall_draws[metric].append(statistics.fmean(
          draw[metric_index] for draw in draws
      ))
  return (
      {metric: _ci(values, confidence) for metric, values in overall_draws.items()},
      {
          category: {
              metric: _ci(values, confidence)
              for metric, values in metrics.items()
          }
          for category, metrics in category_draws.items()
      },
  )


def _pair_conditions(
    c1_index: Mapping[SemanticKey, dict[str, Any]],
    c2_index: Mapping[SemanticKey, dict[str, Any]],
) -> tuple[
    dict[SemanticKey, tuple[dict[str, Any], dict[str, Any]]],
    dict[str, Any],
]:
  c1_keys = set(c1_index)
  c2_keys = set(c2_index)
  paired: dict[SemanticKey, tuple[dict[str, Any], dict[str, Any]]] = {}
  mismatches: list[dict[str, Any]] = []
  for key in sorted(c1_keys & c2_keys):
    reasons = _pair_mismatch_reasons(c1_index[key], c2_index[key])
    if reasons:
      mismatches.append({
          "semantic_key": list(key),
          "reasons": reasons,
          "c1_pkl_path": c1_index[key].get("pkl_path", ""),
          "c2_pkl_path": c2_index[key].get("pkl_path", ""),
      })
    else:
      paired[key] = (c1_index[key], c2_index[key])
  return paired, {
      "c1_only": [list(key) for key in sorted(c1_keys - c2_keys)],
      "c2_only": [list(key) for key in sorted(c2_keys - c1_keys)],
      "provenance_mismatches": mismatches,
  }


def _legacy_semantic_lookup(
    raw: Mapping[str, Sequence[Harvest]],
) -> dict[SlotKey, SemanticKey]:
  candidates: dict[SlotKey, set[SemanticKey]] = collections.defaultdict(set)
  for harvests in raw.values():
    for harvest in harvests:
      for legacy_slot, row in harvest.rows.items():
        try:
          candidates[legacy_slot].add(_semantic_key(row))
        except (KeyError, TypeError, ValueError, OverflowError):
          pass
  return {
      legacy: next(iter(semantic_keys))
      for legacy, semantic_keys in candidates.items()
      if len(semantic_keys) == 1
  }


def analyze_harvests(
    condition_harvests: Mapping[str, Sequence[Harvest]],
    *,
    frozen_cohort: Mapping[str, Any] | None = None,
    expected_slots: Iterable[SemanticKey] | None = None,
    allow_incomplete: bool = False,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 1729,
    confidence: float = 0.95,
) -> dict[str, Any]:
  """Audit and compare in-memory strict harvests for all three conditions."""
  if bootstrap_replicates < 1:
    raise ValueError("bootstrap_replicates must be positive")
  if not 0.0 < confidence < 1.0:
    raise ValueError("confidence must be between zero and one")
  if frozen_cohort is not None and expected_slots is not None:
    raise ValueError("Use frozen_cohort or expected_slots, not both")
  for condition in CONDITIONS:
    if condition not in condition_harvests or not condition_harvests[condition]:
      raise ValueError(f"Missing harvests for required condition {condition}")

  raw = {condition: list(condition_harvests[condition]) for condition in CONDITIONS}
  merged = {
      condition: _merge_harvests(raw[condition]) for condition in CONDITIONS
  }
  indexes: dict[str, dict[SemanticKey, dict[str, Any]]] = {}
  issues: list[dict[str, Any]] = []
  for condition in CONDITIONS:
    index, semantic_issues = _semantic_index(merged[condition])
    indexes[condition] = index
    issues.extend({"condition": condition, **issue} for issue in semantic_issues)
    if merged[condition].invalid_records:
      issues.append({
          "type": "condition_has_invalid_harvest_artifacts",
          "condition": condition,
          "count": len(merged[condition].invalid_records),
      })
    if merged[condition].duplicate_roster:
      issues.append({
          "type": "condition_has_duplicate_roster_cells",
          "condition": condition,
          "cells": [
              list(cell) for cell in sorted(merged[condition].duplicate_roster)
          ],
      })
    cells, row_issues = _normalize_rows(index.values())
    issues.extend({
        "condition": condition,
        **issue,
    } for issue in row_issues)
    issues.extend({
        "condition": condition,
        **issue,
    } for issue in _matrix_issues(
        cells, DEFAULT_AW_APPS, merged[condition].roster
    ))

  if frozen_cohort is not None:
    expected = expected_slots_from_cohort(frozen_cohort)
    declared_conditions = set(frozen_cohort.get("conditions", CONDITIONS))
    missing_conditions = sorted(set(CONDITIONS) - declared_conditions)
    if missing_conditions:
      issues.append({
          "type": "frozen_cohort_missing_conditions",
          "conditions": missing_conditions,
      })
    if {
        "models", "categories", "n_task_combinations"
    }.issubset(frozen_cohort):
      for condition in CONDITIONS:
        validation = _primary_harvest_validation(
            merged[condition], dict(frozen_cohort), condition
        )
        if not validation["valid"]:
          issues.append({
              "type": "frozen_cohort_condition_mismatch",
              "condition": condition,
              "detail": validation,
          })
    schedule_basis = "frozen_cohort_manifest"
  elif expected_slots is not None:
    expected = set(expected_slots)
    schedule_basis = "explicit_in_memory_expected_slots"
  else:
    expected = set().union(*(set(index) for index in indexes.values()))
    schedule_basis = "observed_union_no_universal_missing_detection"
  if not expected:
    issues.append({"type": "empty_scheduled_cohort"})

  for condition, index in indexes.items():
    missing = sorted(expected - set(index))
    extra = sorted(set(index) - expected)
    if missing:
      issues.append({
          "type": "condition_missing_scheduled_valid_cells",
          "condition": condition,
          "count": len(missing),
          "cells": [list(key) for key in missing],
      })
    if extra:
      issues.append({
          "type": "condition_has_unscheduled_valid_cells",
          "condition": condition,
          "count": len(extra),
          "cells": [list(key) for key in extra],
      })

  for condition in ("c2_g", "c2_o"):
    plan_reuse = _plan_reuse_validation(merged[condition])
    if not plan_reuse["valid"]:
      issues.append({
          "type": "condition_plan_not_reused_across_apps",
          "condition": condition,
          "detail": plan_reuse,
      })

  legacy_to_semantic = _legacy_semantic_lookup(raw)
  condition_audits = {
      condition: _raw_condition_counts(
          raw[condition], indexes[condition], expected, legacy_to_semantic
      )
      for condition in CONDITIONS
  }

  contrasts: dict[str, Any] = {}
  for c1_condition, c2_condition in CONTRASTS:
    label = f"{c1_condition}_vs_{c2_condition}"
    pair_index, pairing_audit = _pair_conditions(
        indexes[c1_condition], indexes[c2_condition]
    )
    pair_keys = set(pair_index)
    missing_exact_pairs = sorted(expected - pair_keys)
    if pairing_audit["provenance_mismatches"]:
      issues.append({
          "type": "paired_provenance_mismatch",
          "contrast": label,
          "count": len(pairing_audit["provenance_mismatches"]),
      })
    if missing_exact_pairs:
      issues.append({
          "type": "contrast_missing_exact_pairs",
          "contrast": label,
          "count": len(missing_exact_pairs),
      })

    models = _hierarchical_point(pair_index)
    for model, model_report in models.items():
      draw_seed = _stable_draw_seed(bootstrap_seed, model, label)
      overall_ci, category_ci = _bootstrap_model_contrast(
          _nested_pairs_for_model(pair_index, model),
          replicates=bootstrap_replicates,
          seed=draw_seed,
          confidence=confidence,
      )
      model_report["overall"]["bootstrap_ci"] = overall_ci
      model_report["overall"]["draw_seed"] = draw_seed
      for category, intervals in category_ci.items():
        model_report["categories"][category]["bootstrap_ci"] = intervals
    contrasts[label] = {
        "signed_contrast": f"{c2_condition} - {c1_condition}",
        "scheduled_cells": len(expected),
        "exact_valid_pairs": len(pair_index),
        "missing_exact_pair_cells": len(missing_exact_pairs),
        "pairing_audit": pairing_audit,
        "raw": _transition_counts(pair_index.values()),
        "models": models,
    }

  report = {
      "schema_version": 1,
      "strictly_valid": not issues,
      "allow_incomplete": allow_incomplete,
      "conditions": list(CONDITIONS),
      "hierarchy": [
          "instances_within_semantic_template",
          "semantic_templates_within_app",
          "apps_within_category",
          "categories_within_model",
      ],
      "schedule": {
          "basis": schedule_basis,
          "scheduled_cells_per_condition": len(expected),
          "sha256": _schedule_sha256(expected),
      },
      "bootstrap": {
          "method": "paired hierarchical percentile bootstrap",
          "with_replacement": True,
          "resampling_levels": [
              "categories",
              "apps_within_selected_category",
              "semantic_templates_within_selected_app",
              "paired_instances_within_selected_template",
          ],
          "replicates": bootstrap_replicates,
          "base_seed": bootstrap_seed,
          "model_contrast_seed_derivation": (
              "uint64(sha256(base_seed|model|contrast)[:8])"
          ),
          "confidence": confidence,
          "interval": "percentile",
      },
      "raw_condition_audits": condition_audits,
      "contrasts": contrasts,
      "interpretation": {
          "c1_fail_c2_pass": "planning-responsive under the supplied plan",
          "c1_pass_c2_fail": "plan-harmed/treatment regression",
          "c1_fail_c2_fail": "unresolved; not a grounding diagnosis",
      },
      "validation": {"n_issues": len(issues), "issues": issues},
  }
  if issues and not allow_incomplete:
    raise PairedContrastValidationError(report)
  return report


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
  confidence_label = f"{100 * report['bootstrap']['confidence']:g}% CI"
  lines = [
      "# CATBench paired hierarchical C1/C2 contrasts",
      "",
      f"Strict validity: **{'PASS' if report['strictly_valid'] else 'FAIL'}**",
      "",
      "The signed contrast is C2 − C1. Scores average paired instances → "
      "semantic templates → apps → categories.",
      "",
      "## Artifact accounting",
      "",
      "| Condition | Scheduled | Valid | Success | Failure | Invalid attempts | "
      "Invalid cells | No artifact | Missing valid | Replacements |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for condition in CONDITIONS:
    audit = report["raw_condition_audits"][condition]
    lines.append(
        f"| {condition} | {audit['scheduled_cells']} | "
        f"{audit['selected_valid_outcome_cells']} | "
        f"{audit['valid_successes']} | {audit['valid_failures']} | "
        f"{audit['infrastructure_or_harvest_invalid_attempts']} | "
        f"{audit['invalid_unique_scheduled_cells']} | "
        f"{audit['missing_terminal_artifact_cells']} | "
        f"{audit['missing_valid_outcome_cells']} | "
        f"{audit['replacement_attempts']} |"
    )
  for contrast, contrast_report in report["contrasts"].items():
    lines.extend([
        "",
        f"## {contrast}",
        "",
        f"| Model | n | C1 SR | C2 SR | C2−C1 | {confidence_label} | "
        "responsive | harmed | exact p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model, model_report in sorted(contrast_report["models"].items()):
      overall = model_report["overall"]
      raw = overall["raw"]
      interval = overall["bootstrap_ci"]["delta_c2_minus_c1"]
      lines.append(
          f"| {model} | {raw['exact_valid_pairs']} | "
          f"{overall['c1_sr']:.3f} | {overall['c2_sr']:.3f} | "
          f"{overall['delta_c2_minus_c1']:+.3f} | "
          f"[{interval[0]:+.3f}, {interval[1]:+.3f}] | "
          f"{raw['c1_fail_c2_pass']} | {raw['c1_pass_c2_fail']} | "
          f"{raw['mcnemar_exact_p_two_sided']} |"
      )
  if report["validation"]["issues"]:
    lines.extend(["", "## Validation issues", ""])
    for issue in report["validation"]["issues"]:
      lines.append(f"- `{json.dumps(issue, sort_keys=True)}`")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  for condition in CONDITIONS:
    parser.add_argument(
        f"--{condition}_manifest",
        action="append",
        default=[],
        help=f"{condition} matrix manifest; repeat for disjoint shards.",
    )
  parser.add_argument(
      "--frozen_cohort_manifest",
      "--primary_cohort_manifest",
      dest="frozen_cohort_manifest",
      default="",
      help="Optional frozen semantic cohort JSON.",
  )
  parser.add_argument(
      "--selected_triplets",
      default="",
      help=(
          "Committed schedule-consumer selected_triplets.jsonl. Required "
          "with the frozen primary cohort."
      ),
  )
  parser.add_argument("--allow_incomplete", action="store_true")
  parser.add_argument("--bootstrap_replicates", type=int, default=10000)
  parser.add_argument("--bootstrap_seed", type=int, default=1729)
  parser.add_argument("--confidence", type=float, default=0.95)
  parser.add_argument("--out_dir", required=True)
  args = parser.parse_args(argv)

  frozen_cohort = None
  frozen_cohort_path = None
  if args.frozen_cohort_manifest:
    frozen_cohort_path = Path(
        args.frozen_cohort_manifest
    ).expanduser().absolute()
    frozen_cohort = json.loads(
        frozen_cohort_path.read_text(encoding="utf-8")
    )
  ordinary_inputs = any(
      getattr(args, f"{condition}_manifest") for condition in CONDITIONS
  )
  selection_audit = None
  if frozen_cohort is not None:
    if args.allow_incomplete:
      parser.error(
          "--allow_incomplete is prohibited for primary paired inference; "
          "pending/exhausted triplets require a separate predeclared "
          "attrition-bounds analysis"
      )
    if not args.selected_triplets:
      parser.error(
          "--selected_triplets is required for frozen primary reporting"
      )
    if ordinary_inputs:
      parser.error(
          "Frozen primary reporting ingests only committed selected triplets; "
          "condition manifests are legacy/development inputs"
      )
    try:
      selected, selection_audit = _harvest_selected_triplets(
          Path(args.selected_triplets), frozen_cohort_path
      )
    except SelectedTripletValidationError as exc:
      parser.error(f"invalid committed triplet selection: {exc}")
    condition_harvests = {
        condition: [selected[condition]] for condition in CONDITIONS
    }
    selection_path = Path(args.selected_triplets).expanduser().resolve()
    manifest_paths = {
        condition: [selection_path] for condition in CONDITIONS
    }
  else:
    if args.selected_triplets:
      parser.error(
          "--selected_triplets requires --frozen_cohort_manifest"
      )
    if any(
        not getattr(args, f"{condition}_manifest")
        for condition in CONDITIONS
    ):
      parser.error(
          "Legacy/development mode requires a manifest for every condition"
      )
    manifest_paths = {
        condition: [
            Path(value).expanduser().resolve()
            for value in getattr(args, f"{condition}_manifest")
        ]
        for condition in CONDITIONS
    }
    condition_harvests = {
        condition: [_harvest(path, condition) for path in paths]
        for condition, paths in manifest_paths.items()
    }

  exit_code = 0
  try:
    report = analyze_harvests(
        condition_harvests,
        frozen_cohort=frozen_cohort,
        allow_incomplete=args.allow_incomplete,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
    )
  except PairedContrastValidationError as exc:
    report = exc.report
    exit_code = 2
  report["inputs"] = {
      "manifests": {
          condition: [str(path) for path in paths]
          for condition, paths in manifest_paths.items()
      },
      "frozen_cohort_manifest": args.frozen_cohort_manifest or None,
      "selection_audit": selection_audit,
      "inference_role": (
          "primary_cluster_aware_paired_hierarchical_inference"
          if frozen_cohort is not None else "legacy_or_development"
      ),
  }
  out_dir = Path(args.out_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  json_path = out_dir / "paired_hierarchical_contrasts.json"
  json_path.write_text(
      json.dumps(report, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  _write_markdown(out_dir / "paired_hierarchical_contrasts.md", report)
  print(f"Wrote {json_path}")
  if exit_code:
    print(
        "Strict paired contrast validation failed; inspect the JSON audit.",
        file=sys.stderr,
    )
  return exit_code


if __name__ == "__main__":
  raise SystemExit(main())
