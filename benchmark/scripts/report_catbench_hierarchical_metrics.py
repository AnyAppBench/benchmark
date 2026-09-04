#!/usr/bin/env python3
"""Report strictly macro-averaged CATBench cross-app metrics.

The scoring hierarchy is intentionally explicit:

1. average repeated instances within each semantic task template;
2. average semantic templates within an app;
3. average apps within the AndroidWorld (AW) or new-app group; and
4. average categories for each model.

Consequently, neither categories with more apps nor templates with more
instances receive extra weight.  ``delta_new_minus_aw`` is always New - AW.

Input manifests are harvested through the conservative reader in
``pair_baseline_and_breakdown.py``.  By default, any invalid artifact,
duplicate experimental slot, incomplete model/category/app cell, or
non-common semantic task-instance roster makes the report invalid and the
program exits with status 2.  ``--allow_incomplete`` retains an explicitly
flagged diagnostic report, but must not be used for headline paper results.
The frozen primary release is read only through its committed whole-triplet
selection state; legacy matrix manifests remain available only without a
primary cohort.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
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
    _plan_reuse_validation,
    _primary_harvest_validation,
)


DEFAULT_AW_APPS = {
    "sms": "sms_simple_sms_messenger",
    "files": "files_material_files",
    "maps": "maps_osmand",
    "contacts": "contacts_google_contacts",
    "clock": "clock_google_clock",
}

SemanticSlot = tuple[str, int]


class MatrixValidationError(ValueError):
  """Raised when a matrix is not eligible for strict headline reporting."""

  def __init__(self, report: dict[str, Any]):
    super().__init__(
        "CATBench matrix failed strict completeness/semantic-roster checks"
    )
    self.report = report


@dataclasses.dataclass(frozen=True)
class _Cell:
  model: str
  category: str
  app_id: str
  semantic_task_id: str
  instance_id: int
  success: int
  instance_seed: Any
  semantic_goal_sha256: Any = None
  task_random_seed: Any = None

  @property
  def semantic_slot(self) -> SemanticSlot:
    return (self.semantic_task_id, self.instance_id)


def _stable_key(value: Any) -> str:
  return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def _mean(values: Sequence[float]) -> float | None:
  return statistics.fmean(values) if values else None


def _raw_counts(cells: Iterable[_Cell]) -> dict[str, int]:
  materialized = list(cells)
  return {
      "successes": sum(cell.success for cell in materialized),
      "episodes": len(materialized),
  }


def _normalize_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[_Cell], list[dict[str, Any]]]:
  cells: list[_Cell] = []
  issues: list[dict[str, Any]] = []
  seen: dict[tuple[str, str, str, str, int], int] = collections.Counter()
  for row_index, row in enumerate(rows):
    missing = [
        field
        for field in (
            "model",
            "category",
            "app_id",
            "semantic_task_id",
            "instance_id",
            "is_successful",
            "instance_seed",
        )
        if row.get(field) is None or row.get(field) == ""
    ]
    if missing:
      issues.append({
          "type": "missing_required_row_fields",
          "row_index": row_index,
          "fields": missing,
      })
      continue
    try:
      instance_id = int(row["instance_id"])
    except (TypeError, ValueError, OverflowError):
      issues.append({
          "type": "invalid_instance_id",
          "row_index": row_index,
          "value": str(row.get("instance_id")),
      })
      continue
    success_value = row["is_successful"]
    if isinstance(success_value, bool):
      success = int(success_value)
    elif success_value in (0, 1, 0.0, 1.0):
      success = int(success_value)
    else:
      issues.append({
          "type": "non_binary_success",
          "row_index": row_index,
          "value": str(success_value),
      })
      continue
    cell = _Cell(
        model=str(row["model"]),
        category=str(row["category"]),
        app_id=str(row["app_id"]),
        semantic_task_id=str(row["semantic_task_id"]),
        instance_id=instance_id,
        success=success,
        instance_seed=row["instance_seed"],
        semantic_goal_sha256=row.get("semantic_goal_sha256"),
        task_random_seed=row.get("task_random_seed"),
    )
    key = (
        cell.model,
        cell.category,
        cell.app_id,
        cell.semantic_task_id,
        cell.instance_id,
    )
    seen[key] += 1
    cells.append(cell)

  duplicate_keys = [key for key, count in seen.items() if count > 1]
  for key in sorted(duplicate_keys):
    issues.append({
        "type": "duplicate_semantic_experimental_slot",
        "slot": list(key),
        "count": seen[key],
    })
  if duplicate_keys:
    duplicate_set = set(duplicate_keys)
    cells = [
        cell
        for cell in cells
        if (
            cell.model,
            cell.category,
            cell.app_id,
            cell.semantic_task_id,
            cell.instance_id,
        )
        not in duplicate_set
    ]
  return cells, issues


def _matrix_issues(
    cells: Sequence[_Cell],
    aw_apps: Mapping[str, str],
    expected_roster: set[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
  """Validate a complete, app-common semantic task-instance matrix."""
  issues: list[dict[str, Any]] = []
  if not cells:
    issues.append({"type": "empty_matrix"})
  cell_roster = {(c.model, c.category, c.app_id) for c in cells}
  if expected_roster is not None:
    for roster_cell in sorted(expected_roster - cell_roster):
      issues.append({
          "type": "roster_cell_without_valid_episodes",
          "cell": list(roster_cell),
      })
    for roster_cell in sorted(cell_roster - expected_roster):
      issues.append({
          "type": "episode_cell_outside_manifest_roster",
          "cell": list(roster_cell),
      })

  models = sorted({cell.model for cell in cells})
  categories_by_model = {
      model: {cell.category for cell in cells if cell.model == model}
      for model in models
  }
  if models:
    expected_categories = set.union(*categories_by_model.values())
    for model in models:
      missing = sorted(expected_categories - categories_by_model[model])
      if missing:
        issues.append({
            "type": "model_missing_categories",
            "model": model,
            "categories": missing,
        })

  categories = sorted({cell.category for cell in cells})
  for category in categories:
    mapped_aw = aw_apps.get(category)
    if mapped_aw is None:
      issues.append({
          "type": "missing_aw_app_mapping",
          "category": category,
      })
    category_models = sorted({
        cell.model for cell in cells if cell.category == category
    })
    apps_by_model = {
        model: {
            cell.app_id
            for cell in cells
            if cell.model == model and cell.category == category
        }
        for model in category_models
    }
    if apps_by_model:
      expected_apps = set.union(*apps_by_model.values())
      for model in category_models:
        missing_apps = sorted(expected_apps - apps_by_model[model])
        if missing_apps:
          issues.append({
              "type": "model_category_missing_apps",
              "model": model,
              "category": category,
              "apps": missing_apps,
          })
        if mapped_aw is not None and mapped_aw not in apps_by_model[model]:
          issues.append({
              "type": "mapped_aw_app_absent",
              "model": model,
              "category": category,
              "aw_app": mapped_aw,
          })
        new_apps = apps_by_model[model] - ({mapped_aw} if mapped_aw else set())
        if not new_apps:
          issues.append({
              "type": "no_new_apps",
              "model": model,
              "category": category,
          })

    semantic_slots_by_model: dict[str, set[SemanticSlot]] = {}
    for model in category_models:
      app_slots: dict[str, set[SemanticSlot]] = {}
      for app_id in apps_by_model[model]:
        app_slots[app_id] = {
            cell.semantic_slot
            for cell in cells
            if cell.model == model
            and cell.category == category
            and cell.app_id == app_id
        }
      expected_slots = set.union(*app_slots.values()) if app_slots else set()
      semantic_slots_by_model[model] = expected_slots
      for app_id, slots in sorted(app_slots.items()):
        missing_slots = sorted(expected_slots - slots)
        extra_slots = sorted(slots - set.intersection(*app_slots.values()))
        if missing_slots:
          issues.append({
              "type": "app_missing_semantic_instances",
              "model": model,
              "category": category,
              "app_id": app_id,
              "semantic_slots": [list(slot) for slot in missing_slots],
          })
        # ``extra_slots`` is useful when every other app is the incomplete one.
        if extra_slots and not missing_slots:
          issues.append({
              "type": "app_has_noncommon_semantic_instances",
              "model": model,
              "category": category,
              "app_id": app_id,
              "semantic_slots": [list(slot) for slot in extra_slots],
          })

      for semantic_slot in sorted(expected_slots):
        matched = [
            cell
            for cell in cells
            if cell.model == model
            and cell.category == category
            and cell.semantic_slot == semantic_slot
        ]
        for field in (
            "instance_seed",
            "semantic_goal_sha256",
            "task_random_seed",
        ):
          values = [getattr(cell, field) for cell in matched]
          if not any(value is not None for value in values):
            continue
          value_keys = {
              "<missing>" if value is None else _stable_key(value)
              for value in values
          }
          if len(value_keys) > 1:
            issues.append({
                "type": "semantic_instance_provenance_differs_across_apps",
                "model": model,
                "category": category,
                "semantic_slot": list(semantic_slot),
                "field": field,
            })

    if semantic_slots_by_model:
      expected_model_slots = set.union(*semantic_slots_by_model.values())
      for model, slots in sorted(semantic_slots_by_model.items()):
        missing_slots = sorted(expected_model_slots - slots)
        if missing_slots:
          issues.append({
              "type": "model_category_missing_semantic_instances",
              "model": model,
              "category": category,
              "semantic_slots": [list(slot) for slot in missing_slots],
          })

      # The benchmark instance must also be common across evaluated models,
      # not merely across apps for one model.
      if len(category_models) > 1:
        for semantic_slot in sorted(expected_model_slots):
          matched = [
              cell
              for cell in cells
              if cell.category == category
              and cell.semantic_slot == semantic_slot
          ]
          for field in (
              "instance_seed",
              "semantic_goal_sha256",
              "task_random_seed",
          ):
            values = [getattr(cell, field) for cell in matched]
            if not any(value is not None for value in values):
              continue
            if len({
                "<missing>" if value is None else _stable_key(value)
                for value in values
            }) > 1:
              issues.append({
                  "type": (
                      "semantic_instance_provenance_differs_across_models"
                  ),
                  "category": category,
                  "semantic_slot": list(semantic_slot),
                  "field": field,
              })
  return issues


def _template_metrics(cells: Sequence[_Cell]) -> dict[str, Any]:
  by_template: dict[str, list[_Cell]] = collections.defaultdict(list)
  for cell in cells:
    by_template[cell.semantic_task_id].append(cell)
  output: dict[str, Any] = {}
  for template, template_cells in sorted(by_template.items()):
    counts = _raw_counts(template_cells)
    output[template] = {
        "sr": counts["successes"] / counts["episodes"],
        "raw": counts,
        "instance_ids": sorted(cell.instance_id for cell in template_cells),
    }
  return output


def _app_metrics(cells: Sequence[_Cell]) -> dict[str, Any]:
  templates = _template_metrics(cells)
  return {
      "sr": statistics.fmean(item["sr"] for item in templates.values()),
      "n_semantic_templates": len(templates),
      "raw": _raw_counts(cells),
      "templates": templates,
  }


def _percentile(values: Sequence[float], probability: float) -> float:
  ordered = sorted(values)
  if not ordered:
    raise ValueError("Cannot compute a percentile of an empty sequence")
  position = (len(ordered) - 1) * probability
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  fraction = position - lower
  return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _ci(values: Sequence[float], confidence: float) -> list[float]:
  alpha = (1.0 - confidence) / 2.0
  return [_percentile(values, alpha), _percentile(values, 1.0 - alpha)]


def _resample_category(
    app_cells: Mapping[str, Sequence[_Cell]],
    aw_app: str,
    rng: random.Random,
) -> tuple[float, float, float]:
  """Paired template/instance resample for one model and category."""
  apps = sorted(app_cells)
  new_apps = [app for app in apps if app != aw_app]
  common_templates = sorted(
      set.intersection(*[
          {cell.semantic_task_id for cell in app_cells[app]} for app in apps
      ])
  )
  if aw_app not in app_cells or not new_apps or not common_templates:
    raise ValueError("Category cannot be hierarchically resampled")

  lookup: dict[str, dict[str, dict[int, int]]] = {}
  for app in apps:
    lookup[app] = collections.defaultdict(dict)
    for cell in app_cells[app]:
      lookup[app][cell.semantic_task_id][cell.instance_id] = cell.success

  sampled_templates = [rng.choice(common_templates) for _ in common_templates]
  sampled_semantic_instances: list[tuple[str, list[int]]] = []
  for template in sampled_templates:
    common_instance_ids = sorted(
        set.intersection(*[
            set(lookup[candidate][template]) for candidate in apps
        ])
    )
    if not common_instance_ids:
      raise ValueError("Template has no common instances across apps")
    sampled_semantic_instances.append((
        template,
        [rng.choice(common_instance_ids) for _ in common_instance_ids],
    ))
  sampled_app_rates: dict[str, float] = {}
  for app in apps:
    sampled_template_rates: list[float] = []
    for template, sampled_instances in sampled_semantic_instances:
      sampled_template_rates.append(statistics.fmean(
          lookup[app][template][instance_id]
          for instance_id in sampled_instances
      ))
    sampled_app_rates[app] = statistics.fmean(sampled_template_rates)

  sampled_new_apps = [rng.choice(new_apps) for _ in new_apps]
  aw_sr = sampled_app_rates[aw_app]
  new_sr = statistics.fmean(sampled_app_rates[app] for app in sampled_new_apps)
  return aw_sr, new_sr, new_sr - aw_sr


def _bootstrap_model(
    category_app_cells: Mapping[str, Mapping[str, Sequence[_Cell]]],
    aw_apps: Mapping[str, str],
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]]]:
  if replicates < 1:
    raise ValueError("bootstrap_replicates must be positive")
  if not 0.0 < confidence < 1.0:
    raise ValueError("confidence must be between zero and one")
  rng = random.Random(seed)
  categories = sorted(category_app_cells)
  overall_draws = {"aw_sr": [], "new_sr": [], "delta_new_minus_aw": []}
  category_draws = {
      category: {"aw_sr": [], "new_sr": [], "delta_new_minus_aw": []}
      for category in categories
  }
  for _ in range(replicates):
    # Category-level intervals use one paired lower-level draw per category.
    category_samples = {
        category: _resample_category(
            category_app_cells[category], aw_apps[category], rng
        )
        for category in categories
    }
    for category, sample in category_samples.items():
      for key, value in zip(
          ("aw_sr", "new_sr", "delta_new_minus_aw"), sample
      ):
        category_draws[category][key].append(value)
    # The overall interval independently resamples the complete hierarchy.
    # A category selected twice gets two lower-level draws, while each draw
    # keeps AW and new-app outcomes paired on semantic task instances.
    sampled_category_results = [
        _resample_category(
            category_app_cells[category], aw_apps[category], rng
        )
        for category in [rng.choice(categories) for _ in categories]
    ]
    for metric_index, metric in enumerate(
        ("aw_sr", "new_sr", "delta_new_minus_aw")
    ):
      overall_draws[metric].append(statistics.fmean(
          sample[metric_index] for sample in sampled_category_results
      ))
  overall_ci = {
      metric: _ci(values, confidence) for metric, values in overall_draws.items()
  }
  category_ci = {
      category: {
          metric: _ci(values, confidence) for metric, values in draws.items()
      }
      for category, draws in category_draws.items()
  }
  return overall_ci, category_ci


def _score_cells(
    cells: Sequence[_Cell],
    aw_apps: Mapping[str, str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
  by_model: dict[str, list[_Cell]] = collections.defaultdict(list)
  for cell in cells:
    by_model[cell.model].append(cell)
  models_report: dict[str, Any] = {}
  for model_index, (model, model_cells) in enumerate(sorted(by_model.items())):
    by_category: dict[str, list[_Cell]] = collections.defaultdict(list)
    for cell in model_cells:
      by_category[cell.category].append(cell)
    category_reports: dict[str, Any] = {}
    bootstrap_input: dict[str, dict[str, list[_Cell]]] = {}
    for category, category_cells in sorted(by_category.items()):
      aw_app = aw_apps.get(category)
      if aw_app is None:
        continue
      by_app: dict[str, list[_Cell]] = collections.defaultdict(list)
      for cell in category_cells:
        by_app[cell.app_id].append(cell)
      if aw_app not in by_app:
        continue
      new_apps = sorted(app for app in by_app if app != aw_app)
      if not new_apps:
        continue
      app_reports = {
          app: _app_metrics(app_cells)
          for app, app_cells in sorted(by_app.items())
      }
      aw_sr = app_reports[aw_app]["sr"]
      new_app_rates = [app_reports[app]["sr"] for app in new_apps]
      new_sr = statistics.fmean(new_app_rates)
      category_reports[category] = {
          "aw_app": aw_app,
          "apps": app_reports,
          "groups": {
              "aw": {
                  "apps": [aw_app],
                  "sr": aw_sr,
                  "raw": _raw_counts(by_app[aw_app]),
              },
              "new": {
                  "apps": new_apps,
                  "sr": new_sr,
                  "population_sd_across_app_sr": statistics.pstdev(
                      new_app_rates
                  ),
                  "raw": _raw_counts(
                      cell for app in new_apps for cell in by_app[app]
                  ),
              },
          },
          "delta_new_minus_aw": new_sr - aw_sr,
          "raw": _raw_counts(category_cells),
      }
      bootstrap_input[category] = dict(by_app)

    if not category_reports:
      continue
    overall_aw = statistics.fmean(
        report["groups"]["aw"]["sr"] for report in category_reports.values()
    )
    overall_new = statistics.fmean(
        report["groups"]["new"]["sr"] for report in category_reports.values()
    )
    all_new_app_rates = [
        report["apps"][app]["sr"]
        for report in category_reports.values()
        for app in report["groups"]["new"]["apps"]
    ]
    bootstrap_error = None
    try:
      overall_ci, category_ci = _bootstrap_model(
          bootstrap_input,
          aw_apps,
          replicates=bootstrap_replicates,
          seed=bootstrap_seed + model_index,
          confidence=confidence,
      )
    except ValueError as exc:
      # This path is reachable only for an incomplete diagnostic matrix.  A
      # strict matrix has already established common apps/templates/instances.
      overall_ci = {
          "aw_sr": None,
          "new_sr": None,
          "delta_new_minus_aw": None,
      }
      category_ci = {}
      bootstrap_error = str(exc)
    for category, intervals in category_ci.items():
      category_reports[category]["bootstrap_ci"] = intervals
    models_report[model] = {
        "categories": category_reports,
        "overall": {
            "aw_sr": overall_aw,
            "new_sr": overall_new,
            "delta_new_minus_aw": overall_new - overall_aw,
            "population_sd_across_all_new_app_sr": statistics.pstdev(
                all_new_app_rates
            ),
            "n_categories": len(category_reports),
            "raw": _raw_counts(model_cells),
            "bootstrap_ci": overall_ci,
            "bootstrap_error": bootstrap_error,
        },
    }
  return models_report


def analyze_rows(
    rows: Iterable[Mapping[str, Any]],
    aw_apps: Mapping[str, str] | None = None,
    *,
    allow_incomplete: bool = False,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 1729,
    confidence: float = 0.95,
    expected_roster: set[tuple[str, str, str]] | None = None,
    preexisting_issues: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
  """Validate and score already-harvested in-memory result rows.

  This public entry point keeps unit tests independent from Android/emulator
  state.  Production CLI use first obtains the rows via strict manifest
  harvesting, then calls this function.
  """
  if bootstrap_replicates < 1:
    raise ValueError("bootstrap_replicates must be positive")
  if not 0.0 < confidence < 1.0:
    raise ValueError("confidence must be between zero and one")
  resolved_aw_apps = {**DEFAULT_AW_APPS, **dict(aw_apps or {})}
  cells, row_issues = _normalize_rows(rows)
  issues = [dict(issue) for issue in preexisting_issues]
  issues.extend(row_issues)
  issues.extend(_matrix_issues(cells, resolved_aw_apps, expected_roster))
  models = _score_cells(
      cells,
      resolved_aw_apps,
      bootstrap_replicates=bootstrap_replicates,
      bootstrap_seed=bootstrap_seed,
      confidence=confidence,
  )
  report = {
      "schema_version": 1,
      "strictly_valid": not issues,
      "allow_incomplete": allow_incomplete,
      "hierarchy": [
          "instances_within_semantic_template",
          "semantic_templates_within_app",
          "apps_within_group",
          "categories_within_model",
      ],
      "delta_definition": "New - AW",
      "aw_app_mapping": resolved_aw_apps,
      "bootstrap": {
          "method": (
              "paired hierarchical resampling of categories, new apps, "
              "semantic templates, and app-common instances"
          ),
          "replicates": bootstrap_replicates,
          "seed": bootstrap_seed,
          "confidence": confidence,
      },
      "validation": {
          "n_issues": len(issues),
          "issues": issues,
      },
      "models": models,
  }
  if issues and not allow_incomplete:
    raise MatrixValidationError(report)
  return report


def _merge_harvests(harvests: Sequence[Harvest]) -> Harvest:
  rows: dict[SlotKey, dict[str, Any]] = {}
  invalid_records: list[dict[str, Any]] = []
  invalid_slots: dict[SlotKey, list[dict[str, Any]]] = {}
  roster: set[tuple[str, str, str]] = set()
  duplicate_roster: set[tuple[str, str, str]] = set()
  selection_audits: list[dict[str, Any]] = []
  for harvest_index, harvest in enumerate(harvests):
    invalid_records.extend(harvest.invalid_records)
    for slot, records in harvest.invalid_slots.items():
      invalid_slots.setdefault(slot, []).extend(records)
      rows.pop(slot, None)
    roster.update(harvest.roster)
    duplicate_roster.update(harvest.duplicate_roster)
    if harvest.selection_audit is not None:
      selection_audits.append(harvest.selection_audit)
    for slot, row in harvest.rows.items():
      if slot in rows:
        invalid = {
            "slot": list(slot),
            "issues": ["duplicate_experimental_slot_across_manifests"],
            "manifest_index": harvest_index,
            "pkl_path": row.get("pkl_path", ""),
        }
        invalid_records.append(invalid)
        invalid_slots.setdefault(slot, []).append(invalid)
        rows.pop(slot, None)
      elif slot in invalid_slots:
        invalid = {
            "slot": list(slot),
            "issues": ["experimental_slot_invalid_in_another_manifest"],
            "manifest_index": harvest_index,
            "pkl_path": row.get("pkl_path", ""),
        }
        invalid_records.append(invalid)
        invalid_slots[slot].append(invalid)
      else:
        rows[slot] = row
  if len(selection_audits) > 1:
    invalid_records.append({
        "slot": None,
        "issues": ["multiple_selected_triplet_sources_cannot_be_merged"],
        "selection_source_count": len(selection_audits),
    })
  return Harvest(
      rows=rows,
      invalid_records=invalid_records,
      invalid_slots=invalid_slots,
      roster=roster,
      duplicate_roster=duplicate_roster,
      selection_audit=(
          selection_audits[0] if len(selection_audits) == 1 else None
      ),
  )


def _parse_aw_apps(values: Sequence[str]) -> dict[str, str]:
  mappings: dict[str, str] = {}
  for value in values:
    if "=" not in value:
      raise ValueError(f"--aw_app must be category=app_id, got: {value!r}")
    category, app_id = (part.strip() for part in value.split("=", 1))
    if not category or not app_id:
      raise ValueError(f"--aw_app must be category=app_id, got: {value!r}")
    if category in mappings and mappings[category] != app_id:
      raise ValueError(f"Conflicting --aw_app values for category {category!r}")
    mappings[category] = app_id
  return mappings


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
  lines = [
      "# CATBench hierarchical metrics",
      "",
      f"Strict validity: **{'PASS' if report['strictly_valid'] else 'FAIL'}**",
      "",
      "Scores average instances → semantic templates → apps → "
      "categories. Delta is **New − AW**.",
      "",
      "| Model | Categories | AW SR | New SR | New−AW | New-app SD |",
      "|---|---:|---:|---:|---:|---:|",
  ]
  for model, model_report in sorted(report["models"].items()):
    overall = model_report["overall"]
    lines.append(
        f"| {model} | {overall['n_categories']} | {overall['aw_sr']:.3f} | "
        f"{overall['new_sr']:.3f} | "
        f"{overall['delta_new_minus_aw']:+.3f} | "
        f"{overall['population_sd_across_all_new_app_sr']:.3f} |"
    )
  if report["validation"]["issues"]:
    lines.extend(["", "## Validation issues", ""])
    for issue in report["validation"]["issues"]:
      lines.append(f"- `{json.dumps(issue, sort_keys=True)}`")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--manifest",
      action="append",
      default=[],
      help="Matrix manifest; repeat to merge disjoint matrix shards.",
  )
  parser.add_argument(
      "--condition",
      required=True,
      choices=("baseline", "breakdown", "c1", "c2_g", "c2_o"),
      help="Expected explicit catbench_condition in every episode.",
  )
  parser.add_argument(
      "--aw_app",
      action="append",
      default=[],
      help=(
          "AndroidWorld app as category=app_id; repeat as needed. First-five "
          "CATBench categories have built-in defaults."
      ),
  )
  parser.add_argument("--allow_incomplete", action="store_true")
  parser.add_argument(
      "--primary_cohort_manifest",
      default="",
      help="Frozen cohort JSON required for strict primary-paper reporting.",
  )
  parser.add_argument(
      "--selected_triplets",
      default="",
      help=(
          "Committed schedule-consumer selected_triplets.jsonl. Required "
          "for primary-cohort reporting."
      ),
  )
  parser.add_argument("--bootstrap_replicates", type=int, default=10000)
  parser.add_argument("--bootstrap_seed", type=int, default=1729)
  parser.add_argument("--confidence", type=float, default=0.95)
  parser.add_argument("--out_dir", required=True)
  args = parser.parse_args(argv)

  primary_cohort = None
  primary_cohort_path = None
  if args.primary_cohort_manifest:
    primary_cohort_path = Path(
        args.primary_cohort_manifest
    ).expanduser().absolute()
    primary_cohort = json.loads(
        primary_cohort_path.read_text(encoding="utf-8")
    )
  selection_audit = None
  if primary_cohort is not None:
    if args.allow_incomplete:
      parser.error(
          "--allow_incomplete is prohibited for primary point estimates; "
          "pending/exhausted triplets require a separate predeclared "
          "attrition-bounds analysis"
      )
    if not args.selected_triplets:
      parser.error(
          "--selected_triplets is required for primary-cohort reporting"
      )
    if args.manifest:
      parser.error(
          "Primary reporting ingests only committed selected triplets; "
          "--manifest is reserved for non-primary legacy/development runs"
      )
    if args.condition not in {"c1", "c2_g", "c2_o"}:
      parser.error("Primary reporting condition must be c1, c2_g, or c2_o")
    try:
      selected, selection_audit = _harvest_selected_triplets(
          Path(args.selected_triplets), primary_cohort_path
      )
    except SelectedTripletValidationError as exc:
      parser.error(f"invalid committed triplet selection: {exc}")
    harvest = selected[args.condition]
    manifest_paths = [Path(args.selected_triplets).expanduser().resolve()]
  else:
    if args.selected_triplets:
      parser.error(
          "--selected_triplets requires --primary_cohort_manifest"
      )
    if not args.manifest:
      parser.error(
          "Non-primary legacy/development reporting requires --manifest"
      )
    manifest_paths = [
        Path(value).expanduser().resolve() for value in args.manifest
    ]
    harvest = _merge_harvests([
        _harvest(path, expected_condition=args.condition)
        for path in manifest_paths
    ])
  preexisting_issues: list[dict[str, Any]] = []
  for invalid in harvest.invalid_records:
    preexisting_issues.append({
        "type": "strict_harvest_invalid_artifact",
        "detail": invalid,
    })
  for roster_cell in sorted(harvest.duplicate_roster):
    preexisting_issues.append({
        "type": "duplicate_manifest_roster_cell",
        "cell": list(roster_cell),
    })
  if args.condition in {"breakdown", "c2_g", "c2_o"}:
    plan_reuse = _plan_reuse_validation(harvest)
    for issue in plan_reuse["issues"]:
      preexisting_issues.append({
          "type": "breakdown_plan_not_reused_across_apps",
          "detail": issue,
      })
  if primary_cohort is not None:
    primary_validation = _primary_harvest_validation(
        harvest, primary_cohort, args.condition
    )
    if not primary_validation["valid"]:
      preexisting_issues.append({
          "type": "frozen_primary_cohort_mismatch",
          "detail": primary_validation,
      })

  aw_apps = _parse_aw_apps(args.aw_app)
  if primary_cohort is not None:
    frozen_aw_apps = {
        category: spec["aw_app_id"]
        for category, spec in primary_cohort["categories"].items()
    }
    if args.aw_app and aw_apps != frozen_aw_apps:
      preexisting_issues.append({
          "type": "aw_app_mapping_differs_from_frozen_cohort",
          "expected": frozen_aw_apps,
          "actual": aw_apps,
      })
    aw_apps = frozen_aw_apps

  exit_code = 0
  try:
    report = analyze_rows(
        harvest.rows.values(),
        aw_apps,
        allow_incomplete=args.allow_incomplete,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
        expected_roster=harvest.roster,
        preexisting_issues=preexisting_issues,
    )
  except MatrixValidationError as exc:
    report = exc.report
    exit_code = 2

  report["inputs"] = {
      "condition": args.condition,
      "manifests": [str(path) for path in manifest_paths],
      "eligible_rows": len(harvest.rows),
      "invalid_artifacts": len(harvest.invalid_records),
      "selection_audit": selection_audit,
      "inference_role": (
          "primary_hierarchical_point_estimate"
          if primary_cohort is not None else "legacy_or_development"
      ),
  }
  out_dir = Path(args.out_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  json_path = out_dir / "hierarchical_metrics.json"
  json_path.write_text(
      json.dumps(report, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  _write_markdown(out_dir / "hierarchical_metrics.md", report)
  print(f"Wrote {json_path}")
  if exit_code:
    print(
        "Strict matrix validation failed; see hierarchical_metrics.json.",
        file=sys.stderr,
    )
  return exit_code


if __name__ == "__main__":
  raise SystemExit(main())
