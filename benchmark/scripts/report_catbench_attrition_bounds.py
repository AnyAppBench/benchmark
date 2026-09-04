#!/usr/bin/env python3
"""Report fail-closed missing-data bounds for exhausted CATBench triplets.

This reporter is used only when the frozen consumer has completed every
authorized attempt but one or more full C1/C2-G/C2-O blocks remain
``exhausted_invalid``.  It validates the committed hash-chained consumer state,
loads only genuinely selected episode artifacts, and assigns counterfactual
zeros/ones to exhausted cells solely to calculate predeclared sensitivity and
identification bounds.  Assigned values are never represented as episodes,
verifier outcomes, or replacement results.

Pending runs are rejected.  If no block is exhausted, the ordinary strict
hierarchical reporters remain the primary path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from pair_baseline_and_breakdown import (  # noqa: E402
    SelectedTripletValidationError,
    _harvest_selected_triplets,
)
import report_catbench_hierarchical_metrics as hierarchical  # noqa: E402
from report_catbench_paired_hierarchical_contrasts import (  # noqa: E402
    SemanticKey,
    expected_slots_from_cohort,
)


CONDITIONS = ("c1", "c2_g", "c2_o")
CONTRASTS = (("c1", "c2_g"), ("c1", "c2_o"), ("c2_g", "c2_o"))
Assignment = Callable[[SemanticKey], int]


def _semantic_key(row: Mapping[str, Any]) -> SemanticKey:
  return (
      str(row["model"]),
      str(row["category"]),
      str(row["app_id"]),
      str(row["semantic_task_id"]),
      int(row["instance_id"]),
  )


def _canonical(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _shared_provenance(
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
  output: dict[tuple[str, str, int], dict[str, Any]] = {}
  fields = ("instance_seed", "semantic_goal_sha256", "task_random_seed")
  for rows in condition_rows.values():
    for row in rows:
      slot = (
          str(row["category"]),
          str(row["semantic_task_id"]),
          int(row["instance_id"]),
      )
      candidate = {field: row.get(field) for field in fields}
      previous = output.get(slot)
      if previous is not None and _canonical(previous) != _canonical(candidate):
        raise ValueError(
            "Selected conditions disagree on semantic-instance provenance: "
            f"{slot!r}"
        )
      output[slot] = candidate
  return output


def _scenario_rows(
    observed_rows: Sequence[Mapping[str, Any]],
    expected: set[SemanticKey],
    provenance: Mapping[tuple[str, str, int], Mapping[str, Any]],
    assignment: Assignment,
    *,
    task_random_seed: int,
) -> tuple[list[dict[str, Any]], int]:
  observed: dict[SemanticKey, dict[str, Any]] = {}
  for raw_row in observed_rows:
    row = dict(raw_row)
    key = _semantic_key(row)
    if key in observed:
      raise ValueError(f"Duplicate selected semantic cell: {key!r}")
    if key not in expected:
      raise ValueError(f"Selected semantic cell is outside the cohort: {key!r}")
    observed[key] = row

  rows = list(observed.values())
  missing = sorted(expected - set(observed))
  for model, category, app_id, semantic_task_id, instance_id in missing:
    slot = (category, semantic_task_id, instance_id)
    shared = dict(provenance.get(slot) or {})
    instance_seed = shared.get("instance_seed")
    if instance_seed is None:
      instance_seed = f"attrition-bound:{category}:{semantic_task_id}:{instance_id}"
    row: dict[str, Any] = {
        "model": model,
        "category": category,
        "app_id": app_id,
        "semantic_task_id": semantic_task_id,
        "instance_id": instance_id,
        "instance_seed": instance_seed,
        "task_random_seed": shared.get("task_random_seed", task_random_seed),
        "is_successful": bool(assignment((
            model, category, app_id, semantic_task_id, instance_id
        ))),
        "attrition_assignment_not_observed": True,
    }
    if shared.get("semantic_goal_sha256") is not None:
      row["semantic_goal_sha256"] = shared["semantic_goal_sha256"]
    rows.append(row)
  return rows, len(missing)


def _compact_models(models: Mapping[str, Any]) -> dict[str, Any]:
  compact: dict[str, Any] = {}
  for model, report in sorted(models.items()):
    overall = report["overall"]
    categories: dict[str, Any] = {}
    for category, category_report in sorted(report["categories"].items()):
      aw_sr = category_report["groups"]["aw"]["sr"]
      new_sr = category_report["groups"]["new"]["sr"]
      categories[category] = {
          "aw_sr": aw_sr,
          "new_sr": new_sr,
          "delta_new_minus_aw": category_report["delta_new_minus_aw"],
          "app_sr": {
              app_id: app_report["sr"]
              for app_id, app_report in sorted(
                  category_report["apps"].items()
              )
          },
      }
    aw_sr = overall["aw_sr"]
    new_sr = overall["new_sr"]
    compact[model] = {
        "overall": {
            "aw_sr": aw_sr,
            "new_sr": new_sr,
            "delta_new_minus_aw": overall["delta_new_minus_aw"],
            "retention": (new_sr / aw_sr if aw_sr else None),
        },
        "categories": categories,
    }
  return compact


def _score_scenario(
    rows: Sequence[Mapping[str, Any]],
    aw_apps: Mapping[str, str],
    expected: set[SemanticKey],
) -> dict[str, Any]:
  cells, row_issues = hierarchical._normalize_rows(rows)  # pylint: disable=protected-access
  expected_roster = {(key[0], key[1], key[2]) for key in expected}
  issues = list(row_issues)
  issues.extend(hierarchical._matrix_issues(  # pylint: disable=protected-access
      cells, aw_apps, expected_roster
  ))
  if issues:
    raise ValueError(f"Bound scenario failed hierarchy validation: {issues[:3]}")
  models = hierarchical._score_cells(  # pylint: disable=protected-access
      cells,
      aw_apps,
      bootstrap_replicates=1,
      bootstrap_seed=0,
      confidence=0.95,
  )
  return _compact_models(models)


def _metric_bounds(
    all_fail: Mapping[str, Any],
    all_success: Mapping[str, Any],
    delta_lower: Mapping[str, Any],
    delta_upper: Mapping[str, Any],
) -> dict[str, Any]:
  output: dict[str, Any] = {}
  for model in sorted(all_fail):
    model_output: dict[str, Any] = {"categories": {}}
    for level in ("overall",):
      model_output[level] = {
          "aw_sr": [
              all_fail[model][level]["aw_sr"],
              all_success[model][level]["aw_sr"],
          ],
          "new_sr": [
              all_fail[model][level]["new_sr"],
              all_success[model][level]["new_sr"],
          ],
          "delta_new_minus_aw": [
              delta_lower[model][level]["delta_new_minus_aw"],
              delta_upper[model][level]["delta_new_minus_aw"],
          ],
      }
    for category in sorted(all_fail[model]["categories"]):
      model_output["categories"][category] = {
          "aw_sr": [
              all_fail[model]["categories"][category]["aw_sr"],
              all_success[model]["categories"][category]["aw_sr"],
          ],
          "new_sr": [
              all_fail[model]["categories"][category]["new_sr"],
              all_success[model]["categories"][category]["new_sr"],
          ],
          "delta_new_minus_aw": [
              delta_lower[model]["categories"][category][
                  "delta_new_minus_aw"
              ],
              delta_upper[model]["categories"][category][
                  "delta_new_minus_aw"
              ],
          ],
      }
    output[model] = model_output
  return output


def _difference(
    later: Mapping[str, Any], earlier: Mapping[str, Any]
) -> dict[str, Any]:
  output: dict[str, Any] = {}
  for model in sorted(later):
    output[model] = {"categories": {}}
    for level in ("overall",):
      output[model][level] = {
          metric: later[model][level][metric] - earlier[model][level][metric]
          for metric in ("aw_sr", "new_sr", "delta_new_minus_aw")
      }
    for category in sorted(later[model]["categories"]):
      output[model]["categories"][category] = {
          metric: (
              later[model]["categories"][category][metric]
              - earlier[model]["categories"][category][metric]
          )
          for metric in ("aw_sr", "new_sr", "delta_new_minus_aw")
      }
  return output


def _contrast_bounds(
    earlier: Mapping[str, Mapping[str, Any]],
    later: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
  lower_aw_new = _difference(later["all_fail"], earlier["all_success"])
  upper_aw_new = _difference(later["all_success"], earlier["all_fail"])
  lower_delta = _difference(later["delta_lower"], earlier["delta_upper"])
  upper_delta = _difference(later["delta_upper"], earlier["delta_lower"])
  output: dict[str, Any] = {}
  for model in sorted(lower_aw_new):
    output[model] = {"categories": {}}
    for level in ("overall",):
      output[model][level] = {
          "aw_sr_difference": [
              lower_aw_new[model][level]["aw_sr"],
              upper_aw_new[model][level]["aw_sr"],
          ],
          "new_sr_difference": [
              lower_aw_new[model][level]["new_sr"],
              upper_aw_new[model][level]["new_sr"],
          ],
          "app_substitution_delta_difference": [
              lower_delta[model][level]["delta_new_minus_aw"],
              upper_delta[model][level]["delta_new_minus_aw"],
          ],
      }
    for category in sorted(lower_aw_new[model]["categories"]):
      output[model]["categories"][category] = {
          "aw_sr_difference": [
              lower_aw_new[model]["categories"][category]["aw_sr"],
              upper_aw_new[model]["categories"][category]["aw_sr"],
          ],
          "new_sr_difference": [
              lower_aw_new[model]["categories"][category]["new_sr"],
              upper_aw_new[model]["categories"][category]["new_sr"],
          ],
          "app_substitution_delta_difference": [
              lower_delta[model]["categories"][category][
                  "delta_new_minus_aw"
              ],
              upper_delta[model]["categories"][category][
                  "delta_new_minus_aw"
              ],
          ],
      }
  return output


def _transition_counts(
    earlier_rows: Sequence[Mapping[str, Any]],
    later_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
  earlier = {_semantic_key(row): bool(row["is_successful"]) for row in earlier_rows}
  later = {_semantic_key(row): bool(row["is_successful"]) for row in later_rows}
  if set(earlier) != set(later):
    raise ValueError("Selected condition rows are not exact whole-triplet pairs")
  counts = {
      "pass_pass": 0,
      "earlier_pass_later_fail": 0,
      "earlier_fail_later_pass": 0,
      "fail_fail": 0,
  }
  for key in sorted(earlier):
    pair = (earlier[key], later[key])
    if pair == (True, True):
      counts["pass_pass"] += 1
    elif pair == (True, False):
      counts["earlier_pass_later_fail"] += 1
    elif pair == (False, True):
      counts["earlier_fail_later_pass"] += 1
    else:
      counts["fail_fail"] += 1
  return counts


def build_attrition_report(
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    cohort: Mapping[str, Any],
    selection_audit: Mapping[str, Any],
) -> dict[str, Any]:
  """Build bounds from selected real outcomes plus explicit assignments."""
  if set(condition_rows) != set(CONDITIONS):
    raise ValueError("Attrition report requires c1/c2_g/c2_o rows")
  expected = expected_slots_from_cohort(cohort)
  aw_apps = {
      str(category): str(spec["aw_app_id"])
      for category, spec in cohort["categories"].items()
  }
  provenance = _shared_provenance(condition_rows)
  task_random_seed = int(cohort.get("task_random_seed", 0))

  assignments: dict[str, Assignment] = {
      "all_fail": lambda _key: 0,
      "all_success": lambda _key: 1,
      "delta_lower": lambda key: int(key[2] == aw_apps[key[1]]),
      "delta_upper": lambda key: int(key[2] != aw_apps[key[1]]),
  }
  scenario_scores: dict[str, dict[str, Any]] = {}
  missing_counts: dict[str, int] = {}
  for condition in CONDITIONS:
    scenario_scores[condition] = {}
    for scenario, assignment in assignments.items():
      rows, missing_count = _scenario_rows(
          condition_rows[condition],
          expected,
          provenance,
          assignment,
          task_random_seed=task_random_seed,
      )
      missing_counts[condition] = missing_count
      scenario_scores[condition][scenario] = _score_scenario(
          rows, aw_apps, expected
      )

  if len(set(missing_counts.values())) != 1:
    raise ValueError("Conditions do not have the same exhausted triplet roster")
  missing_count = next(iter(missing_counts.values()))
  exhausted_count = int(selection_audit.get("exhausted_invalid_pairs", -1))
  if missing_count != exhausted_count:
    raise ValueError(
        "Selected-row attrition does not equal the committed exhausted count: "
        f"rows={missing_count}, state={exhausted_count}"
    )

  condition_reports: dict[str, Any] = {}
  for condition in CONDITIONS:
    scores = scenario_scores[condition]
    observed_successes = sum(
        bool(row["is_successful"]) for row in condition_rows[condition]
    )
    condition_reports[condition] = {
        "observed_selected_cells": len(condition_rows[condition]),
        "observed_selected_successes": observed_successes,
        "observed_selected_failures": (
            len(condition_rows[condition]) - observed_successes
        ),
        "exhausted_unobserved_cells": missing_count,
        "all_fail_sensitivity": scores["all_fail"],
        "all_success_sensitivity": scores["all_success"],
        "identification_bounds": _metric_bounds(
            scores["all_fail"],
            scores["all_success"],
            scores["delta_lower"],
            scores["delta_upper"],
        ),
    }

  contrasts: dict[str, Any] = {}
  for earlier, later in CONTRASTS:
    label = f"{earlier}_vs_{later}"
    contrasts[label] = {
        "signed_contrast": f"{later} - {earlier}",
        "observed_selected_transitions": _transition_counts(
            condition_rows[earlier], condition_rows[later]
        ),
        "all_fail_sensitivity": _difference(
            scenario_scores[later]["all_fail"],
            scenario_scores[earlier]["all_fail"],
        ),
        "all_success_sensitivity": _difference(
            scenario_scores[later]["all_success"],
            scenario_scores[earlier]["all_success"],
        ),
        "identification_bounds": _contrast_bounds(
            scenario_scores[earlier], scenario_scores[later]
        ),
    }

  return {
      "schema_version": 1,
      "artifact_role": "attrition_sensitivity_not_observed_rollout_results",
      "release_id": cohort.get("release_id"),
      "strict_state_validated": True,
      "primary_point_estimate_permitted": missing_count == 0,
      "bounds_required": missing_count > 0,
      "assignment_warning": (
          "Zeros and ones assigned to exhausted cells are formal sensitivity "
          "values only; they are not episodes, verifier outputs, or results."
      ),
      "scheduled_triplet_cells_per_condition": len(expected),
      "selected_triplet_cells_per_condition": len(expected) - missing_count,
      "exhausted_triplet_cells_per_condition": missing_count,
      "selection_audit": dict(selection_audit),
      "condition_reports": condition_reports,
      "paired_contrasts": contrasts,
  }


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
  lines = [
      "# CATBench attrition bounds",
      "",
      f"Release: `{report['release_id']}`",
      "",
      f"Scheduled triplets: {report['scheduled_triplet_cells_per_condition']}",
      f"Selected triplets: {report['selected_triplet_cells_per_condition']}",
      f"Exhausted-invalid triplets: "
      f"{report['exhausted_triplet_cells_per_condition']}",
      "",
      "**Assigned bound values are not observed rollout results.**",
      "",
      "A primary complete-roster point estimate is "
      + ("permitted." if report["primary_point_estimate_permitted"] else "blocked."),
  ]
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--selected_triplets", required=True)
  parser.add_argument("--primary_cohort_manifest", required=True)
  parser.add_argument("--out_dir", required=True)
  args = parser.parse_args(argv)

  cohort_path = Path(args.primary_cohort_manifest).expanduser().absolute()
  cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
  try:
    harvests, selection_audit = _harvest_selected_triplets(
        Path(args.selected_triplets),
        cohort_path,
        allow_exhausted=True,
    )
    report = build_attrition_report(
        {
            condition: list(harvests[condition].rows.values())
            for condition in CONDITIONS
        },
        cohort,
        selection_audit,
    )
  except (SelectedTripletValidationError, KeyError, TypeError, ValueError) as exc:
    parser.error(f"invalid committed attrition state: {exc}")

  report["inputs"] = {
      "selected_triplets": str(
          Path(args.selected_triplets).expanduser().absolute()
      ),
      "primary_cohort_manifest": str(cohort_path),
  }
  out_dir = Path(args.out_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  json_path = out_dir / "attrition_bounds.json"
  json_path.write_text(
      json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  _write_markdown(out_dir / "attrition_bounds.md", report)
  print(f"Wrote {json_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
