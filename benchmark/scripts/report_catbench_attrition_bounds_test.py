"""Tests for formal CATBench exhausted-triplet bounds.

The fixtures use real CATBench model, app, and task identifiers.  Assigned
zeros/ones are tested only as explicitly labeled sensitivity bounds; they are
not benchmark episodes or emulator results.
"""

from __future__ import annotations

import report_catbench_attrition_bounds as bounds


MODEL = "UI-Venus-7B"


def _row(
    condition: str, app_id: str, instance_id: int, success: int
) -> dict[str, object]:
  return {
      "model": MODEL,
      "category": "maps",
      "app_id": app_id,
      "semantic_task_id": "MapsSearchPlace",
      "instance_id": instance_id,
      "instance_seed": f"maps:MapsSearchPlace:{instance_id}",
      "task_random_seed": 30,
      "is_successful": bool(success),
      "catbench_condition": condition,
  }


def _fixture():
  cohort = {
      "release_id": "catbench_acl_revision_5cat_v1",
      "task_random_seed": 30,
      "n_task_combinations": 2,
      "models": [MODEL],
      "categories": {
          "maps": {
              "aw_app_id": "maps_osmand",
              "app_ids": ["maps_osmand", "maps_organic_maps"],
              "semantic_task_ids": ["MapsSearchPlace"],
          }
      },
  }
  rows = {
      condition: [
          _row(condition, "maps_osmand", 0, 1),
          _row(condition, "maps_osmand", 1, 1),
          _row(condition, "maps_organic_maps", 0, success),
      ]
      for condition, success in (("c1", 0), ("c2_g", 1), ("c2_o", 1))
  }
  audit = {
      "scheduled_pairs": 4,
      "selected_pairs": 3,
      "exhausted_invalid_pairs": 1,
      "primary_point_estimate_permitted": False,
      "attrition_bounds_required": True,
  }
  return rows, cohort, audit


def test_exhausted_cell_blocks_primary_and_emits_app_gap_bounds():
  rows, cohort, audit = _fixture()
  report = bounds.build_attrition_report(rows, cohort, audit)

  assert report["artifact_role"] == (
      "attrition_sensitivity_not_observed_rollout_results"
  )
  assert not report["primary_point_estimate_permitted"]
  assert report["exhausted_triplet_cells_per_condition"] == 1
  c1 = report["condition_reports"]["c1"]
  assert c1["all_fail_sensitivity"][MODEL]["overall"] == {
      "aw_sr": 1.0,
      "new_sr": 0.0,
      "delta_new_minus_aw": -1.0,
      "retention": 0.0,
  }
  assert c1["all_success_sensitivity"][MODEL]["overall"] == {
      "aw_sr": 1.0,
      "new_sr": 0.5,
      "delta_new_minus_aw": -0.5,
      "retention": 0.5,
  }
  assert c1["identification_bounds"][MODEL]["overall"][
      "delta_new_minus_aw"
  ] == [-1.0, -0.5]


def test_contrast_bounds_preserve_selected_whole_triplet_transitions():
  rows, cohort, audit = _fixture()
  report = bounds.build_attrition_report(rows, cohort, audit)
  contrast = report["paired_contrasts"]["c1_vs_c2_g"]

  assert contrast["observed_selected_transitions"] == {
      "pass_pass": 2,
      "earlier_pass_later_fail": 0,
      "earlier_fail_later_pass": 1,
      "fail_fail": 0,
  }
  assert contrast["identification_bounds"][MODEL]["overall"][
      "new_sr_difference"
  ] == [0.0, 1.0]


def test_missing_count_must_match_committed_exhaustion_audit():
  rows, cohort, audit = _fixture()
  audit["exhausted_invalid_pairs"] = 0
  try:
    bounds.build_attrition_report(rows, cohort, audit)
  except ValueError as exc:
    assert "committed exhausted count" in str(exc)
  else:
    raise AssertionError("Mismatched attrition evidence was accepted")
