"""Unit tests for strict C1/C2 episode pairing.

These tests use only in-memory episode/manifest dictionaries.  They do not
construct benchmark tasks, substitute mock applications, or start Android.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import pair_baseline_and_breakdown as pairing


REAL_RELEASE = "catbench_acl_revision_5cat_v1"
REAL_MODEL = "UI-Venus-7B"


def _selection_orchestration_fixture(
    *, selected_round: int = 1
) -> tuple[list[dict], list[dict], dict, dict, str]:
  """Builds control-plane records with real primary IDs, never app/results."""
  cohort_sha256 = "a" * 64
  cohort = {
      "release_id": REAL_RELEASE,
      "conditions": ["c1", "c2_g", "c2_o"],
      "models": [REAL_MODEL],
      "n_task_combinations": 1,
      "categories": {
          "maps": {
              "app_ids": ["maps_osmand"],
              "semantic_task_ids": ["MapsSearchPlace"],
          }
      },
  }
  paired_key = {
      "model": REAL_MODEL,
      "category": "maps",
      "app_id": "maps_osmand",
      "semantic_task_id": "MapsSearchPlace",
      "instance_id": 0,
  }
  pair_id = pairing._expected_pair_id(  # pylint: disable=protected-access
      REAL_RELEASE, paired_key
  )
  events: list[dict] = []
  finished_ids: list[str] = []
  for round_index in range(selected_round + 1):
    for condition in pairing.PRIMARY_CONDITIONS:
      identity = pairing._expected_attempt_identity(  # pylint: disable=protected-access
          REAL_RELEASE, pair_id, condition, round_index
      )
      provenance = {
          "release_id": REAL_RELEASE,
          "cohort_sha256": cohort_sha256,
          "pair_id": pair_id,
          **identity,
          "model": REAL_MODEL,
          "category": "maps",
          "app_id": "maps_osmand",
          "semantic_task_id": "MapsSearchPlace",
          "instance_id": 0,
          "condition": condition,
      }
      events.append({
          "event": "started",
          "recorded_at": "2026-07-10T00:00:00+00:00",
          **provenance,
      })
      status = "valid_success"
      if round_index < selected_round and condition == "c1":
        status = "invalid_infrastructure"
      events.append({
          "event": "finished",
          "recorded_at": "2026-07-10T00:00:01+00:00",
          **provenance,
          "status": status,
      })
      finished_ids.append(identity["attempt_id"])
  previous = ""
  journal: list[dict] = []
  for sequence, event in enumerate(events):
    body = {
        **event,
        "sequence": sequence,
        "previous_event_sha256": previous,
    }
    event_hash = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    record = {**body, "event_sha256": event_hash}
    journal.append(record)
    previous = event_hash
  selected_attempt_ids = {
      condition: pairing._expected_attempt_identity(  # pylint: disable=protected-access
          REAL_RELEASE, pair_id, condition, selected_round
      )["attempt_id"]
      for condition in pairing.PRIMARY_CONDITIONS
  }
  selection = [{
      "release_id": REAL_RELEASE,
      "pair_id": pair_id,
      "paired_key": paired_key,
      "selection_unit": "full_condition_triplet",
      "selection_status": "selected_complete_triplet",
      "selected_round": selected_round,
      "selected_attempt_ids": selected_attempt_ids,
      "all_finished_attempt_ids": finished_ids,
      "selection_basis": (
          "first complete round with no invalid_infrastructure member; "
          "no cross-round condition mixing"
      ),
  }]
  consumer_manifest = {
      "schema_version": 1,
      "release_id": REAL_RELEASE,
      "release_purpose": "primary_five_category_analysis",
      "analysis_eligible": True,
      "artifact_role": "primary_analysis_candidate",
      "primary_reporter_acceptance_permitted": True,
      "cohort_sha256": cohort_sha256,
      "episode_slot_count": 3,
      "paired_block_count": 1,
      "selective_filters": False,
      "source_revision": "0123456789abcdef",
      **{
          field: "b" * 64
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
          )
      },
  }
  return selection, journal, consumer_manifest, cohort, cohort_sha256


def _item(
    *,
    condition: str | None,
    success: int = 0,
    app_id: str = "clock_clock",
    app_name: str = "Clock",
    package_name: str = "com.best.deskclock",
    app_version: str = "2.29",
    task_template: str = "ClockCreateAlarmForClock",
    goal: str = "In the Clock app, create an alarm for 10:00.",
    instance_id: int = 0,
    instance_seed: int = 73491,
    task_random_seed: int = 30,
    semantic_task_id: str = "ClockCreateAlarm",
    plan_key: str | None = None,
    exception_info: str | None = None,
    episode_status: str | None = None,
    exception_fields: dict | None = None,
    code_revision: str = "0123456789abcdef",
) -> dict:
  metadata = {
      "semantic_task_id": semantic_task_id,
      "semantic_goal_sha256": "semantic-goal-hash",
  }
  if plan_key is not None:
    metadata.update({"plan_key": plan_key, "plan_sha256": "plan-hash"})
  episode = {
      "task_template": task_template,
      "instance_id": instance_id,
      "goal": goal,
      "seed": instance_seed,
      "task_random_seed": task_random_seed,
      "catbench_condition": condition,
      "catbench_episode_status": episode_status or (
          "valid_success" if success else "valid_failure"
      ),
      "is_successful": success,
      "exception_info": exception_info,
      "task_breakdown_metadata": metadata,
  }
  if condition is None:
    episode.pop("catbench_condition")
  if exception_fields:
    episode.update(exception_fields)
  return {
      "episode": episode,
      "job": {
          "model_name": "GUI-Owl-7B",
          "category": "clock",
          "app_id": app_id,
          "app_name": app_name,
          "package_name": package_name,
          "app_version": app_version,
      },
      "manifest": {"code_revision": code_revision},
      "pkl_path": f"/{app_id}/{task_template}.pkl.gz",
      "episode_index": 0,
  }


class HarvestTest(unittest.TestCase):

  def test_missing_condition_is_invalid_not_default_baseline(self):
    result = pairing._harvest_items(  # pylint: disable=protected-access
        [_item(condition=None)], "baseline"
    )
    self.assertFalse(result.rows)
    self.assertEqual(1, len(result.invalid_records))
    self.assertIn(
        "condition:<missing>!=expected:baseline",
        result.invalid_records[0]["issues"],
    )

  def test_infrastructure_exception_is_invalid_not_failure(self):
    result = pairing._harvest_items(  # pylint: disable=protected-access
        [_item(
            condition="baseline",
            success=0,
            exception_info="ADB timeout",
            episode_status="invalid_infrastructure",
        )],
        "baseline",
    )
    self.assertFalse(result.rows)
    self.assertIn("infrastructure_exception", result.invalid_records[0]["issues"])

  def test_wrong_treatment_condition_is_invalid(self):
    result = pairing._harvest_items(  # pylint: disable=protected-access
        [_item(condition="baseline", plan_key="ClockCreateAlarm|instance=0")],
        "breakdown",
    )
    self.assertFalse(result.rows)
    self.assertIn(
        "condition:baseline!=expected:breakdown",
        result.invalid_records[0]["issues"],
    )

  def test_duplicate_slots_reject_every_candidate(self):
    item = _item(condition="baseline")
    result = pairing._harvest_items(  # pylint: disable=protected-access
        [item, copy.deepcopy(item)], "baseline"
    )
    self.assertFalse(result.rows)
    self.assertEqual(2, len(result.invalid_records))
    self.assertTrue(all(
        "duplicate_experimental_slot" in record["issues"]
        for record in result.invalid_records
    ))

  def test_breakdown_requires_plan_key(self):
    result = pairing._harvest_items(  # pylint: disable=protected-access
        [_item(condition="breakdown")], "breakdown"
    )
    self.assertFalse(result.rows)
    self.assertIn("missing_provenance:plan_key", result.invalid_records[0]["issues"])

  def test_declared_typed_agent_parse_failure_is_eligible(self):
    result = pairing._harvest_items(  # pylint: disable=protected-access
        [_item(
            condition="baseline",
            success=0,
            episode_status="valid_failure",
            exception_fields={
                "catbench_exception_stage": "agent",
                "catbench_exception_attribution": (
                    "agent_output_parse_or_malformed_action"
                ),
                "catbench_exception_valid_agent_failure": True,
                "catbench_exception_declared_agent_output": True,
                "catbench_exception_type": (
                    "android_world.agents.episode_exceptions.ActionParseError"
                ),
                "catbench_exception_failure_code": "action_parse_error",
            },
        )],
        "baseline",
    )
    self.assertEqual(1, len(result.rows))
    self.assertFalse(result.invalid_records)

  def test_untyped_generic_agent_error_cannot_claim_valid_failure(self):
    result = pairing._harvest_items(  # pylint: disable=protected-access
        [_item(
            condition="baseline",
            success=0,
            episode_status="valid_failure",
            exception_fields={
                "catbench_exception_stage": "agent",
                "catbench_exception_attribution": "unknown",
                "catbench_exception_valid_agent_failure": False,
                "catbench_exception_declared_agent_output": False,
                "catbench_exception_type": "builtins.ValueError",
            },
        )],
        "baseline",
    )
    self.assertFalse(result.rows)
    self.assertIn(
        "invalid_typed_agent_exception_attribution",
        result.invalid_records[0]["issues"],
    )


class PairingTest(unittest.TestCase):

  def _harvest_pair(self, baseline_item: dict, treatment_item: dict):
    baseline = pairing._harvest_items(  # pylint: disable=protected-access
        [baseline_item], "baseline"
    )
    treatment = pairing._harvest_items(  # pylint: disable=protected-access
        [treatment_item], "breakdown"
    )
    self.assertFalse(baseline.invalid_records)
    self.assertFalse(treatment.invalid_records)
    return pairing._paired_metrics(  # pylint: disable=protected-access
        baseline.rows, treatment.rows
    )

  def test_goal_and_instance_seed_mismatch_are_not_paired(self):
    baseline = _item(condition="baseline")
    treatment = _item(
        condition="breakdown",
        plan_key="ClockCreateAlarm|instance=0|semantic-goal-hash",
        goal="In the Clock app, create an alarm for 14:20.",
        instance_seed=99221,
    )
    report = self._harvest_pair(baseline, treatment)
    self.assertEqual(0, report["n_paired"])
    self.assertEqual(1, report["n_pair_provenance_mismatches"])
    reasons = report["pair_provenance_mismatches"][0]["reasons"]
    self.assertIn("mismatched:goal_sha256", reasons)
    self.assertIn("mismatched:instance_seed", reasons)

  def test_app_and_code_provenance_mismatch_are_not_paired(self):
    baseline = _item(condition="baseline")
    treatment = _item(
        condition="breakdown",
        package_name="com.best.deskclock.repacked",
        app_version="2.30",
        code_revision="fedcba9876543210",
        plan_key="ClockCreateAlarm|instance=0|semantic-goal-hash",
    )
    report = self._harvest_pair(baseline, treatment)
    reasons = report["pair_provenance_mismatches"][0]["reasons"]
    self.assertIn("mismatched:package_name", reasons)
    self.assertIn("mismatched:app_version", reasons)
    self.assertIn("mismatched:code_revision", reasons)

  def test_missing_counterpart_is_separate_from_invalid(self):
    baseline = pairing._harvest_items(  # pylint: disable=protected-access
        [_item(condition="baseline")], "baseline"
    )
    treatment = pairing._harvest_items([], "breakdown")  # pylint: disable=protected-access
    report = pairing._paired_metrics(  # pylint: disable=protected-access
        baseline.rows, treatment.rows
    )
    self.assertEqual(1, report["baseline_only"])
    self.assertEqual(0, report["n_pair_provenance_mismatches"])
    self.assertEqual(0, len(treatment.invalid_records))

  def test_interpretations_and_exact_mcnemar(self):
    baseline_items = []
    treatment_items = []
    outcomes = [(0, 1), (0, 1), (0, 1), (0, 0), (1, 1)]
    for instance_id, (baseline_success, treatment_success) in enumerate(outcomes):
      kwargs = {
          "instance_id": instance_id,
          "instance_seed": 73491 + instance_id,
          "goal": f"In the Clock app, create alarm instance {instance_id}.",
      }
      baseline_items.append(_item(
          condition="baseline", success=baseline_success, **kwargs
      ))
      treatment_items.append(_item(
          condition="breakdown",
          success=treatment_success,
          plan_key=f"ClockCreateAlarm|instance={instance_id}|semantic-goal-hash",
          **kwargs,
      ))
    baseline = pairing._harvest_items(  # pylint: disable=protected-access
        baseline_items, "baseline"
    )
    treatment = pairing._harvest_items(  # pylint: disable=protected-access
        treatment_items, "breakdown"
    )
    report = pairing._paired_metrics(  # pylint: disable=protected-access
        baseline.rows, treatment.rows
    )
    self.assertEqual(3, report["planning_responsive"])
    self.assertEqual(0, report["treatment_regressions"])
    self.assertEqual(1, report["residual_under_plan_assistance"])
    self.assertEqual(1, report["both_successful"])
    self.assertAlmostEqual(0.25, float(report["mcnemar_exact_p_two_sided"]))
    self.assertAlmostEqual(
        0.625,
        float(pairing._mcnemar_exact_p(1, 3)),  # pylint: disable=protected-access
    )
    self.assertNotEqual(
        "0", pairing._mcnemar_exact_p(0, 1150)  # pylint: disable=protected-access
    )


class CohortValidationTest(unittest.TestCase):

  def test_roster_mismatch_is_explicit(self):
    baseline = pairing.Harvest({}, [], {})
    treatment = pairing.Harvest({}, [], {})
    baseline.roster = {
        ("GUI-Owl-7B", "clock", "clock_clock"),
        ("GUI-Owl-7B", "clock", "clock_chrono"),
    }
    treatment.roster = {("GUI-Owl-7B", "clock", "clock_clock")}
    report = pairing._roster_validation(  # pylint: disable=protected-access
        baseline, treatment
    )
    self.assertFalse(report["valid"])
    self.assertEqual(
        [["GUI-Owl-7B", "clock", "clock_chrono"]],
        report["baseline_only_roster"],
    )

  def test_cross_app_plan_key_difference_is_invalid(self):
    clock = _item(
        condition="breakdown",
        plan_key="ClockCreateAlarm|instance=0|clock-plan",
    )
    chrono = _item(
        condition="breakdown",
        app_id="clock_chrono",
        app_name="Chrono",
        package_name="com.vicolo.chrono",
        app_version="0.6.0",
        task_template="ClockCreateAlarmForChrono",
        goal="In the Chrono app, create an alarm for 10:00.",
        plan_key="ClockCreateAlarm|instance=0|chrono-plan",
    )
    treatment = pairing._harvest_items(  # pylint: disable=protected-access
        [clock, chrono], "breakdown"
    )
    report = pairing._plan_reuse_validation(  # pylint: disable=protected-access
        treatment
    )
    self.assertFalse(report["valid"])
    self.assertIn(
        "different_plan_keys_across_apps", report["issues"][0]["reasons"]
    )

  def test_cross_app_instance_seed_difference_is_invalid(self):
    clock = _item(
        condition="breakdown",
        plan_key="ClockCreateAlarm|instance=0|semantic-goal-hash",
    )
    chrono = _item(
        condition="breakdown",
        app_id="clock_chrono",
        app_name="Chrono",
        package_name="com.vicolo.chrono",
        app_version="0.6.0",
        task_template="ClockCreateAlarmForChrono",
        goal="In the Chrono app, create an alarm for 10:00.",
        instance_seed=99221,
        plan_key="ClockCreateAlarm|instance=0|semantic-goal-hash",
    )
    treatment = pairing._harvest_items(  # pylint: disable=protected-access
        [clock, chrono], "breakdown"
    )
    report = pairing._plan_reuse_validation(  # pylint: disable=protected-access
        treatment
    )
    self.assertFalse(report["valid"])
    self.assertIn(
        "different_instance_seeds_across_apps", report["issues"][0]["reasons"]
    )


class SelectedTripletStateTest(unittest.TestCase):
  """Tests only frozen-schedule orchestration with real primary identities."""

  def test_discard_only_g6_release_and_role_are_rejected(self):
    selection, journal, manifest, cohort, cohort_sha = (
        _selection_orchestration_fixture(selected_round=0)
    )
    cohort["release_id"] = "catbench_acl_revision_5cat_g6_dryrun_v1"
    with self.assertRaisesRegex(
        pairing.SelectedTripletValidationError,
        "reserved for the frozen primary release",
    ):
      pairing._validate_selected_triplet_state(  # pylint: disable=protected-access
          selection, journal, manifest, cohort, cohort_sha
      )

    cohort["release_id"] = REAL_RELEASE
    manifest.update({
        "release_purpose": "g6_discard_only_end_to_end_validation",
        "analysis_eligible": False,
        "artifact_role": "discard_only_never_primary_analysis",
        "primary_reporter_acceptance_permitted": False,
    })
    with self.assertRaisesRegex(
        pairing.SelectedTripletValidationError,
        "Consumer manifest mismatch for release_purpose",
    ):
      pairing._validate_selected_triplet_state(  # pylint: disable=protected-access
          selection, journal, manifest, cohort, cohort_sha
      )

  def test_complete_replacement_history_selects_one_whole_valid_round(self):
    fixture = _selection_orchestration_fixture(selected_round=1)

    selected, audit = pairing._validate_selected_triplet_state(  # pylint: disable=protected-access
        *fixture
    )

    self.assertEqual(audit["journal_finished_attempts"], 6)
    self.assertEqual(audit["infrastructure_invalid_prior_attempts"], 1)
    self.assertEqual(audit["pairs_selected_from_replacement_round"], 1)
    for condition in pairing.PRIMARY_CONDITIONS:
      self.assertEqual(len(selected[condition]), 1)
      self.assertEqual(
          selected[condition][0]["finish"]["attempt_index"], 1
      )
      self.assertIn(
          f":{condition}:attempt:r1",
          selected[condition][0]["finish"]["attempt_id"],
      )

  def test_cross_round_condition_mixing_is_rejected(self):
    selection, journal, manifest, cohort, cohort_sha = (
        _selection_orchestration_fixture(selected_round=1)
    )
    pair_id = selection[0]["pair_id"]
    selection[0]["selected_attempt_ids"]["c2_g"] = (
        pairing._expected_attempt_identity(  # pylint: disable=protected-access
            REAL_RELEASE, pair_id, "c2_g", 0
        )["attempt_id"]
    )

    with self.assertRaisesRegex(
        pairing.SelectedTripletValidationError,
        "Selected condition attempt IDs mismatch",
    ):
      pairing._validate_selected_triplet_state(  # pylint: disable=protected-access
          selection, journal, manifest, cohort, cohort_sha
      )

  def test_pending_primary_block_aborts_instead_of_complete_case_selection(self):
    selection, journal, manifest, cohort, cohort_sha = (
        _selection_orchestration_fixture(selected_round=0)
    )
    selection[0]["selection_status"] = "pending"
    selection[0]["selected_round"] = None
    selection[0]["selected_attempt_ids"] = {}

    with self.assertRaisesRegex(
        pairing.SelectedTripletValidationError,
        "Primary pair is not reportable",
    ):
      pairing._validate_selected_triplet_state(  # pylint: disable=protected-access
          selection, journal, manifest, cohort, cohort_sha
      )

  def test_exhausted_block_is_reserved_for_attrition_bounds(self):
    selection, journal, manifest, cohort, cohort_sha = (
        _selection_orchestration_fixture(selected_round=2)
    )
    final_invalid = next(
        event
        for event in journal
        if event["event"] == "finished"
        and event["condition"] == "c1"
        and event["attempt_index"] == 2
    )
    final_invalid["status"] = "invalid_infrastructure"
    previous = ""
    for event in journal:
      event["previous_event_sha256"] = previous
      body = dict(event)
      body.pop("event_sha256")
      event["event_sha256"] = hashlib.sha256(
          json.dumps(
              body,
              sort_keys=True,
              separators=(",", ":"),
              ensure_ascii=False,
          ).encode("utf-8")
      ).hexdigest()
      previous = event["event_sha256"]
    selection[0]["selection_status"] = "exhausted_invalid"
    selection[0]["selected_round"] = None
    selection[0]["selected_attempt_ids"] = {}

    with self.assertRaisesRegex(
        pairing.SelectedTripletValidationError,
        "dedicated attrition-bounds path",
    ):
      pairing._validate_selected_triplet_state(  # pylint: disable=protected-access
          selection, journal, manifest, cohort, cohort_sha
      )

    selected, audit = pairing._validate_selected_triplet_state(  # pylint: disable=protected-access
        selection,
        journal,
        manifest,
        cohort,
        cohort_sha,
        allow_exhausted=True,
    )
    self.assertEqual(audit["scheduled_pairs"], 1)
    self.assertEqual(audit["selected_pairs"], 0)
    self.assertEqual(audit["exhausted_invalid_pairs"], 1)
    self.assertFalse(audit["primary_point_estimate_permitted"])
    self.assertTrue(audit["attrition_bounds_required"])
    for condition in pairing.PRIMARY_CONDITIONS:
      self.assertEqual(selected[condition], [])

  def test_infrastructure_invalid_member_cannot_be_selected(self):
    selection, journal, manifest, cohort, cohort_sha = (
        _selection_orchestration_fixture(selected_round=1)
    )
    selected_id = selection[0]["selected_attempt_ids"]["c2_o"]
    finish = next(
        event
        for event in journal
        if event["event"] == "finished" and event["attempt_id"] == selected_id
    )
    finish["status"] = "invalid_infrastructure"
    # Re-hash the modified journal so this test reaches the selection-status
    # gate instead of merely exercising tamper detection.
    previous = ""
    for event in journal:
      event["previous_event_sha256"] = previous
      body = dict(event)
      body.pop("event_sha256")
      event["event_sha256"] = hashlib.sha256(
          json.dumps(
              body,
              sort_keys=True,
              separators=(",", ":"),
              ensure_ascii=False,
          ).encode("utf-8")
      ).hexdigest()
      previous = event["event_sha256"]

    with self.assertRaisesRegex(
        pairing.SelectedTripletValidationError,
        "Selected round contains infrastructure-invalid member",
    ):
      pairing._validate_selected_triplet_state(  # pylint: disable=protected-access
          selection, journal, manifest, cohort, cohort_sha
      )

  def test_journal_identity_or_hash_tamper_is_rejected(self):
    selection, journal, manifest, cohort, cohort_sha = (
        _selection_orchestration_fixture(selected_round=1)
    )
    journal[-1]["app_id"] = "maps_comaps"

    with self.assertRaisesRegex(
        pairing.SelectedTripletValidationError,
        "event SHA-256 mismatch",
    ):
      pairing._validate_selected_triplet_state(  # pylint: disable=protected-access
          selection, journal, manifest, cohort, cohort_sha
      )

  def test_replacement_without_prior_infrastructure_trigger_is_rejected(self):
    selection, journal, manifest, cohort, cohort_sha = (
        _selection_orchestration_fixture(selected_round=1)
    )
    initial_failure = next(
        event
        for event in journal
        if event["event"] == "finished"
        and event["condition"] == "c1"
        and event["attempt_index"] == 0
    )
    initial_failure["status"] = "valid_failure"
    previous = ""
    for event in journal:
      event["previous_event_sha256"] = previous
      body = dict(event)
      body.pop("event_sha256")
      event["event_sha256"] = hashlib.sha256(
          json.dumps(
              body,
              sort_keys=True,
              separators=(",", ":"),
              ensure_ascii=False,
          ).encode("utf-8")
      ).hexdigest()
      previous = event["event_sha256"]

    with self.assertRaisesRegex(
        pairing.SelectedTripletValidationError,
        "Replacement round lacks infrastructure trigger",
    ):
      pairing._validate_selected_triplet_state(  # pylint: disable=protected-access
          selection, journal, manifest, cohort, cohort_sha
      )


if __name__ == "__main__":
  unittest.main()
