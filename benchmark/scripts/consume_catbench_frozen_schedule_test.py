"""Tests for the real-identity frozen schedule consumer.

The execution doubles below are in-memory control-flow doubles only.  Every
model, app, semantic task, condition, and schedule identifier is from the real
CATBench primary cohort; no synthetic App A/App B or placeholder benchmark task
is introduced.
No emulator, model endpoint, or external API is launched by these tests.
"""

from __future__ import annotations

import copy
import csv
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
from typing import Any, Mapping
from unittest import mock

from absl.testing import absltest

import build_catbench_frozen_schedule as schedule_builder
import consume_catbench_frozen_schedule as consumer
import preflight_task_breakdowns as plan_preflight
import write_catbench_plan_approval_template as approval_template


REAL_RELEASE = "catbench_acl_revision_5cat_v1"
REAL_MODEL = "UI-Venus-7B"


def _real_paired_key(
    *,
    category: str,
    app_id: str,
    semantic_task_id: str,
    instance_id: int = 0,
) -> dict[str, Any]:
  return {
      "model": REAL_MODEL,
      "category": category,
      "app_id": app_id,
      "semantic_task_id": semantic_task_id,
      "instance_id": instance_id,
  }


def _real_triplet(
    key: Mapping[str, Any],
    block_order: int,
    condition_order: tuple[str, str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  pair_id = schedule_builder._pair_id(REAL_RELEASE, key)  # pylint: disable=protected-access
  rows: list[dict[str, Any]] = []
  initial_slots: dict[str, dict[str, str]] = {}
  for within, condition in enumerate(condition_order):
    identity = schedule_builder._attempt_identity(  # pylint: disable=protected-access
        REAL_RELEASE, pair_id, condition, 0
    )
    initial_slots[condition] = identity
    rows.append({
        "release_id": REAL_RELEASE,
        "cohort_sha256": "a" * 64,
        "schedule_seed": 20260710,
        "suite_family": "android_world",
        "task_random_seed": 30,
        "n_task_combinations": 3,
        "episode_runtime_policy_sha256": (
            schedule_builder.episode_runtime_policy_sha256()
        ),
        "global_order": block_order * 3 + within,
        "block_order": block_order,
        "within_block_order": within,
        "pair_id": pair_id,
        **key,
        "condition": condition,
        **identity,
        "attempt_index": 0,
        "snapshot_family_id": f"{REAL_RELEASE}:snapshot:{pair_id}",
        "is_replacement": False,
    })
  authorized = []
  for round_index in (1, 2):
    authorized.append({
        "round_index": round_index,
        "scheduled": False,
        "condition_attempts": {
            condition: schedule_builder._attempt_identity(  # pylint: disable=protected-access
                REAL_RELEASE, pair_id, condition, round_index
            )
            for condition in consumer.CONDITIONS
        },
    })
  ledger = {
      "release_id": REAL_RELEASE,
      "pair_id": pair_id,
      "paired_key": dict(key),
      "max_replacement_rounds": 2,
      "selection_unit": "full_condition_triplet",
      "outcome_selected_replacement_permitted": False,
      "initial_condition_slots": {
          condition: initial_slots[condition]
          for condition in consumer.CONDITIONS
      },
      "authorized_replacement_rounds": authorized,
      "replacement_rounds": [],
  }
  return rows, ledger


def _small_real_bundle(
    triplets: list[tuple[list[dict[str, Any]], dict[str, Any]]]
) -> consumer.FrozenBundle:
  schedule = tuple(row for rows, _ in triplets for row in rows)
  ledgers = tuple(ledger for _, ledger in triplets)
  return consumer.FrozenBundle(
      cohort={"release_id": REAL_RELEASE},
      cohort_sha256="a" * 64,
      schedule_manifest={},
      schedule_manifest_sha256="b" * 64,
      schedule=schedule,
      ledger_seed=ledgers,
      ledger_schema_sha256="c" * 64,
  )


def _real_pin(app_id: str) -> dict[str, str]:
  with consumer.DEFAULT_PINS.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
      if row["app_id"] == app_id:
        return dict(row)
  raise AssertionError(f"Real app pin not found: {app_id}")


def _primary_conformance_fixture(evidence_root: Path) -> tuple[
    consumer.FrozenBundle,
    dict[str, dict[str, str]],
    dict[str, str],
    dict[str, Any],
]:
  """Build an in-memory gate fixture from the real frozen primary roster."""
  evidence_root.mkdir(parents=True, exist_ok=False)

  def write_evidence(relative_path: str, content: str) -> str:
    path = evidence_root.joinpath(*relative_path.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (content + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()

  cohort_path = schedule_builder.DEFAULT_COHORT
  cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
  bundle = consumer.FrozenBundle(
      cohort=cohort,
      cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
      schedule_manifest={},
      schedule_manifest_sha256="a" * 64,
      schedule=(),
      ledger_seed=(),
      ledger_schema_sha256="b" * 64,
  )
  pins = consumer.load_app_pins(consumer.DEFAULT_PINS, cohort)
  base_snapshot = {
      "snapshot_id": "catbench-primary-api33-frozen-base",
      "snapshot_sha256": "c" * 64,
  }
  records: list[dict[str, Any]] = []
  for category, spec in cohort["categories"].items():
    for app_id in spec["app_ids"]:
      pin = pins[app_id]
      for semantic_task_id in spec["semantic_task_ids"]:
        adapter_id = f"{category}/{app_id}/{semantic_task_id}"
        evidence: dict[str, dict[str, Any]] = {}
        for case_name in consumer.VERIFIER_CONFORMANCE_CASES:
          expected_success = case_name in {
              "primitive_action_positive",
              "reset_replay",
          }
          evidence_id = f"{adapter_id}/{case_name}"
          action_path = f"adapters/{adapter_id}/cases/{case_name}/action.jsonl"
          state_path = f"adapters/{adapter_id}/cases/{case_name}/state.json"
          evidence[case_name] = {
              "case": case_name,
              "expected_success": expected_success,
              "observed_success": expected_success,
              "passed": True,
              "skipped": False,
              "overwritten": False,
              "evidence_id": evidence_id,
              "action_trace_path": action_path,
              "action_trace_sha256": write_evidence(
                  action_path, f"action:{evidence_id}"
              ),
              "state_evidence_path": state_path,
              "state_evidence_sha256": write_evidence(
                  state_path, f"state:{evidence_id}"
              ),
          }
        identity_path = f"adapters/{adapter_id}/app_identity.json"
        adapter_path = f"adapters/{adapter_id}/adapter_evidence.json"
        records.append({
            "category": category,
            "app_id": app_id,
            "semantic_task_id": semantic_task_id,
            "package_name": pin["package_name"],
            "version_name": pin["version_name"],
            "version_code": pin["version_code"],
            "apk_sha256": pin["apk_sha256"],
            "approved": True,
            "skipped": False,
            "overwritten": False,
            "app_identity_evidence_path": identity_path,
            "app_identity_evidence_sha256": write_evidence(
                identity_path, f"identity:{adapter_id}"
            ),
            "adapter_evidence_path": adapter_path,
            "adapter_evidence_sha256": write_evidence(
                adapter_path, f"adapter:{adapter_id}"
            ),
            "evidence": evidence,
        })
  collection_tool_path = "header/collection_tool.py"
  manifest_evidence_path = "header/manifest_evidence.json"
  approval_evidence_path = "header/approval_evidence.json"
  payload = {
      "schema_version": 1,
      "release_id": cohort["release_id"],
      "cohort_sha256": bundle.cohort_sha256,
      "base_snapshot_id": base_snapshot["snapshot_id"],
      "base_snapshot_sha256": base_snapshot["snapshot_sha256"],
      "docker_image": consumer.FROZEN_DOCKER_IMAGE,
      "qualification_policy": consumer.VERIFIER_CONFORMANCE_POLICY,
      "artifact_role": consumer.VERIFIER_CONFORMANCE_ARTIFACT_ROLE,
      "analysis_eligible": False,
      "approved": True,
      "attestor_id": "real-cohort-conformance-attestor",
      "attested_at": "2026-07-11T00:00:00+00:00",
      "approver_id": "real-cohort-release-approver",
      "approved_at": "2026-07-11T00:00:00+00:00",
      "collection_tool_path": collection_tool_path,
      "collection_tool_sha256": write_evidence(
          collection_tool_path, "real test collection tool bytes"
      ),
      "manifest_evidence_path": manifest_evidence_path,
      "manifest_evidence_sha256": write_evidence(
          manifest_evidence_path, "real-roster test manifest evidence bytes"
      ),
      "approval_evidence_path": approval_evidence_path,
      "approval_evidence_sha256": write_evidence(
          approval_evidence_path, "real-roster test approval evidence bytes"
      ),
      "records": records,
  }
  return bundle, pins, base_snapshot, payload


def _execution_fixture(
    root: Path, *, condition: str = "c1"
) -> tuple[
    consumer.SubprocessEpisodeExecutor,
    consumer.AttemptSpec,
    consumer.RealTask,
    consumer.FrozenModel,
    dict[str, str],
]:
  triplet = _real_triplet(
      _real_paired_key(
          category="sms",
          app_id="sms_simple_sms_messenger",
          semantic_task_id="SmsSend",
      ),
      0,
      ("c1", "c2_g", "c2_o"),
  )
  attempt = consumer.AttemptSpec.from_row(
      next(row for row in triplet[0] if row["condition"] == condition)
  )
  base_bundle = _small_real_bundle([triplet])
  bundle = consumer.FrozenBundle(
      cohort={
          "release_id": REAL_RELEASE,
          "suite_family": "android_world",
          "episode_runtime_policy": copy.deepcopy(
              schedule_builder.FROZEN_EPISODE_RUNTIME_POLICY
          ),
      },
      cohort_sha256=base_bundle.cohort_sha256,
      schedule_manifest={
          "episode_runtime_policy": copy.deepcopy(
              schedule_builder.FROZEN_EPISODE_RUNTIME_POLICY
          ),
          "episode_runtime_policy_sha256": (
              schedule_builder.episode_runtime_policy_sha256()
          ),
      },
      schedule_manifest_sha256=base_bundle.schedule_manifest_sha256,
      schedule=base_bundle.schedule,
      ledger_seed=base_bundle.ledger_seed,
      ledger_schema_sha256=base_bundle.ledger_schema_sha256,
  )
  task = consumer.RealTask(
      category="sms",
      app_id="sms_simple_sms_messenger",
      semantic_task_id="SmsSend",
      task_template="SmsSendForSimpleSMSMessenger",
      package_name="com.simplemobiletools.smsmessenger",
  )
  runner = Path(consumer.__file__).resolve()
  model = consumer.FrozenModel(
      name=REAL_MODEL,
      revision="f3c6e7264df2a3d75db2f25b3a63a6955a0f062d",
      runner=runner,
      runner_sha256=consumer._sha256_path(runner),  # pylint: disable=protected-access
      args=("--model_name=inclusionAI/UI-Venus-Navi-7B",),
  )
  pin = _real_pin("sms_simple_sms_messenger")
  context = consumer.ExecutionContext(
      bundle=bundle,
      real_tasks={("sms", "sms_simple_sms_messenger", "SmsSend"): task},
      models={REAL_MODEL: model},
      app_pins={"sms_simple_sms_messenger": pin},
      model_config_sha256="d" * 64,
      model_endpoint_attestation_sha256="e" * 64,
      app_pins_sha256="f" * 64,
      installed_app_attestation_sha256="1" * 64,
      episode_runtime_policy=copy.deepcopy(
          schedule_builder.FROZEN_EPISODE_RUNTIME_POLICY
      ),
      episode_runtime_environment=dict(
          schedule_builder.FROZEN_EPISODE_RUNTIME_POLICY["environment"]
      ),
      episode_runtime_policy_sha256=(
          schedule_builder.episode_runtime_policy_sha256()
      ),
      source_revision="2" * 40,
      c2_g_breakdown=root / "real_c2_g_breakdowns.json",
      c2_o_breakdown=root / "real_c2_o_breakdowns.json",
      c2_g_sha256="3" * 64,
      c2_o_sha256="4" * 64,
      base_snapshot={
          "snapshot_id": "catbench-api33-frozen-base",
          "snapshot_sha256": "5" * 64,
      },
      base_snapshot_manifest_sha256="6" * 64,
      snapshot_hook=root / "catbench_snapshot_hook",
      snapshot_hook_sha256="7" * 64,
      python_bin="python",
      adb_path="adb",
      device_serial="emulator-5554",
      console_port=5554,
      grpc_port=8554,
      output_root=root,
  )
  return (
      consumer.SubprocessEpisodeExecutor(context),
      attempt,
      task,
      model,
      pin,
  )


def _valid_episode(
    executor: consumer.SubprocessEpisodeExecutor,
    attempt: consumer.AttemptSpec,
    task: consumer.RealTask,
    model: consumer.FrozenModel,
    pin: Mapping[str, str],
    *,
    runner_config_sha256: str,
) -> dict[str, Any]:
  context = executor.context
  release_policy = schedule_builder.release_policy(attempt.release_id)
  semantic_goal_sha256 = "8" * 64
  episode: dict[str, Any] = {
      "task_template": task.task_template,
      "instance_id": attempt.instance_id,
      "semantic_task_id": attempt.semantic_task_id,
      "semantic_goal_sha256": semantic_goal_sha256,
      "semantic_parameter_sha256": "9" * 64,
      "catbench_condition": attempt.condition,
      "catbench_condition_config_valid": True,
      "catbench_episode_status": "valid_failure",
      "release_id": attempt.release_id,
      "release_purpose": release_policy["release_purpose"],
      "artifact_role": release_policy["artifact_role"],
      "analysis_eligible": release_policy["analysis_eligible"],
      "cohort_sha256": attempt.cohort_sha256,
      "episode_runtime_policy_sha256": (
          context.episode_runtime_policy_sha256
      ),
      "schedule_manifest_sha256": context.bundle.schedule_manifest_sha256,
      "code_revision": context.source_revision,
      "package_name": task.package_name,
      "app_id": attempt.app_id,
      "model_name": model.name,
      "model_revision": model.revision,
      "runner_config_sha256": runner_config_sha256,
      "model_config_sha256": context.model_config_sha256,
      "model_endpoint_attestation_sha256": (
          context.model_endpoint_attestation_sha256
      ),
      "app_pins_sha256": context.app_pins_sha256,
      "installed_app_attestation_sha256": (
          context.installed_app_attestation_sha256
      ),
      "pair_id": attempt.pair_id,
      "slot_id": attempt.slot_id,
      "attempt_id": attempt.attempt_id,
      "attempt_index": attempt.attempt_index,
      "snapshot_family_id": attempt.snapshot_family_id,
      "snapshot_clone_id": attempt.snapshot_clone_id,
      "app_version": pin["version_name"],
      "app_version_code": pin["version_code"],
      "apk_sha256": pin["apk_sha256"],
      "task_random_seed": attempt.task_random_seed,
      "n_task_combinations": attempt.n_task_combinations,
      "schedule_seed": attempt.schedule_seed,
      "plan_file_sha256": (
          ""
          if attempt.condition == "c1"
          else context.c2_g_sha256
          if attempt.condition == "c2_g"
          else context.c2_o_sha256
      ),
      "is_successful": 0.0,
      "exception_info": None,
  }
  if attempt.condition == "c1":
    episode["task_breakdown_metadata"] = {}
    episode["task_breakdown_text"] = ""
  else:
    breakdown_text = "1. Open the messaging app.\n2. Compose and send the message."
    episode["task_breakdown_text"] = breakdown_text
    episode["task_breakdown_metadata"] = {
        "task_template": task.task_template,
        "instance_id": attempt.instance_id,
        "semantic_task_id": attempt.semantic_task_id,
        "semantic_goal_sha256": semantic_goal_sha256,
        "plan_key": (
            f"{attempt.semantic_task_id}|instance={attempt.instance_id}|"
            f"{semantic_goal_sha256}"
        ),
        "plan_sha256": hashlib.sha256(breakdown_text.encode()).hexdigest(),
        "condition": "application_independent_breakdown_prepend",
    }
  return episode


class _InMemoryRealEpisodeExecutor:
  """Returns declared statuses without launching any external process."""

  def __init__(self, status_fn):
    self.status_fn = status_fn
    self.calls: list[consumer.AttemptSpec] = []

  def execute(self, attempt: consumer.AttemptSpec) -> consumer.AttemptOutcome:
    self.calls.append(attempt)
    status = self.status_fn(attempt)
    return consumer.AttemptOutcome(
        status=status,
        artifact_path=f"/in-memory-test/{attempt.attempt_id}.pkl.gz",
        result_contract_path=f"/in-memory-test/{attempt.attempt_id}.json",
        reason_code="in_memory_control_flow_test",
        is_successful=1.0 if status == "valid_success" else 0.0,
        artifact_sha256="d" * 64,
    )


class PlanApprovalWorksheetTest(absltest.TestCase):
  """The worksheet binds real CATBench identities but never asserts approval."""

  def _roster(self, plan_path: Path) -> tuple[set[str], set[str]]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    return (
        {str(entry["key"]) for entry in payload["breakdowns"]},
        {str(entry["plan_key"]) for entry in payload["breakdowns"]},
    )

  def _write_plan(
      self, root: Path, *, condition: str = "c2_o"
  ) -> tuple[Path, str, Path]:
    cohort_path = root / "catbench_5cat_primary_cohort.json"
    cohort_path.write_text(
        json.dumps({"release_id": REAL_RELEASE}) + "\n", encoding="utf-8"
    )
    cohort_sha256 = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    plan_key = f"SmsSend|instance=0|{'a' * 64}"
    payload = {
        "metadata": {
            "generator_provider": "human" if condition == "c2_o" else "gemini",
            "cohort_release_id": REAL_RELEASE,
            "cohort_manifest_sha256": cohort_sha256,
            "expected_entry_count": 2,
            "expected_semantic_plan_count": 1,
        },
        "breakdowns": [
            {
                "key": (
                    "SmsSendForSimpleSMSMessenger|instance=0|" + "b" * 64
                ),
                "plan_key": plan_key,
                "plan_sha256": "c" * 64,
            },
            {
                "key": "SmsSendForQUIKSMS|instance=0|" + "d" * 64,
                "plan_key": plan_key,
                "plan_sha256": "c" * 64,
            },
        ],
    }
    path = root / f"{condition}_real_identity_plan.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path, plan_key, cohort_path

  def test_complete_shared_plan_roster_writes_only_pending_worksheet(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      plan_path, plan_key, cohort_path = self._write_plan(root)
      expected_exact_keys, expected_plan_keys = self._roster(plan_path)
      worksheet = approval_template.build_template(
          breakdown_path=plan_path,
          cohort_manifest_path=cohort_path,
          condition="c2_o",
          release_id=REAL_RELEASE,
          expected_entry_count=2,
          expected_plan_count=1,
          expected_exact_instance_keys=expected_exact_keys,
          expected_plan_keys=expected_plan_keys,
          c2_g_attempt_audit=None,
      )
      self.assertEqual(worksheet["approval_status"], "pending_human_review")
      self.assertEqual(worksheet["required_entry_count"], 2)
      self.assertEqual(worksheet["required_plan_keys"], [plan_key])
      self.assertEmpty(worksheet["approved_plan_keys"])
      self.assertEmpty(worksheet["reviewers"])

  def test_real_frozen_cohort_rosters_are_exact(self):
    for cohort_path, expected_entries, expected_plans in (
        (schedule_builder.DEFAULT_COHORT, 690, 150),
        (schedule_builder.DEFAULT_G6_DRY_RUN_COHORT, 5, 5),
    ):
      with self.subTest(cohort_path=cohort_path):
        exact_keys, plan_keys = approval_template._expected_cohort_roster(  # pylint: disable=protected-access
            cohort_path
        )
        self.assertLen(exact_keys, expected_entries)
        self.assertLen(plan_keys, expected_plans)

  def test_missing_duplicate_or_inconsistent_entries_are_not_deduplicated(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      plan_path, _, cohort_path = self._write_plan(root)
      payload = json.loads(plan_path.read_text(encoding="utf-8"))
      expected_exact_keys, expected_plan_keys = self._roster(plan_path)
      mutations = {
          "invalid plan_key": lambda value: value["breakdowns"][0].pop(
              "plan_key"
          ),
          "Duplicate exact-instance": lambda value: value["breakdowns"][1].update(
              {"key": value["breakdowns"][0]["key"]}
          ),
          "different plan hashes": lambda value: value["breakdowns"][1].update(
              {"plan_sha256": "e" * 64}
          ),
          "exact-instance roster differs": lambda value: value["breakdowns"][1].update(
              {"key": "SmsSendForMessages|instance=0|" + "e" * 64}
          ),
          "semantic-key roster differs": lambda value: [
              entry.update({"plan_key": f"SmsReply|instance=0|{'f' * 64}"})
              for entry in value["breakdowns"]
          ],
      }
      for expected_error, mutate in mutations.items():
        with self.subTest(expected_error=expected_error):
          candidate = copy.deepcopy(payload)
          mutate(candidate)
          plan_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
          with self.assertRaisesRegex(ValueError, expected_error):
            approval_template.build_template(
                breakdown_path=plan_path,
                cohort_manifest_path=cohort_path,
                condition="c2_o",
                release_id=REAL_RELEASE,
                expected_entry_count=2,
                expected_plan_count=1,
                expected_exact_instance_keys=expected_exact_keys,
                expected_plan_keys=expected_plan_keys,
                c2_g_attempt_audit=None,
            )

  def test_explicit_expected_counts_prevent_partial_worksheet(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      plan_path, _, cohort_path = self._write_plan(Path(tmpdir))
      expected_exact_keys, expected_plan_keys = self._roster(plan_path)
      with self.assertRaisesRegex(ValueError, "has 1 semantic plans; expected 150"):
        approval_template.build_template(
            breakdown_path=plan_path,
            cohort_manifest_path=cohort_path,
            condition="c2_o",
            release_id=REAL_RELEASE,
            expected_entry_count=2,
            expected_plan_count=150,
            expected_exact_instance_keys=expected_exact_keys,
            expected_plan_keys=expected_plan_keys,
            c2_g_attempt_audit=None,
        )

  def test_worksheet_marker_is_rejected_even_if_status_is_changed(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      plan_path, _, cohort_path = self._write_plan(root)
      expected_exact_keys, expected_plan_keys = self._roster(plan_path)
      worksheet = approval_template.build_template(
          breakdown_path=plan_path,
          cohort_manifest_path=cohort_path,
          condition="c2_o",
          release_id=REAL_RELEASE,
          expected_entry_count=2,
          expected_plan_count=1,
          expected_exact_instance_keys=expected_exact_keys,
          expected_plan_keys=expected_plan_keys,
          c2_g_attempt_audit=None,
      )
      worksheet["approval_status"] = "approved_for_primary_release"
      worksheet_path = root / "c2_o_pending_worksheet.json"
      worksheet_path.write_text(json.dumps(worksheet) + "\n", encoding="utf-8")
      with self.assertRaisesRegex(
          consumer.ArtifactValidationError, "pending review worksheet"
      ):
        consumer._validate_two_person_approval(  # pylint: disable=protected-access
            worksheet_path,
            breakdown_path=plan_path,
            condition="c2_o",
            release_id=REAL_RELEASE,
            expected_plan_count=1,
        )


class ReplacementStateTest(absltest.TestCase):

  def test_complete_r0_epoch_precedes_full_triplet_r1(self):
    sms = _real_triplet(
        _real_paired_key(
            category="sms",
            app_id="sms_simple_sms_messenger",
            semantic_task_id="SmsSend",
        ),
        0,
        ("c2_g", "c1", "c2_o"),
    )
    files = _real_triplet(
        _real_paired_key(
            category="files",
            app_id="files_material_files",
            semantic_task_id="FilesCreateFolder",
        ),
        1,
        ("c1", "c2_o", "c2_g"),
    )
    bundle = _small_real_bundle([sms, files])
    sms_pair = sms[1]["pair_id"]

    def status(attempt: consumer.AttemptSpec) -> str:
      if (
          attempt.pair_id == sms_pair
          and attempt.attempt_index == 0
          and attempt.condition == "c1"
      ):
        return "invalid_infrastructure"
      if attempt.condition == "c2_o":
        return "valid_failure"
      return "valid_success"

    executor = _InMemoryRealEpisodeExecutor(status)
    with tempfile.TemporaryDirectory() as tmpdir:
      runner = consumer.ScheduleConsumer(bundle, Path(tmpdir), executor)
      self.assertEqual(runner.run_until_complete(halt_after_invalid=False), 0)
      calls = executor.calls
      self.assertLen(calls, 9)
      # Both complete initial triplets precede any replacement attempt.
      self.assertEqual([item.attempt_index for item in calls[:6]], [0] * 6)
      self.assertEqual([item.pair_id for item in calls[:3]], [sms_pair] * 3)
      self.assertEqual(
          [item.pair_id for item in calls[3:6]], [files[1]["pair_id"]] * 3
      )
      self.assertEqual([item.attempt_index for item in calls[6:]], [1] * 3)
      self.assertEqual(
          [item.condition for item in calls[6:]], ["c2_g", "c1", "c2_o"]
      )

      selections = consumer._read_jsonl(  # pylint: disable=protected-access
          Path(tmpdir) / consumer.SELECTION_FILE
      )
      by_pair = {row["pair_id"]: row for row in selections}
      self.assertEqual(by_pair[sms_pair]["selected_round"], 1)
      self.assertEqual(
          set(by_pair[sms_pair]["selected_attempt_ids"]), set(consumer.CONDITIONS)
      )
      self.assertEqual(by_pair[files[1]["pair_id"]]["selected_round"], 0)

      runtime = consumer._read_jsonl(  # pylint: disable=protected-access
          Path(tmpdir) / consumer.RUNTIME_LEDGER_FILE
      )
      sms_runtime = next(row for row in runtime if row["pair_id"] == sms_pair)
      self.assertLen(sms_runtime["replacement_rounds"], 1)
      invalid_initial = next(
          item.attempt_id
          for item in calls[:3]
          if item.condition == "c1"
      )
      self.assertEqual(
          sms_runtime["replacement_rounds"][0]["trigger_attempt_ids"],
          [invalid_initial],
      )

  def test_valid_failure_never_authorizes_replacement(self):
    sms = _real_triplet(
        _real_paired_key(
            category="sms",
            app_id="sms_fossify_messages",
            semantic_task_id="SmsReply",
            instance_id=2,
        ),
        0,
        ("c1", "c2_g", "c2_o"),
    )
    executor = _InMemoryRealEpisodeExecutor(lambda _: "valid_failure")
    with tempfile.TemporaryDirectory() as tmpdir:
      runner = consumer.ScheduleConsumer(
          _small_real_bundle([sms]), Path(tmpdir), executor
      )
      self.assertEqual(runner.run_until_complete(halt_after_invalid=False), 0)
      self.assertLen(executor.calls, 3)
      selection = consumer._read_jsonl(  # pylint: disable=protected-access
          Path(tmpdir) / consumer.SELECTION_FILE
      )[0]
      self.assertEqual(selection["selected_round"], 0)
      self.assertEqual(selection["selection_status"], "selected_complete_triplet")

  def test_two_full_replacement_rounds_then_exhaustion(self):
    clock = _real_triplet(
        _real_paired_key(
            category="clock",
            app_id="clock_clockyou",
            semantic_task_id="ClockCreateTimer",
            instance_id=1,
        ),
        0,
        ("c2_o", "c2_g", "c1"),
    )

    def status(attempt: consumer.AttemptSpec) -> str:
      return (
          "invalid_infrastructure"
          if attempt.condition == "c1"
          else "valid_success"
      )

    executor = _InMemoryRealEpisodeExecutor(status)
    with tempfile.TemporaryDirectory() as tmpdir:
      runner = consumer.ScheduleConsumer(
          _small_real_bundle([clock]), Path(tmpdir), executor
      )
      self.assertEqual(runner.run_until_complete(halt_after_invalid=False), 0)
      self.assertLen(executor.calls, 9)
      self.assertEqual(
          [item.attempt_index for item in executor.calls],
          [0, 0, 0, 1, 1, 1, 2, 2, 2],
      )
      selection = consumer._read_jsonl(  # pylint: disable=protected-access
          Path(tmpdir) / consumer.SELECTION_FILE
      )[0]
      self.assertEqual(selection["selection_status"], "exhausted_invalid")
      self.assertIsNone(selection["selected_round"])
      self.assertEmpty(selection["selected_attempt_ids"])
      runtime = consumer._read_jsonl(  # pylint: disable=protected-access
          Path(tmpdir) / consumer.RUNTIME_LEDGER_FILE
      )[0]
      self.assertLen(runtime["replacement_rounds"], 2)

  def test_interrupted_started_attempt_is_consumed_as_invalid(self):
    contacts = _real_triplet(
        _real_paired_key(
            category="contacts",
            app_id="contacts_google_contacts",
            semantic_task_id="ContactsAddContact",
        ),
        0,
        ("c1", "c2_g", "c2_o"),
    )
    bundle = _small_real_bundle([contacts])
    first = consumer.AttemptSpec.from_row(contacts[0][0])
    executor = _InMemoryRealEpisodeExecutor(lambda _: "valid_success")
    with tempfile.TemporaryDirectory() as tmpdir:
      initialized = consumer.ScheduleConsumer(bundle, Path(tmpdir), executor)
      initialized.journal.append({
          "event": "started",
          "recorded_at": "2026-07-10T00:00:00+00:00",
          **consumer._attempt_provenance(first),  # pylint: disable=protected-access
      })
      runner = consumer.ScheduleConsumer(bundle, Path(tmpdir), executor)
      self.assertEqual(runner.run_until_complete(), 3)
      self.assertEmpty(executor.calls)
      finished = runner.journal.finished[first.attempt_id]
      self.assertEqual(finished["status"], "invalid_infrastructure")
      self.assertEqual(
          finished["reason_code"],
          "consumer_interrupted_before_terminal_contract",
      )


class BundleValidationTest(absltest.TestCase):

  def _ready_real_cohort(self) -> dict[str, Any]:
    cohort = json.loads(
        schedule_builder.DEFAULT_COHORT.read_text(encoding="utf-8")
    )
    cohort["status"] = "ready"
    for gate in cohort["eligibility_gates"].values():
      gate["status"] = "ready"
    verifier_gate = cohort["eligibility_gates"]["verifier_conformance"]
    verifier_gate.update({
        "qualified_adapter_count": 230,
        "unqualified_adapter_count": 0,
        "approval_status": "approved",
        "evidence_manifest_sha256": "a" * 64,
        "approval_record_sha256": "b" * 64,
        "known_unqualified_scored_adapters": [],
    })
    return cohort

  def _g6_real_cohort(self) -> dict[str, Any]:
    return json.loads(
        schedule_builder.DEFAULT_G6_DRY_RUN_COHORT.read_text(encoding="utf-8")
    )

  def test_complete_real_bundle_recompiles_exactly(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      cohort_path = root / "catbench_5cat_primary_cohort.json"
      cohort_path.write_text(
          json.dumps(self._ready_real_cohort(), indent=2) + "\n",
          encoding="utf-8",
      )
      cohort_sha = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
      schedule, ledger, manifest = schedule_builder.compile_frozen_schedule(
          self._ready_real_cohort(), cohort_sha
      )
      schedule_dir = root / "schedule"
      schedule_builder.write_frozen_schedule(
          schedule_dir, schedule, ledger, manifest
      )
      bundle = consumer.load_and_validate_bundle(schedule_dir, cohort_path)
      self.assertLen(bundle.schedule, 10_350)
      self.assertLen(bundle.ledger_seed, 3_450)
      self.assertEqual(
          {row["app_id"] for row in bundle.schedule},
          {
              app_id
              for spec in bundle.cohort["categories"].values()
              for app_id in spec["app_ids"]
          },
      )

  def test_exact_g6_bundle_recompiles_and_remains_discard_only(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      cohort_path = root / "catbench_5cat_g6_dryrun_cohort.json"
      cohort_path.write_bytes(
          schedule_builder.DEFAULT_G6_DRY_RUN_COHORT.read_bytes()
      )
      cohort = self._g6_real_cohort()
      cohort_sha = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
      schedule, ledger, manifest = schedule_builder.compile_frozen_schedule(
          cohort, cohort_sha
      )
      schedule_dir = root / "schedule"
      schedule_builder.write_frozen_schedule(
          schedule_dir, schedule, ledger, manifest
      )

      bundle = consumer.load_and_validate_bundle(schedule_dir, cohort_path)
      self.assertLen(bundle.schedule, 15)
      self.assertLen(bundle.ledger_seed, 5)
      self.assertFalse(bundle.schedule_manifest["analysis_eligible"])
      self.assertEqual(
          bundle.schedule_manifest["artifact_role"],
          "discard_only_never_primary_analysis",
      )
      real_tasks = consumer.resolve_real_tasks(bundle.cohort)
      self.assertLen(real_tasks, 5)
      self.assertEqual(
          {task.task_template for task in real_tasks.values()},
          {
              "SmsSendForSimpleSMSMessenger",
              "FilesCreateFolderForMaterialFiles",
              "MapsSearchPlaceForOsmAnd",
              "ContactsAddContactForFossifyContacts",
              "ClockCreateAlarmForClock",
          },
      )

      executor = _InMemoryRealEpisodeExecutor(lambda _: "valid_failure")
      state_root = root / "discarded_state"
      runner = consumer.ScheduleConsumer(bundle, state_root, executor)
      self.assertEqual(runner.run_until_complete(halt_after_invalid=False), 0)
      self.assertLen(executor.calls, 15)
      selections = consumer._read_jsonl(  # pylint: disable=protected-access
          state_root / consumer.SELECTION_FILE
      )
      self.assertLen(selections, 5)
      self.assertEqual(
          {row["artifact_role"] for row in selections},
          {"discard_only_never_primary_analysis"},
      )
      self.assertEqual(
          {row["analysis_eligible"] for row in selections}, {False}
      )
      journal = consumer._read_jsonl(  # pylint: disable=protected-access
          state_root / consumer.JOURNAL_FILE
      )
      self.assertEqual(
          {row["artifact_role"] for row in journal},
          {"discard_only_never_primary_analysis"},
      )

  def test_g6_plan_preflight_enumerates_only_five_preregistered_instances(self):
    args = argparse.Namespace(
        suite_family="android_world",
        cohort_manifest=str(schedule_builder.DEFAULT_G6_DRY_RUN_COHORT),
        tasks="",
        categories="sms,files,maps,contacts,clock",
        n_task_combinations=3,
        task_random_seed=30,
        fixed_task_seed=False,
    )
    scheduled = plan_preflight._enumerate_scheduled_tasks(  # pylint: disable=protected-access
        args
    )
    self.assertLen(scheduled, 5)
    self.assertEqual(
        {
            (
                row["category"],
                row["app_id"],
                row["semantic_task_id"],
                row["instance_id"],
            )
            for row in scheduled
        },
        {
            (
                block["category"],
                block["app_id"],
                block["semantic_task_id"],
                block["instance_id"],
            )
            for block in self._g6_real_cohort()["paired_blocks"]
        },
    )

  def test_extra_bundle_artifact_is_rejected(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      cohort = self._ready_real_cohort()
      cohort_path = root / "cohort.json"
      cohort_path.write_text(json.dumps(cohort) + "\n", encoding="utf-8")
      cohort_sha = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
      schedule, ledger, manifest = schedule_builder.compile_frozen_schedule(
          cohort, cohort_sha
      )
      schedule_dir = root / "schedule"
      schedule_builder.write_frozen_schedule(
          schedule_dir, schedule, ledger, manifest
      )
      (schedule_dir / "manual_subset.json").write_text("{}\n", encoding="utf-8")
      with self.assertRaisesRegex(
          consumer.ArtifactValidationError, "exactly the four compiler artifacts"
      ):
        consumer.load_and_validate_bundle(schedule_dir, cohort_path)

  def test_duplicate_json_keys_and_nan_are_rejected(self):
    with self.assertRaisesRegex(
        consumer.ArtifactValidationError, "Duplicate JSON key"
    ):
      consumer._strict_json_loads(  # pylint: disable=protected-access
          '{"app_id":"sms_quik_sms","app_id":"sms_google_messages"}',
          "in-memory-real-id-test",
      )
    with self.assertRaisesRegex(
        consumer.ArtifactValidationError, "Non-finite JSON constant"
    ):
      consumer._strict_json_loads(  # pylint: disable=protected-access
          '{"schedule_seed":NaN}', "in-memory-real-id-test"
      )

  def test_explicitly_blocked_cohort_cannot_be_consumed(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      cohort = self._ready_real_cohort()
      cohort["status"] = "blocked_pending_conformance"
      cohort["eligibility_gates"]["clock_clockyou"]["status"] = "blocked"
      cohort_path = root / "blocked_cohort.json"
      cohort_path.write_text(
          json.dumps(cohort, indent=2) + "\n", encoding="utf-8"
      )
      with self.assertRaises(schedule_builder.ScheduleBuildError):
        consumer.load_and_validate_bundle(
            root / "schedule", cohort_path
        )

  def test_arbitrary_preexisting_consumer_output_is_rejected(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      (root / "manual_selected_failures.json").write_text(
          "[]\n", encoding="utf-8"
      )
      with self.assertRaisesRegex(
          consumer.ArtifactValidationError, "preexisting content"
      ):
        consumer._validate_state_root_before_preflight(  # pylint: disable=protected-access
            root
        )

  def test_cli_has_no_slot_subset_or_retry_override(self):
    destinations = {
        action.dest for action in consumer._build_parser()._actions  # pylint: disable=protected-access
    }
    for prohibited in (
        "models",
        "app_ids",
        "tasks",
        "condition",
        "slot_id",
        "offset",
        "resume",
        "max_replacement_rounds",
        "snapshot_hook_timeout_seconds",
        "episode_runner_timeout_seconds",
        "allow_dirty",
    ):
      self.assertNotIn(prohibited, destinations)
    self.assertIn("model_endpoint_attestation_manifest", destinations)
    self.assertIn("installed_app_attestation_manifest", destinations)
    self.assertIn("verifier_conformance_manifest", destinations)
    self.assertIn("verifier_conformance_evidence_root", destinations)
    # pylint: disable=protected-access
    parser_actions = consumer._build_parser()._actions
    conformance_action = next(
        action
        for action in parser_actions
        if action.dest == "verifier_conformance_manifest"
    )
    self.assertTrue(conformance_action.required)
    evidence_root_action = next(
        action
        for action in parser_actions
        if action.dest == "verifier_conformance_evidence_root"
    )
    self.assertTrue(evidence_root_action.required)

  def test_source_failure_precedes_any_output_write(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      output_root = Path(tmpdir) / "catbench_primary_execution"
      argv = [
          "--schedule_dir=/real/frozen/schedule",
          "--cohort_manifest=/real/frozen/cohort.json",
          "--c2_g_breakdown_file=/real/frozen/c2_g.json",
          "--c2_g_attempt_audit=/real/frozen/c2_g_attempts.jsonl",
          "--c2_o_breakdown_file=/real/frozen/c2_o.json",
          "--c2_g_approval_manifest=/real/frozen/c2_g_approval.json",
          "--c2_o_approval_manifest=/real/frozen/c2_o_approval.json",
          "--base_snapshot_manifest=/real/frozen/base_snapshot.json",
          "--model_endpoint_attestation_manifest=/real/frozen/models.json",
          "--installed_app_attestation_manifest=/real/frozen/apps.json",
          "--verifier_conformance_manifest=/real/frozen/conformance.json",
          "--verifier_conformance_evidence_root=/real/frozen/evidence",
          "--snapshot_hook=/real/frozen/snapshot_hook",
          f"--output_root={output_root}",
          "--app_artifact_root=/real/frozen/apks",
          "--device_serial=emulator-5554",
          "--console_port=5554",
          "--grpc_port=8554",
      ]
      with mock.patch.object(
          consumer,
          "_source_revision_clean",
          side_effect=consumer.ArtifactValidationError("source is dirty"),
      ):
        self.assertEqual(consumer.main(argv), 2)
      self.assertFalse(output_root.exists())

  def test_primary_output_must_be_disjoint_from_source_checkout(self):
    with self.assertRaisesRegex(
        consumer.ArtifactValidationError, "disjoint from the source repository"
    ):
      consumer._validate_prewrite_locations(  # pylint: disable=protected-access
          consumer.REPO_ROOT / "benchmark" / "consumer-output", ()
      )

  def test_model_args_cannot_override_frozen_episode_flags(self):
    consumer._validate_frozen_model_args(  # pylint: disable=protected-access
        REAL_MODEL,
        (
            "--endpoint_url=http://127.0.0.1:8000",
            "--model_name=inclusionAI/UI-Venus-Navi-7B",
        ),
    )
    for prohibited in (
        "--tasks=SmsReplyForSimpleSMSMessenger",
        "--task-random-seed=99",
        "--checkpoint_dir=/tmp/manual-checkpoint",
        "--console_port=5556",
        "--flagfile=/tmp/manual.flags",
        "--nofixed_task_seed=true",
    ):
      with self.subTest(prohibited=prohibited), self.assertRaises(
          consumer.ArtifactValidationError
      ):
        consumer._validate_frozen_model_args(  # pylint: disable=protected-access
            REAL_MODEL, (prohibited,)
        )

  def test_observational_signer_audit_is_not_a_launch_attestation(self):
    cohort_path = schedule_builder.DEFAULT_COHORT
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    pins = consumer.load_app_pins(consumer.DEFAULT_PINS, cohort)
    bundle = consumer.FrozenBundle(
        cohort=cohort,
        cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        schedule_manifest={},
        schedule_manifest_sha256="a" * 64,
        schedule=(),
        ledger_seed=(),
        ledger_schema_sha256="b" * 64,
    )
    observed_audit = (
        consumer.BENCHMARK_ROOT
        / "docs"
        / "audits"
        / "pinned_app_signer_audit.json"
    )
    with self.assertRaisesRegex(
        consumer.ArtifactValidationError,
        "Installed app attestation schema_version",
    ):
      consumer.validate_installed_app_attestation(
          observed_audit,
          bundle=bundle,
          pins=pins,
          pins_sha256=hashlib.sha256(consumer.DEFAULT_PINS.read_bytes()).hexdigest(),
          base_snapshot={
              "snapshot_id": "catbench-api33-frozen-base",
              "snapshot_sha256": "c" * 64,
          },
      )

  def test_installed_attestation_must_include_each_exact_pinned_apk(self):
    cohort_path = schedule_builder.DEFAULT_COHORT
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    pins = consumer.load_app_pins(consumer.DEFAULT_PINS, cohort)
    cohort_sha256 = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    pins_sha256 = hashlib.sha256(consumer.DEFAULT_PINS.read_bytes()).hexdigest()
    bundle = consumer.FrozenBundle(
        cohort=cohort,
        cohort_sha256=cohort_sha256,
        schedule_manifest={},
        schedule_manifest_sha256="a" * 64,
        schedule=(),
        ledger_seed=(),
        ledger_schema_sha256="b" * 64,
    )
    base_snapshot = {
        "snapshot_id": "catbench-api33-frozen-base",
        "snapshot_sha256": "c" * 64,
    }
    payload = {
        "schema_version": 1,
        "release_id": REAL_RELEASE,
        "cohort_sha256": cohort_sha256,
        "app_pins_sha256": pins_sha256,
        "base_snapshot_id": base_snapshot["snapshot_id"],
        "base_snapshot_sha256": base_snapshot["snapshot_sha256"],
        "attestation_scope": "on_device_frozen_base_snapshot",
        "attestation_policy": (
            "exact_installed_apk_bytes_and_fully_verified_signers_for_frozen_roster"
        ),
        "approval_status": "approved_for_primary_release",
        "attestor_id": "release-attestor",
        "attested_at": "2026-07-10T00:00:00+00:00",
        "approver_id": "release-approver",
        "approved_at": "2026-07-10T00:00:00+00:00",
        "attestation_evidence_sha256": "d" * 64,
        "approval_evidence_sha256": "e" * 64,
        "collection_tool_sha256": "f" * 64,
        "apps": [
            {
                "app_id": app_id,
                "category": pin["category"],
                "package_name": pin["package_name"],
                "version_name": pin["version_name"],
                "version_code": pin["version_code"],
                "pinned_artifact_sha256": pin["apk_sha256"],
                "signature_verification_status": (
                    "fully_cryptographically_verified"
                ),
                "installed_apk_sha256": [pin["apk_sha256"]],
                "signer_leaf_certificate_sha256": ["1" * 64],
                "installed_bytes_evidence_sha256": "2" * 64,
                "signature_verification_evidence_sha256": "3" * 64,
                "verification_tool_sha256": "4" * 64,
            }
            for category in cohort["categories"].values()
            for app_id in category["app_ids"]
            for pin in (pins[app_id],)
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "installed_apps.json"
      path.write_text(json.dumps(payload), encoding="utf-8")
      consumer.validate_installed_app_attestation(
          path,
          bundle=bundle,
          pins=pins,
          pins_sha256=pins_sha256,
          base_snapshot=base_snapshot,
      )
      payload["apps"][0]["installed_apk_sha256"] = ["5" * 64]
      path.write_text(json.dumps(payload), encoding="utf-8")
      with self.assertRaisesRegex(
          consumer.ArtifactValidationError, "exact pinned APK bytes"
      ):
        consumer.validate_installed_app_attestation(
            path,
            bundle=bundle,
            pins=pins,
            pins_sha256=pins_sha256,
            base_snapshot=base_snapshot,
        )

  def test_verifier_conformance_accepts_exact_real_230_adapter_roster(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      evidence_root = Path(tmpdir) / "evidence"
      bundle, pins, base_snapshot, payload = _primary_conformance_fixture(
          evidence_root
      )
      self.assertLen(payload["records"], 230)
      path = Path(tmpdir) / "verifier_conformance.json"
      path.write_text(json.dumps(payload), encoding="utf-8")
      validated, inventory_sha256 = (
          consumer.validate_verifier_conformance_manifest(
              path,
              bundle=bundle,
              pins=pins,
              base_snapshot=base_snapshot,
              evidence_root=evidence_root,
          )
      )
      _, repeated_inventory_sha256 = (
          consumer.validate_verifier_conformance_manifest(
              path,
              bundle=bundle,
              pins=pins,
              base_snapshot=base_snapshot,
              evidence_root=evidence_root,
          )
      )
    self.assertTrue(validated["approved"])
    self.assertRegex(inventory_sha256, r"^[0-9a-f]{64}$")
    self.assertEqual(inventory_sha256, repeated_inventory_sha256)
    # pylint: disable=protected-access
    expected_keys = consumer._expected_verifier_adapter_keys(
        bundle
    )
    self.assertEqual(
        {
            (row["category"], row["app_id"], row["semantic_task_id"])
            for row in validated["records"]
        },
        set(expected_keys),
    )

  def test_verifier_conformance_rejects_wrong_release_base_or_docker(self):
    mutations = (
        ("release_id", "wrong-release", "release_id"),
        ("cohort_sha256", "0" * 64, "cohort_sha256"),
        ("base_snapshot_id", "wrong-base", "base_snapshot_id"),
        ("base_snapshot_sha256", "0" * 64, "base_snapshot_sha256"),
        (
            "docker_image",
            "android_world@sha256:" + "0" * 64,
            "docker_image",
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      evidence_root = Path(tmpdir) / "evidence"
      bundle, pins, base_snapshot, valid = _primary_conformance_fixture(
          evidence_root
      )
      path = Path(tmpdir) / "verifier_conformance.json"
      for field, value, expected_error in mutations:
        with self.subTest(field=field):
          payload = copy.deepcopy(valid)
          payload[field] = value
          path.write_text(json.dumps(payload), encoding="utf-8")
          with self.assertRaisesRegex(
              consumer.ArtifactValidationError, expected_error
          ):
            consumer.validate_verifier_conformance_manifest(
                path,
                bundle=bundle,
                pins=pins,
                base_snapshot=base_snapshot,
                evidence_root=evidence_root,
            )

  def test_verifier_conformance_rejects_incomplete_or_reused_evidence(self):
    mutations: tuple[tuple[str, Any, str], ...] = (
        (
            "missing_adapter",
            lambda payload: payload["records"].pop(),
            "record count mismatch",
        ),
        (
            "duplicate_adapter",
            lambda payload: payload["records"].__setitem__(
                -1, copy.deepcopy(payload["records"][0])
            ),
            "Duplicate verifier conformance adapter",
        ),
        (
            "missing_case",
            lambda payload: payload["records"][0]["evidence"].pop("stale"),
            "evidence key set mismatch",
        ),
        (
            "reused_evidence_id",
            lambda payload: payload["records"][1]["evidence"][
                "primitive_action_positive"
            ].__setitem__(
                "evidence_id",
                payload["records"][0]["evidence"][
                    "primitive_action_positive"
                ]["evidence_id"],
            ),
            "Duplicate verifier conformance evidence_id",
        ),
        (
            "wrong_observed_result",
            lambda payload: payload["records"][0]["evidence"][
                "wrong_value"
            ].__setitem__("observed_success", True),
            "wrong observed_success",
        ),
        (
            "missing_evidence_identity",
            lambda payload: payload["records"][0]["evidence"][
                "no_op"
            ].__setitem__("evidence_id", ""),
            "evidence_id must be a non-empty identity",
        ),
        (
            "evidence_hash_mismatch",
            lambda payload: payload["records"][0]["evidence"][
                "partial"
            ].__setitem__("action_trace_sha256", "0" * 64),
            "action_trace_path SHA-256 mismatch",
        ),
        (
            "wrong_apk_identity",
            lambda payload: payload["records"][0].__setitem__(
                "apk_sha256", "0" * 64
            ),
            "apk_sha256 mismatch",
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      evidence_root = Path(tmpdir) / "evidence"
      bundle, pins, base_snapshot, valid = _primary_conformance_fixture(
          evidence_root
      )
      path = Path(tmpdir) / "verifier_conformance.json"
      for name, mutate, expected_error in mutations:
        with self.subTest(name=name):
          payload = copy.deepcopy(valid)
          mutate(payload)
          path.write_text(json.dumps(payload), encoding="utf-8")
          with self.assertRaisesRegex(
              consumer.ArtifactValidationError, expected_error
          ):
            consumer.validate_verifier_conformance_manifest(
                path,
                bundle=bundle,
                pins=pins,
                base_snapshot=base_snapshot,
                evidence_root=evidence_root,
            )

  def test_verifier_conformance_rejects_unsafe_or_nonimmutable_files(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      evidence_root = root / "evidence"
      bundle, pins, base_snapshot, valid = _primary_conformance_fixture(
          evidence_root
      )
      path = root / "verifier_conformance.json"

      def validate(payload: Mapping[str, Any]) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")
        consumer.validate_verifier_conformance_manifest(
            path,
            bundle=bundle,
            pins=pins,
            base_snapshot=base_snapshot,
            evidence_root=evidence_root,
        )

      for name, relative_path, expected_error in (
          ("traversal", "../outside.json", "canonical relative path"),
          ("absolute", "/tmp/outside.json", "canonical relative path"),
          ("missing", "missing/action.jsonl", "missing or unreadable"),
      ):
        with self.subTest(name=name):
          payload = copy.deepcopy(valid)
          payload["records"][0]["evidence"][
              "primitive_action_positive"
          ]["action_trace_path"] = relative_path
          with self.assertRaisesRegex(
              consumer.ArtifactValidationError, expected_error
          ):
            validate(payload)

      with self.subTest(name="reused_path"):
        payload = copy.deepcopy(valid)
        case = payload["records"][0]["evidence"][
            "primitive_action_positive"
        ]
        case["state_evidence_path"] = case["action_trace_path"]
        case["state_evidence_sha256"] = case["action_trace_sha256"]
        with self.assertRaisesRegex(
            consumer.ArtifactValidationError, "Reused.*evidence path"
        ):
          validate(payload)

      malicious = evidence_root / "malicious"
      malicious.mkdir()
      target = evidence_root.joinpath(
          *valid["records"][0]["evidence"]["no_op"][
              "action_trace_path"
          ].split("/")
      )
      link = malicious / "link.json"
      link.symlink_to(target)
      with self.subTest(name="symlink"):
        payload = copy.deepcopy(valid)
        case = payload["records"][0]["evidence"]["no_op"]
        case["action_trace_path"] = "malicious/link.json"
        case["action_trace_sha256"] = hashlib.sha256(
            target.read_bytes()
        ).hexdigest()
        with self.assertRaisesRegex(
            consumer.ArtifactValidationError, "traverses a symlink"
        ):
          validate(payload)

      empty = malicious / "empty.json"
      empty.write_bytes(b"")
      with self.subTest(name="empty"):
        payload = copy.deepcopy(valid)
        case = payload["records"][0]["evidence"]["partial"]
        case["action_trace_path"] = "malicious/empty.json"
        case["action_trace_sha256"] = hashlib.sha256(b"").hexdigest()
        with self.assertRaisesRegex(
            consumer.ArtifactValidationError, "is empty"
        ):
          validate(payload)

      directory = malicious / "directory"
      directory.mkdir()
      with self.subTest(name="non_regular"):
        payload = copy.deepcopy(valid)
        case = payload["records"][0]["evidence"]["unrelated"]
        case["state_evidence_path"] = "malicious/directory"
        case["state_evidence_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            consumer.ArtifactValidationError, "not a regular file"
        ):
          validate(payload)

  def test_verifier_evidence_root_is_disjoint_from_source_and_output(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      evidence_root = root / "evidence"
      evidence_root.mkdir()
      output_root = root / "output"
      self.assertEqual(
          consumer._validate_verifier_evidence_root_location(  # pylint: disable=protected-access
              evidence_root, output_root
          ),
          evidence_root.resolve(),
      )
      with self.assertRaisesRegex(
          consumer.ArtifactValidationError, "disjoint from output_root"
      ):
        consumer._validate_verifier_evidence_root_location(  # pylint: disable=protected-access
            evidence_root, evidence_root / "consumer-state"
        )
      link = root / "evidence-link"
      link.symlink_to(evidence_root, target_is_directory=True)
      with self.assertRaisesRegex(
          consumer.ArtifactValidationError, "non-symlink directory"
      ):
        consumer._validate_verifier_evidence_root_location(  # pylint: disable=protected-access
            link, output_root
        )
    with self.assertRaisesRegex(
        consumer.ArtifactValidationError, "disjoint from the source repository"
    ):
      consumer._validate_verifier_evidence_root_location(  # pylint: disable=protected-access
          consumer.BENCHMARK_ROOT, Path("/tmp/catbench-consumer-state")
      )

  def test_verifier_conformance_rejects_unapproved_skipped_or_overwritten(self):
    mutations: tuple[tuple[str, Any, str], ...] = (
        (
            "manifest_unapproved",
            lambda payload: payload.__setitem__("approved", False),
            "manifest is unapproved",
        ),
        (
            "adapter_unapproved",
            lambda payload: payload["records"][0].__setitem__(
                "approved", False
            ),
            "adapter is unapproved",
        ),
        (
            "adapter_skipped",
            lambda payload: payload["records"][0].__setitem__("skipped", True),
            "adapter was skipped",
        ),
        (
            "adapter_overwritten",
            lambda payload: payload["records"][0].__setitem__(
                "overwritten", True
            ),
            "adapter evidence was overwritten",
        ),
        (
            "case_skipped",
            lambda payload: payload["records"][0]["evidence"][
                "no_op"
            ].__setitem__("skipped", True),
            "case no_op was skipped",
        ),
        (
            "case_overwritten",
            lambda payload: payload["records"][0]["evidence"][
                "partial"
            ].__setitem__("overwritten", True),
            "case partial was overwritten",
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      evidence_root = Path(tmpdir) / "evidence"
      bundle, pins, base_snapshot, valid = _primary_conformance_fixture(
          evidence_root
      )
      path = Path(tmpdir) / "verifier_conformance.json"
      for name, mutate, expected_error in mutations:
        with self.subTest(name=name):
          payload = copy.deepcopy(valid)
          mutate(payload)
          path.write_text(json.dumps(payload), encoding="utf-8")
          with self.assertRaisesRegex(
              consumer.ArtifactValidationError, expected_error
          ):
            consumer.validate_verifier_conformance_manifest(
                path,
                bundle=bundle,
                pins=pins,
                base_snapshot=base_snapshot,
                evidence_root=evidence_root,
            )

  def test_verifier_conformance_uses_strict_duplicate_key_json(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      evidence_root = Path(tmpdir) / "evidence"
      bundle, pins, base_snapshot, _ = _primary_conformance_fixture(
          evidence_root
      )
      path = Path(tmpdir) / "verifier_conformance.json"
      path.write_text(
          '{"approved":true,"approved":false}', encoding="utf-8"
      )
      with self.assertRaisesRegex(
          consumer.ArtifactValidationError, "Duplicate JSON key"
      ):
        consumer.validate_verifier_conformance_manifest(
            path,
            bundle=bundle,
            pins=pins,
            base_snapshot=base_snapshot,
            evidence_root=evidence_root,
        )


class ExecutorContractTest(absltest.TestCase):

  def test_episode_environment_overrides_ambient_runtime_controls(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      executor, attempt, task, model, pin = _execution_fixture(Path(tmpdir))
      policy_environment = dict(
          schedule_builder.FROZEN_EPISODE_RUNTIME_POLICY["environment"]
      )
      poisoned = {key: "ambient-drift" for key in policy_environment}
      with mock.patch.dict(consumer.os.environ, poisoned, clear=False):
        env = executor._episode_env(  # pylint: disable=protected-access
            attempt,
            task,
            model,
            pin,
            "a" * 64,
        )
      self.assertEqual(
          {key: env[key] for key in policy_environment}, policy_environment
      )
      self.assertEqual(
          env["CATBENCH_EPISODE_RUNTIME_POLICY_SHA256"],
          schedule_builder.episode_runtime_policy_sha256(),
      )


  def test_exact_c1_provenance_contract_is_accepted(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      executor, attempt, task, model, pin = _execution_fixture(Path(tmpdir))
      runner_hash = "a" * 64
      episode = _valid_episode(
          executor,
          attempt,
          task,
          model,
          pin,
          runner_config_sha256=runner_hash,
      )
      self.assertEqual(
          executor._validate_episode(  # pylint: disable=protected-access
              episode, attempt, task, model, pin, runner_hash
          ),
          ("valid_failure", 0.0),
      )

  def test_missing_or_wrong_provenance_is_infrastructure_invalid(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      executor, attempt, task, model, pin = _execution_fixture(Path(tmpdir))
      runner_hash = "a" * 64
      valid = _valid_episode(
          executor,
          attempt,
          task,
          model,
          pin,
          runner_config_sha256=runner_hash,
      )
      for field in (
          "task_template",
          "instance_id",
          "release_id",
          "release_purpose",
          "artifact_role",
          "analysis_eligible",
          "episode_runtime_policy_sha256",
          "pair_id",
          "attempt_id",
          "snapshot_clone_id",
          "model_revision",
          "runner_config_sha256",
          "installed_app_attestation_sha256",
          "task_random_seed",
          "plan_file_sha256",
      ):
        for mutation in ("missing", "wrong"):
          with self.subTest(field=field, mutation=mutation):
            episode = copy.deepcopy(valid)
            if mutation == "missing":
              episode.pop(field)
            else:
              episode[field] = "wrong"
            with self.assertRaisesRegex(
                consumer.InfrastructureInvalid,
                f"episode_provenance_mismatch:{field}",
            ):
              executor._validate_episode(  # pylint: disable=protected-access
                  episode, attempt, task, model, pin, runner_hash
              )

  def test_c2_plan_identity_and_content_hash_are_exact(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      executor, attempt, task, model, pin = _execution_fixture(
          Path(tmpdir), condition="c2_g"
      )
      runner_hash = "a" * 64
      valid = _valid_episode(
          executor,
          attempt,
          task,
          model,
          pin,
          runner_config_sha256=runner_hash,
      )
      self.assertEqual(
          executor._validate_episode(  # pylint: disable=protected-access
              valid, attempt, task, model, pin, runner_hash
          ),
          ("valid_failure", 0.0),
      )
      wrong_key = copy.deepcopy(valid)
      wrong_key["task_breakdown_metadata"]["plan_key"] = (
          "SmsReply|instance=0|" + "8" * 64
      )
      with self.assertRaisesRegex(
          consumer.InfrastructureInvalid, "c2_metadata_mismatch:plan_key"
      ):
        executor._validate_episode(  # pylint: disable=protected-access
            wrong_key, attempt, task, model, pin, runner_hash
        )
      tampered_text = copy.deepcopy(valid)
      tampered_text["task_breakdown_text"] += "\n3. Open an unrelated app."
      with self.assertRaisesRegex(
          consumer.InfrastructureInvalid, "c2_plan_sha256_mismatch"
      ):
        executor._validate_episode(  # pylint: disable=protected-access
            tampered_text, attempt, task, model, pin, runner_hash
        )

  def test_checkpoint_must_be_exactly_one_regular_expected_episode(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      checkpoint_dir = Path(tmpdir) / "checkpoint"
      checkpoint_dir.mkdir()
      expected = checkpoint_dir / "SmsSendForSimpleSMSMessenger_0.pkl.gz"
      with gzip.open(expected, "wb") as handle:
        pickle.dump([{"task_template": "SmsSendForSimpleSMSMessenger"}], handle)
      self.assertEqual(
          consumer._load_exact_checkpoint(  # pylint: disable=protected-access
              checkpoint_dir, expected
          )["task_template"],
          "SmsSendForSimpleSMSMessenger",
      )
      extra = checkpoint_dir / "SmsSendForSimpleSMSMessenger_1.pkl.gz"
      with gzip.open(extra, "wb") as handle:
        pickle.dump([{"task_template": "SmsSendForSimpleSMSMessenger"}], handle)
      with self.assertRaisesRegex(
          consumer.InfrastructureInvalid, "checkpoint_file_set_mismatch"
      ):
        consumer._load_exact_checkpoint(  # pylint: disable=protected-access
            checkpoint_dir, expected
        )
      extra.unlink()
      with gzip.open(expected, "wb") as handle:
        pickle.dump(
            [
                {"task_template": "SmsSendForSimpleSMSMessenger"},
                {"task_template": "SmsSendForSimpleSMSMessenger"},
            ],
            handle,
        )
      with self.assertRaisesRegex(
          consumer.InfrastructureInvalid, "checkpoint_not_exactly_one_episode"
      ):
        consumer._load_exact_checkpoint(  # pylint: disable=protected-access
            checkpoint_dir, expected
        )

  def test_primary_runner_failure_survives_snapshot_release_failure(self):
    class _ReleaseFailureExecutor(consumer.SubprocessEpisodeExecutor):

      def _hook(self, operation, attempt, attempt_dir):
        del attempt
        if operation == "clone_activate":
          return attempt_dir / "snapshot_clone_activate_receipt.json"
        raise OSError("release_broke")

      def _device_pin_preflight(self, task, pin):
        del task, pin

      def _runner_config_sha256(self, model):
        del model
        return "a" * 64

    with tempfile.TemporaryDirectory() as tmpdir:
      base_executor, attempt, _, _, _ = _execution_fixture(Path(tmpdir))
      executor = _ReleaseFailureExecutor(base_executor.context)
      with mock.patch.object(
          consumer.subprocess,
          "run",
          return_value=SimpleNamespace(returncode=17),
      ):
        outcome = executor.execute(attempt)
      self.assertEqual(outcome.status, "invalid_infrastructure")
      self.assertEqual(
          outcome.reason_code,
          "runner_exit_code:17;secondary_snapshot_release_failed:release_broke",
      )
      contract = json.loads(Path(outcome.result_contract_path).read_text())
      self.assertEqual(contract["reason_code"], outcome.reason_code)
      self.assertEqual(contract["status"], "invalid_infrastructure")


class StateIntegrityTest(absltest.TestCase):

  def _completed_real_state(self, root: Path) -> tuple[consumer.FrozenBundle, Any]:
    maps = _real_triplet(
        _real_paired_key(
            category="maps",
            app_id="maps_osmand",
            semantic_task_id="MapsSearchPlace",
        ),
        0,
        ("c1", "c2_g", "c2_o"),
    )
    bundle = _small_real_bundle([maps])
    executor = _InMemoryRealEpisodeExecutor(lambda _: "valid_success")
    runner = consumer.ScheduleConsumer(bundle, root, executor)
    self.assertEqual(runner.run_until_complete(halt_after_invalid=False), 0)
    return bundle, executor

  def test_committed_selection_tamper_is_rejected_on_replay(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      bundle, executor = self._completed_real_state(root)
      selection = root / consumer.SELECTION_FILE
      selection.write_text(
          selection.read_text(encoding="utf-8") + "\n", encoding="utf-8"
      )
      with self.assertRaisesRegex(
          consumer.ArtifactValidationError, "hash mismatch"
      ):
        consumer.ScheduleConsumer(bundle, root, executor)

  def test_attempt_journal_hash_tamper_is_rejected(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      self._completed_real_state(root)
      journal = root / consumer.JOURNAL_FILE
      text = journal.read_text(encoding="utf-8")
      self.assertIn("valid_success", text)
      journal.write_text(
          text.replace("valid_success", "valid_failure", 1),
          encoding="utf-8",
      )
      with self.assertRaisesRegex(
          consumer.ArtifactValidationError, "event hash mismatch"
      ):
        consumer.Journal(journal)


if __name__ == "__main__":
  absltest.main()
