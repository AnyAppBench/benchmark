"""Tests for the non-executing frozen CATBench multi-condition scheduler."""

from __future__ import annotations

import collections
import copy
import hashlib
import json
from pathlib import Path
import tempfile

from absl.testing import absltest

import build_catbench_frozen_schedule as scheduler


REAL_MODEL_IDS = {
    "UI-Venus-7B",
    "GUI-Owl-7B",
    "MAI-UI-8B",
    "UI Voyager-4B",
    "Qwen3-VL-8B",
}
REAL_APP_IDS = {
    "sms_simple_sms_messenger",
    "sms_fossify_messages",
    "sms_quik_sms",
    "sms_google_messages",
    "files_material_files",
    "files_amaze",
    "files_fossify_file_manager",
    "files_total_commander",
    "files_x_plore_file_manager",
    "maps_osmand",
    "maps_organic_maps",
    "maps_comaps",
    "contacts_google_contacts",
    "contacts_fossify_contacts",
    "contacts_connect_you",
    "contacts_simple_contacts_pro_se",
    "contacts_right_contact",
    "clock_clock",
    "clock_simple_clock",
    "clock_google_clock",
    "clock_clockyou",
    "clock_chrono",
    "clock_fossify_clock",
}


def _raw_cohort() -> dict:
  return json.loads(scheduler.DEFAULT_COHORT.read_text(encoding="utf-8"))


def _ready_cohort() -> dict:
  cohort = copy.deepcopy(_raw_cohort())
  cohort["status"] = "ready"
  cohort["eligibility_gates"]["clock_clockyou"]["status"] = "ready"
  verifier_gate = cohort["eligibility_gates"]["verifier_conformance"]
  verifier_gate.update({
      "status": "ready",
      "qualified_adapter_count": 230,
      "unqualified_adapter_count": 0,
      "approval_status": "approved",
      "evidence_manifest_sha256": "a" * 64,
      "approval_record_sha256": "b" * 64,
      "known_unqualified_scored_adapters": [],
  })
  return cohort


def _g6_dry_run_cohort() -> dict:
  return json.loads(
      scheduler.DEFAULT_G6_DRY_RUN_COHORT.read_text(encoding="utf-8")
  )


class FrozenScheduleTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.cohort = _ready_cohort()
    cls.schedule, cls.ledger, cls.manifest = (
        scheduler.compile_frozen_schedule(cls.cohort, "a" * 64)
    )

  def test_checked_in_production_cohort_is_blocked_pending_g3(self):
    with self.assertRaisesRegex(
        scheduler.ScheduleBuildError,
        "blocked_pending_g3_verifier_conformance|qualified_adapter_count",
    ):
      scheduler.validate_primary_cohort(_raw_cohort())

  def test_blocked_cohort_writes_no_schedule_artifact(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cohort = _raw_cohort()
      cohort_path = Path(tmpdir) / "blocked_cohort.json"
      cohort_path.write_text(
          json.dumps(cohort, indent=2) + "\n", encoding="utf-8"
      )
      output_dir = Path(tmpdir) / "must_not_exist"
      with self.assertRaises(scheduler.ScheduleBuildError):
        scheduler.build_from_paths(cohort_path, output_dir)
      self.assertFalse(output_dir.exists())

  def test_exact_real_primary_schedule(self):
    self.assertLen(self.schedule, 10_350)
    self.assertLen(self.ledger, 3_450)
    self.assertEqual(
        collections.Counter(row["condition"] for row in self.schedule),
        {"c1": 3_450, "c2_g": 3_450, "c2_o": 3_450},
    )
    self.assertEqual({row["model"] for row in self.schedule}, REAL_MODEL_IDS)
    self.assertEqual({row["app_id"] for row in self.schedule}, REAL_APP_IDS)
    self.assertEqual(self.manifest["launch_capability"], False)
    self.assertEqual(self.manifest["selective_rerun_permitted"], False)
    self.assertTrue(self.manifest["analysis_eligible"])
    self.assertEqual(
        self.manifest["artifact_role"], "primary_analysis_candidate"
    )
    self.assertEqual(
        {row["semantic_origin"] for row in self.schedule},
        {
            scheduler.SEMANTIC_ORIGIN_AW,
            scheduler.SEMANTIC_ORIGIN_NEW,
        },
    )
    self.assertEqual(
        self.manifest["semantic_origin_template_counts"],
        {
            scheduler.SEMANTIC_ORIGIN_AW: 16,
            scheduler.SEMANTIC_ORIGIN_NEW: 34,
        },
    )

  def test_sms_send_to_contact_is_immutable_catbench_new_origin(self):
    sms = self.cohort["categories"]["sms"]
    self.assertEqual(
        sms["semantic_origins"]["SmsSendToContact"],
        scheduler.SEMANTIC_ORIGIN_NEW,
    )
    self.assertEqual(
        collections.Counter(sms["semantic_origins"].values()),
        {
            scheduler.SEMANTIC_ORIGIN_AW: 5,
            scheduler.SEMANTIC_ORIGIN_NEW: 5,
        },
    )
    self.assertEqual(
        {
            row["semantic_origin"]
            for row in self.schedule
            if row["semantic_task_id"] == "SmsSendToContact"
        },
        {scheduler.SEMANTIC_ORIGIN_NEW},
    )

  def test_semantic_origin_drift_is_blocked(self):
    cohort = _ready_cohort()
    cohort["categories"]["sms"]["semantic_origins"]["SmsSendToContact"] = (
        scheduler.SEMANTIC_ORIGIN_AW
    )
    with self.assertRaisesRegex(
        scheduler.ScheduleBuildError, "semantic_origins"
    ):
      scheduler.validate_primary_cohort(cohort)

  def test_g3_gate_cannot_be_marked_ready_with_unqualified_adapters(self):
    for field, value in (
        ("qualified_adapter_count", 229),
        ("unqualified_adapter_count", 1),
        ("approval_status", "not_approved"),
        ("evidence_manifest_sha256", ""),
        (
            "known_unqualified_scored_adapters",
            ["contacts/contacts_right_contact/ContactsAddContact"],
        ),
    ):
      cohort = _ready_cohort()
      cohort["eligibility_gates"]["verifier_conformance"][field] = value
      with self.subTest(field=field), self.assertRaises(
          scheduler.ScheduleBuildError
      ):
        scheduler.validate_primary_cohort(cohort)
    expected_policy_sha256 = scheduler.episode_runtime_policy_sha256()
    self.assertEqual(
        self.manifest["episode_runtime_policy"],
        scheduler.FROZEN_EPISODE_RUNTIME_POLICY,
    )
    self.assertEqual(
        self.manifest["episode_runtime_policy_sha256"],
        expected_policy_sha256,
    )
    self.assertEqual(
        {
            row["episode_runtime_policy_sha256"]
            for row in self.schedule
        },
        {expected_policy_sha256},
    )

  def test_exact_distinct_g6_real_task_schedule_is_discard_only(self):
    cohort = _g6_dry_run_cohort()
    schedule, ledger, manifest = scheduler.compile_frozen_schedule(
        cohort, "b" * 64
    )

    self.assertLen(schedule, 15)
    self.assertLen(ledger, 5)
    self.assertEqual(
        collections.Counter(row["condition"] for row in schedule),
        {"c1": 5, "c2_g": 5, "c2_o": 5},
    )
    paired_keys = [row["paired_key"] for row in ledger]
    expected_keys = (
        scheduler._g6_expected_paired_blocks()  # pylint: disable=protected-access
    )
    self.assertEqual(
        {
            scheduler._canonical_json(key)  # pylint: disable=protected-access
            for key in paired_keys
        },
        {
            scheduler._canonical_json(key)  # pylint: disable=protected-access
            for key in expected_keys
        },
    )
    self.assertEqual({row["model"] for row in schedule}, REAL_MODEL_IDS)
    self.assertEqual(
        {row["category"] for row in schedule},
        {"sms", "files", "maps", "contacts", "clock"},
    )
    self.assertEqual(
        {row["app_id"] for row in schedule},
        {
            "sms_simple_sms_messenger",
            "files_material_files",
            "maps_osmand",
            "contacts_fossify_contacts",
            "clock_clock",
        },
    )
    self.assertEqual({row["analysis_eligible"] for row in schedule}, {False})
    self.assertEqual(
        {row["artifact_role"] for row in schedule},
        {"discard_only_never_primary_analysis"},
    )
    self.assertFalse(manifest["analysis_eligible"])
    self.assertFalse(manifest["primary_reporter_acceptance_permitted"])
    self.assertEqual(
        manifest["artifact_role"], "discard_only_never_primary_analysis"
    )

  def test_g6_identity_or_discard_policy_drift_is_blocked(self):
    mutations = []
    cohort = _g6_dry_run_cohort()
    cohort["paired_blocks"][0]["model"] = "GUI-Owl-7B"
    mutations.append(cohort)
    cohort = _g6_dry_run_cohort()
    cohort["categories"]["contacts"]["app_ids"] = ["contacts_connect_you"]
    cohort["paired_blocks"][3]["app_id"] = "contacts_connect_you"
    mutations.append(cohort)
    cohort = _g6_dry_run_cohort()
    cohort["paired_blocks"][4]["semantic_task_id"] = "ClockCreateTimer"
    mutations.append(cohort)
    cohort = _g6_dry_run_cohort()
    cohort["paired_blocks"][2]["instance_id"] = 1
    mutations.append(cohort)
    cohort = _g6_dry_run_cohort()
    cohort["analysis_eligible"] = True
    mutations.append(cohort)
    cohort = _g6_dry_run_cohort()
    cohort["release_id"] = scheduler.PRIMARY_RELEASE_ID
    mutations.append(cohort)

    for mutation in mutations:
      with self.subTest(mutation=mutation), self.assertRaises(
          scheduler.ScheduleBuildError
      ):
        scheduler.validate_frozen_cohort(mutation)

  def test_unknown_release_cannot_be_used_as_a_small_cohort_escape(self):
    cohort = _g6_dry_run_cohort()
    cohort["release_id"] = "catbench_acl_revision_5cat_g6_dryrun_v2"
    with self.assertRaisesRegex(scheduler.ScheduleBuildError, "Unknown frozen"):
      scheduler.validate_frozen_cohort(cohort)

  def test_g6_bundle_compiles_from_only_the_checked_exact_cohort(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      output_dir = Path(tmpdir) / "g6_schedule"
      manifest = scheduler.build_from_paths(
          scheduler.DEFAULT_G6_DRY_RUN_COHORT, output_dir
      )
      self.assertEqual(manifest["episode_slot_count"], 15)
      self.assertEqual(manifest["paired_block_count"], 5)
      self.assertLen(
          (output_dir / scheduler.SCHEDULE_FILE)
          .read_text(encoding="utf-8")
          .splitlines(),
          15,
      )

  def test_conditions_are_interleaved_in_randomized_complete_blocks(self):
    blocks = collections.defaultdict(list)
    for row in self.schedule:
      blocks[row["block_order"]].append(row)
    self.assertLen(blocks, 3_450)
    condition_orders = set()
    for block_order in range(3_450):
      rows = sorted(
          blocks[block_order], key=lambda row: row["within_block_order"]
      )
      self.assertEqual(
          [row["global_order"] for row in rows],
          list(range(block_order * 3, block_order * 3 + 3)),
      )
      self.assertEqual(
          {row["condition"] for row in rows},
          {"c1", "c2_g", "c2_o"},
      )
      self.assertLen({row["snapshot_clone_id"] for row in rows}, 3)
      self.assertLen({row["snapshot_family_id"] for row in rows}, 1)
      condition_orders.add(tuple(row["condition"] for row in rows))
    self.assertLen(condition_orders, 6)

  def test_schedule_is_deterministic(self):
    schedule_again, ledger_again, manifest_again = (
        scheduler.compile_frozen_schedule(self.cohort, "a" * 64)
    )
    self.assertEqual(schedule_again, self.schedule)
    self.assertEqual(ledger_again, self.ledger)
    self.assertEqual(manifest_again, self.manifest)

  def test_replacement_ledger_allows_only_two_full_triplet_rounds(self):
    schema = scheduler.replacement_ledger_schema()
    replacement = schema["properties"]["replacement_rounds"]
    self.assertEqual(replacement["maxItems"], 2)
    required_conditions = schema["$defs"]["replacementRound"]["properties"][
        "condition_attempts"
    ]["required"]
    self.assertEqual(required_conditions, ["c1", "c2_g", "c2_o"])
    self.assertEqual(
        replacement["prefixItems"][0]["allOf"][1]["properties"][
            "round_index"
        ]["const"],
        1,
    )
    self.assertEqual(
        replacement["prefixItems"][1]["allOf"][1]["properties"][
            "round_index"
        ]["const"],
        2,
    )
    for record in self.ledger:
      self.assertEqual(record["max_replacement_rounds"], 2)
      self.assertEqual(record["selection_unit"], "full_condition_triplet")
      self.assertFalse(record["outcome_selected_replacement_permitted"])
      self.assertEmpty(record["replacement_rounds"])
      self.assertLen(record["authorized_replacement_rounds"], 2)
      for replacement_round in record["authorized_replacement_rounds"]:
        self.assertEqual(replacement_round["scheduled"], False)
        self.assertEqual(
            set(replacement_round["condition_attempts"]),
            {"c1", "c2_g", "c2_o"},
        )

  def test_roster_or_math_drift_is_blocked(self):
    cohort = _ready_cohort()
    cohort["categories"]["clock"]["app_ids"].remove("clock_clockyou")
    with self.assertRaisesRegex(
        scheduler.ScheduleBuildError, "clock must contain 6 apps"
    ):
      scheduler.validate_primary_cohort(cohort)

    cohort = _ready_cohort()
    cohort["expected"]["episodes_all_conditions"] = 9_900
    with self.assertRaisesRegex(
        scheduler.ScheduleBuildError,
        "expected.episodes_all_conditions must equal 10350",
    ):
      scheduler.validate_primary_cohort(cohort)

  def test_count_preserving_nonprimary_real_roster_is_blocked(self):
    cohort = _ready_cohort()
    cohort["models"][0] = "UI-Venus-72B"
    with self.assertRaisesRegex(
        scheduler.ScheduleBuildError, "exact frozen roster"
    ):
      scheduler.validate_primary_cohort(cohort)

    cohort = _ready_cohort()
    cohort["categories"]["maps"]["app_ids"][-1] = "maps_google_maps"
    with self.assertRaisesRegex(
        scheduler.ScheduleBuildError, "exact frozen real-app roster"
    ):
      scheduler.validate_primary_cohort(cohort)

    cohort = _ready_cohort()
    cohort["categories"]["clock"]["semantic_task_ids"][0] = (
        "ClockNavigateToAlarmTab"
    )
    with self.assertRaisesRegex(
        scheduler.ScheduleBuildError, "exact frozen semantic roster"
    ):
      scheduler.validate_primary_cohort(cohort)

  def test_unapproved_or_blocked_gate_is_blocked(self):
    cohort = _ready_cohort()
    cohort["eligibility_gates"]["sms_google_messages"] = {
        "required_for_primary": True,
        "status": "blocked",
    }
    with self.assertRaisesRegex(
        scheduler.ScheduleBuildError,
        "eligibility_gates must contain exactly|required eligibility gate",
    ):
      scheduler.validate_primary_cohort(cohort)

  def test_schedule_seed_or_release_identity_drift_is_blocked(self):
    cohort = _ready_cohort()
    cohort["schedule_seed"] += 1
    with self.assertRaisesRegex(
        scheduler.ScheduleBuildError, "schedule_seed must equal"
    ):
      scheduler.validate_primary_cohort(cohort)

  def test_episode_runtime_policy_drift_is_blocked(self):
    mutations = []
    cohort = _ready_cohort()
    cohort.pop("episode_runtime_policy")
    mutations.append(cohort)
    cohort = _ready_cohort()
    cohort["episode_runtime_policy"]["environment"][
        "CATBENCH_VERIFIER_SETTLE_ATTEMPTS"
    ] = "20"
    mutations.append(cohort)
    cohort = _ready_cohort()
    cohort["episode_runtime_policy"]["environment"][
        "CATBENCH_CLEAR_APP_DATA_METHOD"
    ] = "root_rm"
    mutations.append(cohort)
    cohort = _ready_cohort()
    cohort["episode_runtime_policy"]["environment"][
        "CATBENCH_STRICT_CATEGORY_ISOLATION"
    ] = "0"
    mutations.append(cohort)

    for mutation in mutations:
      with self.subTest(mutation=mutation), self.assertRaisesRegex(
          scheduler.ScheduleBuildError, "episode_runtime_policy"
      ):
        scheduler.validate_primary_cohort(mutation)

    cohort = _ready_cohort()
    cohort["release_id"] = "catbench_acl_revision_5cat_v2"
    with self.assertRaisesRegex(
        scheduler.ScheduleBuildError, "release_id must equal"
    ):
      scheduler.validate_primary_cohort(cohort)

  def test_bundle_hashes_match_written_real_schedule(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      output_dir = Path(tmpdir) / "frozen_schedule"
      scheduler.write_frozen_schedule(
          output_dir, self.schedule, self.ledger, self.manifest
      )
      manifest = json.loads(
          (output_dir / scheduler.MANIFEST_FILE).read_text(encoding="utf-8")
      )
      schedule_path = output_dir / scheduler.SCHEDULE_FILE
      ledger_path = output_dir / scheduler.LEDGER_FILE
      self.assertEqual(
          hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
          manifest["outputs"][scheduler.SCHEDULE_FILE]["sha256"],
      )
      self.assertEqual(
          hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
          manifest["outputs"][scheduler.LEDGER_FILE]["sha256"],
      )
      self.assertLen(schedule_path.read_text(encoding="utf-8").splitlines(), 10_350)
      self.assertLen(ledger_path.read_text(encoding="utf-8").splitlines(), 3_450)

      with self.assertRaisesRegex(
          scheduler.ScheduleBuildError, "Refusing to merge"
      ):
        scheduler.write_frozen_schedule(
            output_dir, self.schedule, self.ledger, self.manifest
        )


if __name__ == "__main__":
  absltest.main()
