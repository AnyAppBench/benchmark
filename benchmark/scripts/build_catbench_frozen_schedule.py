#!/usr/bin/env python3
"""Compile the frozen CATBench ACL multi-condition schedule without running it.

This tool is intentionally incapable of launching an emulator, model, or
replacement episode.  It accepts exactly two named, fully specified releases:
the 5-model/23-app/50-template primary design, or the separate five-block G6
real-task dry run.  It writes:

* ``episode_schedule.jsonl``: exactly 10,350 initial episode slots;
* ``replacement_ledger_seed.jsonl``: one outcome-blind triplet ledger record
  for each of the 3,450 paired experimental keys;
* ``replacement_ledger.schema.json``: a schema that permits at most two full
  C1/C2-G/C2-O replacement rounds after an infrastructure invalidation; and
* ``schedule_manifest.json``: counts, policies, and content hashes.

There is no status override, subset filter, resume option, or launch command.
The G6 release is a distinct discard-only release, never a filtered primary
run. Any release whose cohort or required app/verifier gate is not ready fails
closed. The checked-in primary cohort is intentionally blocked until all 230
frozen task--app adapters have independently approved G3 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
DEFAULT_COHORT = (
    BENCHMARK_ROOT / "configs" / "catbench_5cat_primary_cohort.json"
)
DEFAULT_G6_DRY_RUN_COHORT = (
    BENCHMARK_ROOT / "configs" / "catbench_5cat_g6_dryrun_cohort.json"
)

PRIMARY_CONDITIONS = ("c1", "c2_g", "c2_o")
PRIMARY_RELEASE_ID = "catbench_acl_revision_5cat_v1"
PRIMARY_SCHEDULE_SEED = 20260710
G6_DRY_RUN_RELEASE_ID = "catbench_acl_revision_5cat_g6_dryrun_v1"
G6_DRY_RUN_SCHEDULE_SEED = 20260711
PRIMARY_MODELS = (
    "UI-Venus-7B",
    "GUI-Owl-7B",
    "MAI-UI-8B",
    "UI Voyager-4B",
    "Qwen3-VL-8B",
)
# Behavior-changing CATBench environment controls are part of the release,
# not operator-tunable launch settings.  Values are strings because they are
# applied verbatim to the episode subprocess environment.  In particular,
# strict internal category isolation makes a failed sibling-app reset an
# infrastructure invalidation instead of silently changing the app roster.
FROZEN_EPISODE_RUNTIME_POLICY = {
    "schema_version": 1,
    "environment": {
        "CATBENCH_CLEAR_APP_DATA_METHOD": "pm_clear",
        "CATBENCH_EARLY_STOP_ON_SUCCESS": "1",
        "CATBENCH_PM_CLEAR_TIMEOUT_SEC": "45",
        "CATBENCH_PM_TIMEOUT_SEC": "45",
        "CATBENCH_STRICT_CATEGORY_ISOLATION": "1",
        "CATBENCH_TASK_TIMEOUT_SECONDS": "0",
        "CATBENCH_USE_ISOLATE_CATEGORY_SCRIPT": "0",
        "CATBENCH_VERIFIER_SETTLE_ATTEMPTS": "3",
        "CATBENCH_VERIFIER_SETTLE_INTERVAL_SECONDS": "0.2",
    },
}
PRIMARY_CATEGORY_ROSTERS = {
    "sms": {
        "aw_app_id": "sms_simple_sms_messenger",
        "app_ids": (
            "sms_simple_sms_messenger",
            "sms_fossify_messages",
            "sms_quik_sms",
            "sms_google_messages",
        ),
        "semantic_task_ids": (
            "SmsSend",
            "SmsReply",
            "SmsReplyMostRecent",
            "SmsResend",
            "SmsSendToContact",
            "SmsSendReceivedAddress",
            "SmsCreateDraftMessage",
            "SmsEditDraftMessage",
            "SmsDeleteConversation",
            "SmsForwardMessage",
        ),
    },
    "files": {
        "aw_app_id": "files_material_files",
        "app_ids": (
            "files_material_files",
            "files_amaze",
            "files_fossify_file_manager",
            "files_total_commander",
            "files_x_plore_file_manager",
        ),
        "semantic_task_ids": (
            "FilesCreateFolder",
            "FilesRenameFile",
            "FilesDeleteFile",
            "FilesMoveFile",
            "FilesSaveCopyOfFile",
            "FilesSearchFile",
            "FilesCompressFiles",
            "FilesExtractArchive",
            "FilesViewFileInfo",
            "FilesShareFile",
        ),
    },
    "maps": {
        "aw_app_id": "maps_osmand",
        "app_ids": (
            "maps_osmand",
            "maps_organic_maps",
            "maps_comaps",
        ),
        "semantic_task_ids": (
            "MapsSearchPlace",
            "MapsAddFavorite",
            "MapsRemoveFavorite",
            "MapsAddMarker",
            "MapsDeleteMarker",
            "MapsRecordTrack",
            "MapsGetDirections",
            "MapsSearchNearbyPlace",
            "MapsExportLocation",
            "MapsShareLocation",
        ),
    },
    "contacts": {
        "aw_app_id": "contacts_google_contacts",
        "app_ids": (
            "contacts_google_contacts",
            "contacts_fossify_contacts",
            "contacts_connect_you",
            "contacts_simple_contacts_pro_se",
            "contacts_right_contact",
        ),
        "semantic_task_ids": (
            "ContactsAddContact",
            "ContactsNewContactDraft",
            "ContactsEditContact",
            "ContactsSearchContact",
            "ContactsViewContactDetails",
            "ContactsAddFavoriteContact",
            "ContactsRemoveFavoriteContact",
            "ContactsDeleteContact",
            "ContactsCallContact",
            "ContactsMessageContact",
        ),
    },
    "clock": {
        "aw_app_id": "clock_google_clock",
        "app_ids": (
            "clock_clock",
            "clock_simple_clock",
            "clock_google_clock",
            "clock_clockyou",
            "clock_chrono",
            "clock_fossify_clock",
        ),
        "semantic_task_ids": (
            "ClockCreateAlarm",
            "ClockEditAlarm",
            "ClockEnableAlarm",
            "ClockDeleteAlarm",
            "ClockCreateTimer",
            "ClockStartTimer",
            "ClockStopwatchRunning",
            "ClockPauseStopwatch",
            "ClockStopwatchReset",
            "ClockAddWorldClock",
        ),
    },
}
PRIMARY_CATEGORY_APP_COUNTS = {
    category: len(spec["app_ids"])
    for category, spec in PRIMARY_CATEGORY_ROSTERS.items()
}
SEMANTIC_ORIGIN_AW = "androidworld_intent_adapted_by_catbench"
SEMANTIC_ORIGIN_NEW = "catbench_new_semantic_template"
SEMANTIC_ORIGINS = (SEMANTIC_ORIGIN_AW, SEMANTIC_ORIGIN_NEW)
PRIMARY_SEMANTIC_ORIGINS = {
    "sms": {
        "SmsSend": SEMANTIC_ORIGIN_AW,
        "SmsReply": SEMANTIC_ORIGIN_AW,
        "SmsReplyMostRecent": SEMANTIC_ORIGIN_AW,
        "SmsResend": SEMANTIC_ORIGIN_AW,
        "SmsSendToContact": SEMANTIC_ORIGIN_NEW,
        "SmsSendReceivedAddress": SEMANTIC_ORIGIN_AW,
        "SmsCreateDraftMessage": SEMANTIC_ORIGIN_NEW,
        "SmsEditDraftMessage": SEMANTIC_ORIGIN_NEW,
        "SmsDeleteConversation": SEMANTIC_ORIGIN_NEW,
        "SmsForwardMessage": SEMANTIC_ORIGIN_NEW,
    },
    "files": {
        "FilesCreateFolder": SEMANTIC_ORIGIN_NEW,
        "FilesRenameFile": SEMANTIC_ORIGIN_NEW,
        "FilesDeleteFile": SEMANTIC_ORIGIN_AW,
        "FilesMoveFile": SEMANTIC_ORIGIN_AW,
        "FilesSaveCopyOfFile": SEMANTIC_ORIGIN_AW,
        "FilesSearchFile": SEMANTIC_ORIGIN_NEW,
        "FilesCompressFiles": SEMANTIC_ORIGIN_NEW,
        "FilesExtractArchive": SEMANTIC_ORIGIN_NEW,
        "FilesViewFileInfo": SEMANTIC_ORIGIN_NEW,
        "FilesShareFile": SEMANTIC_ORIGIN_NEW,
    },
    "maps": {
        "MapsSearchPlace": SEMANTIC_ORIGIN_NEW,
        "MapsAddFavorite": SEMANTIC_ORIGIN_AW,
        "MapsRemoveFavorite": SEMANTIC_ORIGIN_NEW,
        "MapsAddMarker": SEMANTIC_ORIGIN_AW,
        "MapsDeleteMarker": SEMANTIC_ORIGIN_NEW,
        "MapsRecordTrack": SEMANTIC_ORIGIN_AW,
        "MapsGetDirections": SEMANTIC_ORIGIN_NEW,
        "MapsSearchNearbyPlace": SEMANTIC_ORIGIN_NEW,
        "MapsExportLocation": SEMANTIC_ORIGIN_NEW,
        "MapsShareLocation": SEMANTIC_ORIGIN_NEW,
    },
    "contacts": {
        "ContactsAddContact": SEMANTIC_ORIGIN_AW,
        "ContactsNewContactDraft": SEMANTIC_ORIGIN_AW,
        "ContactsEditContact": SEMANTIC_ORIGIN_NEW,
        "ContactsSearchContact": SEMANTIC_ORIGIN_NEW,
        "ContactsViewContactDetails": SEMANTIC_ORIGIN_NEW,
        "ContactsAddFavoriteContact": SEMANTIC_ORIGIN_NEW,
        "ContactsRemoveFavoriteContact": SEMANTIC_ORIGIN_NEW,
        "ContactsDeleteContact": SEMANTIC_ORIGIN_NEW,
        "ContactsCallContact": SEMANTIC_ORIGIN_NEW,
        "ContactsMessageContact": SEMANTIC_ORIGIN_NEW,
    },
    "clock": {
        "ClockCreateAlarm": SEMANTIC_ORIGIN_NEW,
        "ClockEditAlarm": SEMANTIC_ORIGIN_NEW,
        "ClockEnableAlarm": SEMANTIC_ORIGIN_NEW,
        "ClockDeleteAlarm": SEMANTIC_ORIGIN_NEW,
        "ClockCreateTimer": SEMANTIC_ORIGIN_AW,
        "ClockStartTimer": SEMANTIC_ORIGIN_NEW,
        "ClockStopwatchRunning": SEMANTIC_ORIGIN_AW,
        "ClockPauseStopwatch": SEMANTIC_ORIGIN_AW,
        "ClockStopwatchReset": SEMANTIC_ORIGIN_NEW,
        "ClockAddWorldClock": SEMANTIC_ORIGIN_NEW,
    },
}
PRIMARY_COUNTS = {
    "category_count": 5,
    "app_count": 23,
    "semantic_template_count": 50,
    "task_app_count": 230,
    "instances_per_model_condition": 690,
    "episodes_per_condition": 3450,
    "episodes_all_conditions": 10350,
    "androidworld_intent_adapted_semantic_template_count": 16,
    "catbench_new_semantic_template_count": 34,
}
G6_DRY_RUN_CATEGORY_ROSTERS = {
    "sms": {
        "aw_app_id": "sms_simple_sms_messenger",
        "app_ids": ("sms_simple_sms_messenger",),
        "semantic_task_ids": ("SmsSend",),
    },
    "files": {
        "aw_app_id": "files_material_files",
        "app_ids": ("files_material_files",),
        "semantic_task_ids": ("FilesCreateFolder",),
    },
    "maps": {
        "aw_app_id": "maps_osmand",
        "app_ids": ("maps_osmand",),
        "semantic_task_ids": ("MapsSearchPlace",),
    },
    # The older pinned Google Contacts build is absent locally.  Fossify is
    # the first hash-matched app in the frozen roster and is a registered real
    # task target; aw_app_id records the canonical reference, not the target.
    "contacts": {
        "aw_app_id": "contacts_google_contacts",
        "app_ids": ("contacts_fossify_contacts",),
        "semantic_task_ids": ("ContactsAddContact",),
    },
    # Google Clock is likewise absent and Clock You is not live-qualified.
    # Clock is the first remaining hash-matched real app in roster order.
    "clock": {
        "aw_app_id": "clock_google_clock",
        "app_ids": ("clock_clock",),
        "semantic_task_ids": ("ClockCreateAlarm",),
    },
}
G6_DRY_RUN_PAIRED_BLOCKS = (
    ("UI-Venus-7B", "sms", "sms_simple_sms_messenger", "SmsSend", 0),
    (
        "GUI-Owl-7B",
        "files",
        "files_material_files",
        "FilesCreateFolder",
        1,
    ),
    ("MAI-UI-8B", "maps", "maps_osmand", "MapsSearchPlace", 2),
    (
        "UI Voyager-4B",
        "contacts",
        "contacts_fossify_contacts",
        "ContactsAddContact",
        0,
    ),
    ("Qwen3-VL-8B", "clock", "clock_clock", "ClockCreateAlarm", 1),
)
G6_DRY_RUN_SELECTION_POLICY = {
    "category_order": ["sms", "files", "maps", "contacts", "clock"],
    "model_assignment": "primary_model_roster_order_one_model_per_category",
    "app_choice": (
        "first_primary_roster_app_with_a_locally_hash_matched_pin_excluding_"
        "missing_google_builds_and_unqualified_clock_you"
    ),
    "semantic_task_choice": "first_frozen_semantic_task_in_each_category",
    "instance_choice": "zero_based_category_index_modulo_three",
    "outcome_information_used": False,
}
G6_DRY_RUN_COUNTS = {
    "category_count": 5,
    "app_count": 5,
    "semantic_template_count": 5,
    "task_app_count": 5,
    "scheduled_instance_count": 5,
    "episodes_per_condition": 5,
    "episodes_all_conditions": 15,
}
RELEASE_PURPOSE = {
    PRIMARY_RELEASE_ID: {
        "release_purpose": "primary_five_category_analysis",
        "analysis_eligible": True,
        "artifact_role": "primary_analysis_candidate",
    },
    G6_DRY_RUN_RELEASE_ID: {
        "release_purpose": "g6_discard_only_end_to_end_validation",
        "analysis_eligible": False,
        "artifact_role": "discard_only_never_primary_analysis",
    },
}
CLOCK_YOU_GATE = "clock_clockyou"
VERIFIER_CONFORMANCE_GATE = "verifier_conformance"
VERIFIER_QUALIFICATION_POLICY = (
    "all_frozen_task_app_adapters_with_primitive_action_positive_reset_replay_"
    "and_six_negative_controls_v1"
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_REPLACEMENT_ROUNDS = 2

SCHEDULE_FILE = "episode_schedule.jsonl"
LEDGER_FILE = "replacement_ledger_seed.jsonl"
LEDGER_SCHEMA_FILE = "replacement_ledger.schema.json"
MANIFEST_FILE = "schedule_manifest.json"


class ScheduleBuildError(ValueError):
  """Raised when a frozen schedule cannot be compiled safely."""


def _canonical_json(value: Any) -> str:
  return json.dumps(
      value,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
  )


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def episode_runtime_policy_sha256(
    policy: Mapping[str, Any] = FROZEN_EPISODE_RUNTIME_POLICY,
) -> str:
  """Return the canonical hash of the frozen episode behavior controls."""
  return _sha256_bytes(_canonical_json(policy).encode("utf-8"))


def _validate_episode_runtime_policy(
    cohort: Mapping[str, Any], errors: list[str]
) -> None:
  observed = cohort.get("episode_runtime_policy")
  if observed != FROZEN_EPISODE_RUNTIME_POLICY:
    errors.append(
        "episode_runtime_policy must equal the exact frozen verifier/reset "
        f"policy; got {observed!r}"
    )


def _require_unique(
    values: Sequence[str], label: str, errors: list[str]
) -> None:
  if len(values) != len(set(values)):
    errors.append(f"{label} contains duplicate identifiers")


def validate_primary_cohort(cohort: Mapping[str, Any]) -> None:
  """Fail closed unless ``cohort`` is the exact ready primary design."""
  errors: list[str] = []
  required = {
      "release_id",
      "status",
      "schedule_seed",
      "suite_family",
      "task_random_seed",
      "n_task_combinations",
      "conditions",
      "episode_runtime_policy",
      "eligibility_gates",
      "models",
      "categories",
      "expected",
  }
  missing = sorted(required - set(cohort))
  if missing:
    errors.append(f"missing required fields: {missing}")

  release_id = str(cohort.get("release_id") or "").strip()
  if release_id != PRIMARY_RELEASE_ID:
    errors.append(
        f"release_id must equal {PRIMARY_RELEASE_ID!r}; got {release_id!r}"
    )
  if cohort.get("status") != "ready":
    errors.append(
        f"cohort status is {cohort.get('status')!r}; expected 'ready'"
    )
  if cohort.get("schedule_seed") != PRIMARY_SCHEDULE_SEED:
    errors.append(
        f"schedule_seed must equal {PRIMARY_SCHEDULE_SEED}; got "
        f"{cohort.get('schedule_seed')!r}"
    )
  if cohort.get("suite_family") != "android_world":
    errors.append("suite_family must be 'android_world'")
  if cohort.get("task_random_seed") != 30:
    errors.append("task_random_seed must equal the frozen value 30")
  if cohort.get("n_task_combinations") != 3:
    errors.append("n_task_combinations must equal 3")
  _validate_episode_runtime_policy(cohort, errors)

  conditions = list(cohort.get("conditions") or [])
  if conditions != list(PRIMARY_CONDITIONS):
    errors.append(
        f"conditions must be exactly {list(PRIMARY_CONDITIONS)!r}; "
        f"got {conditions!r}"
    )

  gates = cohort.get("eligibility_gates")
  gates = gates if isinstance(gates, Mapping) else {}
  required_gate_names = {CLOCK_YOU_GATE, VERIFIER_CONFORMANCE_GATE}
  if set(gates) != required_gate_names:
    errors.append(
        "eligibility_gates must contain exactly the Clock You and full G3 "
        "verifier-conformance gates; "
        f"got {sorted(gates)!r}"
    )
  for gate_name, gate in gates.items():
    if (
        isinstance(gate, Mapping)
        and gate.get("required_for_primary") is True
        and gate.get("status") != "ready"
    ):
      errors.append(
          f"required eligibility gate {gate_name!r} is "
          f"{gate.get('status')!r}; expected 'ready'"
      )
  clock_gate = gates.get(CLOCK_YOU_GATE)
  if not isinstance(clock_gate, Mapping):
    errors.append(f"missing required {CLOCK_YOU_GATE!r} eligibility gate")
  else:
    if clock_gate.get("app_id") != CLOCK_YOU_GATE:
      errors.append("Clock You gate app_id must be 'clock_clockyou'")
    if clock_gate.get("category") != "clock":
      errors.append("Clock You gate category must be 'clock'")
    if clock_gate.get("required_for_primary") is not True:
      errors.append("Clock You gate must be required for the primary cohort")
    if clock_gate.get("status") != "ready":
      errors.append(
          "Clock You eligibility gate is "
          f"{clock_gate.get('status')!r}; expected 'ready'"
      )

  verifier_gate = gates.get(VERIFIER_CONFORMANCE_GATE)
  if not isinstance(verifier_gate, Mapping):
    errors.append(
        f"missing required {VERIFIER_CONFORMANCE_GATE!r} eligibility gate"
    )
  else:
    if verifier_gate.get("required_for_primary") is not True:
      errors.append(
          "Verifier-conformance gate must be required for the primary cohort"
      )
    if verifier_gate.get("status") != "ready":
      errors.append(
          "Verifier-conformance eligibility gate is "
          f"{verifier_gate.get('status')!r}; expected 'ready'"
      )
    if verifier_gate.get("qualification_policy") != (
        VERIFIER_QUALIFICATION_POLICY
    ):
      errors.append("Verifier-conformance qualification_policy is not frozen")
    if verifier_gate.get("adapter_count") != PRIMARY_COUNTS["task_app_count"]:
      errors.append("Verifier-conformance adapter_count must equal 230")
    if verifier_gate.get("qualified_adapter_count") != PRIMARY_COUNTS[
        "task_app_count"
    ]:
      errors.append("Verifier-conformance qualified_adapter_count must equal 230")
    if verifier_gate.get("unqualified_adapter_count") != 0:
      errors.append("Verifier-conformance unqualified_adapter_count must equal 0")
    if verifier_gate.get("approval_status") != "approved":
      errors.append("Verifier-conformance approval_status must be 'approved'")
    for hash_field in (
        "evidence_manifest_sha256",
        "approval_record_sha256",
    ):
      if not HEX_SHA256.fullmatch(str(verifier_gate.get(hash_field) or "")):
        errors.append(
            f"Verifier-conformance {hash_field} must be a 64-hex digest"
        )
    known_unqualified = verifier_gate.get("known_unqualified_scored_adapters")
    if known_unqualified != []:
      errors.append(
          "Verifier-conformance known_unqualified_scored_adapters must be "
          "empty before a primary release"
      )

  models = list(cohort.get("models") or [])
  if models != list(PRIMARY_MODELS):
    errors.append(
        f"models must be the exact frozen roster {list(PRIMARY_MODELS)!r}; "
        f"got {models!r}"
    )
  if not all(isinstance(model, str) and model.strip() for model in models):
    errors.append("every model identifier must be a non-empty string")
  _require_unique(models, "models", errors)

  categories = cohort.get("categories")
  categories = categories if isinstance(categories, Mapping) else {}
  if list(categories) != list(PRIMARY_CATEGORY_APP_COUNTS):
    errors.append(
        "categories must be exactly and in order "
        f"{list(PRIMARY_CATEGORY_APP_COUNTS)!r}; got {list(categories)!r}"
    )

  all_app_ids: list[str] = []
  all_task_ids: list[str] = []
  task_app_count = 0
  for category, expected_app_count in PRIMARY_CATEGORY_APP_COUNTS.items():
    spec = categories.get(category)
    if not isinstance(spec, Mapping):
      errors.append(f"missing category specification: {category}")
      continue
    app_ids = list(spec.get("app_ids") or [])
    task_ids = list(spec.get("semantic_task_ids") or [])
    semantic_origins = spec.get("semantic_origins")
    semantic_origins = (
        semantic_origins if isinstance(semantic_origins, Mapping) else {}
    )
    frozen_roster = PRIMARY_CATEGORY_ROSTERS[category]
    if app_ids != list(frozen_roster["app_ids"]):
      errors.append(
          f"{category}.app_ids must be the exact frozen real-app roster; "
          f"got {app_ids!r}"
      )
    if task_ids != list(frozen_roster["semantic_task_ids"]):
      errors.append(
          f"{category}.semantic_task_ids must be the exact frozen semantic "
          f"roster; got {task_ids!r}"
      )
    expected_origins = PRIMARY_SEMANTIC_ORIGINS[category]
    if dict(semantic_origins) != expected_origins:
      errors.append(
          f"{category}.semantic_origins must equal the exact immutable "
          f"semantic-lineage map; got {dict(semantic_origins)!r}"
      )
    if list(semantic_origins) != task_ids:
      errors.append(
          f"{category}.semantic_origins must follow semantic_task_ids order"
      )
    if len(app_ids) != expected_app_count:
      errors.append(
          f"{category} must contain {expected_app_count} apps; got "
          f"{len(app_ids)}"
      )
    if len(task_ids) != 10:
      errors.append(
          f"{category} must contain 10 semantic tasks; got {len(task_ids)}"
      )
    _require_unique(app_ids, f"{category}.app_ids", errors)
    _require_unique(task_ids, f"{category}.semantic_task_ids", errors)
    aw_app_id = spec.get("aw_app_id")
    if aw_app_id != frozen_roster["aw_app_id"]:
      errors.append(
          f"{category}.aw_app_id must equal "
          f"{frozen_roster['aw_app_id']!r}; got {aw_app_id!r}"
      )
    if aw_app_id not in app_ids:
      errors.append(f"{category}.aw_app_id is absent from its app roster")
    if not all(isinstance(value, str) and value.strip() for value in app_ids):
      errors.append(f"{category}.app_ids contains an invalid identifier")
    if not all(isinstance(value, str) and value.strip() for value in task_ids):
      errors.append(
          f"{category}.semantic_task_ids contains an invalid identifier"
      )
    all_app_ids.extend(app_ids)
    all_task_ids.extend(task_ids)
    task_app_count += len(app_ids) * len(task_ids)

  _require_unique(all_app_ids, "cross-category app roster", errors)
  _require_unique(all_task_ids, "cross-category semantic task roster", errors)
  clock_spec = categories.get("clock")
  clock_apps = (
      list(clock_spec.get("app_ids") or [])
      if isinstance(clock_spec, Mapping)
      else []
  )
  if CLOCK_YOU_GATE not in clock_apps:
    errors.append("clock_clockyou is absent from the frozen clock roster")

  computed = {
      "category_count": len(categories),
      "app_count": len(all_app_ids),
      "semantic_template_count": len(all_task_ids),
      "androidworld_intent_adapted_semantic_template_count": sum(
          origin == SEMANTIC_ORIGIN_AW
          for origins in PRIMARY_SEMANTIC_ORIGINS.values()
          for origin in origins.values()
      ),
      "catbench_new_semantic_template_count": sum(
          origin == SEMANTIC_ORIGIN_NEW
          for origins in PRIMARY_SEMANTIC_ORIGINS.values()
          for origin in origins.values()
      ),
      "task_app_count": task_app_count,
      "instances_per_model_condition": (
          task_app_count * int(cohort.get("n_task_combinations") or 0)
      ),
  }
  computed["episodes_per_condition"] = (
      computed["instances_per_model_condition"] * len(models)
  )
  computed["episodes_all_conditions"] = (
      computed["episodes_per_condition"] * len(conditions)
  )
  expected = cohort.get("expected")
  expected = expected if isinstance(expected, Mapping) else {}
  for field, frozen_value in PRIMARY_COUNTS.items():
    if expected.get(field) != frozen_value:
      errors.append(
          f"expected.{field} must equal {frozen_value}; got "
          f"{expected.get(field)!r}"
      )
    if computed.get(field) != frozen_value:
      errors.append(
          f"computed {field} must equal {frozen_value}; got "
          f"{computed.get(field)!r}"
      )

  if errors:
    raise ScheduleBuildError(
        "Frozen cohort validation failed:\n- " + "\n- ".join(errors)
    )


def _g6_expected_categories() -> dict[str, dict[str, Any]]:
  return {
      category: {
          "aw_app_id": spec["aw_app_id"],
          "app_ids": list(spec["app_ids"]),
          "semantic_task_ids": list(spec["semantic_task_ids"]),
      }
      for category, spec in G6_DRY_RUN_CATEGORY_ROSTERS.items()
  }


def _g6_expected_paired_blocks() -> list[dict[str, Any]]:
  return [
      _paired_key_dict(model, category, app_id, semantic_task_id, instance_id)
      for model, category, app_id, semantic_task_id, instance_id
      in G6_DRY_RUN_PAIRED_BLOCKS
  ]


def validate_g6_dry_run_cohort(cohort: Mapping[str, Any]) -> None:
  """Fail closed unless ``cohort`` is the one preregistered G6 release.

  This is deliberately not a generic small-cohort schema.  Count-preserving
  substitutions, added filters, and another release identifier are rejected.
  Every selected app and task is part of the primary real registry roster.
  """
  errors: list[str] = []
  required = {
      "release_id",
      "status",
      "release_purpose",
      "analysis_eligible",
      "artifact_role",
      "schedule_seed",
      "suite_family",
      "task_random_seed",
      "n_task_combinations",
      "conditions",
      "episode_runtime_policy",
      "eligibility_gates",
      "models",
      "categories",
      "paired_blocks",
      "selection_policy",
      "expected",
  }
  missing = sorted(required - set(cohort))
  extra = sorted(set(cohort) - required)
  if missing:
    errors.append(f"missing required fields: {missing}")
  if extra:
    errors.append(f"unexpected fields are forbidden: {extra}")

  exact_scalars = {
      "release_id": G6_DRY_RUN_RELEASE_ID,
      "status": "ready",
      "release_purpose": RELEASE_PURPOSE[G6_DRY_RUN_RELEASE_ID][
          "release_purpose"
      ],
      "analysis_eligible": False,
      "artifact_role": RELEASE_PURPOSE[G6_DRY_RUN_RELEASE_ID]["artifact_role"],
      "schedule_seed": G6_DRY_RUN_SCHEDULE_SEED,
      "suite_family": "android_world",
      "task_random_seed": 30,
      "n_task_combinations": 3,
  }
  for field, expected in exact_scalars.items():
    if type(cohort.get(field)) is not type(expected) or cohort.get(field) != expected:
      errors.append(
          f"{field} must equal {expected!r}; got {cohort.get(field)!r}"
      )
  if cohort.get("conditions") != list(PRIMARY_CONDITIONS):
    errors.append(
        f"conditions must be exactly {list(PRIMARY_CONDITIONS)!r}; got "
        f"{cohort.get('conditions')!r}"
    )
  _validate_episode_runtime_policy(cohort, errors)
  if cohort.get("eligibility_gates") != {}:
    errors.append("G6 dry-run eligibility_gates must be exactly empty")
  if cohort.get("models") != list(PRIMARY_MODELS):
    errors.append(
        "G6 dry-run models must be the exact five frozen primary models in "
        f"order; got {cohort.get('models')!r}"
    )

  expected_categories = _g6_expected_categories()
  if cohort.get("categories") != expected_categories:
    errors.append(
        "G6 dry-run categories must equal the exact preregistered real-app/"
        "semantic-task roster"
    )
  expected_blocks = _g6_expected_paired_blocks()
  if cohort.get("paired_blocks") != expected_blocks:
    errors.append(
        "paired_blocks must equal the exact five preregistered real blocks"
    )
  if cohort.get("selection_policy") != G6_DRY_RUN_SELECTION_POLICY:
    errors.append("selection_policy differs from the preregistered policy")
  if cohort.get("expected") != G6_DRY_RUN_COUNTS:
    errors.append(
        f"expected must equal {G6_DRY_RUN_COUNTS!r}; got "
        f"{cohort.get('expected')!r}"
    )

  # Redundant structural assertions make accidental edits easier to diagnose
  # even though exact equality above already fails closed.
  if len(expected_blocks) != 5:
    errors.append("internal G6 block roster must contain exactly five blocks")
  if {block["model"] for block in expected_blocks} != set(PRIMARY_MODELS):
    errors.append("G6 blocks must represent every primary model exactly once")
  if [block["category"] for block in expected_blocks] != list(
      G6_DRY_RUN_CATEGORY_ROSTERS
  ):
    errors.append("G6 blocks must contain one block per category in frozen order")
  for block in expected_blocks:
    category = block["category"]
    if block["app_id"] not in PRIMARY_CATEGORY_ROSTERS[category]["app_ids"]:
      errors.append(f"G6 block uses a non-primary real app: {block!r}")
    if block["semantic_task_id"] not in PRIMARY_CATEGORY_ROSTERS[category][
        "semantic_task_ids"
    ]:
      errors.append(f"G6 block uses a non-primary semantic task: {block!r}")
    if block["instance_id"] not in range(3):
      errors.append(f"G6 block has out-of-range K=3 instance: {block!r}")

  if errors:
    raise ScheduleBuildError(
        "Frozen G6 dry-run cohort validation failed:\n- "
        + "\n- ".join(errors)
    )


def validate_frozen_cohort(cohort: Mapping[str, Any]) -> None:
  """Accept exactly the primary release or the exact discard-only G6 release."""
  release_id = str(cohort.get("release_id") or "")
  if release_id == PRIMARY_RELEASE_ID:
    validate_primary_cohort(cohort)
  elif release_id == G6_DRY_RUN_RELEASE_ID:
    validate_g6_dry_run_cohort(cohort)
  else:
    raise ScheduleBuildError(
        "Unknown frozen release_id; only the exact primary and G6 dry-run "
        f"releases are accepted, got {release_id!r}"
    )


def release_policy(release_id: str) -> dict[str, Any]:
  try:
    return dict(RELEASE_PURPOSE[release_id])
  except KeyError as exc:
    raise ScheduleBuildError(f"Unknown frozen release_id: {release_id!r}") from exc


def semantic_origin(category: str, semantic_task_id: str) -> str:
  """Return the immutable semantic-lineage stratum for a frozen template."""
  try:
    return PRIMARY_SEMANTIC_ORIGINS[category][semantic_task_id]
  except KeyError as exc:
    raise ScheduleBuildError(
        "Unknown frozen semantic-origin identity: "
        f"{category}/{semantic_task_id}"
    ) from exc


def load_primary_cohort(path: Path) -> tuple[dict[str, Any], str]:
  try:
    raw = path.read_bytes()
  except OSError as exc:
    raise ScheduleBuildError(f"Unable to read cohort {path}: {exc}") from exc
  try:
    payload = json.loads(raw)
  except json.JSONDecodeError as exc:
    raise ScheduleBuildError(f"Invalid cohort JSON {path}: {exc}") from exc
  if not isinstance(payload, dict):
    raise ScheduleBuildError("Primary cohort must contain a JSON object")
  validate_primary_cohort(payload)
  return payload, _sha256_bytes(raw)


def load_frozen_cohort(path: Path) -> tuple[dict[str, Any], str]:
  """Load one of the two exact frozen releases and return its byte hash."""
  try:
    raw = path.read_bytes()
  except OSError as exc:
    raise ScheduleBuildError(f"Unable to read cohort {path}: {exc}") from exc
  try:
    payload = json.loads(raw)
  except json.JSONDecodeError as exc:
    raise ScheduleBuildError(f"Invalid cohort JSON {path}: {exc}") from exc
  if not isinstance(payload, dict):
    raise ScheduleBuildError("Frozen cohort must contain a JSON object")
  validate_frozen_cohort(payload)
  return payload, _sha256_bytes(raw)


def _paired_key_dict(
    model: str,
    category: str,
    app_id: str,
    semantic_task_id: str,
    instance_id: int,
) -> dict[str, Any]:
  return {
      "model": model,
      "category": category,
      "app_id": app_id,
      "semantic_task_id": semantic_task_id,
      "instance_id": instance_id,
  }


def _pair_id(release_id: str, key: Mapping[str, Any]) -> str:
  digest = _sha256_bytes(
      _canonical_json({"release_id": release_id, "paired_key": key}).encode(
          "utf-8"
      )
  )
  return f"pair_{digest[:24]}"


def _attempt_identity(
    release_id: str,
    pair_id: str,
    condition: str,
    replacement_round: int,
) -> dict[str, str]:
  round_label = f"r{replacement_round}"
  slot_id = f"{pair_id}:{condition}"
  if replacement_round:
    slot_id = f"{slot_id}:replacement:{round_label}"
  return {
      "slot_id": slot_id,
      "attempt_id": f"{pair_id}:{condition}:attempt:{round_label}",
      "snapshot_clone_id": (
          f"{release_id}:snapshot:{pair_id}:{condition}:{round_label}"
      ),
  }


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
  return (
      "".join(_canonical_json(dict(row)) + "\n" for row in rows)
  ).encode("utf-8")


def replacement_ledger_schema() -> dict[str, Any]:
  """JSON Schema for outcome-blind, full-triplet replacement accounting."""
  condition_properties = {
      condition: {"$ref": "#/$defs/attemptIdentity"}
      for condition in PRIMARY_CONDITIONS
  }
  replacement_condition_properties = {
      condition: {"$ref": "#/$defs/replacementAttempt"}
      for condition in PRIMARY_CONDITIONS
  }
  return {
      "$schema": "https://json-schema.org/draft/2020-12/schema",
      "$id": "catbench://schemas/frozen-replacement-ledger-v1",
      "title": "CATBench full-condition-triplet replacement ledger",
      "type": "object",
      "additionalProperties": False,
      "required": [
          "release_id",
          "pair_id",
          "paired_key",
          "max_replacement_rounds",
          "selection_unit",
          "outcome_selected_replacement_permitted",
          "initial_condition_slots",
          "authorized_replacement_rounds",
          "replacement_rounds",
      ],
      "properties": {
          "release_id": {"type": "string", "minLength": 1},
          "pair_id": {"type": "string", "pattern": "^pair_[0-9a-f]{24}$"},
          "paired_key": {"$ref": "#/$defs/pairedKey"},
          "max_replacement_rounds": {"const": MAX_REPLACEMENT_ROUNDS},
          "selection_unit": {"const": "full_condition_triplet"},
          "outcome_selected_replacement_permitted": {"const": False},
          "initial_condition_slots": {
              "type": "object",
              "additionalProperties": False,
              "required": list(PRIMARY_CONDITIONS),
              "properties": condition_properties,
          },
          "authorized_replacement_rounds": {
              "type": "array",
              "minItems": MAX_REPLACEMENT_ROUNDS,
              "maxItems": MAX_REPLACEMENT_ROUNDS,
              "prefixItems": [
                  {
                      "allOf": [
                          {"$ref": "#/$defs/authorizedRound"},
                          {"properties": {"round_index": {"const": 1}}},
                      ]
                  },
                  {
                      "allOf": [
                          {"$ref": "#/$defs/authorizedRound"},
                          {"properties": {"round_index": {"const": 2}}},
                      ]
                  },
              ],
              "items": False,
          },
          "replacement_rounds": {
              "type": "array",
              "maxItems": MAX_REPLACEMENT_ROUNDS,
              "uniqueItems": True,
              "prefixItems": [
                  {
                      "allOf": [
                          {"$ref": "#/$defs/replacementRound"},
                          {"properties": {"round_index": {"const": 1}}},
                      ]
                  },
                  {
                      "allOf": [
                          {"$ref": "#/$defs/replacementRound"},
                          {"properties": {"round_index": {"const": 2}}},
                      ]
                  },
              ],
              "items": False,
          },
      },
      "$defs": {
          "pairedKey": {
              "type": "object",
              "additionalProperties": False,
              "required": [
                  "model",
                  "category",
                  "app_id",
                  "semantic_task_id",
                  "instance_id",
              ],
              "properties": {
                  "model": {"type": "string", "minLength": 1},
                  "category": {"type": "string", "minLength": 1},
                  "app_id": {"type": "string", "minLength": 1},
                  "semantic_task_id": {"type": "string", "minLength": 1},
                  "instance_id": {"type": "integer", "minimum": 0},
              },
          },
          "attemptIdentity": {
              "type": "object",
              "additionalProperties": False,
              "required": ["slot_id", "attempt_id", "snapshot_clone_id"],
              "properties": {
                  "slot_id": {"type": "string", "minLength": 1},
                  "attempt_id": {"type": "string", "minLength": 1},
                  "snapshot_clone_id": {"type": "string", "minLength": 1},
              },
          },
          "authorizedRound": {
              "type": "object",
              "additionalProperties": False,
              "required": [
                  "round_index",
                  "scheduled",
                  "condition_attempts",
              ],
              "properties": {
                  "round_index": {
                      "type": "integer",
                      "minimum": 1,
                      "maximum": MAX_REPLACEMENT_ROUNDS,
                  },
                  "scheduled": {"const": False},
                  "condition_attempts": {
                      "type": "object",
                      "additionalProperties": False,
                      "required": list(PRIMARY_CONDITIONS),
                      "properties": condition_properties,
                  },
              },
          },
          "replacementRound": {
              "type": "object",
              "additionalProperties": False,
              "required": [
                  "round_index",
                  "trigger",
                  "trigger_attempt_ids",
                  "decision_basis",
                  "decided_at",
                  "condition_attempts",
              ],
              "properties": {
                  "round_index": {
                      "type": "integer",
                      "minimum": 1,
                      "maximum": MAX_REPLACEMENT_ROUNDS,
                  },
                  "trigger": {"const": "invalid_infrastructure"},
                  "trigger_attempt_ids": {
                      "type": "array",
                      "minItems": 1,
                      "uniqueItems": True,
                      "items": {"type": "string", "minLength": 1},
                  },
                  "decision_basis": {"type": "string", "minLength": 1},
                  "decided_at": {"type": "string", "format": "date-time"},
                  "condition_attempts": {
                      "type": "object",
                      "additionalProperties": False,
                      "required": list(PRIMARY_CONDITIONS),
                      "properties": replacement_condition_properties,
                  },
              },
          },
          "replacementAttempt": {
              "type": "object",
              "additionalProperties": False,
              "required": [
                  "slot_id",
                  "attempt_id",
                  "snapshot_clone_id",
                  "status",
                  "artifact_path",
              ],
              "properties": {
                  "slot_id": {"type": "string", "minLength": 1},
                  "attempt_id": {"type": "string", "minLength": 1},
                  "snapshot_clone_id": {"type": "string", "minLength": 1},
                  "status": {
                      "enum": [
                          "valid_success",
                          "valid_failure",
                          "invalid_infrastructure",
                          "missing",
                      ]
                  },
                  "artifact_path": {"type": "string"},
              },
          },
      },
  }


def compile_frozen_schedule(
    cohort: Mapping[str, Any], cohort_sha256: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
  """Return initial slots, empty replacement ledgers, and a manifest."""
  validate_frozen_cohort(cohort)
  release_id = str(cohort["release_id"])
  policy = release_policy(release_id)
  conditions = tuple(str(value) for value in cohort["conditions"])
  instances = int(cohort["n_task_combinations"])
  episode_runtime_policy = dict(cohort["episode_runtime_policy"])
  episode_runtime_policy_sha = episode_runtime_policy_sha256(
      episode_runtime_policy
  )

  paired_keys: list[dict[str, Any]] = []
  if release_id == PRIMARY_RELEASE_ID:
    for model in cohort["models"]:
      for category, spec in cohort["categories"].items():
        for app_id in spec["app_ids"]:
          for semantic_task_id in spec["semantic_task_ids"]:
            for instance_id in range(instances):
              paired_keys.append(_paired_key_dict(
                  str(model),
                  str(category),
                  str(app_id),
                  str(semantic_task_id),
                  instance_id,
              ))
    release_counts = PRIMARY_COUNTS
  else:
    # validate_g6_dry_run_cohort has already compared every field and every
    # identity against the one allowed five-block preregistration.
    paired_keys = [dict(block) for block in cohort["paired_blocks"]]
    release_counts = G6_DRY_RUN_COUNTS

  rng = random.Random(int(cohort["schedule_seed"]))
  rng.shuffle(paired_keys)
  condition_permutations = list(itertools.permutations(conditions))
  rng.shuffle(condition_permutations)

  schedule: list[dict[str, Any]] = []
  ledger: list[dict[str, Any]] = []
  pair_ids: set[str] = set()
  slot_ids: set[str] = set()
  for block_order, paired_key in enumerate(paired_keys):
    pair_id = _pair_id(release_id, paired_key)
    if pair_id in pair_ids:
      raise ScheduleBuildError(f"Pair identifier collision: {pair_id}")
    pair_ids.add(pair_id)
    condition_order = condition_permutations[
        block_order % len(condition_permutations)
    ]
    initial_condition_slots: dict[str, dict[str, str]] = {}
    for within_block_order, condition in enumerate(condition_order):
      identity = _attempt_identity(release_id, pair_id, condition, 0)
      if identity["slot_id"] in slot_ids:
        raise ScheduleBuildError(
            f"Episode slot identifier collision: {identity['slot_id']}"
        )
      slot_ids.add(identity["slot_id"])
      initial_condition_slots[condition] = identity
      schedule.append({
          "release_id": release_id,
          "release_purpose": policy["release_purpose"],
          "analysis_eligible": policy["analysis_eligible"],
          "artifact_role": policy["artifact_role"],
          "cohort_sha256": cohort_sha256,
          "schedule_seed": cohort["schedule_seed"],
          "suite_family": cohort["suite_family"],
          "task_random_seed": cohort["task_random_seed"],
          "n_task_combinations": instances,
          "episode_runtime_policy_sha256": episode_runtime_policy_sha,
          "global_order": len(schedule),
          "block_order": block_order,
          "within_block_order": within_block_order,
          "pair_id": pair_id,
          **paired_key,
          "semantic_origin": semantic_origin(
              str(paired_key["category"]),
              str(paired_key["semantic_task_id"]),
          ),
          "condition": condition,
          "slot_id": identity["slot_id"],
          "attempt_id": identity["attempt_id"],
          "attempt_index": 0,
          "snapshot_family_id": f"{release_id}:snapshot:{pair_id}",
          "snapshot_clone_id": identity["snapshot_clone_id"],
          "is_replacement": False,
      })

    authorized_rounds = []
    for replacement_round in range(1, MAX_REPLACEMENT_ROUNDS + 1):
      authorized_rounds.append({
          "round_index": replacement_round,
          "scheduled": False,
          "condition_attempts": {
              condition: _attempt_identity(
                  release_id, pair_id, condition, replacement_round
              )
              for condition in conditions
          },
      })
    ledger.append({
        "release_id": release_id,
        "pair_id": pair_id,
        "paired_key": paired_key,
        "max_replacement_rounds": MAX_REPLACEMENT_ROUNDS,
        "selection_unit": "full_condition_triplet",
        "outcome_selected_replacement_permitted": False,
        "initial_condition_slots": {
            condition: initial_condition_slots[condition]
            for condition in conditions
        },
        "authorized_replacement_rounds": authorized_rounds,
        "replacement_rounds": [],
    })

  condition_counts = {
      condition: sum(row["condition"] == condition for row in schedule)
      for condition in conditions
  }
  if len(schedule) != release_counts["episodes_all_conditions"]:
    raise ScheduleBuildError(
        f"Compiled {len(schedule)} slots; expected "
        f"{release_counts['episodes_all_conditions']}"
    )
  if len(ledger) != release_counts["episodes_per_condition"]:
    raise ScheduleBuildError(
        f"Compiled {len(ledger)} paired ledgers; expected "
        f"{release_counts['episodes_per_condition']}"
    )
  if set(condition_counts.values()) != {
      release_counts["episodes_per_condition"]
  }:
    raise ScheduleBuildError(f"Condition counts are imbalanced: {condition_counts}")

  schedule_bytes = _jsonl_bytes(schedule)
  ledger_bytes = _jsonl_bytes(ledger)
  schema = replacement_ledger_schema()
  schema_bytes = (
      json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
  ).encode("utf-8")
  origin_template_counts = {origin: 0 for origin in SEMANTIC_ORIGINS}
  for category, category_spec in cohort["categories"].items():
    for task_id in category_spec["semantic_task_ids"]:
      origin = semantic_origin(str(category), str(task_id))
      origin_template_counts[origin] += 1
  manifest = {
      "schema_version": 2,
      "builder": "build_catbench_frozen_schedule.py",
      "launch_capability": False,
      "selective_rerun_permitted": False,
      "release_id": release_id,
      "release_purpose": policy["release_purpose"],
      "analysis_eligible": policy["analysis_eligible"],
      "artifact_role": policy["artifact_role"],
      "primary_reporter_acceptance_permitted": policy["analysis_eligible"],
      "cohort_sha256": cohort_sha256,
      "schedule_seed": cohort["schedule_seed"],
      "suite_family": cohort["suite_family"],
      "task_random_seed": cohort["task_random_seed"],
      "n_task_combinations": instances,
      "episode_runtime_policy": episode_runtime_policy,
      "episode_runtime_policy_sha256": episode_runtime_policy_sha,
      "conditions": list(conditions),
      "model_count": len(cohort["models"]),
      "app_count": release_counts["app_count"],
      "semantic_template_count": release_counts["semantic_template_count"],
      "semantic_origin_policy": (
          "immutable_per_semantic_task_lineage_from_frozen_cohort_v1"
      ),
      "semantic_origin_template_counts": origin_template_counts,
      "block_randomization": (
          "seeded_shuffle_of_complete_paired_keys_then_seeded_balanced_cycle_"
          "of_condition_permutations"
      ),
      "paired_block_count": len(ledger),
      "episode_slot_count": len(schedule),
      "condition_counts": condition_counts,
      "snapshot_policy": (
          "unique_condition_specific_clone_identifier_per_pair_and_attempt_round"
      ),
      "replacement_policy": {
          "max_replacement_rounds": MAX_REPLACEMENT_ROUNDS,
          "selection_unit": "full_condition_triplet",
          "trigger": "invalid_infrastructure_only",
          "replacement_rounds_initially_scheduled": 0,
      },
      "outputs": {
          SCHEDULE_FILE: {
              "records": len(schedule),
              "sha256": _sha256_bytes(schedule_bytes),
          },
          LEDGER_FILE: {
              "records": len(ledger),
              "sha256": _sha256_bytes(ledger_bytes),
          },
          LEDGER_SCHEMA_FILE: {
              "sha256": _sha256_bytes(schema_bytes),
          },
      },
  }
  return schedule, ledger, manifest


def _atomic_write(path: Path, data: bytes) -> None:
  tmp_path = path.with_name(path.name + ".tmp")
  try:
    with tmp_path.open("xb") as handle:
      handle.write(data)
      handle.flush()
      os.fsync(handle.fileno())
    tmp_path.replace(path)
  except Exception:
    tmp_path.unlink(missing_ok=True)
    raise


def write_frozen_schedule(
    output_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
  """Write a new immutable schedule bundle; never merge or resume one."""
  targets = [
      output_dir / SCHEDULE_FILE,
      output_dir / LEDGER_FILE,
      output_dir / LEDGER_SCHEMA_FILE,
      output_dir / MANIFEST_FILE,
  ]
  existing = [str(path) for path in targets if path.exists()]
  if existing:
    raise ScheduleBuildError(
        "Refusing to merge with an existing schedule bundle: "
        + ", ".join(existing)
    )
  output_dir.mkdir(parents=True, exist_ok=True)
  schema_bytes = (
      json.dumps(
          replacement_ledger_schema(),
          indent=2,
          sort_keys=True,
          ensure_ascii=False,
      )
      + "\n"
  ).encode("utf-8")
  payloads = {
      output_dir / SCHEDULE_FILE: _jsonl_bytes(schedule),
      output_dir / LEDGER_FILE: _jsonl_bytes(ledger),
      output_dir / LEDGER_SCHEMA_FILE: schema_bytes,
      output_dir / MANIFEST_FILE: (
          json.dumps(
              dict(manifest),
              indent=2,
              sort_keys=True,
              ensure_ascii=False,
          )
          + "\n"
      ).encode("utf-8"),
  }
  for path, data in payloads.items():
    _atomic_write(path, data)


def build_from_paths(cohort_path: Path, output_dir: Path) -> dict[str, Any]:
  """Validate, compile, and write one schedule bundle."""
  cohort, cohort_sha256 = load_frozen_cohort(cohort_path)
  schedule, ledger, manifest = compile_frozen_schedule(
      cohort, cohort_sha256
  )
  write_frozen_schedule(output_dir, schedule, ledger, manifest)
  return manifest


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cohort", default=str(DEFAULT_COHORT))
  parser.add_argument("--output_dir", required=True)
  args = parser.parse_args(argv)
  try:
    manifest = build_from_paths(
        Path(args.cohort).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
    )
  except ScheduleBuildError as exc:
    print(f"BLOCKED: {exc}", file=sys.stderr)
    return 2
  print(
      f"Wrote {manifest['episode_slot_count']} frozen, non-executing episode "
      f"slots across {manifest['paired_block_count']} interleaved blocks."
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
