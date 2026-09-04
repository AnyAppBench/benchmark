#!/usr/bin/env python3
"""Inventory executable verifier provenance for the frozen CATBench cohort.

This is a read-only source audit.  It resolves every task-app adapter in the
frozen cohort through the live AndroidWorld registry, records the method owner
for the task lifecycle and verifier, and separates semantic/template ancestry
from executable-class ancestry.

The audit deliberately does not infer that a verifier is correct from its
source provenance.  Conformance evidence is reported separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Final


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_ROOT.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

from android_world import registry  # pylint: disable=wrong-import-position
from app_generalization_profiles import (  # pylint: disable=wrong-import-position
    get_domain_profiles,
)
from scripts import catbench_primary_cohort  # pylint: disable=wrong-import-position


LOCAL_ANDROIDWORLD_IMPORT_COMMIT: Final[str] = (
    "828f74c848426015db718fb5cfd92c1941589d6c"
)

# These are semantic-intent correspondences, not claims that CATBench executes
# the named upstream class.  All executable classes in the frozen cohort are
# resolved independently below.
ANDROIDWORLD_INTENT_ORIGINS: Final[dict[str, tuple[str, str]]] = {
    "SmsSend": (
        "android_world.task_evals.single.sms", "SimpleSmsSend"
    ),
    "SmsReply": (
        "android_world.task_evals.single.sms", "SimpleSmsReply"
    ),
    "SmsReplyMostRecent": (
        "android_world.task_evals.single.sms", "SimpleSmsReplyMostRecent"
    ),
    "SmsResend": (
        "android_world.task_evals.single.sms", "SimpleSmsResend"
    ),
    "SmsSendReceivedAddress": (
        "android_world.task_evals.single.sms",
        "SimpleSmsSendReceivedAddress",
    ),
    "FilesDeleteFile": (
        "android_world.task_evals.single.files", "FilesDeleteFile"
    ),
    "FilesMoveFile": (
        "android_world.task_evals.single.files", "FilesMoveFile"
    ),
    "FilesSaveCopyOfFile": (
        "android_world.task_evals.single.simple_gallery_pro",
        "SaveCopyOfReceiptTaskEval",
    ),
    "ContactsAddContact": (
        "android_world.task_evals.single.contacts", "ContactsAddContact"
    ),
    "ContactsNewContactDraft": (
        "android_world.task_evals.single.contacts", "ContactsNewContactDraft"
    ),
    "MapsAddFavorite": (
        "android_world.task_evals.single.osmand", "OsmAndFavorite"
    ),
    "MapsAddMarker": (
        "android_world.task_evals.single.osmand", "OsmAndMarker"
    ),
    "MapsRecordTrack": (
        "android_world.task_evals.single.osmand", "OsmAndTrack"
    ),
    "ClockCreateTimer": (
        "android_world.task_evals.single.clock", "ClockTimerEntry"
    ),
    "ClockStopwatchRunning": (
        "android_world.task_evals.single.clock", "ClockStopWatchRunning"
    ),
    "ClockPauseStopwatch": (
        "android_world.task_evals.single.clock", "ClockStopWatchPausedVerify"
    ),
}


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
  return _sha256_bytes(path.read_bytes())


def _relative(path: str | Path | None) -> str:
  if not path:
    return ""
  resolved = Path(path).resolve()
  try:
    return resolved.relative_to(REPO_ROOT).as_posix()
  except ValueError:
    return str(resolved)


def _git(*args: str, check: bool = True) -> str:
  result = subprocess.run(
      ("git", *args),
      cwd=REPO_ROOT,
      check=False,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
  )
  if check and result.returncode:
    raise RuntimeError(
        f"git {' '.join(args)} failed: {result.stderr.strip()}"
    )
  return result.stdout.strip()


def _git_bytes(*args: str) -> bytes | None:
  result = subprocess.run(
      ("git", *args),
      cwd=REPO_ROOT,
      check=False,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  )
  return result.stdout if result.returncode == 0 else None


def _method_record(task_type: type[Any], method_name: str) -> dict[str, Any]:
  owner = None
  function = None
  for candidate in task_type.__mro__:
    if method_name in candidate.__dict__:
      owner = candidate
      function = candidate.__dict__[method_name]
      break
  if owner is None or function is None:
    raise ValueError(f"No {method_name} owner for {task_type.__name__}")
  source_file = inspect.getsourcefile(function) or inspect.getsourcefile(owner)
  source_line = None
  source_hash = ""
  try:
    source, source_line = inspect.getsourcelines(function)
    source_hash = _sha256_bytes("".join(source).encode("utf-8"))
  except (OSError, TypeError):
    pass
  return {
      "method": method_name,
      "owner_class": owner.__name__,
      "owner_module": owner.__module__,
      "source_file": _relative(source_file),
      "source_line": source_line,
      "source_sha256": source_hash,
  }


def _class_source(task_type: type[Any]) -> tuple[str, int | None]:
  source_file = inspect.getsourcefile(task_type)
  source_line = None
  try:
    _, source_line = inspect.getsourcelines(task_type)
  except (OSError, TypeError):
    pass
  return _relative(source_file), source_line


def _upstream_origin_record(semantic_task_id: str) -> dict[str, Any]:
  origin = ANDROIDWORLD_INTENT_ORIGINS.get(semantic_task_id)
  if origin is None:
    return {
        "semantic_origin": "catbench_new_semantic_template",
        "androidworld_reference_class": "",
        "androidworld_reference_source_file": "",
        "androidworld_reference_source_sha256": "",
        "androidworld_reference_local_import_source_sha256": "",
        "androidworld_reference_local_import_status": "not_applicable",
        "official_upstream_revision_verified": False,
    }
  module_name, class_name = origin
  module = importlib.import_module(module_name)
  task_type = getattr(module, class_name)
  source_file, source_line = _class_source(task_type)
  path = REPO_ROOT / source_file
  imported_blob = _git_bytes(
      "show", f"{LOCAL_ANDROIDWORLD_IMPORT_COMMIT}:{source_file}"
  )
  if imported_blob is None:
    local_status = "missing_from_local_import_commit"
  elif imported_blob == path.read_bytes():
    local_status = "unchanged_from_local_androidworld_import"
  else:
    local_status = "modified_after_local_androidworld_import"
  reference_verifier = _method_record(task_type, "is_successful")
  return {
      "semantic_origin": "androidworld_intent_adapted_by_catbench",
      "androidworld_reference_class": class_name,
      "androidworld_reference_module": module_name,
      "androidworld_reference_source_file": source_file,
      "androidworld_reference_source_line": source_line,
      "androidworld_reference_source_sha256": _sha256_file(path),
      "androidworld_reference_local_import_source_sha256": (
          _sha256_bytes(imported_blob) if imported_blob is not None else ""
      ),
      "androidworld_reference_verifier_owner": reference_verifier[
          "owner_class"
      ],
      "androidworld_reference_verifier_source_file": reference_verifier[
          "source_file"
      ],
      "androidworld_reference_verifier_source_line": reference_verifier[
          "source_line"
      ],
      "androidworld_reference_local_import_status": local_status,
      # No upstream remote or pinned import commit exists in this checkout.
      "official_upstream_revision_verified": False,
  }


def _audited_validation_mode(
    category: str,
    app_id: str,
    semantic_task_id: str,
    declared_mode: str,
) -> tuple[str, str]:
  """Return a source-informed strategy summary and metadata discrepancy."""
  if category == "sms":
    return "Telephony/SMS provider durable state", ""
  if category == "files":
    if semantic_task_id == "FilesViewFileInfo":
      return "UI exact-file plus info-surface heuristic", ""
    if semantic_task_id == "FilesShareFile":
      return "IntentResolver-owned exact filename preview", ""
    return "Filesystem predicate under /sdcard/CATBench", ""
  if category == "maps":
    return declared_mode or "undeclared", ""
  if category == "contacts":
    ui_terminal_semantics = {
        "ContactsSearchContact", "ContactsViewContactDetails"
    }
    external_transition_semantics = {
        "ContactsCallContact", "ContactsMessageContact"
    }
    destructive_transition_semantics = {
        "ContactsRemoveFavoriteContact", "ContactsDeleteContact"
    }
    sqlite_app_ids = {
        "contacts_fossify_contacts",
        "contacts_connect_you",
        "contacts_simple_contacts_pro_se",
    }
    if app_id in sqlite_app_ids:
      if semantic_task_id == "ContactsNewContactDraft":
        mode = "Editable UI + SQLite durable absence"
      elif semantic_task_id in ui_terminal_semantics:
        mode = "SQLite durable state + exact-target UI terminal state"
      elif semantic_task_id in external_transition_semantics:
        mode = "SQLite durable state + external intent surface"
      elif semantic_task_id in destructive_transition_semantics:
        mode = "SQLite durable state + exact-target transition latch"
      else:
        mode = "SQLite durable state"
    else:
      if semantic_task_id == "ContactsNewContactDraft":
        mode = "Editable UI + ContactsProvider durable absence"
      elif semantic_task_id in ui_terminal_semantics:
        mode = "ContactsProvider + exact-target UI terminal state"
      elif semantic_task_id in external_transition_semantics:
        mode = "ContactsProvider + external intent surface"
      elif semantic_task_id in destructive_transition_semantics:
        mode = (
            "ContactsProvider durable state + exact-target transition latch"
        )
      else:
        mode = "ContactsProvider durable state"
    discrepancy = ""
    if declared_mode != mode:
      discrepancy = (
          f"declared validation_mode is {declared_mode or 'undeclared'} "
          f"but source-informed audit expects {mode}"
      )
    return mode, discrepancy
  if category == "clock":
    if app_id == "clock_clockyou" and semantic_task_id in {
        "ClockCreateAlarm",
        "ClockEditAlarm",
        "ClockEnableAlarm",
        "ClockDeleteAlarm",
        "ClockAddWorldClock",
    }:
      audited_mode = "Clock You Room SQLite durable state"
      discrepancy = ""
      if declared_mode != audited_mode:
        discrepancy = (
            f"declared validation_mode is {declared_mode or 'undeclared'} "
            "but verifier branches to Clock You Room SQLite"
        )
      return audited_mode, discrepancy
    if app_id == "clock_clockyou" and semantic_task_id in {
        "ClockStopwatchRunning",
        "ClockPauseStopwatch",
        "ClockStopwatchReset",
    }:
      return "Clock You app-specific UI state/lifecycle heuristic", ""
    return declared_mode or "undeclared", ""
  return declared_mode or "undeclared", ""


def _conformance_record(
    category: str, app_id: str, semantic_task_id: str
) -> dict[str, Any]:
  result = {
      "g3_qualified": False,
      "narrow_fixture_status": "none",
      "narrow_fixture_evidence": "",
      "supplemental_helper_evidence": "",
  }
  if app_id == "clock_clockyou":
    result.update({
        "narrow_fixture_status": "42_of_42_state_fixture_cases_passed",
        "narrow_fixture_evidence": (
            "benchmark/docs/audits/clock_you_live_conformance_20260710.json"
        ),
    })
  if category == "files" and semantic_task_id not in {
      "FilesViewFileInfo", "FilesShareFile"
  }:
    result.update({
        "narrow_fixture_status": (
            "shared_storage_predicate_fixture_passed_not_UI_trajectory"
        ),
        "narrow_fixture_evidence": (
            "benchmark/docs/audits/"
            "docker_primary_base_candidate_v3_files_storage_"
            "conformance_r3_20260711.json"
        ),
    })
  if category == "maps":
    result["supplemental_helper_evidence"] = (
        "Exact/reversed/wrong-place GPX/KML/link helper smoke exists, but it "
        "is not per-adapter UI/verifier conformance."
    )
  return result


def _source_record(path: str) -> dict[str, Any]:
  source_path = REPO_ROOT / path
  introduced = _git(
      "log", "--diff-filter=A", "--follow", "--format=%H", "--", path
  ).splitlines()
  status = _git("status", "--porcelain", "--", path)
  return {
      "path": path,
      "sha256": _sha256_file(source_path),
      "introduced_commit": introduced[-1] if introduced else "",
      "worktree_status": status,
  }


def _generated_registry_inventory(
    task_registry: dict[str, type[Any]],
    frozen_task_names: set[str],
    profile_task_names: set[str],
) -> list[dict[str, Any]]:
  """List every registered class whose MRO enters a generated module."""
  rows: list[dict[str, Any]] = []
  for task_name, task_type in sorted(task_registry.items()):
    implementation_type = next((
        candidate for candidate in task_type.__mro__
        if "app_generalization_generated" in candidate.__module__
    ), None)
    if implementation_type is None:
      continue
    implementation_module = implementation_type.__module__
    implementation_file = _relative(inspect.getsourcefile(implementation_type))
    module_leaf = implementation_module.rsplit(".", 1)[-1]
    category = module_leaf.removesuffix("_cross_app_tasks").removesuffix(
        "_tasks"
    )
    rows.append({
        "task_class": task_name,
        "runtime_task_module": task_type.__module__,
        "generated_implementation_class": implementation_type.__name__,
        "generated_implementation_module": implementation_module,
        "generated_implementation_source_file": implementation_file,
        "category_hint": category,
        "semantic_task_id": str(
            getattr(task_type, "catbench_semantic_id", "") or ""
        ),
        "package_name": str(getattr(task_type, "package_name", "") or ""),
        "verifier": _method_record(task_type, "is_successful"),
        "scheduled_in_frozen_5cat_cohort": task_name in frozen_task_names,
        "referenced_by_current_domain_profiles": task_name in profile_task_names,
        "lineage_outside_frozen_scope": (
            "not_adjudicated_by_this_frozen-cohort_audit"
            if task_name not in frozen_task_names else "see_frozen_record"
        ),
    })
  return rows


def _profile_coverage() -> dict[str, Any]:
  profiles = get_domain_profiles()
  domains: dict[str, Any] = {}
  all_task_names: list[str] = []
  for domain, profile in profiles.items():
    task_names = [
        task_name
        for app in profile.apps
        for task_name in app.implemented_tasks
    ]
    all_task_names.extend(task_names)
    domains[domain] = {
        "app_count": len(profile.apps),
        "apps_with_implemented_tasks": sum(
            bool(app.implemented_tasks) for app in profile.apps
        ),
        "apps_with_zero_implemented_tasks": [
            app.app_id for app in profile.apps if not app.implemented_tasks
        ],
        "task_app_reference_count": len(task_names),
    }
  return {
      "domain_count": len(profiles),
      "domains": list(profiles),
      "missing_submitted_categories": [
          category for category in ("finance", "music")
          if category not in profiles
      ],
      "task_app_reference_count": len(all_task_names),
      "unique_task_class_name_count": len(set(all_task_names)),
      "domains_detail": domains,
      "task_names": sorted(set(all_task_names)),
  }


def build_audit(cohort_path: Path) -> dict[str, Any]:
  cohort = catbench_primary_cohort.load(cohort_path)
  task_registry = registry.TaskRegistry().get_registry(
      registry.TaskRegistry.ANDROID_WORLD_FAMILY
  )
  task_names, identities = catbench_primary_cohort.frozen_task_names(
      cohort, task_registry
  )
  profiles = get_domain_profiles()
  profile_coverage = _profile_coverage()
  profile_task_names = set(profile_coverage.pop("task_names"))
  apps = {
      (category, app.app_id): app
      for category, profile in profiles.items()
      for app in profile.apps
  }

  records: list[dict[str, Any]] = []
  source_files: set[str] = set()
  for task_name in task_names:
    category, app_id = identities[task_name]
    app = apps[(category, app_id)]
    task_type = task_registry[task_name]
    semantic_task_id = str(task_type.catbench_semantic_id)
    base_type = task_type.__mro__[1]
    base_source_file, base_source_line = _class_source(base_type)
    source_files.add(base_source_file)
    verifier = _method_record(task_type, "is_successful")
    initializer = _method_record(task_type, "initialize_task")
    teardown = _method_record(task_type, "tear_down")
    declared_mode = str(getattr(task_type, "validation_mode", "") or "")
    audited_mode, metadata_discrepancy = _audited_validation_mode(
        category, app_id, semantic_task_id, declared_mode
    )
    upstream = _upstream_origin_record(semantic_task_id)
    record = {
        "release_id": cohort["release_id"],
        "category": category,
        "semantic_task_id": semantic_task_id,
        "app_id": app_id,
        "app_display_name": app.display_name,
        "package_name": app.package_name,
        "is_androidworld_baseline_app": (
            app_id == cohort["categories"][category]["aw_app_id"]
        ),
        "task_class": task_name,
        "runtime_task_module": task_type.__module__,
        "task_creation": "CATBench dynamic type() fan-out",
        "generated_base_class": base_type.__name__,
        "generated_base_source_file": base_source_file,
        "generated_base_source_line": base_source_line,
        "method_resolution_order": [
            cls.__name__ for cls in task_type.__mro__[:-1]
        ],
        "verifier": verifier,
        "initializer": initializer,
        "teardown": teardown,
        "declared_validation_mode": declared_mode,
        "audited_validation_mode": audited_mode,
        "validation_mode_metadata_discrepancy": metadata_discrepancy,
        **upstream,
        # The live registry resolves a generated subclass for every frozen
        # app-task pair, including AW-baseline apps and AW-intent templates.
        "exact_androidworld_task_class_reused": False,
        "exact_androidworld_verifier_method_reused": False,
        "executable_lineage": (
            "catbench_generated_adapter_of_androidworld_intent"
            if upstream["semantic_origin"].startswith("androidworld")
            else "catbench_generated_new_semantic_adapter"
        ),
        "conformance": _conformance_record(
            category, app_id, semantic_task_id
        ),
    }
    records.append(record)

  semantic_rows: list[dict[str, Any]] = []
  for category, category_spec in cohort["categories"].items():
    for semantic_task_id in category_spec["semantic_task_ids"]:
      rows = [
          row for row in records
          if row["category"] == category
          and row["semantic_task_id"] == semantic_task_id
      ]
      semantic_rows.append({
          "category": category,
          "semantic_task_id": semantic_task_id,
          "semantic_origin": rows[0]["semantic_origin"],
          "androidworld_reference_class": rows[0][
              "androidworld_reference_class"
          ],
          "generated_base_class": rows[0]["generated_base_class"],
          "verifier_owner_class": rows[0]["verifier"]["owner_class"],
          "source_file": rows[0]["generated_base_source_file"],
          "app_adapter_count": len(rows),
          "task_classes": [row["task_class"] for row in rows],
          "declared_validation_modes": sorted({
              row["declared_validation_mode"] or "<undeclared>"
              for row in rows
          }),
          "audited_validation_modes": sorted({
              row["audited_validation_mode"] for row in rows
          }),
          "g3_qualified_adapter_count": sum(
              bool(row["conformance"]["g3_qualified"]) for row in rows
          ),
      })

  aw_semantics = sum(
      row["semantic_origin"].startswith("androidworld")
      for row in semantic_rows
  )
  aw_adapter_rows = sum(
      row["semantic_origin"].startswith("androidworld") for row in records
  )
  source_records = [_source_record(path) for path in sorted(source_files)]
  generated_registry = _generated_registry_inventory(
      task_registry, set(task_names), profile_task_names
  )
  generated_profile_references = sum(
      row["referenced_by_current_domain_profiles"]
      for row in generated_registry
  )
  profile_coverage["generated_registry_reference_count"] = (
      generated_profile_references
  )
  profile_coverage["non_generated_reference_count"] = (
      profile_coverage["task_app_reference_count"]
      - generated_profile_references
  )
  registry_module_counts: dict[str, int] = {}
  for row in generated_registry:
    module = row["generated_implementation_module"]
    registry_module_counts[module] = registry_module_counts.get(module, 0) + 1
  head = _git("rev-parse", "HEAD")
  return {
      "schema_version": 1,
      "audit_type": "catbench_frozen_verifier_source_provenance",
      "audit_date": "2026-07-12",
      "scope": {
          "release_id": cohort["release_id"],
          "cohort_path": _relative(cohort_path),
          "cohort_sha256": _sha256_file(cohort_path),
          "categories": list(cohort["categories"]),
          "scope_rule": (
              "Every task-app adapter resolved by frozen_task_names for the "
              "current five-category replacement cohort"
          ),
          "excluded": (
              "Unscheduled/legacy generated classes, the withdrawn ten-"
              "category artifact grid, and ordinary AndroidWorld tasks not "
              "named by the frozen cohort"
          ),
      },
      "repository": {
          "head_commit": head,
          "head_short": head[:12],
          "remote_urls": _git("remote", "-v").splitlines(),
          "local_androidworld_import_commit": LOCAL_ANDROIDWORLD_IMPORT_COMMIT,
          "official_androidworld_import_revision": None,
          "official_upstream_revision_verified": False,
          "provenance_limitation": (
              "This checkout has only the CATBench origin remote and does not "
              "pin the exact google-research/android_world import revision. "
              "Unchanged/modified labels for reference classes are therefore "
              "relative to the repository's local AndroidWorld import commit, "
              "not a cryptographically verified official upstream revision."
          ),
          "source_files": source_records,
      },
      "audit_implementation": {
          "script_path": _relative(Path(__file__)),
          "script_sha256": _sha256_file(Path(__file__)),
          "registry_path": "benchmark/android_world/registry.py",
          "registry_sha256": _sha256_file(
              BENCHMARK_ROOT / "android_world/registry.py"
          ),
          "profiles_path": "benchmark/app_generalization_profiles.py",
          "profiles_sha256": _sha256_file(
              BENCHMARK_ROOT / "app_generalization_profiles.py"
          ),
          "cohort_resolver_path": (
              "benchmark/scripts/catbench_primary_cohort.py"
          ),
          "cohort_resolver_sha256": _sha256_file(
              BENCHMARK_ROOT / "scripts/catbench_primary_cohort.py"
          ),
      },
      "summary": {
          "category_count": len(cohort["categories"]),
          "semantic_template_count": len(semantic_rows),
          "adapter_count": len(records),
          "androidworld_intent_adapted_semantic_templates": aw_semantics,
          "catbench_new_semantic_templates": len(semantic_rows) - aw_semantics,
          "androidworld_intent_adapted_adapter_rows": aw_adapter_rows,
          "catbench_new_adapter_rows": len(records) - aw_adapter_rows,
          "exact_androidworld_task_classes_reused": sum(
              row["exact_androidworld_task_class_reused"] for row in records
          ),
          "exact_androidworld_verifier_methods_reused": sum(
              row["exact_androidworld_verifier_method_reused"]
              for row in records
          ),
          "g3_qualified_adapters": sum(
              row["conformance"]["g3_qualified"] for row in records
          ),
          "validation_mode_metadata_discrepancies": sum(
              bool(row["validation_mode_metadata_discrepancy"])
              for row in records
          ),
      },
      "universe_distinctions": {
          "frozen_replacement_cohort": {
              "category_count": len(cohort["categories"]),
              "semantic_template_count": len(semantic_rows),
              "task_app_adapter_count": len(records),
              "is_the_exhaustive_per_adapter_scope_of_this_audit": True,
          },
          "current_domain_profiles": profile_coverage,
          "all_registered_generated_classes": {
              "class_count": len(generated_registry),
              "module_counts": dict(sorted(registry_module_counts.items())),
              "warning": (
                  "Registry presence does not imply current scheduling, a "
                  "semantic ID, verifier qualification, or inclusion in the "
                  "frozen cohort."
              ),
          },
          "submitted_legacy_grid": {
              "reported_category_count": 10,
              "reported_app_count": 52,
              "reported_task_app_combination_count": 520,
              "same_as_frozen_replacement_cohort": False,
              "same_as_current_domain_profiles": False,
              "same_as_all_registered_generated_classes": False,
              "warning": (
                  "The submitted 10-category/520-combination claim is a "
                  "legacy artifact universe and cannot be reconstructed by "
                  "treating the current profiles or registry as equivalent."
              ),
          },
      },
      "semantic_templates": semantic_rows,
      "records": records,
      "registered_generated_classes": generated_registry,
  }


def _csv_text(records: list[dict[str, Any]]) -> str:
  fields = (
      "release_id", "category", "semantic_task_id", "app_id",
      "app_display_name", "package_name", "is_androidworld_baseline_app",
      "task_class", "generated_base_class", "generated_base_source_file",
      "generated_base_source_line", "verifier_owner_class",
      "verifier_source_file", "verifier_source_line", "verifier_source_sha256",
      "initializer_owner_class", "initializer_source_file",
      "teardown_owner_class", "semantic_origin", "androidworld_reference_class",
      "androidworld_reference_source_file",
      "androidworld_reference_source_sha256",
      "androidworld_reference_local_import_source_sha256",
      "androidworld_reference_local_import_status",
      "official_upstream_revision_verified", "exact_androidworld_task_class_reused",
      "exact_androidworld_verifier_method_reused", "executable_lineage",
      "declared_validation_mode", "audited_validation_mode",
      "validation_mode_metadata_discrepancy", "g3_qualified",
      "narrow_fixture_status", "narrow_fixture_evidence",
      "supplemental_helper_evidence", "method_resolution_order",
  )
  buffer = io.StringIO()
  writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
  writer.writeheader()
  for record in records:
    writer.writerow({
        **{key: record.get(key, "") for key in fields},
        "verifier_owner_class": record["verifier"]["owner_class"],
        "verifier_source_file": record["verifier"]["source_file"],
        "verifier_source_line": record["verifier"]["source_line"],
        "verifier_source_sha256": record["verifier"]["source_sha256"],
        "initializer_owner_class": record["initializer"]["owner_class"],
        "initializer_source_file": record["initializer"]["source_file"],
        "teardown_owner_class": record["teardown"]["owner_class"],
        "g3_qualified": record["conformance"]["g3_qualified"],
        "narrow_fixture_status": record["conformance"][
            "narrow_fixture_status"
        ],
        "narrow_fixture_evidence": record["conformance"][
            "narrow_fixture_evidence"
        ],
        "supplemental_helper_evidence": record["conformance"][
            "supplemental_helper_evidence"
        ],
        "method_resolution_order": ";".join(
            record["method_resolution_order"]
        ),
    })
  return buffer.getvalue()


def _markdown_text(audit: dict[str, Any], json_name: str, csv_name: str) -> str:
  summary = audit["summary"]
  source_rows = audit["repository"]["source_files"]
  universes = audit["universe_distinctions"]
  profile_coverage = universes["current_domain_profiles"]
  lines = [
      "# CATBench frozen verifier provenance audit (2026-07-12)",
      "",
      "## Scope and conclusion",
      "",
      "This audit resolves every executable task class in the current frozen ",
      "five-category replacement cohort through the live registry: 5 categories, ",
      "50 semantic templates, 23 apps, and 230 task-app adapters. It excludes ",
      "unscheduled compatibility classes and the withdrawn legacy ten-category ",
      "artifact grid.",
      "",
      "**No frozen adapter executes an unchanged AndroidWorld task class or an ",
      "unchanged AndroidWorld verifier method.** Every one of the 230 classes is ",
      "created by CATBench's dynamic per-app fan-out and inherits a CATBench base ",
      "evaluator under `app_generalization_generated`. Sixteen of the 50 semantic ",
      "templates adapt AndroidWorld task intent; the other 34 are CATBench-new. ",
      "This semantic ancestry must not be described as executable-verifier reuse.",
      "The three OsmAnd semantic references (`OsmAndFavorite`, `OsmAndMarker`, ",
      "and `OsmAndTrack`) also come from a local source file modified after the ",
      "repository's AndroidWorld import; the generated Maps evaluators still do ",
      "not execute those reference classes.",
      "",
      f"Machine-readable records: [`{json_name}`]({json_name}) and ",
      f"[`{csv_name}`]({csv_name}).",
      "",
      "## Counts",
      "",
      "| Quantity | Count |",
      "|---|---:|",
      f"| Categories | {summary['category_count']} |",
      f"| Semantic templates | {summary['semantic_template_count']} |",
      f"| Task-app adapter classes | {summary['adapter_count']} |",
      f"| AndroidWorld-intent-adapted templates | {summary['androidworld_intent_adapted_semantic_templates']} |",
      f"| CATBench-new templates | {summary['catbench_new_semantic_templates']} |",
      f"| AndroidWorld-intent-adapted adapter rows | {summary['androidworld_intent_adapted_adapter_rows']} |",
      f"| CATBench-new adapter rows | {summary['catbench_new_adapter_rows']} |",
      f"| Exact AndroidWorld task classes reused | {summary['exact_androidworld_task_classes_reused']} |",
      f"| Exact AndroidWorld verifier methods reused | {summary['exact_androidworld_verifier_methods_reused']} |",
      f"| Fully G3-qualified adapters | {summary['g3_qualified_adapters']} |",
      f"| Declared validation-mode discrepancies | {summary['validation_mode_metadata_discrepancies']} |",
      "",
      "## Do not conflate the four task universes",
      "",
      f"1. **Frozen replacement cohort:** {universes['frozen_replacement_cohort']['task_app_adapter_count']} adapters across five categories. This is the exhaustive per-adapter scope below.",
      f"2. **Current domain profiles:** {profile_coverage['task_app_reference_count']} task-app references across {profile_coverage['domain_count']} profile domains. Finance and Music are absent from `get_domain_profiles()`.",
      f"3. **All registered generated/compatibility classes:** {universes['all_registered_generated_classes']['class_count']} registry entries. Registration does not mean scheduling or qualification; the full list is retained in the JSON.",
      "4. **Submitted legacy grid:** 10 categories, 52 apps, and 520 reported task-app combinations. It is not any of the three current runtime sets above.",
      "",
      "The current Notes profile gives implementations to only 2/7 apps ",
      "(five have zero implemented tasks), and the current Todo profile gives ",
      "implementations to only 2/8 apps (six have zero). The ntodotxt entries ",
      "are explicitly temporary shims over Tasks.org information-retrieval ",
      "evaluators, not an app-native verifier port. Current Maps profiles also ",
      "deschedule Google Maps and MAPS.ME. These gaps prevent reconstructing a ",
      "uniform current ten-category/520-cell benchmark from profile metadata.",
      "",
      "## Semantic-template inventory",
      "",
      "`AW intent` means that CATBench adapted the purpose of a named ",
      "AndroidWorld task; it does not mean the original class or verifier is ",
      "executed.",
      "",
      "| Category | Semantic template | Origin | AndroidWorld reference | CATBench verifier owner | Apps |",
      "|---|---|---|---|---|---:|",
  ]
  for row in audit["semantic_templates"]:
    origin = (
        "AW intent"
        if row["semantic_origin"].startswith("androidworld")
        else "CATBench-new"
    )
    reference = row["androidworld_reference_class"] or "—"
    lines.append(
        f"| {row['category']} | `{row['semantic_task_id']}` | {origin} | "
        f"`{reference}` | `{row['verifier_owner_class']}` | "
        f"{row['app_adapter_count']} |"
    )
  lines.extend([
      "",
      "## Source-file provenance",
      "",
      "All five executable verifier modules were introduced after the local ",
      "AndroidWorld import commit and are currently modified in the working tree. ",
      "The byte hashes below, not `HEAD` alone, identify what this audit inspected.",
      "",
      "| Source | SHA-256 | Introduced commit | Worktree status |",
      "|---|---|---|---|",
  ])
  for row in source_rows:
    lines.append(
        f"| `{row['path']}` | `{row['sha256']}` | "
        f"`{row['introduced_commit'][:12]}` | `{row['worktree_status'] or 'clean'}` |"
    )
  lines.extend([
      "",
      "## Verifier strategy and conformance boundary",
      "",
      "- SMS adapters use shared telephony/SMS-provider state.",
      "- Eight Files semantics use filesystem predicates; ViewInfo is a UI ",
      "  heuristic and Share requires an IntentResolver-owned exact filename.",
      "- Maps mixes filesystem/SQLite predicates with transient UI heuristics.",
      "- Contacts mixes ContactsProvider/private SQLite evidence with UI ",
      "  terminal-state and external-transition checks.",
      "- Clock mixes UI heuristics, Fossify SQLite, and Clock You Room/UI ",
      "  branches. Five Clock You durable-state classes still declare ",
      "  `validation_mode = UI heuristic`; the machine inventory flags these ",
      "  metadata discrepancies without changing benchmark logic.",
      "",
      "Source provenance is not verifier correctness. The current protocol ",
      "records 42/42 narrow Clock You state-fixture cases and 185/185 narrow ",
      "Files storage-predicate cases, but no adapter has the required gold UI ",
      "trajectory, native-state package, reset/replay, and signed adjudication ",
      "for full G3 qualification. Maps helper smoke is also not adapter ",
      "conformance.",
      "",
      "## Important limitations",
      "",
      "1. This checkout has only the CATBench `origin` remote. It does not pin ",
      "   the exact `google-research/android_world` import revision. Reference ",
      "   class unchanged/modified labels in the JSON are relative to local ",
      f"   import commit `{LOCAL_ANDROIDWORLD_IMPORT_COMMIT}`, not a verified official upstream revision.",
      "2. Dynamic generated classes report runtime module `abc`; their actual ",
      "   implementation provenance is the first generated base and method owner, ",
      "   which this audit resolves explicitly.",
      "3. Current source provenance cannot be projected onto legacy C1 pickle ",
      "   files unless each artifact proves the executed code revision and source ",
      "   hashes. The preserved legacy manifests generally do not.",
      "4. The cohort JSON's local `status` field does not override the revision ",
      "   protocol: execution remains blocked while G3/G4 and other launch gates ",
      "   are incomplete.",
      "",
  ])
  return "\n".join(lines)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--cohort",
      default=str(BENCHMARK_ROOT / "configs/catbench_5cat_primary_cohort.json"),
  )
  parser.add_argument(
      "--output_dir",
      default=str(BENCHMARK_ROOT / "docs/audits"),
  )
  parser.add_argument(
      "--stem", default="all_verifier_provenance_audit_20260712"
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  cohort_path = Path(args.cohort).expanduser().resolve()
  output_dir = Path(args.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  audit = build_audit(cohort_path)
  json_path = output_dir / f"{args.stem}.json"
  csv_path = output_dir / f"{args.stem}.csv"
  md_path = output_dir / f"{args.stem}.md"
  json_path.write_text(
      json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  csv_path.write_text(_csv_text(audit["records"]), encoding="utf-8")
  md_path.write_text(
      _markdown_text(audit, json_path.name, csv_path.name), encoding="utf-8"
  )
  print(json.dumps({
      "json": str(json_path),
      "csv": str(csv_path),
      "markdown": str(md_path),
      "summary": audit["summary"],
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
