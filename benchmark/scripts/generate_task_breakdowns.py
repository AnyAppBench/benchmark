#!/usr/bin/env python3
"""Generate CATBench task-breakdown JSON with a text-only Gemini model.

The generated file is consumed at runtime by setting:

  CATBENCH_TASK_BREAKDOWN_FILE=/path/to/breakdowns.json
  CATBENCH_TASK_BREAKDOWN_MODE=prepend
"""

from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_ROOT.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

try:
  import pysqlite3

  sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ModuleNotFoundError:
  pass

import task_breakdowns
import catbench_primary_cohort
from android_world import registry
from android_world import suite_utils
from app_generalization_profiles import get_domain_profiles


DEFAULT_CATEGORIES = ("sms", "files", "maps", "contacts", "clock")
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.1")
DEFAULT_QWEN_MODEL = os.environ.get("QWEN_C2_MODEL", "catbench-judge")
ATTEMPT_AUDIT_SCHEMA_VERSION = 1
_AUDIT_GENESIS_SHA256 = "0" * 64

# App names that should never appear in an application-independent breakdown.
# Generic domain nouns such as "files" and "messages" are intentionally omitted:
# they are valid high-level task concepts and would otherwise reject clean
# application-independent breakdowns for the file-manager and SMS categories.
_APP_NAME_TOKENS = (
    "Google Maps", "OsmAnd", "Kashgar", "KashCal", "Google Calendar",
    "Simple Calendar", "Tasks.org", "Markor", "Joplin", "Standard Notes",
    "Google Keep", "Fossify Clock", "Fossify Calculator", "Fossify Contacts",
    "Fossify Notes", "Fossify Gallery", "Fossify Files", "Fossify SMS",
    "Files by Google", "Amaze", "Gmail", "Open Contacts",
    "OpenContacts", "Google Messages", "Signal", "Telegram", "Quik",
    "Calculator", "Audio Recorder", "Drive", "Chrome",
)
_APP_NAME_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(name) for name in _APP_NAME_TOKENS) + r")\b",
    re.IGNORECASE,
)

_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_COORDINATE_PAIR_PATTERN = re.compile(
    rf"(?<![\d.])(?P<latitude>{_NUMBER_PATTERN})\s*,\s*"
    rf"(?P<longitude>{_NUMBER_PATTERN})(?!\d|\.\d)"
)

# Thinking tokens of gemini-3.x count against the output budget; 4096 produced
# truncated JSON in the 2026-07-13 run.
_GEMINI_MAX_OUTPUT_TOKENS = 8192

FORBIDDEN_PATTERNS = {
    "coordinate_pair": _COORDINATE_PAIR_PATTERN,
    "x_y_coordinate": re.compile(
        rf"\b[xy]\s*[:=]\s*{_NUMBER_PATTERN}(?!\d|\.\d)",
        re.IGNORECASE,
    ),
    "coordinate_word": re.compile(r"\bcoordinates?\b", re.IGNORECASE),
    "accessibility_node": re.compile(
        r"\b(accessibility|node id|resource-id|content-desc)\b",
        re.IGNORECASE,
    ),
    "button_label_instruction": re.compile(
        r"\b(button labeled|tap the .+ button|click the .+ button|"
        r"press the .+ icon|tap the .+ icon|select the .+ tab|"
        r"open the hamburger menu|the FAB|floating action button|"
        r"swipe down (on |from )?(the )?notification (shade|drawer|panel))\b",
        re.IGNORECASE,
    ),
    "app_name_mention": _APP_NAME_PATTERN,
    # Keep parity with preflight_task_breakdowns.FORBIDDEN_PLAN_PATTERNS so a
    # plan accepted here is not rejected later by the launch preflight.
    "android_package_identifier": re.compile(
        r"\b(?:[A-Za-z][A-Za-z0-9_]*\.){2,}[A-Za-z][A-Za-z0-9_]*\b"
    ),
    "low_level_ui_action": re.compile(
        r"\b(tap|click|double[- ]tap|long[- ]press|swipe|scroll)\b",
        re.IGNORECASE,
    ),
    "app_specific_control": re.compile(
        r"\b(button labeled|press the .+ button|press the .+ icon|"
        r"select the .+ tab|open the hamburger menu|the FAB|"
        r"floating action button|notification (shade|drawer|panel))\b",
        re.IGNORECASE,
    ),
}

# Coordinate allowances are occurrence-scoped. Direct labels and clearly
# geographic references may be exempted; unrelated coordinate words may not.
_COORDINATE_LABEL_BEFORE_PATTERN = re.compile(
    r"(?P<label>\b(?:geographic\s+)?coordinates?\b)"
    r"\s*(?:(?:of|at)\s*)?(?::|=)?\s*[\[(]?\s*$",
    re.IGNORECASE,
)
_COORDINATE_LABEL_AFTER_PATTERN = re.compile(
    r"^\s*[\])]?(?:\s+(?:as|in))?(?:\s+the)?\s+"
    r"(?P<label>(?:geographic\s+)?coordinates?\b)",
    re.IGNORECASE,
)
_UI_COORDINATE_PREFIX_PATTERN = re.compile(
    r"(?:\b(?:screen|pixel|display|canvas)\b[^.!?;\n]{0,40}|"
    r"\b(?:tap|click|press|swipe|scroll|drag|long[- ]press)\b"
    r"[^.!?;\n]{0,40}\b(?:at|to|from)\b[^.!?;\n]{0,20})$",
    re.IGNORECASE,
)
_UI_COORDINATE_REFERENCE_PATTERN = re.compile(
    r"\b(?:screen|pixel|display|canvas|tap|click|press|swipe|scroll|drag|"
    r"long[- ]press)\b|\b[xy]\s*[:=]",
    re.IGNORECASE,
)
_GEOGRAPHIC_COORDINATE_REFERENCE_PATTERN = re.compile(
    r"\b(?:geographic|latitude|longitude|location|marker)\b|"
    r"\bcoordinates?\b[^.!?;\n]{0,40}"
    r"\b(?:entered|input|provided)\b",
    re.IGNORECASE,
)

# These are ordinary domain nouns and may legitimately appear in an
# app-independent plan even when one target app happens to use that noun as its
# display name. Branded/multiword app names remain forbidden dynamically.
_GENERIC_APP_DISPLAY_NAMES = {
    "clock",
    "contacts",
    "files",
    "maps",
    "messages",
}


def _canonical_json(value: Any) -> str:
  """Return the one serialization used by every audit-chain hash."""
  return json.dumps(
      value,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
      allow_nan=False,
  )


def _json_sha256(value: Any) -> str:
  return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_json_loads(text: str) -> Any:
  """Parse JSON while rejecting duplicate keys and non-finite numbers."""

  def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f"Duplicate JSON key in attempt audit: {key}")
      result[key] = value
    return result

  def _constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON value in attempt audit: {value}")

  return json.loads(
      text,
      object_pairs_hook=_object,
      parse_constant=_constant,
  )


_SECRET_FIELD_PATTERN = re.compile(
    r"(?:authorization|api[-_ ]?key|access[-_ ]?token|secret)",
    re.IGNORECASE,
)
_AUTH_VALUE_PATTERN = re.compile(
    r"(?i)((?:authorization|api[-_ ]?key|access[-_ ]?token|secret)"
    r"\s*[:=]\s*)(?:bearer\s+)?[^\s,;}\]]+"
)
_QUERY_SECRET_PATTERN = re.compile(
    r"(?i)([?&](?:key|api_key|token|access_token)=)[^&#\s]+"
)


def _redact_text(text: str, secrets: tuple[str, ...]) -> tuple[str, bool]:
  """Remove credentials without hiding ordinary provider failure details."""
  redacted = text
  for secret in secrets:
    if secret:
      redacted = redacted.replace(secret, "[REDACTED]")
  redacted = _AUTH_VALUE_PATTERN.sub(r"\1[REDACTED]", redacted)
  redacted = _QUERY_SECRET_PATTERN.sub(r"\1[REDACTED]", redacted)
  return redacted, redacted != text


def _sanitize_audit_value(value: Any, secrets: tuple[str, ...]) -> Any:
  """Recursively strip secret-bearing fields and redact textual values."""
  if isinstance(value, str):
    return _redact_text(value, secrets)[0]
  if isinstance(value, list):
    return [_sanitize_audit_value(item, secrets) for item in value]
  if isinstance(value, tuple):
    return [_sanitize_audit_value(item, secrets) for item in value]
  if isinstance(value, dict):
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
      normalized_key = str(key)
      if _SECRET_FIELD_PATTERN.search(normalized_key):
        sanitized[normalized_key] = "[REDACTED]"
      else:
        sanitized[normalized_key] = _sanitize_audit_value(item, secrets)
    return sanitized
  return value


def _safe_endpoint_identity(raw_url: str) -> str:
  """Return a credential-free endpoint identity for the audit binding."""
  parsed = urllib.parse.urlsplit(raw_url)
  hostname = parsed.hostname or ""
  if ":" in hostname and not hostname.startswith("["):
    hostname = f"[{hostname}]"
  port = f":{parsed.port}" if parsed.port is not None else ""
  netloc = f"{hostname}{port}"
  return urllib.parse.urlunsplit(
      (parsed.scheme.lower(), netloc, parsed.path, "", "")
  )


def _parse_csv(raw: str | None) -> tuple[str, ...]:
  if not raw:
    return ()
  return tuple(item.strip() for item in raw.split(",") if item.strip())


def _tasks_for_categories(categories: tuple[str, ...]) -> list[str]:
  profiles = get_domain_profiles()
  unknown = [category for category in categories if category not in profiles]
  if unknown:
    raise ValueError(f"Unknown categories: {unknown}. Valid: {sorted(profiles)}")

  tasks: list[str] = []
  seen: set[str] = set()
  for category in categories:
    for app in profiles[category].apps:
      for task_name in app.implemented_tasks:
        if task_name not in seen:
          seen.add(task_name)
          tasks.append(task_name)
  return tasks


def _iter_task_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
  task_registry = registry.TaskRegistry()
  full_registry = task_registry.get_registry(family=args.suite_family)
  task_identities: dict[str, tuple[str, str]] = {}
  selected_dry_run_instances: dict[tuple[str, str, str], int] | None = None
  if args.cohort_manifest:
    if args.tasks or args.limit:
      raise ValueError(
          "Frozen-cohort generation forbids --tasks and --limit."
      )
    cohort = catbench_primary_cohort.load(args.cohort_manifest)
    schedule_issues = catbench_primary_cohort.validate_schedule_args(
        cohort,
        suite_family=args.suite_family,
        categories=_parse_csv(args.categories),
        n_task_combinations=args.n_task_combinations,
        task_random_seed=args.task_random_seed,
        fixed_task_seed=args.fixed_task_seed,
    )
    if schedule_issues:
      raise ValueError(
          "Frozen-cohort schedule mismatch:\n- " + "\n- ".join(schedule_issues)
      )
    task_names, task_identities = catbench_primary_cohort.frozen_task_names(
        cohort, full_registry
    )
    if "paired_blocks" in cohort:
      # G6 is a distinct five-block discard-only release, not all K=3
      # instances of its five task templates.  Enumerate only the immutable
      # identities in that release before any possible provider call.
      schedule_cohort = {
          key: value for key, value in cohort.items() if key != "_path"
      }
      import build_catbench_frozen_schedule as frozen_schedule_builder

      frozen_schedule_builder.validate_g6_dry_run_cohort(schedule_cohort)
      selected_dry_run_instances = {}
      for block in cohort["paired_blocks"]:
        identity = (
            str(block["category"]),
            str(block["app_id"]),
            str(block["semantic_task_id"]),
        )
        if identity in selected_dry_run_instances:
          raise ValueError(f"Duplicate G6 dry-run task identity: {identity}")
        selected_dry_run_instances[identity] = int(block["instance_id"])
  else:
    task_names = list(_parse_csv(args.tasks))
    if not task_names:
      task_names = _tasks_for_categories(_parse_csv(args.categories))
  missing = [task_name for task_name in task_names if task_name not in full_registry]
  if missing:
    raise ValueError(f"Task(s) not in {args.suite_family} registry: {missing}")

  suite = suite_utils.create_suite(
      full_registry,
      n_task_combinations=args.n_task_combinations,
      seed=args.task_random_seed,
      tasks=task_names,
      use_identical_params=args.fixed_task_seed,
  )

  instances: list[dict[str, Any]] = []
  for task_template, tasks in suite.items():
    for instance_id, task in enumerate(tasks):
      task_type = type(task)
      semantic_task_id = str(
          getattr(task_type, "catbench_semantic_id", task_template)
      )
      category, app_id = task_identities.get(task_template, (None, None))
      if selected_dry_run_instances is not None:
        selected_instance = selected_dry_run_instances.get(
            (str(category), str(app_id), semantic_task_id)
        )
        if selected_instance is None or instance_id != selected_instance:
          continue
      app_display_name = getattr(
          task_type, "catbench_app_display_name", None
      )
      semantic_goal = task_breakdowns.app_neutral_goal(
          task.goal, app_display_name
      )
      parameter_json = json.dumps(
          getattr(task, "params", {}),
          sort_keys=True,
          separators=(",", ":"),
          ensure_ascii=False,
          default=str,
      )
      instances.append(
          {
              "task_template": task_template,
              "instance_id": instance_id,
              "goal": task.goal,
              "goal_sha256": task_breakdowns.goal_sha256(task.goal),
              "key": task_breakdowns.make_key(
                  task_template, task.goal, instance_id
              ),
              "semantic_task_id": semantic_task_id,
              "category": category,
              "app_id": app_id,
              "app_display_name": app_display_name,
              "semantic_goal": semantic_goal,
              "semantic_goal_sha256": task_breakdowns.goal_sha256(
                  semantic_goal
              ),
              "semantic_parameter_sha256": hashlib.sha256(
                  parameter_json.encode("utf-8")
              ).hexdigest(),
              "plan_key": task_breakdowns.make_semantic_plan_key(
                  semantic_task_id, instance_id, semantic_goal
              ),
          }
      )
  if args.limit:
    instances = instances[: args.limit]
  return instances


def _prompt_template() -> str:
  """The placeholder-free template used to compute a prompt SHA for versioning."""
  return _prompt_for_goal("__GOAL_PLACEHOLDER__")


def _prompt_sha256() -> str:
  return hashlib.sha256(_prompt_template().encode("utf-8")).hexdigest()


def _prompt_for_goal(goal: str) -> str:
  return f"""You are producing an application-independent task breakdown for a mobile automation benchmark.

You receive only the user instruction below. You do not see the screen, any app metadata beyond the instruction, app-specific UI layout, accessibility tree, screenshots, or action history.

Write a high-level step-by-step breakdown of what must be accomplished. The breakdown must help another agent execute the task, but it must not solve visual grounding for that agent.

Rules:
- Use only the user instruction and ordinary domain knowledge.
- Do not invent screen or pixel coordinates, pixel locations, accessibility-node identifiers, resource IDs, or content descriptions. Geographic latitude/longitude values explicitly present in the user instruction are task values and must be preserved.
- Do not name app-specific controls, button labels, menu labels, tab labels, or screen layouts.
- Do not write low-level UI actions such as "tap the Add button"; use interface-neutral phrasing such as "start the workflow for creating the item".
- You may preserve concrete task values from the instruction, such as names, message text, dates, times, file names, and destinations.
- Keep the breakdown concise, usually 3 to 7 steps.

Return valid JSON only, with this shape:
{{
  "steps": [
    "First high-level accomplishment.",
    "Second high-level accomplishment."
  ],
  "notes": []
}}

User instruction:
{goal}
"""


def _repair_prompt_for_goal(
    goal: str,
    prior_breakdown: dict[str, Any],
    warnings: list[str],
) -> str:
  return f"""{_prompt_for_goal(goal)}

The previous JSON draft was rejected by validation warnings: {warnings}.

Rewrite the breakdown as valid JSON with the same schema, fixing only those
violations. Do not mention or repeat the app name, app brand, app-specific UI
labels, or app-specific controls from the user instruction. Use generic domain
phrasing instead, such as "map", "location", "contact", "file", "message",
"alarm", or "timer" when appropriate. Do not use low-level UI action verbs
(tap, click, double-tap, long-press, swipe, scroll), dotted package
identifiers, screen coordinates, or accessibility identifiers anywhere in the
steps or notes. Preserve concrete task values such as
names, message text, dates, times, file names, and destinations.

Rejected draft:
{json.dumps(prior_breakdown, ensure_ascii=False)}
"""


def _schema_repair_prompt_for_goal(goal: str, error: str) -> str:
  """Retry a response that could not be parsed into the required schema."""
  return f"""{_prompt_for_goal(goal)}

The previous response was rejected before a plan could be accepted:
{error}

Return a fresh JSON object with exactly the requested schema. Do not include a
Markdown fence or any prose outside the JSON object.
"""


def _cohort_identity(path_string: str) -> tuple[str | None, str | None]:
  if not path_string:
    return None, None
  path = Path(path_string).expanduser().resolve()
  cohort = catbench_primary_cohort.load(path)
  return str(cohort.get("release_id")), hashlib.sha256(path.read_bytes()).hexdigest()


def _attempt_audit_binding(
    args: argparse.Namespace,
    resolved_model: str,
    task_items: list[dict[str, Any]],
    openai_base_url: str,
) -> dict[str, Any]:
  """Build the immutable, credential-free identity for one generation run."""
  model_identity = _resolve_model_identity(args, resolved_model)
  cohort_release_id, cohort_manifest_sha256 = _cohort_identity(
      str(getattr(args, "cohort_manifest", ""))
  )
  seed_config = {
      "n_task_combinations": int(args.n_task_combinations),
      "task_random_seed": int(args.task_random_seed),
      "fixed_task_seed": bool(args.fixed_task_seed),
  }
  task_set = [
      {
          "key": item["key"],
          "plan_key": item["plan_key"],
          "semantic_task_id": item["semantic_task_id"],
          "instance_id": item["instance_id"],
          "semantic_goal_sha256": item["semantic_goal_sha256"],
          "semantic_parameter_sha256": item["semantic_parameter_sha256"],
          "category": item.get("category"),
          "app_id": item.get("app_id"),
      }
      for item in task_items
  ]
  config = {
      "provider": args.provider,
      "model": resolved_model,
      "model_identity": model_identity,
      "prompt_sha256": _prompt_sha256(),
      "generator_source_sha256": hashlib.sha256(
          Path(__file__).resolve().read_bytes()
      ).hexdigest(),
      "cohort_release_id": cohort_release_id,
      "cohort_manifest_sha256": cohort_manifest_sha256,
      "suite_family": args.suite_family,
      "categories": list(_parse_csv(args.categories)),
      "tasks": list(_parse_csv(args.tasks)),
      "seed_config": seed_config,
      "temperature": float(args.temperature),
      "max_retry": int(args.max_retry),
      "timeout_sec": float(args.timeout_sec),
      "sleep_seconds": float(args.sleep_seconds),
      "validation_retry": int(args.validation_retry),
      "strict_forbidden_check": bool(args.strict_forbidden_check),
      "plan_reuse_policy": "one_plan_per_semantic_instance_across_apps",
      "provider_endpoint_sha256": hashlib.sha256(
          (
              _safe_endpoint_identity(openai_base_url)
              if _is_openai_compatible_provider(args.provider)
              else "google-gemini-wrapper-default"
          ).encode("utf-8")
      ).hexdigest(),
      "provider_request_config": (
          {
              "temperature": float(args.temperature),
              "response_format": {"type": "json_object"},
          }
          if _is_openai_compatible_provider(args.provider)
          else {
              "temperature": float(args.temperature),
              "top_p": 0.95,
              "max_output_tokens": _GEMINI_MAX_OUTPUT_TOKENS,
              "response_mime_type": "application/json",
              "safety_checks": True,
          }
      ),
      "task_set_sha256": _json_sha256(task_set),
      "task_entry_count": len(task_items),
      "semantic_plan_count": len({item["plan_key"] for item in task_items}),
  }
  return {
      "provider": args.provider,
      "model": resolved_model,
      "model_identity": model_identity,
      "prompt_sha256": config["prompt_sha256"],
      "cohort_release_id": cohort_release_id,
      "cohort_manifest_sha256": cohort_manifest_sha256,
      "seed_config_sha256": _json_sha256(seed_config),
      "task_set_sha256": config["task_set_sha256"],
      "generator_config_sha256": _json_sha256(config),
      "generator_config": config,
  }


def _record_sha256(record: dict[str, Any]) -> str:
  unhashed = dict(record)
  unhashed.pop("record_sha256", None)
  return _json_sha256(unhashed)


def _read_audit_records(path: Path) -> list[dict[str, Any]]:
  try:
    file_stat = path.lstat()
  except FileNotFoundError:
    raise FileNotFoundError(f"Attempt audit does not exist: {path}") from None
  if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
    raise ValueError(f"Attempt audit must be a regular non-symlink file: {path}")
  data = path.read_bytes()
  if not data or not data.endswith(b"\n"):
    raise ValueError(f"Attempt audit is empty or lacks a final newline: {path}")
  records: list[dict[str, Any]] = []
  for line_number, raw_line in enumerate(data.splitlines(), 1):
    try:
      record = _strict_json_loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
      raise ValueError(
          f"Invalid attempt-audit JSON at line {line_number}: {exc}"
      ) from exc
    if not isinstance(record, dict):
      raise ValueError(
          f"Attempt-audit line {line_number} is not a JSON object."
      )
    records.append(record)
  return records


def _validate_audit_records(
    records: list[dict[str, Any]],
    expected_binding: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
  """Verify the whole chain and return accepted plans plus attempt counts."""
  if not records:
    raise ValueError("Attempt audit contains no header.")
  previous = _AUDIT_GENESIS_SHA256
  accepted_by_plan: dict[str, dict[str, Any]] = {}
  attempts_by_plan: dict[str, int] = {}
  closed_plans: set[str] = set()
  terminal_rejections: set[str] = set()
  last_plan_key: str | None = None
  pending_request: dict[str, Any] | None = None
  config_sha = expected_binding["generator_config_sha256"]
  for sequence, record in enumerate(records):
    if record.get("schema_version") != ATTEMPT_AUDIT_SCHEMA_VERSION:
      raise ValueError(f"Unsupported attempt-audit schema at line {sequence + 1}.")
    if record.get("record_sequence") != sequence:
      raise ValueError(f"Attempt-audit sequence mismatch at line {sequence + 1}.")
    if record.get("previous_record_sha256") != previous:
      raise ValueError(f"Attempt-audit chain break at line {sequence + 1}.")
    actual_hash = _record_sha256(record)
    if record.get("record_sha256") != actual_hash:
      raise ValueError(f"Attempt-audit hash mismatch at line {sequence + 1}.")
    previous = actual_hash

    if sequence == 0:
      if record.get("record_type") != "attempt_audit_header":
        raise ValueError("First attempt-audit record is not a header.")
      if record.get("binding") != expected_binding:
        raise ValueError(
            "Attempt-audit binding does not match provider/model/prompt/"
            "cohort/seed/config identity. Start a new audit and output file."
        )
      continue

    if record.get("record_type") != "external_request_attempt":
      raise ValueError(f"Unknown attempt-audit record type at line {sequence + 1}.")
    if record.get("generator_config_sha256") != config_sha:
      raise ValueError(f"Attempt record config mismatch at line {sequence + 1}.")
    identity = record.get("plan_identity")
    request = record.get("request")
    phase = record.get("attempt_phase")
    if not isinstance(identity, dict) or not isinstance(request, dict):
      raise ValueError(f"Malformed attempt record at line {sequence + 1}.")
    plan_key = str(identity.get("plan_key", ""))
    if not plan_key:
      raise ValueError(f"Missing plan key at line {sequence + 1}.")
    if phase == "started":
      if pending_request is not None:
        raise ValueError(
            f"A new request starts before the prior outcome at line {sequence + 1}."
        )
      if "outcome" in record:
        raise ValueError(f"Started request has an outcome at line {sequence + 1}.")
      if plan_key in closed_plans or plan_key in accepted_by_plan:
        raise ValueError(f"Attempt appears after accepted plan {plan_key}.")
      if plan_key in terminal_rejections:
        raise ValueError(f"Attempt appears after terminal rejection {plan_key}.")
      if last_plan_key is not None and plan_key != last_plan_key:
        if last_plan_key not in accepted_by_plan:
          raise ValueError(
              f"Attempt audit leaves {last_plan_key} unresolved before {plan_key}."
          )
        closed_plans.add(last_plan_key)
      last_plan_key = plan_key
      expected_ordinal = attempts_by_plan.get(plan_key, 0) + 1
      if request.get("request_ordinal") != expected_ordinal:
        raise ValueError(
            f"Request ordinal mismatch for {plan_key}: expected "
            f"{expected_ordinal}, got {request.get('request_ordinal')}."
        )
      attempts_by_plan[plan_key] = expected_ordinal
      pending_request = record
      continue
    if phase != "completed":
      raise ValueError(f"Unknown attempt phase at line {sequence + 1}: {phase}")
    if pending_request is None:
      raise ValueError(
          f"Request outcome has no preceding start at line {sequence + 1}."
      )
    if (
        pending_request.get("plan_identity") != identity
        or pending_request.get("request") != request
    ):
      raise ValueError(
          f"Request outcome does not match its start at line {sequence + 1}."
      )
    pending_request = None
    outcome = record.get("outcome")
    if not isinstance(outcome, dict):
      raise ValueError(f"Completed attempt lacks outcome at line {sequence + 1}.")
    status = outcome.get("status")
    if status not in {
        "request_error",
        "parse_rejection",
        "schema_rejection",
        "validation_rejection",
        "accepted",
    }:
      raise ValueError(f"Unknown attempt outcome at line {sequence + 1}: {status}")
    if status != "accepted":
      if not outcome.get("retry_reason") or not outcome.get("validation_rejection"):
        raise ValueError(
            f"Rejected attempt lacks retry reason/evidence at line {sequence + 1}."
        )
      if not isinstance(outcome.get("will_retry"), bool):
        raise ValueError(
            f"Rejected attempt lacks boolean will_retry at line {sequence + 1}."
        )
      if not outcome["will_retry"]:
        terminal_rejections.add(plan_key)
      continue
    if outcome.get("will_retry") or outcome.get("retry_reason"):
      raise ValueError(f"Accepted attempt incorrectly requests retry at line {sequence + 1}.")
    plan = outcome.get("final_accepted_plan")
    if not isinstance(plan, dict):
      raise ValueError(f"Accepted attempt lacks its frozen plan at line {sequence + 1}.")
    plan_text = task_breakdowns.format_breakdown_text({"breakdown": plan})
    if outcome.get("final_accepted_plan_sha256") != hashlib.sha256(
        plan_text.encode("utf-8")
    ).hexdigest():
      raise ValueError(f"Accepted plan hash mismatch at line {sequence + 1}.")
    if plan_key in accepted_by_plan:
      raise ValueError(f"Multiple accepted plans for semantic instance {plan_key}.")
    accepted_by_plan[plan_key] = record
  return accepted_by_plan, attempts_by_plan


class AttemptAuditLog:
  """Append-only, hash-linked record of every external planner request."""

  def __init__(
      self,
      path: Path,
      binding: dict[str, Any],
      secrets: tuple[str, ...],
      records: list[dict[str, Any]],
  ):
    self.path = path
    self.binding = binding
    self.secrets = tuple(secret for secret in secrets if secret)
    self.records = records
    self.accepted_by_plan, self.attempts_by_plan = _validate_audit_records(
        records, binding
    )
    file_stat = path.stat()
    self._device_inode = (file_stat.st_dev, file_stat.st_ino)
    flags = os.O_RDWR | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
      flags |= os.O_NOFOLLOW
    self._session_fd = os.open(path, flags)
    try:
      current_stat = os.fstat(self._session_fd)
      if (current_stat.st_dev, current_stat.st_ino) != self._device_inode:
        raise RuntimeError("Attempt audit was replaced while being opened.")
      fcntl.flock(self._session_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      os.close(self._session_fd)
      self._session_fd = -1
      raise RuntimeError(
          "Attempt audit is already locked by another generator process; "
          "concurrent provider calls cannot be audited safely."
      ) from None
    except BaseException:
      os.close(self._session_fd)
      self._session_fd = -1
      raise

  @classmethod
  def create(
      cls,
      path: Path,
      binding: dict[str, Any],
      secrets: tuple[str, ...] = (),
  ) -> "AttemptAuditLog":
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "schema_version": ATTEMPT_AUDIT_SCHEMA_VERSION,
        "record_type": "attempt_audit_header",
        "record_sequence": 0,
        "previous_record_sha256": _AUDIT_GENESIS_SHA256,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "binding": binding,
        "security_policy": (
            "generated content and sanitized errors only; credentials, "
            "authorization headers, and secret query parameters excluded"
        ),
    }
    header["record_sha256"] = _record_sha256(header)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
      flags |= os.O_NOFOLLOW
    try:
      fd = os.open(path, flags, 0o600)
    except FileExistsError:
      raise FileExistsError(
          f"Attempt audit already exists and is immutable: {path}. "
          "Use --resume only with its exactly bound output; otherwise choose "
          "a new, absent audit path."
      ) from None
    try:
      with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(_canonical_json(header) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    except BaseException:
      raise
    return cls(path, binding, secrets, [header])

  @classmethod
  def resume(
      cls,
      path: Path,
      binding: dict[str, Any],
      secrets: tuple[str, ...] = (),
  ) -> "AttemptAuditLog":
    records = _read_audit_records(path)
    instance = cls(path, binding, secrets, records)
    unresolved = set(instance.attempts_by_plan) - set(instance.accepted_by_plan)
    if unresolved:
      instance.close()
      raise RuntimeError(
          "Attempt audit ends with an unresolved semantic instance; refusing "
          "to guess the prior retry state. Start a new output/audit pair. "
          f"Unresolved: {sorted(unresolved)}"
      )
    return instance

  @property
  def tail_sha256(self) -> str:
    return str(self.records[-1]["record_sha256"])

  @property
  def header_sha256(self) -> str:
    return str(self.records[0]["record_sha256"])

  def close(self) -> None:
    if getattr(self, "_session_fd", -1) >= 0:
      fcntl.flock(self._session_fd, fcntl.LOCK_UN)
      os.close(self._session_fd)
      self._session_fd = -1

  def __del__(self) -> None:
    try:
      self.close()
    except (OSError, AttributeError):
      pass

  def append_attempt(self, event: dict[str, Any]) -> dict[str, Any]:
    sanitized_event = _sanitize_audit_value(event, self.secrets)
    record = {
        "schema_version": ATTEMPT_AUDIT_SCHEMA_VERSION,
        "record_type": "external_request_attempt",
        "record_sequence": len(self.records),
        "previous_record_sha256": self.tail_sha256,
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
        "generator_config_sha256": self.binding["generator_config_sha256"],
        **sanitized_event,
    }
    record["record_sha256"] = _record_sha256(record)
    _validate_audit_records(self.records + [record], self.binding)

    if self._session_fd < 0:
      raise RuntimeError("Attempt audit is closed.")
    fd = os.dup(self._session_fd)
    try:
      with os.fdopen(fd, "r+", encoding="utf-8") as handle:
        current_stat = os.fstat(handle.fileno())
        if (current_stat.st_dev, current_stat.st_ino) != self._device_inode:
          raise RuntimeError("Attempt audit was replaced during generation.")
        handle.seek(0)
        disk_text = handle.read()
        if not disk_text.endswith("\n"):
          raise RuntimeError("Attempt audit tail is truncated; refusing append.")
        disk_records = [
            _strict_json_loads(line) for line in disk_text.splitlines()
        ]
        _validate_audit_records(disk_records, self.binding)
        if (
            len(disk_records) != len(self.records)
            or disk_records[-1].get("record_sha256") != self.tail_sha256
        ):
          raise RuntimeError(
              "Attempt audit changed concurrently; refusing a non-contiguous append."
          )
        handle.seek(0, os.SEEK_END)
        handle.write(_canonical_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    except BaseException:
      raise
    self.records.append(record)
    self.accepted_by_plan, self.attempts_by_plan = _validate_audit_records(
        self.records, self.binding
    )
    return record


def _extract_json(text: str) -> dict[str, Any]:
  cleaned = text.strip()
  if cleaned.startswith("```"):
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
  try:
    payload = json.loads(cleaned)
  except json.JSONDecodeError:
    start = cleaned.find("{")
    if start < 0:
      raise
    # Tolerate trailing junk after the first complete object (a common JSON
    # mode failure is one extra closing brace); truncated objects still fail.
    payload, _ = json.JSONDecoder().raw_decode(cleaned, start)
  if not isinstance(payload, dict):
    raise ValueError("Gemini output must be a JSON object.")
  return payload


def _normalize_breakdown(payload: dict[str, Any]) -> dict[str, Any]:
  steps = payload.get("steps")
  if not isinstance(steps, list):
    raise ValueError("Gemini output JSON must include a list field named steps.")
  cleaned_steps = [str(step).strip() for step in steps if str(step).strip()]
  if not cleaned_steps:
    raise ValueError("Gemini output did not include any non-empty steps.")

  notes = payload.get("notes", [])
  if isinstance(notes, str):
    notes = [notes]
  if not isinstance(notes, list):
    notes = []
  cleaned_notes = [str(note).strip() for note in notes if str(note).strip()]
  return {"steps": cleaned_steps, "notes": cleaned_notes}


def _breakdown_scan_text(breakdown: dict[str, Any]) -> str:
  """Return actual plan strings, preserving whitespace for lexical scans."""
  strings: list[str] = []

  def _collect(value: Any) -> None:
    if isinstance(value, str):
      strings.append(value)
    elif isinstance(value, dict):
      for item in value.values():
        _collect(item)
    elif isinstance(value, (list, tuple)):
      for item in value:
        _collect(item)

  _collect(breakdown)
  return "\n".join(strings)


def _geographic_coordinate_keys(text: str) -> set[tuple[str, str]]:
  """Extract range-valid decimal latitude/longitude pairs from a goal."""
  keys: set[tuple[str, str]] = set()
  for match in _COORDINATE_PAIR_PATTERN.finditer(text):
    latitude_text = match.group("latitude")
    longitude_text = match.group("longitude")
    if "." not in latitude_text and "." not in longitude_text:
      continue
    try:
      latitude = Decimal(latitude_text)
      longitude = Decimal(longitude_text)
    except InvalidOperation:
      continue
    if Decimal("-90") <= latitude <= Decimal("90") and (
        Decimal("-180") <= longitude <= Decimal("180")
    ):
      keys.add((latitude_text, longitude_text))
  return keys


def _has_ui_coordinate_context(text: str, coordinate_start: int) -> bool:
  prefix = text[max(0, coordinate_start - 100):coordinate_start]
  line_start = text.rfind("\n", 0, coordinate_start) + 1
  line_end = text.find("\n", coordinate_start)
  if line_end < 0:
    line_end = len(text)
  line = text[line_start:line_end]
  return bool(
      _UI_COORDINATE_PREFIX_PATTERN.search(prefix)
      or _UI_COORDINATE_REFERENCE_PATTERN.search(line)
  )


def _is_geographic_coordinate_reference(
    text: str, coordinate_match: re.Match[str]
) -> bool:
  line_start = text.rfind("\n", 0, coordinate_match.start()) + 1
  line_end = text.find("\n", coordinate_match.end())
  if line_end < 0:
    line_end = len(text)
  line = text[line_start:line_end]
  return (
      not _UI_COORDINATE_REFERENCE_PATTERN.search(line)
      and bool(_GEOGRAPHIC_COORDINATE_REFERENCE_PATTERN.search(line))
  )


def _coordinate_scan_text(text: str, semantic_goal: str) -> str:
  """Mask only goal-derived geographic coordinate occurrences and labels."""
  goal_coordinates = _geographic_coordinate_keys(semantic_goal)
  if not goal_coordinates:
    return text

  masked_spans: list[tuple[int, int]] = []
  preserved_goal_coordinate = False
  for match in _COORDINATE_PAIR_PATTERN.finditer(text):
    key = (match.group("latitude"), match.group("longitude"))
    if key not in goal_coordinates or _has_ui_coordinate_context(
        text, match.start()
    ):
      continue
    preserved_goal_coordinate = True
    masked_spans.append(match.span())

    before = _COORDINATE_LABEL_BEFORE_PATTERN.search(text[:match.start()])
    if before is not None:
      masked_spans.append(before.span("label"))
    after = _COORDINATE_LABEL_AFTER_PATTERN.search(text[match.end():])
    if after is not None:
      start, end = after.span("label")
      masked_spans.append((match.end() + start, match.end() + end))

  if preserved_goal_coordinate:
    for coordinate_match in FORBIDDEN_PATTERNS["coordinate_word"].finditer(
        text
    ):
      if _is_geographic_coordinate_reference(text, coordinate_match):
        masked_spans.append(coordinate_match.span())

  if not masked_spans:
    return text
  masked = list(text)
  for start, end in masked_spans:
    masked[start:end] = " " * (end - start)
  return "".join(masked)


def _forbidden_warnings(
    breakdown: dict[str, Any],
    app_display_names: tuple[str, ...] = (),
    semantic_goal: str = "",
) -> list[str]:
  text = _breakdown_scan_text(breakdown)
  coordinate_scan_text = _coordinate_scan_text(text, semantic_goal)
  warnings = []
  for name, pattern in FORBIDDEN_PATTERNS.items():
    scan_text = (
        coordinate_scan_text
        if name in {"coordinate_pair", "coordinate_word"}
        else text
    )
    if pattern.search(scan_text):
      warnings.append(name)
  for app_name in app_display_names:
    normalized = app_name.strip()
    if normalized.casefold() in _GENERIC_APP_DISPLAY_NAMES:
      continue
    if re.search(rf"\b{re.escape(normalized)}\b", text, re.IGNORECASE):
      if "app_name_mention" not in warnings:
        warnings.append("app_name_mention")
      break
  return warnings


def _load_existing(path: Path, resume: bool, overwrite: bool) -> dict[str, Any]:
  if not path.exists() or overwrite:
    return {"metadata": {}, "breakdowns": []}
  if not resume:
    raise FileExistsError(
        f"{path} already exists. Use --resume to append or --overwrite."
    )
  with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  if not isinstance(payload, dict):
    raise ValueError(f"Existing output is not a JSON object: {path}")
  payload.setdefault("metadata", {})
  payload.setdefault("breakdowns", [])
  return payload


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = path.with_suffix(path.suffix + ".tmp")
  with tmp_path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
  tmp_path.replace(path)


def _existing_indexes(
    payload: dict[str, Any],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
  existing_keys: set[str] = set()
  by_plan_key: dict[str, dict[str, Any]] = {}
  for entry in payload.get("breakdowns", []):
    if not isinstance(entry, dict):
      continue
    key = entry.get("key")
    if key:
      normalized_key = str(key)
      if normalized_key in existing_keys:
        raise RuntimeError(
            f"Duplicate exact-instance breakdown key: {normalized_key}"
        )
      existing_keys.add(normalized_key)
    plan_key = entry.get("plan_key")
    if plan_key and entry.get("breakdown"):
      normalized_plan_key = str(plan_key)
      prior = by_plan_key.get(normalized_plan_key)
      if prior is not None:
        prior_text = task_breakdowns.format_breakdown_text(prior)
        current_text = task_breakdowns.format_breakdown_text(entry)
        if prior_text != current_text:
          raise RuntimeError(
              "Entries sharing a semantic plan key contain different plans: "
              f"{normalized_plan_key}"
          )
      else:
        by_plan_key[normalized_plan_key] = entry
  return existing_keys, by_plan_key


def _has_exact_instance_key(entry: dict[str, Any]) -> bool:
  """Whether a resumable entry carries the complete generated identity."""
  if (
      not entry.get("plan_key")
      or "instance_id" not in entry
      or not entry.get("task_template")
      or not entry.get("goal")
  ):
    return False
  try:
    expected = task_breakdowns.make_key(
        str(entry["task_template"]),
        str(entry["goal"]),
        int(entry["instance_id"]),
    )
  except (TypeError, ValueError, OverflowError):
    return False
  return entry.get("key") == expected


def _build_entry(
    task_item: dict[str, Any],
    breakdown: dict[str, Any],
    model: str,
    warnings: list[str],
    model_identity: str | None = None,
) -> dict[str, Any]:
  breakdown_text = task_breakdowns.format_breakdown_text(
      {"breakdown": breakdown}
  )
  return {
      "key": task_item["key"],
      "task_template": task_item["task_template"],
      "instance_id": task_item["instance_id"],
      "goal": task_item["goal"],
      "goal_sha256": task_item["goal_sha256"],
      "semantic_task_id": task_item["semantic_task_id"],
      "app_display_name": task_item["app_display_name"],
      "semantic_goal": task_item["semantic_goal"],
      "semantic_goal_sha256": task_item["semantic_goal_sha256"],
      "semantic_parameter_sha256": task_item["semantic_parameter_sha256"],
      "plan_key": task_item["plan_key"],
      "plan_sha256": hashlib.sha256(
          breakdown_text.encode("utf-8")
      ).hexdigest(),
      "generator_model": model,
      "generator_model_identity": model_identity or model,
      "breakdown": breakdown,
      "breakdown_text": breakdown_text,
      "validation_warnings": warnings,
  }


def _chat_url(base_url: str) -> str:
  base = base_url.rstrip("/")
  if base.endswith("/chat/completions"):
    return base
  if base.endswith("/v1"):
    return f"{base}/chat/completions"
  return f"{base}/v1/chat/completions"


def _call_openai_once(
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout_sec: float,
    temperature: float,
) -> str:
  if not api_key:
    raise RuntimeError(
        "OpenAI-compatible provider requires an API key."
    )
  payload = {
      "model": model,
      "messages": [{"role": "user", "content": prompt}],
      "temperature": temperature,
      "response_format": {"type": "json_object"},
  }
  data = json.dumps(payload).encode("utf-8")
  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}",
  }
  req = urllib.request.Request(
      _chat_url(base_url), data=data, headers=headers, method="POST"
  )
  with urllib.request.urlopen(req, timeout=timeout_sec) as response:
    body = response.read().decode("utf-8", errors="replace")
  reply = json.loads(body)
  choices = reply.get("choices") or []
  if not choices:
    raise ValueError(f"OpenAI response has no choices: {body[:500]}")
  content = (choices[0].get("message") or {}).get("content")
  if isinstance(content, list):
    content = "".join(
        item.get("text", "") if isinstance(item, dict) else str(item)
        for item in content
    )
  if not isinstance(content, str) or not content.strip():
    raise ValueError(f"OpenAI response empty content: {body[:500]}")
  return content


def _call_provider_once(
    *,
    args: argparse.Namespace,
    prompt: str,
    resolved_model: str,
    gemini_wrapper: Any,
    openai_base_url: str,
    openai_api_key: str,
) -> str:
  """Make exactly one externally visible attempt; retries live above this."""
  if _is_openai_compatible_provider(args.provider):
    return _call_openai_once(
        prompt=prompt,
        model=resolved_model,
        base_url=openai_base_url,
        api_key=openai_api_key,
        timeout_sec=args.timeout_sec,
        temperature=args.temperature,
    )
  # Call the SDK exactly once. GeminiGcpWrapper.generate() owns an internal
  # retry loop (and prints exceptions), which would make attempts unauditable.
  response = gemini_wrapper.client.models.generate_content(
      model=gemini_wrapper.model_name,
      contents=prompt,
      config=gemini_wrapper._build_generation_config(  # pylint: disable=protected-access
          generation_config={
              "response_mime_type": "application/json",
              "max_output_tokens": _GEMINI_MAX_OUTPUT_TOKENS,
          },
          enable_safety_checks=True,
      ),
  )
  text = getattr(response, "text", None)
  if not isinstance(text, str) or not text.strip():
    raise ValueError("Gemini response did not contain non-empty text.")
  return text


def _plan_identity(item: dict[str, Any]) -> dict[str, Any]:
  return {
      "plan_key": item["plan_key"],
      "semantic_task_id": item["semantic_task_id"],
      "instance_id": item["instance_id"],
      "semantic_goal": item["semantic_goal"],
      "semantic_goal_sha256": item["semantic_goal_sha256"],
      "semantic_parameter_sha256": item["semantic_parameter_sha256"],
  }


def _safe_request_payload(
    args: argparse.Namespace,
    resolved_model: str,
    prompt: str,
    prompt_kind: str,
    validation_attempt: int,
    transport_attempt: int,
    request_ordinal: int,
) -> dict[str, Any]:
  return {
      "provider": args.provider,
      "model": resolved_model,
      "model_identity": _resolve_model_identity(args, resolved_model),
      "prompt": prompt,
      "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
      "prompt_kind": prompt_kind,
      "temperature": args.temperature,
      "response_format": "json_object",
      "validation_attempt": validation_attempt,
      "transport_attempt": transport_attempt,
      "request_ordinal": request_ordinal,
  }


def _append_attempt(
    audit: AttemptAuditLog | None,
    *,
    item: dict[str, Any],
    request: dict[str, Any],
    status: str,
    will_retry: bool,
    retry_reason: str | None,
    raw_response: str | None,
    parsed_candidate: dict[str, Any] | None,
    normalized_candidate: dict[str, Any] | None,
    validation_rejection: dict[str, Any] | None,
    error: BaseException | None,
    final_accepted_plan: dict[str, Any] | None,
) -> None:
  if audit is None:
    return
  safe_error = None
  if error is not None:
    safe_error = {
        "type": type(error).__name__,
        "message": str(error),
    }
  final_plan_sha = None
  if final_accepted_plan is not None:
    plan_text = task_breakdowns.format_breakdown_text(
        {"breakdown": final_accepted_plan}
    )
    final_plan_sha = hashlib.sha256(plan_text.encode("utf-8")).hexdigest()
  audit.append_attempt(
      {
          "attempt_phase": "completed",
          "plan_identity": _plan_identity(item),
          "request": request,
          "outcome": {
              "status": status,
              "will_retry": will_retry,
              "retry_reason": retry_reason,
              "raw_response": raw_response,
              "parsed_candidate": parsed_candidate,
              "normalized_candidate": normalized_candidate,
              "validation_rejection": validation_rejection,
              "error": safe_error,
              "final_accepted_plan": final_accepted_plan,
              "final_accepted_plan_sha256": final_plan_sha,
          },
      }
  )


def _append_attempt_started(
    audit: AttemptAuditLog | None,
    *,
    item: dict[str, Any],
    request: dict[str, Any],
) -> None:
  """Durably record intent before making the corresponding external call."""
  if audit is None:
    return
  audit.append_attempt(
      {
          "attempt_phase": "started",
          "plan_identity": _plan_identity(item),
          "request": request,
      }
  )


def _generate_one_plan(
    *,
    args: argparse.Namespace,
    item: dict[str, Any],
    app_display_names: tuple[str, ...],
    resolved_model: str,
    gemini_wrapper: Any,
    openai_base_url: str,
    openai_api_key: str,
    audit: AttemptAuditLog | None,
) -> tuple[dict[str, Any], list[str], int]:
  """Generate and audit one accepted plan for one semantic instance."""
  prompt = _prompt_for_goal(item["semantic_goal"])
  prompt_kind = "initial"
  request_ordinal = 0
  repair_attempts = 0
  for validation_attempt in range(args.validation_retry + 1):
    text: str | None = None
    for transport_attempt in range(1, args.max_retry + 1):
      request_ordinal += 1
      request = _safe_request_payload(
          args,
          resolved_model,
          prompt,
          prompt_kind,
          validation_attempt,
          transport_attempt,
          request_ordinal,
      )
      _append_attempt_started(audit, item=item, request=request)
      try:
        text = _call_provider_once(
            args=args,
            prompt=prompt,
            resolved_model=resolved_model,
            gemini_wrapper=gemini_wrapper,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
        )
      except Exception as exc:  # provider SDKs expose heterogeneous errors
        will_retry = transport_attempt < args.max_retry
        _append_attempt(
            audit,
            item=item,
            request=request,
            status="request_error",
            will_retry=will_retry,
            retry_reason=(
                "provider_call_error_retry" if will_retry else "provider_call_error_exhausted"
            ),
            raw_response=None,
            parsed_candidate=None,
            normalized_candidate=None,
            validation_rejection={
                "kind": "provider_call_error",
                "detail": type(exc).__name__,
            },
            error=exc,
            final_accepted_plan=None,
        )
        if will_retry:
          time.sleep(min(2**transport_attempt, 10))
          continue
        safe_message, _ = _redact_text(
            f"{type(exc).__name__}: {exc}",
            (
                openai_api_key,
                os.environ.get("GEMINI_API_KEY", ""),
                os.environ.get("GCP_API_KEY", ""),
            ),
        )
        raise RuntimeError(
            f"{args.provider} call failed after {args.max_retry} attempts for "
            f"{item['plan_key']}: {safe_message}"
        ) from None
      break
    if text is None:
      raise AssertionError("Provider retry loop ended without text or error.")

    request = _safe_request_payload(
        args,
        resolved_model,
        prompt,
        prompt_kind,
        validation_attempt,
        transport_attempt,
        request_ordinal,
    )
    try:
      parsed_candidate = _extract_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
      will_retry = validation_attempt < args.validation_retry
      _append_attempt(
          audit,
          item=item,
          request=request,
          status="parse_rejection",
          will_retry=will_retry,
          retry_reason=("invalid_json_retry" if will_retry else "invalid_json_exhausted"),
          raw_response=text,
          parsed_candidate=None,
          normalized_candidate=None,
          validation_rejection={"kind": "invalid_json", "detail": str(exc)},
          error=exc,
          final_accepted_plan=None,
      )
      if not will_retry:
        raise ValueError(
            f"Planner returned invalid JSON for {item['plan_key']}: {exc}"
        ) from exc
      repair_attempts += 1
      prompt = _schema_repair_prompt_for_goal(
          item["semantic_goal"], f"invalid JSON: {type(exc).__name__}"
      )
      prompt_kind = "schema_repair"
      continue

    try:
      breakdown = _normalize_breakdown(parsed_candidate)
    except ValueError as exc:
      will_retry = validation_attempt < args.validation_retry
      _append_attempt(
          audit,
          item=item,
          request=request,
          status="schema_rejection",
          will_retry=will_retry,
          retry_reason=("invalid_schema_retry" if will_retry else "invalid_schema_exhausted"),
          raw_response=text,
          parsed_candidate=parsed_candidate,
          normalized_candidate=None,
          validation_rejection={"kind": "invalid_schema", "detail": str(exc)},
          error=exc,
          final_accepted_plan=None,
      )
      if not will_retry:
        raise ValueError(
            f"Planner returned invalid schema for {item['plan_key']}: {exc}"
        ) from exc
      repair_attempts += 1
      prompt = _schema_repair_prompt_for_goal(
          item["semantic_goal"], f"invalid schema: {str(exc)}"
      )
      prompt_kind = "schema_repair"
      continue

    warnings = _forbidden_warnings(
        breakdown,
        app_display_names,
        semantic_goal=item["semantic_goal"],
    )
    if args.strict_forbidden_check and warnings:
      will_retry = validation_attempt < args.validation_retry
      _append_attempt(
          audit,
          item=item,
          request=request,
          status="validation_rejection",
          will_retry=will_retry,
          retry_reason=(
              "forbidden_detail_retry" if will_retry else "forbidden_detail_exhausted"
          ),
          raw_response=text,
          parsed_candidate=parsed_candidate,
          normalized_candidate=breakdown,
          validation_rejection={"kind": "forbidden_detail", "warnings": warnings},
          error=None,
          final_accepted_plan=None,
      )
      if not will_retry:
        raise ValueError(
            f"Forbidden detail warning(s) for {item['key']}: {warnings}. "
            "Start a new audited generation after correcting the protocol; "
            "the rejected attempt remains immutable."
        )
      repair_attempts += 1
      prompt = _repair_prompt_for_goal(
          item["semantic_goal"], breakdown, warnings
      )
      prompt_kind = "forbidden_detail_repair"
      continue

    _append_attempt(
        audit,
        item=item,
        request=request,
        status="accepted",
        will_retry=False,
        retry_reason=None,
        raw_response=text,
        parsed_candidate=parsed_candidate,
        normalized_candidate=breakdown,
        validation_rejection=None,
        error=None,
        final_accepted_plan=breakdown,
    )
    return breakdown, warnings, repair_attempts
  raise AssertionError("Validation loop ended without accepting or rejecting a plan.")


def _resolve_model(args: argparse.Namespace) -> str:
  if args.model:
    return args.model
  if args.provider == "openai":
    return DEFAULT_OPENAI_MODEL
  if args.provider == "qwen":
    return DEFAULT_QWEN_MODEL
  return DEFAULT_GEMINI_MODEL


def _resolve_model_identity(
    args: argparse.Namespace, resolved_model: str
) -> str:
  identity = str(getattr(args, "model_identity", "") or "").strip()
  return identity or resolved_model


def _is_openai_compatible_provider(provider: str) -> bool:
  return provider in {"openai", "qwen"}


def _sync_output_audit_metadata(
    payload: dict[str, Any], audit: AttemptAuditLog | None
) -> None:
  if audit is None:
    return
  payload.setdefault("metadata", {})["attempt_audit"] = {
      "schema_version": ATTEMPT_AUDIT_SCHEMA_VERSION,
      "path": str(audit.path.resolve()),
      "header_sha256": audit.header_sha256,
      "tail_sha256": audit.tail_sha256,
      "record_count": len(audit.records),
      "generator_config_sha256": audit.binding["generator_config_sha256"],
      "security_policy": "credentials_and_authorization_headers_excluded",
  }


def _verify_output_audit_continuity(
    *,
    payload: dict[str, Any],
    output_preexisted: bool,
    task_items: list[dict[str, Any]],
    audit: AttemptAuditLog,
) -> None:
  """Reject legacy/mixed files and bind every materialized plan to acceptance."""
  allowed_identity: dict[str, dict[str, Any]] = {}
  for item in task_items:
    identity = _plan_identity(item)
    prior = allowed_identity.setdefault(item["plan_key"], identity)
    if prior != identity:
      raise RuntimeError(
          f"Semantic plan identity is not shared exactly: {item['plan_key']}"
      )
  for record in audit.records[1:]:
    identity = record["plan_identity"]
    plan_key = str(identity["plan_key"])
    if plan_key not in allowed_identity or identity != allowed_identity[plan_key]:
      raise RuntimeError(
          f"Attempt audit contains an out-of-cohort or altered identity: {plan_key}"
      )

  metadata = payload.get("metadata") or {}
  recorded_audit = metadata.get("attempt_audit")
  if output_preexisted:
    if not isinstance(recorded_audit, dict):
      raise RuntimeError(
          "Existing output has no attempt-audit binding. Refusing to retrofit "
          "a legacy plan file; start a new output/audit pair."
      )
    immutable_fields = {
        "schema_version": ATTEMPT_AUDIT_SCHEMA_VERSION,
        "path": str(audit.path.resolve()),
        "header_sha256": audit.header_sha256,
        "generator_config_sha256": audit.binding["generator_config_sha256"],
    }
    mismatches = {
        key: (recorded_audit.get(key), value)
        for key, value in immutable_fields.items()
        if recorded_audit.get(key) != value
    }
    if mismatches:
      raise RuntimeError(
          f"Existing output attempt-audit binding mismatch: {mismatches}"
      )
    recorded_count = recorded_audit.get("record_count")
    recorded_tail = recorded_audit.get("tail_sha256")
    if (
        not isinstance(recorded_count, int)
        or isinstance(recorded_count, bool)
        or recorded_count < 1
        or recorded_count > len(audit.records)
        or audit.records[recorded_count - 1].get("record_sha256")
        != recorded_tail
    ):
      raise RuntimeError(
          "Existing output audit tail is not a verified ancestor of the "
          "current append-only chain."
      )

  for entry in payload.get("breakdowns", []):
    if not isinstance(entry, dict):
      raise RuntimeError("Existing breakdown output contains a non-object entry.")
    plan_key = str(entry.get("plan_key", ""))
    accepted = audit.accepted_by_plan.get(plan_key)
    if accepted is None:
      raise RuntimeError(
          "Existing output contains a plan with no accepted audit attempt: "
          f"{plan_key or '<missing plan key>'}."
      )
    accepted_plan = accepted["outcome"]["final_accepted_plan"]
    if entry.get("breakdown") != accepted_plan:
      raise RuntimeError(
          f"Output plan differs from immutable accepted audit plan: {plan_key}"
      )
    accepted_sha = accepted["outcome"]["final_accepted_plan_sha256"]
    if entry.get("plan_sha256") != accepted_sha:
      raise RuntimeError(
          f"Output plan hash differs from immutable attempt audit: {plan_key}"
      )


def generate(args: argparse.Namespace) -> None:
  if args.resume and args.overwrite:
    raise ValueError("--resume and --overwrite are mutually exclusive.")
  if args.cohort_manifest and args.overwrite:
    raise ValueError(
        "Frozen-cohort generation forbids --overwrite. Use new absent output "
        "and audit paths, or resume the exactly bound pair."
    )
  if args.max_retry < 1:
    raise ValueError("--max_retry must be at least 1.")
  if args.validation_retry < 0:
    raise ValueError("--validation_retry cannot be negative.")
  if args.timeout_sec <= 0:
    raise ValueError("--timeout_sec must be positive.")
  if args.sleep_seconds < 0:
    raise ValueError("--sleep_seconds cannot be negative.")
  if not all(
      math.isfinite(float(value))
      for value in (args.temperature, args.timeout_sec, args.sleep_seconds)
  ):
    raise ValueError("Temperature and timing values must be finite.")
  if not args.dedupe_by_goal:
    raise ValueError(
        "Per-app C2 generation is not a valid CATBench diagnostic. Shared "
        "semantic-plan reuse is mandatory."
    )
  if args.cohort_manifest and not args.strict_forbidden_check:
    raise ValueError(
        "Frozen primary C2-G generation requires strict forbidden-detail "
        "validation; ablation outputs must use a separate non-primary file."
    )
  task_items = _iter_task_instances(args)
  app_display_names = tuple(sorted({
      str(item["app_display_name"])
      for item in task_items
      if item.get("app_display_name")
  }))
  if args.dry_run:
    print(
        f"Would write {len(task_items)} app entries from "
        f"{len({item['plan_key'] for item in task_items})} unique semantic "
        "plans."
    )
    for item in task_items[:20]:
      print(item["plan_key"], item["semantic_goal"])
    if len(task_items) > 20:
      print(f"... {len(task_items) - 20} more")
    return

  resolved_model = _resolve_model(args)
  model_identity = _resolve_model_identity(args, resolved_model)
  print(
      f"[generator] provider={args.provider} model={resolved_model} "
      f"strict={args.strict_forbidden_check} prompt_sha256={_prompt_sha256()[:12]}",
      flush=True,
  )

  if args.provider == "qwen":
    openai_api_key = (
        args.openai_api_key or os.environ.get("QWEN_C2_API_KEY", "")
    )
    openai_base_url = (
        args.openai_base_url or os.environ.get("QWEN_C2_BASE_URL", "")
    )
    if args.cohort_manifest and not str(
        getattr(args, "model_identity", "") or ""
    ).strip():
      raise ValueError(
          "Frozen-cohort Qwen generation requires an explicit nonempty "
          "--model_identity for the underlying served checkpoint."
      )
    if not openai_base_url:
      raise ValueError(
          "Qwen generation requires an explicit endpoint via "
          "--openai_base_url or QWEN_C2_BASE_URL."
      )
  else:
    openai_api_key = (
        args.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    )
    openai_base_url = (
        args.openai_base_url
        or os.environ.get("OPENAI_BASE_URL", "")
        or "https://api.openai.com/v1/chat/completions"
    )

  output_path = Path(args.output).expanduser().resolve()
  audit_path_string = str(getattr(args, "audit_log", "") or "")
  if args.cohort_manifest and not audit_path_string:
    raise ValueError(
        "Non-dry-run frozen-cohort generation requires --audit_log at a new, "
        "absent path. No external request may be made without it."
    )
  audit_path = (
      Path(audit_path_string).expanduser().resolve()
      if audit_path_string
      else None
  )
  if audit_path is not None and audit_path == output_path:
    raise ValueError("--audit_log and --output must be different files.")
  if audit_path is not None and (
      args.allow_prompt_mismatch or args.allow_provider_mismatch
  ):
    raise ValueError(
        "Audited generation forbids mismatch overrides. Start a new, absent "
        "output/audit pair for a changed prompt or provider."
    )

  output_preexisted = output_path.exists() and not args.overwrite
  payload = _load_existing(output_path, args.resume, args.overwrite)
  existing_entries = [
      entry
      for entry in payload.get("breakdowns", [])
      if isinstance(entry, dict)
  ]
  legacy_entries = [
      entry
      for entry in existing_entries
      if not _has_exact_instance_key(entry)
  ]
  if args.resume and legacy_entries:
    raise RuntimeError(
        "Existing breakdown entries predate exact semantic-instance keys. "
        "Use --overwrite and regenerate; mixing legacy per-app plans with the "
        "paired C2 protocol is invalid."
    )
  existing_keys, by_plan_key = _existing_indexes(payload)

  prompt_sha = _prompt_sha256()
  prior_prompt_sha = (payload.get("metadata") or {}).get("prompt_sha256")
  if (
      args.resume
      and prior_prompt_sha
      and prior_prompt_sha != prompt_sha
      and not args.allow_prompt_mismatch
  ):
    raise RuntimeError(
        f"Prompt template changed since last write ({prior_prompt_sha[:12]} "
        f"!= {prompt_sha[:12]}). Use --overwrite to start fresh or "
        "--allow_prompt_mismatch to keep mixed cohorts (not recommended)."
    )

  prior_provider = (payload.get("metadata") or {}).get("generator_provider")
  if (
      args.resume
      and prior_provider
      and prior_provider != args.provider
      and not args.allow_provider_mismatch
  ):
    raise RuntimeError(
        f"Provider changed since last write ({prior_provider} != "
        f"{args.provider}). Generate per-provider files separately; use "
        "--allow_provider_mismatch only if you know what you are doing."
    )

  gemini_wrapper = None
  if args.provider == "gemini":
    from android_world.agents import infer  # heavy; only loaded when needed

    # Provider-internal retries would hide request attempts. The outer loop
    # owns retries so every externally visible attempt enters the audit first.
    gemini_wrapper = infer.GeminiGcpWrapper(
        model_name=resolved_model,
        max_retry=1,
        temperature=args.temperature,
    )

  audit: AttemptAuditLog | None = None
  if audit_path is not None:
    binding = _attempt_audit_binding(
        args, resolved_model, task_items, openai_base_url
    )
    secrets = (
        openai_api_key,
        os.environ.get("GEMINI_API_KEY", ""),
        os.environ.get("GCP_API_KEY", ""),
    )
    if args.resume:
      if not audit_path.exists():
        raise FileNotFoundError(
            "--resume requires the original attempt audit; refusing to "
            f"retrofit or reconstruct it: {audit_path}"
        )
      audit = AttemptAuditLog.resume(audit_path, binding, secrets)
    else:
      audit = AttemptAuditLog.create(audit_path, binding, secrets)
    _verify_output_audit_continuity(
        payload=payload,
        output_preexisted=output_preexisted,
        task_items=task_items,
        audit=audit,
    )

  cohort_release_id, cohort_manifest_sha256 = _cohort_identity(
      str(args.cohort_manifest)
  )
  payload["metadata"] = {
      **payload.get("metadata", {}),
      "generator_provider": args.provider,
      "generator_model": resolved_model,
      "generator_model_identity": model_identity,
      "prompt_sha256": prompt_sha,
      "created_or_updated_at": dt.datetime.now(dt.UTC).isoformat(),
      "suite_family": args.suite_family,
      "categories": list(_parse_csv(args.categories)),
      "tasks": list(_parse_csv(args.tasks)),
      "n_task_combinations": args.n_task_combinations,
      "task_random_seed": args.task_random_seed,
      "fixed_task_seed": args.fixed_task_seed,
      "generation_policy": {
          "temperature": float(args.temperature),
          "max_retry": int(args.max_retry),
          "timeout_sec": float(args.timeout_sec),
          "sleep_seconds": float(args.sleep_seconds),
          "validation_retry": int(args.validation_retry),
          "strict_forbidden_check": bool(args.strict_forbidden_check),
          "response_contract": (
              "provider_json_mode_then_common_schema_and_forbidden_detail_"
              "validation"
          ),
          "selection_policy": (
              "first_accepted_machine_valid_plan_no_best_of_n"
          ),
      },
      "condition": "application_independent_breakdown_prepend",
      "semantic_pairing_version": 2,
      "plan_reuse_policy": "one_plan_per_semantic_instance_across_apps",
      "planner_input_app_identity": "replaced_with_[TARGET_APP]",
      "expected_entry_count": len(task_items),
      "expected_semantic_plan_count": len({
          item["plan_key"] for item in task_items
      }),
      "generator_observation_policy": (
          "text-only: no screen, UI layout, screen/pixel coordinates, "
          "accessibility nodes, or app-specific labels"
      ),
      "forbidden_patterns": sorted(FORBIDDEN_PATTERNS.keys()),
      "cohort_release_id": cohort_release_id,
      "cohort_manifest_sha256": cohort_manifest_sha256,
  }
  _sync_output_audit_metadata(payload, audit)
  # Establish the bound output before the first provider call. In overwrite
  # mode this also removes any superseded output content before generation.
  _write_payload(output_path, payload)

  generated_count = 0
  for index, item in enumerate(task_items, 1):
    if item["key"] in existing_keys:
      print(f"[{index}/{len(task_items)}] skip existing {item['key']}")
      continue

    repair_attempts = 0
    reused = by_plan_key.get(item["plan_key"])
    recovered = (
        audit.accepted_by_plan.get(item["plan_key"])
        if audit is not None
        else None
    )
    if reused:
      breakdown = reused["breakdown"]
      warnings = list(reused.get("validation_warnings", []))
      print(
          f"[{index}/{len(task_items)}] reuse semantic plan "
          f"{item['plan_key']} for {item['key']}"
      )
    elif recovered:
      breakdown = recovered["outcome"]["final_accepted_plan"]
      warnings = _forbidden_warnings(
          breakdown,
          app_display_names,
          semantic_goal=item["semantic_goal"],
      )
      if args.strict_forbidden_check and warnings:
        raise RuntimeError(
            "Accepted audit plan no longer passes the bound validator: "
            f"{item['plan_key']} warnings={warnings}"
        )
      print(
          f"[{index}/{len(task_items)}] recover accepted audit plan "
          f"{item['plan_key']} for {item['key']}"
      )
    else:
      print(f"[{index}/{len(task_items)}] generate {item['key']}")
      breakdown, warnings, repair_attempts = _generate_one_plan(
          args=args,
          item=item,
          app_display_names=app_display_names,
          resolved_model=resolved_model,
          gemini_wrapper=gemini_wrapper,
          openai_base_url=openai_base_url,
          openai_api_key=openai_api_key,
          audit=audit,
      )
      if args.sleep_seconds:
        time.sleep(args.sleep_seconds)

    entry = _build_entry(
        item,
        breakdown,
        resolved_model,
        warnings,
        model_identity=model_identity,
    )
    entry["generator_provider"] = args.provider
    if repair_attempts:
      entry["repair_attempts"] = repair_attempts
    payload["breakdowns"].append(entry)
    existing_keys.add(item["key"])
    by_plan_key.setdefault(item["plan_key"], entry)
    generated_count += 1
    _sync_output_audit_metadata(payload, audit)
    _write_payload(output_path, payload)

  _sync_output_audit_metadata(payload, audit)
  _write_payload(output_path, payload)
  if audit is not None:
    audit.close()
  print(
      f"Wrote {len(payload['breakdowns'])} total breakdown entries to"
      f" {output_path} ({generated_count} new)."
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", required=True, help="Output JSON path.")
  parser.add_argument(
      "--audit_log",
      default="",
      help=(
          "Append-only JSONL record of every external request attempt, raw "
          "response, rejection, and accepted plan. Mandatory at a new, absent "
          "path for non-dry-run frozen-cohort generation; --resume requires "
          "the original verified audit."
      ),
  )
  parser.add_argument(
      "--provider",
      choices=("gemini", "openai", "qwen"),
      default="gemini",
      help="Backend used to generate the breakdowns.",
  )
  parser.add_argument(
      "--model",
      default="",
      help=(
          "Generator model name. Defaults: "
          f"gemini={DEFAULT_GEMINI_MODEL}, openai={DEFAULT_OPENAI_MODEL}, "
          f"qwen={DEFAULT_QWEN_MODEL}."
      ),
  )
  parser.add_argument(
      "--model_identity",
      default="",
      help=(
          "Underlying served-checkpoint identity. Mandatory and nonempty for "
          "frozen-cohort Qwen generation because --model may be an endpoint "
          "alias."
      ),
  )
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--max_retry", type=int, default=3)
  parser.add_argument("--timeout_sec", type=float, default=120.0)
  parser.add_argument("--sleep_seconds", type=float, default=0.0)
  parser.add_argument(
      "--openai_base_url",
      default="",
      help=(
          "OpenAI-compatible base URL. OpenAI defaults to its public API; "
          "Qwen falls back to QWEN_C2_BASE_URL and has no public default."
      ),
  )
  parser.add_argument(
      "--openai_api_key",
      default="",
      help=(
          "OpenAI-compatible API key. Falls back to OPENAI_API_KEY for the "
          "OpenAI provider or QWEN_C2_API_KEY for the Qwen provider."
      ),
  )
  parser.add_argument(
      "--suite_family",
      default=registry.TaskRegistry.ANDROID_WORLD_FAMILY,
      choices=[
          registry.TaskRegistry.ANDROID_WORLD_FAMILY,
          registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
          registry.TaskRegistry.MINIWOB_FAMILY,
          registry.TaskRegistry.ANDROID_FAMILY,
          registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
      ],
  )
  parser.add_argument(
      "--categories",
      default=",".join(DEFAULT_CATEGORIES),
      help="Comma-separated CATBench app-generalization categories.",
  )
  parser.add_argument(
      "--tasks",
      default="",
      help="Comma-separated task names. Overrides --categories when set.",
  )
  parser.add_argument(
      "--n_task_combinations",
      type=int,
      default=task_breakdowns.DEFAULT_N_TASK_COMBINATIONS,
      help="Number of exact task instances per template (revision default: 3).",
  )
  parser.add_argument("--task_random_seed", type=int, default=30)
  parser.add_argument("--fixed_task_seed", action="store_true")
  parser.add_argument("--limit", type=int, default=0)
  parser.add_argument(
      "--cohort_manifest",
      default="",
      help=(
          "Frozen primary cohort. Resolves its exact 23 real apps and rejects "
          "selective task/limit generation."
      ),
  )
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--overwrite", action="store_true")
  parser.add_argument("--dry_run", action="store_true")
  parser.add_argument(
      "--dedupe_by_goal",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Deprecated compatibility flag. Shared semantic-plan reuse is always "
          "enforced for valid C2 runs; --no-dedupe_by_goal is rejected."
      ),
  )
  parser.add_argument(
      "--strict_forbidden_check",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Fail if generated JSON appears to include forbidden grounding "
          "details. Default on; pass --no-strict_forbidden_check only for "
          "ablation diagnosis."
      ),
  )
  parser.add_argument(
      "--validation_retry",
      type=int,
      default=3,
      help=(
          "When strict forbidden-detail validation fails, retry generation "
          "with a repair prompt this many times before aborting."
      ),
  )
  parser.add_argument(
      "--allow_prompt_mismatch",
      action="store_true",
      help=(
          "Allow --resume even when the prompt template SHA changed since "
          "the last write. Not recommended; produces mixed cohorts."
      ),
  )
  parser.add_argument(
      "--allow_provider_mismatch",
      action="store_true",
      help=(
          "Allow --resume to write entries from a different provider into "
          "an existing file. Not recommended; produces mixed-provider files."
      ),
  )
  return parser.parse_args()


def main() -> None:
  generate(parse_args())


if __name__ == "__main__":
  main()
