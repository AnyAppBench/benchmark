#!/usr/bin/env python3
"""Preflight check that every scheduled CATBench task has a breakdown entry.

This is what we should have written before the first published condition run.
Reads the breakdown JSON the runner will use, enumerates the suite at the
SAME seed / n_task_combinations / fixed_task_seed / suite_family the runner
will use, and reports:

  - missing entries (task_template + instance_id + goal_sha256 not in the file)
  - extra entries (in the file but not in the suite schedule)
  - entries whose validation_warnings list is non-empty
  - any runner/file metadata mismatches (seed, family, ...)

Exits non-zero on missing entries (so it fails CI / pre-run wrappers cleanly).
"""

from __future__ import annotations

import argparse
import collections
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
import sys
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
import build_catbench_frozen_schedule as frozen_schedule_builder
from android_world import registry
from android_world import suite_utils
from app_generalization_profiles import get_domain_profiles


HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
GENERIC_APP_DISPLAY_NAMES = frozenset({
    "clock",
    "contacts",
    "files",
    "maps",
    "messages",
})
_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_COORDINATE_PAIR_PATTERN = re.compile(
    rf"(?<![\d.])(?P<latitude>{_NUMBER_PATTERN})\s*,\s*"
    rf"(?P<longitude>{_NUMBER_PATTERN})(?!\d|\.\d)"
)

FORBIDDEN_PLAN_PATTERNS = {
    "android_package_identifier": re.compile(
        r"\b(?:[A-Za-z][A-Za-z0-9_]*\.){2,}[A-Za-z][A-Za-z0-9_]*\b"
    ),
    "coordinate_pair": _COORDINATE_PAIR_PATTERN,
    "x_y_coordinate": re.compile(
        rf"\b[xy]\s*[:=]\s*{_NUMBER_PATTERN}(?!\d|\.\d)",
        re.IGNORECASE,
    ),
    "coordinate_word": re.compile(r"\bcoordinates?\b", re.IGNORECASE),
    "accessibility_or_resource_identifier": re.compile(
        r"\b(accessibility|node id|resource-id|content-desc)\b",
        re.IGNORECASE,
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


def _strict_json_loads(raw: str, source: str) -> Any:
  def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f"duplicate JSON key {key!r} in {source}")
      result[key] = value
    return result

  def _constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} in {source}")

  return json.loads(
      raw,
      object_pairs_hook=_object,
      parse_constant=_constant,
  )


def _tasks_for_categories(categories: tuple[str, ...]) -> list[str]:
  profiles = get_domain_profiles()
  unknown = [c for c in categories if c not in profiles]
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


def _enumerate_scheduled_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
  task_registry = registry.TaskRegistry()
  full_registry = task_registry.get_registry(family=args.suite_family)
  task_identities: dict[str, tuple[str, str]] = {}
  selected_dry_run_instances: dict[tuple[str, str, str], int] | None = None
  if args.cohort_manifest:
    if args.tasks:
      raise ValueError("Frozen-cohort preflight forbids --tasks.")
    cohort = catbench_primary_cohort.load(args.cohort_manifest)
    schedule_issues = catbench_primary_cohort.validate_schedule_args(
        cohort,
        suite_family=args.suite_family,
        categories=tuple(c for c in args.categories.split(",") if c),
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
      # A filtered primary preflight is forbidden.  The only smaller plan
      # surface is the exact separately named G6 release, whose five selected
      # task instances are part of its immutable cohort identity.
      frozen_schedule_builder.validate_g6_dry_run_cohort({
          key: value for key, value in cohort.items() if key != "_path"
      })
      selected_dry_run_instances = {}
      for block in cohort["paired_blocks"]:
        key = (
            str(block["category"]),
            str(block["app_id"]),
            str(block["semantic_task_id"]),
        )
        if key in selected_dry_run_instances:
          raise ValueError(f"Duplicate G6 dry-run task identity: {key}")
        selected_dry_run_instances[key] = int(block["instance_id"])
  else:
    task_names = [t for t in args.tasks.split(",") if t]
    if not task_names:
      task_names = _tasks_for_categories(
          tuple(c for c in args.categories.split(",") if c)
      )
  missing = [t for t in task_names if t not in full_registry]
  if missing:
    raise ValueError(f"Task(s) not in {args.suite_family} registry: {missing}")
  suite = suite_utils.create_suite(
      full_registry,
      n_task_combinations=args.n_task_combinations,
      seed=args.task_random_seed,
      tasks=task_names,
      use_identical_params=args.fixed_task_seed,
  )
  scheduled: list[dict[str, Any]] = []
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
      scheduled.append(
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
  return scheduled


def _index_breakdowns(
    payload: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
  index: dict[str, dict[str, Any]] = {}
  duplicates: list[str] = []
  invalid: list[str] = []
  for entry_index, entry in enumerate(payload.get("breakdowns", [])):
    if not isinstance(entry, dict):
      invalid.append(f"entry[{entry_index}] is not a JSON object")
      continue
    template = entry.get("task_template")
    goal_hash = entry.get("goal_sha256")
    instance_id = entry.get("instance_id")
    if not template or not goal_hash or instance_id is None:
      invalid.append(
          f"entry[{entry_index}] missing task_template, instance_id, or "
          "goal_sha256"
      )
      continue
    try:
      normalized_instance_id = int(instance_id)
    except (TypeError, ValueError, OverflowError):
      invalid.append(
          f"entry[{entry_index}] has invalid instance_id={instance_id!r}"
      )
      continue
    key = (
        f"{template}|instance={normalized_instance_id}|{goal_hash}"
    )
    if entry.get("key") != key:
      invalid.append(
          f"entry[{entry_index}] key={entry.get('key')!r}; expected {key!r}"
      )
      continue
    if key in index:
      duplicates.append(key)
    else:
      index[key] = entry
  return index, sorted(duplicates), invalid


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
    for coordinate_match in FORBIDDEN_PLAN_PATTERNS[
        "coordinate_word"
    ].finditer(text):
      if _is_geographic_coordinate_reference(text, coordinate_match):
        masked_spans.append(coordinate_match.span())

  if not masked_spans:
    return text
  masked = list(text)
  for start, end in masked_spans:
    masked[start:end] = " " * (end - start)
  return "".join(masked)


def _plan_content_errors(
    entry: dict[str, Any],
    *,
    key: str,
    app_display_names: tuple[str, ...],
    semantic_goal: str = "",
) -> list[str]:
  """Validate the exact plan bytes instead of trusting self-declared warnings."""
  errors: list[str] = []
  breakdown = entry.get("breakdown")
  if not isinstance(breakdown, dict):
    return [f"{key}: breakdown must be an object with steps and notes"]
  if set(breakdown) != {"steps", "notes"}:
    errors.append(
        f"{key}: breakdown keys must be exactly ['notes', 'steps']"
    )
  steps = breakdown.get("steps")
  notes = breakdown.get("notes")
  if (
      not isinstance(steps, list)
      or not steps
      or any(not isinstance(step, str) or step != step.strip() or not step for step in steps)
  ):
    errors.append(f"{key}: steps must be a non-empty list of stripped strings")
  if (
      not isinstance(notes, list)
      or any(not isinstance(note, str) or note != note.strip() or not note for note in notes)
  ):
    errors.append(f"{key}: notes must be a list of non-empty stripped strings")

  canonical_text = task_breakdowns.format_breakdown_text(
      {"breakdown": breakdown}
  )
  if not canonical_text:
    errors.append(f"{key}: plan text is empty")
  stored_text = entry.get("breakdown_text")
  if not isinstance(stored_text, str) or stored_text.strip() != canonical_text:
    errors.append(f"{key}: breakdown_text is not the canonical steps text")
  plan_sha256 = entry.get("plan_sha256")
  if not HEX_SHA256.fullmatch(str(plan_sha256 or "")):
    errors.append(f"{key}: plan_sha256 is not a lowercase SHA-256")
  elif hashlib.sha256(canonical_text.encode("utf-8")).hexdigest() != plan_sha256:
    errors.append(f"{key}: plan_sha256 does not match canonical plan text")

  # A warning list is generator output, not evidence.  Recompute the scan for
  # generated and human-authored plans alike.  Human reviewers remain
  # responsible for semantic correctness and less lexical UI leakage.
  if entry.get("validation_warnings") != []:
    errors.append(f"{key}: validation_warnings must be the exact empty list")
  scan_text = _breakdown_scan_text(breakdown)
  coordinate_scan_text = _coordinate_scan_text(scan_text, semantic_goal)
  for warning, pattern in FORBIDDEN_PLAN_PATTERNS.items():
    candidate_text = (
        coordinate_scan_text
        if warning in {"coordinate_pair", "coordinate_word"}
        else scan_text
    )
    if pattern.search(candidate_text):
      errors.append(f"{key}: forbidden plan detail: {warning}")
  for app_name in app_display_names:
    normalized = " ".join(app_name.strip().split())
    if not normalized or normalized.casefold() in GENERIC_APP_DISPLAY_NAMES:
      continue
    if re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", scan_text, re.IGNORECASE):
      errors.append(f"{key}: forbidden target-app name: {normalized}")
      break
  return errors


def _condition_role_errors(
    payload: dict[str, Any], condition: str
) -> list[str]:
  """Check generated-vs-human role fields before human approval is accepted."""
  errors: list[str] = []
  metadata = payload.get("metadata")
  if not isinstance(metadata, dict):
    return ["metadata must be a JSON object"]
  provider = str(metadata.get("generator_provider") or "").strip().lower()
  model = str(metadata.get("generator_model") or "").strip()
  entries = payload.get("breakdowns")
  if not isinstance(entries, list):
    return ["breakdowns must be a JSON list"]
  if condition == "c2_g":
    if not provider or provider == "human":
      errors.append("metadata.generator_provider must name a non-human C2-G planner")
    if not model:
      errors.append("metadata.generator_model is required for C2-G")
    if not HEX_SHA256.fullmatch(str(metadata.get("prompt_sha256") or "")):
      errors.append("metadata.prompt_sha256 must be a lowercase SHA-256 for C2-G")
    audit = metadata.get("attempt_audit")
    if not isinstance(audit, dict):
      errors.append("metadata.attempt_audit is required for frozen C2-G")
    for index, entry in enumerate(entries):
      if not isinstance(entry, dict):
        continue
      if str(entry.get("generator_provider") or "").strip().lower() != provider:
        errors.append(f"entry[{index}] generator_provider differs from metadata")
      if str(entry.get("generator_model") or "").strip() != model:
        errors.append(f"entry[{index}] generator_model differs from metadata")
  elif condition == "c2_o":
    if provider != "human":
      errors.append("metadata.generator_provider must be 'human' for C2-O")
    if metadata.get("authoring_policy") != "two_human_authors_app_neutral":
      errors.append(
          "metadata.authoring_policy must be two_human_authors_app_neutral"
      )
    if metadata.get("author_input_policy") != (
        "app_neutral_goal_only_no_app_or_ui_observation"
    ):
      errors.append("metadata.author_input_policy is missing or not app-neutral")
    authors = metadata.get("authors")
    if not isinstance(authors, list) or len(authors) != 2:
      errors.append("metadata.authors must contain exactly two human authors")
      author_ids: set[str] = set()
    else:
      author_ids = set()
      for author in authors:
        if not isinstance(author, dict):
          errors.append("metadata.authors contains a malformed author")
          continue
        author_id = str(author.get("author_id") or "").strip()
        if not author_id:
          errors.append("C2-O author_id is missing")
        else:
          author_ids.add(author_id)
        if not str(author.get("authored_at") or "").strip():
          errors.append(f"C2-O author {author_id or '<missing>'} lacks authored_at")
        if not HEX_SHA256.fullmatch(
            str(author.get("authorship_evidence_sha256") or "")
        ):
          errors.append(
              f"C2-O author {author_id or '<missing>'} lacks evidence SHA-256"
          )
      if len(author_ids) != 2:
        errors.append("C2-O author identities must be distinct")
    expected_author_ids = sorted(author_ids)
    for index, entry in enumerate(entries):
      if not isinstance(entry, dict):
        continue
      if str(entry.get("generator_provider") or "").strip().lower() != "human":
        errors.append(f"entry[{index}] generator_provider must be human")
      raw_author_ids = entry.get("author_ids")
      if (
          not isinstance(raw_author_ids, list)
          or raw_author_ids != expected_author_ids
      ):
        errors.append(
            f"entry[{index}] author_ids must equal the two sorted metadata authors"
        )
  else:
    errors.append(f"unsupported plan condition: {condition!r}")
  return errors


def _semantic_pairing_errors(
    scheduled: list[dict[str, Any]],
    index: dict[str, dict[str, Any]],
) -> list[str]:
  """Returns violations of exact-instance and one-shared-plan pairing."""
  errors: list[str] = []
  schedule_by_pair: dict[tuple[str, int], list[dict[str, Any]]] = (
      collections.defaultdict(list)
  )
  plan_hashes: dict[str, set[str]] = collections.defaultdict(set)
  app_display_names = tuple(sorted({
      str(item["app_display_name"])
      for item in scheduled
      if item.get("app_display_name")
  }))

  for item in scheduled:
    pair = (item["semantic_task_id"], item["instance_id"])
    schedule_by_pair[pair].append(item)
    entry = index.get(item["key"])
    if entry is None:
      continue
    if entry.get("goal") != item["goal"]:
      errors.append(f"{item['key']}: goal text differs from scheduled goal")
    if entry.get("semantic_goal") != item["semantic_goal"]:
      errors.append(
          f"{item['key']}: semantic_goal text differs from scheduled app-neutral goal"
      )
    for field in (
        "semantic_task_id",
        "semantic_goal_sha256",
        "semantic_parameter_sha256",
        "plan_key",
        "plan_sha256",
    ):
      if not entry.get(field):
        errors.append(f"{item['key']}: missing {field}")
    for field in (
        "semantic_task_id",
        "semantic_goal_sha256",
        "semantic_parameter_sha256",
        "plan_key",
    ):
      if entry.get(field) and str(entry[field]) != str(item[field]):
        errors.append(
            f"{item['key']}: {field} file={entry[field]!r} "
            f"scheduled={item[field]!r}"
        )
    breakdown_text = task_breakdowns.format_breakdown_text(entry)
    computed_plan_hash = hashlib.sha256(
        breakdown_text.encode("utf-8")
    ).hexdigest()
    if entry.get("plan_sha256") and entry["plan_sha256"] != computed_plan_hash:
      errors.append(f"{item['key']}: plan_sha256 does not match breakdown text")
    if entry.get("plan_key"):
      plan_hashes[str(entry["plan_key"])].add(computed_plan_hash)
    errors.extend(
        _plan_content_errors(
            entry,
            key=item["key"],
            app_display_names=app_display_names,
            semantic_goal=item["semantic_goal"],
        )
    )

  for pair, items in sorted(schedule_by_pair.items()):
    parameter_hashes = {item["semantic_parameter_sha256"] for item in items}
    goal_hashes = {item["semantic_goal_sha256"] for item in items}
    plan_keys = {item["plan_key"] for item in items}
    if len(parameter_hashes) != 1:
      errors.append(
          f"{pair}: sampled parameters differ across apps "
          f"({len(parameter_hashes)} hashes)"
      )
    if len(goal_hashes) != 1:
      errors.append(
          f"{pair}: app-neutral instructions differ across apps "
          f"({len(goal_hashes)} hashes)"
      )
    if len(plan_keys) != 1:
      errors.append(
          f"{pair}: plan keys differ across apps ({len(plan_keys)} keys)"
      )

  for plan_key, hashes in sorted(plan_hashes.items()):
    if len(hashes) != 1:
      errors.append(
          f"{plan_key}: apps received {len(hashes)} different breakdowns"
      )
  return errors


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--breakdown_file", required=True)
  parser.add_argument(
      "--suite_family",
      default=registry.TaskRegistry.ANDROID_WORLD_FAMILY,
  )
  parser.add_argument(
      "--categories",
      default="sms,files,maps,contacts,clock",
      help="Comma-separated CATBench categories the runner will schedule.",
  )
  parser.add_argument(
      "--tasks",
      default="",
      help="Comma-separated task names. Overrides --categories.",
  )
  parser.add_argument(
      "--n_task_combinations",
      type=int,
      default=task_breakdowns.DEFAULT_N_TASK_COMBINATIONS,
      help="Number of exact task instances per template (revision default: 3).",
  )
  parser.add_argument("--task_random_seed", type=int, default=30)
  parser.add_argument("--fixed_task_seed", action="store_true")
  parser.add_argument(
      "--cohort_manifest",
      default="",
      help=(
          "Exact frozen cohort. The primary release covers all 690 app "
          "instances; the separately named G6 release covers only its five "
          "preregistered discard-only instances."
      ),
  )
  parser.add_argument(
      "--report_json",
      default="",
      help="Optional path to write a machine-readable preflight report.",
  )
  parser.add_argument(
      "--fail_on_warnings",
      action="store_true",
      help="Exit non-zero if any entry has a non-empty validation_warnings list.",
  )
  parser.add_argument(
      "--condition",
      choices=("c2_g", "c2_o"),
      default="",
      help=(
          "When supplied, enforce the generated C2-G or human C2-O role and "
          "provenance fields. The frozen schedule consumer always supplies it."
      ),
  )
  args = parser.parse_args()

  breakdown_path = Path(args.breakdown_file).expanduser().resolve()
  if not breakdown_path.exists():
    print(f"FAIL: breakdown file not found: {breakdown_path}", file=sys.stderr)
    return 2

  try:
    payload = _strict_json_loads(
        breakdown_path.read_text(encoding="utf-8"), str(breakdown_path)
    )
  except (OSError, json.JSONDecodeError, ValueError) as exc:
    print(f"FAIL: invalid breakdown JSON: {exc}", file=sys.stderr)
    return 2
  if not isinstance(payload, dict):
    print("FAIL: breakdown root must be a JSON object", file=sys.stderr)
    return 2
  metadata = payload.get("metadata") or {}

  # Metadata compatibility (H2 fix).
  mismatches = task_breakdowns.validate_runner_compatibility(
      runner_seed=args.task_random_seed,
      runner_n_task_combinations=args.n_task_combinations,
      runner_fixed_task_seed=args.fixed_task_seed,
      runner_suite_family=args.suite_family,
      path=str(breakdown_path),
  )

  scheduled = _enumerate_scheduled_tasks(args)
  scheduled_keys = {s["key"] for s in scheduled}
  index, duplicate_keys, invalid_entries = _index_breakdowns(payload)
  index_keys = set(index.keys())

  missing = sorted(scheduled_keys - index_keys)
  extras = sorted(index_keys - scheduled_keys)
  semantic_errors = _semantic_pairing_errors(scheduled, index)
  if args.condition:
    semantic_errors.extend(_condition_role_errors(payload, args.condition))
  required_metadata = {
      "semantic_pairing_version": 2,
      "plan_reuse_policy": "one_plan_per_semantic_instance_across_apps",
      "planner_input_app_identity": "replaced_with_[TARGET_APP]",
      "expected_entry_count": len(scheduled),
      "expected_semantic_plan_count": len({
          item["plan_key"] for item in scheduled
      }),
  }
  if args.cohort_manifest:
    cohort_path = Path(args.cohort_manifest).expanduser().resolve()
    cohort = catbench_primary_cohort.load(cohort_path)
    required_metadata.update({
        "cohort_release_id": cohort["release_id"],
        "cohort_manifest_sha256": hashlib.sha256(
            cohort_path.read_bytes()
        ).hexdigest(),
    })
  for field, expected in required_metadata.items():
    if metadata.get(field) != expected:
      semantic_errors.append(
          f"metadata.{field}={metadata.get(field)!r}; expected {expected!r}"
      )
  warnings_rows: list[dict[str, Any]] = []
  for key, entry in index.items():
    warns = entry.get("validation_warnings") or []
    if warns:
      warnings_rows.append(
          {
              "key": key,
              "task_template": entry.get("task_template"),
              "warnings": list(warns),
          }
      )

  print("=" * 78)
  print(f"Preflight for breakdown file: {breakdown_path}")
  print(f"  generator_provider : {metadata.get('generator_provider', '?')}")
  print(f"  generator_model    : {metadata.get('generator_model', '?')}")
  print(f"  prompt_sha256      : {(metadata.get('prompt_sha256') or '?')[:12]}")
  print(f"  written_at         : {metadata.get('created_or_updated_at', '?')}")
  print(f"  Scheduled tasks    : {len(scheduled)}")
  print(f"  Indexed entries    : {len(index)}")
  print(f"  Missing entries    : {len(missing)}")
  print(f"  Extra entries      : {len(extras)}")
  print(f"  Duplicate entries  : {len(duplicate_keys)}")
  print(f"  Invalid entries    : {len(invalid_entries)}")
  print(f"  Entries w/ warns   : {len(warnings_rows)}")
  print(f"  Metadata mismatches: {len(mismatches)}")
  print(f"  Pairing violations : {len(semantic_errors)}")
  for note in mismatches:
    print(f"    * {note}")
  if missing[:20]:
    print(f"  First {min(20, len(missing))} missing keys:")
    for key in missing[:20]:
      print(f"    {key}")
  if warnings_rows[:10]:
    print("  First 10 entries with validation_warnings:")
    for row in warnings_rows[:10]:
      print(f"    {row['key']}: {row['warnings']}")
  if invalid_entries[:20]:
    print("  First 20 invalid entries:")
    for error in invalid_entries[:20]:
      print(f"    {error}")
  if semantic_errors[:20]:
    print("  First 20 semantic-pairing violations:")
    for error in semantic_errors[:20]:
      print(f"    {error}")

  report = {
      "breakdown_file": str(breakdown_path),
      "metadata": metadata,
      "scheduled_count": len(scheduled),
      "indexed_count": len(index),
      "missing": missing,
      "extras": extras,
      "duplicate_keys": duplicate_keys,
      "invalid_entries": invalid_entries,
      "warnings": warnings_rows,
      "metadata_mismatches": mismatches,
      "semantic_pairing_errors": semantic_errors,
  }
  if args.report_json:
    out = Path(args.report_json).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote JSON report: {out}")

  exit_code = 0
  if (
      missing
      or extras
      or duplicate_keys
      or invalid_entries
      or mismatches
      or semantic_errors
  ):
    print(
        "FAIL: roster, metadata, duplicate, or semantic-pairing violation.",
        file=sys.stderr,
    )
    exit_code = 1
  if args.fail_on_warnings and warnings_rows:
    print("FAIL: --fail_on_warnings set and warnings present.", file=sys.stderr)
    exit_code = 1
  if exit_code == 0:
    print("OK: breakdown file is consistent with the scheduled suite.")
  return exit_code


if __name__ == "__main__":
  raise SystemExit(main())
