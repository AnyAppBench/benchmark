"""Utilities for CATBench application-independent task breakdowns.

The breakdown condition is opt-in through environment variables:

  CATBENCH_TASK_BREAKDOWN_FILE=/path/to/breakdowns.json
  CATBENCH_TASK_BREAKDOWN_MODE=prepend
  CATBENCH_TASK_BREAKDOWN_REQUIRED=1

When enabled, the original task goal remains the evaluation goal. The augmented
prompt goal is only what the evaluated agent sees.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


PROMPT_GOAL_KEY = "prompt_goal"
TASK_BREAKDOWN_TEXT_KEY = "task_breakdown_text"
TASK_BREAKDOWN_METADATA_KEY = "task_breakdown_metadata"

ENV_BREAKDOWN_FILE = "CATBENCH_TASK_BREAKDOWN_FILE"
ENV_BREAKDOWN_MODE = "CATBENCH_TASK_BREAKDOWN_MODE"
ENV_BREAKDOWN_REQUIRED = "CATBENCH_TASK_BREAKDOWN_REQUIRED"

DEFAULT_MODE = "prepend"
DEFAULT_N_TASK_COMBINATIONS = 3
OFF_MODES = {"", "0", "false", "off", "none", "disabled"}
ORIGINAL_FIRST_MODES = {"original_first", "original-first", "append"}
TARGET_APP_PLACEHOLDER = "[TARGET_APP]"

# Exact prompt-goal layout markers. ``build_prompt_goal`` composes the prompt
# from these and ``original_goal_from_prompt_goal`` parses it back, so they
# must stay in sync (agents key their debug-artifact directories by the
# original instruction while ``agent.step`` receives the augmented prompt).
BREAKDOWN_HEADER = (
    "Application-independent task breakdown generated before observing the "
    "screen:\n"
)
ORIGINAL_HEADER = "Original user instruction:\n"
ORIGINAL_FIRST_HEADER = "Original user instruction (authoritative):\n"
PREPEND_TRAILER = (
    "Use the breakdown only as high-level task structure. You must still "
    "ground every action in the live application interface, and you must not "
    "assume coordinates, accessibility-node identifiers, or app-specific UI "
    "labels from the breakdown."
)
ORIGINAL_FIRST_TRAILER = (
    "Follow the original instruction exactly, including the named target app. "
    "Do not substitute another app. Use the breakdown only as high-level task "
    "structure. You must still ground every action in the live application "
    "interface, and you must not assume coordinates, accessibility-node "
    "identifiers, or app-specific UI labels from the breakdown."
)


def normalize_goal(goal: str) -> str:
  return " ".join(str(goal).strip().split())


def goal_sha256(goal: str) -> str:
  return hashlib.sha256(normalize_goal(goal).encode("utf-8")).hexdigest()


def make_key(task_template: str, goal: str, instance_id: int) -> str:
  """Returns the exact key for one scheduled task instance.

  ``goal_sha256`` is not a sufficient instance identifier: two independently
  sampled instances may legitimately render the same instruction.  Requiring
  the suite-local instance ID prevents K>1 schedules from collapsing those
  trials during generation, preflight, or runtime lookup.
  """
  return (
      f"{task_template}|instance={int(instance_id)}|{goal_sha256(goal)}"
  )


def app_neutral_goal(goal: str, app_display_name: str | None) -> str:
  """Replaces the routed app name while preserving the semantic instruction.

  CATBench goals necessarily identify the target application, but the C2
  planner must not receive that app identity.  Generated task classes expose
  the exact display name used in their template.  Replacing that one routing
  field gives every app implementation of a canonical task the same planner
  input once their sampled parameters are paired.

  Raises:
    ValueError: when an app name was supplied but is absent from the goal.  A
      silent fallback here would permit app-specific C2 plans.
  """
  normalized_goal = normalize_goal(goal)
  display_name = normalize_goal(app_display_name or "")
  if not display_name:
    return normalized_goal
  pattern = re.compile(re.escape(display_name), flags=re.IGNORECASE)
  if not pattern.search(normalized_goal):
    raise ValueError(
        f"Target app name {display_name!r} is absent from goal {goal!r}."
    )
  return normalize_goal(pattern.sub(TARGET_APP_PLACEHOLDER, normalized_goal, count=1))


def make_semantic_plan_key(
    semantic_task_id: str,
    instance_id: int,
    semantic_goal: str,
) -> str:
  """Stable key for one plan shared across all apps in a paired instance."""
  return (
      f"{semantic_task_id}|instance={int(instance_id)}|"
      f"{goal_sha256(semantic_goal)}"
  )


def effective_mode() -> str:
  """Returns the runtime breakdown mode: ``"off"`` unless a file is set.

  Mirrors ``is_enabled``: an unset ``CATBENCH_TASK_BREAKDOWN_MODE`` defaults
  to ``DEFAULT_MODE`` whenever ``CATBENCH_TASK_BREAKDOWN_FILE`` is set.
  Manifests must record this value rather than re-deriving their own default.
  """
  path = os.environ.get(ENV_BREAKDOWN_FILE, "").strip()
  mode = os.environ.get(ENV_BREAKDOWN_MODE, DEFAULT_MODE).strip().lower()
  if not path or mode in OFF_MODES:
    return "off"
  return mode


def is_enabled() -> bool:
  return effective_mode() != "off"


def is_required() -> bool:
  return os.environ.get(ENV_BREAKDOWN_REQUIRED, "").strip().lower() in {
      "1",
      "true",
      "yes",
      "required",
  }


@functools.lru_cache(maxsize=32)
def _load_payload_cached(path: str, mtime_ns: int) -> dict[str, Any]:
  # mtime_ns is part of the cache key so the cache invalidates when the
  # underlying file is rewritten (e.g. the generator appending new entries
  # mid-experiment).
  del mtime_ns  # unused inside the function body; only used in the cache key
  with Path(path).expanduser().open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  if not isinstance(payload, dict):
    raise ValueError(f"Breakdown file must contain a JSON object: {path}")
  return payload


def _load_payload(path: str) -> dict[str, Any]:
  expanded = Path(path).expanduser()
  try:
    mtime_ns = expanded.stat().st_mtime_ns
  except FileNotFoundError as exc:
    raise FileNotFoundError(f"Breakdown file not found: {expanded}") from exc
  return _load_payload_cached(str(expanded), mtime_ns)


def clear_payload_cache() -> None:
  """Drop the lru_cache. Mostly for tests."""
  _load_payload_cached.cache_clear()


def _entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
  if "breakdowns" in payload:
    raw_entries = payload["breakdowns"]
  elif "tasks" in payload:
    raw_entries = payload["tasks"]
  else:
    raw_entries = None

  if isinstance(raw_entries, list):
    return [entry for entry in raw_entries if isinstance(entry, dict)]

  if isinstance(raw_entries, dict):
    entries = []
    for key, value in raw_entries.items():
      if isinstance(value, dict):
        entry = dict(value)
        entry.setdefault("key", key)
      else:
        entry = {"key": key, "breakdown": value}
      entries.append(entry)
    return entries

  # Allow a compact mapping-only file: {"Task|hash": {"steps": [...]}, ...}.
  entries = []
  for key, value in payload.items():
    if key == "metadata":
      continue
    if isinstance(value, dict):
      entry = dict(value)
      entry.setdefault("key", key)
    else:
      entry = {"key": key, "breakdown": value}
    entries.append(entry)
  return entries


def _entry_goal_hash(entry: dict[str, Any]) -> str | None:
  value = entry.get("goal_sha256") or entry.get("goal_hash")
  return str(value) if value else None


def _entry_task(entry: dict[str, Any]) -> str | None:
  value = entry.get("task_template") or entry.get("task_name")
  return str(value) if value else None


def _entry_matches(
    entry: dict[str, Any],
    task_template: str,
    goal: str,
    instance_id: int,
) -> bool:
  # Cross-template guard: when the entry was written with a task_template
  # field, we require it to match. Without this guard, two task templates
  # whose user instructions happen to be identical could swap breakdowns.
  task_name = _entry_task(entry)
  if task_name and task_name != task_template:
    return False

  # Fail closed for legacy entries. Goal text/hash alone is not unique when
  # n_task_combinations > 1, so an entry without an exact instance identity
  # must never be selected for a scheduled C2 episode.
  try:
    entry_instance_id = int(entry["instance_id"])
  except (KeyError, TypeError, ValueError, OverflowError):
    return False
  if entry_instance_id != int(instance_id):
    return False

  normalized = normalize_goal(goal)
  goal_hash = goal_sha256(goal)
  entry_hash = _entry_goal_hash(entry)
  entry_goal = entry.get("goal")
  entry_key = str(entry.get("key", ""))
  expected_key = make_key(task_template, goal, instance_id)

  if entry_key != expected_key:
    return False

  if entry_hash is not None:
    return entry_hash == goal_hash
  if entry_goal:
    return normalize_goal(str(entry_goal)) == normalized
  return True  # Exact key is sufficient for compact mapping-only payloads.


def load_metadata(path: str | None = None) -> dict[str, Any]:
  """Return the payload's metadata dict (or empty if missing/disabled)."""
  path = path or os.environ.get(ENV_BREAKDOWN_FILE, "")
  if not path:
    return {}
  payload = _load_payload(os.path.expanduser(path))
  metadata = payload.get("metadata") if isinstance(payload, dict) else None
  return metadata if isinstance(metadata, dict) else {}


def validate_runner_compatibility(
    runner_seed: int | None = None,
    runner_n_task_combinations: int | None = None,
    runner_fixed_task_seed: bool | None = None,
    runner_suite_family: str | None = None,
    path: str | None = None,
) -> list[str]:
  """Compare runner config against breakdown-file metadata.

  Returns a list of human-readable mismatch messages. Empty means compatible.
  Pass only the fields the caller can observe; None means "do not check".
  """
  metadata = load_metadata(path)
  if not metadata:
    return []
  mismatches: list[str] = []
  if (
      runner_seed is not None
      and "task_random_seed" in metadata
      and int(metadata["task_random_seed"]) != int(runner_seed)
  ):
    mismatches.append(
        f"task_random_seed mismatch: file={metadata['task_random_seed']} "
        f"runner={runner_seed}"
    )
  if (
      runner_n_task_combinations is not None
      and "n_task_combinations" in metadata
      and int(metadata["n_task_combinations"]) != int(runner_n_task_combinations)
  ):
    mismatches.append(
        f"n_task_combinations mismatch: file={metadata['n_task_combinations']} "
        f"runner={runner_n_task_combinations}"
    )
  if (
      runner_fixed_task_seed is not None
      and "fixed_task_seed" in metadata
      and bool(metadata["fixed_task_seed"]) != bool(runner_fixed_task_seed)
  ):
    mismatches.append(
        f"fixed_task_seed mismatch: file={metadata['fixed_task_seed']} "
        f"runner={runner_fixed_task_seed}"
    )
  if (
      runner_suite_family is not None
      and metadata.get("suite_family")
      and metadata["suite_family"] != runner_suite_family
  ):
    mismatches.append(
        f"suite_family mismatch: file={metadata['suite_family']} "
        f"runner={runner_suite_family}"
    )
  return mismatches


def lookup_breakdown(
    task_template: str,
    goal: str,
    instance_id: int,
    path: str | None = None,
) -> dict[str, Any] | None:
  path = path or os.environ.get(ENV_BREAKDOWN_FILE, "")
  if not path:
    return None
  payload = _load_payload(os.path.expanduser(path))
  matches = [
      entry
      for entry in _entries(payload)
      if _entry_matches(entry, task_template, goal, instance_id)
  ]
  if len(matches) > 1:
    raise ValueError(
        "Duplicate task breakdown entries for "
        f"{make_key(task_template, goal, instance_id)}"
    )
  return matches[0] if matches else None


def _numbered_steps(steps: list[Any]) -> str:
  cleaned = [str(step).strip() for step in steps if str(step).strip()]
  return "\n".join(f"{idx}. {step}" for idx, step in enumerate(cleaned, 1))


def format_breakdown_text(entry: dict[str, Any]) -> str:
  if entry.get("breakdown_text"):
    return str(entry["breakdown_text"]).strip()

  breakdown = entry.get("breakdown", entry)
  if isinstance(breakdown, str):
    return breakdown.strip()
  if isinstance(breakdown, list):
    return _numbered_steps(breakdown)
  if isinstance(breakdown, dict):
    steps = breakdown.get("steps")
    if isinstance(steps, list):
      return _numbered_steps(steps)
    if isinstance(steps, str):
      return steps.strip()
  return ""


def build_prompt_goal(
    original_goal: str,
    breakdown_text: str,
    mode: str | None = None,
) -> str:
  mode = (mode or os.environ.get(ENV_BREAKDOWN_MODE, DEFAULT_MODE)).strip().lower()
  if mode in ORIGINAL_FIRST_MODES:
    return (
        f"{ORIGINAL_FIRST_HEADER}{original_goal}\n\n"
        f"{BREAKDOWN_HEADER}{breakdown_text}\n\n"
        f"{ORIGINAL_FIRST_TRAILER}"
    )
  return (
      f"{BREAKDOWN_HEADER}{breakdown_text}\n\n"
      f"{ORIGINAL_HEADER}{original_goal}\n\n"
      f"{PREPEND_TRAILER}"
  )


def original_goal_from_prompt_goal(goal: str) -> str:
  """Recovers the original instruction from a ``build_prompt_goal`` output.

  Returns ``goal`` unchanged when it is not an augmented prompt goal, so
  callers can apply it unconditionally (e.g. agents resolving the debug
  artifact directory that was keyed by the original ``task.goal``).
  """
  if not isinstance(goal, str):
    return goal
  if goal.startswith(ORIGINAL_FIRST_HEADER):
    body = goal[len(ORIGINAL_FIRST_HEADER):]
    marker = "\n\n" + BREAKDOWN_HEADER
    idx = body.find(marker)
    return body[:idx] if idx >= 0 else goal
  if goal.startswith(BREAKDOWN_HEADER):
    marker = "\n\n" + ORIGINAL_HEADER
    idx = goal.rfind(marker)
    if idx < 0:
      return goal
    body = goal[idx + len(marker):]
    trailer = "\n\n" + PREPEND_TRAILER
    if body.endswith(trailer):
      return body[: -len(trailer)]
    # Tolerate a trailer edited after the prompt was built.
    t_idx = body.rfind("\n\nUse the breakdown only")
    return body[:t_idx] if t_idx >= 0 else body
  return goal


def build_prompt_context(
    task_template: str,
    original_goal: str,
    instance_id: int | None = None,
) -> dict[str, Any]:
  """Returns prompt-goal context for the breakdown condition."""
  context = {
      "enabled": is_enabled(),
      "found": False,
      "source_file": os.environ.get(ENV_BREAKDOWN_FILE, ""),
      PROMPT_GOAL_KEY: original_goal,
      TASK_BREAKDOWN_TEXT_KEY: "",
      TASK_BREAKDOWN_METADATA_KEY: {},
  }
  if not context["enabled"]:
    return context

  if instance_id is None:
    message = (
        "Missing runtime instance_id for exact task-breakdown lookup: "
        f"{task_template} / {goal_sha256(original_goal)}"
    )
    if is_required():
      raise KeyError(message)
    context[TASK_BREAKDOWN_METADATA_KEY] = {
        "lookup_missing": True,
        "missing_instance_id": True,
        "goal_sha256": goal_sha256(original_goal),
        "task_template": task_template,
    }
    return context

  entry = lookup_breakdown(task_template, original_goal, instance_id)
  if entry is None:
    message = (
        "Missing task breakdown for "
        f"{make_key(task_template, original_goal, instance_id)}"
    )
    if is_required():
      raise KeyError(message)
    context[TASK_BREAKDOWN_METADATA_KEY] = {
        "lookup_missing": True,
        "instance_id": int(instance_id),
        "goal_sha256": goal_sha256(original_goal),
        "task_template": task_template,
    }
    return context

  breakdown_text = format_breakdown_text(entry)
  if not breakdown_text:
    message = f"Empty task breakdown for {task_template}"
    if is_required():
      raise ValueError(message)
    # Mark the empty case explicitly so downstream analysis can filter it out
    # of the condition pool (M5 in the senior review).
    context[TASK_BREAKDOWN_METADATA_KEY] = {
        "empty_breakdown": True,
        "instance_id": int(instance_id),
        "goal_sha256": goal_sha256(original_goal),
        "task_template": task_template,
    }
    return context

  mode = os.environ.get(ENV_BREAKDOWN_MODE, DEFAULT_MODE)
  context.update(
      {
          "found": True,
          PROMPT_GOAL_KEY: build_prompt_goal(original_goal, breakdown_text, mode),
          TASK_BREAKDOWN_TEXT_KEY: breakdown_text,
          TASK_BREAKDOWN_METADATA_KEY: {
              "source_file": context["source_file"],
              "key": entry.get(
                  "key", make_key(task_template, original_goal, instance_id)
              ),
              "task_template": task_template,
              "instance_id": int(instance_id),
              "goal_sha256": goal_sha256(original_goal),
              "generator_provider": entry.get("generator_provider"),
              "generator_model": entry.get("generator_model"),
              "semantic_task_id": entry.get("semantic_task_id"),
              "semantic_goal_sha256": entry.get("semantic_goal_sha256"),
              "plan_key": entry.get("plan_key"),
              "plan_sha256": entry.get("plan_sha256"),
              "condition": f"application_independent_breakdown_{mode}",
          },
      }
  )
  return context
