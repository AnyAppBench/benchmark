#!/usr/bin/env python3
"""Audit a suffix-0 C2 cohort assembled from one or more roots per model.

The July C2 launch generated three semantic instances per task template.  For a
deadline K=1 analysis, this tool selects only ``*_0.pkl.gz`` and verifies those
artifacts against the frozen Qwen plan JSON before reporting scores.  Repeating
``--model MODEL=ROOT`` allows a model to be assembled from disjoint shards.

The command exits zero only when every requested model has exactly the same 230
instance-0 cells, every artifact agrees with the frozen goal and plan, and no
artifact is unreadable, duplicated, or marked as an environment skip.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import io
import json
import pickle
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))


CATEGORIES = ("sms", "files", "maps", "contacts", "clock")
DISPLAY_CATEGORY = {
    "sms": "SMS",
    "files": "Files",
    "maps": "Maps",
    "contacts": "Contacts",
    "clock": "Clock",
}
PREFIX_CATEGORY = {
    "Sms": "sms",
    "Files": "files",
    "Maps": "maps",
    "Contacts": "contacts",
    "Clock": "clock",
}
SKIP_EXCEPTION_MARKERS = (
    "[skipped_uninstalled]",
    "[skipped_environment]",
    "_EnvironmentNetworkError",
    "network/connectivity error dialog visible",
)


@dataclass(frozen=True)
class LoadedArtifact:
  path: str
  episode: dict[str, Any] | None
  error: str | None


def _sha256_text(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _category_from_template(task_template: str) -> str | None:
  for prefix, category in PREFIX_CATEGORY.items():
    if task_template.startswith(prefix):
      return category
  return None


def _category_and_app(path: Path, task_template: str) -> tuple[str, str]:
  parts = path.parts
  for index, part in enumerate(parts[:-1]):
    if part in CATEGORIES:
      app = parts[index + 1] if index + 1 < len(parts) - 1 else "unknown"
      return part, app
  return _category_from_template(task_template) or "unknown", "unknown"


def _parse_pickle_scalar_after_marker(buffer: bytes, marker_end: int) -> float | None:
  """Decode the small scalar stored immediately after a pickle dict key."""
  pos = marker_end
  while pos < len(buffer) and buffer[pos] == 0x94:  # MEMOIZE
    pos += 1
  if pos >= len(buffer):
    return None
  opcode = buffer[pos]
  if opcode == ord("G") and pos + 9 <= len(buffer):  # BINFLOAT
    return float(struct.unpack(">d", buffer[pos + 1 : pos + 9])[0])
  if opcode == ord("K") and pos + 2 <= len(buffer):  # BININT1
    return float(buffer[pos + 1])
  if opcode == ord("M") and pos + 3 <= len(buffer):  # BININT2
    return float(int.from_bytes(buffer[pos + 1 : pos + 3], "little"))
  if opcode == ord("J") and pos + 5 <= len(buffer):  # BININT
    return float(int.from_bytes(buffer[pos + 1 : pos + 5], "little", signed=True))
  if opcode == 0x88:  # NEWTRUE
    return 1.0
  if opcode in (0x89, ord("N")):  # NEWFALSE / NONE
    return 0.0
  return None


def _read_artifact(job: tuple[str, dict[str, Any] | None]) -> LoadedArtifact:
  """Stream-scan a checkpoint without materializing embedded screenshots."""
  raw_path, expected = job
  path = Path(raw_path)
  task_template = path.name.removesuffix(".pkl.gz").removesuffix("_0")
  targets: dict[str, bytes] = {
      "task_template": task_template.encode("utf-8"),
      "catbench_condition": b"c2_g",
  }
  if expected is not None:
    for label, field in (
        ("goal", "goal"),
        ("semantic_goal", "semantic_goal"),
        ("semantic_task_id", "semantic_task_id"),
        ("semantic_goal_sha256", "semantic_goal_sha256"),
        ("semantic_parameter_sha256", "semantic_parameter_sha256"),
        ("task_breakdown_text", "breakdown_text"),
        ("metadata.key", "key"),
        ("metadata.goal_sha256", "goal_sha256"),
        ("metadata.plan_sha256", "plan_sha256"),
        ("metadata.plan_key", "plan_key"),
        ("metadata.generator_provider", "generator_provider"),
    ):
      value = expected.get(field)
      if isinstance(value, str) and value:
        targets[label] = value.encode("utf-8")
  missing_targets = set(targets)
  success: float | None = None
  skipped_marker: str | None = None
  success_key = b"\x8c\ris_successful"
  tail_size = max([256, *(len(value) + 64 for value in targets.values())])
  try:
    tail = b""
    with gzip.open(path, "rb") as stream:
      while True:
        chunk = stream.read(1 << 20)
        if not chunk:
          break
        combined = tail + chunk
        for label in tuple(missing_targets):
          if targets[label] in combined:
            missing_targets.remove(label)
        if skipped_marker is None:
          for marker in SKIP_EXCEPTION_MARKERS:
            if marker.encode("utf-8") in combined:
              skipped_marker = marker
              break
        if success is None:
          index = combined.find(success_key)
          if index >= 0:
            success = _parse_pickle_scalar_after_marker(
                combined, index + len(success_key)
            )
        tail = combined[-tail_size:]
    if success is None:
      return LoadedArtifact(raw_path, None, "could not decode is_successful")
    metadata = {}
    if expected is not None:
      metadata = {
          "goal_sha256": expected.get("goal_sha256"),
          "plan_sha256": expected.get("plan_sha256"),
          "plan_key": expected.get("plan_key"),
          "generator_provider": expected.get("generator_provider"),
      }
    compact = {
        "goal": expected.get("goal") if expected else None,
        "semantic_goal": expected.get("semantic_goal") if expected else None,
        "semantic_goal_sha256": (
            expected.get("semantic_goal_sha256") if expected else None
        ),
        "semantic_parameter_sha256": (
            expected.get("semantic_parameter_sha256") if expected else None
        ),
        "semantic_task_id": expected.get("semantic_task_id") if expected else None,
        "task_template": task_template,
        "instance_id": 0,
        "task_breakdown_text": expected.get("breakdown_text") if expected else None,
        "task_breakdown_metadata": metadata,
        "catbench_condition": "c2_g" if "catbench_condition" not in missing_targets else None,
        "is_successful": success,
        "exception_info": skipped_marker,
        "_identity_mismatches": sorted(missing_targets),
    }
    return LoadedArtifact(raw_path, compact, None)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return LoadedArtifact(raw_path, None, f"{type(exc).__name__}: {exc}")


def _parse_model_roots(values: list[str]) -> collections.OrderedDict[str, list[Path]]:
  result: collections.OrderedDict[str, list[Path]] = collections.OrderedDict()
  for value in values:
    if "=" not in value:
      raise ValueError(f"--model must be MODEL=ROOT, got: {value!r}")
    model, raw_root = value.split("=", 1)
    model = model.strip()
    root = Path(raw_root).expanduser().resolve()
    if not model or not raw_root:
      raise ValueError(f"--model must be MODEL=ROOT, got: {value!r}")
    result.setdefault(model, []).append(root)
  return result


def _load_expected_plans(plan_json: Path) -> dict[str, dict[str, Any]]:
  payload = json.loads(plan_json.read_text(encoding="utf-8"))
  rows = payload.get("breakdowns", [])
  expected: dict[str, dict[str, Any]] = {}
  for row in rows:
    if not isinstance(row, dict) or int(row.get("instance_id", -1)) != 0:
      continue
    task_template = str(row.get("task_template") or "")
    if not task_template:
      raise ValueError("plan JSON contains an instance-0 row without task_template")
    if task_template in expected:
      raise ValueError(f"duplicate instance-0 plan for {task_template}")
    expected[task_template] = row
  return expected


def _is_skipped(episode: dict[str, Any]) -> bool:
  info = episode.get("exception_info") or episode.get("EXCEPTION_INFO") or ""
  return isinstance(info, str) and any(marker in info for marker in SKIP_EXCEPTION_MARKERS)


def _identity_mismatches(
    episode: dict[str, Any], expected: dict[str, Any]
) -> list[str]:
  streaming_mismatches = episode.get("_identity_mismatches")
  if isinstance(streaming_mismatches, list):
    return [str(item) for item in streaming_mismatches]
  mismatches: list[str] = []
  metadata = episode.get("task_breakdown_metadata")
  if not isinstance(metadata, dict):
    metadata = {}

  exact_fields = (
      "goal",
      "semantic_goal",
      "semantic_goal_sha256",
      "semantic_parameter_sha256",
  )
  for field in exact_fields:
    if episode.get(field) != expected.get(field):
      mismatches.append(field)

  if int(episode.get("instance_id", -1)) != 0:
    mismatches.append("instance_id")
  if episode.get("semantic_task_id") != expected.get("semantic_task_id"):
    mismatches.append("semantic_task_id")
  if episode.get("task_breakdown_text") != expected.get("breakdown_text"):
    mismatches.append("task_breakdown_text")
  if metadata.get("goal_sha256") != expected.get("goal_sha256"):
    mismatches.append("metadata.goal_sha256")
  if metadata.get("plan_sha256") != expected.get("plan_sha256"):
    mismatches.append("metadata.plan_sha256")
  if metadata.get("plan_key") != expected.get("plan_key"):
    mismatches.append("metadata.plan_key")
  if metadata.get("generator_provider") != expected.get("generator_provider"):
    mismatches.append("metadata.generator_provider")
  goal = episode.get("goal")
  if not isinstance(goal, str) or _sha256_text(goal) != expected.get("goal_sha256"):
    mismatches.append("computed_goal_sha256")
  if episode.get("catbench_condition") != "c2_g":
    mismatches.append("catbench_condition")
  return sorted(set(mismatches))


def _format_score(successes: int, valid: int) -> str:
  if not valid:
    return "--"
  return f"{successes}/{valid} ({100.0 * successes / valid:.1f}%)"


def _artifact_digest(rows: dict[str, dict[str, Any]]) -> str:
  payload = []
  for task_template in sorted(rows):
    episode = rows[task_template]["episode"]
    metadata = episode.get("task_breakdown_metadata") or {}
    payload.append(
        "|".join(
            (
                task_template,
                str(episode.get("semantic_goal_sha256") or ""),
                str(episode.get("semantic_parameter_sha256") or ""),
                str(metadata.get("plan_sha256") or ""),
            )
        )
    )
  return _sha256_text("\n".join(payload))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--plan-json", required=True)
  parser.add_argument(
      "--model",
      action="append",
      required=True,
      metavar="MODEL=ROOT",
      help="Repeat for disjoint shards of the same model.",
  )
  parser.add_argument("--out", required=True, help="Markdown output path.")
  parser.add_argument("--workers", type=int, default=8)
  parser.add_argument(
      "--expected-cells",
      type=int,
      default=230,
      help="Fail if the plan or any model does not contain this many cells.",
  )
  args = parser.parse_args()

  plan_json = Path(args.plan_json).expanduser().resolve()
  out_path = Path(args.out).expanduser().resolve()
  model_roots = _parse_model_roots(args.model)
  expected = _load_expected_plans(plan_json)
  expected_tasks = set(expected)
  expected_by_category = collections.Counter(
      _category_from_template(task_template) or "unknown"
      for task_template in expected_tasks
  )

  files_by_model: dict[str, list[str]] = {}
  missing_roots: list[str] = []
  for model, roots in model_roots.items():
    paths: set[str] = set()
    for root in roots:
      if not root.exists():
        missing_roots.append(f"{model}: {root}")
        continue
      paths.update(str(path.resolve()) for path in root.rglob("*_0.pkl.gz"))
    files_by_model[model] = sorted(paths)

  load_jobs = []
  for paths in files_by_model.values():
    for path in paths:
      task_template = Path(path).name.removesuffix(".pkl.gz").removesuffix("_0")
      load_jobs.append((path, expected.get(task_template)))
  loaded_by_path: dict[str, LoadedArtifact] = {}
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max(1, args.workers)
  ) as executor:
    for loaded in executor.map(_read_artifact, load_jobs):
      loaded_by_path[loaded.path] = loaded

  audits: dict[str, dict[str, Any]] = {}
  for model, paths in files_by_model.items():
    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    read_errors: list[LoadedArtifact] = []
    for path in paths:
      loaded = loaded_by_path[path]
      if loaded.error or loaded.episode is None:
        read_errors.append(loaded)
        continue
      task_template = str(loaded.episode.get("task_template") or "")
      category, app = _category_and_app(Path(path), task_template)
      by_task[task_template].append(
          {
              "path": path,
              "episode": loaded.episode,
              "category": category,
              "app": app,
          }
      )

    duplicates = {key: rows for key, rows in by_task.items() if len(rows) > 1}
    unique_rows = {key: rows[0] for key, rows in by_task.items() if len(rows) == 1}
    unknown = sorted(set(unique_rows) - expected_tasks)
    missing = sorted(expected_tasks - set(unique_rows))
    identity_mismatches: list[tuple[str, list[str], str]] = []
    skipped: list[str] = []
    stats: dict[str, dict[str, int]] = {
        category: {"present": 0, "valid": 0, "success": 0}
        for category in CATEGORIES
    }
    for task_template, row in unique_rows.items():
      category = row["category"]
      episode = row["episode"]
      if category in stats:
        stats[category]["present"] += 1
      if task_template in expected:
        mismatches = _identity_mismatches(episode, expected[task_template])
        if mismatches:
          identity_mismatches.append((task_template, mismatches, row["path"]))
      if _is_skipped(episode):
        skipped.append(task_template)
        continue
      if category in stats:
        stats[category]["valid"] += 1
        try:
          success = float(episode.get("is_successful") or 0.0) >= 0.5
        except (TypeError, ValueError):
          success = False
        stats[category]["success"] += int(success)

    clean_expected_rows = {
        key: row for key, row in unique_rows.items() if key in expected
    }
    total_present = sum(bucket["present"] for bucket in stats.values())
    complete = (
        len(expected) == args.expected_cells
        and len(unique_rows) == args.expected_cells
        and total_present == args.expected_cells
        and not read_errors
        and not duplicates
        and not unknown
        and not missing
        and not identity_mismatches
        and not skipped
    )
    audits[model] = {
        "stats": stats,
        "total_present": total_present,
        "read_errors": read_errors,
        "duplicates": duplicates,
        "unknown": unknown,
        "missing": missing,
        "identity_mismatches": identity_mismatches,
        "skipped": skipped,
        "digest": _artifact_digest(clean_expected_rows),
        "complete": complete,
    }

  lines = [
      "# C2 suffix-0 (K=1) cohort audit",
      "",
      f"- Frozen Qwen plan: `{plan_json}`",
      f"- Expected instance-0 cells: **{len(expected)}**",
      "- Selection rule: only checkpoint filenames ending in `_0.pkl.gz`.",
      "- A model is COMPLETE only with the exact frozen roster, matching goal/plan "
      "hashes, no duplicates/read errors, and no environment-skipped episode.",
      "",
      "## Coverage",
      "",
      "| Model | SMS | Files | Maps | Contacts | Clock | Total | Missing | "
      "Duplicates | Identity errors | Skipped | Status |",
      "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|",
  ]
  for model, audit in audits.items():
    stats = audit["stats"]
    cells = [
        f"{stats[category]['present']}/{expected_by_category[category]}"
        for category in CATEGORIES
    ]
    lines.append(
        "| "
        + " | ".join(
            [
                model,
                *cells,
                f"{audit['total_present']}/{len(expected)}",
                str(len(audit["missing"])),
                str(len(audit["duplicates"])),
                str(len(audit["identity_mismatches"])),
                str(len(audit["skipped"])),
                "**COMPLETE**" if audit["complete"] else "INCOMPLETE",
            ]
        )
        + " |"
    )

  lines.extend(
      [
          "",
          "## Success rates",
          "",
          "Rates are `successful/valid`; environment-skipped episodes are excluded "
          "and separately flagged above.",
          "",
          "| Model | SMS | Files | Maps | Contacts | Clock | Overall |",
          "|:--|--:|--:|--:|--:|--:|--:|",
      ]
  )
  for model, audit in audits.items():
    stats = audit["stats"]
    scores = [
        _format_score(stats[category]["success"], stats[category]["valid"])
        for category in CATEGORIES
    ]
    total_success = sum(stats[category]["success"] for category in CATEGORIES)
    total_valid = sum(stats[category]["valid"] for category in CATEGORIES)
    lines.append(
        "| "
        + " | ".join([model, *scores, _format_score(total_success, total_valid)])
        + " |"
    )

  lines.extend(["", "## Frozen-identity digest", ""])
  for model, audit in audits.items():
    lines.append(f"- {model}: `{audit['digest']}`")
  if len({audit["digest"] for audit in audits.values() if audit["complete"]}) > 1:
    lines.append("- **ERROR:** complete models do not share one identity digest.")
    for audit in audits.values():
      audit["complete"] = False

  diagnostics: list[str] = []
  diagnostics.extend(f"Missing root — {item}" for item in missing_roots)
  for model, audit in audits.items():
    if audit["missing"]:
      diagnostics.append(f"{model} missing: {', '.join(audit['missing'])}")
    for task_template, rows in audit["duplicates"].items():
      diagnostics.append(
          f"{model} duplicate {task_template}: "
          + ", ".join(row["path"] for row in rows)
      )
    for loaded in audit["read_errors"]:
      diagnostics.append(f"{model} read error {loaded.path}: {loaded.error}")
    for task_template, fields, path in audit["identity_mismatches"]:
      diagnostics.append(
          f"{model} identity mismatch {task_template} "
          f"({', '.join(fields)}): {path}"
      )
    if audit["unknown"]:
      diagnostics.append(f"{model} unknown tasks: {', '.join(audit['unknown'])}")
    if audit["skipped"]:
      diagnostics.append(f"{model} skipped: {', '.join(audit['skipped'])}")

  lines.extend(["", "## Diagnostics", ""])
  if diagnostics:
    lines.extend(f"- {item}" for item in diagnostics)
  else:
    lines.append("- None.")

  all_complete = (
      not missing_roots
      and len(expected) == args.expected_cells
      and all(audit["complete"] for audit in audits.values())
  )
  lines.extend(
      [
          "",
          f"**Cohort status: {'COMPLETE' if all_complete else 'INCOMPLETE'}**",
          "",
      ]
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text("\n".join(lines), encoding="utf-8")
  print(out_path)
  print(f"cohort_status={'COMPLETE' if all_complete else 'INCOMPLETE'}")
  return 0 if all_complete else 2


if __name__ == "__main__":
  raise SystemExit(main())
