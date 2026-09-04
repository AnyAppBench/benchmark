#!/usr/bin/env python3
"""Write legacy-manifest paper-style tables from a live Markdown report.

This path does not implement the frozen K=3 whole-triplet selection contract;
primary C1 app-level tables must use ``report_c1_app_level_frozen.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import collections
import datetime as dt
import gzip
import json
import math
import os
import re
import statistics
import struct
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "markdown_all_models.md"
DEFAULT_OUT = ROOT / "markdown_paper_tables_5cat.md"
DEFAULT_TASK_CACHE = ROOT / ".cache" / "catbench_paper_task_results_5cat.json"
DEFAULT_MODEL_CONFIG = ROOT / "benchmark" / "configs" / "catbench_5cat_models.json"

EXCLUDED_MODELS = {"MobileRL-9B"}

CATEGORY_LABELS = {
    "sms": "SMS",
    "files": "File Manager",
    "maps": "Maps",
    "contacts": "Contacts",
    "clock": "Clock",
}

CATEGORY_PREFIXES = {
    "SMS": "Sms",
    "File Manager": "Files",
    "Maps": "Maps",
    "Contacts": "Contacts",
    "Clock": "Clock",
}


CATEGORY_SPECS = collections.OrderedDict(
    [
        (
            "SMS",
            {
                "label": "SMS",
                "tasks": [
                    ("Send", True),
                    ("Reply", True),
                    ("ReplyMostRecent", True),
                    ("Resend", True),
                    ("SendToContact", False),
                    ("SendReceivedAddress", True),
                    ("CreateDraftMessage", False),
                    ("EditDraftMessage", False),
                    ("DeleteConversation", False),
                    ("ForwardMessage", False),
                ],
                "aw_apps": ["Simple SMS Messenger"],
                "new_apps": ["Fossify Messages", "QUIK SMS", "Messages"],
            },
        ),
        (
            "File Manager",
            {
                "label": "File Manager",
                "tasks": [
                    ("CreateFolder", False),
                    ("RenameFile", False),
                    ("DeleteFile", True),
                    ("MoveFile", True),
                    ("SaveCopyOfFile", True),
                    ("SearchFile", False),
                    ("CompressFiles", False),
                    ("ExtractArchive", False),
                    ("ViewFileInfo", False),
                    ("ShareFile", False),
                ],
                "aw_apps": ["Material Files"],
                "new_apps": [
                    "Amaze File Manager",
                    "Fossify File Manager",
                    "Total Commander",
                    "X-plore File Manager",
                ],
            },
        ),
        (
            "Maps",
            {
                "label": "Maps",
                "tasks": [
                    ("SearchPlace", False),
                    ("AddFavorite", True),
                    ("RemoveFavorite", False),
                    ("AddMarker", True),
                    ("DeleteMarker", False),
                    ("RecordTrack", True),
                    ("GetDirections", False),
                    ("SearchNearbyPlace", False),
                    ("ExportLocation", False),
                    ("ShareLocation", False),
                ],
                "aw_apps": ["OsmAnd~"],
                "new_apps": ["Organic Maps", "CoMaps"],
            },
        ),
        (
            "Contacts",
            {
                "label": "Contacts",
                "tasks": [
                    ("AddContact", True),
                    ("NewContactDraft", True),
                    ("EditContact", False),
                    ("SearchContact", False),
                    ("ViewContactDetails", False),
                    ("AddFavoriteContact", False),
                    ("RemoveFavoriteContact", False),
                    ("DeleteContact", False),
                    ("CallContact", False),
                    ("MessageContact", False),
                ],
                "aw_apps": ["Google Contacts"],
                "new_apps": [
                    "Fossify Contacts",
                    "Connect You",
                    "Simple Contacts Pro SE",
                    "Right Contact",
                ],
            },
        ),
        (
            "Clock",
            {
                "label": "Clock",
                "tasks": [
                    ("CreateAlarm", False),
                    ("EditAlarm", False),
                    ("EnableAlarm", False),
                    ("DeleteAlarm", False),
                    ("CreateTimer", True),
                    ("StartTimer", False),
                    ("StopwatchRunning", True),
                    ("PauseStopwatch", True),
                    ("StopwatchReset", False),
                    ("AddWorldClock", False),
                ],
                "aw_apps": ["Google Clock"],
                "new_apps": [
                    "Clock",
                    "Simple Clock",
                    "Clock You",
                    "Chrono",
                    "Fossify Clock",
                ],
            },
        ),
    ]
)

FAMILY_ORDER = ["GUI-Spec.", "General", "Closed", "Hybrid", "Agent"]


def _split_md_row(line: str) -> list[str]:
  return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_pair(value: str) -> tuple[int, int]:
  match = re.search(r"(\d+)\s*/\s*(\d+)", value)
  if not match:
    return (0, 0)
  return (int(match.group(1)), int(match.group(2)))


def _section_lines(text: str, heading: str) -> list[str]:
  marker = f"## {heading}"
  start = text.find(marker)
  if start < 0:
    return []
  rest = text[start + len(marker):]
  end_match = re.search(r"\n## ", rest)
  if end_match:
    rest = rest[:end_match.start()]
  return rest.splitlines()


def _parse_app_rows(source: Path) -> list[dict[str, object]]:
  lines = _section_lines(source.read_text(encoding="utf-8"), "App-Level Progress")
  rows: list[dict[str, object]] = []
  for line in lines:
    if not line.startswith("|") or line.startswith("|---"):
      continue
    cells = _split_md_row(line)
    if len(cells) < 10 or cells[0] == "Family":
      continue
    if cells[1] in EXCLUDED_MODELS:
      continue
    success, scheduled = _parse_pair(cells[6])
    completed, completed_scheduled = _parse_pair(cells[7])
    rows.append(
        {
            "family": cells[0],
            "model": cells[1],
            "category": cells[2],
            "app_type": cells[3],
            "app": cells[4],
            "success": success,
            "scheduled": scheduled,
            "completed": completed,
            "completed_scheduled": completed_scheduled,
        }
    )
  return rows


def _load_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as handle:
    return json.load(handle)


def _manifest_from_source(source: Path) -> Path:
  text = source.read_text(encoding="utf-8")
  match = re.search(r"^Manifest:\s*`([^`]+)`", text, flags=re.MULTILINE)
  if not match:
    raise ValueError(f"Could not find Manifest line in {source}")
  return Path(match.group(1)).expanduser().resolve()


def _task_names(job: dict[str, Any]) -> list[str]:
  for arg in job.get("command", []):
    if isinstance(arg, str) and arg.startswith("--tasks="):
      return [name for name in arg.removeprefix("--tasks=").split(",") if name]
  return []


def _checkpoint_paths_for_output(output_path: Path) -> list[Path]:
  if not output_path.exists():
    return []
  run_dirs = [
      path for path in output_path.iterdir()
      if path.is_dir() and path.name.startswith("run_")
  ]
  if not run_dirs:
    return sorted(output_path.rglob("*.pkl.gz"))

  def run_dir_key(path: Path) -> tuple[int, int, str]:
    try:
      checkpoint_count = sum(1 for _ in path.rglob("*.pkl.gz"))
      mtime = path.stat().st_mtime_ns
    except OSError:
      return (0, 0, path.name)
    return (checkpoint_count, mtime, path.name)

  best_run_dir = max(run_dirs, key=run_dir_key)
  return sorted(best_run_dir.rglob("*.pkl.gz"))


def _checkpoint_task_name(path: Path) -> str:
  name = path.name.removesuffix(".pkl.gz")
  return re.sub(r"_\d+$", "", name)


def _load_task_cache(path: Path) -> dict[str, Any]:
  if not path.exists():
    return {"version": 1, "items": {}}
  try:
    payload = _load_json(path)
  except (OSError, json.JSONDecodeError):
    return {"version": 1, "items": {}}
  if not isinstance(payload, dict):
    return {"version": 1, "items": {}}
  payload.setdefault("version", 1)
  payload.setdefault("items", {})
  return payload


def _save_task_cache(path: Path, cache: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _checkpoint_tail(path: Path, keep_bytes: int = 131_072) -> bytes:
  tail = bytearray()
  with gzip.open(path, "rb") as handle:
    while True:
      chunk = handle.read(1024 * 1024)
      if not chunk:
        break
      tail.extend(chunk)
      if len(tail) > keep_bytes:
        del tail[:-keep_bytes]
  return bytes(tail)


def _scan_checkpoint_status(path: Path) -> tuple[bool, bool]:
  tail = _checkpoint_tail(path)
  key = b"is_successful"
  idxs: list[int] = []
  start = 0
  while True:
    idx = tail.find(key, start)
    if idx < 0:
      break
    idxs.append(idx)
    start = idx + 1
  if not idxs:
    raise ValueError(f"Could not find is_successful in {path}")
  success: bool | None = None
  for idx in idxs:
    value_pos = tail.find(b"\x47", idx + len(key), idx + len(key) + 16)
    if value_pos >= 0 and value_pos + 9 <= len(tail):
      success_value = struct.unpack(">d", tail[value_pos + 1:value_pos + 9])[0]
      success = (not math.isnan(success_value)) and success_value >= 0.5
      break
    true_pos = tail.find(b"\x88", idx + len(key), idx + len(key) + 16)
    false_pos = tail.find(b"\x89", idx + len(key), idx + len(key) + 16)
    if true_pos >= 0 or false_pos >= 0:
      success = true_pos >= 0 and (false_pos < 0 or true_pos < false_pos)
      break
  if success is None:
    raise ValueError(f"Could not parse is_successful value in {path}")
  skipped = b"[skipped_uninstalled]" in tail
  return success, skipped


def _checkpoint_status(path: Path, cache: dict[str, Any]) -> tuple[bool, bool]:
  stat = path.stat()
  key = str(path)
  items = cache.setdefault("items", {})
  cached = items.get(key)
  if (
      isinstance(cached, dict)
      and cached.get("mtime_ns") == stat.st_mtime_ns
      and cached.get("size") == stat.st_size
  ):
    return bool(cached.get("success")), bool(cached.get("skipped"))
  success, skipped = _scan_checkpoint_status(path)
  items[key] = {
      "mtime_ns": stat.st_mtime_ns,
      "size": stat.st_size,
      "success": success,
      "skipped": skipped,
  }
  return success, skipped


def _cached_checkpoint_status(path: Path, cache: dict[str, Any]) -> tuple[bool, bool] | None:
  try:
    stat = path.stat()
  except OSError:
    return None
  cached = cache.setdefault("items", {}).get(str(path))
  if (
      isinstance(cached, dict)
      and cached.get("mtime_ns") == stat.st_mtime_ns
      and cached.get("size") == stat.st_size
  ):
    return bool(cached.get("success")), bool(cached.get("skipped"))
  return None


def _scan_checkpoint_cache_item(path_text: str) -> tuple[str, dict[str, object]]:
  path = Path(path_text)
  stat = path.stat()
  success, skipped = _scan_checkpoint_status(path)
  return path_text, {
      "mtime_ns": stat.st_mtime_ns,
      "size": stat.st_size,
      "success": success,
      "skipped": skipped,
  }


def _canonical_task_name(category: str, full_task: str) -> str:
  prefix = CATEGORY_PREFIXES.get(category, "")
  inner = full_task.removeprefix(prefix)
  template_names = sorted(
      [name for name, _ in CATEGORY_SPECS[category]["tasks"]],
      key=len,
      reverse=True,
  )
  for template_name in template_names:
    if inner == template_name or inner.startswith(f"{template_name}For"):
      return template_name
  raise ValueError(f"Could not map task {full_task!r} to a {category} template")


def _task_origin_lookup() -> dict[tuple[str, str], bool]:
  out: dict[tuple[str, str], bool] = {}
  for category, spec in CATEGORY_SPECS.items():
    for task_name, is_aw in spec["tasks"]:
      out[(category, task_name)] = is_aw
  return out


def _load_task_records(
    manifest: Path,
    rows: list[dict[str, object]],
    cache_path: Path,
) -> list[dict[str, object]]:
  manifest_payload = _load_json(manifest)
  row_by_key = {
      (str(row["model"]), str(row["category"]), str(row["app"])): row
      for row in rows
  }
  task_origin = _task_origin_lookup()
  cache = _load_task_cache(cache_path)
  record_specs: list[dict[str, object]] = []
  pending_paths: set[str] = set()
  missing: list[str] = []
  for job in manifest_payload.get("jobs", []):
    category = CATEGORY_LABELS.get(str(job.get("category", "")))
    if not category:
      continue
    model = str(job.get("model_name", ""))
    app_name = str(job.get("app_name", ""))
    row = row_by_key.get((model, category, app_name))
    if not row:
      continue
    output_path = Path(str(job.get("output_path", ""))).expanduser()
    expected = set(_task_names(job))
    paths_by_task = {
        _checkpoint_task_name(path): path
        for path in _checkpoint_paths_for_output(output_path)
    }
    for full_task in sorted(expected):
      checkpoint_path = paths_by_task.get(full_task)
      if checkpoint_path is None:
        missing.append(f"{model}/{category}/{app_name}/{full_task}")
        continue
      canonical = _canonical_task_name(category, full_task)
      path_text = str(checkpoint_path)
      if _cached_checkpoint_status(checkpoint_path, cache) is None:
        pending_paths.add(path_text)
      record_specs.append(
          {
              "model": model,
              "family": str(row["family"]),
              "category": category,
              "app": app_name,
              "app_type": str(row["app_type"]),
              "task": full_task,
              "template": canonical,
              "is_aw": task_origin[(category, canonical)],
              "checkpoint": path_text,
          }
      )
  if missing:
    sample = "; ".join(missing[:5])
    raise RuntimeError(
        f"Missing {len(missing)} expected checkpoints; first missing: {sample}"
    )
  if pending_paths:
    max_workers = min(len(pending_paths), int(os.environ.get("CATBENCH_TABLE_SCAN_WORKERS", "12")))
    max_workers = max(1, max_workers)
    print(
        f"Scanning {len(pending_paths)} checkpoint metadata files with {max_workers} workers...",
        file=sys.stderr,
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
      futures = {
          executor.submit(_scan_checkpoint_cache_item, path_text): path_text
          for path_text in sorted(pending_paths)
      }
      for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
        path_text, item = future.result()
        cache.setdefault("items", {})[path_text] = item
        if done % 100 == 0 or done == len(futures):
          print(
              f"  scanned {done}/{len(futures)} checkpoint metadata files",
              file=sys.stderr,
              flush=True,
          )
          _save_task_cache(cache_path, cache)
  _save_task_cache(cache_path, cache)
  records: list[dict[str, object]] = []
  for spec in record_specs:
    cached = cache["items"][str(spec["checkpoint"])]
    records.append(
        {
            **spec,
            "success": bool(cached["success"]),
            "skipped": bool(cached["skipped"]),
        }
    )
  return records


def _verify_task_records(
    records: list[dict[str, object]],
    rows: list[dict[str, object]],
) -> None:
  by_app: dict[tuple[str, str, str], collections.Counter[str]] = collections.defaultdict(collections.Counter)
  for record in records:
    if record["skipped"]:
      continue
    key = (str(record["model"]), str(record["category"]), str(record["app"]))
    by_app[key]["scheduled"] += 1
    if record["success"]:
      by_app[key]["success"] += 1
  mismatches = []
  for row in rows:
    key = (str(row["model"]), str(row["category"]), str(row["app"]))
    stats = by_app.get(key, collections.Counter())
    expected = (int(row["success"]), int(row["scheduled"]))
    actual = (int(stats["success"]), int(stats["scheduled"]))
    if actual != expected:
      mismatches.append((key, expected, actual))
  if mismatches:
    details = "; ".join(
        f"{key}: report={expected}, checkpoints={actual}"
        for key, expected, actual in mismatches[:5]
    )
    raise RuntimeError(
        f"Checkpoint task metadata does not match markdown app totals "
        f"({len(mismatches)} mismatches): {details}"
    )


def _model_order(rows: list[dict[str, object]]) -> list[str]:
  order: list[str] = []
  for row in rows:
    model = str(row["model"])
    if model not in order:
      order.append(model)
  return order


def _model_roster(
    rows: list[dict[str, object]],
    model_config: Path,
) -> collections.OrderedDict[str, list[str]]:
  grouped: collections.OrderedDict[str, list[str]] = collections.OrderedDict(
      (family, []) for family in FAMILY_ORDER
  )
  seen: set[str] = set()
  if model_config.exists():
    payload = _load_json(model_config)
    for model_info in payload.get("models", []):
      if not isinstance(model_info, dict):
        continue
      model = str(model_info.get("name", ""))
      if not model or model in EXCLUDED_MODELS:
        continue
      family = str(model_info.get("group", "Other"))
      grouped.setdefault(family, [])
      grouped[family].append(model)
      seen.add(model)
  row_families = _family_for_model(rows)
  for model in _model_order(rows):
    if model in seen or model in EXCLUDED_MODELS:
      continue
    family = row_families.get(model, "Other")
    grouped.setdefault(family, [])
    grouped[family].append(model)
  return collections.OrderedDict((family, models) for family, models in grouped.items() if models)


def _roster_model_order(
    roster: collections.OrderedDict[str, list[str]],
) -> list[str]:
  return [model for models in roster.values() for model in models]


def _family_for_model(rows: list[dict[str, object]]) -> dict[str, str]:
  return {str(row["model"]): str(row["family"]) for row in rows}


def _group_models(rows: list[dict[str, object]]) -> collections.OrderedDict[str, list[str]]:
  families = _family_for_model(rows)
  seen = _model_order(rows)
  grouped: collections.OrderedDict[str, list[str]] = collections.OrderedDict()
  for family in FAMILY_ORDER:
    models = [model for model in seen if families.get(model) == family]
    if models:
      grouped[family] = models
  for model in seen:
    family = families.get(model, "Other")
    if family not in grouped:
      grouped[family] = []
    if model not in grouped[family]:
      grouped[family].append(model)
  return grouped


def _latex_escape(text: str) -> str:
  replacements = {
      "\\": r"\textbackslash{}",
      "&": r"\&",
      "%": r"\%",
      "$": r"\$",
      "#": r"\#",
      "_": r"\_",
      "{": r"\{",
      "}": r"\}",
      "~": r"\texttildelow{}",
      "^": r"\textasciicircum{}",
  }
  return "".join(replacements.get(ch, ch) for ch in text)


def _latex_model(text: str) -> str:
  escaped = _latex_escape(text)
  return escaped.replace("-dagger", r"$^\dagger$")


def _task_tex(task: tuple[str, bool]) -> str:
  name, is_aw = task
  escaped = _latex_escape(name)
  return rf"\awtask{{{escaped}}}" if is_aw else escaped


def _makecell(items: list[str], per_line: int = 3) -> str:
  lines = [
      ", ".join(items[idx:idx + per_line])
      for idx in range(0, len(items), per_line)
  ]
  return r"\makecell[l]{" + r",\\".join(lines) + "}"


def _pct(success: int, scheduled: int) -> float:
  return 100.0 * success / scheduled if scheduled else 0.0


def _signed(value: float) -> str:
  return f"{value:+.1f}"


def _app_index(rows: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
  out = {}
  for row in rows:
    out[(str(row["model"]), str(row["category"]), str(row["app"]))] = row
  return out


def _cell_for_app(
    app_rows: dict[tuple[str, str, str], dict[str, object]],
    model: str,
    category: str,
    app: str,
) -> str:
  row = app_rows.get((model, category, app))
  if not row:
    return "--"
  return f"{row['success']}/{row['scheduled']}"


def _task_stats_index(
    records: list[dict[str, object]],
) -> dict[tuple[str, str, str, bool], collections.Counter[str]]:
  out: dict[tuple[str, str, str, bool], collections.Counter[str]] = collections.defaultdict(collections.Counter)
  for record in records:
    if record["skipped"]:
      continue
    key = (
        str(record["model"]),
        str(record["category"]),
        str(record["app"]),
        bool(record["is_aw"]),
    )
    out[key]["scheduled"] += 1
    if record["success"]:
      out[key]["success"] += 1
  return out


def _cell_for_task_origin(
    task_stats: dict[tuple[str, str, str, bool], collections.Counter[str]],
    model: str,
    category: str,
    app: str,
    is_aw: bool,
) -> str:
  stats = task_stats.get((model, category, app, is_aw))
  if not stats or not stats["scheduled"]:
    return "--"
  return f"{int(stats['success'])}/{int(stats['scheduled'])}"


def _app_header_label(app: str) -> str:
  words = app.split()
  if len(words) <= 1:
    return rf"\textbf{{{_latex_escape(app)}}}"
  lines = []
  for idx in range(0, len(words), 2):
    lines.append(r"\textbf{" + _latex_escape(" ".join(words[idx:idx + 2])) + "}")
  return r"\makecell{" + r"\\".join(lines) + "}"


def _overview_table(completed_model_count: int | None = None) -> str:
  total_templates = sum(len(spec["tasks"]) for spec in CATEGORY_SPECS.values())
  aw_entries = sum(len(spec["aw_apps"]) for spec in CATEGORY_SPECS.values())
  new_entries = sum(len(spec["new_apps"]) for spec in CATEGORY_SPECS.values())
  total_entries = aw_entries + new_entries
  total_task_app_combinations = sum(
      len(spec["tasks"]) * (len(spec["aw_apps"]) + len(spec["new_apps"]))
      for spec in CATEGORY_SPECS.values()
  )
  completed_note = (
      f"{completed_model_count} completed model rows are represented in the "
      "source report."
      if completed_model_count
      else (
          f"each fully completed model row adds {total_task_app_combinations} "
          "nominal task--app attempts."
      )
  )
  lines = [
      r"\begin{table*}[t]",
      r"\centering",
      rf"\caption{{\textbf{{CATBench five-category overview.}} This completed run evaluates cross-app generalization in five functional categories. Task templates marked with superscript ``AW'' preserve AndroidWorld task intent; unmarked templates are CATBench additions. Bold app names denote the AndroidWorld-original baseline app used for comparison. Total: {total_templates} task templates, {total_entries} app entries, and {total_task_app_combinations} nominal task--app combinations per model.}}",
      r"\label{tab:catbench_5cat_overview}",
      "",
      r"\scriptsize",
      r"\setlength{\tabcolsep}{2.4pt}",
      r"\renewcommand{\arraystretch}{0.88}",
      r"\begin{adjustbox}{max totalsize={\textwidth}{0.72\textheight},center}",
      r"\begin{threeparttable}",
      r"\begin{tabular}{llcllcc}",
      r"\toprule",
      r"\textbf{Category} & \textbf{Task Templates} & \textbf{\# Tmpl.} & \textbf{AW Orig. App(s)} & \textbf{Newly Installed Apps} & \textbf{\# New} & \textbf{\# Apps} \\",
      r"\midrule",
  ]
  for spec in CATEGORY_SPECS.values():
    tasks = [_task_tex(task) for task in spec["tasks"]]
    aw_apps = [rf"\textbf{{{_latex_escape(app)}}}" for app in spec["aw_apps"]]
    new_apps = [_latex_escape(app) for app in spec["new_apps"]]
    lines.extend(
        [
            "",
            f"{_latex_escape(str(spec['label']))}",
            f"& {_makecell(tasks, 4)}",
            f"& {len(tasks)}",
            f"& {_makecell(aw_apps, 2)}",
            f"& {_makecell(new_apps, 3)}",
            f"& {len(new_apps)}",
            f"& {len(aw_apps) + len(new_apps)} " + r"\\",
            r"\midrule",
        ]
    )
  lines.extend(
      [
          r"\rowcolor{gray!15}",
          rf"\textbf{{Total}} & & \textbf{{{total_templates}}} & \textbf{{{aw_entries} AW app entries}} & \textbf{{{new_entries} newly installed apps}} & \textbf{{{new_entries}}} & \textbf{{{total_entries}}} \\",
          r"\bottomrule",
          r"\end{tabular}",
          r"\begin{tablenotes}[flushleft]",
          r"\footnotesize",
          r"\item The current result file covers the five active categories in the source report: SMS, File Manager, Maps, Contacts, and Clock.",
          rf"\item With category-specific task-template counts and {total_entries} app entries, this scope contains {total_task_app_combinations} nominal task--app combinations per model; {completed_note}",
          r"\end{tablenotes}",
          r"\end{threeparttable}",
          r"\end{adjustbox}",
          r"\end{table*}",
      ]
  )
  return "\n".join(lines)


def _app_level_table(
    category: str,
    rows: list[dict[str, object]],
    records: list[dict[str, object]],
    roster: collections.OrderedDict[str, list[str]],
) -> str:
  spec = CATEGORY_SPECS[category]
  apps = list(spec["aw_apps"]) + list(spec["new_apps"])
  task_stats = _task_stats_index(records)
  grouped = roster
  align = "ll" + (r">{\columncolor{awcol}}c>{\columncolor{catcol}}c" * len(apps))
  header_apps = []
  cmidrules = []
  col = 3
  for app in apps:
    label = _app_header_label(app)
    if app in spec["aw_apps"]:
      label = rf"\cellcolor{{awshade}}{label}"
    header_apps.append(rf"\multicolumn{{2}}{{c}}{{{label}}}")
    cmidrules.append(rf"\cmidrule(lr){{{col}-{col + 1}}}")
    col += 2
  provenance_headers = []
  for _ in apps:
    provenance_headers.extend([r"\textbf{AW}", r"\textbf{New}"])
  aw_count = sum(1 for _, is_aw in spec["tasks"] if is_aw)
  new_count = sum(1 for _, is_aw in spec["tasks"] if not is_aw)
  lines = [
      r"\begin{table*}[t]",
      r"\centering",
      rf"\caption{{\textbf{{App-level success rates for {_latex_escape(category)}.}} Each app is split by template provenance: AW-inherited task templates ({aw_count}) and newly added CATBench templates ({new_count}). Cells show successful completed tasks out of completed templates in the corrected manifest; all included rows have no skipped tasks.}}",
      rf"\label{{tab:app_level_{category.lower().replace(' ', '_')}}}",
      r"\scriptsize",
      r"\setlength{\tabcolsep}{3pt}",
      r"\renewcommand{\arraystretch}{1.08}",
      r"\begin{threeparttable}",
      r"\begin{adjustbox}{max width=\textwidth}",
      rf"\begin{{tabular}}{{{align}}}",
      r"\toprule",
      r"& & " + " & ".join(header_apps) + r" \\",
      "".join(cmidrules),
      r"\textbf{Family} & \textbf{Model} & " + " & ".join(provenance_headers) + r" \\",
      r"\midrule",
  ]
  for family, models in grouped.items():
    for idx, model in enumerate(models):
      family_cell = (
          rf"\multirow{{{len(models)}}}{{*}}{{\rotatebox{{90}}{{\scriptsize {_latex_escape(family)}}}}}"
          if idx == 0
          else ""
      )
      cells = []
      for app in apps:
        cells.append(_cell_for_task_origin(task_stats, model, category, app, True))
        cells.append(_cell_for_task_origin(task_stats, model, category, app, False))
      lines.append(f"{family_cell} & {_latex_model(model)} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
  mean_cells = []
  for app in apps:
    for is_aw in [True, False]:
      rates = []
      for model in _roster_model_order(roster):
        stats = task_stats.get((model, category, app, is_aw))
        if stats and stats["scheduled"]:
          rates.append(_pct(int(stats["success"]), int(stats["scheduled"])))
      mean_cells.append("--" if not rates else rf"{statistics.mean(rates):.1f}\%")
  lines.extend(
      [
          r"\rowcolor{appavgrow}",
          r"\multicolumn{2}{l}{\textbf{Mean success rate}} & " + " & ".join(mean_cells) + r" \\",
          r"\bottomrule",
          r"\end{tabular}",
          r"\end{adjustbox}",
          r"\begin{tablenotes}[flushleft]",
          r"\footnotesize",
          r"\item Shaded header cells mark AndroidWorld-original app baselines. AW/New subcolumns refer to template provenance, not app provenance.",
          r"\end{tablenotes}",
          r"\end{threeparttable}",
          r"\end{table*}",
      ]
  )
  return "\n".join(lines)


def _provenance_gap_rows(
    rows: list[dict[str, object]],
    records: list[dict[str, object]],
    roster: collections.OrderedDict[str, list[str]],
) -> list[dict[str, object]]:
  del rows
  grouped = roster
  out: list[dict[str, object]] = []
  for family, models in grouped.items():
    for model in models:
      row: dict[str, object] = {"family": family, "model": model}
      for provenance_key, is_aw in [("aw_templates", True), ("new_templates", False)]:
        aw_success = aw_sched = new_success = new_sched = 0
        per_aw_app: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(collections.Counter)
        per_new_app: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(collections.Counter)
        for record in records:
          if record["model"] != model or bool(record["is_aw"]) != is_aw or record["skipped"]:
            continue
          if record["app_type"] == "AW Orig.":
            aw_sched += 1
            if record["success"]:
              aw_success += 1
            app_key = (str(record["category"]), str(record["app"]))
            per_aw_app[app_key]["scheduled"] += 1
            if record["success"]:
              per_aw_app[app_key]["success"] += 1
          elif record["app_type"] == "New":
            new_sched += 1
            if record["success"]:
              new_success += 1
            app_key = (str(record["category"]), str(record["app"]))
            per_new_app[app_key]["scheduled"] += 1
            if record["success"]:
              per_new_app[app_key]["success"] += 1
        aw_app_rates = [
            _pct(int(stats["success"]), int(stats["scheduled"]))
            for stats in per_aw_app.values()
            if stats["scheduled"]
        ]
        new_app_rates = [
            _pct(int(stats["success"]), int(stats["scheduled"]))
            for stats in per_new_app.values()
            if stats["scheduled"]
        ]
        if not aw_app_rates and not new_app_rates:
          row[f"{provenance_key}_aw_rate"] = None
          row[f"{provenance_key}_new_rate"] = None
          row[f"{provenance_key}_std"] = None
          row[f"{provenance_key}_delta"] = None
        else:
          aw_rate = statistics.mean(aw_app_rates) if aw_app_rates else _pct(aw_success, aw_sched)
          new_rate = statistics.mean(new_app_rates) if new_app_rates else _pct(new_success, new_sched)
          row[f"{provenance_key}_aw_rate"] = aw_rate
          row[f"{provenance_key}_new_rate"] = new_rate
          row[f"{provenance_key}_std"] = (
              statistics.pstdev(new_app_rates) if len(new_app_rates) > 1 else 0.0
          )
          row[f"{provenance_key}_delta"] = new_rate - aw_rate
        row[f"{provenance_key}_aw_count"] = f"{aw_success}/{aw_sched}"
        row[f"{provenance_key}_new_count"] = f"{new_success}/{new_sched}"
      out.append(row)
  return out


def _gap_table(
    rows: list[dict[str, object]],
    records: list[dict[str, object]],
    roster: collections.OrderedDict[str, list[str]],
) -> str:
  grouped = collections.OrderedDict()
  for row in _provenance_gap_rows(rows, records, roster):
    grouped.setdefault(row["family"], []).append(row)
  all_rows = [row for family_rows in grouped.values() for row in family_rows]
  aw_template_count = sum(
      1 for spec in CATEGORY_SPECS.values() for _, is_aw in spec["tasks"] if is_aw
  )
  new_template_count = sum(
      1 for spec in CATEGORY_SPECS.values() for _, is_aw in spec["tasks"] if not is_aw
  )

  def avg(field: str) -> float:
    values = [float(row[field]) for row in all_rows if row.get(field) is not None]
    return statistics.mean(values) if values else 0.0

  def fmt_rate(value: object) -> str:
    return "--" if value is None else f"{float(value):.1f}"

  def fmt_delta(value: object) -> str:
    return "--" if value is None else _signed(float(value))

  lines = [
      r"\begin{table*}[t]",
      r"\centering",
      r"\footnotesize",
      r"\setlength{\tabcolsep}{3.5pt}",
      r"\renewcommand{\arraystretch}{1.05}",
      "",
      rf"\caption{{\textbf{{Cross-app generalization gap, decomposed by template provenance.}} \textbf{{(a)}} {aw_template_count} templates inherited or adapted from AndroidWorld: same task intent, different app, isolating the cross-app effect. \textbf{{(b)}} {new_template_count} newly introduced CATBench templates, confirming the gap is not only an artifact of the AW task distribution. \textbf{{AW app}}: success on AndroidWorld baseline app(s). \textbf{{New}}: average over newly installed apps in the same categories. \textbf{{Std}}: app-level variation across new apps. $\boldsymbol{{\Delta}}$: percentage-point change (negative = drop).}}",
      r"\label{tab:catbench_5cat_cross_app_gap}",
      "",
      r"\begin{adjustbox}{max width=\textwidth}",
      r"\begin{tabular}{ll>{\columncolor{awcol}}c>{\columncolor{catcol}}c>{\columncolor{stdcol}}c>{\columncolor{deltacol}}c>{\columncolor{awcol}}c>{\columncolor{catcol}}c>{\columncolor{stdcol}}c>{\columncolor{deltacol}}c}",
      r"\toprule",
      rf"& & \multicolumn{{4}}{{c}}{{\textbf{{(a) AW-inherited templates ({aw_template_count})}}}} & \multicolumn{{4}}{{c}}{{\textbf{{(b) New CATBench templates ({new_template_count})}}}} \\",
      r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
      r"\textbf{Family} & \textbf{Model} & \textbf{AW app} & \textbf{New} & \textbf{Std} & $\boldsymbol{\Delta}$ & \textbf{AW app} & \textbf{New} & \textbf{Std} & $\boldsymbol{\Delta}$ \\",
      r"\midrule",
  ]
  for family, family_rows in grouped.items():
    for idx, row in enumerate(family_rows):
      family_cell = (
          rf"\multirow{{{len(family_rows)}}}{{*}}{{\rotatebox{{90}}{{\scriptsize {_latex_escape(str(family))}}}}}"
          if idx == 0
          else ""
      )
      lines.append(
          f"{family_cell} & {_latex_model(str(row['model']))} "
          f"& {fmt_rate(row['aw_templates_aw_rate'])} "
          f"& {fmt_rate(row['aw_templates_new_rate'])} "
          f"& {fmt_rate(row['aw_templates_std'])} "
          f"& {fmt_delta(row['aw_templates_delta'])} "
          f"& {fmt_rate(row['new_templates_aw_rate'])} "
          f"& {fmt_rate(row['new_templates_new_rate'])} "
          f"& {fmt_rate(row['new_templates_std'])} "
          f"& {fmt_delta(row['new_templates_delta'])} " + r"\\"
      )
    lines.append(r"\midrule")
  lines.extend(
      [
          r"\rowcolor{avgrow}",
          rf"& \textbf{{Average}} & \textbf{{{avg('aw_templates_aw_rate'):.1f}}} & \textbf{{{avg('aw_templates_new_rate'):.1f}}} & \textbf{{{avg('aw_templates_std'):.1f}}} & \textbf{{{_signed(avg('aw_templates_delta'))}}} & \textbf{{{avg('new_templates_aw_rate'):.1f}}} & \textbf{{{avg('new_templates_new_rate'):.1f}}} & \textbf{{{avg('new_templates_std'):.1f}}} & \textbf{{{_signed(avg('new_templates_delta'))}}} \\",
          r"\bottomrule",
          r"\end{tabular}",
          r"\end{adjustbox}",
          "",
          r"\vspace{0.3em}",
          r"\begin{minipage}{\textwidth}",
          r"\scriptsize",
          r"\textit{Notes.} Success rates are percentages computed from the same completed checkpoint records that produce \texttt{markdown\_all\_models.md}; $\Delta$ is in percentage points.",
          r"\end{minipage}",
          r"\end{table*}",
      ]
  )
  return "\n".join(lines)


def _macro_block() -> str:
  return "\n".join(
      [
          r"% Suggested LaTeX helpers for the tables below:",
          r"\newcommand{\awtask}[1]{#1\textsuperscript{\tiny AW}}",
          r"\definecolor{awshade}{RGB}{235,242,255}",
          r"\definecolor{awcol}{RGB}{245,249,255}",
          r"\definecolor{catcol}{RGB}{248,248,248}",
          r"\definecolor{stdcol}{RGB}{252,247,236}",
          r"\definecolor{deltacol}{RGB}{252,241,241}",
          r"\definecolor{appavgrow}{RGB}{238,238,238}",
          r"\definecolor{avgrow}{RGB}{230,230,230}",
      ]
  )


def _write_markdown(
    source: Path,
    out: Path,
    manifest: Path,
    task_cache: Path,
    model_config: Path,
) -> None:
  rows = _parse_app_rows(source)
  roster = _model_roster(rows, model_config)
  records = _load_task_records(manifest, rows, task_cache)
  _verify_task_records(records, rows)
  now = dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S %Z")
  completed_records = [record for record in records if not record["skipped"]]
  successful_records = [record for record in completed_records if record["success"]]
  completed_models = sorted({str(row["model"]) for row in rows})
  skipped_records = [record for record in records if record["skipped"]]
  parts = [
      "# CATBench Paper-Style Tables (5-Category)",
      "",
      f"Generated: `{now}`",
      "",
      f"Source report: `{source}`",
      f"Manifest: `{manifest}`",
      f"Model roster: `{model_config}`",
      f"Task metadata cache: `{task_cache}`",
      "",
      "This Markdown contains LaTeX table blocks in the style of the paper tables. "
      f"It reflects the completed five-category source rows represented by the manifest: "
      f"`{len(completed_records)}/{len(records)}` completed task records and "
      f"`{len(skipped_records)}` skipped task records.",
      "",
      f"The AW/New template split below is computed from real checkpoint metadata: "
      f"`{len(successful_records)}/{len(completed_records)}` successful completed task records. "
      "Before writing, the generator verifies that the per-task checkpoint totals match the app-level totals in the source report.",
      "Models without completed results are kept as `--` placeholder rows so the tables can be filled in later.",
      "",
      "Note: the example you provided describes the full 10-category, 55-app CATBench overview. "
      f"The current completed source rows contain {len(completed_models)} model(s) over the five active categories: SMS, File Manager, Maps, Contacts, and Clock.",
      "",
      "## LaTeX Helper Macros",
      "",
      "```latex",
      _macro_block(),
      "```",
      "",
      "## Benchmark Overview",
      "",
      "```latex",
      _overview_table(len(completed_models)),
      "```",
      "",
      "## App-Level Results",
      "",
  ]
  for category in CATEGORY_SPECS:
    parts.extend(
        [
            f"### {category}",
            "",
            "```latex",
            _app_level_table(category, rows, records, roster),
            "```",
            "",
        ]
    )
  parts.extend(
      [
          "## Cross-App Generalization Gap",
          "",
          "```latex",
          _gap_table(rows, records, roster),
          "```",
          "",
      ]
  )
  out.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", default=str(DEFAULT_SOURCE))
  parser.add_argument("--out", default=str(DEFAULT_OUT))
  parser.add_argument("--model_config", default=str(DEFAULT_MODEL_CONFIG))
  parser.add_argument(
      "--manifest",
      default="",
      help="CATBench matrix manifest. Defaults to the Manifest line in --source.",
  )
  parser.add_argument("--task_cache", default=str(DEFAULT_TASK_CACHE))
  args = parser.parse_args()
  source = Path(args.source).expanduser().resolve()
  manifest = (
      Path(args.manifest).expanduser().resolve()
      if args.manifest
      else _manifest_from_source(source)
  )
  out = Path(args.out).expanduser().resolve()
  task_cache = Path(args.task_cache).expanduser().resolve()
  model_config = Path(args.model_config).expanduser().resolve()
  _write_markdown(source, out, manifest, task_cache, model_config)
  print(f"Wrote {Path(args.out).expanduser().resolve()}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
