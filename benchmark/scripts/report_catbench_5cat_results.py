#!/usr/bin/env python3
"""Aggregate legacy/diagnostic CATBench five-category runs.

This reporter is intentionally not valid for the frozen prospective C1
release.  Primary K=3 results must go through
``report_c1_app_level_frozen.py``, which verifies the immutable schedule,
replacement journal, complete triplets, semantic origins, and typed terminal
contract.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import io
import json
import math
import os
import pickle
import re
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

from app_generalization_profiles import get_domain_profiles  # pylint: disable=wrong-import-position


CATEGORY_ORDER = ("sms", "files", "maps", "contacts", "clock")
CONFIG_PATH = BENCHMARK_ROOT / "configs" / "catbench_5cat_models.json"
BASELINE_APP_BY_CATEGORY = {
    "sms": "sms_simple_sms_messenger",
    "files": "files_material_files",
    "maps": "maps_osmand",
    "contacts": "contacts_google_contacts",
    "clock": "clock_google_clock",
}
SCREENSHOT_FIELDS = (
    "raw_screenshot",
    "screenshot",
    "before_screenshot",
    "after_screenshot",
    "before_screenshot_with_som",
    "after_screenshot_with_som",
)
SCREENSHOT_PATH_FIELDS = (
    "agentprog_screenshots",
)
SKIP_EXCEPTION_MARKERS = (
    "[skipped_uninstalled]",
    "[skipped_environment]",
    "_EnvironmentNetworkError",
    "network/connectivity error dialog visible",
)


class LegacyReportError(ValueError):
  """Raised when a manifest is not eligible for this legacy reporter."""


def _read_pkl_gz(path: Path) -> Any:
  with path.open("rb") as handle:
    raw = handle.read()
  with gzip.open(io.BytesIO(raw), "rb") as gz:
    return pickle.load(gz)


def _load_episodes(path: Path) -> list[dict[str, Any]]:
  episodes: list[dict[str, Any]] = []
  if not path.exists():
    return episodes
  for file_path in sorted(path.rglob("*.pkl.gz")):
    try:
      payload = _read_pkl_gz(file_path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"warning: failed to read {file_path}: {exc}", file=sys.stderr)
      continue
    if isinstance(payload, list):
      episodes.extend(ep for ep in payload if isinstance(ep, dict))
  return episodes


def _is_skipped(ep: dict[str, Any]) -> bool:
  status = ep.get("catbench_episode_status")
  if status == "invalid_infrastructure":
    return True
  if status is not None and status not in {"valid_success", "valid_failure"}:
    # Unknown typed outcomes are never silently converted to task failures.
    return True
  info = ep.get("exception_info") or ep.get("EXCEPTION_INFO") or ""
  return isinstance(info, str) and any(
      marker in info for marker in SKIP_EXCEPTION_MARKERS
  )


def _rate(episodes: list[dict[str, Any]]) -> float | None:
  valid = [ep for ep in episodes if not _is_skipped(ep)]
  if not valid:
    return None
  successes = [
      1.0 if float(ep.get("is_successful") or 0.0) >= 0.5 else 0.0
      for ep in valid
  ]
  return 100.0 * sum(successes) / len(successes)


def _fmt_rate(value: float | None) -> str:
  return "--" if value is None else f"{value:.1f}"


def _safe_json(value: Any) -> Any:
  if value is None or isinstance(value, (str, int, float, bool)):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
      return None
    return value
  if isinstance(value, dict):
    return {str(k): _safe_json(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_safe_json(v) for v in value]
  return str(value)


def _episode_steps(ep: dict[str, Any]) -> list[dict[str, Any]]:
  step_data = ep.get("episode_data") or {}
  if not isinstance(step_data, dict):
    return []
  length = 0
  for value in step_data.values():
    if isinstance(value, list):
      length = max(length, len(value))
  fields = (
      "step_number",
      "prompt_system",
      "prompt_user",
      "action_prompt",
      "summary_prompt",
      "response",
      "action_output",
      "action_raw_response",
      "thought",
      "thinking",
      "action_reason",
      "reason",
      "action_desc",
      "action",
      "action_output_json",
      "tool_call",
      "summary",
      "agentprog_task_dir",
      "agentprog_workflow_path",
      "agentprog_image_dir",
      "agentprog_meta_info_dir",
      "agentprog_log_path",
      "agentprog_workflow",
      "agentprog_global_variables",
      "agentprog_answer",
  )
  steps = []
  for idx in range(length):
    row = {"step": idx + 1}
    for field in fields:
      seq = step_data.get(field)
      if isinstance(seq, list) and idx < len(seq):
        row[field] = _safe_json(seq[idx])
    steps.append(row)
  return steps


def _save_image_value(value: Any, dest: Path) -> str | None:
  if value is None:
    return None
  if isinstance(value, str):
    source = Path(value)
    if source.exists() and source.is_file():
      dest.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(source, dest)
      return str(dest)
    return None

  try:
    import numpy as np  # pylint: disable=import-outside-toplevel
    from PIL import Image  # pylint: disable=import-outside-toplevel
  except ImportError:
    return None

  try:
    if isinstance(value, np.ndarray):
      array = value
      if array.dtype != np.uint8:
        array = array.astype(np.uint8)
      dest.parent.mkdir(parents=True, exist_ok=True)
      Image.fromarray(array).save(dest)
      return str(dest)
    if hasattr(value, "save"):
      dest.parent.mkdir(parents=True, exist_ok=True)
      value.save(dest)
      return str(dest)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"warning: failed to write screenshot {dest}: {exc}", file=sys.stderr)
  return None


def _dump_episode_screenshots(
    out_root: Path,
    model: str,
    category: str,
    app_id: str,
    index: int,
    task_name: str,
    ep: dict[str, Any],
) -> dict[int, dict[str, Any]]:
  step_data = ep.get("episode_data") or {}
  if not isinstance(step_data, dict):
    return {}
  screenshot_root = (
      out_root / "screenshots" / _slug(model) / category / app_id
      / f"{index:04d}_{_slug(str(task_name))}"
  )
  refs: dict[int, dict[str, Any]] = {}

  for field in SCREENSHOT_FIELDS:
    seq = step_data.get(field)
    if not isinstance(seq, list):
      continue
    for step_idx, value in enumerate(seq):
      path = _save_image_value(
          value, screenshot_root / f"step_{step_idx + 1:03d}_{field}.png"
      )
      if path:
        refs.setdefault(step_idx, {})[field] = path

  for field in SCREENSHOT_PATH_FIELDS:
    seq = step_data.get(field)
    if not isinstance(seq, list):
      continue
    for step_idx, values in enumerate(seq):
      if not values:
        continue
      copied: list[str] = []
      if isinstance(values, str):
        values = [values]
      if isinstance(values, (list, tuple)):
        for img_idx, value in enumerate(values):
          path = _save_image_value(
              value,
              screenshot_root
              / f"step_{step_idx + 1:03d}_{field}_{img_idx:03d}.png",
          )
          if path:
            copied.append(path)
      if copied:
        refs.setdefault(step_idx, {})[field] = copied
  return refs


def _dump_prompt_traces(
    out_root: Path,
    model: str,
    category: str,
    app_id: str,
    episodes: list[dict[str, Any]],
) -> tuple[int, int]:
  trace_root = out_root / "prompts_reasoning" / _slug(model) / category / app_id
  trace_root.mkdir(parents=True, exist_ok=True)
  count = 0
  screenshot_count = 0
  for index, ep in enumerate(episodes):
    steps = _episode_steps(ep)
    task_name = ep.get("task_template") or ep.get("name") or f"episode_{index}"
    screenshot_refs = _dump_episode_screenshots(
        out_root, model, category, app_id, index, str(task_name), ep
    )
    if not steps and not screenshot_refs:
      continue
    if screenshot_refs:
      screenshot_count += 1
    for step_index, refs in screenshot_refs.items():
      while len(steps) <= step_index:
        steps.append({"step": len(steps) + 1})
      steps[step_index]["screenshots"] = refs
    payload = {
        "model": model,
        "category": category,
        "app_id": app_id,
        "task_template": task_name,
        "goal": ep.get("goal"),
        "is_successful": _safe_json(ep.get("is_successful")),
        "steps": steps,
    }
    file_name = f"{index:04d}_{_slug(str(task_name))}.json"
    with (trace_root / file_name).open("w", encoding="utf-8") as handle:
      json.dump(payload, handle, indent=2)
    count += 1
  return count, screenshot_count


def _slug(value: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _manifest_payload(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  if not isinstance(payload, dict):
    raise LegacyReportError("Manifest root must be a JSON object")
  return payload


def _jobs_from_manifest(path: Path) -> list[dict[str, Any]]:
  payload = _manifest_payload(path)
  jobs = payload.get("jobs", [])
  if not isinstance(jobs, list):
    raise LegacyReportError("Manifest jobs must be a list")
  if payload.get("analysis_eligible") is True or any(
      isinstance(job, dict) and job.get("analysis_eligible") is True
      for job in jobs
  ):
    raise LegacyReportError(
        "Analysis-eligible manifests must use "
        "report_c1_app_level_frozen.py; this legacy reporter does not verify "
        "the frozen schedule or whole K=3 triplets."
    )
  return jobs


def _model_order(config_path: Path, jobs: list[dict[str, Any]]) -> list[str]:
  names: list[str] = []
  if config_path.exists():
    with config_path.open("r", encoding="utf-8") as handle:
      payload = json.load(handle)
    for model in payload.get("models", []):
      name = model.get("name")
      if isinstance(name, str) and name and name not in names:
        names.append(name)
  for job in jobs:
    name = job.get("model_name")
    if isinstance(name, str) and name and name not in names:
      names.append(name)
  return names


def _model_groups(config_path: Path) -> dict[str, str]:
  if not config_path.exists():
    return {}
  with config_path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  return {
      model["name"]: model.get("group", "")
      for model in payload.get("models", [])
      if isinstance(model.get("name"), str)
  }


def _empty_app_stats(display_name: str) -> dict[str, Any]:
  return {
      "app_name": display_name,
      "success_rate": None,
      "num_episodes": 0,
      "num_skipped": 0,
      "output_path": "",
  }


def _summarize(
    manifest: Path,
    out_dir: Path,
    model_config: Path,
    dump_prompt_traces: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  profiles = get_domain_profiles()
  app_labels = {
      category: {app.app_id: app.display_name for app in profiles[category].apps}
      for category in CATEGORY_ORDER
  }
  jobs = _jobs_from_manifest(manifest)
  model_names = _model_order(model_config, jobs)
  per_model: dict[str, dict[str, dict[str, dict[str, Any]]]] = collections.defaultdict(
      lambda: collections.defaultdict(dict)
  )
  for model in model_names:
    for category in CATEGORY_ORDER:
      for app in profiles[category].apps:
        per_model[model][category][app.app_id] = _empty_app_stats(
            app.display_name
        )
  trace_count = 0
  screenshot_episode_count = 0

  for job in jobs:
    model = job["model_name"]
    category = job["category"]
    app_id = job["app_id"]
    episodes = _load_episodes(Path(job["output_path"]))
    rate = _rate(episodes)
    per_model[model][category][app_id] = {
        "app_name": app_labels.get(category, {}).get(app_id, job.get("app_name", app_id)),
        "success_rate": rate,
        "num_episodes": len([ep for ep in episodes if not _is_skipped(ep)]),
        "num_skipped": len([ep for ep in episodes if _is_skipped(ep)]),
        "output_path": job["output_path"],
    }
    if dump_prompt_traces:
      new_traces, new_screenshot_dirs = _dump_prompt_traces(
          out_dir, model, category, app_id, episodes
      )
      trace_count += new_traces
      screenshot_episode_count += new_screenshot_dirs

  main_rows: list[dict[str, Any]] = []
  for model in model_names:
    categories = per_model[model]
    aw_rates = []
    new_app_rates = []
    for category in CATEGORY_ORDER:
      app_stats = categories.get(category, {})
      baseline = BASELINE_APP_BY_CATEGORY[category]
      if baseline in app_stats and app_stats[baseline]["success_rate"] is not None:
        aw_rates.append(app_stats[baseline]["success_rate"])
      for app_id, stats in app_stats.items():
        if app_id == baseline:
          continue
        if stats["success_rate"] is not None:
          new_app_rates.append(stats["success_rate"])
    aw_avg = statistics.mean(aw_rates) if aw_rates else None
    new_avg = statistics.mean(new_app_rates) if new_app_rates else None
    new_std = statistics.pstdev(new_app_rates) if len(new_app_rates) > 1 else 0.0 if new_app_rates else None
    delta = new_avg - aw_avg if aw_avg is not None and new_avg is not None else None
    retain = new_avg / aw_avg if aw_avg and new_avg is not None else None
    main_rows.append(
        {
            "model": model,
            "aw_orig_sr": aw_avg,
            "cat_new_avg_sr": new_avg,
            "new_app_std": new_std,
            "delta": delta,
            "rel_retain": retain,
        }
    )

  average_row = {"model": "Average", "is_average": True}
  for key in (
      "aw_orig_sr",
      "cat_new_avg_sr",
      "new_app_std",
      "delta",
      "rel_retain",
  ):
    values = [row[key] for row in main_rows if row.get(key) is not None]
    average_row[key] = statistics.mean(values) if values else None
  main_rows.append(average_row)

  details = {
      "per_model": per_model,
      "model_order": model_names,
      "prompt_trace_files": trace_count,
      "screenshot_episode_dirs": screenshot_episode_count,
  }
  return main_rows, details


def _write_text(main_rows: list[dict[str, Any]], details: dict[str, Any], out_dir: Path) -> None:
  profiles = get_domain_profiles()
  rows = [("Model", "AW Orig. SR", "CAT New Avg. SR", "New-App Std.", "Delta", "Rel. Retain")]
  for row in main_rows:
    retain = row["rel_retain"]
    rows.append(
        (
            row["model"],
            _fmt_rate(row["aw_orig_sr"]),
            _fmt_rate(row["cat_new_avg_sr"]),
            _fmt_rate(row["new_app_std"]),
            _fmt_rate(row["delta"]),
            "--" if retain is None else f"{retain:.3f}",
        )
    )
  widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
  lines = [
      "  ".join(str(r[i]).ljust(widths[i]) for i in range(len(r)))
      for r in rows
  ]
  (out_dir / "main_cross_app_gap.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

  detail_lines: list[str] = []
  for model in details.get("model_order", sorted(details["per_model"])):
    categories = details["per_model"].get(model, {})
    detail_lines.append(model)
    detail_lines.append("=" * len(model))
    for category in CATEGORY_ORDER:
      app_stats = categories.get(category, {})
      if not app_stats:
        continue
      detail_lines.append(category)
      baseline = BASELINE_APP_BY_CATEGORY[category]
      for app in profiles[category].apps:
        app_id = app.app_id
        stats = app_stats.get(app_id)
        if not stats:
          continue
        mark = "AW" if app_id == baseline else "NEW"
        detail_lines.append(
            f"  [{mark}] {stats['app_name']}: {_fmt_rate(stats['success_rate'])} "
            f"(n={stats['num_episodes']}, skipped={stats['num_skipped']})"
        )
    detail_lines.append("")
  (out_dir / "per_model_category_app.txt").write_text(
      "\n".join(detail_lines), encoding="utf-8"
  )


def _latex_model_name(name: str) -> str:
  if name.endswith("-dagger"):
    return name.removesuffix("-dagger") + r"$^\dagger$"
  return name


def _latex_rate(value: float | None, retain: bool = False) -> str:
  if value is None:
    return "--"
  return f"{value:.3f}" if retain else f"{value:.1f}"


def _write_latex_rows(
    main_rows: list[dict[str, Any]],
    model_config: Path,
    out_dir: Path,
) -> None:
  groups = _model_groups(model_config)
  grouped: collections.OrderedDict[str, list[dict[str, Any]]] = collections.OrderedDict()
  average = None
  for row in main_rows:
    if row.get("is_average"):
      average = row
      continue
    group = groups.get(row["model"], "")
    grouped.setdefault(group, []).append(row)

  lines = [
      "% Auto-generated by report_catbench_5cat_results.py.",
      "% Paste inside the main_cross_app_gap tabular body.",
  ]
  for group, rows in grouped.items():
    if not rows:
      continue
    label = group or "Other"
    for idx, row in enumerate(rows):
      prefix = (
          rf"\multirow{{{len(rows)}}}{{*}}{{\rotatebox{{90}}{{\scriptsize {label}}}}}"
          if idx == 0 else ""
      )
      lines.append(
          f"{prefix}\n"
          f"& {_latex_model_name(row['model'])}\n"
          f"& {_latex_rate(row['aw_orig_sr'])} "
          f"& {_latex_rate(row['cat_new_avg_sr'])} "
          f"& {_latex_rate(row['new_app_std'])} "
          f"& {_latex_rate(row['delta'])} "
          f"& {_latex_rate(row['rel_retain'], retain=True)} \\\\"
      )
    lines.append(r"\midrule")

  if average:
    lines.append(r"\rowcolor{midgray}")
    lines.append(
        rf"& \textbf{{Average}}"
        f"\n& {_latex_rate(average['aw_orig_sr'])} "
        f"& {_latex_rate(average['cat_new_avg_sr'])} "
        f"& {_latex_rate(average['new_app_std'])} "
        f"& {_latex_rate(average['delta'])} "
        f"& {_latex_rate(average['rel_retain'], retain=True)} \\\\"
    )
  (out_dir / "main_cross_app_gap_rows.tex").write_text(
      "\n\n".join(lines) + "\n", encoding="utf-8"
  )


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True)
  parser.add_argument("--model_config", default=str(CONFIG_PATH))
  parser.add_argument("--out_dir", default="")
  parser.add_argument(
      "--skip_prompt_traces",
      action="store_true",
      help="Only write score tables; skip prompt/reasoning JSON and screenshots.",
  )
  args = parser.parse_args()

  manifest = Path(args.manifest).expanduser().resolve()
  model_config = Path(args.model_config).expanduser().resolve()
  out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else manifest.parent / "reports"
  out_dir.mkdir(parents=True, exist_ok=True)

  main_rows, details = _summarize(
      manifest,
      out_dir,
      model_config,
      dump_prompt_traces=not args.skip_prompt_traces,
  )
  with (out_dir / "main_cross_app_gap.json").open("w", encoding="utf-8") as handle:
    json.dump(_safe_json(main_rows), handle, indent=2)
  with (out_dir / "per_model_category_app.json").open("w", encoding="utf-8") as handle:
    json.dump(_safe_json(details), handle, indent=2)
  _write_text(main_rows, details, out_dir)
  _write_latex_rows(main_rows, model_config, out_dir)

  print(f"Wrote reports to {out_dir}")
  print(f"Prompt/reasoning trace files: {details['prompt_trace_files']}")
  print(f"Screenshot episode dirs: {details['screenshot_episode_dirs']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
