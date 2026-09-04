#!/usr/bin/env python3
"""Fast score-only CATBench manifest summarizer.

Unlike report_catbench_5cat_results.py, this script does not dump prompt traces
or screenshots. It parallel-loads pkl.gz episodes and writes the same core score
tables used by the paper report:

  - main_cross_app_gap.json
  - per_model_category_app.json
  - main_cross_app_gap.txt
  - per_model_category_app.txt
  - main_cross_app_gap_rows.tex
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gzip
import io
import json
import math
import pickle
import statistics
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

from app_generalization_profiles import get_domain_profiles  # pylint: disable=wrong-import-position
from report_catbench_5cat_results import (  # pylint: disable=wrong-import-position
    BASELINE_APP_BY_CATEGORY,
    CATEGORY_ORDER,
    CONFIG_PATH,
    SKIP_EXCEPTION_MARKERS,
    _empty_app_stats,
    _fmt_rate,
    _latex_model_name,
    _latex_rate,
    _model_groups,
    _model_order,
    _safe_json,
)


def _read_pkl_gz(path: Path) -> Any:
  with path.open("rb") as handle:
    raw = handle.read()
  with gzip.open(io.BytesIO(raw), "rb") as gz:
    return pickle.load(gz)


def _is_skipped(ep: dict[str, Any]) -> bool:
  info = ep.get("exception_info") or ep.get("EXCEPTION_INFO") or ""
  return isinstance(info, str) and any(
      marker in info for marker in SKIP_EXCEPTION_MARKERS
  )


def _episode_files(output_path: str) -> list[str]:
  path = Path(output_path)
  if not path.exists():
    return []
  return [str(p) for p in sorted(path.rglob("*.pkl.gz"))]


def _read_episode_result(task: tuple[int, str]) -> list[dict[str, Any]]:
  job_index, raw_path = task
  path = Path(raw_path)
  try:
    payload = _read_pkl_gz(path)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return [{
        "job_index": job_index,
        "path": raw_path,
        "read_error": str(exc),
    }]
  episodes = payload if isinstance(payload, list) else [payload]
  rows: list[dict[str, Any]] = []
  for episode_index, ep in enumerate(episodes):
    if not isinstance(ep, dict):
      continue
    skipped = _is_skipped(ep)
    success = None
    if not skipped:
      try:
        success = 1.0 if float(ep.get("is_successful") or 0.0) >= 0.5 else 0.0
      except (TypeError, ValueError):
        success = 0.0
    rows.append(
        {
            "job_index": job_index,
            "path": raw_path,
            "episode_index": episode_index,
            "task_template": ep.get("task_template") or ep.get("name") or path.name,
            "skipped": skipped,
            "success": success,
        }
    )
  return rows


def _load_jobs(path: Path) -> list[dict[str, Any]]:
  with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  return [job for job in payload.get("jobs", []) if isinstance(job, dict)]


def _rate(successes: list[float]) -> float | None:
  if not successes:
    return None
  return 100.0 * sum(successes) / len(successes)


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
  (out_dir / "main_cross_app_gap.txt").write_text(
      "\n".join(lines) + "\n", encoding="utf-8"
  )

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
      "% Auto-generated by summarize_catbench_scores_fast.py.",
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


def _summarize(
    manifest: Path,
    model_config: Path,
    workers: int,
    max_tasks_per_child: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  profiles = get_domain_profiles()
  app_labels = {
      category: {app.app_id: app.display_name for app in profiles[category].apps}
      for category in CATEGORY_ORDER
  }
  jobs = _load_jobs(manifest)
  present_models = {
      str(job.get("model_name"))
      for job in jobs
      if isinstance(job.get("model_name"), str)
  }
  model_names = [
      model for model in _model_order(model_config, jobs) if model in present_models
  ]

  per_job: dict[int, dict[str, Any]] = {
      index: {
          "successes": [],
          "num_skipped": 0,
          "read_errors": [],
      }
      for index, _ in enumerate(jobs)
  }
  tasks: list[tuple[int, str]] = []
  for index, job in enumerate(jobs):
    for episode_path in _episode_files(str(job.get("output_path") or "")):
      tasks.append((index, episode_path))

  pool_kwargs: dict[str, Any] = {"max_workers": workers}
  if max_tasks_per_child > 0:
    pool_kwargs["max_tasks_per_child"] = max_tasks_per_child
  with concurrent.futures.ProcessPoolExecutor(**pool_kwargs) as executor:
    for rows in executor.map(_read_episode_result, tasks, chunksize=4):
      for row in rows:
        bucket = per_job[row["job_index"]]
        if row.get("read_error"):
          bucket["read_errors"].append(row)
        elif row.get("skipped"):
          bucket["num_skipped"] += 1
        elif row.get("success") is not None:
          bucket["successes"].append(float(row["success"]))

  per_model: dict[str, dict[str, dict[str, dict[str, Any]]]] = collections.defaultdict(
      lambda: collections.defaultdict(dict)
  )
  for model in model_names:
    for category in CATEGORY_ORDER:
      for app in profiles[category].apps:
        per_model[model][category][app.app_id] = _empty_app_stats(app.display_name)

  read_errors: list[dict[str, Any]] = []
  for index, job in enumerate(jobs):
    model = job["model_name"]
    category = job["category"]
    app_id = job["app_id"]
    stats = per_job[index]
    read_errors.extend(stats["read_errors"])
    per_model[model][category][app_id] = {
        "app_name": app_labels.get(category, {}).get(app_id, job.get("app_name", app_id)),
        "success_rate": _rate(stats["successes"]),
        "num_episodes": len(stats["successes"]),
        "num_skipped": stats["num_skipped"],
        "num_read_errors": len(stats["read_errors"]),
        "output_path": job["output_path"],
    }

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
    new_std = (
        statistics.pstdev(new_app_rates)
        if len(new_app_rates) > 1
        else 0.0 if new_app_rates else None
    )
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
      "read_errors": read_errors,
      "num_episode_files": len(tasks),
  }
  return main_rows, details


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True)
  parser.add_argument("--model_config", default=str(CONFIG_PATH))
  parser.add_argument("--out_dir", required=True)
  parser.add_argument("--workers", type=int, default=8)
  parser.add_argument(
      "--max_tasks_per_child",
      type=int,
      default=0,
      help=(
          "Recycle worker processes after this many files. Default 0 keeps "
          "fork workers alive for speed."
      ),
  )
  args = parser.parse_args()

  manifest = Path(args.manifest).expanduser().resolve()
  model_config = Path(args.model_config).expanduser().resolve()
  out_dir = Path(args.out_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  main_rows, details = _summarize(
      manifest,
      model_config,
      max(1, args.workers),
      max(0, args.max_tasks_per_child),
  )
  (out_dir / "main_cross_app_gap.json").write_text(
      json.dumps(_safe_json(main_rows), indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  (out_dir / "per_model_category_app.json").write_text(
      json.dumps(_safe_json(details), indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  _write_text(main_rows, details, out_dir)
  _write_latex_rows(main_rows, model_config, out_dir)

  print(f"Wrote score-only reports to {out_dir}")
  print(f"Episode files read: {details['num_episode_files']}")
  print(f"Read errors: {len(details['read_errors'])}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
