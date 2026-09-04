#!/usr/bin/env python3
"""Build an aggregate markdown report for CATBench failure-judge outputs."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
from pathlib import Path


FAILURE_MODES = (
    "planning",
    "grounding",
    "mixed_planning_grounding",
    "execution_tooling",
    "environment_or_evaluator",
    "unknown",
)


def _read_json(path: Path) -> dict:
  try:
    with path.open("r", encoding="utf-8") as handle:
      payload = json.load(handle)
  except (OSError, json.JSONDecodeError):
    return {}
  return payload if isinstance(payload, dict) else {}


def _read_status(path: Path) -> dict[str, dict[str, str]]:
  if not path.exists():
    return {}
  rows: dict[str, dict[str, str]] = {}
  lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
  if not lines:
    return rows
  header = lines[0].split("\t")
  for line in lines[1:]:
    parts = line.split("\t")
    if len(parts) != len(header):
      continue
    row = dict(zip(header, parts))
    model = row.get("model", "")
    if model:
      rows[model] = row
  return rows


def _latest_log(default_pattern: str) -> Path | None:
  matches = [Path(path) for path in glob.glob(default_pattern)]
  matches = [path for path in matches if path.exists()]
  if not matches:
    return None
  return max(matches, key=lambda path: path.stat().st_mtime)


def _parse_log_progress(path: Path | None) -> dict[str, str]:
  if not path or not path.exists():
    return {}
  text = path.read_text(encoding="utf-8", errors="replace")
  starts = re.findall(r"\[judge-all\] START model=(.*?) manifest=", text)
  progress = re.findall(r"\[(\d+)/(\d+)\] judging ([^ ]+) ", text)
  out: dict[str, str] = {}
  if starts:
    out["current_model"] = starts[-1]
  if progress:
    done, total, model = progress[-1]
    out["progress_model"] = model
    out["progress"] = f"{done}/{total}"
  fails = re.findall(r"\[judge-all\] FAIL model=(.*?) exit=(\d+)", text)
  if fails:
    out["last_failure"] = ", ".join(f"{model} exit={code}" for model, code in fails[-3:])
  return out


def _model_from_summary(summary: dict, fallback: str) -> str:
  by_model = summary.get("counts_by_model")
  if isinstance(by_model, dict) and len(by_model) == 1:
    return next(iter(by_model))
  return fallback


def _summary_rows(summary_paths: list[Path]) -> dict[str, dict]:
  rows: dict[str, dict] = {}
  for summary_path in summary_paths:
    summary = _read_json(summary_path)
    if not summary:
      continue
    model = _model_from_summary(summary, summary_path.parents[2].name)
    rows[model] = {
        "summary": summary,
        "summary_path": summary_path,
        "summary_md": summary_path.with_suffix(".md"),
        "out_dir": summary_path.parent,
    }
  return rows


def _count(summary: dict, mode: str) -> int:
  counts = summary.get("counts_by_failure_mode", {})
  return int(counts.get(mode, 0)) if isinstance(counts, dict) else 0


def _rel(path: Path, root: Path) -> str:
  try:
    return str(path.resolve().relative_to(root.resolve()))
  except ValueError:
    return str(path)


def _write_report(
    out_path: Path,
    root: Path,
    summaries: dict[str, dict],
    status: dict[str, dict[str, str]],
    log_progress: dict[str, str],
    log_path: Path | None,
) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  all_models = sorted(set(summaries) | set(status))
  aggregate = {mode: 0 for mode in FAILURE_MODES}
  category_counts: dict[str, dict[str, int]] = {}
  for item in summaries.values():
    summary = item["summary"]
    for mode in FAILURE_MODES:
      aggregate[mode] += _count(summary, mode)
    by_category = summary.get("counts_by_category", {})
    if isinstance(by_category, dict):
      for category, counts in by_category.items():
        if not isinstance(counts, dict):
          continue
        bucket = category_counts.setdefault(category, {mode: 0 for mode in FAILURE_MODES})
        for mode in FAILURE_MODES:
          bucket[mode] += int(counts.get(mode, 0))

  lines: list[str] = []
  lines.append("# CATBench Failure Judge Report")
  lines.append("")
  lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
  if log_path:
    lines.append(f"Judge log: `{_rel(log_path, root)}`")
  lines.append("")
  if log_progress:
    lines.append("## Current Judge State")
    lines.append("")
    lines.append(f"- Current model: `{log_progress.get('current_model', 'unknown')}`")
    if "progress" in log_progress:
      lines.append(
          f"- Last progress: `{log_progress.get('progress_model', 'unknown')}` "
          f"`{log_progress['progress']}`"
      )
    if "last_failure" in log_progress:
      lines.append(f"- Recent failed judge passes: `{log_progress['last_failure']}`")
    lines.append("")

  lines.append("## Model Status")
  lines.append("")
  header = ["Model", "Status", "Cases", *FAILURE_MODES, "Summary"]
  lines.append("| " + " | ".join(header) + " |")
  lines.append("|" + "|".join(["---"] * len(header)) + "|")
  for model in all_models:
    item = summaries.get(model)
    summary = item["summary"] if item else {}
    status_text = status.get(model, {}).get("status")
    if not status_text and item:
      status_text = "summary_available"
    status_text = status_text or "pending/running"
    cells = [
        model,
        status_text,
        str(summary.get("total_cases", "--")),
    ]
    cells.extend(str(_count(summary, mode)) if summary else "--" for mode in FAILURE_MODES)
    if item:
      summary_md = item["summary_md"]
      cells.append(f"[md]({_rel(summary_md, root)})")
    else:
      cells.append("--")
    lines.append("| " + " | ".join(cells) + " |")

  lines.append("")
  lines.append("## Aggregate Failure Modes")
  lines.append("")
  lines.append("| Failure mode | Count |")
  lines.append("|---|---:|")
  for mode in FAILURE_MODES:
    lines.append(f"| {mode} | {aggregate[mode]} |")

  if category_counts:
    lines.append("")
    lines.append("## Aggregate By Category")
    lines.append("")
    lines.append("| Category | " + " | ".join(FAILURE_MODES) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(FAILURE_MODES)) + "|")
    for category in sorted(category_counts):
      counts = category_counts[category]
      lines.append(
          f"| {category} | "
          + " | ".join(str(counts.get(mode, 0)) for mode in FAILURE_MODES)
          + " |"
      )

  lines.append("")
  lines.append("## Notes")
  lines.append("")
  lines.append("- This report is incremental: rerun the builder after the judge finishes.")
  lines.append("- Failed judge passes indicate the judge call failed, not necessarily the benchmark run.")
  lines.append("- Per-model markdown files contain high-confidence examples and rationales.")
  lines.append("")
  out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
  parser.add_argument(
      "--summary_glob",
      default="runs/**/failure_mode_analysis/gemini3_maintex/failure_mode_summary.json",
  )
  parser.add_argument(
      "--status_file",
      default="/tmp/catbench_judge_all_maintex_status.tsv",
  )
  parser.add_argument(
      "--log_file",
      default="",
      help="Judge log. Defaults to newest /tmp/catbench_judge_all_maintex_*.log.",
  )
  parser.add_argument(
      "--out",
      default="runs/failure_mode_analysis_maintex_report.md",
  )
  args = parser.parse_args()

  root = Path(args.root).expanduser().resolve()
  summary_paths = sorted(root.glob(args.summary_glob))
  status = _read_status(Path(args.status_file).expanduser())
  log_path = Path(args.log_file).expanduser() if args.log_file else _latest_log(
      "/tmp/catbench_judge_all_maintex_*.log"
  )
  report_path = Path(args.out).expanduser()
  if not report_path.is_absolute():
    report_path = root / report_path
  _write_report(
      out_path=report_path,
      root=root,
      summaries=_summary_rows(summary_paths),
      status=status,
      log_progress=_parse_log_progress(log_path),
      log_path=log_path,
  )
  print(report_path)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
