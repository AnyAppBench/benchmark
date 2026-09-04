#!/usr/bin/env python3
"""Build exact CATBench C1 repair targets from failure-mode judge output."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CASE_CSV = (
    "$HOME/anyappbench_results/"
    "20260525_c1_failure_judge_selected_gemini_vlm_no_qwen30b/"
    "failure_mode_case_level.csv"
)
DEFAULT_OUT_DIR = "benchmark/configs/c1_issue_repair_targets"
DEFAULT_MODES = ("environment_or_evaluator", "execution_tooling")


def _repair_track(row: dict[str, str]) -> str:
  mode = row["chart_failure_mode"]
  category = row["category"]
  rationale = row.get("rationale", "").lower()
  if mode == "execution_tooling":
    if "open_app" in rationale or "open app" in rationale:
      return "action_schema_open_app"
    if any(
        term in rationale
        for term in (
            "set-of-mark",
            "som",
            "bounding box",
            "numeric index",
            "fab",
            "floating action",
            "play button",
        )
    ):
      return "som_detection"
    if "scroll" in rationale or "overshoot" in rationale:
      return "scroll_control"
    if "input" in rationale or "keyboard" in rationale or "text" in rationale:
      return "text_entry_or_focus"
    return "execution_tooling"

  if category == "maps" and any(
      term in rationale
      for term in ("connection failure", "download", "internet", "network")
  ):
    return "env_network_or_map_seed"
  if category in {"sms", "contacts"} and any(
      term in rationale
      for term in (
          "missing",
          "not found",
          "no stored",
          "no contacts",
          "required message",
          "required contact",
      )
  ):
    return "env_seed_data"
  if "validator" in rationale or "fulfilled" in rationale or "successfully" in rationale:
    return "validator_audit"
  return "env_or_evaluator_audit"


def _target_from_row(row: dict[str, str]) -> dict[str, str]:
  return {
      "model": row["model_name"],
      "category": row["category"],
      "app_id": row["app_id"],
      "task": row["task_template"],
      "episode_id": row.get("episode_id", ""),
      "chart_failure_mode": row["chart_failure_mode"],
      "repair_track": _repair_track(row),
      "pkl_path": row.get("pkl_path", ""),
      "rationale": row.get("rationale", ""),
  }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
    handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def _counter_rows(counter: Counter[tuple[str, ...]], fieldnames: list[str]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for key, count in sorted(counter.items()):
    row = {name: value for name, value in zip(fieldnames, key, strict=True)}
    row["count"] = count
    rows.append(row)
  return rows


def build_targets(case_csv: Path, out_dir: Path, stamp: str, modes: set[str]) -> dict[str, Path]:
  with case_csv.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

  selected = [row for row in rows if row["chart_failure_mode"] in modes]
  targets = [_target_from_row(row) for row in selected]
  deduped: list[dict[str, str]] = []
  seen: set[tuple[str, str, str, str]] = set()
  for target in targets:
    key = (
        target["model"],
        target["category"],
        target["app_id"],
        target["task"],
    )
    if key in seen:
      continue
    seen.add(key)
    deduped.append(target)

  generated_at = datetime.now(timezone.utc).isoformat()
  base_payload = {
      "source_case_csv": str(case_csv),
      "generated_at_utc": generated_at,
      "selection": {"chart_failure_mode": sorted(modes)},
  }

  written: dict[str, Path] = {}
  groups = {
      "env_evaluator": [
          target for target in deduped
          if target["chart_failure_mode"] == "environment_or_evaluator"
      ],
      "execution_tooling": [
          target for target in deduped
          if target["chart_failure_mode"] == "execution_tooling"
      ],
      "env_execution": deduped,
  }
  for slug, group_targets in groups.items():
    path = out_dir / f"{stamp}_{slug}_targets.json"
    _write_json(
        path,
        {
            **base_payload,
            "target_count": len(group_targets),
            "targets": group_targets,
        },
    )
    written[slug] = path

  batch_counter = Counter(
      (
          target["chart_failure_mode"],
          target["repair_track"],
          target["model"],
          target["category"],
          target["app_id"],
      )
      for target in deduped
  )
  batch_rows = _counter_rows(
      batch_counter,
      ["chart_failure_mode", "repair_track", "model", "category", "app_id"],
  )
  batch_csv = out_dir / f"{stamp}_issue_repair_batches.csv"
  _write_csv(
      batch_csv,
      batch_rows,
      ["chart_failure_mode", "repair_track", "model", "category", "app_id", "count"],
  )
  written["batches_csv"] = batch_csv

  summary_counter = Counter(
      (target["chart_failure_mode"], target["repair_track"]) for target in deduped
  )
  summary_rows = _counter_rows(
      summary_counter,
      ["chart_failure_mode", "repair_track"],
  )
  summary_csv = out_dir / f"{stamp}_issue_repair_summary.csv"
  _write_csv(
      summary_csv,
      summary_rows,
      ["chart_failure_mode", "repair_track", "count"],
  )
  written["summary_csv"] = summary_csv

  summary_json = out_dir / f"{stamp}_issue_repair_summary.json"
  _write_json(
      summary_json,
      {
          **base_payload,
          "source_rows": len(rows),
          "selected_rows": len(selected),
          "target_count": len(deduped),
          "mode_counts": Counter(
              target["chart_failure_mode"] for target in deduped
          ),
          "repair_track_counts": Counter(
              target["repair_track"] for target in deduped
          ),
          "model_counts": Counter(target["model"] for target in deduped),
          "category_counts": Counter(target["category"] for target in deduped),
      },
  )
  written["summary_json"] = summary_json
  return written


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--case_csv", default=DEFAULT_CASE_CSV)
  parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
  parser.add_argument("--stamp", default="20260526")
  parser.add_argument("--mode", action="append", choices=DEFAULT_MODES)
  args = parser.parse_args()

  modes = set(args.mode or DEFAULT_MODES)
  written = build_targets(
      Path(args.case_csv).expanduser().resolve(),
      Path(args.out_dir).expanduser().resolve(),
      args.stamp,
      modes,
  )
  for name, path in written.items():
    print(f"{name}: {path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
