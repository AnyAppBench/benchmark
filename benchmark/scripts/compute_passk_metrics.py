#!/usr/bin/env python3
"""Compute pass@1, pass@k, and FRR from a passk_index.json.

The pass@k_index.json is produced by run_passk_attempts.py and lists the per-
attempt manifests. This script:

  1. Reads each attempt's manifest and all its pkl.gz checkpoint files.
  2. Groups episodes by (model, category, app_id, task_template).
  3. Computes:
       - pass@1  : SR of attempt 1 only
       - pass@k  : SR if ANY of k attempts succeeded
       - FRR     : Failure Recovery Rate with harmonic decay weighting
                   (sum_{tasks that failed at attempt 1} 1 / first_success_attempt
                    divided by count of tasks that failed at attempt 1).
  4. Writes pass_k_summary.{json,md} with per-model / per-category breakdowns.

If a judge JSONL is supplied (--judge_jsonl), the script can optionally use the
judge's *verdict* instead of the recorded `is_successful` flag — this is what
MemGUI does to bypass noisy evaluators.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from classify_catbench_failures import _read_pkl_gz, _is_skipped  # noqa: E402


def _read_manifest(path: Path) -> list[dict[str, Any]]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  jobs = payload.get("jobs", [])
  return [job for job in jobs if isinstance(job, dict)]


def _collect_attempt_results(
    manifest: Path,
    judge_by_id: dict[str, dict[str, Any]] | None,
) -> dict[tuple[str, str, str, str], bool]:
  """Returns {(model, category, app_id, task_template): success_bool}."""
  results: dict[tuple[str, str, str, str], bool] = {}
  for job in _read_manifest(manifest):
    model = str(job.get("model_name") or "")
    category = str(job.get("category") or "")
    app_id = str(job.get("app_id") or "")
    output_path = Path(str(job.get("output_path") or "")).expanduser()
    if not output_path.exists():
      continue
    for pkl in sorted(output_path.rglob("*.pkl.gz")):
      try:
        payload = _read_pkl_gz(pkl)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"warning: skipping {pkl}: {exc}", file=sys.stderr)
        continue
      eps = payload if isinstance(payload, list) else [payload]
      for ep_idx, ep in enumerate(eps):
        if not isinstance(ep, dict) or _is_skipped(ep):
          continue
        task = str(ep.get("task_template") or pkl.stem)
        key = (model, category, app_id, task)
        if judge_by_id is not None:
          # Try to match a judge row for this episode.
          for jr in judge_by_id.values():
            if (
                jr.get("pkl_path") == str(pkl)
                and str(jr.get("model_name")) == model
                and str(jr.get("task_template")) == task
            ):
              verdict = (jr.get("judgment") or {}).get("verdict")
              if verdict in {"success", "failure"}:
                results[key] = verdict == "success"
                break
          else:
            results[key] = float(ep.get("is_successful") or 0.0) >= 0.5
        else:
          results[key] = float(ep.get("is_successful") or 0.0) >= 0.5
  return results


def _harmonic_first_success(successes: list[bool]) -> float | None:
  """Returns 1 / (1-indexed attempt of first success), or None if none succeed."""
  for idx, succ in enumerate(successes, start=1):
    if succ:
      return 1.0 / idx
  return None


def _aggregate(
    per_attempt: list[dict[tuple[str, str, str, str], bool]],
) -> dict[str, Any]:
  all_keys = set()
  for attempt in per_attempt:
    all_keys.update(attempt.keys())

  per_task: list[dict[str, Any]] = []
  for key in sorted(all_keys):
    successes = [attempt.get(key, False) for attempt in per_attempt]
    per_task.append(
        {
            "model": key[0], "category": key[1], "app_id": key[2], "task": key[3],
            "successes": successes,
        }
    )

  def rate(predicate) -> float:
    if not per_task:
      return 0.0
    return sum(1 for row in per_task if predicate(row)) / len(per_task)

  def frr(rows: list[dict[str, Any]]) -> dict[str, float]:
    failed_at_1 = [row for row in rows if not row["successes"][0]]
    if not failed_at_1:
      return {"frr": 0.0, "n_failed_at_1": 0}
    weights = [_harmonic_first_success(row["successes"][1:]) for row in failed_at_1]
    weights = [w for w in weights if w is not None]
    return {
        "frr": sum(weights) / len(failed_at_1),
        "n_failed_at_1": len(failed_at_1),
        "n_recovered": len(weights),
    }

  by_model: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  by_category: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  for row in per_task:
    by_model[row["model"]].append(row)
    by_category[row["category"]].append(row)

  def group_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
      return {"pass_at_1": 0.0, "pass_at_k": 0.0, "n": 0}
    p1 = sum(1 for row in rows if row["successes"][0]) / len(rows)
    pk = sum(1 for row in rows if any(row["successes"])) / len(rows)
    out = {"pass_at_1": p1, "pass_at_k": pk, "n": len(rows)}
    out.update(frr(rows))
    return out

  return {
      "k": len(per_attempt),
      "overall": group_metrics(per_task),
      "by_model": {m: group_metrics(rows) for m, rows in by_model.items()},
      "by_category": {c: group_metrics(rows) for c, rows in by_category.items()},
      "per_task": per_task,
  }


def _write_markdown(out_path: Path, report: dict[str, Any]) -> None:
  k = report["k"]
  lines = [f"# CATBench pass@{k} Summary", ""]
  overall = report["overall"]
  lines.append(
      f"Overall: pass@1={overall['pass_at_1']:.3f}, "
      f"pass@{k}={overall['pass_at_k']:.3f}, "
      f"FRR={overall.get('frr', 0):.3f} "
      f"(n_tasks={overall['n']}, recovered={overall.get('n_recovered', 0)}/"
      f"{overall.get('n_failed_at_1', 0)})"
  )
  lines.append("")
  lines.append("## By Model")
  lines.append("")
  lines.append("| Model | n | pass@1 | pass@k | FRR | failed@1 | recovered |")
  lines.append("|---|---:|---:|---:|---:|---:|---:|")
  for model in sorted(report["by_model"]):
    metrics = report["by_model"][model]
    lines.append(
        f"| {model} | {metrics['n']} | {metrics['pass_at_1']:.3f} | "
        f"{metrics['pass_at_k']:.3f} | {metrics.get('frr', 0):.3f} | "
        f"{metrics.get('n_failed_at_1', 0)} | {metrics.get('n_recovered', 0)} |"
    )
  lines.append("")
  lines.append("## By Category")
  lines.append("")
  lines.append("| Category | n | pass@1 | pass@k | FRR |")
  lines.append("|---|---:|---:|---:|---:|")
  for category in sorted(report["by_category"]):
    metrics = report["by_category"][category]
    lines.append(
        f"| {category} | {metrics['n']} | {metrics['pass_at_1']:.3f} | "
        f"{metrics['pass_at_k']:.3f} | {metrics.get('frr', 0):.3f} |"
    )
  out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--passk_index", required=True)
  parser.add_argument(
      "--judge_jsonl",
      default="",
      help=(
          "Optional failure_mode_judgments.jsonl whose `verdict` overrides the "
          "recorded is_successful. Use to bypass noisy evaluators."
      ),
  )
  parser.add_argument("--out_dir", required=True)
  args = parser.parse_args()

  out_dir = Path(args.out_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  judge_by_id: dict[str, dict[str, Any]] | None = None
  if args.judge_jsonl:
    judge_by_id = {}
    with Path(args.judge_jsonl).expanduser().open("r", encoding="utf-8") as handle:
      for line in handle:
        line = line.strip()
        if not line:
          continue
        row = json.loads(line)
        judge_by_id[row.get("episode_id", "")] = row

  index = json.loads(Path(args.passk_index).expanduser().read_text(encoding="utf-8"))
  per_attempt: list[dict[tuple[str, str, str, str], bool]] = []
  for attempt in index.get("attempts", []):
    manifest_path = Path(attempt["manifest"]).expanduser()
    if not manifest_path.exists():
      print(f"warning: missing manifest {manifest_path}", file=sys.stderr)
      per_attempt.append({})
      continue
    per_attempt.append(_collect_attempt_results(manifest_path, judge_by_id))

  report = _aggregate(per_attempt)
  (out_dir / "pass_k_summary.json").write_text(
      json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
  )
  _write_markdown(out_dir / "pass_k_summary.md", report)
  print(f"Wrote {out_dir / 'pass_k_summary.md'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
