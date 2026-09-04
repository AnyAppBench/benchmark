#!/usr/bin/env python3
"""Compute precision/recall/F1/confusion of the LLM judge vs. human labels.

Inputs:
  --annotations   annotations.jsonl produced by annotate_judge_cases.py
  --judge_jsonl   failure_mode_judgments.jsonl from classify_catbench_failures.py

Outputs (under --out_dir):
  - report.json    machine-readable metrics
  - report.md      human-readable markdown table
  - mismatches.jsonl  per-case rows where judge != human (for review)
  - agreement.json    inter-rater agreement on the double_annotation pool
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


FAILURE_MODES = (
    "planning",
    "grounding",
    "mixed_planning_grounding",
    "execution_tooling",
    "environment_or_evaluator",
    "unknown",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError:
        continue
  return rows


def _consolidate_humans(annotations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  """For the double_annotation pool keep both labels; for primary pool the
  most recent label wins. Returns {episode_id: {primary, doubles[],
  annotators[]}}."""
  by_episode: dict[str, dict[str, Any]] = collections.defaultdict(
      lambda: {
          "primary": None,
          "doubles": [],
          "_doubles_by_annotator": {},
          "annotators": [],
      }
  )
  for row in annotations:
    episode_id = row.get("episode_id")
    pool = row.get("pool", "primary")
    if not episode_id:
      continue
    entry = by_episode[episode_id]
    entry["annotators"].append(row.get("annotator", "?"))
    if pool == "primary":
      entry["primary"] = row
    else:
      entry["_doubles_by_annotator"][row.get("annotator", "?")] = row
  for entry in by_episode.values():
    entry["doubles"] = list(entry["_doubles_by_annotator"].values())
    del entry["_doubles_by_annotator"]
  return by_episode


def _metric_table(
    rows: list[tuple[str, str]],
    labels: tuple[str, ...],
) -> dict[str, Any]:
  """rows: list of (human_label, judge_label). Returns per-label P/R/F1 + macro/micro."""
  tp = collections.Counter()
  fp = collections.Counter()
  fn = collections.Counter()
  confusion: dict[str, collections.Counter[str]] = {
      label: collections.Counter() for label in labels
  }
  for human, judge in rows:
    if human not in labels:
      human = "unknown"
    if judge not in labels:
      judge = "unknown"
    confusion[human][judge] += 1
    if human == judge:
      tp[human] += 1
    else:
      fp[judge] += 1
      fn[human] += 1

  per_label = {}
  macro_f1 = 0.0
  macro_n = 0
  for label in labels:
    p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
    r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    support = tp[label] + fn[label]
    per_label[label] = {
        "precision": p, "recall": r, "f1": f1, "support": support
    }
    if support > 0:
      macro_f1 += f1
      macro_n += 1
  macro_f1 = macro_f1 / macro_n if macro_n else 0.0
  total_tp = sum(tp.values())
  micro_acc = total_tp / len(rows) if rows else 0.0
  return {
      "per_label": per_label,
      "macro_f1": macro_f1,
      "micro_accuracy": micro_acc,
      "n": len(rows),
      "confusion": {label: dict(counter) for label, counter in confusion.items()},
  }


def _agreement(doubles: list[dict[str, Any]]) -> dict[str, Any]:
  """Cohen's kappa on the doubly-annotated subset."""
  pairs: list[tuple[str, str]] = []
  for entry in doubles:
    primary = entry.get("primary")
    secondary = None
    for candidate in entry["doubles"]:
      if not primary or candidate.get("annotator") != primary.get("annotator"):
        secondary = candidate
        break
    if not primary or not secondary:
      continue
    a = str(primary["human_label"]["primary_failure_mode"])
    b = str(secondary["human_label"]["primary_failure_mode"])
    pairs.append((a, b))
  if not pairs:
    return {"n": 0}
  agreement = sum(1 for a, b in pairs if a == b) / len(pairs)
  marginals_a = collections.Counter(a for a, _ in pairs)
  marginals_b = collections.Counter(b for _, b in pairs)
  pe = 0.0
  n = len(pairs)
  for label in set(marginals_a) | set(marginals_b):
    pe += (marginals_a[label] / n) * (marginals_b[label] / n)
  kappa = (agreement - pe) / (1 - pe) if pe < 1 else 0.0
  return {"n": n, "agreement": agreement, "cohens_kappa": kappa}


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
  lines = ["# Judge vs. Human Validation Report", ""]
  lines.append(
      "Scope: human annotations cover *failure-mode classification only*. "
      "The binary pass/fail verdict is owned by the programmatic AW/CATBench "
      "validator and is not subject to human override."
  )
  lines.append("")
  lines.append(f"Annotations used: {report['n_used']} (skipped: {report['n_skipped']})")
  lines.append("")
  lines.append("## Failure-mode classification")
  lines.append("")
  multi = report["multiclass"]
  lines.append(
      f"Macro-F1: {multi['macro_f1']:.3f}  Micro-Acc: {multi['micro_accuracy']:.3f}  "
      f"(N={multi['n']})"
  )
  lines.append("")
  lines.append("| Mode | Precision | Recall | F1 | Support |")
  lines.append("|---|---:|---:|---:|---:|")
  for label, metrics in multi["per_label"].items():
    lines.append(
        f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} "
        f"| {metrics['f1']:.3f} | {metrics['support']} |"
    )
  lines.append("")
  lines.append("## Confusion matrix (rows = human, cols = judge)")
  lines.append("")
  header_labels = list(multi["per_label"].keys())
  lines.append("| | " + " | ".join(header_labels) + " |")
  lines.append("|---|" + "---:|" * len(header_labels))
  for human in header_labels:
    cells = [str(multi["confusion"][human].get(judge, 0)) for judge in header_labels]
    lines.append(f"| {human} | " + " | ".join(cells) + " |")
  lines.append("")
  if report.get("agreement", {}).get("n"):
    agr = report["agreement"]
    lines.append("## Inter-rater agreement (double_annotation pool)")
    lines.append("")
    lines.append(
        f"- N pairs: {agr['n']}  Raw agreement: {agr['agreement']:.3f}  "
        f"Cohen's κ: {agr['cohens_kappa']:.3f}"
    )
    lines.append("")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--annotations", required=True)
  parser.add_argument("--judge_jsonl", required=True)
  parser.add_argument("--out_dir", required=True)
  args = parser.parse_args()

  out_dir = Path(args.out_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  annotations = _load_jsonl(Path(args.annotations).expanduser())
  judge_rows = _load_jsonl(Path(args.judge_jsonl).expanduser())
  judge_by_id = {row.get("episode_id"): row for row in judge_rows if row.get("episode_id")}

  human_by_id = _consolidate_humans(annotations)

  multiclass_rows: list[tuple[str, str]] = []
  mismatches: list[dict[str, Any]] = []
  used = 0
  skipped = 0
  for episode_id, entry in human_by_id.items():
    primary = entry["primary"]
    judge = judge_by_id.get(episode_id)
    if not primary or not judge:
      skipped += 1
      continue
    used += 1
    human_label = primary["human_label"]
    judgment = judge.get("judgment") or {}
    human_mode = str(human_label["primary_failure_mode"])
    judge_mode = str(judgment.get("primary_failure_mode") or "unknown")
    multiclass_rows.append((human_mode, judge_mode))

    if human_mode != judge_mode:
      mismatches.append(
          {
              "episode_id": episode_id,
              "model_name": primary.get("model_name"),
              "category": primary.get("category"),
              "task_template": primary.get("task_template"),
              "human": human_label,
              "judge": judgment,
              "frames_dir": primary.get("frames_dir"),
          }
      )

  multi_metrics = _metric_table(multiclass_rows, FAILURE_MODES)
  agreement = _agreement([entry for entry in human_by_id.values() if entry["doubles"]])

  report = {
      "n_used": used,
      "n_skipped": skipped,
      "multiclass": multi_metrics,
      "agreement": agreement,
  }
  (out_dir / "report.json").write_text(
      json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
  )
  _write_markdown(out_dir / "report.md", report)
  with (out_dir / "mismatches.jsonl").open("w", encoding="utf-8") as handle:
    for row in mismatches:
      handle.write(json.dumps(row, ensure_ascii=False) + "\n")
  if agreement.get("n"):
    (out_dir / "agreement.json").write_text(
        json.dumps(agreement, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

  print(f"Wrote {out_dir / 'report.md'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
