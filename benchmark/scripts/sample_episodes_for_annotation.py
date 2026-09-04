#!/usr/bin/env python3
"""Stratified sampler that picks CATBench episodes for human annotation.

Strategy answers the practical question: "we have many models running — do we
annotate every model?" The short answer is *no*, you stratify.

The sampler takes a CATBench matrix manifest (or a directory of pkl.gz files)
plus an optional judge-output JSONL (failure_mode_judgments.jsonl) and emits a
balanced sample manifest of ~N episodes.

Stratification axes (in order of precedence):
  1. Model        — every model gets a minimum quota so no model dominates.
  2. Category     — within each model, balance categories (SMS, Files, ...).
  3. Verdict      — validator-failures only by default; successes are sampled
                    only for the separate validator-audit pass.
  4. Judge label  — within failures, balance across the failure modes the
                    judge already assigned (so we can compute per-mode F1).
  5. Confidence   — boost low-confidence cases (most informative for F1).

Outputs:
  - sample_manifest.jsonl  : one JSON record per sampled episode (annotation queue).
  - sample_summary.md      : human-readable breakdown of the sample composition.

Companion scripts:
  - annotate_judge_cases.py        : walks the manifest and captures labels.
  - validate_judge_vs_human.py     : computes F1/confusion vs human labels.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from classify_catbench_failures import (  # noqa: E402
    _collect_records,
    _read_pkl_gz,
)


CONFIDENCE_WEIGHT = {"low": 3.0, "medium": 1.5, "high": 1.0}


def _load_judge_rows(path: Path | None) -> dict[str, dict[str, Any]]:
  """Index existing judge rows by episode_id (or pkl_path+episode_index)."""
  if not path or not path.exists():
    return {}
  rows: dict[str, dict[str, Any]] = {}
  with path.open("r", encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError:
        continue
      key = str(row.get("episode_id") or "")
      if not key:
        key = f"{row.get('pkl_path')}::{row.get('episode_index', 0)}"
      rows[key] = row
  return rows


def _record_signature(record: Any) -> str:
  """Stable signature for a record across runs."""
  return getattr(record, "episode_id", "") or (
      f"{record.pkl_path}::{record.episode_index}"
  )


def _bucket_records(
    records: list[Any],
    judge_rows: dict[str, dict[str, Any]],
) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
  """Group records by (model, category, verdict, judge_label)."""
  buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = (
      collections.defaultdict(list)
  )
  for record in records:
    signature = _record_signature(record)
    judge_row = judge_rows.get(signature, {})
    judgment = judge_row.get("judgment") or {}
    verdict = "success" if record.is_successful >= 0.5 else "failure"
    judge_label = str(judgment.get("primary_failure_mode") or "unjudged")
    confidence = str(judgment.get("confidence") or "low")
    weight = CONFIDENCE_WEIGHT.get(confidence, 1.0)
    bucket_key = (record.model_name, record.category, verdict, judge_label)
    buckets[bucket_key].append(
        {
            "record": record,
            "weight": weight,
            "judge_row": judge_row,
        }
    )
  return buckets


def _quota_per_model(total: int, models: list[str], min_per_model: int) -> dict[str, int]:
  if not models:
    return {}
  per_model = max(min_per_model, total // len(models))
  remainder = total - per_model * len(models)
  quotas = {model: per_model for model in models}
  for model in models[:max(0, remainder)]:
    quotas[model] += 1
  return quotas


def _sample_within_model(
    buckets_for_model: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    quota: int,
    rng: random.Random,
    failure_share: float,
) -> list[dict[str, Any]]:
  # Split quota into success / failure halves.
  failure_quota = int(round(quota * failure_share))
  success_quota = quota - failure_quota

  def pick_from(verdict_filter: str, count: int) -> list[dict[str, Any]]:
    if count <= 0:
      return []
    sub_buckets = {
        key: items
        for key, items in buckets_for_model.items()
        if key[2] == verdict_filter and items
    }
    if not sub_buckets:
      return []
    # Round-robin across sub-buckets, weighted by confidence.
    keys = list(sub_buckets.keys())
    rng.shuffle(keys)
    picked: list[dict[str, Any]] = []
    while keys and len(picked) < count:
      progress = False
      for key in list(keys):
        items = sub_buckets[key]
        if not items:
          keys.remove(key)
          continue
        weights = [item["weight"] for item in items]
        item = rng.choices(items, weights=weights, k=1)[0]
        items.remove(item)
        picked.append(item)
        progress = True
        if len(picked) >= count:
          break
      if not progress:
        break
    return picked

  return pick_from("failure", failure_quota) + pick_from("success", success_quota)


def _serialize_pick(pick: dict[str, Any], pool: str) -> dict[str, Any]:
  record = pick["record"]
  judge_row = pick["judge_row"] or {}
  judgment = judge_row.get("judgment") or {}
  return {
      "pool": pool,
      "episode_id": _record_signature(record),
      "model_name": record.model_name,
      "category": record.category,
      "app_id": record.app_id,
      "app_name": record.app_name,
      "task_template": record.task_template,
      "goal": record.goal,
      "is_successful": record.is_successful,
      "finish_dtime": record.finish_dtime,
      "pkl_path": str(record.pkl_path),
      "episode_index": record.episode_index,
      "output_path": str(record.output_path),
      "judge_label": judgment.get("primary_failure_mode") or "unjudged",
      "judge_confidence": judgment.get("confidence") or "",
      "judge_verdict": judgment.get("verdict") or "",
      "judge_rationale": judgment.get("rationale") or "",
  }


def _write_summary(out_path: Path, picks: list[dict[str, Any]]) -> None:
  by_model = collections.Counter(pick["model_name"] for pick in picks)
  by_category = collections.Counter(pick["category"] for pick in picks)
  by_judge = collections.Counter(pick["judge_label"] for pick in picks)
  by_verdict = collections.Counter(
      "success" if pick["is_successful"] >= 0.5 else "failure" for pick in picks
  )
  by_pool = collections.Counter(pick["pool"] for pick in picks)
  lines = ["# Annotation Sample Composition", ""]
  lines.append(f"Total picked: {len(picks)}")
  lines.append("")
  for title, counter in (
      ("By pool", by_pool),
      ("By model", by_model),
      ("By category", by_category),
      ("By verdict", by_verdict),
      ("By judge label", by_judge),
  ):
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| key | count |")
    lines.append("|---|---:|")
    for key, count in counter.most_common():
      lines.append(f"| {key} | {count} |")
    lines.append("")
  out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--manifest",
      required=True,
      help="CATBench matrix manifest JSON (the same one classify_catbench_failures uses).",
  )
  parser.add_argument(
      "--judge_jsonl",
      default="",
      help=(
          "Optional path to failure_mode_judgments.jsonl from a prior judge "
          "run. Used to stratify on judge label/confidence."
      ),
  )
  parser.add_argument("--out_dir", required=True)
  parser.add_argument(
      "--total",
      type=int,
      default=200,
      help="Target total number of episodes to sample (default 200).",
  )
  parser.add_argument(
      "--min_per_model",
      type=int,
      default=8,
      help="Minimum episodes per model — prevents one model from dominating.",
  )
  parser.add_argument(
      "--failure_share",
      type=float,
      default=1.0,
      help=(
          "Fraction of the sample that should be validator-failed episodes. "
          "Default 1.0 because the new pipeline only annotates failures — "
          "the validator owns pass/fail, the human and the VLM only "
          "classify the failure mode. Set to a smaller value ONLY for the "
          "optional validator-audit pass."
      ),
  )
  parser.add_argument(
      "--double_annotation_share",
      type=float,
      default=0.1,
      help=(
          "Fraction of picked episodes copied into a 'double_annotation' pool "
          "for inter-rater agreement (default 10%%)."
      ),
  )
  parser.add_argument("--seed", type=int, default=20260520)
  parser.add_argument("--model", action="append", default=[])
  parser.add_argument("--category", action="append", default=[])
  args = parser.parse_args()

  rng = random.Random(args.seed)
  manifest = Path(args.manifest).expanduser().resolve()
  out_dir = Path(args.out_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  judge_rows = _load_judge_rows(
      Path(args.judge_jsonl).expanduser() if args.judge_jsonl else None
  )

  records = _collect_records(
      manifest=manifest,
      include_successes=True,
      model_filter=set(args.model),
      category_filter=set(args.category),
      app_filter=set(),
      task_regex=None,
      newest_first=True,
      max_records=0,
      keep_images=False,
  )
  if not records:
    print("No records found for the given filters.", file=sys.stderr)
    return 1

  buckets = _bucket_records(records, judge_rows)
  models = sorted({record.model_name for record in records})
  quotas = _quota_per_model(args.total, models, args.min_per_model)

  picked: list[dict[str, Any]] = []
  for model_name in models:
    model_buckets = {
        key: list(items) for key, items in buckets.items() if key[0] == model_name
    }
    picks = _sample_within_model(
        model_buckets, quotas[model_name], rng, args.failure_share
    )
    for pick in picks:
      picked.append(_serialize_pick(pick, pool="primary"))

  # Double-annotation pool — copy a random subset for inter-rater agreement.
  double_n = int(round(len(picked) * args.double_annotation_share))
  if double_n > 0:
    double_picks = rng.sample(picked, double_n)
    for pick in double_picks:
      pick_copy = dict(pick)
      pick_copy["pool"] = "double_annotation"
      picked.append(pick_copy)

  manifest_path = out_dir / "sample_manifest.jsonl"
  with manifest_path.open("w", encoding="utf-8") as handle:
    for pick in picked:
      handle.write(json.dumps(pick, ensure_ascii=False) + "\n")
  _write_summary(out_dir / "sample_summary.md", picked)

  print(f"Wrote {len(picked)} sampled episodes to {manifest_path}")
  print(f"Summary: {out_dir / 'sample_summary.md'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
