#!/usr/bin/env python3
"""Freeze a 10%-of-cohort CATBench C1 human-validation sample.

This design is for the frozen 2,195 verifier-failed C1 episodes in Clock,
Contacts, Files, Maps, and SMS.  The primary sample is a proportionate,
category-stratified simple random sample: 220 unique episodes, using the
pre-declared category quotas 54/49/37/42/38.  Selection is deliberately blind
to *both* judges: no Gemini or Qwen label, score, confidence, rationale, or
cross-judge agreement can affect inclusion.

The output layout is compatible with ``serve_annotation_app.py``:

* ``sample_manifest.jsonl`` contains 220 primary rows plus 22 uniformly drawn
  duplicate rows for a second annotator;
* ``calibration_manifest.jsonl`` contains 20 disjoint pilots (four/category);
* public manifests contain no judge outputs; and
* private crosswalk and judge-sample files must remain sealed until annotation
  is complete.

This script only freezes artifacts.  It never launches an annotation server.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import prepare_cross_judge_human_validation as shared


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = shared.DEFAULT_SOURCE
MAIN_SEED = 20260712
DOUBLE_ANNOTATION_SEED = 20260713
PILOT_SEED = 20260711

SOURCE_CATEGORY_COUNTS: dict[str, int] = {
    "clock": 539,
    "contacts": 486,
    "files": 373,
    "maps": 422,
    "sms": 375,
}
CATEGORY_QUOTAS: dict[str, int] = {
    "clock": 54,
    "contacts": 49,
    "files": 37,
    "maps": 42,
    "sms": 38,
}
PRIMARY_N = sum(CATEGORY_QUOTAS.values())
DOUBLE_ANNOTATION_N = 22
PILOT_PER_CATEGORY = 4
PILOT_N = PILOT_PER_CATEGORY * len(shared.CATEGORIES)


def _derived_seed(seed: int, stage: str, stratum: str) -> int:
  """Derive a stable per-stratum seed without depending on row order."""
  material = f"catbench|{seed}|{stage}|{stratum}".encode("utf-8")
  return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _category_counts(rows: Sequence[Mapping[str, Any]]) -> collections.Counter[str]:
  return collections.Counter(str(row["category"]) for row in rows)


def _validate_frozen_denominator(rows: Sequence[Mapping[str, Any]]) -> None:
  actual = _category_counts(rows)
  expected = collections.Counter(SOURCE_CATEGORY_COUNTS)
  if actual != expected:
    raise ValueError(
        "Frozen category denominator mismatch: "
        f"expected={dict(expected)}, found={dict(actual)}"
    )
  if sum(actual.values()) != shared.EXPECTED_SOURCE_ROWS:
    raise ValueError(
        f"Expected {shared.EXPECTED_SOURCE_ROWS} source rows, "
        f"found {sum(actual.values())}"
    )


def _uniform_category_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    quotas: Mapping[str, int],
    seed: int,
    stage: str,
) -> tuple[list[Mapping[str, Any]], dict[str, dict[str, Any]]]:
  """Draw an order-invariant simple random sample within each category."""
  grouped: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
  for row in rows:
    grouped[str(row["category"])].append(row)

  selected: list[Mapping[str, Any]] = []
  audits: dict[str, dict[str, Any]] = {}
  for category in shared.CATEGORIES:
    candidates = sorted(grouped.get(category, []), key=lambda row: str(row["episode_id"]))
    target = int(quotas.get(category, 0))
    if len(candidates) < target:
      raise ValueError(
          f"Insufficient {stage} candidates for {category}: "
          f"need {target}, found {len(candidates)}"
      )
    stratum_seed = _derived_seed(seed, stage, category)
    chosen = random.Random(stratum_seed).sample(candidates, target)
    inclusion_probability = target / len(candidates)
    analysis_weight = len(candidates) / target if target else 0.0
    for draw_index, row in enumerate(chosen):
      episode_id = str(row["episode_id"])
      selected.append(row)
      audits[episode_id] = {
          "selection_stage": stage,
          "selection_design": "category_stratified_simple_random_without_replacement",
          "selection_stratum": {"category": category},
          "source_stratum_n": len(candidates),
          "sample_stratum_n": target,
          "uniform_inclusion_probability": inclusion_probability,
          "inverse_probability_analysis_weight": analysis_weight,
          "draw_index_within_category": draw_index,
          "derived_stratum_seed": stratum_seed,
          "qwen_used_for_sampling": False,
          "gemini_used_for_sampling": False,
          "judge_agreement_used_for_sampling": False,
      }
  return selected, audits


def _select_primary(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> tuple[list[Mapping[str, Any]], dict[str, dict[str, Any]]]:
  selected, audits = _uniform_category_sample(
      rows, quotas=CATEGORY_QUOTAS, seed=seed, stage="primary"
  )
  if len(selected) != PRIMARY_N:
    raise AssertionError(f"primary size is {len(selected)}, expected {PRIMARY_N}")
  if len({str(row["episode_id"]) for row in selected}) != PRIMARY_N:
    raise AssertionError("primary sample does not contain 220 unique episodes")
  if _category_counts(selected) != collections.Counter(CATEGORY_QUOTAS):
    raise AssertionError("primary category quotas were not met")
  return selected, audits


def _select_pilots(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_episode_ids: set[str],
    seed: int,
) -> tuple[list[Mapping[str, Any]], dict[str, dict[str, Any]]]:
  remaining = [
      row for row in rows if str(row["episode_id"]) not in excluded_episode_ids
  ]
  quotas = {category: PILOT_PER_CATEGORY for category in shared.CATEGORIES}
  pilots, audits = _uniform_category_sample(
      remaining, quotas=quotas, seed=seed, stage="calibration"
  )
  pilot_ids = {str(row["episode_id"]) for row in pilots}
  if len(pilots) != PILOT_N or len(pilot_ids) != PILOT_N:
    raise AssertionError("calibration sample is not 20 unique episodes")
  if pilot_ids & excluded_episode_ids:
    raise AssertionError("calibration sample overlaps the primary sample")
  return pilots, audits


def _randomized_public_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    pool: str,
    sample_prefix: str,
    seed: int,
) -> list[dict[str, Any]]:
  ordered = list(rows)
  random.Random(_derived_seed(seed, "display_order", pool)).shuffle(ordered)
  return [
      shared._public_row(
          row, pool=pool, sample_id=f"{sample_prefix}-{index:03d}"
      )
      for index, row in enumerate(ordered, 1)
  ]


def _select_double_annotation(
    primary_public: Sequence[Mapping[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
  ordered = sorted(primary_public, key=lambda row: str(row["episode_id"]))
  chosen = random.Random(seed).sample(ordered, DOUBLE_ANNOTATION_N)
  duplicates: list[dict[str, Any]] = []
  audits: dict[str, dict[str, Any]] = {}
  for index, original in enumerate(chosen, 1):
    duplicate = copy.deepcopy(dict(original))
    duplicate["pool"] = "double_annotation"
    duplicate["sample_id"] = f"double-{index:03d}"
    duplicate["duplicated_from_sample_id"] = original["sample_id"]
    duplicates.append(duplicate)
    episode_id = str(original["episode_id"])
    audits[episode_id] = {
        "selection_stage": "double_annotation",
        "selection_design": "simple_random_without_replacement_from_primary",
        "duplicated_from_sample_id": original["sample_id"],
        "uniform_draw_seed": seed,
        "source_n": PRIMARY_N,
        "sample_n": DOUBLE_ANNOTATION_N,
        "uniform_inclusion_probability": DOUBLE_ANNOTATION_N / PRIMARY_N,
        "qwen_used_for_sampling": False,
        "gemini_used_for_sampling": False,
        "judge_agreement_used_for_sampling": False,
    }
  if len({row["episode_id"] for row in duplicates}) != DOUBLE_ANNOTATION_N:
    raise AssertionError("double-annotation sample is not 22 unique episodes")
  return duplicates, audits


def _write_protocol(path: Path, *, out_dir: Path) -> None:
  relative = out_dir
  try:
    relative = out_dir.relative_to(REPO_ROOT)
  except ValueError:
    pass
  text = f"""# CATBench 10%-cohort human-annotation protocol

This frozen bundle contains 220 unique primary cases (a category-stratified
10% sample of the 2,195 five-category C1 verifier failures), 22 uniformly
sampled primary cases for independent second annotation, and 20 disjoint
calibration pilots excluded from evaluation. Primary and pilot selection used
no Gemini or Qwen outputs and no cross-judge agreement.

The category quotas are Clock=54, Contacts=49, Files=37, Maps=42, and SMS=38.
For pooled population estimates, use the category-specific inverse-probability
weights recorded in `sampling_config.json`; also report category-stratified
metrics. Do not inspect private judge files until human labels are frozen.

Primary annotator:

```bash
python3 \\
  benchmark/scripts/serve_annotation_app.py \\
  --sample_manifest {relative}/sample_manifest.jsonl \\
  --out_dir {relative}/primary_annotations \\
  --annotator ttran --pool primary --blind --max_frames 0 --max_dim 896 \\
  --host 127.0.0.1 --port 8875
```

Second annotator (22 repeated cases only):

```bash
python3 \\
  benchmark/scripts/serve_annotation_app.py \\
  --sample_manifest {relative}/sample_manifest.jsonl \\
  --out_dir {relative}/primary_annotations \\
  --annotator second_rater --pool double_annotation --blind \\
  --max_frames 0 --max_dim 896 --host 127.0.0.1 --port 8876
```

Calibration pilots are in `calibration_manifest.jsonl`. They must remain
excluded from judge-accuracy and human-agreement estimates.
"""
  path.write_text(text, encoding="utf-8")


def _write_summary(
    path: Path,
    *,
    primary: Sequence[Mapping[str, Any]],
    doubles: Sequence[Mapping[str, Any]],
    pilots: Sequence[Mapping[str, Any]],
    config_sha256: str,
) -> None:
  lines = [
      "# CATBench 10%-Cohort Human-Validation Sample",
      "",
      "Primary inclusion was blind to both judges and used category-stratified "
      "simple random sampling without replacement.",
      "",
      "| audit check | result |",
      "|---|---:|",
      f"| frozen C1 failure denominator | PASS ({sum(SOURCE_CATEGORY_COUNTS.values())}) |",
      f"| unique primary episodes | PASS ({len(primary)}) |",
      f"| uniform second-rater duplicates | PASS ({len(doubles)}) |",
      f"| disjoint calibration pilots | PASS ({len(pilots)}) |",
      "| Gemini outputs used for selection | NO |",
      "| Qwen outputs used for selection | NO |",
      "| cross-judge agreement used for selection | NO |",
      f"| sampling config SHA-256 | `{config_sha256}` |",
      "",
  ]
  shared._counter_table(lines, "Source by category", SOURCE_CATEGORY_COUNTS)
  shared._counter_table(lines, "Primary by category", _category_counts(primary))
  shared._counter_table(
      lines,
      "Primary by model (realized, not quota-controlled)",
      collections.Counter(str(row["model_name"]) for row in primary),
  )
  shared._counter_table(
      lines,
      "Primary by app (realized, not quota-controlled)",
      collections.Counter(str(row["app_id"]) for row in primary),
  )
  shared._counter_table(lines, "Calibration by category", _category_counts(pilots))
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
    source_path: Path,
    main_seed: int = MAIN_SEED,
    double_seed: int = DOUBLE_ANNOTATION_SEED,
    pilot_seed: int = PILOT_SEED,
    all_source_pkl_audited: bool = True,
) -> dict[str, Any]:
  """Select and write the complete, frozen 220-case annotation bundle."""
  _validate_frozen_denominator(rows)
  out_dir.mkdir(parents=True, exist_ok=True)

  primary, primary_audits = _select_primary(rows, seed=main_seed)
  primary_ids = {str(row["episode_id"]) for row in primary}
  pilots, pilot_audits = _select_pilots(
      rows, excluded_episode_ids=primary_ids, seed=pilot_seed
  )
  missing = shared._missing_pkl_paths([*primary, *pilots])
  if missing:
    preview = ", ".join(f"{episode_id}:{path}" for episode_id, path in missing[:5])
    raise FileNotFoundError("Selected annotation trajectories are missing: " + preview)
  if not all(shared._safe_failure(row.get("is_successful")) for row in [*primary, *pilots]):
    raise AssertionError("a selected episode is not a validator failure")

  primary_public = _randomized_public_rows(
      primary,
      pool="primary",
      sample_prefix="primary",
      seed=main_seed,
  )
  pilot_public = _randomized_public_rows(
      pilots,
      pool="calibration",
      sample_prefix="calibration",
      seed=pilot_seed,
  )
  double_public, double_audits = _select_double_annotation(
      primary_public, seed=double_seed
  )
  sample_manifest = [*primary_public, *double_public]
  shared._assert_public_blinding(sample_manifest)
  shared._assert_public_blinding(pilot_public)

  source_by_id = {str(row["episode_id"]): row for row in rows}
  private_rows: list[dict[str, Any]] = []
  for public in primary_public:
    episode_id = str(public["episode_id"])
    private_rows.append(
        shared._private_row(public, source_by_id[episode_id], primary_audits[episode_id])
    )
  for public in double_public:
    episode_id = str(public["episode_id"])
    private_rows.append(
        shared._private_row(public, source_by_id[episode_id], double_audits[episode_id])
    )
  for public in pilot_public:
    episode_id = str(public["episode_id"])
    private_rows.append(
        shared._private_row(public, source_by_id[episode_id], pilot_audits[episode_id])
    )

  gemini_sample = [
      {
          "episode_id": str(public["episode_id"]),
          "judgment": copy.deepcopy(
              dict(shared._gemini_judgment(source_by_id[str(public["episode_id"])]))
          ),
      }
      for public in primary_public
  ]
  qwen_sample = [
      {
          "episode_id": str(public["episode_id"]),
          "judgment": copy.deepcopy(
              dict(shared._qwen_judgment(source_by_id[str(public["episode_id"])]))
          ),
      }
      for public in primary_public
  ]

  category_design = {
      category: {
          "source_n": SOURCE_CATEGORY_COUNTS[category],
          "sample_n": CATEGORY_QUOTAS[category],
          "inclusion_probability": (
              CATEGORY_QUOTAS[category] / SOURCE_CATEGORY_COUNTS[category]
          ),
          "inverse_probability_analysis_weight": (
              SOURCE_CATEGORY_COUNTS[category] / CATEGORY_QUOTAS[category]
          ),
      }
      for category in shared.CATEGORIES
  }
  config_without_hash: dict[str, Any] = {
      "schema_version": "catbench_proportional_human_validation_v1",
      "source_jsonl": str(source_path.resolve()),
      "source_sha256": shared._sha256_file(source_path),
      "source_rows": len(rows),
      "source_category_counts": SOURCE_CATEGORY_COUNTS,
      "all_source_pkl_audited": all_source_pkl_audited,
      "selected_pkl_audited": True,
      "sampling_design": "category_stratified_simple_random_without_replacement",
      "primary_n": PRIMARY_N,
      "primary_fraction_of_frozen_cohort": PRIMARY_N / len(rows),
      "category_quotas": CATEGORY_QUOTAS,
      "category_sampling_design": category_design,
      "double_annotation_n": DOUBLE_ANNOTATION_N,
      "double_annotation_fraction": DOUBLE_ANNOTATION_N / PRIMARY_N,
      "calibration_n": PILOT_N,
      "calibration_per_category": PILOT_PER_CATEGORY,
      "main_seed": main_seed,
      "double_annotation_seed": double_seed,
      "pilot_seed": pilot_seed,
      "selection_fields_used": ["episode_id", "category"],
      "selection_fields_forbidden": [
          "gemini_judgment",
          "qwen_judgment",
          "Gemini-Qwen agreement",
      ],
      "analysis_note": (
          "Use category-specific inverse-probability weights for pooled "
          "population estimates because rounded quotas give slightly unequal "
          "inclusion probabilities."
      ),
  }
  config_sha256 = shared._sha256_bytes(shared._canonical_json(config_without_hash))
  config = {**config_without_hash, "config_sha256": config_sha256}

  outputs = {
      "sample_manifest.jsonl": sample_manifest,
      "calibration_manifest.jsonl": pilot_public,
      "private_crosswalk.jsonl": private_rows,
      "gemini_judge_sample.jsonl": gemini_sample,
      "qwen_judge_sample.jsonl": qwen_sample,
  }
  for name, output_rows in outputs.items():
    shared._write_jsonl(out_dir / name, output_rows)
  (out_dir / "sampling_config.json").write_text(
      json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  _write_summary(
      out_dir / "sample_summary.md",
      primary=primary,
      doubles=double_public,
      pilots=pilots,
      config_sha256=config_sha256,
  )
  _write_protocol(out_dir / "annotation_protocol.md", out_dir=out_dir)

  artifact_names = [
      *outputs,
      "sampling_config.json",
      "sample_summary.md",
      "annotation_protocol.md",
  ]
  hashes = {
      "schema_version": "catbench_annotation_artifact_hashes_v1",
      "config_sha256": config_sha256,
      "artifacts": {
          name: {
              "sha256": shared._sha256_file(out_dir / name),
              "bytes": (out_dir / name).stat().st_size,
          }
          for name in artifact_names
      },
  }
  (out_dir / "artifact_hashes.json").write_text(
      json.dumps(hashes, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  return {
      "primary": primary,
      "pilots": pilots,
      "sample_manifest": sample_manifest,
      "private_crosswalk": private_rows,
      "config": config,
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source_jsonl", type=Path, default=DEFAULT_SOURCE)
  parser.add_argument("--out_dir", type=Path, required=True)
  parser.add_argument("--seed", type=int, default=MAIN_SEED)
  parser.add_argument("--double_seed", type=int, default=DOUBLE_ANNOTATION_SEED)
  parser.add_argument("--pilot_seed", type=int, default=PILOT_SEED)
  parser.add_argument(
      "--skip_all_source_pkl_audit",
      action="store_true",
      help=(
          "Skip source-wide trajectory stat calls after a prior exact-cohort "
          "audit. Selected primary and calibration trajectories are always checked."
      ),
  )
  args = parser.parse_args(argv)

  source_path = args.source_jsonl.expanduser().resolve()
  out_dir = args.out_dir.expanduser().resolve()
  rows = shared._read_jsonl(source_path)
  all_source_pkl_audited = not args.skip_all_source_pkl_audit
  shared._validate_source(
      rows, require_exact_source=True, check_pkl=all_source_pkl_audited
  )
  _validate_frozen_denominator(rows)
  result = prepare_artifacts(
      rows,
      out_dir=out_dir,
      source_path=source_path,
      main_seed=args.seed,
      double_seed=args.double_seed,
      pilot_seed=args.pilot_seed,
      all_source_pkl_audited=all_source_pkl_audited,
  )
  print(f"Frozen unique primary episodes: {len(result['primary'])}")
  print(f"Server queue rows (primary + duplicates): {len(result['sample_manifest'])}")
  print(f"Disjoint calibration episodes: {len(result['pilots'])}")
  print(f"Artifacts: {out_dir}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
