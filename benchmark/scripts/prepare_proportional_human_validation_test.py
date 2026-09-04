#!/usr/bin/env python3

from __future__ import annotations

import collections
import copy
import json
import tempfile
import unittest
from pathlib import Path

import prepare_cross_judge_human_validation as shared
import prepare_proportional_human_validation as sample


def _synthetic_rows(pkl_path: Path) -> list[dict]:
  rows: list[dict] = []
  models = tuple(shared.MODEL_CATEGORY_QUOTAS)
  labels = shared.FAILURE_MODES
  counter = 0
  for category, count in sample.SOURCE_CATEGORY_COUNTS.items():
    for index in range(count):
      counter += 1
      app_id = (
          shared.AW_APP_BY_CATEGORY[category]
          if index % 7 == 0
          else f"{category}_new_{index % 4}"
      )
      rows.append(
          {
              "episode_id": f"episode-{counter:05d}",
              "status": "ok",
              "model_name": models[index % len(models)],
              "category": category,
              "app_id": app_id,
              "app_name": app_id,
              "task_template": f"{category}_task_{index % 10}",
              "goal": f"goal {counter}",
              "is_successful": 0.0,
              "pkl_path": str(pkl_path),
              "episode_index": 0,
              "gemini_judgment": {
                  "primary_failure_mode": labels[index % len(labels)],
                  "confidence": ("low", "medium", "high")[index % 3],
                  "rationale": "private Gemini output",
              },
              "qwen_judgment": {
                  "primary_failure_mode": labels[(index + 1) % len(labels)],
                  "confidence": ("high", "medium", "low")[index % 3],
                  "rationale": "private Qwen output",
              },
          }
      )
  return rows


class PrepareProportionalHumanValidationTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.pkl_path = self.root / "episode.pkl.gz"
    self.pkl_path.write_bytes(b"synthetic trajectory")
    self.rows = _synthetic_rows(self.pkl_path)

  def tearDown(self):
    self.temporary.cleanup()

  def test_primary_is_exact_reproducible_and_blind_to_both_judges(self):
    selected, audits = sample._select_primary(self.rows, seed=sample.MAIN_SEED)
    self.assertEqual(sample.PRIMARY_N, len(selected))
    self.assertEqual(sample.PRIMARY_N, len({row["episode_id"] for row in selected}))
    self.assertEqual(
        collections.Counter(sample.CATEGORY_QUOTAS),
        collections.Counter(row["category"] for row in selected),
    )

    changed = copy.deepcopy(self.rows)
    for index, row in enumerate(changed):
      row["gemini_judgment"] = {
          "primary_failure_mode": shared.FAILURE_MODES[-1 - index % 6],
          "confidence": "low",
          "rationale": "changed",
      }
      row["qwen_judgment"] = {
          "primary_failure_mode": shared.FAILURE_MODES[index % 6],
          "confidence": "high",
          "rationale": "changed",
      }
    changed_selected, changed_audits = sample._select_primary(
        changed, seed=sample.MAIN_SEED
    )
    self.assertEqual(
        [row["episode_id"] for row in selected],
        [row["episode_id"] for row in changed_selected],
    )
    self.assertEqual(audits, changed_audits)

  def test_bundle_sizes_blinding_and_disjointness(self):
    source = self.root / "source.jsonl"
    with source.open("w", encoding="utf-8") as handle:
      for row in self.rows:
        handle.write(json.dumps(row) + "\n")
    out = self.root / "out"
    result = sample.prepare_artifacts(
        self.rows, out_dir=out, source_path=source
    )

    manifest = shared._read_jsonl(out / "sample_manifest.jsonl")
    pilots = shared._read_jsonl(out / "calibration_manifest.jsonl")
    private = shared._read_jsonl(out / "private_crosswalk.jsonl")
    primary = [row for row in manifest if row["pool"] == "primary"]
    doubles = [row for row in manifest if row["pool"] == "double_annotation"]
    self.assertEqual(242, len(manifest))
    self.assertEqual(220, len(primary))
    self.assertEqual(22, len(doubles))
    self.assertEqual(20, len(pilots))
    self.assertEqual(262, len(private))
    self.assertEqual(
        collections.Counter(sample.CATEGORY_QUOTAS),
        collections.Counter(row["category"] for row in primary),
    )
    self.assertEqual(
        {category: 4 for category in shared.CATEGORIES},
        dict(collections.Counter(row["category"] for row in pilots)),
    )
    primary_ids = {row["episode_id"] for row in primary}
    self.assertTrue({row["episode_id"] for row in doubles} <= primary_ids)
    self.assertTrue(primary_ids.isdisjoint({row["episode_id"] for row in pilots}))
    shared._assert_public_blinding(manifest)
    shared._assert_public_blinding(pilots)
    self.assertEqual(220, len(shared._read_jsonl(out / "gemini_judge_sample.jsonl")))
    self.assertEqual(220, len(shared._read_jsonl(out / "qwen_judge_sample.jsonl")))
    self.assertEqual(220, len(result["primary"]))
    config = json.loads((out / "sampling_config.json").read_text())
    self.assertEqual(["episode_id", "category"], config["selection_fields_used"])
    self.assertEqual(0.1, config["double_annotation_fraction"])

  def test_denominator_mismatch_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "denominator mismatch"):
      sample._validate_frozen_denominator(self.rows[:-1])


if __name__ == "__main__":
  unittest.main()
