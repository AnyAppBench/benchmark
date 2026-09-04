#!/usr/bin/env python3

from __future__ import annotations

import collections
import copy
import json
import tempfile
import unittest
from pathlib import Path

import prepare_cross_judge_human_validation as sample


def _synthetic_rows(pkl_path: Path) -> list[dict]:
  rows = []
  counter = 0
  app_by_category = {
      category: (sample.AW_APP_BY_CATEGORY[category], f"{category}_new_a", f"{category}_new_b")
      for category in sample.CATEGORIES
  }
  confidences = tuple(sample.CONFIDENCE_WEIGHTS)
  qwen_labels = tuple(reversed(sample.FAILURE_MODES))
  # Three candidates for every model/category/Gemini-label edge make all
  # audited flow margins feasible while still exercising weighted draws.
  for model in sample.MODEL_CATEGORY_QUOTAS:
    for category in sample.CATEGORIES:
      for label_index, label in enumerate(sample.FAILURE_MODES):
        for repeat in range(3):
          counter += 1
          app_id = app_by_category[category][(label_index + repeat) % 3]
          rows.append(
              {
                  "episode_id": f"episode-{counter:05d}",
                  "status": "ok",
                  "judge_model": "catbench-judge",
                  "model_name": model,
                  "category": category,
                  "app_id": app_id,
                  "app_name": app_id.replace("_", " ").title(),
                  "task_template": f"{category.title()}Task{label_index}",
                  "goal": f"Synthetic goal {counter}",
                  "is_successful": 0.0,
                  "pkl_path": str(pkl_path),
                  "episode_index": repeat,
                  "gemini_judgment": {
                      "primary_failure_mode": label,
                      "confidence": confidences[(counter + repeat) % 3],
                      "rationale": f"private Gemini rationale {counter}",
                      "planning_score": 1,
                      "grounding_score": 1,
                      "evidence": ["private Gemini evidence"],
                  },
                  "qwen_judgment": {
                      "primary_failure_mode": qwen_labels[label_index],
                      "confidence": confidences[(counter + 1) % 3],
                      "rationale": f"private Qwen rationale {counter}",
                      "planning_score": 2,
                      "grounding_score": 2,
                      "evidence": ["private Qwen evidence"],
                  },
              }
          )
  return rows


class PrepareCrossJudgeHumanValidationTest(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.root = Path(self.temp_dir.name)
    self.pkl_path = self.root / "episode.pkl.gz"
    self.pkl_path.write_bytes(b"test artifact")
    self.rows = _synthetic_rows(self.pkl_path)

  def tearDown(self):
    self.temp_dir.cleanup()

  def test_binding_allocation_and_selection_are_qwen_blind(self):
    selected, _, allocation = sample._select_primary(
        self.rows, seed=sample.MAIN_SEED
    )
    self.assertEqual(200, len(selected))
    self.assertEqual(200, len({row["episode_id"] for row in selected}))
    self.assertEqual(200, sum(allocation.values()))

    by_model = collections.Counter(row["model_name"] for row in selected)
    self.assertEqual(
        {
            model: sum(category_quotas.values())
            for model, category_quotas in sample.MODEL_CATEGORY_QUOTAS.items()
        },
        dict(by_model),
    )
    self.assertEqual(
        {category: 40 for category in sample.CATEGORIES},
        dict(collections.Counter(row["category"] for row in selected)),
    )
    self.assertEqual(
        {label: target for label, target in zip(sample.FAILURE_MODES, (38, 38, 38, 38, 38, 10))},
        dict(collections.Counter(sample._gemini_label(row) for row in selected)),
    )

    changed_qwen = copy.deepcopy(self.rows)
    for index, row in enumerate(changed_qwen):
      row["qwen_judgment"] = {
          "primary_failure_mode": sample.FAILURE_MODES[index % len(sample.FAILURE_MODES)],
          "confidence": "low" if index % 2 else "high",
          "rationale": "entirely changed and must not affect selection",
      }
    changed, _, changed_allocation = sample._select_primary(
        changed_qwen, seed=sample.MAIN_SEED
    )
    self.assertEqual(allocation, changed_allocation)
    self.assertEqual(
        [row["episode_id"] for row in selected],
        [row["episode_id"] for row in changed],
    )

  def test_bundle_is_balanced_blinded_disjoint_and_validator_compatible(self):
    source_path = self.root / "source.jsonl"
    with source_path.open("w", encoding="utf-8") as handle:
      for row in self.rows:
        handle.write(json.dumps(row) + "\n")
    out_dir = self.root / "out"
    result = sample.prepare_artifacts(
        self.rows,
        out_dir=out_dir,
        source_path=source_path,
    )

    manifest = [json.loads(line) for line in (out_dir / "sample_manifest.jsonl").read_text().splitlines()]
    pilot = [json.loads(line) for line in (out_dir / "calibration_manifest.jsonl").read_text().splitlines()]
    private = [json.loads(line) for line in (out_dir / "private_crosswalk.jsonl").read_text().splitlines()]
    self.assertEqual(220, len(manifest))
    self.assertEqual(200, sum(row["pool"] == "primary" for row in manifest))
    self.assertEqual(20, sum(row["pool"] == "double_annotation" for row in manifest))
    self.assertEqual(20, len(pilot))
    self.assertEqual(240, len(private))
    self.assertEqual(
        {category: 4 for category in sample.CATEGORIES},
        dict(collections.Counter(row["category"] for row in pilot)),
    )
    self.assertTrue(
        {row["episode_id"] for row in manifest if row["pool"] == "primary"}.isdisjoint(
            {row["episode_id"] for row in pilot}
        )
    )
    sample._assert_public_blinding(manifest)
    sample._assert_public_blinding(pilot)
    serialized_public = json.dumps([manifest, pilot]).lower()
    self.assertNotIn("rationale", serialized_public)
    # Model names such as Qwen3-VL-8B and Gemini-3-Pro are backend-required
    # episode metadata; judge labels/rationales are what must remain private.

    for name in ("gemini_judge_sample.jsonl", "qwen_judge_sample.jsonl"):
      judge_rows = [json.loads(line) for line in (out_dir / name).read_text().splitlines()]
      self.assertEqual(200, len(judge_rows))
      self.assertTrue(all(set(row) == {"episode_id", "judgment"} for row in judge_rows))
    self.assertTrue((out_dir / "sampling_config.json").is_file())
    self.assertTrue((out_dir / "artifact_hashes.json").is_file())
    self.assertTrue((out_dir / "sample_summary.md").is_file())
    self.assertEqual(200, len(result["primary"]))

  def test_source_validation_rejects_success_duplicate_and_missing_pkl(self):
    base = copy.deepcopy(self.rows[0])
    base["is_successful"] = 1.0
    duplicate = copy.deepcopy(base)
    duplicate["pkl_path"] = str(self.root / "missing.pkl.gz")
    with self.assertRaisesRegex(ValueError, "validator failure") as context:
      sample._validate_source(
          [base, duplicate], require_exact_source=False, check_pkl=True
      )
    message = str(context.exception)
    self.assertIn("duplicate episode IDs", message)
    self.assertIn("pkl does not exist", message)


if __name__ == "__main__":
  unittest.main()
