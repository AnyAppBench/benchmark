#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import compare_failure_judges as subject  # noqa: E402


class CompareFailureJudgesTest(unittest.TestCase):

  def test_outputs_exclude_errors_without_converting_unknown(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      gemini_csv = root / "gemini.csv"
      qwen_jsonl = root / "qwen.jsonl"
      out_dir = root / "out"

      gemini_rows = [
          self._gemini("e1", "M1", "clock", "planning"),
          self._gemini("e2", "M1", "maps", "grounding"),
          self._gemini("e3", "M1", "sms", "unknown"),
          self._gemini("e4", "M2", "contacts", "environment_or_evaluator"),
          self._gemini("e5", "M2", "files", "grounding"),
          self._gemini("e6", "M2", "clock", "planning"),
          self._gemini("e8", "M2", "sms", "unknown"),
          self._gemini("e9", "M2", "maps", "planning"),
          # The default five-category filter must remove this row.
          self._gemini("outside", "M3", "todo", "planning"),
      ]
      with gemini_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gemini_rows[0]))
        writer.writeheader()
        writer.writerows(gemini_rows)

      qwen_rows = [
          {
              "episode_id": "e1",
              "qwen_judgment": self._judgment("planning"),
              "usage": {"num_images": 6, "model": "catbench-judge"},
              "gemini_usage": {"num_images": 6},
          },
          {
              "episode_id": "e2",
              "judgment": self._judgment("mixed_planning_grounding"),
              "evidence_parity": {"historical_gemini_num_images": 4},
          },
          {
              "episode_id": "e3",
              "primary_failure_mode": "unknown",
              "judge_error": True,
              "status": "error",
          },
          {
              "episode_id": "e5",
              "primary_failure_mode": "grounding",
              "error": "0",
              "gemini_usage": {"num_images": 0},
          },
          {
              "episode_id": "e6",
              "qwen_judgment": self._judgment("grounding"),
              "gemini_usage": {"num_images": "not-recorded"},
              "evidence_parity": {"historical_gemini_num_images": 0},
          },
          {
              "episode_id": "e8",
              "judgment": self._judgment("unknown"),
              "gemini_usage": {"num_images": 0},
          },
          {
              "episode_id": "e9",
              "judgment": self._judgment("not_a_mode"),
              "gemini_usage": {"num_images": 0},
          },
          {"episode_id": "outside", "judgment": self._judgment("planning")},
      ]
      qwen_jsonl.write_text(
          "".join(json.dumps(row) + "\n" for row in qwen_rows),
          encoding="utf-8",
      )

      result = subject.main(
          [
              "--gemini_csv",
              str(gemini_csv),
              "--qwen_jsonl",
              str(qwen_jsonl),
              "--out_dir",
              str(out_dir),
              "--expected_n",
              "8",
              "--bootstrap_replicates",
              "200",
          ]
      )
      self.assertEqual(result, 0)

      coverage = json.loads((out_dir / "coverage.json").read_text())
      self.assertEqual(coverage["gemini_roster_cases"], 8)
      self.assertEqual(coverage["matched_qwen_episode_ids"], 7)
      self.assertEqual(coverage["valid_pairs"], 5)
      self.assertEqual(coverage["excluded_pairs"], 3)
      self.assertEqual(coverage["missing_qwen_rows"], 1)
      self.assertEqual(coverage["qwen_explicit_error_pairs"], 1)
      self.assertEqual(coverage["qwen_invalid_or_missing_label_pairs"], 1)
      self.assertEqual(coverage["qwen_extra_episode_ids"], 1)
      self.assertEqual(coverage["historical_evidence_unknown_pairs"], 2)
      self.assertEqual(coverage["historical_evidence_count_conflicts"], 0)

      overall = json.loads((out_dir / "overall.json").read_text())
      self.assertEqual(overall["n"], 5)
      self.assertAlmostEqual(overall["raw_agreement"], 0.6)
      self.assertAlmostEqual(overall["expected_chance_agreement"], 0.28)
      self.assertAlmostEqual(overall["cohens_kappa"], 4.0 / 9.0)
      self.assertEqual(overall["bootstrap"]["replicates_requested"], 200)
      self.assertEqual(overall["bootstrap"]["raw_agreement_replicates"], 200)
      self.assertEqual(
          overall["bootstrap"]["cluster_fields"],
          list(subject.BOOTSTRAP_CLUSTER_FIELDS),
      )
      self.assertEqual(len(overall["raw_agreement_ci95"]), 2)
      self.assertEqual(len(overall["cohens_kappa_ci95"]), 2)
      self.assertEqual(overall["confusion_counts"]["unknown"]["unknown"], 1)
      self.assertEqual(
          overall["confusion_counts"]["grounding"]["mixed_planning_grounding"],
          1,
      )

      paired = self._load_jsonl(out_dir / "paired_cases.jsonl")
      paired_by_id = {row["episode_id"]: row for row in paired}
      self.assertTrue(paired_by_id["e8"]["valid_pair"])
      self.assertEqual(paired_by_id["e8"]["qwen_label"], "unknown")
      self.assertFalse(paired_by_id["e3"]["valid_pair"])
      self.assertEqual(paired_by_id["e3"]["qwen_label"], "unknown")
      self.assertIn("judge_error", paired_by_id["e3"]["qwen_error"])
      self.assertEqual(paired_by_id["e4"]["qwen_error"], "missing_qwen_row")
      self.assertFalse(paired_by_id["e9"]["valid_pair"])
      self.assertIn("invalid_primary_failure_mode", paired_by_id["e9"]["qwen_error"])
      self.assertEqual(paired_by_id["e1"]["qwen_num_images"], 6)
      self.assertEqual(paired_by_id["e1"]["historical_num_images"], 6)
      self.assertEqual(paired_by_id["e1"]["evidence_type"], "visual")
      self.assertEqual(paired_by_id["e2"]["historical_num_images"], 4)
      self.assertEqual(
          paired_by_id["e2"]["evidence_count_source"],
          "evidence_parity.historical_gemini_num_images",
      )
      self.assertEqual(paired_by_id["e4"]["historical_num_images"], "")
      self.assertEqual(paired_by_id["e4"]["evidence_type"], "unknown")
      self.assertEqual(paired_by_id["e3"]["evidence_type"], "unknown")
      self.assertEqual(paired_by_id["e5"]["evidence_type"], "zero_image")
      self.assertEqual(paired_by_id["e6"]["evidence_type"], "zero_image")

      with (out_dir / "by_evidence.csv").open(newline="", encoding="utf-8") as handle:
        evidence = {row["evidence_type"]: row for row in csv.DictReader(handle)}
      self.assertEqual(set(evidence), {"unknown", "visual", "zero_image"})
      self.assertEqual(evidence["visual"]["roster_n"], "2")
      self.assertEqual(evidence["visual"]["valid_pair_n"], "2")
      self.assertEqual(evidence["zero_image"]["roster_n"], "4")
      self.assertEqual(evidence["zero_image"]["valid_pair_n"], "3")
      self.assertEqual(evidence["unknown"]["roster_n"], "2")
      self.assertEqual(evidence["unknown"]["valid_pair_n"], "0")

      with (out_dir / "by_category.csv").open(newline="", encoding="utf-8") as handle:
        categories = {row["category"]: row for row in csv.DictReader(handle)}
      self.assertEqual(set(categories), {"clock", "contacts", "files", "maps", "sms"})
      self.assertEqual(categories["contacts"]["valid_pair_n"], "0")
      self.assertEqual(categories["contacts"]["cohens_kappa"], "")

      with (out_dir / "label_shift_by_model.csv").open(newline="", encoding="utf-8") as handle:
        model_shifts = list(csv.DictReader(handle))
      self.assertEqual(len(model_shifts), 2 * len(subject.FAILURE_MODES))
      self.assertTrue((out_dir / "confusion_counts.csv").exists())
      self.assertTrue((out_dir / "overall.md").exists())
      markdown = (out_dir / "overall.md").read_text(encoding="utf-8")
      self.assertIn("## By historical Gemini evidence", markdown)
      self.assertIn("| zero_image | 4 | 3 | 1 |", markdown)
      self.assertIn("cluster-bootstrap CI", markdown)

  def test_duplicate_qwen_rows_are_audited_and_last_wins(self) -> None:
    rows = [
        {"episode_id": "e1", "judgment": self._judgment("planning")},
        {"episode_id": "e1", "judgment": self._judgment("grounding")},
        {"judgment": self._judgment("unknown")},
    ]
    indexed, audit = subject._index_qwen_rows(rows)
    self.assertEqual(subject._extract_qwen(indexed["e1"])["label"], "grounding")
    self.assertEqual(audit["qwen_duplicate_episode_ids"], 1)
    self.assertEqual(audit["qwen_duplicate_rows"], 1)
    self.assertEqual(audit["qwen_conflicting_duplicate_episode_ids"], 1)
    self.assertEqual(audit["qwen_rows_without_episode_id"], 1)

  def test_degenerate_kappa_is_reported_as_undefined(self) -> None:
    metrics = subject._agreement_metrics([("planning", "planning")])
    self.assertEqual(metrics["raw_agreement"], 1.0)
    self.assertIsNone(metrics["cohens_kappa"])

  def test_historical_evidence_requires_explicit_count(self) -> None:
    self.assertEqual(
        subject._historical_evidence({"status": "error"})["evidence_type"],
        "unknown",
    )
    self.assertEqual(
        subject._historical_evidence({"gemini_usage": {}})["evidence_type"],
        "unknown",
    )
    self.assertEqual(
        subject._historical_evidence(
            {"gemini_usage": {"num_images": False}}
        )["evidence_type"],
        "unknown",
    )
    self.assertEqual(
        subject._historical_evidence(
            {"gemini_usage": {"num_images": 1.5}}
        )["evidence_type"],
        "unknown",
    )
    self.assertEqual(
        subject._historical_evidence(
            {"gemini_usage": {"num_images": 0}}
        )["evidence_type"],
        "zero_image",
    )
    self.assertEqual(
        subject._historical_evidence(
            {
                "gemini_usage": {"num_images": "invalid"},
                "evidence_parity": {"historical_gemini_num_images": 3},
            }
        )["evidence_type"],
        "visual",
    )
    conflict = subject._historical_evidence(
        {
            "gemini_usage": {"num_images": 0},
            "evidence_parity": {"historical_gemini_num_images": 6},
        }
    )
    self.assertEqual(conflict["historical_num_images"], 0)
    self.assertEqual(
        conflict["evidence_count_source"], "gemini_usage.num_images"
    )
    self.assertTrue(conflict["evidence_count_conflict"])

  def test_cluster_bootstrap_is_deterministic_and_can_be_disabled(self) -> None:
    rows = []
    labels = [
        ("planning", "planning"),
        ("grounding", "grounding"),
        ("planning", "grounding"),
        ("grounding", "planning"),
    ]
    for index, (gemini, qwen) in enumerate(labels):
      rows.append(
          {
              "valid_pair": True,
              "gemini_label": gemini,
              "qwen_label": qwen,
              "model_name": "M",
              "category": "clock",
              "app_id": f"app{index // 2}",
              "task_template": f"task{index // 2}",
          }
      )
    first = subject._cluster_bootstrap(rows, 250)
    second = subject._cluster_bootstrap(rows, 250)
    permuted = subject._cluster_bootstrap(list(reversed(rows)), 250)
    self.assertEqual(first, second)
    self.assertEqual(first, permuted)
    self.assertEqual(first["clusters"], 2)
    self.assertEqual(first["raw_agreement_replicates"], 250)
    self.assertIsNotNone(first["raw_agreement_ci95"])
    self.assertGreater(first["cohens_kappa_replicates"], 0)
    disabled = subject._cluster_bootstrap(rows, 0)
    self.assertIsNone(disabled["raw_agreement_ci95"])
    self.assertEqual(disabled["raw_agreement_replicates"], 0)
    with self.assertRaises(ValueError):
      subject._cluster_bootstrap(rows, -1)
    incomplete = [dict(rows[0], app_id="")]
    with self.assertRaises(ValueError):
      subject._cluster_bootstrap(incomplete, 1)

  @staticmethod
  def _gemini(
      episode_id: str, model: str, category: str, label: str
  ) -> dict[str, str]:
    return {
        "episode_id": episode_id,
        "model_name": model,
        "category": category,
        "app_id": f"{category}_app",
        "app_name": "App",
        "task_template": f"Task{episode_id}",
        "primary_failure_mode": label,
        "confidence": "high",
        "planning_score": "1",
        "grounding_score": "2",
        "rationale": "Gemini rationale",
        "pkl_path": f"/{episode_id}.pkl.gz",
        "jsonl_path": "/gemini.jsonl",
    }

  @staticmethod
  def _judgment(label: str) -> dict[str, object]:
    return {
        "primary_failure_mode": label,
        "planning_score": 1,
        "grounding_score": 2,
        "confidence": "high",
        "rationale": "Qwen rationale",
        "evidence": ["step 1"],
    }

  @staticmethod
  def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
  unittest.main()
