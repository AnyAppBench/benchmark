"""Tests for fail-closed C2 planner-artifact comparison."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

import compare_c2_planner_artifacts as subject
import task_breakdowns


def _sha(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CompareC2PlannerArtifactsTest(unittest.TestCase):

  def _payload(
      self,
      *,
      provider: str,
      model: str,
      model_identity: str,
      plan_prefix: str,
  ) -> dict:
    categories = (
        ("sms", 4),
        ("files", 5),
        ("maps", 3),
        ("contacts", 5),
        ("clock", 6),
    )
    entries = []
    for category, app_count in categories:
      for task_index in range(10):
        semantic_task_id = f"{category.title()}Task{task_index}"
        for instance_id in range(3):
          semantic_goal = (
              f"Complete {category} task {task_index} instance {instance_id} "
              "in [TARGET_APP]"
          )
          semantic_goal_sha = task_breakdowns.goal_sha256(semantic_goal)
          parameter_sha = _sha(
              f"parameters|{category}|{task_index}|{instance_id}"
          )
          plan_key = task_breakdowns.make_semantic_plan_key(
              semantic_task_id, instance_id, semantic_goal
          )
          breakdown = {
              "steps": [
                  f"{plan_prefix} prepare task {task_index}.",
                  f"{plan_prefix} preserve instance {instance_id} values.",
                  f"{plan_prefix} complete and verify the outcome.",
              ],
              "notes": [],
          }
          breakdown_text = task_breakdowns.format_breakdown_text(
              {"breakdown": breakdown}
          )
          plan_sha = _sha(breakdown_text)
          for app_index in range(app_count):
            display_name = f"{category.title()}App{app_index}"
            goal = semantic_goal.replace("[TARGET_APP]", display_name)
            task_template = f"{semantic_task_id}For{display_name}"
            entries.append({
                "key": task_breakdowns.make_key(
                    task_template, goal, instance_id
                ),
                "task_template": task_template,
                "instance_id": instance_id,
                "goal": goal,
                "goal_sha256": task_breakdowns.goal_sha256(goal),
                "semantic_task_id": semantic_task_id,
                "app_display_name": display_name,
                "semantic_goal": semantic_goal,
                "semantic_goal_sha256": semantic_goal_sha,
                "semantic_parameter_sha256": parameter_sha,
                "plan_key": plan_key,
                "plan_sha256": plan_sha,
                "generator_provider": provider,
                "generator_model": model,
                "generator_model_identity": model_identity,
                "breakdown": copy.deepcopy(breakdown),
                "breakdown_text": breakdown_text,
                "validation_warnings": [],
            })
    self.assertEqual(len(entries), 690)
    self.assertEqual(len({entry["plan_key"] for entry in entries}), 150)
    return {
        "metadata": {
            "generator_provider": provider,
            "generator_model": model,
            "generator_model_identity": model_identity,
            "prompt_sha256": "a" * 64,
            "cohort_release_id": "catbench_acl_revision_5cat_v1",
            "cohort_manifest_sha256": "b" * 64,
            "suite_family": "android_world",
            "categories": [item[0] for item in categories],
            "tasks": [],
            "n_task_combinations": 3,
            "task_random_seed": 30,
            "fixed_task_seed": False,
            "generation_policy": {
                "temperature": 0.0,
                "max_retry": 3,
                "timeout_sec": 120.0,
                "sleep_seconds": 0.0,
                "validation_retry": 3,
                "strict_forbidden_check": True,
                "response_contract": (
                    "provider_json_mode_then_common_schema_and_forbidden_"
                    "detail_validation"
                ),
                "selection_policy": (
                    "first_accepted_machine_valid_plan_no_best_of_n"
                ),
            },
            "semantic_pairing_version": 2,
            "plan_reuse_policy": (
                "one_plan_per_semantic_instance_across_apps"
            ),
            "planner_input_app_identity": "replaced_with_[TARGET_APP]",
            "expected_entry_count": 690,
            "expected_semantic_plan_count": 150,
            "attempt_audit": {
                "header_sha256": "c" * 64,
                "tail_sha256": "d" * 64,
                "generator_config_sha256": "e" * 64,
                "record_count": 151,
            },
        },
        "breakdowns": entries,
    }

  def _write(self, path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")

  def _valid_pair(self, root: Path) -> tuple[Path, Path, dict, dict]:
    gemini = self._payload(
        provider="gemini",
        model="gemini-3.1-pro-preview",
        model_identity="gemini-3.1-pro-preview",
        plan_prefix="Gemini",
    )
    qwen = self._payload(
        provider="qwen",
        model="catbench-judge",
        model_identity="Qwen/Qwen3-VL-30B-A3B-Instruct",
        plan_prefix="Qwen",
    )
    gemini_path = root / "gemini.json"
    qwen_path = root / "qwen.json"
    self._write(gemini_path, gemini)
    self._write(qwen_path, qwen)
    return gemini_path, qwen_path, gemini, qwen

  def test_valid_pair_reports_identity_and_text_counts_without_quality(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      gemini_path, qwen_path, _, _ = self._valid_pair(Path(temp))

      report = subject.compare_artifacts(gemini_path, qwen_path)

    self.assertEqual(report["status"], "pass")
    self.assertFalse(report["quality_assessment_performed"])
    self.assertEqual(
        report["artifacts"]["qwen"]["model_identity"],
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
    )
    self.assertEqual(report["plan_text_comparison"]["identical_count"], 0)
    self.assertEqual(report["plan_text_comparison"]["different_count"], 150)

  def test_rejects_prompt_mismatch(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      gemini_path, qwen_path, _, qwen = self._valid_pair(root)
      qwen["metadata"]["prompt_sha256"] = "f" * 64
      self._write(qwen_path, qwen)

      with self.assertRaisesRegex(subject.ComparisonError, "prompt_sha256"):
        subject.compare_artifacts(gemini_path, qwen_path)

  def test_rejects_generation_policy_drift(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      gemini_path, qwen_path, _, qwen = self._valid_pair(root)
      qwen["metadata"]["generation_policy"]["temperature"] = 0.7
      self._write(qwen_path, qwen)

      with self.assertRaisesRegex(subject.ComparisonError, "generation_policy"):
        subject.compare_artifacts(gemini_path, qwen_path)

  def test_rejects_nonidentical_cross_app_plan_reuse(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      gemini_path, qwen_path, _, qwen = self._valid_pair(root)
      entry = qwen["breakdowns"][1]
      entry["breakdown"]["steps"][0] = "Changed for only this app."
      entry["breakdown_text"] = task_breakdowns.format_breakdown_text(entry)
      entry["plan_sha256"] = _sha(entry["breakdown_text"])
      self._write(qwen_path, qwen)

      with self.assertRaisesRegex(subject.ComparisonError, "byte-identically"):
        subject.compare_artifacts(gemini_path, qwen_path)

  def test_rejects_any_validation_warning(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      gemini_path, qwen_path, _, qwen = self._valid_pair(root)
      qwen["breakdowns"][0]["validation_warnings"] = ["app_name_mention"]
      self._write(qwen_path, qwen)

      with self.assertRaisesRegex(subject.ComparisonError, "non-empty"):
        subject.compare_artifacts(gemini_path, qwen_path)


if __name__ == "__main__":
  unittest.main()
