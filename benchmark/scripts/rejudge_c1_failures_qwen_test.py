"""Focused offline tests for the paired Qwen C1 failure rejudge runner."""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import rejudge_c1_failures_qwen as runner


VALID_JUDGMENT = {
    "primary_failure_mode": "planning",
    "planning_score": 3,
    "grounding_score": 0,
    "confidence": "high",
    "rationale": "The plan omitted a required operation.",
    "evidence": ["step 2"],
}


class RejudgeRunnerTest(unittest.TestCase):

  def _case(
      self,
      root: Path,
      *,
      episode_id: str = "episode-a",
      historical_images: int = 1,
  ) -> runner.FrozenCase:
    pkl_path = root / f"{episode_id}.pkl.gz"
    pkl_path.touch()
    jsonl_path = root / "failure_mode_judgments.jsonl"
    flat = {
        "source": "fixture",
        "episode_id": episode_id,
        "model_name": "FixtureAgent",
        "category": "sms",
        "app_id": "fixture.sms",
        "app_name": "Fixture SMS",
        "task_template": "FixtureSendSms",
        "pkl_path": str(pkl_path),
        "jsonl_path": str(jsonl_path),
    }
    payload = {
        **{key: flat[key] for key in (
            "episode_id",
            "model_name",
            "category",
            "app_id",
            "app_name",
            "task_template",
            "pkl_path",
        )},
        "goal": "Send a fixture message.",
        "key_step_indices": [0, 2],
        "steps": [{"step": 1}, {"step": 3}],
    }
    gemini_row = {
        **flat,
        "case_payload": payload,
        "goal": payload["goal"],
        "is_successful": 0.0,
        "usage": {"num_images": historical_images},
        "judgment": dict(VALID_JUDGMENT),
    }
    return runner.FrozenCase(flat=flat, gemini_row=gemini_row)

  def test_load_frozen_cases_joins_exact_source_row(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      case = self._case(root)
      Path(case.flat["jsonl_path"]).write_text(
          json.dumps(case.gemini_row) + "\n", encoding="utf-8"
      )
      csv_path = root / "merged.csv"
      with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case.flat))
        writer.writeheader()
        writer.writerow(case.flat)

      loaded = runner._load_frozen_cases(csv_path, {"sms"}, expected_n=1)

      self.assertEqual([case.episode_id], [item.episode_id for item in loaded])
      self.assertEqual(case.gemini_row["case_payload"], loaded[0].gemini_row["case_payload"])

  def test_validate_judgment_rejects_bool_score_and_extra_field(self) -> None:
    invalid_bool = {**VALID_JUDGMENT, "planning_score": True}
    with self.assertRaisesRegex(ValueError, "planning_score"):
      runner._validate_judgment(invalid_bool)

    invalid_extra = {**VALID_JUDGMENT, "unconstrained_note": "not in schema"}
    with self.assertRaisesRegex(ValueError, "unexpected fields"):
      runner._validate_judgment(invalid_extra)

  def test_config_hash_preserves_v1_resume_and_ignores_runtime_tuning(self) -> None:
    config = {
        "runner_version": runner.RUNNER_VERSION,
        "merged_csv_sha256": "merged-a",
        "historical_jsonl_sha256": {"source.jsonl": "jsonl-a"},
        "historical_judge_config_sha256": {"judge_config.json": "config-a"},
        "model": "catbench-judge",
        "system_prompt_sha1": "prompt-a",
        "output_schema_sha1": "schema-a",
        "evidence": {"max_frames": 6},
        "workers": 4,
        "base_url": "https://first.invalid/v1",
    }
    first = runner._config_hash(config)
    operational_change = {**config, "workers": 12, "base_url": "https://second.invalid/v1"}
    roster_change = {
        **config,
        "merged_csv_sha256": "merged-b",
    }

    self.assertEqual(first, runner._config_hash(operational_change))
    # Historical hashes are recorded and audited, but intentionally do not
    # alter the v1 key: this keeps already-produced d018... caches resumable.
    source_change = {
        **config,
        "historical_jsonl_sha256": {"source.jsonl": "jsonl-b"},
    }
    self.assertEqual(first, runner._config_hash(source_change))
    self.assertNotEqual(first, runner._config_hash(roster_change))

  def test_historical_artifact_audit_checks_original_prompt_and_settings(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      case = self._case(root)
      jsonl_path = Path(case.flat["jsonl_path"])
      jsonl_path.write_text(json.dumps(case.gemini_row) + "\n", encoding="utf-8")
      expected_config = {
          "judge_backend": "gemini",
          "system_prompt_sha1": hashlib.sha1(
              runner.classifier.SYSTEM_PROMPT.encode("utf-8")
          ).hexdigest(),
          "with_screenshots": True,
          "screenshot_max_frames": 6,
          "max_steps": 6,
          "smart_steps": True,
          "screenshot_max_dim": 896,
      }
      config_path = root / "judge_config.json"
      config_path.write_text(json.dumps(expected_config), encoding="utf-8")

      jsonl_hashes, config_hashes = runner._historical_artifact_hashes([case])

      self.assertIn(str(jsonl_path), jsonl_hashes)
      self.assertIn(str(config_path), config_hashes)
      config_path.write_text(
          json.dumps({**expected_config, "max_steps": 5}), encoding="utf-8"
      )
      with self.assertRaisesRegex(ValueError, "max_steps"):
        runner._historical_artifact_hashes([case])

  def test_judge_one_writes_secret_free_cache_and_resumes_without_call(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      case = self._case(root, historical_images=1)
      image = {"step": 1, "field": "screenshot", "jpeg_base64": "YWJj"}
      usage = {"num_images": 1, "model": "fixture-served-model"}
      secret = "fixture-secret-token"
      with (
          mock.patch.object(runner, "_resolve_episode", return_value=({"episode_data": {}}, 0)),
          mock.patch.object(
              runner.classifier,
              "_extract_screenshots_for_judge",
              return_value=[image],
          ),
          mock.patch.object(
              runner.classifier,
              "_call_judge",
              return_value=(dict(VALID_JUDGMENT), usage),
          ) as call_judge,
      ):
        result = runner._judge_one(
            case,
            out_dir=root,
            config_hash="config-a",
            model="fixture-qwen",
            base_url="https://fixture.invalid/v1",
            api_key=secret,
            timeout_sec=10,
            max_retries=0,
            resume=True,
        )

      self.assertEqual("fixture-qwen", result["judge_model"])
      self.assertFalse(result["cache_hit"])
      self.assertEqual(
          hashlib.sha256(b"YWJj").hexdigest(),
          result["evidence_parity"]["qwen_image_sha256"][0],
      )
      self.assertIs(
          case.gemini_row["case_payload"],
          call_judge.call_args.kwargs["case_payload"],
      )
      cache_path = root / "cache" / "episode-a_config-a.json"
      self.assertNotIn(secret, cache_path.read_text(encoding="utf-8"))

      with mock.patch.object(
          runner.classifier,
          "_call_judge",
          side_effect=AssertionError("resume unexpectedly called the endpoint"),
      ):
        resumed = runner._judge_one(
            case,
            out_dir=root,
            config_hash="config-a",
            model="fixture-qwen",
            base_url="https://fixture.invalid/v1",
            api_key=secret,
            timeout_sec=10,
            max_retries=0,
            resume=True,
        )
      self.assertTrue(resumed["cache_hit"])

  def test_legacy_cache_is_fingerprinted_without_rejudging(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      case = self._case(root, historical_images=0)
      legacy = {
          **runner._result_metadata(case, "config-a", "fixture-qwen"),
          "status": "ok",
          "cache_hit": False,
          "evidence_parity": {"image_count_matches": True},
          "qwen_judgment": dict(VALID_JUDGMENT),
      }
      cache_path = root / "cache" / "episode-a_config-a.json"
      runner._atomic_json(cache_path, legacy)

      with mock.patch.object(
          runner.classifier,
          "_call_judge",
          side_effect=AssertionError("legacy migration called the endpoint"),
      ):
        resumed = runner._judge_one(
            case,
            out_dir=root,
            config_hash="config-a",
            model="fixture-qwen",
            base_url="https://fixture.invalid/v1",
            api_key="secret",
            timeout_sec=10,
            max_retries=0,
            resume=True,
        )

      self.assertTrue(resumed["cache_hit"])
      self.assertTrue(resumed["cache_format_migrated"])
      self.assertEqual(
          runner._input_fingerprint(case), resumed["input_fingerprint"]
      )
      persisted = json.loads(cache_path.read_text(encoding="utf-8"))
      self.assertEqual(resumed["input_fingerprint"], persisted["input_fingerprint"])

  def test_invalid_cache_is_ignored_and_rejudged(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      case = self._case(root, historical_images=0)
      cache_path = root / "cache" / "episode-a_config-a.json"
      runner._atomic_json(
          cache_path,
          {
              **runner._result_metadata(case, "config-a", "fixture-qwen"),
              "status": "ok",
              "evidence_parity": {"image_count_matches": True},
              "qwen_judgment": {**VALID_JUDGMENT, "planning_score": 99},
          },
      )
      with (
          mock.patch.object(runner, "_resolve_episode", return_value=({"episode_data": {}}, 0)),
          mock.patch.object(
              runner.classifier, "_extract_screenshots_for_judge", return_value=[]
          ),
          mock.patch.object(
              runner.classifier,
              "_call_judge",
              return_value=(dict(VALID_JUDGMENT), {"num_images": 0}),
          ) as call_judge,
      ):
        result = runner._judge_one(
            case,
            out_dir=root,
            config_hash="config-a",
            model="fixture-qwen",
            base_url="https://fixture.invalid/v1",
            api_key="secret",
            timeout_sec=10,
            max_retries=0,
            resume=True,
        )

      call_judge.assert_called_once()
      self.assertFalse(result["cache_hit"])

  def test_image_count_mismatch_fails_before_endpoint_call(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      case = self._case(root, historical_images=1)
      with (
          mock.patch.object(runner, "_resolve_episode", return_value=({"episode_data": {}}, 0)),
          mock.patch.object(
              runner.classifier, "_extract_screenshots_for_judge", return_value=[]
          ),
          mock.patch.object(runner.classifier, "_call_judge") as call_judge,
      ):
        with self.assertRaisesRegex(ValueError, "Image-parity failure"):
          runner._judge_one(
              case,
              out_dir=root,
              config_hash="config-a",
              model="fixture-qwen",
              base_url="https://fixture.invalid/v1",
              api_key="secret",
              timeout_sec=10,
              max_retries=0,
              resume=False,
          )
      call_judge.assert_not_called()

  def test_error_artifact_redacts_api_key(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      case = self._case(Path(tmp))
      secret = "do-not-persist-this-key"
      row = runner._error_result(
          case,
          "config-a",
          "fixture-qwen",
          RuntimeError(f"Authorization: Bearer {secret}"),
          secrets=(secret,),
      )
      self.assertNotIn(secret, json.dumps(row))
      self.assertIn("<redacted>", row["qwen_error"])

  def test_atomic_json_remains_valid_under_concurrent_writers(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "shared.json"
      with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(runner._atomic_json, path, {"writer": index, "blob": "x" * 1000})
            for index in range(32)
        ]
        for future in futures:
          future.result()

      payload = json.loads(path.read_text(encoding="utf-8"))
      self.assertIn(payload["writer"], range(32))
      self.assertEqual("x" * 1000, payload["blob"])


if __name__ == "__main__":
  unittest.main()
