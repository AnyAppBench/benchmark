"""Tests for the exact-cell CATBench diagnostic runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from unittest import mock

from absl.testing import absltest

import exact_task_params
import run_catbench_5cat_matrix as matrix
import run_catbench_target_cells as target_cells


class BuildJobsTest(absltest.TestCase):

  def test_build_jobs_passes_pinned_app_metadata_to_matrix_builder(self):
    models = matrix._load_models(  # pylint: disable=protected-access
        matrix.CONFIG_PATH, {"UI-Venus-7B"}
    )
    app_pins = matrix._load_app_pins(  # pylint: disable=protected-access
        matrix.APP_PINS_PATH
    )
    target = (
        "UI-Venus-7B",
        "sms",
        "sms_simple_sms_messenger",
        "SmsSendForSimpleSMSMessenger",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      jobs = target_cells._build_jobs(  # pylint: disable=protected-access
          targets=[target],
          models=models,
          app_pins=app_pins,
          output_root=Path(tmpdir),
          run_id="diagnostic",
          python_bin="python",
          n_task_combinations=3,
          task_random_seed=30,
          runner_args=[],
          resume_existing=False,
          condition="c1",
          instance_id=2,
          code_revision="b" * 40,
          source_snapshot_sha256="c" * 64,
          release_purpose="revision_rerun_candidate",
          artifact_role="invalid_episode_replacement_candidate",
          analysis_eligible=False,
          model_config_sha256="d" * 64,
          app_pins_sha256="e" * 64,
      )

    self.assertLen(jobs, 1)
    job = jobs[0]
    pin = app_pins[("sms", "sms_simple_sms_messenger")]
    self.assertEqual(pin["package_name"], job.package_name)
    self.assertEqual(pin["version_name"], job.app_version)
    self.assertEqual(pin["version_code"], job.app_version_code)
    self.assertEqual(pin["apk_sha256"], job.apk_sha256)
    self.assertEqual((target[3],), job.task_templates)
    self.assertEqual("c1", job.condition)
    self.assertEqual(2, job.instance_id)
    self.assertEqual("b" * 40, job.code_revision)
    self.assertEqual("c" * 64, job.source_snapshot_sha256)
    self.assertEqual("revision_rerun_candidate", job.release_purpose)
    self.assertEqual(
        "invalid_episode_replacement_candidate", job.artifact_role
    )
    self.assertFalse(job.analysis_eligible)
    self.assertEqual("d" * 64, job.model_config_sha256)
    self.assertEqual("e" * 64, job.app_pins_sha256)
    self.assertIn(f"--tasks={target[3]}", job.command)

  def test_dry_run_manifest_serializes_candidate_provenance(self):
    target = (
        "UI-Venus-7B|sms|sms_simple_sms_messenger|"
        "SmsSendForSimpleSMSMessenger"
    )
    source_snapshot = "f" * 64
    with tempfile.TemporaryDirectory() as tmpdir:
      argv = [
          "run_catbench_target_cells.py",
          "--env_file=/dev/null",
          f"--target={target}",
          "--emulators=5800:8800:-:5039",
          f"--output_root={tmpdir}",
          "--run_id=revision_candidate",
          "--condition=c1",
          "--n_task_combinations=3",
          "--task_random_seed=30",
          "--instance_id=2",
          f"--source_snapshot_sha256={source_snapshot}",
          "--release_purpose=revision_rerun_candidate",
          "--artifact_role=invalid_episode_replacement_candidate",
          "--analysis_eligible=false",
          "--dry_run",
      ]
      with mock.patch.object(target_cells.sys, "argv", argv):
        with mock.patch.dict(
            os.environ,
            {
                "CATBENCH_INSTANCE_ID": "",
                "CATBENCH_SOURCE_SNAPSHOT_SHA256": "",
                "CATBENCH_TASK_BREAKDOWN_FILE": "",
                "CATBENCH_TASK_BREAKDOWN_MODE": "",
                "CATBENCH_TASK_BREAKDOWN_REQUIRED": "",
            },
            clear=False,
        ):
          self.assertEqual(0, target_cells.main())

      manifest = json.loads(
          (Path(tmpdir) / "revision_candidate" /
           "catbench_5cat_manifest.json").read_text(encoding="utf-8")
      )

    self.assertEqual("revision_rerun_candidate", manifest["release_purpose"])
    self.assertEqual(
        "invalid_episode_replacement_candidate", manifest["artifact_role"]
    )
    self.assertFalse(manifest["analysis_eligible"])
    self.assertEqual(source_snapshot, manifest["source_snapshot_sha256"])
    self.assertEqual(2, manifest["matrix_args"]["instance_id"])
    self.assertEqual(2, manifest["matrix_args"]["catbench_instance_id"])
    self.assertLen(manifest["jobs"], 1)
    job = manifest["jobs"][0]
    self.assertEqual(2, job["instance_id"])
    self.assertEqual(source_snapshot, job["source_snapshot_sha256"])
    self.assertEqual(manifest["model_config_sha256"], job["model_config_sha256"])
    self.assertEqual(manifest["app_pins_sha256"], job["app_pins_sha256"])
    self.assertFalse(job["analysis_eligible"])

  def test_target_runner_cannot_self_promote_candidate(self):
    argv = [
        "run_catbench_target_cells.py",
        "--env_file=/dev/null",
        "--emulators=5800:8800:-:5039",
        "--run_id=must_fail",
        "--analysis_eligible=true",
        "--dry_run",
    ]
    with mock.patch.object(target_cells.sys, "argv", argv):
      with mock.patch.dict(
          os.environ,
          {
              "CATBENCH_INSTANCE_ID": "",
              "CATBENCH_SOURCE_SNAPSHOT_SHA256": "",
          },
          clear=False,
      ):
        with self.assertRaisesRegex(ValueError, "must remain"):
          target_cells.main()

  def test_exact_override_is_projected_and_bound_into_manifest_jobs(self):
    target_task = "SmsSendForSimpleSMSMessenger"
    target = (
        "UI-Venus-7B|sms|sms_simple_sms_messenger|" + target_task
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      source = Path(tmpdir) / "canonical.json"
      source.write_text('{"audited":true}\n', encoding="utf-8")
      source_hash = exact_task_params.file_sha256(source)
      payload = {
          "schema_version": 1,
          "mode": exact_task_params.MODE,
          "source": {"file": str(source), "sha256": source_hash},
          "overrides": {
              target_task: {
                  "instance_id": 0,
                  "params": {"number": "123", "message": "hello", "seed": 9},
                  "expected_goal": (
                      "Using the Simple SMS Messenger app, send an SMS to 123 "
                      "with the message: `hello`."
                  ),
                  "expected_seed": 9,
              }
          },
      }
      override_file = Path(tmpdir) / "override.json"
      override_file.write_text(json.dumps(payload), encoding="utf-8")
      override_hash = exact_task_params.file_sha256(override_file)
      argv = [
          "run_catbench_target_cells.py",
          "--env_file=/dev/null",
          f"--target={target}",
          f"--exact_task_params_file={override_file}",
          f"--exact_task_params_sha256={override_hash}",
          "--emulators=5800:8800:-:5039",
          f"--output_root={tmpdir}",
          "--run_id=exact_candidate",
          "--condition=c1",
          "--n_task_combinations=1",
          "--task_random_seed=30",
          "--instance_id=0",
          "--dry_run",
      ]
      with mock.patch.object(target_cells.sys, "argv", argv):
        with mock.patch.dict(
            os.environ,
            {
                **{name: "" for name in exact_task_params.ENV_NAMES},
                "CATBENCH_INSTANCE_ID": "",
                "CATBENCH_SOURCE_SNAPSHOT_SHA256": "",
                "CATBENCH_TASK_BREAKDOWN_FILE": "",
                "CATBENCH_TASK_BREAKDOWN_MODE": "",
                "CATBENCH_TASK_BREAKDOWN_REQUIRED": "",
            },
            clear=False,
        ):
          self.assertEqual(0, target_cells.main())

      manifest = json.loads(
          (Path(tmpdir) / "exact_candidate" /
           "catbench_5cat_manifest.json").read_text(encoding="utf-8")
      )
      override = manifest["exact_task_params_override"]
      self.assertTrue(override["enabled"])
      self.assertEqual(exact_task_params.MODE, override["mode"])
      self.assertEqual(str(override_file.resolve()), override["override_file"])
      self.assertEqual(override_hash, override["override_sha256"])
      self.assertEqual(str(source.resolve()), override["source_file"])
      self.assertEqual(source_hash, override["source_sha256"])
      self.assertLen(override["effective_job_files"], 1)
      job = manifest["jobs"][0]
      effective = Path(job["exact_task_params_override_file"])
      self.assertTrue(effective.is_file())
      self.assertEqual(
          exact_task_params.file_sha256(effective),
          job["exact_task_params_override_sha256"],
      )
      self.assertEqual(
          exact_task_params.MODE, job["exact_task_params_override_mode"]
      )
      self.assertTrue(job["exact_goal_override_enabled"])
      self.assertEqual(
          job["exact_task_params_override_sha256"],
          job["exact_goal_mapping_sha256"],
      )
      self.assertEqual(str(source.resolve()), job["exact_task_params_source_file"])
      projected = json.loads(effective.read_text(encoding="utf-8"))
      self.assertEqual([target_task], list(projected["overrides"]))

  def test_exact_override_rejects_claimed_k_greater_than_one(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      source = Path(tmpdir) / "canonical.json"
      source.write_text('{"audited":true}\n', encoding="utf-8")
      payload = {
          "schema_version": 1,
          "mode": exact_task_params.MODE,
          "source": {
              "file": str(source),
              "sha256": exact_task_params.file_sha256(source),
          },
          "overrides": {
              "SmsSendForSimpleSMSMessenger": {
                  "instance_id": 0,
                  "params": {"seed": 9},
                  "expected_goal": "goal",
                  "expected_seed": 9,
              }
          },
      }
      override_file = Path(tmpdir) / "override.json"
      override_file.write_text(json.dumps(payload), encoding="utf-8")
      argv = [
          "run_catbench_target_cells.py",
          "--env_file=/dev/null",
          f"--exact_task_params_file={override_file}",
          "--exact_task_params_sha256="
          + exact_task_params.file_sha256(override_file),
          "--emulators=5800:8800:-:5039",
          "--run_id=must_fail",
          "--condition=c1",
          "--n_task_combinations=3",
          "--instance_id=0",
          "--dry_run",
      ]
      with mock.patch.object(target_cells.sys, "argv", argv):
        with mock.patch.dict(
            os.environ,
            {
                **{name: "" for name in exact_task_params.ENV_NAMES},
                "CATBENCH_INSTANCE_ID": "",
                "CATBENCH_SOURCE_SNAPSHOT_SHA256": "",
            },
            clear=False,
        ):
          with self.assertRaisesRegex(ValueError, "n_task_combinations=1"):
            target_cells.main()

  def test_exact_exclusion_preserves_full_source_set_gate(self):
    sms_task = "SmsSendForSimpleSMSMessenger"
    blocked_task = "MapsRecordTrackForCoMaps"
    with tempfile.TemporaryDirectory() as tmpdir:
      source = Path(tmpdir) / "canonical.json"
      source.write_text('{"audited":true}\n', encoding="utf-8")
      payload = {
          "schema_version": 1,
          "mode": exact_task_params.MODE,
          "source": {
              "file": str(source),
              "sha256": exact_task_params.file_sha256(source),
          },
          "overrides": {
              sms_task: {
                  "instance_id": 0,
                  "params": {
                      "number": "123", "message": "hello", "seed": 9
                  },
                  "expected_goal": "Archived SMS instruction.",
                  "expected_seed": 9,
              },
              blocked_task: {
                  "instance_id": 0,
                  "params": {"track_name": "Old Track", "seed": 10},
                  "expected_goal": "Archived track instruction.",
                  "expected_seed": 10,
              },
          },
      }
      override_file = Path(tmpdir) / "override.json"
      override_file.write_text(json.dumps(payload), encoding="utf-8")
      argv = [
          "run_catbench_target_cells.py",
          "--env_file=/dev/null",
          "--target=UI-Venus-7B|sms|sms_simple_sms_messenger|" + sms_task,
          "--target=GUI-Owl-7B|maps|maps_comaps|" + blocked_task,
          f"--exact_task_params_file={override_file}",
          "--exact_task_params_sha256="
          + exact_task_params.file_sha256(override_file),
          "--exclude_exact_task=" + blocked_task,
          "--emulators=5800:8800:-:5039",
          f"--output_root={tmpdir}",
          "--run_id=excluded_incompatible",
          "--condition=c1",
          "--n_task_combinations=1",
          "--instance_id=0",
          "--dry_run",
      ]
      with mock.patch.object(target_cells.sys, "argv", argv):
        with mock.patch.dict(
            os.environ,
            {
                **{name: "" for name in exact_task_params.ENV_NAMES},
                "CATBENCH_INSTANCE_ID": "",
                "CATBENCH_SOURCE_SNAPSHOT_SHA256": "",
                "CATBENCH_TASK_BREAKDOWN_FILE": "",
                "CATBENCH_TASK_BREAKDOWN_MODE": "",
                "CATBENCH_TASK_BREAKDOWN_REQUIRED": "",
            },
            clear=False,
        ):
          self.assertEqual(0, target_cells.main())
      manifest = json.loads(
          (Path(tmpdir) / "excluded_incompatible" /
           "catbench_5cat_manifest.json").read_text(encoding="utf-8")
      )
      override = manifest["exact_task_params_override"]
      self.assertEqual([blocked_task], override["excluded_task_classes"])
      self.assertEqual(1, override["excluded_target_count"])
      self.assertLen(manifest["jobs"], 1)
      self.assertEqual((sms_task,), tuple(manifest["jobs"][0]["task_templates"]))


if __name__ == "__main__":
  absltest.main()
