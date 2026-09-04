#!/usr/bin/env python3
"""Tests for the read-only pinned model artifact audit.

Fixtures use the real primary model identity but tiny byte strings. They are
filesystem-integrity unit tests only and never stand in for checkpoint weights,
an endpoint, an inference, or a reported model result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import audit_pinned_model_artifacts as audit


MODEL_NAME = "UI-Venus-7B"
REPOSITORY = "inclusionAI/UI-Venus-Navi-7B"
REVISION = "f3c6e7264df2a3d75db2f25b3a63a6955a0f062d"


class PinnedModelArtifactAuditTest(unittest.TestCase):

  def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
    cohort = root / "cohort.json"
    config = root / "models.json"
    cache = root / "hub"
    repository_root = cache / "models--inclusionAI--UI-Venus-Navi-7B"
    blobs = repository_root / "blobs"
    snapshot = repository_root / "snapshots" / REVISION
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    weight_bytes = b"unit-test-only-not-model-weights"
    weight_sha = hashlib.sha256(weight_bytes).hexdigest()
    (blobs / weight_sha).write_bytes(weight_bytes)
    (snapshot / "model.safetensors").symlink_to(
        Path("../../blobs") / weight_sha
    )
    metadata_bytes = b'{"model_type":"qwen2_5_vl"}\n'
    metadata_blob = blobs / hashlib.sha1(metadata_bytes).hexdigest()
    metadata_blob.write_bytes(metadata_bytes)
    (snapshot / "config.json").symlink_to(
        Path("../../blobs") / metadata_blob.name
    )
    cohort.write_text(json.dumps({"models": [MODEL_NAME]}) + "\n")
    config.write_text(json.dumps({
        "models": [{
            "name": MODEL_NAME,
            "repository": REPOSITORY,
            "revision": REVISION,
        }]
    }) + "\n")
    return cohort, config, cache, snapshot

  def test_audit_hashes_exact_content_addressed_weight(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cohort, config, cache, _ = self._fixture(Path(tmpdir))
      report = audit.audit_models(
          cohort_path=cohort,
          model_config_path=config,
          cache_root=cache,
          workers=2,
      )
      self.assertTrue(report["valid"])
      self.assertEqual(report["valid_models"], 1)
      self.assertFalse(report["external_artifact_fetch_performed"])
      self.assertFalse(report["inference_endpoint_contacted"])
      self.assertIn("network_backed", report["cache_storage"])
      model = report["models"][0]
      self.assertEqual(model["repository"], REPOSITORY)
      self.assertEqual(model["weight_file_count"], 1)
      self.assertEqual(
          model["weight_files"][0]["blob_name"],
          model["weight_files"][0]["sha256"],
      )

  def test_audit_rejects_weight_symlink_outside_repository_blobs(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      cohort, config, cache, snapshot = self._fixture(root)
      (snapshot / "model.safetensors").unlink()
      outside = root / ("a" * 64)
      outside.write_bytes(b"outside")
      (snapshot / "model.safetensors").symlink_to(outside)
      with self.assertRaisesRegex(audit.AuditError, "repository's blobs"):
        audit.audit_models(
            cohort_path=cohort,
            model_config_path=config,
            cache_root=cache,
            workers=1,
        )

  def test_audit_accepts_validated_symlinked_weight_index(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cohort, config, cache, snapshot = self._fixture(Path(tmpdir))
      repository_root = snapshot.parents[1]
      index_bytes = json.dumps({
          "weight_map": {"layer.weight": "model.safetensors"}
      }).encode("utf-8")
      index_blob = repository_root / "blobs" / hashlib.sha1(index_bytes).hexdigest()
      index_blob.write_bytes(index_bytes)
      (snapshot / "model.safetensors.index.json").symlink_to(
          Path("../../blobs") / index_blob.name
      )
      report = audit.audit_models(
          cohort_path=cohort,
          model_config_path=config,
          cache_root=cache,
          workers=1,
      )
      self.assertTrue(report["valid"])
      self.assertEqual(report["models"][0]["weight_file_count"], 1)

  def test_audit_rejects_missing_weight_file(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cohort, config, cache, snapshot = self._fixture(Path(tmpdir))
      (snapshot / "model.safetensors").unlink()
      with self.assertRaisesRegex(audit.AuditError, "no model weight files"):
        audit.audit_models(
            cohort_path=cohort,
            model_config_path=config,
            cache_root=cache,
            workers=1,
        )

  def test_output_is_exclusive(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      output = Path(tmpdir) / "report.json"
      audit._write_exclusive(output, {"valid": True})
      with self.assertRaisesRegex(audit.AuditError, "refusing to overwrite"):
        audit._write_exclusive(output, {"valid": True})


if __name__ == "__main__":
  unittest.main()
