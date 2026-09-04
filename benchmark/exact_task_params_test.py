"""Tests for the fail-closed exact task-parameter interchange contract."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from absl.testing import absltest

import exact_task_params


def _payload(source: Path) -> dict[str, object]:
  return {
      "schema_version": 1,
      "mode": exact_task_params.MODE,
      "source": {
          "file": str(source.resolve()),
          "sha256": exact_task_params.file_sha256(source),
      },
      "overrides": {
          "Task1": {
              "instance_id": 0,
              "params": {"value": "alpha", "seed": 17},
              "expected_goal": "Do alpha",
              "expected_seed": 17,
          }
      },
  }


class ExactTaskParamsTest(absltest.TestCase):

  def _write(self, directory: str, payload: object) -> Path:
    path = Path(directory) / "overrides.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

  def _source(self, directory: str) -> Path:
    path = Path(directory) / "canonical.json"
    path.write_text('{"audited":true}\n', encoding="utf-8")
    return path

  def test_loads_byte_pinned_closed_schema(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = self._write(tmpdir, _payload(self._source(tmpdir)))
      digest = exact_task_params.file_sha256(path)
      bundle = exact_task_params.load_bundle(
          path, expected_sha256=digest
      )

    self.assertEqual(digest, bundle.sha256)
    self.assertEqual(exact_task_params.MODE, bundle.mode)
    self.assertEqual(17, bundle.overrides["Task1"]["expected_seed"])

  def test_rejects_hash_mismatch(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = self._write(tmpdir, _payload(self._source(tmpdir)))
      with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
        exact_task_params.load_bundle(
            path, expected_sha256="0" * 64
        )

  def test_rejects_extra_root_and_entry_fields(self):
    for location in ("root", "entry"):
      with self.subTest(location=location):
        with tempfile.TemporaryDirectory() as tmpdir:
          payload = _payload(self._source(tmpdir))
          if location == "root":
            payload["unexpected"] = True
          else:
            payload["overrides"]["Task1"]["unexpected"] = True
          path = self._write(tmpdir, payload)
          with self.assertRaisesRegex(ValueError, "fields mismatch"):
            exact_task_params.load_bundle(
                path,
                expected_sha256=exact_task_params.file_sha256(path),
            )

  def test_rejects_seed_mismatch(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      payload = _payload(self._source(tmpdir))
      payload["overrides"]["Task1"]["expected_seed"] = 18
      path = self._write(tmpdir, payload)
      with self.assertRaisesRegex(ValueError, "seed mismatch"):
        exact_task_params.load_bundle(
            path, expected_sha256=exact_task_params.file_sha256(path)
        )

  def test_requires_exact_task_set_and_registry_membership(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = self._write(tmpdir, _payload(self._source(tmpdir)))
      bundle = exact_task_params.load_bundle(
          path, expected_sha256=exact_task_params.file_sha256(path)
      )
      with self.assertRaisesRegex(ValueError, "missing=.*Task2"):
        exact_task_params.require_exact_task_names(
            bundle, ["Task1", "Task2"]
        )
      with self.assertRaisesRegex(ValueError, "absent from the registry"):
        exact_task_params.require_exact_task_names(
            bundle, ["Task1"], registry_names=["DifferentTask"]
        )


if __name__ == "__main__":
  absltest.main()
