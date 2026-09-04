#!/usr/bin/env python3
"""Mechanically project an audited historical candidate into runtime schema.

The script never generates or repairs a value. It verifies every available
hash/identity field in the canonical candidate, requires duplicate task-class
rows to agree exactly, and emits the closed schema consumed by the target-only
override path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

import exact_task_params  # pylint: disable=wrong-import-position


def _canonical_json(value: Any) -> str:
  return json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
  )


def _sha256_text(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build(source: Path, expected_source_sha256: str) -> dict[str, Any]:
  source = source.expanduser().resolve()
  expected_hash = expected_source_sha256.strip().lower()
  actual_hash = exact_task_params.file_sha256(source)
  if actual_hash != expected_hash:
    raise ValueError(
        f"Canonical source SHA-256 mismatch: expected={expected_hash} "
        f"actual={actual_hash}"
    )
  payload = json.loads(source.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError("Canonical candidate root must be an object.")
  if payload.get("schema_version") != 1:
    raise ValueError("Canonical candidate schema_version must be 1.")
  if payload.get("condition") != "C1":
    raise ValueError("Canonical candidate condition must be C1.")
  if payload.get("analysis_eligible") is not False:
    raise ValueError("Canonical candidate must be analysis-ineligible.")
  if payload.get("instance_id") != 0:
    raise ValueError("Canonical candidate must contain instance_id=0.")
  if payload.get("n_task_combinations") != 1:
    raise ValueError("Canonical candidate must come from a K=1 source run.")
  if payload.get("task_random_seed") != 30:
    raise ValueError("Canonical candidate task_random_seed must be 30.")
  rows = payload.get("rows")
  if not isinstance(rows, list) or not rows:
    raise ValueError("Canonical candidate rows must be a non-empty list.")
  if payload.get("row_count") != len(rows):
    raise ValueError("Canonical candidate row_count does not match rows.")
  rows_hash = _sha256_text(_canonical_json(rows))
  if rows_hash != payload.get("rows_canonical_sha256"):
    raise ValueError("Canonical candidate rows_canonical_sha256 is invalid.")

  overrides: dict[str, dict[str, Any]] = {}
  identities: set[str] = set()
  for index, row in enumerate(rows):
    if not isinstance(row, dict):
      raise ValueError(f"Canonical row #{index + 1} must be an object.")
    task_name = row.get("task_template")
    if not isinstance(task_name, str) or not task_name:
      raise ValueError(f"Canonical row #{index + 1} lacks task_template.")
    identity = row.get("identity")
    expected_identity = "|".join(
        str(row.get(field) or "")
        for field in ("model", "category", "app_id", "task_template")
    )
    if identity != expected_identity or identity in identities:
      raise ValueError(f"Canonical row identity is invalid/duplicate: {identity!r}")
    identities.add(identity)
    if row.get("instance_id") != 0:
      raise ValueError(f"Canonical row {identity} is not instance 0.")
    params = row.get("params")
    goal = row.get("goal")
    seed = row.get("seed")
    if not isinstance(params, dict):
      raise ValueError(f"Canonical row {identity} params must be an object.")
    if not isinstance(goal, str) or not goal:
      raise ValueError(f"Canonical row {identity} goal must be non-empty.")
    if isinstance(seed, bool) or not isinstance(seed, int):
      raise ValueError(f"Canonical row {identity} seed must be an integer.")
    if params.get("seed") != seed:
      raise ValueError(f"Canonical row {identity} params.seed mismatch.")
    params_json = _canonical_json(params)
    if row.get("params_canonical_json") != params_json:
      raise ValueError(f"Canonical row {identity} params JSON mismatch.")
    if row.get("params_sha256") != _sha256_text(params_json):
      raise ValueError(f"Canonical row {identity} params SHA-256 mismatch.")
    without_seed = dict(params)
    without_seed.pop("seed", None)
    if row.get("params_without_seed_sha256") != _sha256_text(
        _canonical_json(without_seed)
    ):
      raise ValueError(
          f"Canonical row {identity} seed-free params SHA-256 mismatch."
      )
    if row.get("goal_sha256") != _sha256_text(goal):
      raise ValueError(f"Canonical row {identity} goal SHA-256 mismatch.")
    entry = {
        "instance_id": 0,
        "params": params,
        "expected_goal": goal,
        "expected_seed": seed,
    }
    existing = overrides.get(task_name)
    if existing is not None and existing != entry:
      raise ValueError(
          f"Duplicate task class {task_name} has conflicting exact values."
      )
    overrides[task_name] = entry

  if payload.get("unique_identity_count") != len(identities):
    raise ValueError("Canonical candidate unique_identity_count is invalid.")
  if payload.get("unique_task_template_count") != len(overrides):
    raise ValueError("Canonical candidate unique_task_template_count is invalid.")
  return {
      "schema_version": exact_task_params.SCHEMA_VERSION,
      "mode": exact_task_params.MODE,
      "source": {"file": str(source), "sha256": actual_hash},
      "overrides": {
          name: overrides[name] for name in sorted(overrides)
      },
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", required=True)
  parser.add_argument("--source_sha256", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()

  result = build(Path(args.source), args.source_sha256)
  output = Path(args.output).expanduser().resolve()
  serialized = json.dumps(
      result, ensure_ascii=False, sort_keys=True, indent=2
  ) + "\n"
  if output.exists() and output.read_text(encoding="utf-8") != serialized:
    raise ValueError(f"Refusing to overwrite different output: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(serialized, encoding="utf-8")
  digest = exact_task_params.file_sha256(output)
  exact_task_params.load_bundle(output, expected_sha256=digest)
  print(f"Output: {output}")
  print(f"SHA-256: {digest}")
  print(f"Tasks: {len(result['overrides'])}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
