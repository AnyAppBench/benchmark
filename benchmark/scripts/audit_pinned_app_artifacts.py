#!/usr/bin/env python3
"""Verify frozen CATBench app hashes against the local APK/XAPK inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COHORT = BENCHMARK_ROOT / "configs" / "catbench_5cat_primary_cohort.json"
DEFAULT_PINS = BENCHMARK_ROOT / "configs" / "app_versions_pinned.csv"
DEFAULT_APPS = BENCHMARK_ROOT / "app_generalization_apps.csv"
DEFAULT_ARTIFACT_ROOT = Path(
    "$HOME/anyappbench_apks"
)


def _sha256_stream(handle: io.BufferedIOBase) -> str:
  digest = hashlib.sha256()
  for chunk in iter(lambda: handle.read(1024 * 1024), b""):
    digest.update(chunk)
  return digest.hexdigest()


def _sha256_path(path: Path) -> str:
  with path.open("rb") as handle:
    return _sha256_stream(handle)


def _csv_index(path: Path, key: str) -> dict[str, dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    return {str(row[key]): dict(row) for row in csv.DictReader(handle)}


def audit(
    cohort_path: Path,
    pins_path: Path,
    apps_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
  cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
  pins = _csv_index(pins_path, "app_id")
  apps = _csv_index(apps_path, "app_id")
  frozen_app_ids = [
      app_id
      for spec in cohort["categories"].values()
      for app_id in spec["app_ids"]
  ]
  rows: list[dict[str, Any]] = []
  for app_id in frozen_app_ids:
    pin = pins.get(app_id, {})
    inventory = apps.get(app_id, {})
    artifact_path = artifact_root / str(inventory.get("apk_filename") or "")
    expected_hash = str(pin.get("apk_sha256") or "")
    matches: list[dict[str, str]] = []
    artifact_hash = ""
    error = ""
    if not artifact_path.is_file():
      error = "artifact_missing"
    else:
      with artifact_path.open("rb") as handle:
        artifact_hash = _sha256_stream(handle)
      if artifact_hash == expected_hash:
        matches.append({"scope": "artifact_file", "member": ""})
      if zipfile.is_zipfile(artifact_path):
        with zipfile.ZipFile(artifact_path) as archive:
          for member in archive.namelist():
            if not member.lower().endswith(".apk"):
              continue
            with archive.open(member) as handle:
              member_hash = _sha256_stream(handle)
            if member_hash == expected_hash:
              matches.append({"scope": "zip_apk_member", "member": member})
    if not error and not matches:
      error = "pinned_hash_not_found_in_local_artifact_or_apk_members"
    rows.append({
        "app_id": app_id,
        "category": str(pin.get("category") or ""),
        "package_name": str(pin.get("package_name") or ""),
        "pinned_version_name": str(pin.get("version_name") or ""),
        "pinned_version_code": str(pin.get("version_code") or ""),
        "pinned_sha256": expected_hash,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "matches": matches,
        "valid": bool(matches) and not error,
        "error": error,
    })
  valid = sum(row["valid"] for row in rows)
  return {
      "audit_type": "local_frozen_app_artifact_hash_preflight",
      "cohort_release_id": cohort["release_id"],
      "cohort_manifest": str(cohort_path),
      "cohort_manifest_sha256": _sha256_path(cohort_path),
      "pins_file": str(pins_path),
      "pins_file_sha256": _sha256_path(pins_path),
      "app_inventory_file": str(apps_path),
      "app_inventory_file_sha256": _sha256_path(apps_path),
      "artifact_root": str(artifact_root),
      "expected_apps": len(frozen_app_ids),
      "valid_apps": valid,
      "invalid_apps": len(rows) - valid,
      "valid": valid == len(rows),
      "apps": rows,
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cohort_manifest", default=str(DEFAULT_COHORT))
  parser.add_argument("--pins", default=str(DEFAULT_PINS))
  parser.add_argument("--apps", default=str(DEFAULT_APPS))
  parser.add_argument("--artifact_root", default=str(DEFAULT_ARTIFACT_ROOT))
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  report = audit(
      Path(args.cohort_manifest).expanduser().resolve(),
      Path(args.pins).expanduser().resolve(),
      Path(args.apps).expanduser().resolve(),
      Path(args.artifact_root).expanduser().resolve(),
  )
  output = Path(args.output).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(
      json.dumps(report, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  print(json.dumps({
      "expected_apps": report["expected_apps"],
      "valid_apps": report["valid_apps"],
      "invalid_apps": report["invalid_apps"],
      "invalid_app_ids": [
          row["app_id"] for row in report["apps"] if not row["valid"]
      ],
  }, indent=2))
  return 0 if report["valid"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
