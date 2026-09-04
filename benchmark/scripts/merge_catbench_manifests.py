#!/usr/bin/env python3
"""Merge per-model CATBench manifests into one aggregate manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def _load_jobs(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  jobs = payload.get("jobs", [])
  return [job for job in jobs if isinstance(job, dict)]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("manifests", nargs="*", help="Input manifest paths.")
  parser.add_argument("--out", required=True, help="Aggregate manifest path.")
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Mark the aggregate manifest as a dry-run manifest.",
  )
  args = parser.parse_args()

  jobs: list[dict[str, Any]] = []
  sources: list[str] = []
  for raw_path in args.manifests:
    path = Path(raw_path).expanduser().resolve()
    loaded = _load_jobs(path)
    if not loaded and not path.exists():
      continue
    jobs.extend(loaded)
    sources.append(str(path))

  out_path = Path(args.out).expanduser().resolve()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
      "created_at": dt.datetime.now().isoformat(),
      "dry_run": args.dry_run,
      "merged_from": sources,
      "jobs": jobs,
  }
  with out_path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)

  print(f"Merged {len(jobs)} jobs from {len(sources)} manifest(s) into {out_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
