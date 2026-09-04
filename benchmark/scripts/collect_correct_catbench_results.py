#!/usr/bin/env python3
"""Collect CATBench episodes from multiple runs into one corrected manifest.

The CATBench matrix runner writes one job per (model, category, app). Targeted
repair runs may contain only one task inside that job, so manifest-level merging
is too coarse: it either keeps stale episodes or drops unrelated good episodes.

This script merges at the pkl.gz episode-file level. Input manifests are applied
in order; later manifests override earlier ones when they contain the same
(model, category, app_id, episode file name). The output is a normal
catbench_5cat_manifest.json whose output paths point at a collected tree of
symlinks or copied pkl.gz files, so existing report scripts can read it.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import shutil
from pathlib import Path
from typing import Any


EpisodeKey = tuple[str, str, str, str]
JobKey = tuple[str, str, str]


def _slug(value: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
  with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  return [job for job in payload.get("jobs", []) if isinstance(job, dict)]


def _episode_files(output_path: Path) -> list[Path]:
  if not output_path.exists():
    return []
  return sorted(output_path.rglob("*.pkl.gz"))


def _passes_model_filter(
    model: str,
    include_models: set[str],
    exclude_patterns: list[re.Pattern[str]],
) -> bool:
  if include_models and model not in include_models:
    return False
  return not any(pattern.search(model) for pattern in exclude_patterns)


def _copy_or_link(source: Path, dest: Path, mode: str) -> None:
  dest.parent.mkdir(parents=True, exist_ok=True)
  if dest.exists() or dest.is_symlink():
    dest.unlink()
  if mode == "copy":
    shutil.copy2(source, dest)
  else:
    dest.symlink_to(source)


def _safe_json(value: Any) -> Any:
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  if isinstance(value, dict):
    return {str(key): _safe_json(val) for key, val in value.items()}
  if isinstance(value, (list, tuple)):
    return [_safe_json(val) for val in value]
  return str(value)


def _write_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as handle:
    json.dump(_safe_json(payload), handle, indent=2, ensure_ascii=False)
    handle.write("\n")


def _write_audit_md(
    path: Path,
    *,
    manifest_paths: list[Path],
    selected: dict[EpisodeKey, dict[str, Any]],
    overridden: list[dict[str, Any]],
    job_counts: dict[JobKey, int],
) -> None:
  source_counts = collections.Counter(
      row["source_manifest"] for row in selected.values()
  )
  model_counts = collections.Counter(key[0] for key in selected)
  category_counts = collections.Counter(key[1] for key in selected)

  lines = [
      "# Corrected CATBench Collection",
      "",
      f"Created: {dt.datetime.now().isoformat()}",
      "",
      "## Source Priority",
      "",
  ]
  for index, manifest in enumerate(manifest_paths, start=1):
    lines.append(f"{index}. `{manifest}`")
  lines.extend(["", "Later manifests override earlier manifests at episode-file level.", ""])

  lines.extend(["## Episode Counts", ""])
  lines.append(f"- Selected episodes: {len(selected)}")
  lines.append(f"- Overridden stale episodes: {len(overridden)}")
  lines.append(f"- Output jobs: {len(job_counts)}")
  lines.append("")

  lines.extend(["## By Source", ""])
  for source, count in sorted(source_counts.items()):
    lines.append(f"- `{source}`: {count}")
  lines.append("")

  lines.extend(["## By Model", ""])
  for model, count in sorted(model_counts.items()):
    lines.append(f"- {model}: {count}")
  lines.append("")

  lines.extend(["## By Category", ""])
  for category, count in sorted(category_counts.items()):
    lines.append(f"- {category}: {count}")
  lines.append("")

  lines.extend(["## By Job", ""])
  for (model, category, app_id), count in sorted(job_counts.items()):
    lines.append(f"- {model} / {category} / {app_id}: {count}")

  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--manifest",
      action="append",
      required=True,
      help=(
          "Input catbench_5cat_manifest.json. Repeat in source priority order; "
          "later manifests override earlier ones."
      ),
  )
  parser.add_argument("--out_dir", required=True, help="Output collection root.")
  parser.add_argument(
      "--run_id",
      default="corrected_collected",
      help="Run id directory name under --out_dir.",
  )
  parser.add_argument(
      "--model",
      action="append",
      default=[],
      help="Optional model allowlist. Repeat for multiple models.",
  )
  parser.add_argument(
      "--exclude_model_regex",
      action="append",
      default=[],
      help="Regex for model names to exclude. Repeat for multiple patterns.",
  )
  parser.add_argument(
      "--mode",
      choices=("symlink", "copy"),
      default="symlink",
      help="Store selected pkl.gz files as symlinks or physical copies.",
  )
  args = parser.parse_args()

  manifest_paths = [Path(raw).expanduser().resolve() for raw in args.manifest]
  include_models = set(args.model)
  exclude_patterns = [re.compile(raw) for raw in args.exclude_model_regex]
  out_dir = Path(args.out_dir).expanduser().resolve()
  run_root = out_dir / args.run_id
  data_root = run_root / "matrix" / args.run_id

  selected: dict[EpisodeKey, dict[str, Any]] = {}
  overridden: list[dict[str, Any]] = []
  source_job_counts: dict[str, int] = collections.Counter()
  source_episode_counts: dict[str, int] = collections.Counter()

  for priority, manifest_path in enumerate(manifest_paths):
    jobs = _load_manifest(manifest_path)
    source_job_counts[str(manifest_path)] += len(jobs)
    for job in jobs:
      model = str(job.get("model_name") or "")
      category = str(job.get("category") or "")
      app_id = str(job.get("app_id") or "")
      if not model or not category or not app_id:
        continue
      if not _passes_model_filter(model, include_models, exclude_patterns):
        continue
      output_path = Path(str(job.get("output_path") or "")).expanduser()
      for episode_path in _episode_files(output_path):
        episode_name = episode_path.name
        key = (model, category, app_id, episode_name)
        row = {
            "key": key,
            "priority": priority,
            "source_manifest": str(manifest_path),
            "source_path": str(episode_path.resolve()),
            "source_job": job,
        }
        if key in selected:
          overridden.append(
              {
                  "key": key,
                  "old_source_manifest": selected[key]["source_manifest"],
                  "old_source_path": selected[key]["source_path"],
                  "new_source_manifest": str(manifest_path),
                  "new_source_path": str(episode_path.resolve()),
              }
          )
        selected[key] = row
        source_episode_counts[str(manifest_path)] += 1

  if not selected:
    raise SystemExit("No episodes selected. Check manifest paths and filters.")

  job_to_episodes: dict[JobKey, list[tuple[EpisodeKey, dict[str, Any]]]] = (
      collections.defaultdict(list)
  )
  for key, row in selected.items():
    job_to_episodes[key[:3]].append((key, row))

  manifest_jobs: list[dict[str, Any]] = []
  for job_key, items in sorted(job_to_episodes.items()):
    model, category, app_id = job_key
    model_slug = _slug(model)
    dest_job_dir = data_root / model_slug / category / app_id
    dest_run_dir = dest_job_dir / "collected"
    app_name = app_id
    source_jobs = []
    task_names: list[str] = []
    for key, row in sorted(items):
      source_path = Path(row["source_path"])
      _copy_or_link(source_path, dest_run_dir / key[3], args.mode)
      source_job = dict(row["source_job"])
      app_name = str(source_job.get("app_name") or app_name)
      task_name = re.sub(r"_\d+$", "", key[3].removesuffix(".pkl.gz"))
      if task_name not in task_names:
        task_names.append(task_name)
      source_jobs.append(
          {
              "source_manifest": row["source_manifest"],
              "source_output_path": source_job.get("output_path"),
              "episode_file": key[3],
          }
      )

    manifest_jobs.append(
        {
            "model_name": model,
            "category": category,
            "app_id": app_id,
            "app_name": app_name,
            "command": [
                "collect_correct_catbench_results.py",
                "--tasks=" + ",".join(sorted(task_names)),
            ],
            "output_path": str(dest_job_dir),
            "exit_code": 0,
            "collected_episode_count": len(items),
            "collected_sources": source_jobs,
        }
    )

  manifest_payload = {
      "created_at": dt.datetime.now().isoformat(),
      "dry_run": False,
      "run_id": args.run_id,
      "collection_mode": args.mode,
      "merged_episode_level": True,
      "source_priority": [str(path) for path in manifest_paths],
      "jobs": manifest_jobs,
  }
  manifest_path = data_root / "catbench_5cat_manifest.json"
  _write_json(manifest_path, manifest_payload)

  audit = {
      "created_at": dt.datetime.now().isoformat(),
      "run_id": args.run_id,
      "out_dir": str(out_dir),
      "manifest": str(manifest_path),
      "mode": args.mode,
      "include_models": sorted(include_models),
      "exclude_model_regex": args.exclude_model_regex,
      "source_priority": [str(path) for path in manifest_paths],
      "source_job_counts": dict(source_job_counts),
      "source_episode_counts_before_override": dict(source_episode_counts),
      "selected_episode_count": len(selected),
      "overridden_episode_count": len(overridden),
      "output_job_count": len(manifest_jobs),
      "overridden": overridden,
      "selected": [
          {
              "model_name": key[0],
              "category": key[1],
              "app_id": key[2],
              "episode_file": key[3],
              "source_manifest": row["source_manifest"],
              "source_path": row["source_path"],
          }
          for key, row in sorted(selected.items())
      ],
  }
  _write_json(run_root / "merge_audit.json", audit)
  _write_audit_md(
      run_root / "merge_audit.md",
      manifest_paths=manifest_paths,
      selected=selected,
      overridden=overridden,
      job_counts={key: len(items) for key, items in job_to_episodes.items()},
  )

  print(f"Wrote corrected manifest: {manifest_path}")
  print(f"Selected episodes: {len(selected)}")
  print(f"Overridden stale episodes: {len(overridden)}")
  print(f"Output jobs: {len(manifest_jobs)}")
  print(f"Audit: {run_root / 'merge_audit.md'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
