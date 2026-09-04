#!/usr/bin/env python3
"""Run an exact set of CATBench model/category/app/task diagnostic cells.

This narrow diagnostic runner reuses ``run_catbench_5cat_matrix.py`` for model
config expansion, manifest writing, preflight, and scheduling, but accepts
exact task cells from a JSON file or repeated CLI flags.  It is not a consumer
of the frozen episode schedule, so its outputs remain analysis-ineligible
replacement candidates unless a separate audited merge selects them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_ROOT.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import run_catbench_5cat_matrix as matrix  # pylint: disable=wrong-import-position
import exact_task_params as exact_task_params_lib  # pylint: disable=wrong-import-position


Target = tuple[str, str, str, str]


def _load_targets(path: Path) -> list[Target]:
  with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  if isinstance(payload, dict):
    raw_targets = payload.get("targets", [])
  else:
    raw_targets = payload
  targets: list[Target] = []
  for index, item in enumerate(raw_targets):
    if not isinstance(item, dict):
      raise ValueError(f"Target #{index + 1} is not an object: {item!r}")
    try:
      targets.append(
          (
              str(item["model"]),
              str(item["category"]),
              str(item["app_id"]),
              str(item["task"]),
          )
      )
    except KeyError as exc:
      raise ValueError(f"Target #{index + 1} is missing {exc}") from exc
  return targets


def _parse_target_flag(raw: str) -> Target:
  parts = [part.strip() for part in raw.split("|")]
  if len(parts) != 4 or any(not part for part in parts):
    raise ValueError(
        "--target must be formatted as 'MODEL|CATEGORY|APP_ID|TASK'"
    )
  return (parts[0], parts[1], parts[2], parts[3])


def _arg_value(command: list[str], prefix: str) -> str:
  for arg in command:
    if arg.startswith(prefix):
      return arg[len(prefix):]
  return ""


def _compile_exact_task_regex(tasks: list[str]) -> re.Pattern[str]:
  return re.compile(r"^(?:" + "|".join(re.escape(task) for task in tasks) + r")$")


def _attach_exact_task_params(
    jobs: list[matrix.Job],
    *,
    bundle: exact_task_params_lib.ExactTaskParamsBundle,
    output_root: Path,
    run_id: str,
) -> list[dict[str, str]]:
  """Writes exact per-job projections and binds their hashes to each job."""
  projection_dir = output_root / run_id / "exact_task_params"
  projection_dir.mkdir(parents=True, exist_ok=True)
  artifacts: list[dict[str, str]] = []
  for job in jobs:
    identity = "__".join((job.model_name, job.category, job.app_id))
    safe_identity = re.sub(r"[^A-Za-z0-9_.-]+", "_", identity)
    task_set_hash = hashlib.sha256(
        json.dumps(
            list(job.task_templates),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    path = projection_dir / f"{safe_identity}__{task_set_hash}.json"
    payload = exact_task_params_lib.projected_payload(
        bundle, job.task_templates
    )
    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
      raise ValueError(
          f"Refusing to replace a different exact-parameter projection: {path}"
      )
    path.write_text(serialized, encoding="utf-8")
    projection_hash = exact_task_params_lib.file_sha256(path)
    projected = exact_task_params_lib.load_bundle(
        path,
        expected_sha256=projection_hash,
        expected_mode=bundle.mode,
    )
    exact_task_params_lib.require_exact_task_names(
        projected, job.task_templates
    )
    job.exact_task_params_override_file = str(path)
    job.exact_task_params_override_sha256 = projection_hash
    job.exact_task_params_override_mode = bundle.mode
    job.exact_task_params_source_file = str(bundle.source_path)
    job.exact_task_params_source_sha256 = bundle.source_sha256
    job.exact_goal_override_enabled = True
    job.exact_goal_mapping_sha256 = projection_hash
    job.runner_config_sha256 = hashlib.sha256(
        json.dumps(
            {
                "base_runner_config_sha256": job.runner_config_sha256,
                "exact_task_params_override_sha256": projection_hash,
                "exact_task_params_override_mode": bundle.mode,
                "exact_task_params_source_sha256": bundle.source_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    artifacts.append({
        "model": job.model_name,
        "category": job.category,
        "app_id": job.app_id,
        "file": str(path),
        "sha256": projection_hash,
    })
  return artifacts


def _build_jobs(
    *,
    targets: list[Target],
    models: list[dict[str, Any]],
    app_pins: dict[tuple[str, str], dict[str, str]],
    output_root: Path,
    run_id: str,
    python_bin: str,
    n_task_combinations: int,
    task_random_seed: int,
    runner_args: list[str],
    resume_existing: bool,
    condition: str = "",
    instance_id: int | None = None,
    code_revision: str = "",
    source_snapshot_sha256: str = "",
    release_purpose: str = "",
    artifact_role: str = "",
    analysis_eligible: bool = False,
    model_config_sha256: str = "",
    app_pins_sha256: str = "",
) -> list[matrix.Job]:
  model_by_name = {str(model.get("name")): model for model in models}
  grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
  for model, category, app_id, task in targets:
    if task not in grouped[(model, category, app_id)]:
      grouped[(model, category, app_id)].append(task)

  jobs: list[matrix.Job] = []
  missing_models = sorted({model for model, _, _, _ in targets} - set(model_by_name))
  if missing_models:
    raise ValueError(f"Unknown model(s): {', '.join(missing_models)}")

  for (model_name, category, app_id), tasks in sorted(grouped.items()):
    task_regex = _compile_exact_task_regex(tasks)
    built = matrix._task_jobs_for_model(  # pylint: disable=protected-access
        model=model_by_name[model_name],
        categories=(category,),
        output_root=output_root,
        run_id=run_id,
        python_bin=python_bin,
        n_task_combinations=n_task_combinations,
        task_random_seed=task_random_seed,
        extra_runner_args=runner_args,
        resume_existing=resume_existing,
        task_regex=task_regex,
        app_ids={app_id},
        app_pins=app_pins,
        code_revision=code_revision,
        source_snapshot_sha256=source_snapshot_sha256,
        release_id=run_id,
        release_purpose=release_purpose,
        artifact_role=artifact_role,
        analysis_eligible=analysis_eligible,
        condition=condition,
        instance_id=instance_id,
        model_config_sha256=model_config_sha256,
        app_pins_sha256=app_pins_sha256,
    )
    if len(built) != 1:
      raise ValueError(
          "Expected exactly one job for "
          f"{model_name}/{category}/{app_id}, built {len(built)}."
      )
    actual_tasks = set(_arg_value(built[0].command, "--tasks=").split(","))
    wanted_tasks = set(tasks)
    if actual_tasks != wanted_tasks:
      missing = sorted(wanted_tasks - actual_tasks)
      extra = sorted(actual_tasks - wanted_tasks)
      raise ValueError(
          f"Task mismatch for {model_name}/{category}/{app_id}: "
          f"missing={missing}, extra={extra}"
      )
    jobs.append(built[0])
  return jobs


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model_config", default=str(matrix.CONFIG_PATH))
  parser.add_argument("--env_file", default=str(matrix.ENV_FILE_PATH))
  parser.add_argument(
      "--app_pins",
      default=str(matrix.APP_PINS_PATH),
      help="Pinned app-version manifest used by the matrix job builder.",
  )
  parser.add_argument("--targets_json", action="append", default=[])
  parser.add_argument(
      "--exact_task_params_file",
      default="",
      help=(
          "Audited closed-schema mapping from task class names to exact "
          "historical instance-0 params, expected goals, and expected seeds."
      ),
  )
  parser.add_argument(
      "--exact_task_params_sha256",
      default="",
      help=(
          "Required expected SHA-256 of --exact_task_params_file. The runner "
          "and each child process verify it before use."
      ),
  )
  parser.add_argument(
      "--exclude_exact_task",
      action="append",
      default=[],
      help=(
          "Explicitly block an incompatible task class after validating the "
          "full source roster/override one-to-one mapping. Exact overrides "
          "only; can be repeated."
      ),
  )
  parser.add_argument(
      "--target",
      action="append",
      default=[],
      help="Exact cell formatted as MODEL|CATEGORY|APP_ID|TASK.",
  )
  parser.add_argument(
      "--exclude_model",
      action="append",
      default=[],
      help="Drop targets for this model name. Can be repeated.",
  )
  parser.add_argument(
      "--include_model",
      action="append",
      default=[],
      help="Keep only targets for this model name. Can be repeated.",
  )
  parser.add_argument("--emulators", required=True)
  parser.add_argument("--output_root", default=str(matrix._default_output_root()))  # pylint: disable=protected-access
  parser.add_argument("--run_id", required=True)
  parser.add_argument("--python", default=sys.executable)
  parser.add_argument(
      "--adb",
      default=os.environ.get("ADB", "adb"),
      help="ADB executable used for installed package/version preflight.",
  )
  parser.add_argument(
      "--condition",
      choices=("development", "c1", "c2_g", "c2_o"),
      default=os.environ.get("CATBENCH_CONDITION") or "development",
      help="Explicit diagnostic condition persisted in every job and manifest.",
  )
  parser.add_argument("--n_task_combinations", type=int, default=1)
  parser.add_argument("--task_random_seed", type=int, default=30)
  parser.add_argument(
      "--instance_id",
      type=int,
      default=None,
      help=(
          "Retain one zero-based instance from the original K-instance "
          "schedule. If omitted, CATBENCH_INSTANCE_ID is honored."
      ),
  )
  parser.add_argument(
      "--source_snapshot_sha256",
      default="",
      help=(
          "Optional SHA-256 identity for an archived dirty source snapshot; "
          "CATBENCH_SOURCE_SNAPSHOT_SHA256 is honored if omitted."
      ),
  )
  parser.add_argument(
      "--release_purpose",
      default="revision_rerun_candidate",
      help="Purpose label persisted in the manifest, jobs, and episodes.",
  )
  parser.add_argument(
      "--artifact_role",
      default="invalid_episode_replacement_candidate",
      help="Artifact-role label persisted in the manifest, jobs, and episodes.",
  )
  parser.add_argument(
      "--analysis_eligible",
      type=matrix._parse_bool,  # pylint: disable=protected-access
      default=False,
      metavar="true|false",
      help=(
          "Target-cell candidates must remain false until an audited merge "
          "selects them."
      ),
  )
  parser.add_argument("--max_parallel", type=int, default=0)
  parser.add_argument("--launch_stagger_seconds", type=float, default=0.0)
  parser.add_argument("--job_timeout_seconds", type=float, default=0.0)
  parser.add_argument("--prelaunch_delay_seconds", type=float, default=0.0)
  parser.add_argument("--skip_aw_env_preflight", action="store_true")
  parser.add_argument("--dry_run", action="store_true")
  parser.add_argument("--continue_on_error", action="store_true")
  parser.add_argument("--resume_existing", action="store_true")
  parser.add_argument("--runner_arg", action="append", default=[])
  args = parser.parse_args()

  env_file = Path(args.env_file).expanduser()
  loaded_env = matrix._load_env_file(env_file)  # pylint: disable=protected-access
  if loaded_env:
    print(f"Loaded {loaded_env} variables from {env_file}", flush=True)
  leaked_exact_environment = [
      name for name in exact_task_params_lib.ENV_NAMES
      if os.environ.get(name, "").strip()
  ]
  if leaked_exact_environment:
    raise ValueError(
        "Exact task-parameter overrides must be supplied explicitly on the "
        "target-runner command line, not inherited through the environment: "
        + ", ".join(leaked_exact_environment)
    )
  if bool(args.exact_task_params_file) != bool(args.exact_task_params_sha256):
    raise ValueError(
        "--exact_task_params_file and --exact_task_params_sha256 are required "
        "together."
    )
  if args.exclude_exact_task and not args.exact_task_params_file:
    raise ValueError(
        "--exclude_exact_task is available only with a byte-pinned exact "
        "task-parameter override."
    )
  if args.n_task_combinations < 1:
    raise ValueError("n_task_combinations must be positive.")
  selected_instance_id = matrix._resolve_instance_id(  # pylint: disable=protected-access
      args.instance_id, args.n_task_combinations
  )
  source_snapshot_sha256 = (
      matrix._resolve_source_snapshot_sha256(  # pylint: disable=protected-access
          args.source_snapshot_sha256
      )
  )
  args.release_purpose = str(args.release_purpose).strip()
  args.artifact_role = str(args.artifact_role).strip()
  if not args.release_purpose or not args.artifact_role:
    raise ValueError("release_purpose and artifact_role must be non-empty.")
  if args.analysis_eligible:
    raise ValueError(
        "Target-cell outputs are replacement candidates and must remain "
        "analysis_eligible=false until an audited merge selects them."
    )

  exact_bundle = None
  if args.exact_task_params_file:
    if args.n_task_combinations != 1:
      raise ValueError(
          "Exact historical task-parameter overrides require "
          "--n_task_combinations=1; they cannot claim a K>1 schedule."
      )
    if selected_instance_id != 0:
      raise ValueError(
          "Exact historical task-parameter overrides require --instance_id=0."
      )
    if args.condition != "c1":
      raise ValueError(
          "Exact historical task-parameter overrides are restricted to C1."
      )
    if (
        args.release_purpose != "revision_rerun_candidate"
        or args.artifact_role != "invalid_episode_replacement_candidate"
    ):
      raise ValueError(
          "Exact historical task-parameter overrides require the fixed "
          "revision-rerun replacement-candidate provenance labels."
      )
    if args.resume_existing:
      raise ValueError(
          "Exact historical task-parameter replacements cannot resume an "
          "existing checkpoint."
      )
    protected_runner_prefixes = (
        "--tasks",
        "--n_task_combinations",
        "--task_random_seed",
        "--suite_family",
        "--output_path",
        "--checkpoint_dir",
    )
    conflicts = [
        value for value in args.runner_arg
        if value.startswith(protected_runner_prefixes)
    ]
    if conflicts:
      raise ValueError(
          "Exact task-parameter replacements forbid runner arguments that "
          "can change suite identity: " + ", ".join(conflicts)
      )
    exact_bundle = exact_task_params_lib.load_bundle(
        Path(args.exact_task_params_file),
        expected_sha256=args.exact_task_params_sha256,
    )

  targets: list[Target] = []
  for raw_path in args.targets_json:
    targets.extend(_load_targets(Path(raw_path).expanduser().resolve()))
  targets.extend(_parse_target_flag(raw) for raw in args.target)
  if not targets:
    raise ValueError("No targets provided.")
  excluded_exact_targets: list[Target] = []
  if exact_bundle is not None:
    full_exact_task_names = sorted({task for _, _, _, task in targets})
    exact_task_params_lib.require_exact_task_names(
        exact_bundle,
        full_exact_task_names,
    )
    excluded_task_names = set(args.exclude_exact_task)
    unknown_exclusions = sorted(excluded_task_names - set(full_exact_task_names))
    if unknown_exclusions:
      raise ValueError(
          "Exact task exclusions are absent from the fully validated source "
          "roster: " + ", ".join(unknown_exclusions)
      )
    if excluded_task_names:
      excluded_exact_targets = [
          target for target in targets if target[3] in excluded_task_names
      ]
      targets = [
          target for target in targets if target[3] not in excluded_task_names
      ]
      if not targets:
        raise ValueError("Exact task exclusions removed every target.")
    exact_task_names = sorted({task for _, _, _, task in targets})
    # Validate against the same registry/classes used by every model runner,
    # before writing a launchable manifest or touching an emulator.
    from android_world import registry as task_registry_lib  # pylint: disable=import-outside-toplevel
    from android_world import suite_utils  # pylint: disable=import-outside-toplevel

    task_registry = task_registry_lib.TaskRegistry().get_registry(
        family="android_world"
    )
    projected_overrides = {
        name: exact_bundle.overrides[name] for name in exact_task_names
    }
    suite_utils.validate_exact_task_params_entries(
        task_registry, projected_overrides, exact_task_names
    )
  if args.exclude_model:
    excluded = set(args.exclude_model)
    targets = [
        target for target in targets
        if target[0] not in excluded
    ]
  if args.include_model:
    included = set(args.include_model)
    targets = [
        target for target in targets
        if target[0] in included
    ]
  if not targets:
    raise ValueError("Model filters removed every target.")
  if args.condition == "c1":
    leaked = [
        name
        for name in (
            "CATBENCH_TASK_BREAKDOWN_FILE",
            "CATBENCH_TASK_BREAKDOWN_MODE",
            "CATBENCH_TASK_BREAKDOWN_REQUIRED",
        )
        if os.environ.get(name)
    ]
    if leaked:
      raise ValueError(
          "C1 diagnostic inherited C2 breakdown state: " + ", ".join(leaked)
      )

  selected_models = {model for model, _, _, _ in targets}
  models = matrix._load_models(  # pylint: disable=protected-access
      Path(args.model_config).expanduser().resolve(), selected_models
  )
  app_pins = matrix._load_app_pins(  # pylint: disable=protected-access
      Path(args.app_pins).expanduser().resolve()
  )
  model_config_path = Path(args.model_config).expanduser().resolve()
  app_pins_path = Path(args.app_pins).expanduser().resolve()
  model_config_sha256 = hashlib.sha256(
      model_config_path.read_bytes()
  ).hexdigest()
  app_pins_sha256 = hashlib.sha256(app_pins_path.read_bytes()).hexdigest()
  code_revision, dirty_source = matrix._source_revision()  # pylint: disable=protected-access
  output_root = Path(args.output_root).expanduser().resolve()
  emulators = matrix._parse_emulators(args.emulators)  # pylint: disable=protected-access
  max_parallel = args.max_parallel or len(emulators)

  jobs = _build_jobs(
      targets=targets,
      models=models,
      app_pins=app_pins,
      output_root=output_root,
      run_id=args.run_id,
      python_bin=args.python,
      n_task_combinations=args.n_task_combinations,
      task_random_seed=args.task_random_seed,
      runner_args=args.runner_arg,
      resume_existing=args.resume_existing,
      condition=args.condition,
      instance_id=selected_instance_id,
      code_revision=code_revision,
      source_snapshot_sha256=source_snapshot_sha256,
      release_purpose=args.release_purpose,
      artifact_role=args.artifact_role,
      analysis_eligible=False,
      model_config_sha256=model_config_sha256,
      app_pins_sha256=app_pins_sha256,
  )
  exact_task_params_artifacts: list[dict[str, str]] = []
  if exact_bundle is not None:
    exact_task_params_artifacts = _attach_exact_task_params(
        jobs,
        bundle=exact_bundle,
        output_root=output_root,
        run_id=args.run_id,
    )
  manifest_path = output_root / args.run_id / "catbench_5cat_manifest.json"
  manifest_provenance: dict[str, Any] = {
      "release_id": args.run_id,
      "release_purpose": args.release_purpose,
      "artifact_role": args.artifact_role,
      "analysis_eligible": False,
      "condition": args.condition,
      "git_revision": code_revision,
      "dirty_source": dirty_source,
      "source_snapshot_sha256": source_snapshot_sha256,
      "model_config_sha256": model_config_sha256,
      "app_pins_sha256": app_pins_sha256,
      "exact_task_params_override": {
          "enabled": exact_bundle is not None,
          "mode": exact_bundle.mode if exact_bundle is not None else "",
          "override_file": (
              str(exact_bundle.path) if exact_bundle is not None else ""
          ),
          "override_sha256": (
              exact_bundle.sha256 if exact_bundle is not None else ""
          ),
          "source_file": (
              str(exact_bundle.source_path) if exact_bundle is not None else ""
          ),
          "source_sha256": (
              exact_bundle.source_sha256 if exact_bundle is not None else ""
          ),
          "effective_job_files": exact_task_params_artifacts,
          "exact_goal_override_enabled": exact_bundle is not None,
          "exact_goal_mapping_source": "overrides.<task>.expected_goal",
          "exact_goal_bundle_sha256": (
              exact_bundle.sha256 if exact_bundle is not None else ""
          ),
          "excluded_task_classes": sorted(set(args.exclude_exact_task)),
          "excluded_target_count": len(excluded_exact_targets),
          "excluded_targets": [
              {
                  "model": model,
                  "category": category,
                  "app_id": app_id,
                  "task": task,
              }
              for model, category, app_id, task in excluded_exact_targets
          ],
      },
      "device_app_preflight": [],
      "matrix_args": {
          "condition": args.condition,
          "n_task_combinations": args.n_task_combinations,
          "task_random_seed": args.task_random_seed,
          "instance_id": selected_instance_id,
          "catbench_instance_id": selected_instance_id,
          "release_purpose": args.release_purpose,
          "artifact_role": args.artifact_role,
          "analysis_eligible": False,
          "source_snapshot_sha256": source_snapshot_sha256,
          "exact_task_params_override_mode": (
              exact_bundle.mode if exact_bundle is not None else ""
          ),
          "exact_task_params_source_file": (
              str(exact_bundle.source_path) if exact_bundle is not None else ""
          ),
          "exact_task_params_source_sha256": (
              exact_bundle.source_sha256 if exact_bundle is not None else ""
          ),
          "excluded_exact_task_classes": sorted(
              set(args.exclude_exact_task)
          ),
      },
  }
  matrix._write_manifest(  # pylint: disable=protected-access
      manifest_path, jobs, args.dry_run, manifest_provenance
  )
  print(f"Manifest: {manifest_path}", flush=True)
  print(
      f"Targets: {len(targets)}  Jobs: {len(jobs)}  "
      f"Models: {len(selected_models)}  Emulators: {len(emulators)}",
      flush=True,
  )

  categories = tuple(sorted({category for _, category, _, _ in targets}))
  if not args.dry_run:
    app_preflight_code, app_preflight = matrix._preflight_pinned_apps(  # pylint: disable=protected-access
        emulators, jobs, adb_path=args.adb
    )
    manifest_provenance["device_app_preflight"] = app_preflight
    matrix._write_manifest(  # pylint: disable=protected-access
        manifest_path, jobs, args.dry_run, manifest_provenance
    )
    if app_preflight_code != 0:
      print(
          "Installed package/version preflight failed; see manifest.",
          file=sys.stderr,
          flush=True,
      )
      return app_preflight_code
  if not args.dry_run and not args.skip_aw_env_preflight:
    preflight_code = matrix._preflight_aw_env(  # pylint: disable=protected-access
        emulators,
        categories=categories,
        python_bin=args.python,
    )
    if preflight_code != 0:
      return preflight_code
  if not args.dry_run and args.prelaunch_delay_seconds > 0:
    print(
        f"Waiting {args.prelaunch_delay_seconds:.1f}s before launching jobs",
        flush=True,
    )
    time.sleep(args.prelaunch_delay_seconds)

  code = matrix._run_jobs(  # pylint: disable=protected-access
      jobs,
      emulators=emulators,
      max_parallel=max_parallel,
      dry_run=args.dry_run,
      continue_on_error=args.continue_on_error,
      launch_stagger_seconds=args.launch_stagger_seconds,
      job_timeout_seconds=args.job_timeout_seconds,
  )
  matrix._write_manifest(  # pylint: disable=protected-access
      manifest_path, jobs, args.dry_run, manifest_provenance
  )
  return code


if __name__ == "__main__":
  raise SystemExit(main())
