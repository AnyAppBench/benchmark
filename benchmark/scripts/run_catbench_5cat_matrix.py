#!/usr/bin/env python3
"""Run CATBench five-category model/app matrix across emulator ports."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import signal
import shlex
import socket
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_ROOT.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

from app_generalization_profiles import get_domain_profiles  # pylint: disable=wrong-import-position
import exact_task_params as exact_task_params_lib  # pylint: disable=wrong-import-position
from scripts import build_catbench_frozen_schedule as frozen_schedule  # pylint: disable=wrong-import-position
import task_breakdowns as task_breakdowns_lib  # pylint: disable=wrong-import-position


DEFAULT_CATEGORIES = ("sms", "files", "maps", "contacts", "clock")
CONFIG_PATH = BENCHMARK_ROOT / "configs" / "catbench_5cat_models.json"
ENV_FILE_PATH = BENCHMARK_ROOT / "configs" / "catbench.env"
APP_PINS_PATH = BENCHMARK_ROOT / "configs" / "app_versions_pinned.csv"
APP_INVENTORY_PATH = BENCHMARK_ROOT / "app_generalization_apps.csv"
DEFAULT_APP_ARTIFACT_ROOT = Path(
    os.environ.get(
        "CATBENCH_APP_ARTIFACT_ROOT",
        "$HOME/anyappbench_apks",
    )
)
PRIMARY_COHORT_PATH = (
    BENCHMARK_ROOT / "configs" / "catbench_5cat_primary_cohort.json"
)
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")
HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclasses.dataclass(frozen=True)
class Emulator:
  console_port: int
  grpc_port: int
  serial_override: str | None = None
  adb_server_port: int | None = None

  @property
  def serial(self) -> str:
    return self.serial_override or f"emulator-{self.console_port}"


@dataclasses.dataclass
class Job:
  model_name: str
  category: str
  app_id: str
  app_name: str
  command: list[str]
  output_path: str
  package_name: str = ""
  app_version: str = ""
  app_version_code: str = ""
  apk_sha256: str = ""
  task_templates: tuple[str, ...] = ()
  n_task_combinations: int = 0
  task_random_seed: int = 0
  instance_id: int | None = None
  code_revision: str = ""
  source_snapshot_sha256: str = ""
  release_id: str = ""
  release_purpose: str = ""
  artifact_role: str = ""
  analysis_eligible: bool | None = None
  condition: str = ""
  runner_config_sha256: str = ""
  model_revision: str = ""
  model_config_sha256: str = ""
  app_pins_sha256: str = ""
  exact_task_params_override_file: str = ""
  exact_task_params_override_sha256: str = ""
  exact_task_params_override_mode: str = ""
  exact_task_params_source_file: str = ""
  exact_task_params_source_sha256: str = ""
  exact_goal_override_enabled: bool = False
  exact_goal_mapping_sha256: str = ""
  checkpoint_dir: str | None = None
  console_port: int | None = None
  grpc_port: int | None = None
  adb_server_port: int | None = None
  exit_code: int | None = None


def _parse_bool(value: str) -> bool:
  normalized = str(value).strip().lower()
  if normalized in {"1", "true", "yes"}:
    return True
  if normalized in {"0", "false", "no"}:
    return False
  raise argparse.ArgumentTypeError(
      f"Expected a boolean (true/false); got {value!r}."
  )


def _resolve_instance_id(
    cli_value: int | None, n_task_combinations: int
) -> int | None:
  """Resolves an explicit/frozen instance selector without silent drift."""
  raw_environment = os.environ.get("CATBENCH_INSTANCE_ID", "").strip()
  environment_value: int | None = None
  if raw_environment:
    try:
      environment_value = int(raw_environment)
    except ValueError as error:
      raise ValueError(
          "CATBENCH_INSTANCE_ID must be a zero-based integer; "
          f"got {raw_environment!r}."
      ) from error
  if (
      cli_value is not None
      and environment_value is not None
      and cli_value != environment_value
  ):
    raise ValueError(
        "--instance_id conflicts with CATBENCH_INSTANCE_ID: "
        f"{cli_value} != {environment_value}."
    )
  selected = cli_value if cli_value is not None else environment_value
  if selected is not None and not 0 <= selected < n_task_combinations:
    raise ValueError(
        f"instance_id={selected} is outside the scheduled range "
        f"[0, {n_task_combinations})."
    )
  return selected


def _resolve_source_snapshot_sha256(cli_value: str) -> str:
  """Resolves and validates the optional dirty-source snapshot identity."""
  explicit = str(cli_value or "").strip()
  inherited = os.environ.get("CATBENCH_SOURCE_SNAPSHOT_SHA256", "").strip()
  if explicit and inherited and explicit.lower() != inherited.lower():
    raise ValueError(
        "--source_snapshot_sha256 conflicts with "
        "CATBENCH_SOURCE_SNAPSHOT_SHA256."
    )
  value = explicit or inherited
  if value and not HEX_SHA256.fullmatch(value):
    raise ValueError(
        "source_snapshot_sha256 must be exactly 64 hexadecimal characters."
    )
  return value.lower()


def _default_output_root() -> Path:
  host = socket.gethostname().split(".", 1)[0].lower()
  user = os.environ.get("USER", "ttran")
  if host in {"odin2", "loki3"}:
    return Path(f"$HOME/{user}/catbench_runs/five_category")
  return Path(f"$HOME/catbench_runs/five_category")


def _expand_env(value: str) -> str:
  def replace(match: re.Match[str]) -> str:
    name = match.group(1)
    default = match.group(3)
    value = os.environ.get(name)
    return value if value else default or ""

  return ENV_PATTERN.sub(replace, value)


def _load_env_file(path: Path, override: bool = False) -> int:
  """Loads simple KEY=VALUE and export KEY=VALUE lines into os.environ."""
  if not path.exists():
    return 0
  loaded = 0
  with path.open("r", encoding="utf-8") as handle:
    for raw_line in handle:
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      try:
        tokens = shlex.split(line, comments=True, posix=True)
      except ValueError:
        continue
      if tokens and tokens[0] == "export":
        tokens = tokens[1:]
      for token in tokens:
        if "=" not in token:
          continue
        name, value = token.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
          continue
        if not override and name in os.environ:
          continue
        value = _expand_env(value)
        os.environ[name] = value
        loaded += 1
  return loaded


def _load_models(path: Path, selected: set[str] | None) -> list[dict[str, Any]]:
  with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  models = payload.get("models", [])
  out = []
  for model in models:
    if selected and model.get("name") not in selected:
      continue
    if not selected and model.get("enabled", True) is False:
      continue
    out.append(model)
  return out


def _load_app_pins(path: Path) -> dict[tuple[str, str], dict[str, str]]:
  if not path.exists():
    raise FileNotFoundError(f"Pinned app-version manifest not found: {path}")
  pins: dict[tuple[str, str], dict[str, str]] = {}
  with path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
      key = (str(row.get("category") or ""), str(row.get("app_id") or ""))
      if not all(key):
        raise ValueError(f"Malformed app pin row: {row}")
      if key in pins:
        raise ValueError(f"Duplicate app pin row: {key}")
      pins[key] = {str(field): str(value or "") for field, value in row.items()}
  return pins


def _load_primary_cohort(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  required = {
      "release_id",
      "status",
      "models",
      "categories",
      "n_task_combinations",
      "task_random_seed",
      "conditions",
      "expected",
  }
  missing = sorted(required - set(payload))
  if missing:
    raise ValueError(f"Primary cohort manifest missing fields: {missing}")
  return payload


def _primary_release_violations(
    cohort: dict[str, Any],
    *,
    categories: tuple[str, ...],
    models: list[dict[str, Any]],
    n_task_combinations: int,
    task_random_seed: int,
) -> list[str]:
  """Returns launch-blocking deviations from the frozen primary cohort."""
  issues: list[str] = []
  try:
    frozen_schedule.validate_primary_cohort(cohort)
  except frozen_schedule.ScheduleBuildError as exc:
    issues.append(f"frozen_primary_cohort_gate={exc}")
  if cohort.get("status") != "ready":
    issues.append(f"cohort_status={cohort.get('status')!r}, expected 'ready'")
  for gate_name, gate in (cohort.get("eligibility_gates") or {}).items():
    if gate.get("required_for_primary") is True and gate.get("status") != "ready":
      issues.append(
          f"eligibility_gate[{gate_name}]={gate.get('status')!r}, "
          "expected 'ready'"
      )
  expected_categories = tuple(cohort["categories"])
  if categories != expected_categories:
    issues.append(
        f"categories={list(categories)!r}, expected {list(expected_categories)!r}"
    )
  actual_models = [str(model.get("name")) for model in models]
  expected_models = list(cohort["models"])
  if actual_models != expected_models:
    issues.append(f"models={actual_models!r}, expected {expected_models!r}")
  for model in models:
    if not str(model.get("revision") or "").strip():
      issues.append(
          f"model {model.get('name')!r} lacks an immutable revision/checkpoint "
          "identifier"
      )
  if n_task_combinations != int(cohort["n_task_combinations"]):
    issues.append(
        f"n_task_combinations={n_task_combinations}, expected "
        f"{cohort['n_task_combinations']}"
    )
  if task_random_seed != int(cohort["task_random_seed"]):
    issues.append(
        f"task_random_seed={task_random_seed}, expected "
        f"{cohort['task_random_seed']}"
    )

  profiles = get_domain_profiles()
  for category, spec in cohort["categories"].items():
    enabled = [
        app for app in profiles[category].apps if app.implemented_tasks
    ]
    actual_app_ids = [app.app_id for app in enabled]
    expected_app_ids = list(spec["app_ids"])
    if actual_app_ids != expected_app_ids:
      issues.append(
          f"{category} enabled apps={actual_app_ids!r}, "
          f"expected {expected_app_ids!r}"
      )
    for app in enabled:
      if len(app.implemented_tasks) != len(spec["semantic_task_ids"]):
        issues.append(
            f"{category}/{app.app_id} has {len(app.implemented_tasks)} tasks, "
            f"expected {len(spec['semantic_task_ids'])}"
        )
  return issues


def _source_revision() -> tuple[str, bool]:
  try:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip())
    return revision, dirty
  except FileNotFoundError:
    git_dir = REPO_ROOT / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
      ref_name = head.removeprefix("ref: ")
      ref_path = git_dir / ref_name
      if ref_path.is_file():
        revision = ref_path.read_text(encoding="utf-8").strip()
      else:
        revision = ""
        packed_refs = git_dir / "packed-refs"
        if packed_refs.is_file():
          for raw_line in packed_refs.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "^")):
              continue
            value, name = line.split(" ", 1)
            if name == ref_name:
              revision = value
              break
        if not revision:
          raise RuntimeError(f"Unable to resolve Git ref {ref_name!r}.")
    else:
      revision = head
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
      raise RuntimeError(f"Invalid Git revision in {git_dir}: {revision!r}")
    # Without Git we cannot prove cleanliness, so retain the dirty-source gate.
    return revision.lower(), True


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _parse_emulators(raw: str) -> list[Emulator]:
  emulators = []
  for item in raw.split(","):
    item = item.strip()
    if not item:
      continue
    parts = item.split(":")
    if len(parts) == 1:
      console_port = int(parts[0])
      emulators.append(Emulator(console_port, 8554 + (console_port - 5554) // 2))
    elif len(parts) == 2:
      console, grpc = parts
      emulators.append(Emulator(int(console), int(grpc)))
    elif len(parts) == 3:
      console, grpc, adb_serial_or_port = parts
      serial = (
          f"localhost:{adb_serial_or_port}"
          if adb_serial_or_port.isdigit()
          else adb_serial_or_port
      )
      emulators.append(Emulator(int(console), int(grpc), serial))
    elif len(parts) == 4:
      console, grpc, adb_serial_or_port, adb_server_port = parts
      serial = (
          None
          if adb_serial_or_port in {"", "-"}
          else (
              f"localhost:{adb_serial_or_port}"
              if adb_serial_or_port.isdigit()
              else adb_serial_or_port
          )
      )
      emulators.append(
          Emulator(
              int(console),
              int(grpc),
              serial,
              int(adb_server_port),
          )
      )
    else:
      raise ValueError(
          "Emulators must be console, console:grpc, or "
          "console:grpc:adb_serial_or_port[:adb_server_port]."
      )
  if not emulators:
    raise ValueError("At least one emulator port must be provided.")
  return emulators


def _resolve_runner(path: str) -> str:
  candidate = Path(_expand_env(path))
  if candidate.is_absolute():
    return str(candidate)
  for base in (REPO_ROOT, BENCHMARK_ROOT, Path.cwd()):
    resolved = (base / candidate).resolve()
    if resolved.exists():
      return str(resolved)
  return str((REPO_ROOT / candidate).resolve())


def _latest_checkpoint_dir(output_path: Path) -> Path | None:
  if not output_path.is_dir():
    return None
  run_dirs = [path for path in output_path.iterdir() if path.is_dir() and path.name.startswith("run_")]
  if not run_dirs:
    return None
  return max(run_dirs, key=lambda path: (path.name, path.stat().st_mtime_ns))


def _task_jobs_for_model(
    model: dict[str, Any],
    categories: Iterable[str],
    output_root: Path,
    run_id: str,
    python_bin: str,
    n_task_combinations: int,
    task_random_seed: int,
    extra_runner_args: list[str],
    resume_existing: bool,
    task_regex: re.Pattern[str] | None = None,
    app_ids: set[str] | None = None,
    app_pins: dict[tuple[str, str], dict[str, str]] | None = None,
    code_revision: str = "",
    source_snapshot_sha256: str = "",
    release_id: str = "",
    release_purpose: str = "",
    artifact_role: str = "",
    analysis_eligible: bool | None = None,
    condition: str = "",
    instance_id: int | None = None,
    model_config_sha256: str = "",
    app_pins_sha256: str = "",
) -> list[Job]:
  profiles = get_domain_profiles()
  runner = _resolve_runner(model["runner_script"])
  model_args = [_expand_env(arg) for arg in model.get("args", [])]
  model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model["name"])
  jobs: list[Job] = []
  for category in categories:
    profile = profiles[category]
    for app in profile.apps:
      if app_ids is not None and app.app_id not in app_ids:
        continue
      implemented_tasks = tuple(app.implemented_tasks)
      if task_regex is not None:
        implemented_tasks = tuple(
            task for task in implemented_tasks if task_regex.search(task)
        )
      if not implemented_tasks:
        continue
      pin = (app_pins or {}).get((category, app.app_id))
      if pin is None:
        raise ValueError(
            f"Missing pinned app version for {category}/{app.app_id}."
        )
      if pin.get("package_name") != str(app.package_name or ""):
        raise ValueError(
            f"Pinned package mismatch for {category}/{app.app_id}: "
            f"profile={app.package_name!r} pin={pin.get('package_name')!r}"
        )
      out_dir = output_root / run_id / model_slug / category / app.app_id
      task_csv = ",".join(implemented_tasks)
      command = [
          python_bin,
          runner,
          "--suite_family=android_world",
          f"--tasks={task_csv}",
          f"--n_task_combinations={n_task_combinations}",
          f"--task_random_seed={task_random_seed}",
          f"--output_path={out_dir}",
          *model_args,
          *extra_runner_args,
      ]
      runner_config_sha256 = hashlib.sha256(
          json.dumps(
              [arg for arg in command if not arg.startswith("--output_path=")],
              ensure_ascii=False,
              separators=(",", ":"),
          ).encode("utf-8")
      ).hexdigest()
      checkpoint_dir = None
      if resume_existing:
        latest = _latest_checkpoint_dir(out_dir)
        if latest is not None:
          checkpoint_dir = str(latest)
          command.append(f"--checkpoint_dir={checkpoint_dir}")
      jobs.append(
          Job(
              model_name=model["name"],
              category=category,
              app_id=app.app_id,
              app_name=app.display_name,
              command=command,
            output_path=str(out_dir),
            package_name=str(app.package_name or ""),
            app_version=pin.get("version_name", ""),
            app_version_code=pin.get("version_code", ""),
            apk_sha256=pin.get("apk_sha256", ""),
            task_templates=implemented_tasks,
            n_task_combinations=n_task_combinations,
            task_random_seed=task_random_seed,
            instance_id=instance_id,
            code_revision=code_revision,
            source_snapshot_sha256=source_snapshot_sha256,
            release_id=release_id,
            release_purpose=release_purpose,
            artifact_role=artifact_role,
            analysis_eligible=analysis_eligible,
            condition=condition,
            runner_config_sha256=runner_config_sha256,
            model_revision=str(model.get("revision") or ""),
            model_config_sha256=model_config_sha256,
            app_pins_sha256=app_pins_sha256,
            checkpoint_dir=checkpoint_dir,
          )
      )
  return jobs


def _write_manifest(
    path: Path,
    jobs: list[Job],
    dry_run: bool,
    provenance: dict[str, Any] | None = None,
) -> None:
  payload = {
      "created_at": dt.datetime.now().isoformat(),
      "dry_run": dry_run,
      **(provenance or {}),
      "jobs": [dataclasses.asdict(job) for job in jobs],
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)


def _run_jobs(
    jobs: list[Job],
    emulators: list[Emulator],
    max_parallel: int,
    dry_run: bool,
    continue_on_error: bool,
    launch_stagger_seconds: float = 0.0,
    job_timeout_seconds: float = 0.0,
    plan_file_sha256: str | None = None,
    cohort_sha256: str = "",
) -> int:
  if dry_run:
    for idx, job in enumerate(jobs):
      emu = emulators[idx % len(emulators)]
      job.console_port = emu.console_port
      job.grpc_port = emu.grpc_port
      job.adb_server_port = emu.adb_server_port
      rendered = [
          *job.command,
          f"--console_port={emu.console_port}",
          f"--grpc_port={emu.grpc_port}",
      ]
      print("[DRY]", job.model_name, job.category, job.app_name)
      print(" ".join(rendered))
    return 0

  # Run-level provenance stamped into every episode by suite_utils. The plan
  # digest is derived from the same CATBENCH_TASK_BREAKDOWN_FILE the runner
  # reads; the cohort digest is only exported when the caller knows it.
  breakdown_file = os.environ.get(
      task_breakdowns_lib.ENV_BREAKDOWN_FILE, ""
  ).strip()
  if breakdown_file and not plan_file_sha256:
    plan_file_sha256 = _file_sha256(
        Path(breakdown_file).expanduser().resolve()
    )
  run_provenance_fields = {
      "CATBENCH_PLAN_FILE_SHA256": (
          plan_file_sha256 if breakdown_file else ""
      ),
  }
  if cohort_sha256:
    run_provenance_fields["CATBENCH_COHORT_SHA256"] = cohort_sha256

  pending = list(jobs)
  running: list[tuple[subprocess.Popen[Any], Job, Emulator, float]] = []
  timed_out_pids: set[int] = set()
  failures = 0
  max_parallel = min(max_parallel, len(emulators))
  available_emulators = list(emulators[:max_parallel])

  def terminate_proc(proc: subprocess.Popen[Any]) -> bool:
    if proc.poll() is not None:
      return True
    try:
      os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
      return True
    except OSError:
      proc.terminate()
    try:
      proc.wait(timeout=30)
      return True
    except subprocess.TimeoutExpired:
      pass
    try:
      os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
      return True
    except OSError:
      proc.kill()
    try:
      proc.wait(timeout=30)
      return True
    except subprocess.TimeoutExpired:
      return False

  while pending or running:
    while pending and available_emulators:
      emu = available_emulators.pop(0)
      job = pending.pop(0)
      job.console_port = emu.console_port
      job.grpc_port = emu.grpc_port
      job.adb_server_port = emu.adb_server_port
      command = [
          *job.command,
          f"--console_port={emu.console_port}",
          f"--grpc_port={emu.grpc_port}",
      ]
      env = os.environ.copy()
      env["ANDROID_SERIAL"] = emu.serial
      if emu.adb_server_port is not None:
        env["ADB_SERVER_PORT"] = str(emu.adb_server_port)
        env["ANDROID_ADB_SERVER_PORT"] = str(emu.adb_server_port)
      exact_episode_fields = {
          "CATBENCH_CODE_REVISION": job.code_revision,
          "CATBENCH_SOURCE_SNAPSHOT_SHA256": job.source_snapshot_sha256,
          "CATBENCH_RELEASE_ID": job.release_id,
          "CATBENCH_RELEASE_PURPOSE": job.release_purpose,
          "CATBENCH_ARTIFACT_ROLE": job.artifact_role,
          "CATBENCH_CONDITION": job.condition,
          "CATBENCH_APP_ID": job.app_id,
          "CATBENCH_APP_VERSION": job.app_version,
          "CATBENCH_APP_VERSION_CODE": job.app_version_code,
          "CATBENCH_APK_SHA256": job.apk_sha256,
          "CATBENCH_MODEL_NAME": job.model_name,
          "CATBENCH_MODEL_REVISION": job.model_revision,
          "CATBENCH_RUNNER_CONFIG_SHA256": job.runner_config_sha256,
          "CATBENCH_MODEL_CONFIG_SHA256": job.model_config_sha256,
          "CATBENCH_APP_PINS_SHA256": job.app_pins_sha256,
          exact_task_params_lib.ENV_FILE: (
              job.exact_task_params_override_file
          ),
          exact_task_params_lib.ENV_SHA256: (
              job.exact_task_params_override_sha256
          ),
          exact_task_params_lib.ENV_MODE: (
              job.exact_task_params_override_mode
          ),
          exact_task_params_lib.ENV_SOURCE_FILE: (
              job.exact_task_params_source_file
          ),
          exact_task_params_lib.ENV_SOURCE_SHA256: (
              job.exact_task_params_source_sha256
          ),
          exact_task_params_lib.ENV_GOAL_OVERRIDE_ENABLED: (
              "1" if job.exact_goal_override_enabled else ""
          ),
          exact_task_params_lib.ENV_GOAL_MAPPING_SHA256: (
              job.exact_goal_mapping_sha256
          ),
      }
      for name, value in exact_episode_fields.items():
        if value:
          env[name] = value
        else:
          env.pop(name, None)
      for name, value in run_provenance_fields.items():
        if value:
          env[name] = value
        else:
          env.pop(name, None)
      if job.analysis_eligible is not None:
        env["CATBENCH_ANALYSIS_ELIGIBLE"] = (
            "1" if job.analysis_eligible else "0"
        )
      else:
        env.pop("CATBENCH_ANALYSIS_ELIGIBLE", None)
      if job.instance_id is not None:
        env["CATBENCH_INSTANCE_ID"] = str(job.instance_id)
      else:
        env.pop("CATBENCH_INSTANCE_ID", None)
      if job.n_task_combinations > 0:
        env["CATBENCH_TASK_RANDOM_SEED"] = str(job.task_random_seed)
        env["CATBENCH_N_TASK_COMBINATIONS"] = str(
            job.n_task_combinations
        )
      else:
        env.pop("CATBENCH_TASK_RANDOM_SEED", None)
        env.pop("CATBENCH_N_TASK_COMBINATIONS", None)
      Path(job.output_path).mkdir(parents=True, exist_ok=True)
      print(
          f"[RUN] model={job.model_name} category={job.category} "
          f"app={job.app_name} emulator={emu.serial}",
          flush=True,
      )
      proc = subprocess.Popen(
          command, cwd=str(REPO_ROOT), env=env, start_new_session=True
      )
      running.append((proc, job, emu, time.monotonic()))
      if launch_stagger_seconds > 0 and pending and available_emulators:
        time.sleep(launch_stagger_seconds)

    still_running: list[tuple[subprocess.Popen[Any], Job, Emulator, float]] = []
    now = time.monotonic()
    for proc, job, emu, started_at in running:
      code = proc.poll()
      if code is None:
        elapsed = now - started_at
        if job_timeout_seconds > 0 and elapsed >= job_timeout_seconds:
          if proc.pid not in timed_out_pids:
            print(
                f"[TIMEOUT] model={job.model_name} category={job.category} "
                f"app={job.app_name} emulator={emu.serial} "
                f"elapsed_seconds={elapsed:.1f} "
                f"timeout_seconds={job_timeout_seconds:.1f}",
                flush=True,
            )
            timed_out_pids.add(proc.pid)
          if not terminate_proc(proc):
            still_running.append((proc, job, emu, started_at))
            continue
          code = proc.returncode if proc.returncode is not None else -signal.SIGKILL
          if code == 0:
            code = -124
        else:
          still_running.append((proc, job, emu, started_at))
          continue
      job.exit_code = code
      status = "OK" if code == 0 else "ERR"
      print(
          f"[{status}] model={job.model_name} category={job.category} "
          f"app={job.app_name} emulator={emu.serial} exit={code}",
          flush=True,
      )
      if code != 0:
        failures += 1
        if not continue_on_error:
          for other_proc, _, _, _ in running:
            if other_proc is proc:
              continue
            if other_proc.poll() is not None:
              continue
            terminate_proc(other_proc)
          return code
      available_emulators.append(emu)
    running = still_running
    if running:
      time.sleep(2)
  return 1 if failures else 0


def _preflight_aw_env(
    emulators: list[Emulator],
    *,
    categories: tuple[str, ...],
    python_bin: str,
) -> int:
  """Repairs/checks AW-like emulator state before CATBench jobs."""
  if "maps" not in categories:
    return 0

  script = SCRIPT_DIR / "preflight_catbench_aw_env.py"
  if not script.exists():
    print(f"[PREFLIGHT] missing script: {script}", file=sys.stderr, flush=True)
    return 1

  for emu in emulators:
    command = [
        python_bin,
        str(script),
        "--serial",
        emu.serial,
        "--repair",
    ]
    print(f"[PREFLIGHT] {' '.join(command)}", flush=True)
    env = os.environ.copy()
    env["ANDROID_SERIAL"] = emu.serial
    if emu.adb_server_port is not None:
      env["ADB_SERVER_PORT"] = str(emu.adb_server_port)
      env["ANDROID_ADB_SERVER_PORT"] = str(emu.adb_server_port)
    result = subprocess.run(command, check=False, env=env)
    if result.returncode != 0:
      return result.returncode
  return 0


def _preflight_pinned_apps(
    emulators: list[Emulator],
    jobs: list[Job],
    *,
    adb_path: str,
) -> tuple[int, list[dict[str, Any]]]:
  """Checks every scheduled real package version on every target device."""
  expected: dict[str, tuple[str, str]] = {}
  for job in jobs:
    value = (job.app_version, job.app_version_code)
    prior = expected.setdefault(job.package_name, value)
    if prior != value:
      raise ValueError(
          f"Conflicting pins for package {job.package_name}: {prior} vs {value}"
      )
  report: list[dict[str, Any]] = []
  failures = 0
  for emulator in emulators:
    for package_name, (expected_name, expected_code) in sorted(expected.items()):
      adb_command = [adb_path]
      if emulator.adb_server_port is not None:
        adb_command.extend(["-P", str(emulator.adb_server_port)])
      adb_command.extend([
          "-s",
          emulator.serial,
          "shell",
          "dumpsys",
          "package",
          package_name,
      ])
      completed = subprocess.run(
          adb_command,
          check=False,
          capture_output=True,
          text=True,
          timeout=30,
      )
      output = completed.stdout or ""
      code_match = re.search(r"\bversionCode=(\d+)\b", output)
      name_match = re.search(r"^\s*versionName=(.+?)\s*$", output, re.MULTILINE)
      actual_code = code_match.group(1) if code_match else ""
      actual_name = name_match.group(1) if name_match else ""
      reasons: list[str] = []
      if completed.returncode != 0 or not actual_code or not actual_name:
        reasons.append("package_missing_or_unreadable")
      if actual_code and actual_code != str(expected_code):
        reasons.append("version_code_mismatch")
      if actual_name and actual_name != str(expected_name):
        reasons.append("version_name_mismatch")
      failures += bool(reasons)
      report.append({
          "serial": emulator.serial,
          "adb_server_port": emulator.adb_server_port,
          "package_name": package_name,
          "expected_version_name": expected_name,
          "expected_version_code": expected_code,
          "actual_version_name": actual_name,
          "actual_version_code": actual_code,
          "valid": not reasons,
          "reasons": reasons,
      })
  return (1 if failures else 0), report


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model_config", default=str(CONFIG_PATH))
  parser.add_argument(
      "--env_file",
      default=str(ENV_FILE_PATH),
      help=(
          "Optional env file loaded before expanding model config arguments. "
          "Existing environment variables are preserved."
      ),
  )
  parser.add_argument("--models", default="", help="Comma-separated model names.")
  parser.add_argument(
      "--categories",
      default=",".join(DEFAULT_CATEGORIES),
      help="Comma-separated categories to run.",
  )
  parser.add_argument(
      "--emulators",
      default=os.environ.get("CATBENCH_EMULATORS", "5554:8554"),
      help=(
          "Comma-separated console:grpc[:serial[:adb_server_port]] specs, "
          "e.g. 5576:8576:emulator-5576:5041."
      ),
  )
  parser.add_argument("--output_root", default=str(_default_output_root()))
  parser.add_argument("--run_id", default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
  parser.add_argument("--python", default=sys.executable)
  parser.add_argument(
      "--adb",
      default=os.environ.get("ADB", "adb"),
      help="ADB executable used for installed package/version preflight.",
  )
  parser.add_argument("--n_task_combinations", type=int, default=3)
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
          "Optional SHA-256 identity for a reproducibly archived dirty source "
          "snapshot; CATBENCH_SOURCE_SNAPSHOT_SHA256 is honored if omitted."
      ),
  )
  parser.add_argument(
      "--app_pins",
      default=str(APP_PINS_PATH),
      help="Frozen CSV containing package versions and APK SHA-256 values.",
  )
  parser.add_argument(
      "--app_artifact_root",
      default=str(DEFAULT_APP_ARTIFACT_ROOT),
      help="Directory containing the frozen APK/XAPK inventory.",
  )
  parser.add_argument(
      "--condition",
      choices=("development", "c1", "c2_g", "c2_o"),
      default="development",
      help="Explicit episode condition persisted in every release artifact.",
  )
  parser.add_argument(
      "--release_purpose",
      default="development_matrix",
      help="Purpose label persisted in the matrix, jobs, and episodes.",
  )
  parser.add_argument(
      "--artifact_role",
      default="development_matrix_not_primary_analysis",
      help="Artifact-role label persisted in the matrix, jobs, and episodes.",
  )
  parser.add_argument(
      "--analysis_eligible",
      type=_parse_bool,
      default=False,
      metavar="true|false",
      help=(
          "Whether outputs are eligible for analysis. True is rejected unless "
          "strict frozen-primary mode passes all existing gates."
      ),
  )
  parser.add_argument(
      "--primary_release_manifest",
      default="",
      help=(
          "Enable strict primary-release mode using this frozen cohort JSON. "
          "Selective filters, resume, dirty source, and roster drift then fail."
      ),
  )
  parser.add_argument(
      "--allow_dirty_source",
      action="store_true",
      help="Development only: permit a non-dry launch from a dirty worktree.",
  )
  parser.add_argument("--max_parallel", type=int, default=0)
  parser.add_argument(
      "--launch_stagger_seconds",
      type=float,
      default=float(os.environ.get("CATBENCH_LAUNCH_STAGGER_SECONDS", "0")),
      help=(
          "Seconds to wait between launching concurrent emulator jobs. "
          "Useful for avoiding ADB startup races."
      ),
  )
  parser.add_argument(
      "--job_timeout_seconds",
      type=float,
      default=float(os.environ.get("CATBENCH_JOB_TIMEOUT_SECONDS", "0")),
      help=(
          "Per app-level job wall-clock timeout in seconds. 0 disables the "
          "timeout. Timed-out jobs are terminated, counted as failures, and "
          "the emulator is released for the next job when --continue_on_error "
          "is set."
      ),
  )
  parser.add_argument(
      "--prelaunch_delay_seconds",
      type=float,
      default=float(os.environ.get("CATBENCH_PRELAUNCH_DELAY_SECONDS", "0")),
      help=(
          "Seconds to wait after writing the manifest and before launching the "
          "first emulator job. This lets live watchers attach to the run."
      ),
  )
  parser.add_argument(
      "--skip_aw_env_preflight",
      action="store_true",
      default=os.environ.get("CATBENCH_SKIP_AW_ENV_PREFLIGHT", "") == "1",
      help=(
          "Skip the AndroidWorld-environment preflight. By default the matrix "
          "repairs the AW OsmAnd offline map before Maps jobs."
      ),
  )
  parser.add_argument(
      "--skip_app_pin_preflight",
      action="store_true",
      help="Development only: skip installed package/version verification.",
  )
  parser.add_argument(
      "--skip_app_artifact_preflight",
      action="store_true",
      help="Development only: skip local APK/XAPK SHA-256 verification.",
  )
  parser.add_argument("--dry_run", action="store_true")
  parser.add_argument("--continue_on_error", action="store_true")
  parser.add_argument(
      "--resume_existing",
      action="store_true",
      help=(
          "Resume each app-run from the newest existing run_* checkpoint "
          "directory under its output path. Fully completed tasks are skipped "
          "by the AndroidWorld checkpointer; an interrupted current task starts "
          "again from its last completed checkpoint."
      ),
  )
  parser.add_argument(
      "--runner_arg",
      action="append",
      default=[],
      help="Extra argument forwarded to every runner. Can be repeated.",
  )
  parser.add_argument(
      "--task_regex",
      default="",
      help=(
          "Optional regular expression applied to task class names after the "
          "category/app profile is selected. Apps with no matching tasks are "
          "skipped."
      ),
  )
  parser.add_argument(
      "--app_ids",
      default="",
      help=(
          "Optional comma-separated profile app IDs to include "
          "(for example sms_simple_sms_messenger,clock_google_clock)."
      ),
  )
  args = parser.parse_args()
  env_file = Path(args.env_file).expanduser()
  loaded_env = _load_env_file(env_file)
  if loaded_env:
    print(f"Loaded {loaded_env} variables from {env_file}", flush=True)
  leaked_exact_overrides = [
      name for name in exact_task_params_lib.ENV_NAMES
      if os.environ.get(name, "").strip()
  ]
  if leaked_exact_overrides:
    raise ValueError(
        "Full matrix and primary runs forbid exact task-parameter overrides; "
        "use run_catbench_target_cells.py for audited invalid-episode "
        "replacement candidates. Leaked variables: "
        + ", ".join(leaked_exact_overrides)
    )
  if args.n_task_combinations < 1:
    raise ValueError("n_task_combinations must be positive.")
  selected_instance_id = _resolve_instance_id(
      args.instance_id, args.n_task_combinations
  )
  source_snapshot_sha256 = _resolve_source_snapshot_sha256(
      args.source_snapshot_sha256
  )
  args.release_purpose = str(args.release_purpose).strip()
  args.artifact_role = str(args.artifact_role).strip()
  if not args.release_purpose or not args.artifact_role:
    raise ValueError("release_purpose and artifact_role must be non-empty.")

  selected = {m.strip() for m in args.models.split(",") if m.strip()} or None
  categories = tuple(c.strip() for c in args.categories.split(",") if c.strip())
  profiles = get_domain_profiles()
  unknown = [c for c in categories if c not in profiles]
  if unknown:
    raise ValueError(f"Unknown categories: {unknown}. Valid: {sorted(profiles)}")

  models = _load_models(Path(args.model_config), selected)
  if not models:
    raise ValueError("No models selected.")

  primary_cohort = None
  primary_cohort_path = None
  if args.primary_release_manifest:
    primary_cohort_path = Path(
        args.primary_release_manifest
    ).expanduser().resolve()
    primary_cohort = _load_primary_cohort(primary_cohort_path)
    prohibited = []
    if args.allow_dirty_source:
      prohibited.append("--allow_dirty_source")
    if args.resume_existing:
      prohibited.append("--resume_existing")
    if args.task_regex:
      prohibited.append("--task_regex")
    if args.app_ids:
      prohibited.append("--app_ids")
    if args.runner_arg:
      prohibited.append("--runner_arg")
    if args.skip_app_pin_preflight:
      prohibited.append("--skip_app_pin_preflight")
    if args.skip_app_artifact_preflight:
      prohibited.append("--skip_app_artifact_preflight")
    if selected_instance_id is not None:
      prohibited.append("--instance_id/CATBENCH_INSTANCE_ID")
    if prohibited:
      raise ValueError(
          "Primary-release mode forbids selective/development options: "
          + ", ".join(prohibited)
      )
    if args.condition not in set(primary_cohort["conditions"]):
      raise ValueError(
          f"Primary-release condition must be one of "
          f"{primary_cohort['conditions']}; got {args.condition!r}."
      )
    breakdown_file = os.environ.get("CATBENCH_TASK_BREAKDOWN_FILE", "")
    if args.condition == "c1" and breakdown_file:
      raise ValueError("C1 must not set CATBENCH_TASK_BREAKDOWN_FILE.")
    if args.condition in {"c2_g", "c2_o"} and not breakdown_file:
      raise ValueError(
          f"{args.condition} requires CATBENCH_TASK_BREAKDOWN_FILE."
      )
    issues = _primary_release_violations(
        primary_cohort,
        categories=categories,
        models=models,
        n_task_combinations=args.n_task_combinations,
        task_random_seed=args.task_random_seed,
    )
    if issues:
      raise RuntimeError(
          "Frozen primary-cohort preflight failed:\n- " + "\n- ".join(issues)
      )

  if args.analysis_eligible:
    if primary_cohort is None:
      raise ValueError(
          "analysis_eligible=true requires strict --primary_release_manifest "
          "mode; revision/development runs must remain ineligible."
      )
    expected_labels = (
        "primary_five_category_analysis",
        "primary_analysis_candidate",
    )
    if (args.release_purpose, args.artifact_role) != expected_labels:
      raise ValueError(
          "analysis_eligible=true requires release_purpose="
          "primary_five_category_analysis and artifact_role="
          "primary_analysis_candidate."
      )

  emulators = _parse_emulators(args.emulators)
  code_revision, dirty_source = _source_revision()
  if dirty_source and primary_cohort is not None:
    raise RuntimeError(
        "Primary-release mode requires a clean frozen source tree, including "
        "for its release dry run."
    )
  if dirty_source and not args.dry_run and not args.allow_dirty_source:
    raise RuntimeError(
        "Refusing to launch from a dirty worktree. Commit the frozen release "
        "or use --allow_dirty_source only for discarded development smoke tests."
    )
  app_pins_path = Path(args.app_pins).expanduser().resolve()
  app_pins = _load_app_pins(app_pins_path)
  model_config_path = Path(args.model_config).expanduser().resolve()
  model_config_sha256 = _file_sha256(model_config_path)
  app_pins_sha256 = _file_sha256(app_pins_path)
  app_artifact_report: dict[str, Any] | None = None
  if primary_cohort is not None and not args.skip_app_artifact_preflight:
    import audit_pinned_app_artifacts  # pylint: disable=import-outside-toplevel

    app_artifact_report = audit_pinned_app_artifacts.audit(
        primary_cohort_path,
        app_pins_path,
        APP_INVENTORY_PATH,
        Path(args.app_artifact_root).expanduser().resolve(),
    )
    if not app_artifact_report["valid"]:
      invalid_app_ids = [
          row["app_id"]
          for row in app_artifact_report["apps"]
          if not row["valid"]
      ]
      raise RuntimeError(
          "Frozen app-artifact preflight failed for: "
          + ", ".join(invalid_app_ids)
      )
  max_parallel = args.max_parallel or len(emulators)
  output_root = Path(args.output_root).expanduser().resolve()
  task_regex = re.compile(args.task_regex) if args.task_regex else None
  app_ids = {item.strip() for item in args.app_ids.split(",") if item.strip()} or None

  job_groups: list[list[Job]] = []
  release_id = (
      str(primary_cohort["release_id"])
      if primary_cohort is not None
      else args.run_id
  )
  for model in models:
    job_groups.append(
        _task_jobs_for_model(
            model=model,
            categories=categories,
            output_root=output_root,
            run_id=args.run_id,
            python_bin=args.python,
            n_task_combinations=args.n_task_combinations,
            task_random_seed=args.task_random_seed,
            extra_runner_args=args.runner_arg,
            resume_existing=args.resume_existing,
            task_regex=task_regex,
            app_ids=app_ids,
            app_pins=app_pins,
            code_revision=code_revision,
            source_snapshot_sha256=source_snapshot_sha256,
            release_id=release_id,
            release_purpose=args.release_purpose,
            artifact_role=args.artifact_role,
            analysis_eligible=args.analysis_eligible,
            condition=args.condition,
            instance_id=selected_instance_id,
            model_config_sha256=model_config_sha256,
            app_pins_sha256=app_pins_sha256,
        )
    )
  jobs: list[Job] = []
  max_group_len = max((len(group) for group in job_groups), default=0)
  for idx in range(max_group_len):
    for group in job_groups:
      if idx < len(group):
        jobs.append(group[idx])

  if primary_cohort is not None:
    expected_jobs = len(primary_cohort["models"]) * int(
        primary_cohort["expected"]["app_count"]
    )
    if len(jobs) != expected_jobs:
      raise RuntimeError(
          f"Primary release scheduled {len(jobs)} app jobs; expected "
          f"{expected_jobs}."
      )

  device_app_preflight: list[dict[str, Any]] = []
  device_app_preflight_code = 0
  if not args.dry_run and not args.skip_app_pin_preflight:
    device_app_preflight_code, device_app_preflight = _preflight_pinned_apps(
        emulators, jobs, adb_path=args.adb
    )

  manifest_path = output_root / args.run_id / "catbench_5cat_manifest.json"
  task_breakdown_file = os.environ.get("CATBENCH_TASK_BREAKDOWN_FILE", "")
  task_breakdown_sha256 = (
      _file_sha256(Path(task_breakdown_file).expanduser().resolve())
      if task_breakdown_file
      else ""
  )
  primary_cohort_sha256 = (
      _file_sha256(primary_cohort_path) if primary_cohort_path else ""
  )
  provenance = {
      "release_id": release_id,
      "release_purpose": args.release_purpose,
      "artifact_role": args.artifact_role,
      "analysis_eligible": args.analysis_eligible,
      "git_revision": code_revision,
      "dirty_source": dirty_source,
      "source_snapshot_sha256": source_snapshot_sha256,
      "model_config_sha256": model_config_sha256,
      "app_pins_file": str(app_pins_path),
      "app_pins_sha256": app_pins_sha256,
      "app_artifact_preflight": app_artifact_report,
      "primary_cohort_file": str(primary_cohort_path or ""),
      "primary_cohort_sha256": primary_cohort_sha256,
      "device_app_preflight": device_app_preflight,
      "matrix_args": {
          "condition": args.condition,
          "categories": list(categories),
          "n_task_combinations": args.n_task_combinations,
          "task_random_seed": args.task_random_seed,
          "instance_id": selected_instance_id,
          "catbench_instance_id": selected_instance_id,
          "release_purpose": args.release_purpose,
          "artifact_role": args.artifact_role,
          "analysis_eligible": args.analysis_eligible,
          "source_snapshot_sha256": source_snapshot_sha256,
          "task_breakdown_file": task_breakdown_file,
          "task_breakdown_sha256": task_breakdown_sha256,
          # Record the mode the runner will actually apply (an unset MODE
          # defaults to "prepend" whenever a file is set), not a local default.
          "task_breakdown_mode": task_breakdowns_lib.effective_mode(),
          "task_breakdown_required": task_breakdowns_lib.is_required(),
      },
  }
  _write_manifest(manifest_path, jobs, args.dry_run, provenance)
  print(f"Manifest: {manifest_path}")
  print(f"Jobs: {len(jobs)}  Models: {len(models)}  Emulators: {len(emulators)}")
  if device_app_preflight_code != 0:
    print(
        "Installed package/version preflight failed; see manifest.",
        file=sys.stderr,
    )
    return device_app_preflight_code
  if not args.dry_run and not args.skip_aw_env_preflight:
    preflight_code = _preflight_aw_env(
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

  code = _run_jobs(
      jobs,
      emulators=emulators,
      max_parallel=max_parallel,
      dry_run=args.dry_run,
      continue_on_error=args.continue_on_error,
      launch_stagger_seconds=args.launch_stagger_seconds,
      job_timeout_seconds=args.job_timeout_seconds,
      plan_file_sha256=task_breakdown_sha256,
      cohort_sha256=primary_cohort_sha256,
  )
  _write_manifest(manifest_path, jobs, args.dry_run, provenance)
  return code


if __name__ == "__main__":
  raise SystemExit(main())
