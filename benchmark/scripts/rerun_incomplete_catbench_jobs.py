#!/usr/bin/env python3
"""Resume incomplete CATBench manifest jobs in place.

This is intentionally narrower than run_catbench_5cat_matrix.py: it reads an
existing manifest, finds app-level jobs whose latest checkpoint has fewer
episodes than scheduled, and reruns those exact commands with --checkpoint_dir.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any


@dataclasses.dataclass
class Emulator:
  console_port: int
  grpc_port: int

  @property
  def serial(self) -> str:
    return f"emulator-{self.console_port}"


@dataclasses.dataclass
class PendingJob:
  job: dict[str, Any]
  checkpoint_dir: Path
  expected: int
  existing: int


def _parse_emulators(raw: str) -> list[Emulator]:
  emulators: list[Emulator] = []
  for item in raw.split(","):
    item = item.strip()
    if not item:
      continue
    console, grpc = item.split(":", 1)
    emulators.append(Emulator(int(console), int(grpc)))
  if not emulators:
    raise ValueError("No emulators provided.")
  return emulators


def _latest_checkpoint_dir(output_path: Path) -> Path | None:
  if not output_path.is_dir():
    return None
  run_dirs = [
      path for path in output_path.iterdir()
      if path.is_dir() and path.name.startswith("run_")
  ]
  if not run_dirs:
    return None
  return max(run_dirs, key=lambda path: (path.name, path.stat().st_mtime_ns))


def _arg_value(command: list[str], prefix: str, default: str = "") -> str:
  for arg in command:
    if arg.startswith(prefix):
      return arg[len(prefix):]
  return default


def _expected_checkpoint_count(command: list[str]) -> int:
  tasks = [task for task in _arg_value(command, "--tasks=").split(",") if task]
  n_raw = _arg_value(command, "--n_task_combinations=", "1")
  try:
    n = int(n_raw)
  except ValueError:
    n = 1
  return len(tasks) * n


def _missing_task_templates(command: list[str], checkpoint_dir: Path) -> list[str]:
  tasks = [task for task in _arg_value(command, "--tasks=").split(",") if task]
  n_raw = _arg_value(command, "--n_task_combinations=", "1")
  try:
    n = int(n_raw)
  except ValueError:
    n = 1
  if n != 1:
    return tasks
  return [
      task for task in tasks
      if not (checkpoint_dir / f"{task}_0.pkl.gz").exists()
  ]


def _checkpoint_count(checkpoint_dir: Path | None) -> int:
  if checkpoint_dir is None or not checkpoint_dir.is_dir():
    return 0
  return sum(1 for _ in checkpoint_dir.glob("*.pkl.gz"))


def _strip_runtime_args(command: list[str]) -> list[str]:
  prefixes = ("--checkpoint_dir=", "--console_port=", "--grpc_port=")
  return [arg for arg in command if not arg.startswith(prefixes)]


def _replace_or_append_arg(
    command: list[str], prefix: str, value: str
) -> list[str]:
  replacement = f"{prefix}{value}"
  replaced = False
  updated: list[str] = []
  for arg in command:
    if arg.startswith(prefix):
      updated.append(replacement)
      replaced = True
    else:
      updated.append(arg)
  if not replaced:
    updated.append(replacement)
  return updated


def _parse_endpoint_pools(raw_items: list[str]) -> dict[str, list[str]]:
  pools: dict[str, list[str]] = {}
  for raw in raw_items:
    model, sep, urls = raw.partition("=")
    model = model.strip()
    if not sep or not model:
      raise ValueError(
          "--endpoint_pool entries must look like MODEL=url1,url2"
      )
    pool = [url.strip() for url in urls.split(",") if url.strip()]
    if not pool:
      raise ValueError(f"No endpoints provided for model {model!r}.")
    pools[model] = pool
  return pools


def _incomplete_jobs(manifest: Path) -> list[PendingJob]:
  payload = json.loads(manifest.read_text(encoding="utf-8"))
  pending: list[PendingJob] = []
  for job in payload.get("jobs", []):
    if not isinstance(job, dict):
      continue
    command = [str(arg) for arg in job.get("command", [])]
    output_path = Path(str(job.get("output_path", ""))).expanduser()
    checkpoint_dir = _latest_checkpoint_dir(output_path)
    expected = _expected_checkpoint_count(command)
    existing = _checkpoint_count(checkpoint_dir)
    if expected and existing < expected and checkpoint_dir is not None:
      pending.append(
          PendingJob(
              job=job,
              checkpoint_dir=checkpoint_dir,
              expected=expected,
              existing=existing,
          )
      )
  return pending


def _terminate_proc(proc: subprocess.Popen[Any]) -> None:
  if proc.poll() is not None:
    return
  try:
    os.killpg(proc.pid, signal.SIGTERM)
  except ProcessLookupError:
    return
  except OSError:
    proc.terminate()
  try:
    proc.wait(timeout=30)
    return
  except subprocess.TimeoutExpired:
    pass
  try:
    os.killpg(proc.pid, signal.SIGKILL)
  except ProcessLookupError:
    return
  except OSError:
    proc.kill()
  proc.wait(timeout=30)


def _run(
    pending: list[PendingJob],
    emulators: list[Emulator],
    max_parallel: int,
    max_per_model: int,
    launch_stagger_seconds: float,
    job_timeout_seconds: float,
    endpoint_pools: dict[str, list[str]],
) -> int:
  available = list(emulators[:max_parallel])
  running: list[
      tuple[subprocess.Popen[Any], PendingJob, Emulator, float, Any]
  ] = []
  running_by_model: dict[str, int] = {}
  endpoint_cursor: dict[str, int] = {}
  failures = 0

  def acquire_job() -> PendingJob | None:
    for idx, item in enumerate(pending):
      model = str(item.job.get("model_name", ""))
      if running_by_model.get(model, 0) < max_per_model:
        return pending.pop(idx)
    return None

  while pending or running:
    launched = False
    while available and pending:
      item = acquire_job()
      if item is None:
        break
      emu = available.pop(0)
      job = item.job
      model = str(job.get("model_name", ""))
      command = [
          *_strip_runtime_args([str(arg) for arg in job.get("command", [])]),
          f"--checkpoint_dir={item.checkpoint_dir}",
          f"--console_port={emu.console_port}",
          f"--grpc_port={emu.grpc_port}",
      ]
      missing_tasks = _missing_task_templates(command, item.checkpoint_dir)
      if not missing_tasks:
        available.append(emu)
        print(
            "[DONE] "
            f"model={model} category={job.get('category')} app={job.get('app_name')} "
            f"exit=0 checkpoint={item.existing}/{item.expected}",
            flush=True,
        )
        continue
      command = _replace_or_append_arg(
          command, "--tasks=", ",".join(missing_tasks)
      )
      if model in endpoint_pools:
        pool = endpoint_pools[model]
        cursor = endpoint_cursor.get(model, 0)
        endpoint = pool[cursor % len(pool)]
        endpoint_cursor[model] = cursor + 1
        command = _replace_or_append_arg(command, "--endpoint_url=", endpoint)
      env = os.environ.copy()
      env["ANDROID_SERIAL"] = emu.serial
      Path(str(job["output_path"])).mkdir(parents=True, exist_ok=True)
      log_path = (
          item.checkpoint_dir
          / f"resume_{int(time.time())}_{emu.serial}.log"
      )
      log_handle = log_path.open("a", encoding="utf-8", buffering=1)
      print(
          "[RESUME] "
          f"model={model} category={job.get('category')} app={job.get('app_name')} "
          f"checkpoint={item.existing}/{item.expected} emulator={emu.serial} "
          f"log={log_path}",
          flush=True,
      )
      proc = subprocess.Popen(
          command,
          cwd=str(Path(__file__).resolve().parents[2]),
          env=env,
          start_new_session=True,
          stdout=log_handle,
          stderr=subprocess.STDOUT,
      )
      running.append((proc, item, emu, time.monotonic(), log_handle))
      running_by_model[model] = running_by_model.get(model, 0) + 1
      launched = True
      if launch_stagger_seconds > 0 and available and pending:
        time.sleep(launch_stagger_seconds)

    still_running: list[
        tuple[subprocess.Popen[Any], PendingJob, Emulator, float, Any]
    ] = []
    for proc, item, emu, started_at, log_handle in running:
      code = proc.poll()
      if code is None and job_timeout_seconds > 0:
        elapsed = time.monotonic() - started_at
        if elapsed >= job_timeout_seconds:
          print(
              "[TIMEOUT] "
              f"model={item.job.get('model_name')} category={item.job.get('category')} "
              f"app={item.job.get('app_name')} emulator={emu.serial} "
              f"elapsed_seconds={elapsed:.1f}",
              flush=True,
          )
          _terminate_proc(proc)
          code = proc.returncode if proc.returncode is not None else -124
      if code is None:
        still_running.append((proc, item, emu, started_at, log_handle))
        continue

      log_handle.close()
      model = str(item.job.get("model_name", ""))
      running_by_model[model] = max(0, running_by_model.get(model, 1) - 1)
      available.append(emu)
      new_count = _checkpoint_count(item.checkpoint_dir)
      print(
          "[DONE] "
          f"model={model} category={item.job.get('category')} app={item.job.get('app_name')} "
          f"exit={code} checkpoint={new_count}/{item.expected}",
          flush=True,
      )
      if code != 0 or new_count < item.expected:
        failures += 1
    running = still_running

    if not launched:
      time.sleep(5)

  return 1 if failures else 0


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True)
  parser.add_argument("--emulators", required=True)
  parser.add_argument("--max_parallel", type=int, default=3)
  parser.add_argument("--max_per_model", type=int, default=1)
  parser.add_argument("--launch_stagger_seconds", type=float, default=10.0)
  parser.add_argument("--job_timeout_seconds", type=float, default=0.0)
  parser.add_argument(
      "--endpoint_pool",
      action="append",
      default=[],
      help=(
          "Optional MODEL=url1,url2 pool. Matching jobs have --endpoint_url "
          "replaced round-robin at launch time."
      ),
  )
  parser.add_argument(
      "--list_only",
      action="store_true",
      help="Only print incomplete jobs; do not launch any reruns.",
  )
  args = parser.parse_args()

  manifest = Path(args.manifest).expanduser().resolve()
  pending = _incomplete_jobs(manifest)
  print(f"Incomplete jobs: {len(pending)}", flush=True)
  for item in pending:
    print(
        "  "
        f"{item.job.get('model_name')} | {item.job.get('category')} | "
        f"{item.job.get('app_id')} | {item.existing}/{item.expected}",
        flush=True,
    )
  if not pending:
    return 0
  if args.list_only:
    return 0
  emulators = _parse_emulators(args.emulators)
  endpoint_pools = _parse_endpoint_pools(args.endpoint_pool)
  return _run(
      pending=pending,
      emulators=emulators,
      max_parallel=max(1, min(args.max_parallel, len(emulators))),
      max_per_model=max(1, args.max_per_model),
      launch_stagger_seconds=args.launch_stagger_seconds,
      job_timeout_seconds=args.job_timeout_seconds,
      endpoint_pools=endpoint_pools,
  )


if __name__ == "__main__":
  raise SystemExit(main())
