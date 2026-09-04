#!/usr/bin/env python3
"""Record per-job Android emulator videos for a live CATBench matrix run.

The recorder tails the matrix log, starts one `adb shell screenrecord` process
for each `[RUN] ... emulator=...` line, and stops/pulls the recording when the
matching `[OK]` or `[ERR]` line appears. Android's screenrecord has a 180 second
limit, so long app-jobs are stored as multiple MP4 segments.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


RUN_RE = re.compile(
    r"^\[RUN\] model=(?P<model>.*?) category=(?P<category>\S+) "
    r"app=(?P<app>.*?) emulator=(?P<serial>emulator-\d+)"
)
DONE_RE = re.compile(
    r"^\[(?P<status>OK|ERR)\] model=(?P<model>.*?) category=(?P<category>\S+) "
    r"app=(?P<app>.*?) emulator=(?P<serial>emulator-\d+) exit=(?P<exit>-?\d+)"
)


def _now() -> str:
  return dt.datetime.now().isoformat(timespec="seconds")


def _safe_slug(value: str) -> str:
  slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
  return slug.strip("_") or "unknown"


def _run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      cmd,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      timeout=timeout,
      check=False,
  )


@dataclasses.dataclass(frozen=True)
class JobKey:
  model: str
  category: str
  app: str


@dataclasses.dataclass
class JobInfo:
  key: JobKey
  serial: str
  output_path: Path
  status: str | None = None
  exit_code: int | None = None


class JobRecorder:
  """Records one app-job into its output_path/videos directory."""

  def __init__(
      self,
      job: JobInfo,
      adb: str,
      segment_seconds: int,
      bit_rate: str,
      size: str,
      index_path: Path,
      log_path: Path,
  ):
    self.job = job
    self.adb = adb
    self.segment_seconds = segment_seconds
    self.bit_rate = bit_rate
    self.size = size
    self.index_path = index_path
    self.log_path = log_path
    self.stop_event = threading.Event()
    self.thread = threading.Thread(target=self._run, daemon=True)
    self.video_dir = job.output_path / "videos"
    self.segment = 0

  def start(self) -> None:
    self.video_dir.mkdir(parents=True, exist_ok=True)
    self._write_metadata("start")
    self.thread.start()

  def stop(self, status: str | None = None, exit_code: int | None = None) -> None:
    self.job.status = status
    self.job.exit_code = exit_code
    self.stop_event.set()
    self._interrupt_screenrecord()

  def join(self, timeout: float | None = None) -> None:
    self.thread.join(timeout=timeout)

  def _record_index(self, payload: dict[str, Any]) -> None:
    payload = {
        "time": _now(),
        "model": self.job.key.model,
        "category": self.job.key.category,
        "app": self.job.key.app,
        "serial": self.job.serial,
        **payload,
    }
    with self.index_path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(payload, sort_keys=True) + "\n")

  def _log(self, message: str) -> None:
    line = f"[{_now()}] {self.job.serial} {self.job.key.model} {self.job.key.category}/{self.job.key.app}: {message}\n"
    with self.log_path.open("a", encoding="utf-8") as handle:
      handle.write(line)

  def _write_metadata(self, event: str) -> None:
    metadata = {
        "event": event,
        "time": _now(),
        "model": self.job.key.model,
        "category": self.job.key.category,
        "app": self.job.key.app,
        "serial": self.job.serial,
        "output_path": str(self.job.output_path),
        "segment_seconds": self.segment_seconds,
        "bit_rate": self.bit_rate,
        "size": self.size or None,
    }
    if self.job.status is not None:
      metadata["status"] = self.job.status
      metadata["exit_code"] = self.job.exit_code
    path = self.video_dir / "recording_metadata.jsonl"
    with path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(metadata, sort_keys=True) + "\n")

  def _interrupt_screenrecord(self) -> None:
    _run([self.adb, "-s", self.job.serial, "shell", "pkill", "-2", "screenrecord"], timeout=5)

  def _device_path(self) -> str:
    return (
        "/sdcard/catbench_recordings/"
        f"{_safe_slug(self.job.key.model)}_{_safe_slug(self.job.key.category)}_"
        f"{_safe_slug(self.job.key.app)}_{os.getpid()}_{self.segment:03d}.mp4"
    )

  def _run(self) -> None:
    _run([self.adb, "-s", self.job.serial, "shell", "mkdir", "-p", "/sdcard/catbench_recordings"], timeout=10)
    self._log("recording started")
    while not self.stop_event.is_set():
      device_path = self._device_path()
      local_path = self.video_dir / f"segment_{self.segment:03d}.mp4"
      cmd = [
          self.adb,
          "-s",
          self.job.serial,
          "shell",
          "screenrecord",
          "--bit-rate",
          self.bit_rate,
          "--time-limit",
          str(self.segment_seconds),
      ]
      if self.size:
        cmd.extend(["--size", self.size])
      cmd.append(device_path)

      self._log(f"segment {self.segment:03d} start")
      proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
      while proc.poll() is None and not self.stop_event.is_set():
        time.sleep(1)
      if self.stop_event.is_set() and proc.poll() is None:
        try:
          proc.send_signal(signal.SIGINT)
          proc.wait(timeout=5)
        except Exception:
          self._interrupt_screenrecord()
          try:
            proc.wait(timeout=5)
          except Exception:
            proc.kill()
      else:
        proc.wait()

      pull = _run([self.adb, "-s", self.job.serial, "pull", device_path, str(local_path)], timeout=120)
      _run([self.adb, "-s", self.job.serial, "shell", "rm", "-f", device_path], timeout=10)
      if local_path.exists() and local_path.stat().st_size > 0:
        self._record_index({
            "event": "segment",
            "segment": self.segment,
            "path": str(local_path),
            "bytes": local_path.stat().st_size,
        })
        self._log(f"segment {self.segment:03d} saved: {local_path}")
        self.segment += 1
      else:
        self._log(f"segment {self.segment:03d} pull failed or empty: {pull.stdout.strip()[:500]}")
        if self.stop_event.is_set():
          break
        time.sleep(2)
    self._write_metadata("stop")
    self._record_index({
        "event": "job_stop",
        "status": self.job.status,
        "exit_code": self.job.exit_code,
        "segments": self.segment,
        "video_dir": str(self.video_dir),
    })
    self._log("recording stopped")


def _load_output_paths(manifest: Path) -> dict[JobKey, Path]:
  payload = json.loads(manifest.read_text(encoding="utf-8"))
  out: dict[JobKey, Path] = {}
  for job in payload.get("jobs", []):
    key = JobKey(job["model_name"], job["category"], job["app_name"])
    out[key] = Path(job["output_path"])
  return out


def _events_from_line(line: str) -> tuple[str, JobInfo | None]:
  match = RUN_RE.match(line)
  if match:
    key = JobKey(match.group("model"), match.group("category"), match.group("app"))
    return "run", JobInfo(key=key, serial=match.group("serial"), output_path=Path())
  match = DONE_RE.match(line)
  if match:
    key = JobKey(match.group("model"), match.group("category"), match.group("app"))
    return "done", JobInfo(
        key=key,
        serial=match.group("serial"),
        output_path=Path(),
        status=match.group("status"),
        exit_code=int(match.group("exit")),
    )
  return "", None


def _scan_active_jobs(matrix_log: Path, output_paths: dict[JobKey, Path]) -> dict[JobKey, JobInfo]:
  active: dict[JobKey, JobInfo] = {}
  if not matrix_log.exists():
    return active
  with matrix_log.open("r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
      event, job = _events_from_line(line.rstrip("\n"))
      if not job:
        continue
      if event == "run" and job.key in output_paths:
        job.output_path = output_paths[job.key]
        active[job.key] = job
      elif event == "done":
        active.pop(job.key, None)
  return active


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True)
  parser.add_argument("--matrix_log", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--segment_seconds", type=int, default=170)
  parser.add_argument("--bit_rate", default="4M")
  parser.add_argument("--size", default="")
  parser.add_argument("--poll_seconds", type=float, default=1.0)
  parser.add_argument("--run_root", default="")
  args = parser.parse_args()

  manifest = Path(args.manifest).expanduser().resolve()
  matrix_log = Path(args.matrix_log).expanduser().resolve()
  run_root = Path(args.run_root).expanduser().resolve() if args.run_root else matrix_log.parent.parent
  recorder_dir = run_root / "videos"
  recorder_dir.mkdir(parents=True, exist_ok=True)
  index_path = recorder_dir / "video_index.jsonl"
  log_path = recorder_dir / "video_recorder.log"

  output_paths = _load_output_paths(manifest)
  recorders: dict[JobKey, JobRecorder] = {}

  def start_job(job: JobInfo) -> None:
    if job.key in recorders or job.key not in output_paths:
      return
    job.output_path = output_paths[job.key]
    recorder = JobRecorder(
        job=job,
        adb=args.adb,
        segment_seconds=min(args.segment_seconds, 180),
        bit_rate=args.bit_rate,
        size=args.size,
        index_path=index_path,
        log_path=log_path,
    )
    recorders[job.key] = recorder
    recorder.start()

  for job in _scan_active_jobs(matrix_log, output_paths).values():
    start_job(job)

  with log_path.open("a", encoding="utf-8") as handle:
    handle.write(f"[{_now()}] recorder started manifest={manifest} matrix_log={matrix_log}\n")

  position = matrix_log.stat().st_size if matrix_log.exists() else 0
  try:
    while True:
      if not matrix_log.exists():
        time.sleep(args.poll_seconds)
        continue
      with matrix_log.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(position)
        for line in handle:
          event, job = _events_from_line(line.rstrip("\n"))
          if job is None:
            continue
          if event == "run":
            start_job(job)
          elif event == "done":
            recorder = recorders.pop(job.key, None)
            if recorder is not None:
              recorder.stop(job.status, job.exit_code)
              recorder.join(timeout=180)
        position = handle.tell()
      time.sleep(args.poll_seconds)
  except KeyboardInterrupt:
    pass
  finally:
    for recorder in list(recorders.values()):
      recorder.stop("INTERRUPTED", None)
    for recorder in list(recorders.values()):
      recorder.join(timeout=180)
    with log_path.open("a", encoding="utf-8") as handle:
      handle.write(f"[{_now()}] recorder stopped\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
