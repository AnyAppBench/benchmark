#!/usr/bin/env python3
"""Run the 6 AW-canonical Clock/Maps tasks N times per agent for reproduction.

Purpose: defend the paper claim "for the AW-canonical task x AW-canonical app
subset, CATBench's measurement matches the published AW numbers within X pp"
against a reviewer.

This script does NOT use the cross-app matrix runner. It schedules the 6
AW-canonical task classes (which point at AW's original validators in
``android_world/task_evals/single/clock.py`` and ``.../osmand.py``) and
invokes per-agent runners that already accept ``--tasks=`` flags.

Outputs:
  --out_root/<agent_short>/trial_<i>/<TaskName>_*.pkl.gz   raw run artifacts
  --out_root/index.json                                     trial manifest

Pair with ``report_aw_reproduction.py`` to produce the comparison table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_ROOT.parent


# The 6 AW-canonical tasks. Names and modules verified against the live
# AW registry. Do not change without re-verifying.
AW_CANONICAL_TASKS: tuple[str, ...] = (
    "ClockTimerEntry",
    "ClockStopWatchRunning",
    "ClockStopWatchPausedVerify",
    "OsmAndFavorite",
    "OsmAndMarker",
    "OsmAndTrack",
)


# Default agent recipes. Each entry is (display_name, runner_script,
# extra_args_template). The template may reference ${TASKS}, ${TRIAL},
# ${OUT_DIR}, ${SEED}, ${EMULATOR} which are substituted per invocation.
#
# The user is expected to customise these via ``--agent_config`` to match
# their endpoints / model names. The defaults below assume the conda env at
# python3 and the matrix-runner
# environment variables (CATBENCH_EMULATORS, CATBENCH_ADB_PORT, model
# server endpoints).
DEFAULT_AGENT_RECIPES: tuple[dict[str, str], ...] = (
    {
        "display": "M3A-Venus",
        "runner": "benchmark/run_m3a_venus.py",
        "extra_args": (
            "--tasks=${TASKS} --output_path=${OUT_DIR} --task_random_seed=${SEED}"
        ),
    },
    {
        "display": "GPT-5.1",
        "runner": "benchmark/run_openai_python_action.py",
        "extra_args": (
            "--tasks=${TASKS} --output_path=${OUT_DIR} --task_random_seed=${SEED}"
            " --model=${OPENAI_MODEL}"
        ),
    },
    {
        "display": "Mobile-Agent-v3",
        "runner": "benchmark/run_mobile_agent_v3.py",
        "extra_args": (
            "--tasks=${TASKS} --output_path=${OUT_DIR} --task_random_seed=${SEED}"
        ),
    },
)


def _substitute(template: str, substitutions: dict[str, str]) -> str:
  result = template
  for key, value in substitutions.items():
    result = result.replace("${" + key + "}", value)
  return result


def _safe_short_name(name: str) -> str:
  return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _load_agent_recipes(path: str | None) -> list[dict[str, str]]:
  if not path:
    return list(DEFAULT_AGENT_RECIPES)
  with Path(path).expanduser().open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  recipes = payload.get("agents") if isinstance(payload, dict) else payload
  if not isinstance(recipes, list):
    raise ValueError(
        f"--agent_config must contain a list of recipes, got {type(recipes).__name__}"
    )
  validated: list[dict[str, str]] = []
  for entry in recipes:
    if not isinstance(entry, dict):
      raise ValueError(f"Recipe entry must be a dict, got: {entry!r}")
    if not all(key in entry for key in ("display", "runner", "extra_args")):
      raise ValueError(
          f"Recipe must have keys display, runner, extra_args: {entry!r}"
      )
    validated.append(entry)
  return validated


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out_root", required=True)
  parser.add_argument(
      "--trials",
      type=int,
      default=3,
      help="Number of independent runs per (agent, task) pair (default 3).",
  )
  parser.add_argument(
      "--base_seed",
      type=int,
      default=30,
      help="Trial i uses task_random_seed = base_seed + i.",
  )
  parser.add_argument(
      "--agent",
      action="append",
      default=[],
      help=(
          "Display name(s) of agents to run (subset of the configured "
          "recipes). Repeat the flag. Empty = run all configured recipes."
      ),
  )
  parser.add_argument(
      "--agent_config",
      default="",
      help=(
          "Optional JSON file overriding the default agent recipes "
          "(list of {display, runner, extra_args}). Use this to point at "
          "your local model endpoints and conda envs."
      ),
  )
  parser.add_argument(
      "--python",
      default=os.environ.get("PYTHON_BIN", sys.executable),
      help="Python interpreter used to invoke each agent runner.",
  )
  parser.add_argument(
      "--env",
      action="append",
      default=[],
      help="Extra KEY=VALUE env vars to set per invocation. Repeat the flag.",
  )
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Print every planned command without launching.",
  )
  parser.add_argument(
      "--tasks",
      default=",".join(AW_CANONICAL_TASKS),
      help=(
          "Override the canonical task list. Default is the 6 AW-canonical "
          "Clock+Maps tasks; do not change unless you know what you are doing."
      ),
  )
  args = parser.parse_args()

  out_root = Path(args.out_root).expanduser().resolve()
  out_root.mkdir(parents=True, exist_ok=True)
  recipes = _load_agent_recipes(args.agent_config)
  if args.agent:
    wanted = set(args.agent)
    recipes = [r for r in recipes if r["display"] in wanted]
    if not recipes:
      print(
          f"error: --agent {args.agent} matched no configured recipe.",
          file=sys.stderr,
      )
      return 2

  extra_env: dict[str, str] = {}
  for raw in args.env:
    if "=" not in raw:
      print(f"warning: ignoring malformed --env {raw!r}", file=sys.stderr)
      continue
    key, value = raw.split("=", 1)
    extra_env[key.strip()] = value

  tasks_csv = args.tasks
  invocations: list[dict[str, Any]] = []
  for recipe in recipes:
    short = _safe_short_name(recipe["display"])
    runner_path = (REPO_ROOT / recipe["runner"]).resolve()
    if not runner_path.exists():
      print(
          f"warning: runner not found, skipping {recipe['display']}: {runner_path}",
          file=sys.stderr,
      )
      continue
    for trial in range(1, args.trials + 1):
      seed = args.base_seed + trial - 1
      trial_dir = out_root / short / f"trial_{trial:02d}"
      trial_dir.mkdir(parents=True, exist_ok=True)
      substitutions = {
          "TASKS": tasks_csv,
          "TRIAL": str(trial),
          "OUT_DIR": str(trial_dir),
          "SEED": str(seed),
          "EMULATOR": os.environ.get("CATBENCH_EMULATOR", "5554:8554"),
          "OPENAI_MODEL": os.environ.get("OPENAI_MODEL", "gpt-5.1"),
      }
      extra = _substitute(recipe["extra_args"], substitutions)
      command = [args.python, str(runner_path), *shlex.split(extra)]
      invocations.append(
          {
              "agent": recipe["display"],
              "trial": trial,
              "seed": seed,
              "out_dir": str(trial_dir),
              "tasks": tasks_csv.split(","),
              "command": command,
          }
      )

  index_path = out_root / "index.json"
  index_path.write_text(
      json.dumps(
          {
              "generated_at": dt.datetime.now().isoformat(),
              "trials_per_agent": args.trials,
              "tasks": tasks_csv.split(","),
              "invocations": invocations,
          },
          indent=2,
          ensure_ascii=False,
      )
      + "\n",
      encoding="utf-8",
  )
  print(f"Wrote trial index: {index_path}")
  print(f"Planned {len(invocations)} agent x trial invocations.")
  if args.dry_run:
    for inv in invocations:
      print("  " + " ".join(shlex.quote(part) for part in inv["command"]))
    return 0

  failures: list[dict[str, Any]] = []
  for inv in invocations:
    print("=" * 78, flush=True)
    print(
        f"[{inv['agent']} trial {inv['trial']}/{args.trials}] seed={inv['seed']}",
        flush=True,
    )
    env = os.environ.copy()
    env.update(extra_env)
    rc = subprocess.run(inv["command"], env=env, check=False).returncode
    if rc != 0:
      failures.append({**inv, "exit_code": rc})
      print(f"  exited with code {rc}", file=sys.stderr)

  status_path = out_root / "run_status.json"
  status_path.write_text(
      json.dumps(
          {
              "completed_at": dt.datetime.now().isoformat(),
              "total_invocations": len(invocations),
              "failed": len(failures),
              "failures": failures,
          },
          indent=2,
          ensure_ascii=False,
      )
      + "\n",
      encoding="utf-8",
  )
  print(f"Wrote run status: {status_path}")
  return 0 if not failures else 1


if __name__ == "__main__":
  raise SystemExit(main())
