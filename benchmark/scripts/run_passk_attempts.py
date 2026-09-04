#!/usr/bin/env python3
"""Run the CATBench matrix k times to support pass@k / FRR evaluation.

This is a thin wrapper around `run_catbench_5cat_matrix.py`. It invokes the
matrix runner k times with k distinct (seed, output_root) pairs and writes a
top-level `passk_index.json` that links the k attempt manifests.

What it does NOT do:
  - It does not persist agent long-term memory across attempts. MemGUI's FRR
    metric only makes sense for agents with cross-session memory (Agent-S2,
    AgentProg with plan-cache, etc.). For memory-less agents, pass@k is
    informative as "would another seed have succeeded?" only.
  - It does not snapshot-reset the emulator AVD; it relies on the matrix
    runner's existing per-task reset behavior.

Usage:
  python3 run_passk_attempts.py \
    --base_runner benchmark/scripts/run_catbench_5cat_matrix.py \
    --output_root $HOME/anyappbench_runs/passk \
    --k 3 \
    --base_seed 30 \
    -- \
    --run_id passk_smoke --models gpt-5.1 ...

Arguments after `--` are forwarded verbatim to the base runner, except that
`--run_id`, `--task_random_seed`, and `--output_root` are overridden per
attempt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


def _strip_overridden_args(forwarded: list[str], overrides: tuple[str, ...]) -> list[str]:
  """Remove any forwarded flags we override per attempt (run_id, seed, output_root)."""
  cleaned: list[str] = []
  skip_next = False
  for arg in forwarded:
    if skip_next:
      skip_next = False
      continue
    matched = False
    for override in overrides:
      if arg == override:
        matched = True
        skip_next = True
        break
      if arg.startswith(f"{override}="):
        matched = True
        break
    if not matched:
      cleaned.append(arg)
  return cleaned


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--base_runner",
      default="benchmark/scripts/run_catbench_5cat_matrix.py",
      help="Path to the underlying matrix runner.",
  )
  parser.add_argument("--output_root", required=True)
  parser.add_argument(
      "--k", type=int, default=3, help="Number of attempts per task (default 3)."
  )
  parser.add_argument(
      "--base_seed",
      type=int,
      default=30,
      help="Seed for attempt 1. Attempt i uses base_seed + i - 1.",
  )
  parser.add_argument(
      "--run_id_prefix",
      default="",
      help="Prefix for the per-attempt run_id. Defaults to a timestamp.",
  )
  parser.add_argument(
      "--python",
      default=sys.executable,
      help="Python interpreter used to invoke the base runner.",
  )
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Print the planned attempt invocations without launching them.",
  )
  parser.add_argument(
      "forwarded",
      nargs=argparse.REMAINDER,
      help="Args forwarded to the base runner after `--`.",
  )
  args = parser.parse_args()

  output_root = Path(args.output_root).expanduser().resolve()
  output_root.mkdir(parents=True, exist_ok=True)
  prefix = args.run_id_prefix or dt.datetime.now().strftime("passk_%Y%m%d_%H%M%S")

  forwarded = list(args.forwarded)
  if forwarded and forwarded[0] == "--":
    forwarded = forwarded[1:]
  forwarded = _strip_overridden_args(
      forwarded, ("--run_id", "--task_random_seed", "--output_root")
  )

  attempt_records: list[dict[str, object]] = []
  for attempt_idx in range(1, args.k + 1):
    run_id = f"{prefix}_attempt{attempt_idx}"
    seed = args.base_seed + (attempt_idx - 1)
    attempt_dir = output_root / run_id
    command = [
        args.python,
        args.base_runner,
        f"--run_id={run_id}",
        f"--output_root={output_root}",
        f"--task_random_seed={seed}",
        *forwarded,
    ]
    attempt_records.append(
        {
            "attempt": attempt_idx,
            "run_id": run_id,
            "seed": seed,
            "manifest": str(attempt_dir / "catbench_5cat_manifest.json"),
            "command": command,
        }
    )
    print(f"[attempt {attempt_idx}/{args.k}] {' '.join(command)}", flush=True)
    if args.dry_run:
      continue
    proc = subprocess.run(command, check=False)
    if proc.returncode != 0:
      print(
          f"warning: attempt {attempt_idx} exited with code {proc.returncode}",
          file=sys.stderr,
      )

  index_path = output_root / "passk_index.json"
  index_path.write_text(
      json.dumps(
          {
              "k": args.k,
              "base_seed": args.base_seed,
              "output_root": str(output_root),
              "attempts": attempt_records,
              "generated_at": dt.datetime.now().isoformat(),
          },
          indent=2,
          ensure_ascii=False,
      )
      + "\n",
      encoding="utf-8",
  )
  print(f"Wrote pass@k index: {index_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
