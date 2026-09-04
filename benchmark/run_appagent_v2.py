#!/usr/bin/env python3
"""Best-effort AppAgent-v2-compatible runner.

The official AppAgent v2 AndroidWorld adapter is not included in this checkout.
This runner provides a runnable screenshot/action baseline with an AppAgent-like
single-action loop so result pipelines can be exercised. Keep the config row
disabled unless you intentionally want this compatibility fallback.
"""

from __future__ import annotations

import os
import subprocess
import sys


DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.expanduser(os.environ.get("CATBENCH_RUNS_DIR", "~/catbench_runs")),
    "appagent_v2_lite",
)


def _has_flag(argv: list[str], flag_name: str) -> bool:
  flag = f"--{flag_name}"
  return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def main() -> int:
  args = list(sys.argv[1:])
  if not _has_flag(args, "model_name"):
    args.append("--model_name=appagent-v2")
  if not _has_flag(args, "prompt_style"):
    args.append("--prompt_style=appagent_v2_lite")
  if not _has_flag(args, "output_path"):
    args.append(f"--output_path={DEFAULT_OUTPUT_PATH}")
  script_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "run_openai_python_action.py"
  )
  return subprocess.call([sys.executable, script_path, *args])


if __name__ == "__main__":
  raise SystemExit(main())
