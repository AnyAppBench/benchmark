#!/usr/bin/env python3
"""Run MobileRL-9B on AndroidWorld tasks using the author's prompt grammar."""

from __future__ import annotations

import os
import subprocess
import sys


DEFAULT_MODEL_NAME = "xuyifan/MobileRL-9B"
DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.expanduser(os.environ.get("CATBENCH_RUNS_DIR", "~/catbench_runs")),
    "mobilerl_9b",
)
DEFAULT_MAX_NEW_TOKENS = os.environ.get("MOBILERL_MAX_NEW_TOKENS", "512")
DEFAULT_MAX_STEPS = os.environ.get("MOBILERL_MAX_STEPS", "20")


def _has_flag(argv: list[str], flag_name: str) -> bool:
  flag = f"--{flag_name}"
  return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def main() -> int:
  args = list(sys.argv[1:])
  if not _has_flag(args, "model_name"):
    args.append(f"--model_name={DEFAULT_MODEL_NAME}")
  if not _has_flag(args, "prompt_style"):
    args.append("--prompt_style=mobilerl_point_think")
  if not _has_flag(args, "image_max_pixels"):
    args.append("--image_max_pixels=500000")
  if not _has_flag(args, "max_new_tokens"):
    args.append(f"--max_new_tokens={DEFAULT_MAX_NEW_TOKENS}")
  if not _has_flag(args, "max_steps"):
    args.append(f"--max_steps={DEFAULT_MAX_STEPS}")
  if not _has_flag(args, "output_path"):
    args.append(f"--output_path={DEFAULT_OUTPUT_PATH}")

  script_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "run_openai_python_action.py"
  )
  return subprocess.call([sys.executable, script_path, *args])


if __name__ == "__main__":
  raise SystemExit(main())
