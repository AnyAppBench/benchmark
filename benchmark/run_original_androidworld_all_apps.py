"""Run original AndroidWorld suite (all apps/tasks) without editing run.py.

This wrapper delegates execution to benchmark/run.py while enforcing full-suite
defaults:
- suite_family=android_world
- tasks unset (run all tasks across apps)
- agent_name=qwen_vlm
- qwen_model_id=namhokaist/SFT_True_Final
"""

from __future__ import annotations

import os
import subprocess
import sys


DEFAULT_AGENT_NAME = "qwen_vlm"
DEFAULT_SUITE_FAMILY = "android_world"
DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.expanduser(os.environ.get("CATBENCH_RUNS_DIR", "~/catbench_runs")),
    "original_androidworld_all_apps",
)
DEFAULT_QWEN_MODEL_ID = "namhokaist/SFT_True_Final"


def _has_flag(argv: list[str], flag_name: str) -> bool:
    flag = f"--{flag_name}"
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def main() -> int:
    args = list(sys.argv[1:])

    if _has_flag(args, "tasks"):
        raise SystemExit(
            "This runner is for full AndroidWorld evaluation across all apps. "
            "Remove --tasks to run the complete suite."
        )

    if not _has_flag(args, "agent_name"):
        args.append(f"--agent_name={DEFAULT_AGENT_NAME}")

    if not _has_flag(args, "suite_family"):
        args.append(f"--suite_family={DEFAULT_SUITE_FAMILY}")

    if not _has_flag(args, "output_path"):
        args.append(f"--output_path={DEFAULT_OUTPUT_PATH}")

    if not _has_flag(args, "n_task_combinations"):
        args.append("--n_task_combinations=1")

    if not _has_flag(args, "qwen_model_id"):
        args.append(f"--qwen_model_id={DEFAULT_QWEN_MODEL_ID}")

    if not _has_flag(args, "qwen_allow_infeasible"):
        args.append("--qwen_allow_infeasible=false")

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
    cmd = [sys.executable, script_path, *args]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())