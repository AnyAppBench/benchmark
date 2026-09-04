"""Run MAI-UI models on Android World tasks.

This is a thin wrapper around run_uitars.py with MAI-UI defaults.

Examples:
  python benchmark/run_maiui.py --maiui_variant=8b --suite_family=android_world
  python benchmark/run_maiui.py --maiui_variant=2b --suite_family=android_world
"""

import os
import subprocess
import sys


MODEL_BY_VARIANT = {
    "8b": "Tongyi-MAI/MAI-UI-8B",
    "2b": "Tongyi-MAI/MAI-UI-2B",
}


def _default_runs_root() -> str:
    explicit = os.environ.get("CATBENCH_RUNS_DIR")
    if explicit:
        return os.path.expanduser(explicit)

    user_name = os.environ.get("USER", "")
    candidates = []
    if user_name:
        candidates.append(os.path.join("$HOME", user_name, "android_world_runs"))
    candidates.append(os.path.expanduser("~/catbench_runs"))

    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except OSError:
            continue

    return os.path.expanduser("~/catbench_runs")


DEFAULT_RUNS_ROOT = _default_runs_root()

OUTPUT_BY_VARIANT = {
    "8b": os.path.join(
        DEFAULT_RUNS_ROOT,
        "maiui_8b",
    ),
    "2b": os.path.join(
        DEFAULT_RUNS_ROOT,
        "maiui_2b",
    ),
}


def _has_flag(argv: list[str], flag_name: str) -> bool:
    flag = f"--{flag_name}"
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _extract_flag(argv: list[str], flag_name: str, default: str) -> tuple[str, list[str]]:
    """Extracts --flag=value or --flag value from argv and returns the remainder."""
    long_flag = f"--{flag_name}"
    value = default
    remaining: list[str] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == long_flag:
            if i + 1 < len(argv):
                value = argv[i + 1]
                i += 2
                continue
            i += 1
            continue
        if arg.startswith(f"{long_flag}="):
            value = arg.split("=", 1)[1]
            i += 1
            continue

        remaining.append(arg)
        i += 1

    return value, remaining


def _check_agent_dependency() -> tuple[bool, str]:
    """Verifies a MAI-UI-compatible backend is importable."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    for path in (project_root, script_dir):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        from benchmark.agent_import_utils import resolve_uitars_agent_class
    except ModuleNotFoundError:
        from agent_import_utils import resolve_uitars_agent_class

    try:
        _, backend_source = resolve_uitars_agent_class(project_root)
    except ModuleNotFoundError as exc:
        message = (
            "Could not resolve a MAI-UI agent backend.\n"
            f"{exc}\n\n"
            "Expected one of:\n"
            "1. External adapter module agents/uitars/adapters/android_world.py, or\n"
            "2. Local in-repo agent backend under benchmark/android_world/agents."
        )
        return False, message

    return True, backend_source


def main() -> int:
    args = list(sys.argv[1:])

    backend_ok, backend_message = _check_agent_dependency()
    if not backend_ok:
        print(backend_message)
        return 2

    print(f"Using MAI-UI backend: {backend_message}")

    variant, args = _extract_flag(args, "maiui_variant", "8b")
    variant = variant.lower()

    if variant not in MODEL_BY_VARIANT:
        print(
            "Invalid --maiui_variant. Use one of: "
            + ", ".join(sorted(MODEL_BY_VARIANT.keys()))
        )
        return 2

    if not _has_flag(args, "model_name"):
        args.append(f"--model_name={MODEL_BY_VARIANT[variant]}")

    if not _has_flag(args, "output_path"):
        args.append(f"--output_path={OUTPUT_BY_VARIANT[variant]}")

    if not _has_flag(args, "endpoint_min_context_len"):
        args.append("--endpoint_min_context_len=16384")

    if not _has_flag(args, "endpoint_require_loopback"):
        args.append("--endpoint_require_loopback=true")

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_uitars.py")
    cmd = [sys.executable, script_path, *args]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
