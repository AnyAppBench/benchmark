"""Run cross-app generalization slices for AndroidWorld.

This script orchestrates experiments to test whether models can solve
comparable intents across similar apps. It uses mapped AndroidWorld task
templates where evaluators already exist and generates porting scaffolds for
unsupported apps using canonical task names.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict
from typing import Any

from app_generalization_profiles import AppProfile
from app_generalization_profiles import get_domain_profiles


DEFAULT_OUTPUT_ROOT = os.path.join(
    os.path.expanduser(os.environ.get("CATBENCH_RUNS_DIR", "~/catbench_runs")),
    "app_generalization",
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)


def _build_parser() -> argparse.ArgumentParser:
    domain_choices = ["all", *sorted(get_domain_profiles().keys()), "tasks"]
    parser = argparse.ArgumentParser(
        description=(
            "Run app-generalization study cohorts (notes/todo/clock) using an existing "
            "benchmark runner."
        )
    )
    parser.add_argument(
        "--runner_script",
        type=str,
        default="benchmark/run_maiui.py",
        help=(
            "Runner script to execute for each supported app cohort "
            "(for example benchmark/run_maiui.py)."
        ),
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=tuple(domain_choices),
        help="Which domain cohort to run.",
    )
    parser.add_argument(
        "--suite_family",
        type=str,
        default="android_world",
        help="Suite family forwarded to the runner.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where per-app run outputs and manifest are written.",
    )
    parser.add_argument(
        "--include_optional",
        action="store_true",
        help="Include optional harder variants in the manifest and run plan.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and write manifest without executing runs.",
    )
    parser.add_argument(
        "--print_profile",
        action="store_true",
        help="Print profile details before execution.",
    )
    parser.add_argument(
        "--write_scaffolds",
        action="store_true",
        help="Generate task-porting scaffold files for unsupported apps.",
    )
    parser.add_argument(
        "--scaffold_root",
        type=str,
        default="benchmark/android_world/task_evals",
        help="Root directory for generated scaffold files.",
    )
    parser.add_argument(
        "--shared_vllm",
        action="store_true",
        help=(
            "Launch a single vLLM server once and reuse it for every app run, "
            "instead of restarting it for each app (saves 5-10 min per app). "
            "Reads --model_name, --device, --endpoint_url from passthrough flags "
            "or uses their defaults."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Shared vLLM server helpers
# ---------------------------------------------------------------------------

def _extract_passthrough_flag(passthrough: list[str], name: str, default: str) -> str:
    """Return the value of --name=value from passthrough, or default."""
    for arg in passthrough:
        if arg.startswith(f"--{name}="):
            return arg.split("=", 1)[1]
    return default


def _launch_shared_vllm(
    model_name: str,
    device: str,
    endpoint_url: str,
    gpu_memory_utilization: float = 0.75,
) -> "subprocess.Popen[bytes]":
    """Start a single vLLM server to be shared across all app runs."""
    parsed = urllib.parse.urlparse(endpoint_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]

    env = os.environ.copy()
    if device.startswith("cuda:"):
        gpu_idx = device.split(":", 1)[1]
        env["CUDA_VISIBLE_DEVICES"] = gpu_idx
        print(f"[SHARED_VLLM] Pinning to GPU {gpu_idx} (CUDA_VISIBLE_DEVICES={gpu_idx})", flush=True)
    elif device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd += ["--device", "cpu"]

    print(f"[SHARED_VLLM] Launching: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, env=env)
    print(f"[SHARED_VLLM] PID={proc.pid} — waiting for model to load…", flush=True)
    return proc


def _wait_for_shared_vllm(endpoint_url: str, model_name: str, timeout_sec: int = 1800) -> None:
    """Poll until the shared vLLM endpoint is serving the expected model."""
    base = endpoint_url.rstrip("/")
    if base.endswith("/models"):
        models_url = base
    elif base.endswith("/v1"):
        models_url = f"{base}/models"
    else:
        models_url = f"{base}/v1/models"

    deadline = time.time() + timeout_sec
    attempt = 0
    print(f"[SHARED_VLLM] Waiting for {model_name!r} at {models_url} (timeout={timeout_sec}s)", flush=True)
    while time.time() < deadline:
        attempt += 1
        try:
            req = urllib.request.Request(models_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                payload = json.loads(resp.read().decode())
            ids = [m.get("id") for m in payload.get("data", []) if isinstance(m, dict)]
            if model_name in ids:
                print(f"[SHARED_VLLM] Endpoint ready: {model_name!r} is serving.", flush=True)
                return
        except Exception:
            pass
        print(f"[SHARED_VLLM] Not ready yet (attempt {attempt})…", flush=True)
        time.sleep(5.0)
    raise RuntimeError(
        f"[SHARED_VLLM] vLLM server did not become ready within {timeout_sec}s"
    )


def _resolve_runner_path(path: str) -> str:
    if os.path.isabs(path):
        return path

    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(_SCRIPT_DIR, path)),
        os.path.abspath(os.path.join(_REPO_ROOT, path)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _status_for_app(app: AppProfile, include_optional: bool) -> str:
    if app.optional and not include_optional:
        return "skipped_optional"
    if not app.implemented_tasks:
        return "unsupported_no_task_eval"
    return "supported"


def _run_command(command: list[str], dry_run: bool, label: str) -> int:
    rendered = _serialize_command(command)
    if dry_run:
        print(f"[DRY] {label}: {rendered}", flush=True)
        return 0
    print(f"[RUN] {label}: {rendered}", flush=True)
    return subprocess.call(command)


def _serialize_command(command: list[str]) -> str:
    return " ".join(command)


def _to_pascal(value: str) -> str:
    tokens = value.replace("-", "_").split("_")
    return "".join(token.capitalize() for token in tokens if token)


def _build_porting_targets(app: AppProfile, canonical_tasks: tuple[str, ...]) -> list[dict[str, str]]:
    app_pascal = _to_pascal(app.app_id)
    targets = []
    for source_task in canonical_tasks:
        targets.append(
            {
                "source_task": source_task,
                "target_task_name": f"{source_task}For{app_pascal}",
            }
        )
    return targets


def _render_scaffold_module(
    domain: str,
    app: AppProfile,
    canonical_tasks: tuple[str, ...],
    task_family: str,
) -> str:
    app_pascal = _to_pascal(app.app_id)
    storage_hint = (
        "sqlite_or_information_retrieval"
        if task_family == "information_retrieval"
        else "file_or_sqlite"
    )
    parts: list[str] = [
        '"""Autogenerated task-porting scaffold for app generalization.',
        "",
        "Generated by benchmark/run_app_generalization.py.",
        "Use docs/tasks_guide.md to implement each task fully.",
        '"""',
        "",
        "from typing import Any",
        "",
        "from android_world.env import interface",
        "from android_world.task_evals import task_eval",
        "",
        f"# Domain: {domain}",
        f"# Target app: {app.display_name}",
        f"# Storage hint from guide: {storage_hint}",
        "_APP_NAME = \"TODO_REPLACE_WITH_ANDROIDWORLD_APP_NAME\"",
        "",
    ]

    for source_task in canonical_tasks:
        class_name = f"{source_task}For{app_pascal}"
        parts.extend(
            [
                f"class {class_name}(task_eval.TaskEval):",
                f'  """Port {source_task} to {app.display_name} ({domain})."""',
                "",
                (
                    "  template = "
                    f'"TODO: port {source_task} behavior to {app.display_name}."'
                ),
                "  complexity = 1.0",
                "  schema = {'type': 'object', 'properties': {}, 'required': []}",
                "",
                "  def __init__(self, params: dict[str, Any]):",
                "    super().__init__(params)",
                "    # Replace with concrete validator-backed task implementation.",
                "    self.impl_task: task_eval.TaskEval | None = None",
                "",
                "  @property",
                "  def app_names(self) -> tuple[str, ...]:",
                "    return (_APP_NAME,)",
                "",
                "  @classmethod",
                "  def generate_random_params(cls) -> dict[str, Any]:",
                "    # Port random parameter generation from the source task.",
                "    return {'seed': 0}",
                "",
                "  def initialize_task(self, env: interface.AsyncEnv) -> None:",
                "    super().initialize_task(env)",
                "    if self.impl_task is None:",
                "      raise NotImplementedError(",
                "          'TODO: build validator-backed impl_task per docs/tasks_guide.md.'",
                "      )",
                "    self.impl_task.initialize_task(env)",
                "",
                "  def is_successful(self, env: interface.AsyncEnv) -> float:",
                "    super().is_successful(env)",
                "    if self.impl_task is None:",
                "      raise NotImplementedError(",
                "          'TODO: implement app-specific validator success logic.'",
                "      )",
                "    return self.impl_task.is_successful(env)",
                "",
                "  def tear_down(self, env: interface.AsyncEnv) -> None:",
                "    if self.impl_task is not None:",
                "      self.impl_task.tear_down(env)",
                "    super().tear_down(env)",
                "",
                "",
            ]
        )

    return "\n".join(parts)


def _write_scaffold_file(
    scaffold_root: str,
    domain: str,
    app: AppProfile,
    canonical_tasks: tuple[str, ...],
    task_family: str,
) -> str:
    scaffold_dir = _resolve_scaffold_dir(scaffold_root, task_family)
    os.makedirs(scaffold_dir, exist_ok=True)
    filename = f"{domain}_{app.app_id}_tasks.py"
    file_path = os.path.join(scaffold_dir, filename)
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(
            _render_scaffold_module(
                domain,
                app,
                canonical_tasks,
                task_family,
            )
        )
    return file_path


def _resolve_scaffold_dir(scaffold_root: str, task_family: str) -> str:
    if task_family == "information_retrieval":
        return os.path.join(
            scaffold_root,
            "information_retrieval",
            "app_generalization_generated",
        )
    return os.path.join(scaffold_root, "single", "app_generalization_generated")


def main() -> int:
    parser = _build_parser()
    args, passthrough = parser.parse_known_args()

    runner_script = _resolve_runner_path(args.runner_script)
    if not os.path.isfile(runner_script):
        print(f"Runner script not found: {runner_script}")
        return 2

    # --shared_vllm: launch the model server once and share it across all apps.
    # Each runner script receives --mode=endpoint so it skips launching its own.
    if args.shared_vllm:
        model_name   = _extract_passthrough_flag(passthrough, "model_name", "xuyifan/MobileRL-9B")
        device       = _extract_passthrough_flag(passthrough, "device", "cuda:0")
        endpoint_url = _extract_passthrough_flag(passthrough, "endpoint_url", "http://127.0.0.1:8000")
        gpu_mem_util = float(_extract_passthrough_flag(passthrough, "gpu_memory_utilization", "0.75"))

        _vllm_proc = _launch_shared_vllm(model_name, device, endpoint_url, gpu_memory_utilization=gpu_mem_util)
        atexit.register(
            lambda: _vllm_proc.terminate()
            if _vllm_proc and _vllm_proc.poll() is None
            else None
        )
        _wait_for_shared_vllm(endpoint_url, model_name)

        # Strip any existing --mode / --endpoint_url from passthrough, then
        # inject endpoint mode so runner scripts connect instead of re-launching.
        passthrough = [
            a for a in passthrough
            if not a.startswith("--mode=") and not a.startswith("--endpoint_url=")
        ]
        passthrough.append("--mode=endpoint")
        passthrough.append(f"--endpoint_url={endpoint_url}")

    all_profiles = get_domain_profiles()
    requested_domain = "todo" if args.domain == "tasks" else args.domain
    selected_domains = (
        list(all_profiles.keys())
        if requested_domain == "all"
        else [requested_domain]
    )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = os.path.join(args.output_root, f"manifest_{timestamp}")
    _ensure_dir(manifest_dir)

    manifest: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(),
        "runner_script": runner_script,
        "dry_run": args.dry_run,
        "include_optional": args.include_optional,
        "suite_family": args.suite_family,
        "passthrough_flags": passthrough,
        "domains": {},
    }

    if args.print_profile:
        print("Selected domains:", ", ".join(selected_domains))

    total_runs = 0
    total_skipped = 0
    total_skipped_optional = 0
    total_unsupported = 0
    total_failed = 0
    total_scaffolds = 0
    unsupported_details: list[str] = []
    optional_skipped_details: list[str] = []

    for domain in selected_domains:
        profile = all_profiles[domain]
        domain_entries: list[dict[str, Any]] = []

        if args.print_profile:
            print(f"\n[{domain}] intents: {', '.join(profile.intents)}")

        for app in profile.apps:
            status = _status_for_app(app, args.include_optional)
            app_out_dir = os.path.join(args.output_root, domain, app.app_id)
            porting_targets = _build_porting_targets(app, profile.canonical_tasks)

            entry: dict[str, Any] = {
                "app": asdict(app),
                "status": status,
                "canonical_tasks": list(profile.canonical_tasks),
                "implemented_tasks": list(app.implemented_tasks),
                "porting_targets": porting_targets,
                "output_dir": app_out_dir,
                "runner_command": None,
                "exit_code": None,
                "scaffold_file": None,
            }

            if status != "supported":
                total_skipped += 1
                if status == "skipped_optional":
                    total_skipped_optional += 1
                    optional_skipped_details.append(
                        f"{domain}/{app.app_id} ({app.display_name})"
                    )
                elif status == "unsupported_no_task_eval":
                    total_unsupported += 1
                    reason = app.notes or "No implemented task evaluators configured."
                    unsupported_details.append(
                        f"{domain}/{app.app_id} ({app.display_name}): {reason}"
                    )
                if args.write_scaffolds and status == "unsupported_no_task_eval":
                    scaffold_file = _write_scaffold_file(
                        args.scaffold_root,
                        domain,
                        app,
                        profile.canonical_tasks,
                        profile.task_family,
                    )
                    total_scaffolds += 1
                    entry["scaffold_file"] = scaffold_file
                domain_entries.append(entry)
                continue

            _ensure_dir(app_out_dir)
            tasks_value = ",".join(app.implemented_tasks)
            command = [
                sys.executable,
                runner_script,
                f"--suite_family={args.suite_family}",
                f"--tasks={tasks_value}",
                f"--output_path={app_out_dir}",
                *passthrough,
            ]

            entry["runner_command"] = _serialize_command(command)
            run_label = f"{domain}/{app.app_id}"
            exit_code = _run_command(command, args.dry_run, run_label)
            entry["exit_code"] = exit_code

            if exit_code == 0:
                total_runs += 1
                print(f"[OK ] {run_label}", flush=True)
            else:
                total_failed += 1
                entry["status"] = "failed"
                print(f"[ERR] {run_label} (exit={exit_code})", flush=True)

            domain_entries.append(entry)

        manifest["domains"][domain] = {
            "task_family": profile.task_family,
            "intents": profile.intents,
            "canonical_tasks": profile.canonical_tasks,
            "apps": domain_entries,
        }

    manifest_path = os.path.join(manifest_dir, "app_generalization_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nApp-generalization run complete")
    print(f"Manifest: {manifest_path}")
    print(f"Successful runs: {total_runs}")
    print(f"Failed runs: {total_failed}")
    print(f"Unsupported (no task eval): {total_unsupported}")
    print(f"Skipped optional: {total_skipped_optional}")
    print(f"Skipped/unsupported: {total_skipped}")
    print(f"Scaffold files generated: {total_scaffolds}")

    if unsupported_details:
        print("\nUnsupported app cohorts (installed app alone is not enough):")
        for line in unsupported_details:
            print(f"- {line}")

    if optional_skipped_details:
        print("\nOptional cohorts skipped (pass --include_optional to include):")
        for line in optional_skipped_details:
            print(f"- {line}")

    if total_failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
