"""Evaluate UI-TARS on different versions of Markor app."""

import os
import sys
import time
import json
import glob
import subprocess
import shutil
from collections.abc import Sequence

user_tmp_dir = os.path.expanduser('~/tmp')
os.makedirs(user_tmp_dir, exist_ok=True)
os.environ['TMPDIR'] = user_tmp_dir

from absl import app, flags, logging

import pysqlite3
sys.modules['sqlite3'] = sys.modules['pysqlite3']

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmark.agent_import_utils import import_optional_agent

AndroidWorldUITARSAgent = import_optional_agent(
    project_root,
    'agents.uitars.adapters.android_world',
    'AndroidWorldUITARSAgent',
)
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.env import env_launcher

logging.set_verbosity(logging.INFO)

_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    os.path.expanduser('~/androidsdk/platform-tools/adb'),
    'Path to adb.',
)
_CONSOLE_PORT = flags.DEFINE_integer('console_port', 5554, 'Console port.')
_APK_DIR = flags.DEFINE_string(
    'apk_dir',
    os.path.join(script_dir, 'my_apks', 'markor'),
    'Directory containing APK versions.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    os.path.join(
        os.path.expanduser(os.environ.get('CATBENCH_RUNS_DIR', '~/catbench_runs')),
        'markor_eval_latest',
    ),
    'Output directory.',
)
_MODEL_NAME = flags.DEFINE_string('model_name', 'ByteDance-Seed/UI-TARS-1.5-7B', 'Model name.')
_MODE = flags.DEFINE_string('mode', 'endpoint', 'Mode: local or endpoint.')
_ENDPOINT_URL = flags.DEFINE_string('endpoint_url', 'http://127.0.0.1:8001', 'Endpoint URL.')
_MAX_STEPS = flags.DEFINE_integer('max_steps', 30, 'Max steps.')

MARKOR_PACKAGE = "net.gsantner.markor"
MARKOR_TASKS = [
    "MarkorAddNoteHeader",
    "MarkorChangeNoteContent",
    "MarkorCreateFolder",
    "MarkorCreateNote",
    "MarkorCreateNoteAndSms",
    "MarkorCreateNoteFromClipboard",
    "MarkorDeleteAllNotes",
    "MarkorDeleteNewestNote",
    "MarkorDeleteNote",
    "MarkorEditNote",
    "MarkorMergeNotes",
    "MarkorMoveNote",
    "MarkorTranscribeReceipt",
    "MarkorTranscribeVideo",
]


def install_apk(apk_path, adb_path, console_port):
    print(f"Installing {apk_path}...")
    cmd_prefix = [adb_path, '-s', f'emulator-{console_port}']
    
    subprocess.run(cmd_prefix + ['uninstall', MARKOR_PACKAGE], capture_output=True)
    
    result = subprocess.run(cmd_prefix + ['install', '-r', apk_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to install {apk_path}: {result.stderr}")
        return False
    
    print(f"Successfully installed {apk_path}")
    
    # Launch app once to trigger tutorial
    print(f"Launching {MARKOR_PACKAGE} for first-time setup...")
    subprocess.run(cmd_prefix + ['shell', 'monkey', '-p', MARKOR_PACKAGE, '-c', 'android.intent.category.LAUNCHER', '1'], capture_output=True)
    time.sleep(3.0)
    
    # Dismiss tutorial screens using exact emulator coordinates
    print(f"Dismissing Markor tutorial screens...")
    # The arrow button is at x:978.9, y:2249.8
    for i in range(5):
        print(f"  Dismissing tutorial screen {i+1}...")
        subprocess.run(cmd_prefix + ['shell', 'input', 'tap', '979', '2250'], capture_output=True)
        time.sleep(1.5)
    
    # Go back to home screen (don't force-stop, let tutorial state persist)
    print(f"Returning to home screen...")
    subprocess.run(cmd_prefix + ['shell', 'input', 'keyevent', 'KEYCODE_HOME'], capture_output=True)
    time.sleep(1.0)
    
    print(f"Setup complete for {apk_path}")
    return True


def run_all_tasks_for_version(version_name, adb_path, console_port):
    print(f"Running all Markor tasks on {version_name}...")
    
    env = env_launcher.load_and_setup_env(
        console_port=console_port,
        emulator_setup=False,
        adb_path=adb_path,
    )
    
    task_registry = registry.TaskRegistry()
    all_tasks = task_registry.get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)
    
    markor_task_classes = {}
    for task_name in MARKOR_TASKS:
        task_class = all_tasks.get(task_name)
        if task_class:
            markor_task_classes[task_name] = task_class
        else:
            print(f"Warning: Task {task_name} not found in registry")
    
    if not markor_task_classes:
        print("No valid Markor tasks found")
        env.close()
        return None
    
    suite = suite_utils.create_suite(
        markor_task_classes,
        n_task_combinations=1,
        seed=42,
    )
    
    run_dir = os.path.join(_OUTPUT_PATH.value, version_name)
    
    # Clean up any previous run directory to avoid checkpoint conflicts
    if os.path.exists(run_dir):
        print(f"Removing previous run directory: {run_dir}")
        shutil.rmtree(run_dir)
    
    checkpointer = checkpointer_lib.IncrementalCheckpointer(checkpointer_lib.create_run_directory(run_dir))
    
    agent = AndroidWorldUITARSAgent(
        env,
        model_name=_MODEL_NAME.value,
        device='cuda:0',
        save_failed_tasks=True,
        failed_tasks_dir=run_dir,
        mode=_MODE.value,
        endpoint_url=_ENDPOINT_URL.value,
    )
    agent.name = 'uitars'
    
    # CRITICAL: Set max_steps explicitly - this overrides task complexity-based budgets
    if _MAX_STEPS.value:
        agent.max_steps = _MAX_STEPS.value
        print(f"[CONFIG] Set agent.max_steps = {agent.max_steps}")
    else:
        print(f"[CONFIG] Using default task complexity-based step budgets")
    
    suite_utils.run(suite, agent, checkpointer=checkpointer, demo_mode=False)
    
    env.close()
    return run_dir


def _main() -> None:
    apk_files = []
    
    # Only run baseline version
    baseline_apk = os.path.join(
        os.path.expanduser(os.environ.get('CATBENCH_APK_DATA_DIR', '~/tmp/android_world/app_data')),
        'net.gsantner.markor_146.apk',
    )
    
    if os.path.exists(baseline_apk):
        apk_files.append(("v2.10.9_baseline", baseline_apk))
    else:
        print(f"Error: Baseline APK not found at {baseline_apk}")
        return
    
    print(f"Testing baseline version: v2.10.9")

    results_summary = {}

    for version_name, apk_path in apk_files:
        print(f"\n{'='*60}")
        print(f"Testing version: {version_name}")
        print(f"{'='*60}\n")
        
        success = install_apk(apk_path, _ADB_PATH.value, _CONSOLE_PORT.value)
        if not success:
            results_summary[version_name] = {"status": "failed", "error": "APK installation failed"}
            continue
        
        try:
            result_dir = run_all_tasks_for_version(version_name, _ADB_PATH.value, _CONSOLE_PORT.value)
            results_summary[version_name] = {"status": "completed", "log_dir": result_dir}
        except Exception as e:
            print(f"Error running tasks for {version_name}: {e}")
            results_summary[version_name] = {"status": "failed", "error": str(e)}

    with open(os.path.join(_OUTPUT_PATH.value, "summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    print("All evaluations completed.")


def main(argv: Sequence[str]) -> None:
    del argv
    _main()


if __name__ == '__main__':
    app.run(main)
