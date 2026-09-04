#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${CONDA_ENV:-catbench311}"
AUTO_REPAIR="${AUTO_REPAIR:-0}"
ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-}"

if [[ -n "$ANDROID_SDK_ROOT" ]] && [[ ! -x "$ANDROID_SDK_ROOT/emulator/emulator" || ! -x "$ANDROID_SDK_ROOT/platform-tools/adb" ]]; then
  echo "WARN: ANDROID_SDK_ROOT is set but invalid: $ANDROID_SDK_ROOT"
  ANDROID_SDK_ROOT=""
fi

if [[ -z "$ANDROID_SDK_ROOT" ]]; then
  for candidate in \
    "$HOME/android-sdk" \
    "$HOME/Android/Sdk"; do
    if [[ -x "$candidate/emulator/emulator" && -x "$candidate/platform-tools/adb" ]]; then
      ANDROID_SDK_ROOT="$candidate"
      break
    fi
  done
fi

if [[ -z "$ANDROID_SDK_ROOT" ]]; then
  echo "ERROR: Could not detect a valid Android SDK root."
  echo "Set ANDROID_SDK_ROOT to a directory containing emulator/emulator and platform-tools/adb"
  exit 2
fi

export ANDROID_SDK_ROOT
export ANDROID_HOME="$ANDROID_SDK_ROOT"
export PATH="$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/benchmark:${PYTHONPATH:-}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not found."
  exit 2
fi

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "ERROR: conda environment '$CONDA_ENV' does not exist."
  exit 2
fi

if ! command -v adb >/dev/null 2>&1; then
  echo "ERROR: adb is not available on PATH."
  exit 2
fi

if ! command -v emulator >/dev/null 2>&1; then
  echo "ERROR: emulator is not available on PATH."
  echo "Expected in: $ANDROID_SDK_ROOT/emulator"
  exit 2
fi

if [[ "$AUTO_REPAIR" == "1" ]]; then
  echo "AUTO_REPAIR=1 detected. Repairing MAI-UI dependency stack in '$CONDA_ENV'..."
  conda run -n "$CONDA_ENV" python -m pip install --upgrade --force-reinstall \
    "torch==2.10.0" \
    "torchaudio==2.10.0" \
    "torchvision==0.25.0" \
    "transformers==4.57.6" \
    "huggingface_hub==0.36.2" \
    "tokenizers==0.22.2" \
    "protobuf==5.29.6" \
    "cuda-bindings==12.9.4" \
    "opencv-python-headless==4.13.0.92" \
    "numpy==2.2.6" \
    "pandas==2.3.0"
fi

if ! conda run -n "$CONDA_ENV" python -m pip check >/dev/null; then
  echo "ERROR: Python dependency conflicts detected in '$CONDA_ENV'."
  echo "Run again with AUTO_REPAIR=1 to apply known-good MAI-UI pins."
  conda run -n "$CONDA_ENV" python -m pip check || true
  exit 2
fi

if ! conda run -n "$CONDA_ENV" python -c "from packaging.version import Version; import cv2, torch, transformers, vllm, google.protobuf as protobuf; torch_v = Version(torch.__version__.split('+')[0]); tf_v = Version(transformers.__version__); vllm_v = Version(vllm.__version__); pb_v = Version(protobuf.__version__); cv_v = Version(cv2.__version__); assert torch_v.release[:2] == (2, 10), f'torch must be 2.10.x, got {torch.__version__}'; assert Version('4.56.0') <= tf_v < Version('5.0.0'), f'transformers must be >=4.56,<5, got {transformers.__version__}'; assert vllm_v.release[:2] == (0, 19), f'vllm must be 0.19.x, got {vllm.__version__}'; assert pb_v >= Version('5.29.6'), f'protobuf must be >=5.29.6, got {protobuf.__version__}'; assert cv_v >= Version('4.13.0'), f'opencv-python-headless must be >=4.13.0, got {cv2.__version__}'"
then
  echo "ERROR: MAI-UI core version checks failed in '$CONDA_ENV'."
  echo "Run with AUTO_REPAIR=1 or inspect package versions manually."
  exit 2
fi

conda run -n "$CONDA_ENV" python -c "import android_env, dm_env, pysqlite3" >/dev/null

agent_check_output="$({
  PROJECT_ROOT_ENV="$PROJECT_ROOT" conda run -n "$CONDA_ENV" python -c "import os, sys; project_root = os.environ['PROJECT_ROOT_ENV']; benchmark_root = os.path.join(project_root, 'benchmark'); [sys.path.insert(0, p) for p in (project_root, benchmark_root) if p not in sys.path];
try:
  from benchmark.agent_import_utils import resolve_uitars_agent_class
except ModuleNotFoundError:
  from agent_import_utils import resolve_uitars_agent_class
try:
  _, source = resolve_uitars_agent_class(project_root)
  print(f'AGENT_OK:{source}')
except ModuleNotFoundError as exc:
  print(exc)";
} 2>&1)"

agent_ok_line="$(grep '^AGENT_OK:' <<<"$agent_check_output" || true)"
if [[ -z "$agent_ok_line" ]]; then
  if [[ -n "$agent_check_output" ]]; then
    echo "$agent_check_output"
  fi
  echo "ERROR: Could not resolve a MAI-UI/UI-TARS agent backend."
  echo "Expected one of:"
  echo "  1) external module agents.uitars.adapters.android_world (set CATBENCH_AGENT_ROOT), or"
  echo "  2) local in-repo backend benchmark/android_world/agents/qwen_vlm.py"
  exit 2
fi

echo "Agent backend detected: ${agent_ok_line#AGENT_OK:}"

echo "Environment looks ready."
echo "CONDA_ENV=$CONDA_ENV"
echo "ANDROID_SDK_ROOT=$ANDROID_SDK_ROOT"
echo "PYTHONPATH=$PYTHONPATH"

echo ""
echo "Run from repository root:"
echo "cd \"$PROJECT_ROOT\""
echo "conda run -n $CONDA_ENV python benchmark/run_app_generalization.py --runner_script benchmark/run_maiui.py --maiui_variant=2b --domain all --include_optional --write_scaffolds --device=cuda:0 --suite_family=android_world"
echo ""
echo "Run from benchmark directory:"
echo "cd \"$PROJECT_ROOT/benchmark\""
echo "conda run -n $CONDA_ENV python run_app_generalization.py --runner_script run_maiui.py --maiui_variant=2b --domain all --include_optional --write_scaffolds --device=cuda:0 --suite_family=android_world"
