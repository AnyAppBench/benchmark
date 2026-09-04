#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"
INCLUDE_OPTIONAL_APPS="${INCLUDE_OPTIONAL_APPS:-0}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
ADB="${ADB:-adb}"
ADB_SERIAL="${ADB_SERIAL:-}"

if [[ "${SKIP_BOOTSTRAP}" != "1" ]]; then
  echo "[1/2] Bootstrapping app-generalization APKs..."
  INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL_APPS}" \
  FORCE_DOWNLOAD="${FORCE_DOWNLOAD}" \
  ADB="${ADB}" \
  ADB_SERIAL="${ADB_SERIAL}" \
  bash "${SCRIPT_DIR}/bootstrap_app_generalization_apks.sh"
else
  echo "[1/2] SKIP_BOOTSTRAP=1, skipping APK bootstrap."
fi

echo "[2/2] Running app-generalization benchmark..."
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_app_generalization.py" "$@"
