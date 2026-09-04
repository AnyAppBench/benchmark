#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATALOG_FILE="${CATALOG_FILE:-${SCRIPT_DIR}/app_generalization_apps.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/anyappbench_apks}"
LOCAL_APK_DIR="${LOCAL_APK_DIR:-}"
INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-0}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
ADB="${ADB:-adb}"
ADB_SERIAL="${ADB_SERIAL:-}"
BOOTSTRAP_SCRIPT="${SCRIPT_DIR}/bootstrap_app_generalization_apps.py"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${BOOTSTRAP_SCRIPT}" ]]; then
  echo "Bootstrap script not found: ${BOOTSTRAP_SCRIPT}"
  exit 1
fi

args=(
  --catalog-file "${CATALOG_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --adb "${ADB}"
)

if [[ "${INCLUDE_OPTIONAL}" == "1" ]]; then
  args+=(--include-optional)
fi

if [[ "${FORCE_DOWNLOAD}" == "1" ]]; then
  args+=(--force-download)
fi

if [[ -n "${ADB_SERIAL}" ]]; then
  args+=(--serial "${ADB_SERIAL}")
fi

if [[ -n "${LOCAL_APK_DIR}" ]]; then
  args+=(--local-apk-dir "${LOCAL_APK_DIR}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi

python3 "${BOOTSTRAP_SCRIPT}" "${args[@]}"
