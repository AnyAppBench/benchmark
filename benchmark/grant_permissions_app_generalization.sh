#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATALOG_FILE="${CATALOG_FILE:-${SCRIPT_DIR}/app_generalization_apps.csv}"
INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-0}"
ADB="${ADB:-adb}"
BOOTSTRAP_SCRIPT="${SCRIPT_DIR}/bootstrap_app_generalization_apps.py"
ADB_SERIAL="${ADB_SERIAL:-}"

if [[ ! -f "${BOOTSTRAP_SCRIPT}" ]]; then
  echo "Bootstrap script not found: ${BOOTSTRAP_SCRIPT}"
  exit 1
fi

args=(
  --catalog-file "${CATALOG_FILE}"
  --adb "${ADB}"
  --skip-download
  --skip-install
)

if [[ "${INCLUDE_OPTIONAL}" == "1" ]]; then
  args+=(--include-optional)
fi

if [[ -n "${ADB_SERIAL}" ]]; then
  args+=(--serial "${ADB_SERIAL}")
fi

python3 "${BOOTSTRAP_SCRIPT}" "${args[@]}"
