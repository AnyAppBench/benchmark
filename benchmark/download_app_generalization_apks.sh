#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATALOG_FILE="${CATALOG_FILE:-${SCRIPT_DIR}/app_generalization_apps.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/anyappbench_apks}"
INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-0}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
BOOTSTRAP_SCRIPT="${SCRIPT_DIR}/bootstrap_app_generalization_apps.py"

if [[ ! -f "${BOOTSTRAP_SCRIPT}" ]]; then
  echo "Bootstrap script not found: ${BOOTSTRAP_SCRIPT}"
  exit 1
fi

args=(
  --catalog-file "${CATALOG_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --skip-install
  --skip-grant
)

if [[ "${INCLUDE_OPTIONAL}" == "1" ]]; then
  args+=(--include-optional)
fi

if [[ "${FORCE_DOWNLOAD}" == "1" ]]; then
  args+=(--force-download)
fi

python3 "${BOOTSTRAP_SCRIPT}" "${args[@]}"
