#!/usr/bin/env bash
# isolate_category.sh -- disable every app in a CATBench category except a
# target package, by reading benchmark/app_generalization_apps.csv.
#
# Usage:
#   scripts/isolate_category.sh clock com.vicolo.chrono
#       -> disables every other clock app for user 0, leaves Chrono enabled.
#
#   scripts/isolate_category.sh clock --restore
#       -> re-enables every clock app for user 0.
#
# The category is the prefix of app_id in the CSV (e.g. "clock_chrono"
# -> "clock"). Run from the benchmark/ directory or a parent.
#
# Mirrors the automatic isolation done by
# android_world/task_evals/single/app_generalization_generated/_cross_app_base.py
# Use this script for ad-hoc manual testing outside the benchmark runner.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <category> <target_package | --restore>" >&2
  exit 64
fi

CATEGORY="$1"
TARGET="$2"

# Locate the CSV: try CWD, then benchmark/, then parent of this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV=""
for candidate in \
    "./app_generalization_apps.csv" \
    "./benchmark/app_generalization_apps.csv" \
    "${SCRIPT_DIR}/../app_generalization_apps.csv"; do
  if [[ -f "$candidate" ]]; then
    CSV="$(readlink -f "$candidate")"
    break
  fi
done

if [[ -z "$CSV" ]]; then
  echo "error: app_generalization_apps.csv not found" >&2
  exit 1
fi

# Pull every package whose app_id starts with "<category>_".
mapfile -t PKGS < <(
  awk -F',' -v cat="${CATEGORY}_" '
    NR == 1 { next }
    $1 ~ "^"cat { print $3 }
  ' "$CSV"
)

if [[ ${#PKGS[@]} -eq 0 ]]; then
  echo "error: no packages found for category '${CATEGORY}' in $CSV" >&2
  exit 1
fi

ADB_BIN="${ADB_BIN:-${ADB:-adb}}"
ADB_CMD=("${ADB_BIN}")
if [[ -n "${ADB_SERVER_PORT:-}" ]]; then
  ADB_CMD+=("-P" "${ADB_SERVER_PORT}")
fi
ADB_SERIAL="${ADB_SERIAL:-${ANDROID_SERIAL:-}}"
if [[ -n "${ADB_SERIAL}" ]]; then
  ADB_CMD+=("-s" "${ADB_SERIAL}")
fi
ADB_COMMAND_TIMEOUT_SEC="${ADB_COMMAND_TIMEOUT_SEC:-45}"

adb_call() {
  if command -v timeout >/dev/null 2>&1; then
    timeout --kill-after=5s "${ADB_COMMAND_TIMEOUT_SEC}s" "${ADB_CMD[@]}" "$@"
  else
    "${ADB_CMD[@]}" "$@"
  fi
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|required|strict) return 0 ;;
    *) return 1 ;;
  esac
}

contains_line() {
  local needle="$1"
  grep -Fxq "$needle"
}

contains_value() {
  local needle="$1"
  shift
  local value
  for value in "$@"; do
    if [[ "$value" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "$TARGET" == "--restore" ]]; then
  echo "Re-enabling ${#PKGS[@]} package(s) in category '${CATEGORY}':"
  for pkg in "${PKGS[@]}"; do
    echo "  pm enable --user 0 $pkg"
    adb_call shell pm enable --user 0 "$pkg" >/dev/null 2>&1 || true
  done
  exit 0
fi

# Sanity-check: target must be in the category.
if ! contains_value "$TARGET" "${PKGS[@]}"; then
  echo "error: '$TARGET' is not a registered package in category '${CATEGORY}'." >&2
  echo "Known packages:" >&2
  printf '  %s\n' "${PKGS[@]}" >&2
  exit 1
fi

# Heal any prior crash that left siblings disabled, then disable everything
# except the target.
echo "Isolating '${TARGET}' in category '${CATEGORY}':"
for pkg in "${PKGS[@]}"; do
  if [[ "$pkg" == "$TARGET" ]]; then
    echo "  pm enable --user 0 $pkg   (target)"
    adb_call shell pm enable --user 0 "$pkg" >/dev/null 2>&1 || true
  else
    echo "  pm disable-user --user 0 $pkg"
    adb_call shell pm disable-user --user 0 "$pkg" >/dev/null 2>&1 || true
  fi
done

if is_true "${CATBENCH_STRICT_CATEGORY_ISOLATION:-}"; then
  mapfile -t INSTALLED < <(
    adb_call shell pm list packages 2>/dev/null \
      | sed 's/\r$//' \
      | sed 's/^package://'
  )
  mapfile -t DISABLED < <(
    adb_call shell pm list packages -d 2>/dev/null \
      | sed 's/\r$//' \
      | sed 's/^package://'
  )

  errors=0
  if contains_value "$TARGET" "${DISABLED[@]}"; then
    echo "error: target package is disabled after isolation: $TARGET" >&2
    errors=1
  fi

  for pkg in "${PKGS[@]}"; do
    [[ "$pkg" == "$TARGET" ]] && continue
    if ! contains_value "$pkg" "${INSTALLED[@]}"; then
      continue
    fi
    if ! contains_value "$pkg" "${DISABLED[@]}"; then
      echo "error: sibling package is still enabled after isolation: $pkg" >&2
      errors=1
    fi
  done

  if [[ "${errors}" -ne 0 ]]; then
    exit 1
  fi
fi
