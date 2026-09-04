#!/usr/bin/env bash
# CATBench resource-contract wrapper for the legacy emulator launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="${CATBENCH_BASE_EMULATOR_LAUNCHER:-${SCRIPT_DIR}/start_emu_headless.sh}"

# These values are part of the CATBench runtime contract. A future resource
# change must edit this wrapper, update its provenance pin, and receive a new
# release approval; an ambient environment variable cannot silently change it.
readonly PINNED_MEMORY_MB=4096
readonly PINNED_CORES=2
REQUESTED_MEMORY_MB="${CATBENCH_EMULATOR_MEMORY_MB:-${PINNED_MEMORY_MB}}"
REQUESTED_CORES="${CATBENCH_EMULATOR_CORES:-${PINNED_CORES}}"

die() {
  echo "ERROR: $*" >&2
  exit 64
}

[[ "${REQUESTED_MEMORY_MB}" =~ ^[1-9][0-9]*$ ]] \
  || die "CATBENCH_EMULATOR_MEMORY_MB must be a positive integer"
[[ "${REQUESTED_CORES}" =~ ^[1-9][0-9]*$ ]] \
  || die "CATBENCH_EMULATOR_CORES must be a positive integer"
[[ "${REQUESTED_MEMORY_MB}" == "${PINNED_MEMORY_MB}" ]] \
  || die "CATBench memory drift: requested ${REQUESTED_MEMORY_MB} MiB; pinned ${PINNED_MEMORY_MB} MiB"
[[ "${REQUESTED_CORES}" == "${PINNED_CORES}" ]] \
  || die "CATBench core drift: requested ${REQUESTED_CORES}; pinned ${PINNED_CORES}"

print_contract() {
  echo "CATBENCH_EMULATOR_RESOURCE_CONTRACT memory_mb=${PINNED_MEMORY_MB} cores=${PINNED_CORES}"
}

if [[ "${CATBENCH_PRINT_EMULATOR_RESOURCE_CONTRACT:-0}" == "1" ]]; then
  print_contract
  exit 0
fi

[[ -f "${BASE_LAUNCHER}" && ! -L "${BASE_LAUNCHER}" ]] \
  || die "base emulator launcher must be a regular non-symlink file: ${BASE_LAUNCHER}"

REAL_EMULATOR="$(command -v emulator || true)"
[[ -n "${REAL_EMULATOR}" ]] || die "emulator executable is not on PATH"

# The legacy launcher predates nounset and reads this optional variable as a
# bare expansion.  Because it is sourced below under this wrapper's `set -u`,
# bind the legacy default explicitly instead of weakening nounset globally.
HW_ACCEL_OVERRIDE="${HW_ACCEL_OVERRIDE:-}"

# The legacy launcher invokes `nohup emulator ...`. Intercept only that call,
# preserve every unrelated argument, and replace each unique resource flag.
# The direct `emulator -accel-check` probe remains untouched.
nohup() {
  if [[ "${1:-}" != "emulator" ]]; then
    command nohup "$@"
    return
  fi
  shift
  local args=("$@")
  local index memory_count=0 core_count=0
  for ((index = 0; index < ${#args[@]}; index++)); do
    case "${args[index]}" in
      -memory)
        ((index + 1 < ${#args[@]})) \
          || die "emulator -memory flag has no value"
        memory_count=$((memory_count + 1))
        args[index + 1]="${PINNED_MEMORY_MB}"
        index=$((index + 1))
        ;;
      -cores)
        ((index + 1 < ${#args[@]})) \
          || die "emulator -cores flag has no value"
        core_count=$((core_count + 1))
        args[index + 1]="${PINNED_CORES}"
        index=$((index + 1))
        ;;
    esac
  done
  [[ "${memory_count}" == "1" ]] \
    || die "expected exactly one emulator -memory flag; found ${memory_count}"
  [[ "${core_count}" == "1" ]] \
    || die "expected exactly one emulator -cores flag; found ${core_count}"
  print_contract
  command nohup "${REAL_EMULATOR}" "${args[@]}"
}

# shellcheck source=/dev/null
source "${BASE_LAUNCHER}"
