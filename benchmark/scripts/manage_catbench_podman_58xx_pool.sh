#!/usr/bin/env bash
# Manage an isolated, diagnostic-only Podman emulator pool.

set -euo pipefail

ACTION="${1:-status}"
ACTION_WORKER_INDEX="${2:-}"
# Default to the checkout this script lives in, so the pool works from any
# clone location. Override with REPO_ROOT when running out of tree.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PODMAN_BIN="${PODMAN_BIN:-/usr/bin/podman}"
PODMAN_ROOT="${CATBENCH_PODMAN_ROOT:-/tmp/catbench_qwen_c2_podman/storage}"
PODMAN_RUNROOT="${CATBENCH_PODMAN_RUNROOT:-/tmp/catbench_qwen_c2_podman/runroot}"
ADB="${ADB:-$HOME/Android/Sdk/platform-tools/adb}"
ANDROID_HOME="${CATBENCH_ANDROID_HOME:-$HOME/Android/Sdk}"
IMAGE="${CATBENCH_PODMAN_EMULATOR_IMAGE:-}"
BUILD_TAG="${CATBENCH_PODMAN_BUILD_TAG:-localhost/catbench-emulator-runtime:unapproved-candidate}"
RUNTIME_DISPOSITION="${CATBENCH_PODMAN_RUNTIME_DISPOSITION:-}"
AVD_ROOT="${CATBENCH_PODMAN_AVD_ROOT:-/tmp/catbench_qwen_c2_avds_5800}"
NAME_PREFIX="catbench-qwen-c2-emu"
AVD_PREFIX="catbench_qwen_c2_5800"
POOL_SIZE="${CATBENCH_PODMAN_POOL_SIZE:-8}"
FIRST_ADB_SERVER_PORT="${CATBENCH_PODMAN_FIRST_ADB_SERVER_PORT:-5041}"
EMULATOR_TIMEOUT="${EMULATOR_TIMEOUT:-600}"
START_SCRIPT="${REPO_ROOT}/benchmark/docker_setup/start_catbench_emu_headless.sh"
BASE_START_SCRIPT="${REPO_ROOT}/benchmark/docker_setup/start_emu_headless.sh"
EMULATOR_MEMORY_MB="${CATBENCH_EMULATOR_MEMORY_MB:-4096}"
EMULATOR_CORES="${CATBENCH_EMULATOR_CORES:-2}"
CONTAINERFILE="${REPO_ROOT}/benchmark/docker_setup/Containerfile.emulator-runtime"

export ADB_LOCAL_TRANSPORT_MAX_PORT=5900
export ADB_MDNS_AUTO_CONNECT=

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "${POOL_SIZE}" =~ ^[1-9][0-9]*$ ]] \
  || die "CATBENCH_PODMAN_POOL_SIZE must be a positive integer"
(( POOL_SIZE <= 32 )) \
  || die "CATBENCH_PODMAN_POOL_SIZE must be at most 32"
LAST_WORKER_INDEX="$((POOL_SIZE - 1))"

require_worker_index() {
  [[ "${ACTION_WORKER_INDEX}" =~ ^[0-9]+$ ]] \
    && (( ACTION_WORKER_INDEX <= LAST_WORKER_INDEX )) \
    || die "worker index must be an integer in [0, ${LAST_WORKER_INDEX}]"
}

worker_values() {
  local index="$1"
  WORKER_CONSOLE_PORT="$((5800 + 2 * index))"
  WORKER_ADB_TRANSPORT_PORT="$((WORKER_CONSOLE_PORT + 1))"
  WORKER_GRPC_PORT="$((8800 + index))"
  WORKER_ADB_SERVER_PORT="$((FIRST_ADB_SERVER_PORT + index))"
  WORKER_SERIAL="emulator-${WORKER_CONSOLE_PORT}"
  WORKER_NAME="${NAME_PREFIX}-${index}"
  WORKER_AVD_NAME="${AVD_PREFIX}_${index}"
  WORKER_AVD_HOME="${AVD_ROOT}/avd_home_${index}"
}

container_exists() {
  podman_cmd container exists "$1"
}

podman_cmd() {
  mkdir -p "${PODMAN_ROOT}" "${PODMAN_RUNROOT}"
  "${PODMAN_BIN}" \
    --root "${PODMAN_ROOT}" \
    --runroot "${PODMAN_RUNROOT}" \
    --storage-opt ignore_chown_errors=true \
    "$@"
}

port_is_listening() {
  local port="$1"
  ss -H -ltn 2>/dev/null \
    | awk -v wanted=":${port}" '$4 ~ wanted "$" {found=1} END {exit(found ? 0 : 1)}'
}

validate_launch_contract() {
  [[ "${RUNTIME_DISPOSITION}" == "diagnostic_only" ]] \
    || die "CATBENCH_PODMAN_RUNTIME_DISPOSITION must be diagnostic_only; this manager has no primary-release approval path"
  [[ "${IMAGE}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die "CATBENCH_PODMAN_EMULATOR_IMAGE must be an exact local sha256 image ID; mutable tags and repository-name mixing are forbidden"
  [[ "${EMULATOR_MEMORY_MB}" == "4096" ]] \
    || die "CATBench emulator memory must match the pinned 4096 MiB contract"
  [[ "${EMULATOR_CORES}" == "2" ]] \
    || die "CATBench emulator cores must match the pinned 2-core contract"
  [[ -f "${START_SCRIPT}" ]] || die "emulator launcher is unavailable"
  [[ -f "${BASE_START_SCRIPT}" ]] || die "base emulator launcher is unavailable"
  CATBENCH_EMULATOR_MEMORY_MB="${EMULATOR_MEMORY_MB}" \
  CATBENCH_EMULATOR_CORES="${EMULATOR_CORES}" \
  CATBENCH_PRINT_EMULATOR_RESOURCE_CONTRACT=1 \
    bash "${START_SCRIPT}"
  echo "CATBENCH_PODMAN_RUNTIME_CONTRACT disposition=diagnostic_only analysis_eligible=false image=${IMAGE}"
}

require_base_runtime() {
  validate_launch_contract
  [[ "${FIRST_ADB_SERVER_PORT}" =~ ^[0-9]+$ ]] \
    || die "CATBENCH_PODMAN_FIRST_ADB_SERVER_PORT must be an integer"
  (( FIRST_ADB_SERVER_PORT >= 1024 \
      && FIRST_ADB_SERVER_PORT + LAST_WORKER_INDEX <= 65535 )) \
    || die "per-worker ADB server ports must remain in [1024, 65535]"
  [[ -x "${PODMAN_BIN}" ]] || die "Podman is unavailable: ${PODMAN_BIN}"
  [[ -x "${ADB}" ]] || die "ADB is unavailable: ${ADB}"
  [[ -d "${ANDROID_HOME}" ]] || die "Android SDK is unavailable: ${ANDROID_HOME}"
  [[ -e /dev/kvm ]] || die "/dev/kvm is unavailable"
  [[ -f "${START_SCRIPT}" ]] || die "emulator launcher is unavailable"
  [[ -f "${BASE_START_SCRIPT}" ]] || die "base emulator launcher is unavailable"
}

prepare_avds() {
  require_base_runtime
  NUM_EMULATORS="${POOL_SIZE}" \
  AVD_PREFIX="${AVD_PREFIX}" \
  DATA_ROOT="${AVD_ROOT}" \
  START_EMULATORS=0 \
  SDKMANAGER=/nonexistent \
  RAM_MB="${EMULATOR_MEMORY_MB}" \
  DISK_SIZE=16G \
  FIRST_CONSOLE_PORT=5800 \
  FIRST_GRPC_PORT=8800 \
  ANDROID_HOME="${ANDROID_HOME}" \
    bash "${REPO_ROOT}/benchmark/scripts/create_android_emulators.sh"
}

build_image() {
  [[ -x "${PODMAN_BIN}" ]] || die "Podman is unavailable: ${PODMAN_BIN}"
  podman_cmd build \
    --pull=missing \
    --tag "${BUILD_TAG}" \
    --file "${CONTAINERFILE}" \
    "${REPO_ROOT}/benchmark"
  local candidate_id
  candidate_id="$(podman_cmd image inspect --format '{{.Id}}' "${BUILD_TAG}")"
  candidate_id="sha256:${candidate_id#sha256:}"
  echo "UNAPPROVED_DIAGNOSTIC_IMAGE tag=${BUILD_TAG} id=${candidate_id}"
  echo "Set CATBENCH_PODMAN_EMULATOR_IMAGE to that exact sha256 ID only after recording its diagnostic provenance."
}

preflight_start() {
  local allow_existing="${1:-0}"
  require_base_runtime
  podman_cmd image exists "${IMAGE}" || die "image is missing: ${IMAGE}"
  local actual_image_id
  actual_image_id="$(podman_cmd image inspect --format '{{.Id}}' "${IMAGE}")"
  # Podman formats .Id as bare hex while its accepted exact-ID reference is
  # sha256:<hex>. Normalize before comparing; otherwise every valid ID is
  # rejected solely because of CLI presentation.
  actual_image_id="sha256:${actual_image_id#sha256:}"
  [[ "${actual_image_id}" == "${IMAGE}" ]] \
    || die "resolved image ID ${actual_image_id} does not equal requested ${IMAGE}"
  local index port
  for index in $(seq 0 "${LAST_WORKER_INDEX}"); do
    worker_values "${index}"
    "${ADB}" -P "${WORKER_ADB_SERVER_PORT}" start-server >/dev/null
    "${ADB}" -P "${WORKER_ADB_SERVER_PORT}" server-status >/dev/null \
      || die "port ${WORKER_ADB_SERVER_PORT} is not a usable ADB server"
    [[ -d "${WORKER_AVD_HOME}/${WORKER_AVD_NAME}.avd" ]] \
      || die "AVD is missing: ${WORKER_AVD_HOME}/${WORKER_AVD_NAME}.avd"
    if container_exists "${WORKER_NAME}"; then
      [[ "${allow_existing}" == "1" ]] \
        || die "container already exists: ${WORKER_NAME}"
      local state
      state="$(podman_cmd inspect --format '{{.State.Status}}' "${WORKER_NAME}")"
      [[ "${state}" == "running" ]] \
        || die "existing container is not running: ${WORKER_NAME} (${state})"
      continue
    fi
    for port in \
      "${WORKER_CONSOLE_PORT}" \
      "${WORKER_ADB_TRANSPORT_PORT}" \
      "${WORKER_GRPC_PORT}"; do
      port_is_listening "${port}" \
        && die "reserved port is already listening: ${port}"
    done
  done
  return 0
}

start_worker() {
  local index="$1"
  worker_values "${index}"
  podman_cmd run -d \
    --name "${WORKER_NAME}" \
    --hostname "${WORKER_NAME}" \
    --network host \
    --device /dev/kvm:/dev/kvm:rwm \
    --security-opt label=disable \
    --label "catbench.runtime_disposition=diagnostic_only" \
    --label "catbench.analysis_eligible=false" \
    --label "catbench.image_id=${IMAGE}" \
    --label "catbench.emulator_memory_mb=${EMULATOR_MEMORY_MB}" \
    --label "catbench.emulator_cores=${EMULATOR_CORES}" \
    --label "catbench.adb_server_port=${WORKER_ADB_SERVER_PORT}" \
    --mount "type=bind,src=${ANDROID_HOME},dst=/host-android-sdk,ro=true" \
    --mount "type=bind,src=${WORKER_AVD_HOME},dst=${WORKER_AVD_HOME}" \
    --mount "type=bind,src=${START_SCRIPT},dst=/catbench/start_catbench_emu_headless.sh,ro=true" \
    --mount "type=bind,src=${BASE_START_SCRIPT},dst=/catbench/start_emu_headless.sh,ro=true" \
    -e DOCKER=true \
    -e EMULATOR_NAME="${WORKER_AVD_NAME}" \
    -e ANDROID_AVD_HOME="${WORKER_AVD_HOME}" \
    -e EMULATOR_CONSOLE_PORT="${WORKER_CONSOLE_PORT}" \
    -e EMULATOR_ADB_PORT="${WORKER_ADB_TRANSPORT_PORT}" \
    -e EMULATOR_GRPC_PORT="${WORKER_GRPC_PORT}" \
    -e ANDROID_SERIAL="${WORKER_SERIAL}" \
    -e ADB_SERVER_PORT="${WORKER_ADB_SERVER_PORT}" \
    -e ANDROID_ADB_SERVER_PORT="${WORKER_ADB_SERVER_PORT}" \
    -e ADB_LOCAL_TRANSPORT_MAX_PORT=5900 \
    -e ADB_MDNS_AUTO_CONNECT= \
    -e EMULATOR_GPU_MODE=lavapipe \
    -e EMULATOR_SAFE_GPU_MODE=lavapipe \
    -e EMULATOR_SEGFAULT_GPU_FALLBACK=lavapipe \
    -e EMULATOR_SAFE_MODE=1 \
    -e EMULATOR_TIMEOUT="${EMULATOR_TIMEOUT}" \
    -e CATBENCH_BASE_EMULATOR_LAUNCHER=/catbench/start_emu_headless.sh \
    -e CATBENCH_EMULATOR_MEMORY_MB="${EMULATOR_MEMORY_MB}" \
    -e CATBENCH_EMULATOR_CORES="${EMULATOR_CORES}" \
    -e CATBENCH_SKIP_EXISTING_EMULATOR_CLEANUP=1 \
    "${IMAGE}" \
    -lc '/catbench/start_catbench_emu_headless.sh && exec sleep infinity' >/dev/null
  echo "STARTED name=${WORKER_NAME} serial=${WORKER_SERIAL} console=${WORKER_CONSOLE_PORT} grpc=${WORKER_GRPC_PORT} adb_server=${WORKER_ADB_SERVER_PORT}"
}

wait_for_worker() {
  local index="$1"
  worker_values "${index}"
  local deadline="$((SECONDS + EMULATOR_TIMEOUT))"
  while (( SECONDS < deadline )); do
    local state booted uid
    state="$(podman_cmd inspect --format '{{.State.Status}}' "${WORKER_NAME}" 2>/dev/null || true)"
    [[ "${state}" == "running" ]] || {
      echo "Worker exited during boot: ${WORKER_NAME}" >&2
      podman_cmd logs --tail 120 "${WORKER_NAME}" >&2 || true
      return 1
    }
    booted="$("${ADB}" -P "${WORKER_ADB_SERVER_PORT}" -s "${WORKER_SERIAL}" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' || true)"
    uid="$("${ADB}" -P "${WORKER_ADB_SERVER_PORT}" -s "${WORKER_SERIAL}" shell id -u 2>/dev/null | tr -d '\r' || true)"
    if [[ "${booted}" == "1" && "${uid}" == "0" ]]; then
      configure_worker_device "${WORKER_SERIAL}" "${WORKER_ADB_SERVER_PORT}"
      echo "READY name=${WORKER_NAME} serial=${WORKER_SERIAL} console=${WORKER_CONSOLE_PORT} grpc=${WORKER_GRPC_PORT} adb_server=${WORKER_ADB_SERVER_PORT}"
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for ${WORKER_NAME}" >&2
  podman_cmd logs --tail 120 "${WORKER_NAME}" >&2 || true
  return 1
}

configure_worker_device() {
  local serial="$1"
  local adb_server_port="$2"
  local adb_device=("${ADB}" -P "${adb_server_port}" -s "${serial}")

  # Reapply benchmark-critical settings through this lane's independent ADB
  # server and exact serial.  No server owns more than one benchmark emulator.
  "${adb_device[@]}" shell settings put secure lock_screen_lock_after_timeout 2147483647
  "${adb_device[@]}" shell settings put system screen_off_timeout 2147483647
  "${adb_device[@]}" shell settings put global stay_on_while_plugged_in 7
  "${adb_device[@]}" shell settings put global window_animation_scale 0
  "${adb_device[@]}" shell settings put global transition_animation_scale 0
  "${adb_device[@]}" shell settings put global animator_duration_scale 0
  "${adb_device[@]}" shell settings put system accelerometer_rotation 0
  "${adb_device[@]}" shell settings put system user_rotation 0
  "${adb_device[@]}" shell logcat -G 2M
  "${adb_device[@]}" shell input keyevent KEYCODE_WAKEUP
  "${adb_device[@]}" shell wm dismiss-keyguard >/dev/null 2>&1 || true

  local screen_timeout auto_rotate
  screen_timeout="$("${adb_device[@]}" shell settings get system screen_off_timeout | tr -d '\r')"
  auto_rotate="$("${adb_device[@]}" shell settings get system accelerometer_rotation | tr -d '\r')"
  [[ "${screen_timeout}" == "2147483647" ]] \
    || die "screen timeout configuration failed for ${serial}: ${screen_timeout}"
  [[ "${auto_rotate}" == "0" ]] \
    || die "rotation configuration failed for ${serial}: ${auto_rotate}"
}

start_pool() {
  preflight_start 1
  local index
  for index in $(seq 0 "${LAST_WORKER_INDEX}"); do
    worker_values "${index}"
    if container_exists "${WORKER_NAME}"; then
      echo "REUSE name=${WORKER_NAME} serial=${WORKER_SERIAL}"
    else
      start_worker "${index}"
    fi
  done
  local failed=0
  for index in $(seq 0 "${LAST_WORKER_INDEX}"); do
    wait_for_worker "${index}" || failed=1
  done
  (( failed == 0 )) || exit 3
  print_specs
}

start_one() {
  require_worker_index
  preflight_start 1
  worker_values "${ACTION_WORKER_INDEX}"
  ! container_exists "${WORKER_NAME}" \
    || die "container already exists: ${WORKER_NAME}"
  start_worker "${ACTION_WORKER_INDEX}"
  wait_for_worker "${ACTION_WORKER_INDEX}"
}

status_pool() {
  require_base_runtime
  local index
  for index in $(seq 0 "${LAST_WORKER_INDEX}"); do
    worker_values "${index}"
    if ! container_exists "${WORKER_NAME}"; then
      echo "MISSING name=${WORKER_NAME} serial=${WORKER_SERIAL}"
      continue
    fi
    local state booted uid
    state="$(podman_cmd inspect --format '{{.State.Status}}' "${WORKER_NAME}")"
    booted="$("${ADB}" -P "${WORKER_ADB_SERVER_PORT}" -s "${WORKER_SERIAL}" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' || true)"
    uid="$("${ADB}" -P "${WORKER_ADB_SERVER_PORT}" -s "${WORKER_SERIAL}" shell id -u 2>/dev/null | tr -d '\r' || true)"
    echo "STATUS name=${WORKER_NAME} state=${state} serial=${WORKER_SERIAL} boot_completed=${booted:-0} uid=${uid:-unknown} console=${WORKER_CONSOLE_PORT} grpc=${WORKER_GRPC_PORT} adb_server=${WORKER_ADB_SERVER_PORT}"
  done
}

stop_pool() {
  require_base_runtime
  local index
  for index in $(seq 0 "${LAST_WORKER_INDEX}"); do
    worker_values "${index}"
    if ! container_exists "${WORKER_NAME}"; then
      echo "MISSING name=${WORKER_NAME}"
      continue
    fi
    local state
    state="$(podman_cmd inspect --format '{{.State.Status}}' "${WORKER_NAME}")"
    if [[ "${state}" == "running" ]]; then
      podman_cmd stop --time 30 "${WORKER_NAME}" >/dev/null
    fi
    podman_cmd rm "${WORKER_NAME}" >/dev/null
    echo "STOPPED name=${WORKER_NAME} avd_preserved=${WORKER_AVD_HOME}"
  done
}

stop_one() {
  require_worker_index
  require_base_runtime
  worker_values "${ACTION_WORKER_INDEX}"
  container_exists "${WORKER_NAME}" || die "container is missing: ${WORKER_NAME}"
  local state
  state="$(podman_cmd inspect --format '{{.State.Status}}' "${WORKER_NAME}")"
  if [[ "${state}" == "running" ]]; then
    podman_cmd stop --time 30 "${WORKER_NAME}" >/dev/null
  fi
  podman_cmd rm "${WORKER_NAME}" >/dev/null
  echo "STOPPED name=${WORKER_NAME} avd_preserved=${WORKER_AVD_HOME}"
}

print_specs() {
  local index specs=""
  for index in $(seq 0 "${LAST_WORKER_INDEX}"); do
    worker_values "${index}"
    [[ -z "${specs}" ]] || specs+=","
    specs+="${WORKER_CONSOLE_PORT}:${WORKER_GRPC_PORT}:-:${WORKER_ADB_SERVER_PORT}"
  done
  echo "CATBENCH_EMULATORS=${specs}"
}

case "${ACTION}" in
  contract) validate_launch_contract ;;
  build) build_image ;;
  prepare) prepare_avds ;;
  preflight) preflight_start ;;
  start) start_pool ;;
  start-one) start_one ;;
  status) status_pool ;;
  stop) stop_pool ;;
  stop-one) stop_one ;;
  specs) print_specs ;;
  *) die "action must be one of: contract, build, prepare, preflight, start, start-one, status, stop, stop-one, specs" ;;
esac
