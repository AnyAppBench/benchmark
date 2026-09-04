#!/usr/bin/env bash
# Manage isolated, headless Docker emulator workers for host-side CATBench runs.
#
# The Docker image is used only as the Android SDK/system-image substrate.  The
# current launcher is bind-mounted read-only, and the stale server/repository
# embedded in the image is never executed.  Each worker has an independent AVD
# volume and independent console, emulator-gRPC, and ADB-server ports.

set -euo pipefail

ACTION="${1:-status}"
ACTION_WORKER_INDEX="${2:-}"
ACTION_VOLUME="${3:-}"
NUM_EMULATORS="${NUM_EMULATORS:-2}"
FIRST_CONSOLE_PORT="${FIRST_CONSOLE_PORT:-5576}"
FIRST_GRPC_PORT="${FIRST_GRPC_PORT:-8576}"
FIRST_ADB_SERVER_PORT="${FIRST_ADB_SERVER_PORT:-5041}"
NAME_PREFIX="${NAME_PREFIX:-catbench-docker-emu}"
VOLUME_PREFIX="${VOLUME_PREFIX:-catbench-docker-avd}"
IMAGE="${IMAGE:-android_world@sha256:6d8b2c148aebd3a1fe626768efe22c01a7a62cdbd2cbbe7d3f973adc57c7dd2f}"
EMULATOR_NAME="${EMULATOR_NAME:-Pixel_6_API_33}"
EMULATOR_TIMEOUT="${EMULATOR_TIMEOUT:-600}"
DOCKER="${DOCKER:-docker}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
START_SCRIPT="${START_SCRIPT:-${BENCHMARK_ROOT}/docker_setup/start_catbench_emu_headless.sh}"
BASE_START_SCRIPT="${BASE_START_SCRIPT:-${BENCHMARK_ROOT}/docker_setup/start_emu_headless.sh}"
EMULATOR_MEMORY_MB="${CATBENCH_EMULATOR_MEMORY_MB:-4096}"
EMULATOR_CORES="${CATBENCH_EMULATOR_CORES:-2}"
ADB="${ADB:-${ANDROID_SDK_ROOT:-${HOME}/Android/Sdk}/platform-tools/adb}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer"
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer"
}

require_port() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be an integer"
  (( value >= 1024 && value <= 65535 )) || die "${name} must be in [1024, 65535]"
}

worker_values() {
  local index="$1"
  WORKER_CONSOLE_PORT="$((FIRST_CONSOLE_PORT + 2 * index))"
  WORKER_ADB_TRANSPORT_PORT="$((WORKER_CONSOLE_PORT + 1))"
  WORKER_GRPC_PORT="$((FIRST_GRPC_PORT + index))"
  WORKER_ADB_SERVER_PORT="$((FIRST_ADB_SERVER_PORT + index))"
  WORKER_SERIAL="emulator-${WORKER_CONSOLE_PORT}"
  WORKER_NAME="${NAME_PREFIX}-${index}"
  WORKER_VOLUME="${VOLUME_PREFIX}-${index}"
}

container_exists() {
  "${DOCKER}" container inspect "$1" >/dev/null 2>&1
}

container_running() {
  [[ "$("${DOCKER}" inspect --format '{{.State.Running}}' "$1" 2>/dev/null || true)" == "true" ]]
}

volume_exists() {
  "${DOCKER}" volume inspect "$1" >/dev/null 2>&1
}

volume_label() {
  local volume="$1"
  local label="$2"
  "${DOCKER}" volume inspect --format "{{index .Labels \"${label}\"}}" "${volume}"
}

port_is_listening() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -H -ltn 2>/dev/null | awk -v wanted=":${port}" '$4 ~ wanted "$" {found=1} END {exit(found ? 0 : 1)}'
    return
  fi
  return 1
}

wait_for_worker() {
  local ready_volume="${1:-${WORKER_VOLUME}}"
  local deadline="$((SECONDS + EMULATOR_TIMEOUT))"
  while (( SECONDS < deadline )); do
    if container_running "${WORKER_NAME}"; then
      local booted
      booted="$(
        "${ADB}" -P "${WORKER_ADB_SERVER_PORT}" -s "${WORKER_SERIAL}" \
          shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' || true
      )"
      if [[ "${booted}" == "1" ]]; then
        echo "READY name=${WORKER_NAME} serial=${WORKER_SERIAL} console=${WORKER_CONSOLE_PORT} grpc=${WORKER_GRPC_PORT} adb_server=${WORKER_ADB_SERVER_PORT} volume=${ready_volume}"
        return 0
      fi
    else
      echo "Worker ${WORKER_NAME} exited during boot" >&2
      "${DOCKER}" logs --tail 120 "${WORKER_NAME}" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for ${WORKER_NAME}" >&2
  "${DOCKER}" logs --tail 120 "${WORKER_NAME}" >&2 || true
  return 1
}

validate_launch_contract() {
  [[ "${IMAGE}" =~ @sha256:[0-9a-f]{64}$ || "${IMAGE}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die "IMAGE must be an immutable repository digest or exact local image ID"
  [[ "${EMULATOR_MEMORY_MB}" == "4096" ]] \
    || die "CATBench emulator memory must match the pinned 4096 MiB contract"
  [[ "${EMULATOR_CORES}" == "2" ]] \
    || die "CATBench emulator cores must match the pinned 2-core contract"
  [[ -f "${START_SCRIPT}" ]] || die "launcher is missing: ${START_SCRIPT}"
  [[ -f "${BASE_START_SCRIPT}" ]] || die "base launcher is missing: ${BASE_START_SCRIPT}"
  CATBENCH_EMULATOR_MEMORY_MB="${EMULATOR_MEMORY_MB}" \
  CATBENCH_EMULATOR_CORES="${EMULATOR_CORES}" \
  CATBENCH_PRINT_EMULATOR_RESOURCE_CONTRACT=1 \
    bash "${START_SCRIPT}"
}

require_runtime() {
  validate_launch_contract
  command -v "${DOCKER}" >/dev/null 2>&1 || die "docker is not installed: ${DOCKER}"
  [[ -x "${ADB}" ]] || die "adb is missing: ${ADB}"
  [[ -f "${START_SCRIPT}" ]] || die "launcher is missing: ${START_SCRIPT}"
  [[ -f "${BASE_START_SCRIPT}" ]] || die "base launcher is missing: ${BASE_START_SCRIPT}"
  [[ -e /dev/kvm ]] || die "/dev/kvm is unavailable"

  RUNTIME_LAUNCHER_SHA="$(sha256sum "${START_SCRIPT}" | awk '{print $1}')"
  RUNTIME_BASE_LAUNCHER_SHA="$(sha256sum "${BASE_START_SCRIPT}" | awk '{print $1}')"
  RUNTIME_IMAGE_ID="$("${DOCKER}" image inspect --format '{{.Id}}' "${IMAGE}")"
}

launch_worker() {
  local index="$1"
  local avd_volume="$2"
  worker_values "${index}"

  "${DOCKER}" run -d \
    --name "${WORKER_NAME}" \
    --hostname "${WORKER_NAME}" \
    --init \
    --network host \
    --device /dev/kvm:/dev/kvm:rwm \
    --stop-timeout 30 \
    --label "catbench.pool=${NAME_PREFIX}" \
    --label "catbench.worker_index=${index}" \
    --label "catbench.image_id=${RUNTIME_IMAGE_ID}" \
    --label "catbench.launcher_sha256=${RUNTIME_LAUNCHER_SHA}" \
    --label "catbench.base_launcher_sha256=${RUNTIME_BASE_LAUNCHER_SHA}" \
    --label "catbench.emulator_memory_mb=${EMULATOR_MEMORY_MB}" \
    --label "catbench.emulator_cores=${EMULATOR_CORES}" \
    --label "catbench.avd_volume=${avd_volume}" \
    --label "catbench.snapshot_clone_id=$(volume_label "${avd_volume}" catbench.snapshot.clone_id)" \
    --mount "type=volume,src=${avd_volume},dst=/root/.android" \
    --mount "type=bind,src=${START_SCRIPT},dst=/catbench/start_catbench_emu_headless.sh,readonly" \
    --mount "type=bind,src=${BASE_START_SCRIPT},dst=/catbench/start_emu_headless.sh,readonly" \
    -e DOCKER=true \
    -e EMULATOR_NAME="${EMULATOR_NAME}" \
    -e EMULATOR_CONSOLE_PORT="${WORKER_CONSOLE_PORT}" \
    -e EMULATOR_ADB_PORT="${WORKER_ADB_TRANSPORT_PORT}" \
    -e EMULATOR_GRPC_PORT="${WORKER_GRPC_PORT}" \
    -e ANDROID_SERIAL="${WORKER_SERIAL}" \
    -e ADB_SERVER_PORT="${WORKER_ADB_SERVER_PORT}" \
    -e ANDROID_ADB_SERVER_PORT="${WORKER_ADB_SERVER_PORT}" \
    -e EMULATOR_GPU_MODE=lavapipe \
    -e EMULATOR_SAFE_GPU_MODE=lavapipe \
    -e EMULATOR_SEGFAULT_GPU_FALLBACK=lavapipe \
    -e EMULATOR_SAFE_MODE=1 \
    -e EMULATOR_TIMEOUT="${EMULATOR_TIMEOUT}" \
    -e CATBENCH_BASE_EMULATOR_LAUNCHER=/catbench/start_emu_headless.sh \
    -e CATBENCH_EMULATOR_MEMORY_MB="${EMULATOR_MEMORY_MB}" \
    -e CATBENCH_EMULATOR_CORES="${EMULATOR_CORES}" \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -lc '/catbench/start_catbench_emu_headless.sh && exec sleep infinity' >/dev/null
  echo "STARTED name=${WORKER_NAME} serial=${WORKER_SERIAL} volume=${avd_volume}"
}

stop_remove_worker() {
  worker_values "$1"
  if ! container_exists "${WORKER_NAME}"; then
    return 0
  fi
  "${ADB}" -P "${WORKER_ADB_SERVER_PORT}" -s "${WORKER_SERIAL}" \
    emu kill >/dev/null 2>&1 || true
  if container_running "${WORKER_NAME}"; then
    "${DOCKER}" stop "${WORKER_NAME}" >/dev/null
  fi
  "${DOCKER}" rm "${WORKER_NAME}" >/dev/null
}

wait_for_emulator_ports_to_close() {
  worker_values "$1"
  local attempt
  for attempt in $(seq 1 40); do
    if ! port_is_listening "${WORKER_CONSOLE_PORT}" \
      && ! port_is_listening "${WORKER_ADB_TRANSPORT_PORT}" \
      && ! port_is_listening "${WORKER_GRPC_PORT}"; then
      return 0
    fi
    sleep 0.25
  done
  die "emulator ports did not close for worker $1"
}

start_pool() {
  require_runtime
  echo "IMAGE ${IMAGE} id=${RUNTIME_IMAGE_ID}"
  echo "LAUNCHER ${START_SCRIPT} sha256=${RUNTIME_LAUNCHER_SHA}"
  echo "BASE_LAUNCHER ${BASE_START_SCRIPT} sha256=${RUNTIME_BASE_LAUNCHER_SHA}"
  echo "RESOURCES memory_mb=${EMULATOR_MEMORY_MB} cores=${EMULATOR_CORES}"

  local index port
  for index in $(seq 0 "$((NUM_EMULATORS - 1))"); do
    worker_values "${index}"
    if container_exists "${WORKER_NAME}"; then
      die "container already exists: ${WORKER_NAME}; use '$0 stop' and rename/remove it explicitly"
    fi
    for port in \
      "${WORKER_CONSOLE_PORT}" \
      "${WORKER_ADB_TRANSPORT_PORT}" \
      "${WORKER_GRPC_PORT}"; do
      if port_is_listening "${port}"; then
        die "host port ${port} is already listening (worker ${index})"
      fi
    done
    if port_is_listening "${WORKER_ADB_SERVER_PORT}"; then
      if ! "${ADB}" -P "${WORKER_ADB_SERVER_PORT}" server-status >/dev/null 2>&1; then
        die "host port ${WORKER_ADB_SERVER_PORT} is occupied by a non-ADB listener (worker ${index})"
      fi
      echo "REUSE adb_server=${WORKER_ADB_SERVER_PORT} worker=${index}"
    fi
  done

  for index in $(seq 0 "$((NUM_EMULATORS - 1))"); do
    worker_values "${index}"
    "${DOCKER}" volume create \
      --label "catbench.pool=${NAME_PREFIX}" \
      --label "catbench.worker_index=${index}" \
      "${WORKER_VOLUME}" >/dev/null
    launch_worker "${index}" "${WORKER_VOLUME}"
  done

  local failed=0
  for index in $(seq 0 "$((NUM_EMULATORS - 1))"); do
    worker_values "${index}"
    wait_for_worker "${WORKER_VOLUME}" || failed=1
  done
  (( failed == 0 )) || exit 3
  print_specs
}

activate_volume() {
  require_runtime
  require_nonnegative_integer WORKER_INDEX "${ACTION_WORKER_INDEX}"
  (( ACTION_WORKER_INDEX < NUM_EMULATORS )) \
    || die "WORKER_INDEX must be smaller than NUM_EMULATORS"
  [[ "${ACTION_VOLUME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$ ]] \
    || die "invalid Docker volume name: ${ACTION_VOLUME}"
  volume_exists "${ACTION_VOLUME}" || die "volume does not exist: ${ACTION_VOLUME}"
  [[ "$(volume_label "${ACTION_VOLUME}" catbench.snapshot.role)" == "episode-clone" ]] \
    || die "volume is not an episode clone: ${ACTION_VOLUME}"
  [[ "$(volume_label "${ACTION_VOLUME}" catbench.snapshot.worker_index)" == "${ACTION_WORKER_INDEX}" ]] \
    || die "episode clone has the wrong worker index"

  worker_values "${ACTION_WORKER_INDEX}"
  if container_exists "${WORKER_NAME}"; then
    local mounted_volume mounted_role
    mounted_volume="$("${DOCKER}" inspect --format '{{range .Mounts}}{{if eq .Destination "/root/.android"}}{{.Name}}{{end}}{{end}}' "${WORKER_NAME}")"
    mounted_role=""
    if [[ -n "${mounted_volume}" ]] && volume_exists "${mounted_volume}"; then
      mounted_role="$(volume_label "${mounted_volume}" catbench.snapshot.role)"
    fi
    if [[ "${mounted_role}" == "episode-clone" && "${mounted_volume}" != "${ACTION_VOLUME}" ]]; then
      die "worker already owns a different unreleased episode clone: ${mounted_volume}"
    fi
    stop_remove_worker "${ACTION_WORKER_INDEX}"
    wait_for_emulator_ports_to_close "${ACTION_WORKER_INDEX}"
  fi

  launch_worker "${ACTION_WORKER_INDEX}" "${ACTION_VOLUME}"
  wait_for_worker "${ACTION_VOLUME}"
}

deactivate_volume() {
  require_runtime
  require_nonnegative_integer WORKER_INDEX "${ACTION_WORKER_INDEX}"
  (( ACTION_WORKER_INDEX < NUM_EMULATORS )) \
    || die "WORKER_INDEX must be smaller than NUM_EMULATORS"
  [[ "${ACTION_VOLUME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$ ]] \
    || die "invalid Docker volume name: ${ACTION_VOLUME}"
  worker_values "${ACTION_WORKER_INDEX}"
  container_exists "${WORKER_NAME}" || die "worker container is missing: ${WORKER_NAME}"
  local mounted_volume
  mounted_volume="$("${DOCKER}" inspect --format '{{range .Mounts}}{{if eq .Destination "/root/.android"}}{{.Name}}{{end}}{{end}}' "${WORKER_NAME}")"
  [[ "${mounted_volume}" == "${ACTION_VOLUME}" ]] \
    || die "worker mounts ${mounted_volume:-no AVD volume}, not ${ACTION_VOLUME}"
  [[ "$(volume_label "${ACTION_VOLUME}" catbench.snapshot.role)" == "episode-clone" ]] \
    || die "mounted volume is not an episode clone"
  stop_remove_worker "${ACTION_WORKER_INDEX}"
  wait_for_emulator_ports_to_close "${ACTION_WORKER_INDEX}"
  echo "DEACTIVATED name=${WORKER_NAME} volume=${ACTION_VOLUME}"
}

detach_baseline_worker() {
  require_runtime
  require_nonnegative_integer WORKER_INDEX "${ACTION_WORKER_INDEX}"
  (( ACTION_WORKER_INDEX < NUM_EMULATORS )) \
    || die "WORKER_INDEX must be smaller than NUM_EMULATORS"
  worker_values "${ACTION_WORKER_INDEX}"
  container_exists "${WORKER_NAME}" || die "worker container is missing: ${WORKER_NAME}"
  local mounted_volume mounted_role
  mounted_volume="$("${DOCKER}" inspect --format '{{range .Mounts}}{{if eq .Destination "/root/.android"}}{{.Name}}{{end}}{{end}}' "${WORKER_NAME}")"
  [[ -n "${mounted_volume}" ]] || die "worker has no named AVD volume"
  mounted_role="$(volume_label "${mounted_volume}" catbench.snapshot.role)"
  [[ "${mounted_role}" != "episode-clone" ]] \
    || die "refusing to detach an episode clone; use deactivate-volume with its exact name"
  stop_remove_worker "${ACTION_WORKER_INDEX}"
  wait_for_emulator_ports_to_close "${ACTION_WORKER_INDEX}"
  echo "DETACHED name=${WORKER_NAME} volume_preserved=${mounted_volume}"
}

attach_baseline_worker() {
  require_runtime
  require_nonnegative_integer WORKER_INDEX "${ACTION_WORKER_INDEX}"
  (( ACTION_WORKER_INDEX < NUM_EMULATORS )) \
    || die "WORKER_INDEX must be smaller than NUM_EMULATORS"
  worker_values "${ACTION_WORKER_INDEX}"
  container_exists "${WORKER_NAME}" && die "worker container already exists: ${WORKER_NAME}"
  volume_exists "${WORKER_VOLUME}" || die "baseline volume is missing: ${WORKER_VOLUME}"
  [[ "$(volume_label "${WORKER_VOLUME}" catbench.snapshot.role)" != "episode-clone" ]] \
    || die "baseline volume name points to an episode clone"
  wait_for_emulator_ports_to_close "${ACTION_WORKER_INDEX}"
  launch_worker "${ACTION_WORKER_INDEX}" "${WORKER_VOLUME}"
  wait_for_worker "${WORKER_VOLUME}"
}

status_pool() {
  local index booted state mounted_volume
  for index in $(seq 0 "$((NUM_EMULATORS - 1))"); do
    worker_values "${index}"
    if ! container_exists "${WORKER_NAME}"; then
      echo "MISSING name=${WORKER_NAME}"
      continue
    fi
    state="$("${DOCKER}" inspect --format '{{.State.Status}}' "${WORKER_NAME}")"
    mounted_volume="$("${DOCKER}" inspect --format '{{range .Mounts}}{{if eq .Destination "/root/.android"}}{{.Name}}{{end}}{{end}}' "${WORKER_NAME}")"
    booted="$(
      "${ADB}" -P "${WORKER_ADB_SERVER_PORT}" -s "${WORKER_SERIAL}" \
        shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' || true
    )"
    echo "STATUS name=${WORKER_NAME} state=${state} serial=${WORKER_SERIAL} boot_completed=${booted:-0} console=${WORKER_CONSOLE_PORT} grpc=${WORKER_GRPC_PORT} adb_server=${WORKER_ADB_SERVER_PORT} volume=${mounted_volume:-none}"
  done
}

stop_pool() {
  local index
  for index in $(seq 0 "$((NUM_EMULATORS - 1))"); do
    worker_values "${index}"
    if ! container_exists "${WORKER_NAME}"; then
      echo "MISSING name=${WORKER_NAME}"
      continue
    fi
    "${ADB}" -P "${WORKER_ADB_SERVER_PORT}" -s "${WORKER_SERIAL}" \
      emu kill >/dev/null 2>&1 || true
    "${DOCKER}" stop "${WORKER_NAME}" >/dev/null
    echo "STOPPED name=${WORKER_NAME} volume_preserved=${WORKER_VOLUME}"
  done
}

print_specs() {
  local index specs=""
  for index in $(seq 0 "$((NUM_EMULATORS - 1))"); do
    worker_values "${index}"
    [[ -z "${specs}" ]] || specs+=","
    specs+="${WORKER_CONSOLE_PORT}:${WORKER_GRPC_PORT}:${WORKER_SERIAL}:${WORKER_ADB_SERVER_PORT}"
  done
  echo "CATBENCH_EMULATORS=${specs}"
}

require_positive_integer NUM_EMULATORS "${NUM_EMULATORS}"
require_port FIRST_CONSOLE_PORT "${FIRST_CONSOLE_PORT}"
require_port FIRST_GRPC_PORT "${FIRST_GRPC_PORT}"
require_port FIRST_ADB_SERVER_PORT "${FIRST_ADB_SERVER_PORT}"

case "${ACTION}" in
  contract) validate_launch_contract ;;
  start) start_pool ;;
  status) status_pool ;;
  stop) stop_pool ;;
  specs) print_specs ;;
  activate-volume) activate_volume ;;
  deactivate-volume) deactivate_volume ;;
  detach-worker) detach_baseline_worker ;;
  attach-worker) attach_baseline_worker ;;
  *) die "action must be one of: contract, start, status, stop, specs, activate-volume, deactivate-volume, detach-worker, attach-worker" ;;
esac
