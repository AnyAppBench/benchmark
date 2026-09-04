#!/usr/bin/env bash
# Create and optionally start multiple AndroidWorld-compatible AVDs.

set -euo pipefail

NUM_EMULATORS="${NUM_EMULATORS:-4}"
AVD_PREFIX="${AVD_PREFIX:-catbench_api33}"
SYSTEM_IMAGE="${SYSTEM_IMAGE:-system-images;android-33;google_apis;x86_64}"
DEVICE_PROFILE="${DEVICE_PROFILE:-pixel_6}"
DATA_ROOT="${DATA_ROOT:-${CATBENCH_AVD_ROOT:-$HOME/${USER}/catbench_avds}}"
START_EMULATORS="${START_EMULATORS:-1}"
HEADLESS="${HEADLESS:-1}"
WIPE_DATA="${WIPE_DATA:-1}"
NO_METRICS="${NO_METRICS:-1}"
RAM_MB="${RAM_MB:-4096}"
DISK_SIZE="${DISK_SIZE:-16G}"
FIRST_CONSOLE_PORT="${FIRST_CONSOLE_PORT:-5554}"
FIRST_GRPC_PORT="${FIRST_GRPC_PORT:-8554}"
ANDROID_HOME="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-${HOME}/Android/Sdk}}"
EMULATOR_EXTRA_ARGS="${EMULATOR_EXTRA_ARGS:-}"

AVDMANAGER="${AVDMANAGER:-${ANDROID_HOME}/cmdline-tools/latest/bin/avdmanager}"
SDKMANAGER="${SDKMANAGER:-${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager}"
EMULATOR="${EMULATOR:-${ANDROID_HOME}/emulator/emulator}"

mkdir -p "${DATA_ROOT}/logs"

if [[ ! -x "${AVDMANAGER}" ]]; then
  echo "avdmanager not found: ${AVDMANAGER}" >&2
  exit 2
fi
if [[ ! -x "${EMULATOR}" ]]; then
  echo "emulator not found: ${EMULATOR}" >&2
  exit 2
fi

if [[ -x "${SDKMANAGER}" ]]; then
  yes | "${SDKMANAGER}" "${SYSTEM_IMAGE}" >/dev/null || true
fi

echo "Creating ${NUM_EMULATORS} AVD(s) under ${DATA_ROOT}"
for i in $(seq 0 "$((NUM_EMULATORS - 1))"); do
  name="${AVD_PREFIX}_${i}"
  avd_home="${DATA_ROOT}/avd_home_${i}"
  mkdir -p "${avd_home}"
  export ANDROID_AVD_HOME="${avd_home}"

  if [[ ! -d "${avd_home}/${name}.avd" ]]; then
    echo "  [create] ${name}"
    echo "no" | "${AVDMANAGER}" create avd \
      --force \
      --name "${name}" \
      --package "${SYSTEM_IMAGE}" \
      --device "${DEVICE_PROFILE}" >/dev/null
  else
    echo "  [exists] ${name}"
  fi

  config="${avd_home}/${name}.avd/config.ini"
  if [[ -f "${config}" ]]; then
    grep -q '^hw.ramSize=' "${config}" \
      && sed -i "s/^hw.ramSize=.*/hw.ramSize=${RAM_MB}/" "${config}" \
      || echo "hw.ramSize=${RAM_MB}" >> "${config}"
    grep -q '^disk.dataPartition.size=' "${config}" \
      && sed -i "s/^disk.dataPartition.size=.*/disk.dataPartition.size=${DISK_SIZE}/" "${config}" \
      || echo "disk.dataPartition.size=${DISK_SIZE}" >> "${config}"
  fi

  console_port="$((FIRST_CONSOLE_PORT + 2 * i))"
  adb_port="$((console_port + 1))"
  grpc_port="$((FIRST_GRPC_PORT + i))"
  echo "  serial=emulator-${console_port} console_port=${console_port} grpc_port=${grpc_port}"

  if [[ "${START_EMULATORS}" == "1" ]]; then
    args=(
      -avd "${name}"
      -ports "${console_port},${adb_port}"
      -grpc "${grpc_port}"
      -no-snapshot
      -no-boot-anim
      -no-audio
    )
    if [[ "${WIPE_DATA}" == "1" ]]; then
      args+=(-wipe-data)
    fi
    if [[ "${NO_METRICS}" == "1" ]]; then
      args+=(-no-metrics)
    fi
    if [[ "${HEADLESS}" == "1" ]]; then
      args+=(-no-window)
    fi
    if [[ -n "${EMULATOR_EXTRA_ARGS}" ]]; then
      # Intentionally uses shell-style splitting for simple emulator flags.
      read -r -a extra_args <<< "${EMULATOR_EXTRA_ARGS}"
      args+=("${extra_args[@]}")
    fi
    log_path="${DATA_ROOT}/logs/${name}.log"
    echo "  [start] ${name}; log=${log_path}"
    ANDROID_AVD_HOME="${avd_home}" nohup setsid "${EMULATOR}" "${args[@]}" >"${log_path}" 2>&1 &
  fi
done

echo
echo "Use these runner flags per emulator:"
for i in $(seq 0 "$((NUM_EMULATORS - 1))"); do
  console_port="$((FIRST_CONSOLE_PORT + 2 * i))"
  grpc_port="$((FIRST_GRPC_PORT + i))"
  echo "  --console_port=${console_port} --grpc_port=${grpc_port}"
done
