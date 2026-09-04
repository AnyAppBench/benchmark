#!/bin/bash

set -euo pipefail

adb_base=(adb)
if [[ -n "${ADB_SERVER_PORT:-}" ]]; then
	adb_base+=(-P "${ADB_SERVER_PORT}")
	export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-${ADB_SERVER_PORT}}"
fi

adb_cmd() {
	"${adb_base[@]}" "$@"
}

adb_device_cmd() {
	if [[ -n "${ANDROID_SERIAL:-}" ]]; then
		adb_cmd -s "${ANDROID_SERIAL}" "$@"
	else
		adb_cmd "$@"
	fi
}

wait_for_adb_device() {
	local wait_timeout="${1:-300}"
	local start_ts
	start_ts=$(date +%s)

	while true; do
		if adb_cmd devices | awk 'NR>1 && $2=="device" { found=1 } END { exit(found?0:1) }'; then
			return 0
		fi

		local now_ts
		now_ts=$(date +%s)
		if (( now_ts - start_ts > wait_timeout )); then
			echo "Timed out waiting for adb device after ${wait_timeout}s"
			adb_cmd devices -l || true
			return 1
		fi
		sleep 2
	done
}

connect_external_emulator() {
	echo "Using external emulator mode. Skipping in-container emulator startup."

	if [[ "${CATBENCH_ALLOW_ADB_KILL_SERVER:-0}" == "1" ]]; then
		adb_cmd kill-server || true
	fi
	adb_cmd start-server

	if [[ -n "${ADB_CONNECT_ADDR:-}" ]]; then
		echo "Connecting adb to ${ADB_CONNECT_ADDR}"
		adb_cmd connect "${ADB_CONNECT_ADDR}" || true
	fi

	wait_for_adb_device "${ADB_WAIT_TIMEOUT:-300}"
	if [[ -n "${ANDROID_SERIAL:-}" ]]; then
		target_serial="${ANDROID_SERIAL}"
	elif adb_cmd devices | awk 'NR>1 && $1=="emulator-5554" && $2=="device" {found=1} END {exit(found?0:1)}'; then
		target_serial="emulator-5554"
	elif [[ -n "${ADB_CONNECT_ADDR:-}" ]]; then
		target_serial="${ADB_CONNECT_ADDR}"
	else
		target_serial=""
	fi

	if [[ -n "${target_serial}" ]]; then
		export ANDROID_SERIAL="${target_serial}"
		echo "Using adb target serial: ${ANDROID_SERIAL}"
		adb_device_cmd root || true
	else
		adb_cmd root || true
	fi
	exec python3 -m server.android_server
}

if [[ "${EXTERNAL_EMULATOR:-0}" == "1" ]]; then
	connect_external_emulator
fi

if [[ "${DOCKER:-}" == "true" ]]; then
	export EMULATOR_GPU_MODE="${EMULATOR_GPU_MODE:-lavapipe}"
	export EMULATOR_SAFE_GPU_MODE="${EMULATOR_SAFE_GPU_MODE:-lavapipe}"
	export EMULATOR_SEGFAULT_GPU_FALLBACK="${EMULATOR_SEGFAULT_GPU_FALLBACK:-lavapipe}"

	if [[ ! -e /dev/kvm ]]; then
		echo "Warning: /dev/kvm is not available. For in-container emulator mode, run Docker with --privileged --device /dev/kvm."
		echo "Warning: If KVM cannot be exposed on this host, use EXTERNAL_EMULATOR=1 mode instead."
	fi
fi

# Start Emulator
#============================================
if ! ./docker_setup/start_emu_headless.sh; then
	if [[ "${EXTERNAL_EMULATOR_FALLBACK:-0}" == "1" ]]; then
		echo "In-container emulator failed. Falling back to external emulator mode."
		connect_external_emulator
	fi
	exit 1
fi
adb_device_cmd root || true
exec python3 -m server.android_server
