#!/bin/bash


# Credits to https://github.com/amrsa1/Android-Emulator-image

BL='\033[0;34m'
G='\033[0;32m'
RED='\033[0;31m'
YE='\033[1;33m'
NC='\033[0m' # No Color

emulator_name=${EMULATOR_NAME}
recovery_attempted=0
emulator_console_port="${EMULATOR_CONSOLE_PORT:-5554}"
emulator_adb_port="${EMULATOR_ADB_PORT:-$((emulator_console_port + 1))}"
emulator_grpc_port="${EMULATOR_GRPC_PORT:-8554}"
target_serial="${ANDROID_SERIAL:-emulator-${emulator_console_port}}"
emulator_camera_back_mode="${EMULATOR_CAMERA_BACK_MODE:-emulated}"
emulator_camera_front_mode="${EMULATOR_CAMERA_FRONT_MODE:-none}"

if [[ ! "${emulator_camera_back_mode}" =~ ^(emulated|virtualscene|videoplayback|webcam[0-9]+|none)$ ]]; then
    echo "Invalid EMULATOR_CAMERA_BACK_MODE: ${emulator_camera_back_mode}" >&2
    exit 64
fi
if [[ ! "${emulator_camera_front_mode}" =~ ^(emulated|webcam[0-9]+|none)$ ]]; then
    echo "Invalid EMULATOR_CAMERA_FRONT_MODE: ${emulator_camera_front_mode}" >&2
    exit 64
fi

adb_base=(adb)
if [[ -n "${ADB_SERVER_PORT:-}" ]]; then
    adb_base+=(-P "${ADB_SERVER_PORT}")
    export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-${ADB_SERVER_PORT}}"
fi

function adb_cmd() {
    "${adb_base[@]}" "$@"
}

function adb_device_cmd() {
    adb_cmd -s "${target_serial}" "$@"
}

function ensure_ipv6_loopback_resolution() {
    # Some Docker environments disable IPv6 entirely; qemu modem setup needs ::1.
    if command -v sysctl >/dev/null 2>&1; then
      sysctl -w net.ipv6.conf.all.disable_ipv6=0 >/dev/null 2>&1 || true
      sysctl -w net.ipv6.conf.default.disable_ipv6=0 >/dev/null 2>&1 || true
      sysctl -w net.ipv6.conf.lo.disable_ipv6=0 >/dev/null 2>&1 || true
    fi

    if command -v ip >/dev/null 2>&1; then
      ip link set lo up >/dev/null 2>&1 || true
      if ! ip -6 addr show dev lo | grep -q '::1/128'; then
        ip -6 addr add ::1/128 dev lo >/dev/null 2>&1 || true
      fi
    fi

    if [[ -w /etc/hosts ]] && ! grep -qE '^[[:space:]]*::1[[:space:]]' /etc/hosts; then
      echo "::1 localhost ip6-localhost ip6-loopback" >> /etc/hosts || true
    fi

    if ! getent hosts ::1 >/dev/null 2>&1; then
      echo "Warning: ::1 is still not resolvable inside container; emulator modem setup may fail."
    fi
}

function disable_gsm_modem_in_avd() {
    avd_home="${ANDROID_AVD_HOME:-${HOME}/.android/avd}"
    avd_config="${avd_home}/${emulator_name}.avd/config.ini"
    if [[ ! -f "$avd_config" ]]; then
      return 0
    fi

    sed -i '/^hw\.gsmModem=/d' "$avd_config"
    sed -i '/^hw\.gsm=/d' "$avd_config"
    echo "hw.gsmModem=no" >> "$avd_config"
    echo "hw.gsm=no" >> "$avd_config"
}

function apply_container_avd_workarounds() {
    requested_gpu_mode="${1:-off}"
    avd_home="${ANDROID_AVD_HOME:-${HOME}/.android/avd}"
    avd_config="${avd_home}/${emulator_name}.avd/config.ini"
    if [[ ! -f "$avd_config" ]]; then
      return 0
    fi

    sed -i '/^hw\.gpu\.enabled=/d' "$avd_config"
    sed -i '/^hw\.gpu\.mode=/d' "$avd_config"
    sed -i '/^hw\.gltransport=/d' "$avd_config"
    sed -i '/^hw\.audioInput=/d' "$avd_config"
    sed -i '/^hw\.audioOutput=/d' "$avd_config"
    sed -i '/^[[:space:]]*hw\.camera\.back[[:space:]]*=/d' "$avd_config"
    sed -i '/^[[:space:]]*hw\.camera\.front[[:space:]]*=/d' "$avd_config"
    sed -i '/^hw\.bluetooth=/d' "$avd_config"

    {
      if [[ "${requested_gpu_mode}" == "off" ]]; then
        echo "hw.gpu.enabled=no"
        echo "hw.gpu.mode=off"
      else
        echo "hw.gpu.enabled=yes"
        echo "hw.gpu.mode=${requested_gpu_mode}"
      fi
      echo "hw.gltransport=pipe"
      echo "hw.audioInput=no"
      echo "hw.audioOutput=no"
      echo "hw.camera.back=${emulator_camera_back_mode}"
      echo "hw.camera.front=${emulator_camera_front_mode}"
      echo "hw.bluetooth=no"
    } >> "$avd_config"
}

function emulator_process_running() {
    if pgrep -f "emulator .*@${emulator_name}" >/dev/null 2>&1; then
      return 0
    fi

    if pgrep -f "emulator .* -avd ${emulator_name}" >/dev/null 2>&1; then
      return 0
    fi

    # Recent emulator builds exec the wrapper into a binary whose command is
    # `/.../qemu-system-x86_64-headless @AVD_NAME ...`.  The path contains the
    # word `emulator` but not `emulator `, so the wrapper patterns above do not
    # match it.  Missing this process caused healthy Docker emulators to be
    # classified as dead during their initial offline/booting interval.
    if pgrep -f "qemu-system[^[:space:]]*[[:space:]]+@${emulator_name}([[:space:]]|$)" >/dev/null 2>&1; then
      return 0
    fi

    return 1
}

function stop_existing_emulator_processes() {
    if [[ "${CATBENCH_SKIP_EXISTING_EMULATOR_CLEANUP:-0}" == "1" ]]; then
      return 0
    fi
    if [[ "${CATBENCH_ALLOW_ADB_EMU_KILL:-0}" == "1" ]]; then
      adb_device_cmd emu kill || true
    fi
    pkill -f "emulator .*@${emulator_name}" >/dev/null 2>&1 || true
    pkill -f "emulator .* -avd ${emulator_name}" >/dev/null 2>&1 || true
    pkill -f "qemu-system[^[:space:]]*[[:space:]]+@${emulator_name}([[:space:]]|$)" >/dev/null 2>&1 || true
    if [[ "${CATBENCH_ALLOW_GLOBAL_QEMU_KILL:-0}" == "1" ]]; then
      pkill -f "qemu-system-x86_64-headless" >/dev/null 2>&1 || true
    fi

    for _ in $(seq 1 10); do
      if ! emulator_process_running; then
        return 0
      fi
      sleep 1
    done
}

function check_hardware_acceleration() {
    if [[ "$HW_ACCEL_OVERRIDE" != "" ]]; then
        hw_accel_flag="$HW_ACCEL_OVERRIDE"
    else
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS-specific hardware acceleration check
      HW_ACCEL_SUPPORT=$(sysctl -a | grep -E -c '(vmx|svm)')
      if [[ $HW_ACCEL_SUPPORT == 0 ]]; then
        hw_accel_flag="-accel off"
        echo "Warning: no accelerator found. Falling back to software acceleration." >&2
      else
        hw_accel_flag="-accel on"
      fi
    else
      # In containers, CPU virtualization flags alone are not enough.
      # KVM must be present and actually usable by the emulator.
      accel_check_output=""
      if command -v emulator >/dev/null 2>&1; then
        accel_check_output="$(emulator -accel-check 2>&1 || true)"
      fi

      if [[ -e /dev/kvm && -r /dev/kvm && -w /dev/kvm ]] \
        && echo "$accel_check_output" | grep -qi "installed and usable"; then
        hw_accel_flag="-accel on"
      else
        hw_accel_flag="-accel off"
        echo "Warning: emulator acceleration is unavailable. Falling back to software acceleration (-accel off)." >&2
        if [[ "$accel_check_output" != "" ]]; then
          echo "$accel_check_output" >&2
        fi
      fi
    fi
    fi

    echo "$hw_accel_flag"
}


hw_accel_flag=$(check_hardware_acceleration)

function default_linux_gpu_mode() {
  if [[ "${DOCKER:-}" == "true" ]]; then
    # lavapipe is more stable than -gpu off/swiftshader in many nested Docker/KVM hosts.
    echo "lavapipe"
  else
    echo "off"
  fi
}

function launch_emulator () {
  stop_existing_emulator_processes
  default_gpu_mode="$(default_linux_gpu_mode)"
  safe_hw_accel_flag="${EMULATOR_SAFE_ACCEL_OVERRIDE:-${hw_accel_flag}}"

  if [[ "${EMULATOR_SAFE_MODE:-0}" == "1" ]]; then
    options="@${emulator_name} -ports ${emulator_console_port},${emulator_adb_port} -no-window -no-snapshot -no-audio -no-metrics -no-boot-anim -memory 2048 -cores 1 -camera-back ${emulator_camera_back_mode} -camera-front ${emulator_camera_front_mode} -no-sim -engine auto ${safe_hw_accel_flag} -grpc ${emulator_grpc_port}"
    linux_gpu_mode="${EMULATOR_GPU_MODE:-${EMULATOR_SAFE_GPU_MODE:-${default_gpu_mode}}}"
  else
    options="@${emulator_name} -ports ${emulator_console_port},${emulator_adb_port} -no-window -no-snapshot -no-audio -no-metrics -no-boot-anim -memory 2048 -cores 1 -camera-back ${emulator_camera_back_mode} -camera-front ${emulator_camera_front_mode} -no-sim -engine auto ${hw_accel_flag} -grpc ${emulator_grpc_port}"
    linux_gpu_mode="${EMULATOR_GPU_MODE:-${default_gpu_mode}}"
  fi

  ensure_ipv6_loopback_resolution
  apply_container_avd_workarounds "${linux_gpu_mode}"

  function run_emulator_once() {
    : > nohup.out

    if [[ "$OSTYPE" == *linux* ]]; then
      echo "${OSTYPE}: emulator ${options} -gpu ${linux_gpu_mode}"
      nohup emulator $options -gpu ${linux_gpu_mode} &
    fi
    if [[ "$OSTYPE" == *darwin* ]] || [[ "$OSTYPE" == *macos* ]]; then
      echo "${OSTYPE}: emulator ${options} -gpu software"
      nohup emulator $options -gpu software &
    fi

    if [ $? -ne 0 ]; then
      echo "Error launching emulator"
      return 1
    fi

    # Wait briefly for either emulator wrapper or qemu backend process to appear.
    for _ in $(seq 1 10); do
      if emulator_process_running; then
        return 0
      fi
      sleep 1
    done

    if ! emulator_process_running; then
      echo "Error: emulator process exited early."
      if [[ -f nohup.out ]]; then
        echo "==== Last emulator logs ===="
        tail -n 80 nohup.out
        echo "============================"
      fi
      return 1
    fi
    return 0
  }

  if run_emulator_once; then
    return 0
  fi

  if [[ -f nohup.out ]] && grep -q "Unable to connect character device modem" nohup.out; then
    echo "Warning: modem socket initialization failed. Disabling GSM modem and retrying launch once."
    disable_gsm_modem_in_avd
    ensure_ipv6_loopback_resolution
    stop_existing_emulator_processes
    sleep 2
    run_emulator_once
    return $?
  fi

  if [[ -f nohup.out ]] && grep -q "Running multiple emulators with the same AVD" nohup.out; then
    echo "Warning: duplicate AVD lock detected. Retrying once with -read-only."
    stop_existing_emulator_processes
    sleep 2
    options="${options} -read-only"
    run_emulator_once
    return $?
  fi

  if [[ -f nohup.out ]] && grep -qE "libunwind: __unw_add_dynamic_fde|Segmentation fault" nohup.out; then
    echo "Warning: emulator native crash detected. Retrying once with safe fallback profile."
    stop_existing_emulator_processes
    sleep 2
    options="${options} -no-snapshot"
    linux_gpu_mode="${EMULATOR_SEGFAULT_GPU_FALLBACK:-${default_gpu_mode}}"
    export EMULATOR_GPU_MODE="${linux_gpu_mode}"
    apply_container_avd_workarounds "${linux_gpu_mode}"
    run_emulator_once
    return $?
  fi

  return 1
}


function check_emulator_status () {
  printf "${G}==> ${BL}Checking emulator booting up status 🧐${NC}\n"
  start_time=$(date +%s)
  spinner=( "⠹" "⠺" "⠼" "⠶" "⠦" "⠧" "⠇" "⠏" )
  i=0
  # Get the timeout value from the environment variable or use the default value of 300 seconds (5 minutes)
  timeout=${EMULATOR_TIMEOUT:-300}

  while true; do
    result=$(adb_device_cmd shell getprop sys.boot_completed 2>&1)

    if [ "$result" == "1" ]; then
      printf "\e[K${G}==> \u2713 Emulator is ready : '$result'           ${NC}\n"
      adb_cmd devices -l
      adb_device_cmd shell input keyevent 82
      return 0  # Return a 0 to indicate emulator has booted successfully
    elif [ "$result" == "" ]; then
      printf "${YE}==> Emulator is partially Booted! 😕 ${spinner[$i]} ${NC}\r"
    else
      printf "${RED}==> $result, please wait ${spinner[$i]} ${NC}\r"
      i=$(( (i+1) % 8 ))
    fi

    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))

    if ! emulator_process_running; then
      printf "${RED}==> Emulator process has exited unexpectedly.${NC}\n"
      if [[ -f nohup.out ]]; then
        echo "==== Last emulator logs ===="
        tail -n 80 nohup.out
        echo "============================"
      fi

      if [[ $recovery_attempted -eq 0 ]]; then
        recovery_attempted=1
        echo "Warning: emulator exited during boot; retrying once with safe launch profile."
        export EMULATOR_SAFE_MODE=1
        if [[ "${DOCKER:-}" == "true" ]]; then
          export EMULATOR_GPU_MODE="${EMULATOR_GPU_MODE:-lavapipe}"
        else
          export EMULATOR_GPU_MODE="${EMULATOR_GPU_MODE:-swiftshader_indirect}"
        fi

        if launch_emulator; then
          start_time=$(date +%s)
          continue
        fi
      fi

      return 1
    fi

    if [ $elapsed_time -gt $timeout ]; then
      printf "${RED}==> Timeout after ${timeout} seconds elapsed 🕛.. ${NC}\n"
      return 1 # Return a 1 to indicate failure if exceeded timeout
    fi
    sleep 4
  done
};


function ensure_root_adb() {
  # CATBench's deterministic state adapters read app-private SQLite files.
  # A boot-complete device is therefore not runner-ready until adbd has been
  # restarted as root and the exact worker serial proves uid 0.  This is
  # intentionally fail-closed: otherwise verifier storage reads can silently
  # degrade into false failures on a freshly cloned AVD.
  root_output="$(adb_device_cmd root 2>&1)" || {
    echo "Failed to restart adbd as root: ${root_output}" >&2
    return 1
  }
  if [[ -n "${root_output}" ]]; then
    echo "${root_output}"
  fi

  local attempt uid
  for attempt in $(seq 1 30); do
    uid="$(adb_device_cmd shell id -u 2>/dev/null | tr -d '\r' || true)"
    if [[ "${uid}" == "0" ]]; then
      echo "ADB_ROOT_READY serial=${target_serial} uid=${uid}"
      return 0
    fi
    sleep 1
  done
  echo "Timed out proving root adbd on ${target_serial}" >&2
  return 1
}


function disable_animation() {
  adb_device_cmd shell "settings put global window_animation_scale 0.0"
  adb_device_cmd shell "settings put global transition_animation_scale 0.0"
  adb_device_cmd shell "settings put global animator_duration_scale 0.0"
};

function hidden_policy() {
  adb_device_cmd shell "settings put global hidden_api_policy_pre_p_apps 1;settings put global hidden_api_policy_p_apps 1;settings put global hidden_api_policy 1"
};

launch_emulator
sleep 2

if check_emulator_status; then
  # Only run the below if the emulator is actually ready
  sleep 1
  ensure_root_adb || exit 1
  sleep 1
  disable_animation
  sleep 1
  hidden_policy
  sleep 1
else
  echo "Emulator failed to start properly, exiting..."
  exit 1
fi
