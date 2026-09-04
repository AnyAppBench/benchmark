#!/usr/bin/env bash
# Launch one coordinated CATBench five-category run for all 11 requested rows.
#
# Rows:
#   GUI-Owl-7B, MobileRL-9B, UI Voyager-4B,
#   Gemini-3-Pro, GPT-5.1, Claude Sonnet 4,
#   Gemini-3-Pro-dagger, GPT-5.1-dagger, Claude Sonnet 4-dagger,
#   Mobile-Agent-v3, AgentProg.

set -euo pipefail

REPO_ROOT="$HOME/AnyAppBench"
cd "${REPO_ROOT}"

PRESET_RESULT_ROOT="${RESULT_ROOT:-}"
PRESET_RUN_ID="${RUN_ID:-}"
PRESET_CATBENCH_EMULATORS="${CATBENCH_EMULATORS:-}"
PRESET_CATBENCH_CATEGORIES="${CATBENCH_CATEGORIES:-}"
PRESET_CATBENCH_TASK_REGEX="${CATBENCH_TASK_REGEX:-}"
PRESET_CATBENCH_APP_IDS="${CATBENCH_APP_IDS:-}"
PRESET_GROUNDER_URL="${UI_VENUS_72B_GROUNDER_URL:-}"
PRESET_GROUNDER_MODEL="${UI_VENUS_72B_GROUNDER_MODEL:-}"
PRESET_GROUNDER_AUTH="${UI_VENUS_GROUNDER_AUTHORIZATION:-}"
PRESET_GROUNDER_NGROK="${UI_VENUS_GROUNDER_NGROK_SKIP_WARNING:-}"
PRESET_CATBENCH_HF_CACHE_ROOT="${CATBENCH_HF_CACHE_ROOT:-}"

set -a
source benchmark/configs/catbench.env
set +a

restore_preset() {
  local name="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    export "${name}=${value}"
  fi
}

restore_preset RESULT_ROOT "${PRESET_RESULT_ROOT}"
restore_preset RUN_ID "${PRESET_RUN_ID}"
restore_preset CATBENCH_EMULATORS "${PRESET_CATBENCH_EMULATORS}"
restore_preset CATBENCH_CATEGORIES "${PRESET_CATBENCH_CATEGORIES}"
restore_preset CATBENCH_TASK_REGEX "${PRESET_CATBENCH_TASK_REGEX}"
restore_preset CATBENCH_APP_IDS "${PRESET_CATBENCH_APP_IDS}"
restore_preset UI_VENUS_72B_GROUNDER_URL "${PRESET_GROUNDER_URL}"
restore_preset UI_VENUS_72B_GROUNDER_MODEL "${PRESET_GROUNDER_MODEL}"
restore_preset UI_VENUS_GROUNDER_AUTHORIZATION "${PRESET_GROUNDER_AUTH}"
restore_preset UI_VENUS_GROUNDER_NGROK_SKIP_WARNING "${PRESET_GROUNDER_NGROK}"
restore_preset CATBENCH_HF_CACHE_ROOT "${PRESET_CATBENCH_HF_CACHE_ROOT}"

DEFAULT_CATBENCH_EMULATORS="5556:8555,5558:8556,5560:8557,5562:8558,5564:8559,5566:8560,5568:8561,5570:8562,5572:8563,5574:8564"
if [[ -z "${PRESET_CATBENCH_EMULATORS}" ]]; then
  export CATBENCH_EMULATORS="${CATBENCH_5CAT_EMULATORS:-${DEFAULT_CATBENCH_EMULATORS}}"
fi

RESULT_ROOT="${RESULT_ROOT:-$HOME/anyappbench_results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_5cat_all11}"
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_ROOT}/logs"
REPORT_DIR="${RUN_ROOT}/reports"
SERVER_LOG_DIR="${RUN_ROOT}/model_servers"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-python3}"
CATBENCH_RECORD_VIDEOS="${CATBENCH_RECORD_VIDEOS:-0}"
CATBENCH_VIDEO_SEGMENT_SECONDS="${CATBENCH_VIDEO_SEGMENT_SECONDS:-170}"
CATBENCH_VIDEO_BIT_RATE="${CATBENCH_VIDEO_BIT_RATE:-4M}"
CATBENCH_VIDEO_SIZE="${CATBENCH_VIDEO_SIZE:-}"
CATBENCH_JOB_TIMEOUT_SECONDS="${CATBENCH_JOB_TIMEOUT_SECONDS:-14400}"
if [[ "${CATBENCH_RECORD_VIDEOS}" == "1" ]]; then
  CATBENCH_PRELAUNCH_DELAY_SECONDS="${CATBENCH_PRELAUNCH_DELAY_SECONDS:-5}"
else
  CATBENCH_PRELAUNCH_DELAY_SECONDS="${CATBENCH_PRELAUNCH_DELAY_SECONDS:-0}"
fi

export CATBENCH_RUNS_DIR="${RUN_ROOT}/runs"
export CATBENCH_HF_CACHE_ROOT="${CATBENCH_HF_CACHE_ROOT:-$HOME/${USER}/hf_cache}"
export XDG_CACHE_HOME="${CATBENCH_HF_CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${CATBENCH_HF_CACHE_ROOT}/xdg_config"
export HF_HOME="${CATBENCH_HF_CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_XET_CACHE="${HF_HOME}/xet"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${CATBENCH_HF_CACHE_ROOT}/torch"
export VLLM_CACHE_ROOT="${CATBENCH_HF_CACHE_ROOT}/vllm"
export VLLM_CONFIG_ROOT="${CATBENCH_HF_CACHE_ROOT}/vllm_config"
export TRITON_CACHE_DIR="${CATBENCH_HF_CACHE_ROOT}/triton/cache"
export TRITON_OVERRIDE_DIR="${CATBENCH_HF_CACHE_ROOT}/triton/override"
export TRITON_DUMP_DIR="${CATBENCH_HF_CACHE_ROOT}/triton/dump"
export OUTLINES_CACHE_DIR="${CATBENCH_HF_CACHE_ROOT}/outlines"
export VLLM_DOWNLOAD_DIR="${HF_HUB_CACHE}"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1

export UI_VOYAGER_URL="${UI_VOYAGER_URL:-http://127.0.0.1:8000/v1}"
export UI_VOYAGER_MODEL="${CATBENCH_UI_VOYAGER_MODEL:-UI-Voyager}"
export MOBILERL_URL="${MOBILERL_URL:-http://127.0.0.1:8001/v1}"
export MOBILERL_MODEL="${CATBENCH_MOBILERL_MODEL:-xuyifan/MobileRL-9B}"
export GUI_OWL_URL="${GUI_OWL_URL:-http://127.0.0.1:8002/v1}"
export GUI_OWL_MODEL="${CATBENCH_GUI_OWL_MODEL:-GUI-Owl-7B}"
export MOBILE_AGENT_V3_URL="${MOBILE_AGENT_V3_URL:-http://127.0.0.1:8002/v1}"
export MOBILE_AGENT_V3_MODEL="${CATBENCH_MOBILE_AGENT_V3_MODEL:-GUI-Owl-7B}"
export AGENTPROG_ROOT="${AGENTPROG_ROOT:-$HOME/AgentProg}"
export AGENTPROG_MODEL="${CATBENCH_AGENTPROG_MODEL:-gemini/gemini-3.1-pro-preview}"
export AGENTPROG_UI_TARS_MODEL="${CATBENCH_AGENTPROG_UI_TARS_MODEL:-ui-tars-1.5-7b}"
export AGENTPROG_UI_TARS_BASE_URL="${AGENTPROG_UI_TARS_BASE_URL:-http://127.0.0.1:8003/v1}"
export AGENTPROG_UI_TARS_API_KEY="${AGENTPROG_UI_TARS_API_KEY:-EMPTY}"
export UI_VENUS_72B_GROUNDER_MODEL="${UI_VENUS_72B_GROUNDER_MODEL:-ui-venus-gd}"
if [[ -z "${AUTODEV_BASE_URL:-}" && -n "${OPENAI_BASE_URL:-}" ]]; then
  export AUTODEV_BASE_URL="${OPENAI_BASE_URL}"
fi

MODEL_LIST="${CATBENCH_MODEL_LIST:-GUI-Owl-7B,MobileRL-9B,UI Voyager-4B,Gemini-3-Pro,GPT-5.1,Claude Sonnet 4,Gemini-3-Pro-dagger,GPT-5.1-dagger,Claude Sonnet 4-dagger,Mobile-Agent-v3,AgentProg}"
CATBENCH_CATEGORIES="${CATBENCH_CATEGORIES:-sms,files,contacts,maps,clock}"
CATBENCH_TASK_REGEX="${CATBENCH_TASK_REGEX:-}"
CATBENCH_APP_IDS="${CATBENCH_APP_IDS:-}"
MOBILERL_LOCAL_MODEL_DEFAULT="/tmp/mobilerl_complete_snapshot"
if [[ ! -e "${MOBILERL_LOCAL_MODEL_DEFAULT}" ]]; then
  MOBILERL_LOCAL_MODEL_DEFAULT="${MOBILERL_MODEL}"
fi

mkdir -p \
  "${LOG_DIR}" \
  "${REPORT_DIR}" \
  "${SERVER_LOG_DIR}" \
  "${CATBENCH_RUNS_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${HF_XET_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${VLLM_CACHE_ROOT}" \
  "${VLLM_CONFIG_ROOT}" \
  "${TRITON_CACHE_DIR}" \
  "${TRITON_OVERRIDE_DIR}" \
  "${TRITON_DUMP_DIR}" \
  "${OUTLINES_CACHE_DIR}" \
  "${VLLM_DOWNLOAD_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

endpoint_base() {
  local url="${1%/}"
  if [[ "${url}" != */v1 ]]; then
    url="${url}/v1"
  fi
  printf '%s' "${url}"
}

wait_for_openai_endpoint() {
  local name="$1"
  local url="$2"
  local attempts="${CATBENCH_DEPENDENCY_WAIT_ATTEMPTS:-720}"
  local sleep_sec="${CATBENCH_DEPENDENCY_WAIT_SECONDS:-10}"
  local base
  base="$(endpoint_base "${url}")"
  log "Waiting for ${name}: ${base}/models"
  for ((i = 1; i <= attempts; i++)); do
    if curl -fsS "${base}/models" >/dev/null 2>&1; then
      log "${name} is ready"
      return 0
    fi
    sleep "${sleep_sec}"
  done
  log "ERROR: ${name} did not become ready"
  return 1
}

configure_grounder_auth() {
  local credentials_file="${UI_VENUS_GROUNDER_CREDENTIALS_FILE:-logs/ngrok-credentials.txt}"
  local basic_auth="${UI_VENUS_GROUNDER_BASIC_AUTH:-}"
  if [[ -z "${UI_VENUS_GROUNDER_AUTHORIZATION:-}" ]]; then
    if [[ -z "${basic_auth}" && -f "${credentials_file}" ]]; then
      basic_auth="$(
        awk 'NF && $1 !~ /^#/ {print; exit}' "${credentials_file}" | tr -d '\r\n'
      )"
    fi
    if [[ -n "${basic_auth}" ]]; then
      export UI_VENUS_GROUNDER_AUTHORIZATION="Basic $(printf '%s' "${basic_auth}" | base64 | tr -d '\n')"
      log "Configured UI-Venus grounder basic auth"
    fi
  fi
  export UI_VENUS_GROUNDER_NGROK_SKIP_WARNING="${UI_VENUS_GROUNDER_NGROK_SKIP_WARNING:-true}"
}

model_selected() {
  local needle="$1"
  local model
  IFS=',' read -r -a selected_models <<< "${MODEL_LIST}"
  for model in "${selected_models[@]}"; do
    model="${model#"${model%%[![:space:]]*}"}"
    model="${model%"${model##*[![:space:]]}"}"
    if [[ "${model}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

needs_ui_voyager() {
  model_selected "UI Voyager-4B"
}

needs_mobilerl() {
  model_selected "MobileRL-9B"
}

needs_gui_owl() {
  model_selected "GUI-Owl-7B" || model_selected "Mobile-Agent-v3"
}

needs_agentprog_uitars() {
  model_selected "AgentProg"
}

needs_grounder() {
  model_selected "Gemini-3-Pro-dagger" \
    || model_selected "GPT-5.1-dagger" \
    || model_selected "Claude Sonnet 4-dagger"
}

wait_for_grounder() {
  local attempts="${CATBENCH_DEPENDENCY_WAIT_ATTEMPTS:-720}"
  local sleep_sec="${CATBENCH_DEPENDENCY_WAIT_SECONDS:-10}"
  local base
  if [[ -z "${UI_VENUS_72B_GROUNDER_URL:-}" ]]; then
    log "ERROR: UI_VENUS_72B_GROUNDER_URL is empty"
    return 1
  fi
  base="$(endpoint_base "${UI_VENUS_72B_GROUNDER_URL}")"
  log "Waiting for UI-Venus grounder: ${base}/models"
  for ((i = 1; i <= attempts; i++)); do
    local args=(-fsS)
    if [[ -n "${UI_VENUS_GROUNDER_AUTHORIZATION:-}" ]]; then
      args+=(-H "Authorization: ${UI_VENUS_GROUNDER_AUTHORIZATION}")
    fi
    if [[ "${UI_VENUS_GROUNDER_NGROK_SKIP_WARNING:-}" == "true" ]]; then
      args+=(-H "ngrok-skip-browser-warning: true")
    fi
    if curl "${args[@]}" "${base}/models" >/dev/null 2>&1; then
      log "UI-Venus grounder is ready"
      return 0
    fi
    sleep "${sleep_sec}"
  done
  log "ERROR: UI-Venus grounder did not become ready"
  return 1
}

wait_for_emulators() {
  local attempts="${CATBENCH_EMULATOR_WAIT_ATTEMPTS:-180}"
  local sleep_sec="${CATBENCH_EMULATOR_WAIT_SECONDS:-5}"
  log "Waiting for emulators: ${CATBENCH_EMULATORS}"
  for ((i = 1; i <= attempts; i++)); do
    local all_ready=1
    IFS=',' read -r -a items <<< "${CATBENCH_EMULATORS}"
    for item in "${items[@]}"; do
      local console_port="${item%%:*}"
      local serial="emulator-${console_port}"
      if [[ "$(adb -s "${serial}" get-state 2>/dev/null || true)" != "device" ]]; then
        all_ready=0
        break
      fi
    done
    if [[ "${all_ready}" == "1" ]]; then
      log "All target emulators are ready"
      return 0
    fi
    sleep "${sleep_sec}"
  done
  log "ERROR: target emulators did not become ready"
  return 1
}

start_vllm_if_needed() {
  local name="$1"
  local model_path="$2"
  local served_name="$3"
  local endpoint_url="$4"
  local gpu_ids="$5"
  local max_model_len="$6"
  local gpu_memory_utilization="$7"
  local extra_args="$8"
  local base
  local port

  base="$(endpoint_base "${endpoint_url}")"
  port="$("${PYTHON_BIN}" -c "import urllib.parse; print(urllib.parse.urlparse('${base}').port or 8000)")"
  if curl -fsS "${base}/models" >/dev/null 2>&1; then
    log "${name} endpoint already running at ${base}"
    return 0
  fi

  log "Starting ${name}: model=${model_path}, served=${served_name}, gpu=${gpu_ids}, port=${port}"
  (
    PYTHON_BIN="${VLLM_PYTHON_BIN}" \
    MODEL_PATH="${model_path}" \
    SERVED_MODEL_NAME="${served_name}" \
    GPU_IDS="${gpu_ids}" \
    PORT="${port}" \
    MAX_MODEL_LEN="${max_model_len}" \
    GPU_MEMORY_UTILIZATION="${gpu_memory_utilization}" \
    OUTPUT_ROOT="${SERVER_LOG_DIR}" \
    EXTRA_VLLM_ARGS="${extra_args}" \
    CATBENCH_HF_CACHE_ROOT="${CATBENCH_HF_CACHE_ROOT}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    XDG_CONFIG_HOME="${XDG_CONFIG_HOME}" \
    HF_HOME="${HF_HOME}" \
    HF_HUB_CACHE="${HF_HUB_CACHE}" \
    HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
    HF_XET_CACHE="${HF_XET_CACHE}" \
    TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
    TORCH_HOME="${TORCH_HOME}" \
    VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}" \
    VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT}" \
    TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
    TRITON_OVERRIDE_DIR="${TRITON_OVERRIDE_DIR}" \
    TRITON_DUMP_DIR="${TRITON_DUMP_DIR}" \
    OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR}" \
    DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR}" \
    VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS}" \
    VLLM_DO_NOT_TRACK="${VLLM_DO_NOT_TRACK}" \
    DO_NOT_TRACK="${DO_NOT_TRACK}" \
      bash benchmark/scripts/launch_vllm_model.sh
  ) >>"${LOG_DIR}/${name}.vllm.outer.log" 2>&1 &
  echo "$!" >"${LOG_DIR}/${name}.vllm.pid"
}

start_live_reporter() {
  (
    for _ in $(seq 1 120); do
      if [[ -f "${MANIFEST}" ]]; then
        "${PYTHON_BIN}" benchmark/scripts/live_catbench_markdown_report.py \
          --manifest "${MANIFEST}" \
          --model_config benchmark/configs/catbench_5cat_models.json \
          --out "${REPO_ROOT}/markdown_all_models.md" \
          --artifact_dir "${REPO_ROOT}/catbench_live_artifacts_all_models" \
          --recent 20 \
          --interval 30 \
          --watch
        exit $?
      fi
      sleep 2
    done
    log "WARNING: live report did not start because manifest was not created: ${MANIFEST}"
  ) >>"${LOG_DIR}/live_report.log" 2>&1 &
  echo "$!" >"${LOG_DIR}/live_report.pid"
  log "Live Markdown report watcher PID: $(cat "${LOG_DIR}/live_report.pid")"
}

stop_live_reporter() {
  if [[ ! -f "${LOG_DIR}/live_report.pid" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${LOG_DIR}/live_report.pid")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
  fi
}

start_video_recorder() {
  if [[ "${CATBENCH_RECORD_VIDEOS}" != "1" ]]; then
    return 0
  fi
  (
    for _ in $(seq 1 120); do
      if [[ -f "${MANIFEST}" ]]; then
        "${PYTHON_BIN}" benchmark/scripts/record_catbench_videos.py \
          --manifest "${MANIFEST}" \
          --matrix_log "${MATRIX_LOG}" \
          --run_root "${RUN_ROOT}" \
          --segment_seconds "${CATBENCH_VIDEO_SEGMENT_SECONDS}" \
          --bit_rate "${CATBENCH_VIDEO_BIT_RATE}" \
          --size "${CATBENCH_VIDEO_SIZE}"
        exit $?
      fi
      sleep 2
    done
    log "WARNING: video recorder did not start because manifest was not created: ${MANIFEST}"
  ) >>"${LOG_DIR}/video_recorder.launch.log" 2>&1 &
  echo "$!" >"${LOG_DIR}/video_recorder.pid"
  log "Video recorder watcher PID: $(cat "${LOG_DIR}/video_recorder.pid")"
}

stop_video_recorder() {
  if [[ ! -f "${LOG_DIR}/video_recorder.pid" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${LOG_DIR}/video_recorder.pid")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill -INT "${pid}" >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do
      if ! kill -0 "${pid}" >/dev/null 2>&1; then
        return 0
      fi
      sleep 1
    done
    kill "${pid}" >/dev/null 2>&1 || true
  fi
}

cleanup_watchers() {
  stop_video_recorder
  stop_live_reporter
}

trap cleanup_watchers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

MATRIX_LOG="${LOG_DIR}/catbench_matrix.log"
MANIFEST="${RUN_ROOT}/matrix/${RUN_ID}/catbench_5cat_manifest.json"

log "CATBench all-11 launcher"
log "Host: $(hostname)"
log "Run root: ${RUN_ROOT}"
log "Python: ${PYTHON_BIN}"
log "vLLM Python: ${VLLM_PYTHON_BIN}"
log "Emulators: ${CATBENCH_EMULATORS}"
log "Models: ${MODEL_LIST}"
log "Categories: ${CATBENCH_CATEGORIES}"
if [[ -n "${CATBENCH_TASK_REGEX}" ]]; then
  log "Task regex: ${CATBENCH_TASK_REGEX}"
fi
if [[ -n "${CATBENCH_APP_IDS}" ]]; then
  log "App IDs: ${CATBENCH_APP_IDS}"
fi
log "Per-job timeout: ${CATBENCH_JOB_TIMEOUT_SECONDS}s"
configure_grounder_auth

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | tee "${LOG_DIR}/gpu_snapshot.txt"
adb devices | tee "${LOG_DIR}/adb_devices.txt"

wait_for_emulators

if needs_ui_voyager; then
  start_vllm_if_needed \
    "ui_voyager_4b" \
    "${UI_VOYAGER_LOCAL_MODEL:-MarsXL/UI-Voyager}" \
    "${UI_VOYAGER_MODEL}" \
    "${UI_VOYAGER_URL}" \
    "${UI_VOYAGER_GPU_IDS:-0}" \
    "${UI_VOYAGER_MAX_MODEL_LEN:-196608}" \
    "${UI_VOYAGER_GPU_MEMORY_UTILIZATION:-0.90}" \
    '--limit-mm-per-prompt {"image":1}'
  wait_for_openai_endpoint "ui_voyager_4b" "${UI_VOYAGER_URL}"
fi

if needs_mobilerl; then
  start_vllm_if_needed \
    "mobilerl_9b" \
    "${MOBILERL_LOCAL_MODEL:-${MOBILERL_LOCAL_MODEL_DEFAULT}}" \
    "${MOBILERL_MODEL}" \
    "${MOBILERL_URL}" \
    "${MOBILERL_GPU_IDS:-1}" \
    "${MOBILERL_MAX_MODEL_LEN:-32768}" \
    "${MOBILERL_GPU_MEMORY_UTILIZATION:-0.85}" \
    '--limit-mm-per-prompt {"image":2}'
  wait_for_openai_endpoint "mobilerl_9b" "${MOBILERL_URL}"
fi

if needs_gui_owl; then
  start_vllm_if_needed \
    "gui_owl_7b" \
    "${GUI_OWL_LOCAL_MODEL:-mPLUG/GUI-Owl-7B}" \
    "${GUI_OWL_MODEL}" \
    "${GUI_OWL_URL}" \
    "${GUI_OWL_GPU_IDS:-2}" \
    "${GUI_OWL_MAX_MODEL_LEN:-8192}" \
    "${GUI_OWL_GPU_MEMORY_UTILIZATION:-0.75}" \
    '--limit-mm-per-prompt {"image":2} --allowed-local-media-path / --enforce-eager'
  wait_for_openai_endpoint "gui_owl_7b" "${GUI_OWL_URL}"
fi

if needs_agentprog_uitars; then
  start_vllm_if_needed \
    "agentprog_uitars15" \
    "${AGENTPROG_UI_TARS_LOCAL_MODEL:-ByteDance-Seed/UI-TARS-1.5-7B}" \
    "${AGENTPROG_UI_TARS_MODEL}" \
    "${AGENTPROG_UI_TARS_BASE_URL}" \
    "${AGENTPROG_UI_TARS_GPU_IDS:-3}" \
    "${AGENTPROG_UI_TARS_MAX_MODEL_LEN:-8192}" \
    "${AGENTPROG_UI_TARS_GPU_MEMORY_UTILIZATION:-0.75}" \
    '--limit-mm-per-prompt {"image":1} --allowed-local-media-path / --enforce-eager'
  wait_for_openai_endpoint "agentprog_uitars15" "${AGENTPROG_UI_TARS_BASE_URL}"
fi

if needs_grounder; then
  wait_for_grounder
fi

log "Starting CATBench all-11 matrix. Log: ${MATRIX_LOG}"
start_live_reporter
start_video_recorder
MATRIX_RESUME_ARGS=()
if [[ "${CATBENCH_RESUME_EXISTING:-0}" == "1" ]]; then
  MATRIX_RESUME_ARGS=(--resume_existing)
fi
MATRIX_FILTER_ARGS=()
if [[ -n "${CATBENCH_TASK_REGEX}" ]]; then
  MATRIX_FILTER_ARGS+=(--task_regex "${CATBENCH_TASK_REGEX}")
fi
if [[ -n "${CATBENCH_APP_IDS}" ]]; then
  MATRIX_FILTER_ARGS+=(--app_ids "${CATBENCH_APP_IDS}")
fi
set +e
"${PYTHON_BIN}" benchmark/scripts/run_catbench_5cat_matrix.py \
  --model_config benchmark/configs/catbench_5cat_models.json \
  --models "${MODEL_LIST}" \
  --categories "${CATBENCH_CATEGORIES}" \
  --emulators "${CATBENCH_EMULATORS}" \
  --output_root "${RUN_ROOT}/matrix" \
  --run_id "${RUN_ID}" \
  --prelaunch_delay_seconds "${CATBENCH_PRELAUNCH_DELAY_SECONDS}" \
  --launch_stagger_seconds "${CATBENCH_LAUNCH_STAGGER_SECONDS:-3}" \
  --job_timeout_seconds "${CATBENCH_JOB_TIMEOUT_SECONDS}" \
  --continue_on_error \
  "${MATRIX_RESUME_ARGS[@]}" \
  "${MATRIX_FILTER_ARGS[@]}" \
  >"${MATRIX_LOG}" 2>&1
MATRIX_STATUS=$?
set -e

log "Matrix finished with exit=${MATRIX_STATUS}. Manifest: ${MANIFEST}"
log "Generating CATBench all-11 report into ${REPORT_DIR}"
set +e
"${PYTHON_BIN}" benchmark/scripts/report_catbench_5cat_results.py \
  --manifest "${MANIFEST}" \
  --model_config benchmark/configs/catbench_5cat_models.json \
  --out_dir "${REPORT_DIR}" \
  >"${LOG_DIR}/report.log" 2>&1
REPORT_STATUS=$?
set -e

"${PYTHON_BIN}" benchmark/scripts/live_catbench_markdown_report.py \
  --manifest "${MANIFEST}" \
  --model_config benchmark/configs/catbench_5cat_models.json \
  --out "${REPO_ROOT}/markdown_all_models.md" \
  --artifact_dir "${REPO_ROOT}/catbench_live_artifacts_all_models" \
  --recent 20 \
  >>"${LOG_DIR}/live_report.log" 2>&1 || true
stop_video_recorder
stop_live_reporter

log "Report generation exit=${REPORT_STATUS}"
log "Done. Summary text: ${REPORT_DIR}/main_cross_app_gap.txt"
exit "${MATRIX_STATUS}"
