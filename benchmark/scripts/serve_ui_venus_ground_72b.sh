#!/usr/bin/env bash
# Persistently host inclusionAI/UI-Venus-Ground-72B behind an OpenAI-compatible
# vLLM endpoint inside a detached tmux session, so a collaborator can call it
# from any client (e.g. benchmark/run_m3a_venus.py) without having to keep an
# interactive shell alive.
#
# Quick start:
#   ./benchmark/scripts/serve_ui_venus_ground_72b.sh start
#   ./benchmark/scripts/serve_ui_venus_ground_72b.sh status
#   ./benchmark/scripts/serve_ui_venus_ground_72b.sh logs       # tail -f
#   ./benchmark/scripts/serve_ui_venus_ground_72b.sh attach     # tmux attach
#   ./benchmark/scripts/serve_ui_venus_ground_72b.sh stop
#
# Once it is up, the collaborator points an OpenAI-compatible client at:
#   base_url = http://<this-host>:${PORT}/v1
#   model    = ${SERVED_MODEL_NAME}
#   api_key  = anything (vLLM ignores it)
#
# Tunables (override via env vars):
#   CONDA_ENV               conda env that has vllm installed (default: catbench311)
#   TMUX_SESSION            tmux session name              (default: ui-venus-gd)
#   MODEL_ID                HF model id to serve           (default: inclusionAI/UI-Venus-Ground-72B)
#   SERVED_MODEL_NAME       --served-model-name            (default: ui-venus-gd)
#   HOST / PORT             bind address                   (default: 0.0.0.0 / 8001)
#   TENSOR_PARALLEL_SIZE    GPUs to shard across           (default: 4)
#   GPU_MEMORY_UTILIZATION  vLLM memory headroom           (default: 0.90)
#   MAX_MODEL_LEN           context window                 (default: 16384)
#   LIMIT_MM_PER_PROMPT     multimodal cap (JSON)          (default: {"image":1})
#   CUDA_VISIBLE_DEVICES    GPUs to expose to vLLM         (default: 0,1,2,3)
#   LOG_DIR                 where the vLLM log goes        (default: <repo>/logs)
#   HF_HOME                 writable HF cache root         (default: <repo>/runs/cache/huggingface)
#   HUGGINGFACE_HUB_CACHE   writable HF hub cache          (default: $HF_HOME/hub)
#   TRANSFORMERS_CACHE      writable transformers cache    (default: $HUGGINGFACE_HUB_CACHE)
#   XDG_CACHE_HOME          writable generic cache root    (default: <repo>/runs/cache)
#   EXTRA_VLLM_ARGS         extra args appended to vllm    (default: empty)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-catbench311}"
TMUX_SESSION="${TMUX_SESSION:-ui-venus-gd}"
MODEL_ID="${MODEL_ID:-inclusionAI/UI-Venus-Ground-72B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ui-venus-gd}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":1}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
HF_HOME="${HF_HOME:-$REPO_ROOT/runs/cache/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HUGGINGFACE_HUB_CACHE}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/runs/cache}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

LOG_FILE="$LOG_DIR/${TMUX_SESSION}.log"

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found on PATH." >&2; exit 2; }
}

build_serve_cmd() {
  # Single-line command run inside the tmux pane. Using `conda run -n` keeps the
  # session reproducible regardless of the launcher's shell init.
  # `LD_LIBRARY_PATH` is prepended with the conda env's lib so vLLM picks up
  # the env's libstdc++ (the system one on some hosts is too old for the
  # CXXABI symbols required by conda-shipped libicui18n / sqlite3).
  local conda_env_prefix
  conda_env_prefix="$(conda info --base)/envs/${CONDA_ENV}"
  printf '%s' "\
export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'; \
export LD_LIBRARY_PATH='${conda_env_prefix}/lib:${LD_LIBRARY_PATH:-}'; \
export HF_HOME='${HF_HOME}'; \
export HUGGINGFACE_HUB_CACHE='${HUGGINGFACE_HUB_CACHE}'; \
export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}'; \
export XDG_CACHE_HOME='${XDG_CACHE_HOME}'; \
exec conda run --no-capture-output -n '${CONDA_ENV}' \
  vllm serve '${MODEL_ID}' \
    --host '${HOST}' \
    --port '${PORT}' \
    --served-model-name '${SERVED_MODEL_NAME}' \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    --trust-remote-code \
    --limit-mm-per-prompt '${LIMIT_MM_PER_PROMPT}' \
    ${EXTRA_VLLM_ARGS} \
  2>&1 | tee -a '${LOG_FILE}'"
}

cmd_start() {
  require tmux
  require conda

  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "tmux session '$TMUX_SESSION' already exists. Use 'attach', 'logs', or 'stop'."
    exit 0
  fi

  mkdir -p "$LOG_DIR"
  : >> "$LOG_FILE"

  local serve_cmd
  serve_cmd="$(build_serve_cmd)"

  echo "Launching tmux session '$TMUX_SESSION'"
  echo "  model      : $MODEL_ID  (served as '$SERVED_MODEL_NAME')"
  echo "  endpoint   : http://$HOST:$PORT/v1"
  echo "  GPUs       : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, TP=$TENSOR_PARALLEL_SIZE"
  echo "  conda env  : $CONDA_ENV"
  echo "  log file   : $LOG_FILE"
  echo "  HF cache   : $HUGGINGFACE_HUB_CACHE"

  tmux new-session -d -s "$TMUX_SESSION" -c "$REPO_ROOT" "$serve_cmd"

  echo
  echo "Started. The model takes a few minutes to load on first run."
  echo "Tail progress:   $0 logs"
  echo "Attach console:  $0 attach   (detach with Ctrl-b d)"
  echo "Stop server:     $0 stop"
}

cmd_stop() {
  require tmux
  if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "No tmux session '$TMUX_SESSION'."
    return 0
  fi
  echo "Killing tmux session '$TMUX_SESSION'."
  tmux kill-session -t "$TMUX_SESSION"
}

cmd_status() {
  require tmux
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "tmux session '$TMUX_SESSION' is RUNNING."
    tmux list-windows -t "$TMUX_SESSION"
  else
    echo "tmux session '$TMUX_SESSION' is NOT running."
  fi

  echo
  if command -v curl >/dev/null 2>&1; then
    echo "Health check (http://$HOST:$PORT/v1/models):"
    curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" \
      || echo "  endpoint not reachable yet"
  fi
}

cmd_logs() {
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "No log file at $LOG_FILE yet."
    exit 1
  fi
  exec tail -F "$LOG_FILE"
}

cmd_attach() {
  require tmux
  exec tmux attach -t "$TMUX_SESSION"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") {start|stop|status|logs|attach}

  start    Launch vLLM serving $MODEL_ID inside tmux session '$TMUX_SESSION'.
  stop     Kill the tmux session (terminates the vLLM server).
  status   Show whether the session is alive and probe /v1/models.
  logs     Tail -F the vLLM log file ($LOG_FILE).
  attach   Attach to the tmux session (detach with Ctrl-b d).

See the comment block at the top of this script for tunable env vars.
EOF
}

case "${1:-start}" in
  start)  cmd_start  ;;
  stop)   cmd_stop   ;;
  status) cmd_status ;;
  logs)   cmd_logs   ;;
  attach) cmd_attach ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac
