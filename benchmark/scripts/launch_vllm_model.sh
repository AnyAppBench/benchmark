#!/usr/bin/env bash
# Launch one OpenAI-compatible vLLM server, optionally split across GPUs.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a HuggingFace id or local path.}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_IDS="${GPU_IDS:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CATBENCH_RUNS_DIR:-$HOME/${USER}/catbench_runs}/model_servers}"
DTYPE="${DTYPE:-bfloat16}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CACHE_ROOT="${CATBENCH_HF_CACHE_ROOT:-$HOME/${USER}/hf_cache}"
export CATBENCH_HF_CACHE_ROOT="${CACHE_ROOT}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${CACHE_ROOT}/xdg_config}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${CACHE_ROOT}/vllm_config}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton/cache}"
export TRITON_OVERRIDE_DIR="${TRITON_OVERRIDE_DIR:-${CACHE_ROOT}/triton/override}"
export TRITON_DUMP_DIR="${TRITON_DUMP_DIR:-${CACHE_ROOT}/triton/dump}"
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-${CACHE_ROOT}/outlines}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${CACHE_ROOT}/flashinfer_workspace}"
export FLASHINFER_CUBIN_DIR="${FLASHINFER_CUBIN_DIR:-${FLASHINFER_WORKSPACE_BASE}/.cache/flashinfer/cubins}"
export FLASHINFER_DUMP_DIR="${FLASHINFER_DUMP_DIR:-${CACHE_ROOT}/flashinfer_dumps}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch_extensions}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${CACHE_ROOT}/numba_cache}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${CACHE_ROOT}/cupy_cache}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_DO_NOT_TRACK="${VLLM_DO_NOT_TRACK:-1}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${HF_HUB_CACHE}}"

mkdir -p \
  "${OUTPUT_ROOT}" \
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
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${FLASHINFER_CUBIN_DIR}" \
  "${FLASHINFER_DUMP_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${CUPY_CACHE_DIR}" \
  "${DOWNLOAD_DIR}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ -z "${TENSOR_PARALLEL_SIZE}" ]]; then
  TENSOR_PARALLEL_SIZE="${#GPU_ARRAY[@]}"
fi

LOG_PATH="${OUTPUT_ROOT}/${SERVED_MODEL_NAME//\//_}_${PORT}.log"

echo "Launching ${SERVED_MODEL_NAME}"
echo "  model: ${MODEL_PATH}"
echo "  GPUs: ${GPU_IDS}"
echo "  tensor_parallel_size: ${TENSOR_PARALLEL_SIZE}"
echo "  endpoint: http://${HOST}:${PORT}/v1"
echo "  log: ${LOG_PATH}"
echo "  HF_HOME: ${HF_HOME}"
echo "  HF_HUB_CACHE: ${HF_HUB_CACHE}"
echo "  HF_XET_CACHE: ${HF_XET_CACHE}"
echo "  XDG_CACHE_HOME: ${XDG_CACHE_HOME}"
echo "  XDG_CONFIG_HOME: ${XDG_CONFIG_HOME}"
echo "  TORCH_HOME: ${TORCH_HOME}"
echo "  TORCH_EXTENSIONS_DIR: ${TORCH_EXTENSIONS_DIR}"
echo "  VLLM_CACHE_ROOT: ${VLLM_CACHE_ROOT}"
echo "  VLLM_CONFIG_ROOT: ${VLLM_CONFIG_ROOT}"
echo "  TRITON_CACHE_DIR: ${TRITON_CACHE_DIR}"
echo "  FLASHINFER_WORKSPACE_BASE: ${FLASHINFER_WORKSPACE_BASE}"
echo "  FLASHINFER_CUBIN_DIR: ${FLASHINFER_CUBIN_DIR}"
echo "  VLLM_NO_USAGE_STATS: ${VLLM_NO_USAGE_STATS}"
echo "  download_dir: ${DOWNLOAD_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --download-dir "${DOWNLOAD_DIR}" \
  ${EXTRA_VLLM_ARGS} 2>&1 | tee "${LOG_PATH}"
