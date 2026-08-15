#!/usr/bin/env bash
set -euo pipefail

readonly GPU_PERCENT="${1:?usage: run_vllm_condition.sh GPU_PERCENT}"
if ! [[ "${GPU_PERCENT}" =~ ^[0-9]+$ ]] \
    || (( GPU_PERCENT < 1 || GPU_PERCENT > 100 )); then
  echo "GPU_PERCENT must be an integer from 1 to 100" >&2
  exit 2
fi

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly IMAGE="${RAGENT_VLLM_IMAGE:-stcc-vllm:0.22.0-audio}"
readonly MODEL_DIR="${RAGENT_MODEL_DIR:-/home/trinq3/models/stcc}"
readonly CACHE_DIR="${RAGENT_VLLM_CACHE_DIR:-/home/trinq3/.cache/vllm-stcc022}"
readonly CONTAINER_PREFIX="${RAGENT_VLLM_CONTAINER_PREFIX:-stcc-vllm022-p}"
readonly CONTAINER_NAME="${CONTAINER_PREFIX}${GPU_PERCENT}"
readonly START_SCRIPT="${RAGENT_VLLM_START_SCRIPT:-${SCRIPT_DIR}/start_vllm_condition.sh}"
readonly JETSON_DRIVER_LIB_DIR="${RAGENT_JETSON_DRIVER_LIB_DIR:-/opt/nvidia/l4t-gpu-libs/nvgpu}"
readonly VLLM_PORT="${RAGENT_VLLM_PORT:-8000}"
readonly VLLM_MODEL_NAME="${RAGENT_VLLM_MODEL_NAME:-stcc}"
readonly VLLM_MAX_MODEL_LEN="${RAGENT_VLLM_MAX_MODEL_LEN:-8192}"
readonly VLLM_GPU_MEMORY_UTILIZATION="${RAGENT_VLLM_GPU_MEMORY_UTILIZATION:-0.70}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container ${CONTAINER_NAME} already exists; stop it explicitly first." >&2
  exit 1
fi
if [[ ! -f "${JETSON_DRIVER_LIB_DIR}/libcuda.so.1.1" ]]; then
  echo "Jetson CUDA driver library was not found in ${JETSON_DRIVER_LIB_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/config.json" ]] \
    || [[ ! -f "${MODEL_DIR}/tools_openai.json" ]]; then
  echo "Standalone STCC checkpoint is incomplete: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${START_SCRIPT}" ]]; then
  echo "Container startup script was not found: ${START_SCRIPT}" >&2
  exit 1
fi

printf 'SERVER_START_ISO=%s\n' "$(date -Is)"
printf 'SERVER_START_EPOCH_NS=%s\n' "$(date +%s%N)"
mkdir -p "${CACHE_DIR}"

exec docker run --rm \
  --name "${CONTAINER_NAME}" \
  --device nvidia.com/gpu=all \
  --network=host \
  --ipc=host \
  --env "GPU_PERCENT=${GPU_PERCENT}" \
  --env "RAGENT_VLLM_PORT=${VLLM_PORT}" \
  --env "RAGENT_VLLM_MODEL_NAME=${VLLM_MODEL_NAME}" \
  --env "RAGENT_VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}" \
  --env "RAGENT_VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}" \
  --env "LD_LIBRARY_PATH=/host-nvgpu:/usr/local/cuda/lib64" \
  --volume "${MODEL_DIR}:/model:ro" \
  --volume "${CACHE_DIR}:/root/.cache/vllm" \
  --volume "${JETSON_DRIVER_LIB_DIR}:/host-nvgpu:ro" \
  --volume "${START_SCRIPT}:/start_vllm_condition.sh:ro" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  /start_vllm_condition.sh
