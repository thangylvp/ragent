#!/usr/bin/env bash
set -euo pipefail

# Runs inside the Jetson vLLM container. Values below 100 use CUDA MPS's
# active-thread percentage as an SM scheduling quota.
readonly GPU_PERCENT="${GPU_PERCENT:?set GPU_PERCENT to an integer from 1 to 100}"
readonly VLLM_PORT="${RAGENT_VLLM_PORT:-8000}"
readonly VLLM_MODEL_NAME="${RAGENT_VLLM_MODEL_NAME:-stcc}"
readonly VLLM_MAX_MODEL_LEN="${RAGENT_VLLM_MAX_MODEL_LEN:-8192}"
readonly VLLM_GPU_MEMORY_UTILIZATION="${RAGENT_VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
if ! [[ "${GPU_PERCENT}" =~ ^[0-9]+$ ]] \
    || (( GPU_PERCENT < 1 || GPU_PERCENT > 100 )); then
  echo "GPU_PERCENT must be an integer from 1 to 100" >&2
  exit 2
fi
if ! [[ "${VLLM_PORT}" =~ ^[0-9]+$ ]] \
    || (( VLLM_PORT < 1 || VLLM_PORT > 65535 )); then
  echo "RAGENT_VLLM_PORT must be an integer from 1 to 65535" >&2
  exit 2
fi
if ! [[ "${VLLM_MAX_MODEL_LEN}" =~ ^[0-9]+$ ]] \
    || (( VLLM_MAX_MODEL_LEN < 1 )); then
  echo "RAGENT_VLLM_MAX_MODEL_LEN must be a positive integer" >&2
  exit 2
fi
if ! [[ "${VLLM_GPU_MEMORY_UTILIZATION}" =~ ^0\.[0-9]*[1-9][0-9]*$|^1\.0+$ ]]; then
  echo "RAGENT_VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1]" >&2
  exit 2
fi

if (( GPU_PERCENT < 100 )); then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/stcc-mps/pipe
  export CUDA_MPS_LOG_DIRECTORY=/tmp/stcc-mps/log
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${GPU_PERCENT}"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
  nvidia-cuda-mps-control -d
  for _attempt in $(seq 1 30); do
    if printf 'get_default_active_thread_percentage\n' \
        | nvidia-cuda-mps-control >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done
  printf 'set_default_active_thread_percentage %s\n' "${GPU_PERCENT}" \
    | nvidia-cuda-mps-control
  printf 'MPS_DEFAULT_ACTIVE_THREAD_PERCENTAGE='
  printf 'get_default_active_thread_percentage\n' | nvidia-cuda-mps-control
else
  printf 'MPS_DISABLED_FULL_GPU=100\n'
fi

exec vllm serve /model \
  --host 0.0.0.0 \
  --port "${VLLM_PORT}" \
  --served-model-name "${VLLM_MODEL_NAME}" \
  --dtype bfloat16 \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --generation-config auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --limit-mm-per-prompt '{"audio":1}' \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
