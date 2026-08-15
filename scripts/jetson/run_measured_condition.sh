#!/usr/bin/env bash
set -euo pipefail

readonly GPU_PERCENT="${1:?usage: run_measured_condition.sh GPU_PERCENT [RUNS]}"
readonly RUNS="${2:-8}"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BENCH_DIR="${RAGENT_FULL_BENCH_DIR:-/home/trinq3/benchmarks/ragent_full}"
readonly RESULT_ROOT="${RAGENT_BENCH_RESULT_ROOT:-${BENCH_DIR}/measurements_20260814}"
readonly RESULT_DIR="${RESULT_ROOT}/p${GPU_PERCENT}"
readonly CONTAINER_PREFIX="${RAGENT_VLLM_CONTAINER_PREFIX:-stcc-vllm022-p}"
readonly CONTAINER_NAME="${CONTAINER_PREFIX}${GPU_PERCENT}"
readonly RUN_FULL_SYSTEM="${RAGENT_RUN_FULL_SYSTEM_SCRIPT:-${SCRIPT_DIR}/run_full_system_condition.sh}"

if ! [[ "${GPU_PERCENT}" =~ ^[0-9]+$ ]] \
    || (( GPU_PERCENT < 1 || GPU_PERCENT > 100 )); then
  echo "GPU_PERCENT must be an integer from 1 to 100" >&2
  exit 2
fi
if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Expected running container ${CONTAINER_NAME} was not found" >&2
  exit 1
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != true ]]; then
  echo "Container ${CONTAINER_NAME} is not running" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"
{
  date -Is
  uname -a
  printf '\n--- memory ---\n'
  free -h
  printf '\n--- power mode ---\n'
  nvpmodel -q 2>&1 || true
  printf '\n--- containers ---\n'
  docker ps --no-trunc
  printf '\n--- benchmark container command ---\n'
  docker inspect -f '{{json .Config.Cmd}}' "${CONTAINER_NAME}"
  printf '\n--- processes by RSS ---\n'
  ps -eo pid,rss,comm,args --sort=-rss | head -20
} >"${RESULT_DIR}/system_before.txt"

if (( GPU_PERCENT < 100 )); then
  docker exec "${CONTAINER_NAME}" /bin/bash -lc \
    'set -euo pipefail
     export CUDA_MPS_PIPE_DIRECTORY=/tmp/stcc-mps/pipe
     printf "DEFAULT_ACTIVE_THREAD_PERCENTAGE="
     printf "get_default_active_thread_percentage\n" | nvidia-cuda-mps-control
     server_list="$(printf "get_server_list\n" | nvidia-cuda-mps-control)"
     printf "MPS_SERVER_LIST=%s\n" "${server_list}"
     printf "MPS_CLIENTS:\n"
     printf "ps\n" | nvidia-cuda-mps-control
     for server_pid in ${server_list}; do
       [[ "${server_pid}" =~ ^[0-9]+$ ]] || continue
       printf "SERVER_%s_ACTIVE_THREAD_PERCENTAGE=" "${server_pid}"
       printf "get_active_thread_percentage %s\n" "${server_pid}" | nvidia-cuda-mps-control
     done' \
    >"${RESULT_DIR}/mps_verification.txt"
else
  printf 'MPS disabled; direct full-GPU baseline.\n' \
    >"${RESULT_DIR}/mps_verification.txt"
fi

tegrastats --interval 100 --logfile "${RESULT_DIR}/tegrastats.log" &
stats_pid=$!
if ! [[ "${stats_pid}" =~ ^[0-9]+$ ]]; then
  echo "Invalid tegrastats PID" >&2
  exit 1
fi
cleanup() {
  if kill -0 "${stats_pid}" 2>/dev/null; then
    kill "${stats_pid}"
    wait "${stats_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export RAGENT_BENCH_RESULT_ROOT="${RESULT_ROOT}"
status=0
"${RUN_FULL_SYSTEM}" \
  "${GPU_PERCENT}" "${RUNS}" >"${RESULT_DIR}/benchmark.log" 2>&1 || status=$?

cleanup
trap - EXIT
{
  date -Is
  free -h
  ps -eo pid,rss,comm,args --sort=-rss | head -20
} >"${RESULT_DIR}/system_after.txt"
exit "${status}"
