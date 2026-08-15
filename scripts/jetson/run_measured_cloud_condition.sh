#!/usr/bin/env bash
set -euo pipefail

readonly GPU_PERCENT="${1:?usage: run_measured_cloud_condition.sh GPU_PERCENT [RUNS]}"
readonly RUNS="${2:-6}"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BENCH_DIR="${RAGENT_CLOUD_BENCH_DIR:-/home/trinq3/benchmarks/ragent_cloud}"
readonly RESULT_ROOT="${RAGENT_CLOUD_BENCH_RESULT_ROOT:-${BENCH_DIR}/measurements_20260814}"
readonly RESULT_DIR="${RESULT_ROOT}/p${GPU_PERCENT}"
readonly CONTAINER_PREFIX="${RAGENT_VLLM_CONTAINER_PREFIX:-stcc-vllm022-p}"
readonly CONTAINER_NAME="${CONTAINER_PREFIX}${GPU_PERCENT}"
readonly RUN_CLOUD="${RAGENT_RUN_CLOUD_SCRIPT:-${SCRIPT_DIR}/run_cloud_condition.sh}"

if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1 \
    || [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != true ]]; then
  echo "Expected running container ${CONTAINER_NAME} was not found" >&2
  exit 1
fi
mkdir -p "${RESULT_DIR}"
{
  printf 'captured_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'gpu_limit_percent=%s\n' "${GPU_PERCENT}"
  printf 'measured_cycles=%s\n' "${RUNS}"
  printf '\n[nvpmodel]\n'
  nvpmodel -q 2>&1 || true
  printf '\n[memory]\n'
  free -h
  printf '\n[container]\n'
  docker inspect -f 'name={{.Name}} image={{.Config.Image}} status={{.State.Status}} pid={{.State.Pid}}' "${CONTAINER_NAME}"
  printf '\n[processes]\n'
  ps -eo pid,ppid,pcpu,pmem,etimes,args --sort=-pcpu | head -n 30
} >"${RESULT_DIR}/system_before.txt"
docker inspect -f '{{json .Config.Cmd}}' "${CONTAINER_NAME}" >"${RESULT_DIR}/vllm_command.json"
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
  printf '%s\n' 'MPS disabled; direct full-GPU baseline.' >"${RESULT_DIR}/mps_verification.txt"
fi
tegrastats --interval 100 --logfile "${RESULT_DIR}/tegrastats.log" &
stats_pid=$!
if ! [[ "${stats_pid}" =~ ^[0-9]+$ ]]; then exit 1; fi
cleanup() {
  if kill -0 "${stats_pid}" 2>/dev/null; then
    kill "${stats_pid}"
    wait "${stats_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
status=0
"${RUN_CLOUD}" "${GPU_PERCENT}" "${RUNS}" \
  >"${RESULT_DIR}/benchmark.log" 2>&1 || status=$?
cleanup
trap - EXIT
{
  printf 'captured_at=%s\n' "$(date --iso-8601=seconds)"
  printf '\n[memory]\n'
  free -h
  printf '\n[container]\n'
  docker inspect -f 'name={{.Name}} image={{.Config.Image}} status={{.State.Status}} pid={{.State.Pid}}' "${CONTAINER_NAME}"
  printf '\n[processes]\n'
  ps -eo pid,ppid,pcpu,pmem,etimes,args --sort=-pcpu | head -n 30
} >"${RESULT_DIR}/system_after.txt"
exit "${status}"
