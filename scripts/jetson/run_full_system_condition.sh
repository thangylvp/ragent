#!/usr/bin/env bash
set -euo pipefail

readonly GPU_PERCENT="${1:?usage: run_full_system_condition.sh GPU_PERCENT [RUNS]}"
readonly RUNS="${2:-8}"
readonly APP_DIR="${RAGENT_JETSON_APP_DIR:-/home/trinq3/ragent}"
readonly BENCH_DIR="${RAGENT_FULL_BENCH_DIR:-/home/trinq3/benchmarks/ragent_full}"
readonly RESULT_ROOT="${RAGENT_BENCH_RESULT_ROOT:-${BENCH_DIR}/measurements_20260814}"
readonly RESULT_DIR="${RESULT_ROOT}/p${GPU_PERCENT}"
readonly PYTHON="${RAGENT_JETSON_PYTHON:-/home/trinq3/venvs/qwen_asr_bench/bin/python}"
readonly MODEL_DIR="${RAGENT_MODEL_DIR:-/home/trinq3/models/stcc}"

if ! [[ "${GPU_PERCENT}" =~ ^[0-9]+$ ]] \
    || (( GPU_PERCENT < 1 || GPU_PERCENT > 100 )); then
  echo "GPU_PERCENT must be an integer from 1 to 100" >&2
  exit 2
fi
if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || (( RUNS < 2 )); then
  echo "RUNS must be an integer of at least 2" >&2
  exit 2
fi

mkdir -p "${RESULT_DIR}" "${BENCH_DIR}/captures"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Jetson Python is not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/tools_openai.json" ]]; then
  echo "Standalone STCC checkpoint is missing: ${MODEL_DIR}" >&2
  exit 1
fi
cd "${APP_DIR}"
export PYTHONPATH="${APP_DIR}/src:${APP_DIR}:${PYTHONPATH:-}"
export WEBTEST_MODEL_MODE=vllm
export WEBTEST_MODEL_DIR="${WEBTEST_MODEL_DIR:-${MODEL_DIR}}"
export WEBTEST_VLLM_BASE_URL="${WEBTEST_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export WEBTEST_VLLM_MODEL="${WEBTEST_VLLM_MODEL:-stcc}"
export WEBTEST_MAX_NEW_TOKENS="${WEBTEST_MAX_NEW_TOKENS:-128}"
export WEBTEST_VAD="${WEBTEST_VAD:-omnivad}"
export WEBTEST_ENHANCER="${WEBTEST_ENHANCER:-fastenhancer_s}"
export WEBTEST_FASTENHANCER_S_MODEL="${WEBTEST_FASTENHANCER_S_MODEL:-${RAGENT_FASTENHANCER_MODEL:-${APP_DIR}/outputs/denoise/models/fastenhancer_s_dns.onnx}}"
export WEBTEST_CAPTURE_DIR="${WEBTEST_CAPTURE_DIR:-${BENCH_DIR}/captures}"
export DEMO_VOICE_CACHE_DIR="${DEMO_VOICE_CACHE_DIR:-${APP_DIR}/outputs/demo/voice}"
export DEMO_STATIC_AUDIO_MANIFEST="${DEMO_STATIC_AUDIO_MANIFEST:-${APP_DIR}/outputs/demo/voice/manifest.json}"
export DEMO_CLOUD_ENABLED=0
export DEMO_TTS_ENABLED=0

exec "${PYTHON}" scripts/benchmark_jetson_full_system.py \
  --gpu-limit "${GPU_PERCENT}" \
  --runs "${RUNS}" \
  --warmup-cycles 1 \
  --variants-root "${BENCH_DIR}/final_variants" \
  --raw-jsonl "${RESULT_DIR}/runs.jsonl" \
  --summary-json "${RESULT_DIR}/summary.json" \
  "${BENCH_DIR}/final_corpus/01_1.39s_radio_source.wav" \
  "${BENCH_DIR}/final_corpus/02_2.00s_fog_lights.wav" \
  "${BENCH_DIR}/final_corpus/03_3.00s_seat_heating.wav" \
  "${BENCH_DIR}/final_corpus/04_4.00s_ambient_green.wav" \
  "${BENCH_DIR}/final_corpus/05_5.00s_ambient_rainbow.wav" \
  "${BENCH_DIR}/final_corpus/06_6.00s_ambient_rainbow.wav" \
  "${BENCH_DIR}/final_corpus/07_6.75s_next_station.wav" \
  "${BENCH_DIR}/final_corpus/08_7.44s_previous_station.wav"
