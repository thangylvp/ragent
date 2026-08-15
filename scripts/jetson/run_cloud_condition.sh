#!/usr/bin/env bash
set -euo pipefail

readonly GPU_PERCENT="${1:?usage: run_cloud_condition.sh GPU_PERCENT [RUNS]}"
readonly RUNS="${2:-6}"
readonly APP_DIR="${RAGENT_JETSON_APP_DIR:-/home/trinq3/ragent}"
readonly BENCH_DIR="${RAGENT_CLOUD_BENCH_DIR:-/home/trinq3/benchmarks/ragent_cloud}"
readonly RESULT_ROOT="${RAGENT_CLOUD_BENCH_RESULT_ROOT:-${BENCH_DIR}/measurements_20260814}"
readonly RESULT_DIR="${RESULT_ROOT}/p${GPU_PERCENT}"
readonly PYTHON="${RAGENT_JETSON_PYTHON:-/home/trinq3/venvs/qwen_asr_bench/bin/python}"
readonly MODEL_DIR="${RAGENT_MODEL_DIR:-/home/trinq3/models/stcc}"
readonly ENV_FILE="${RAGENT_JETSON_ENV:-${APP_DIR}/.env}"

if ! [[ "${GPU_PERCENT}" =~ ^[0-9]+$ ]] || (( GPU_PERCENT < 1 || GPU_PERCENT > 100 )); then
  echo "GPU_PERCENT must be an integer from 1 to 100" >&2
  exit 2
fi
if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || (( RUNS < 1 )); then
  echo "RUNS must be a positive integer" >&2
  exit 2
fi

mkdir -p "${RESULT_DIR}" "${BENCH_DIR}/captures" "${BENCH_DIR}/voice_cache/p${GPU_PERCENT}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Jetson Python is not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/tools_openai.json" ]]; then
  echo "Standalone STCC checkpoint is missing: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Cloud benchmark environment file is missing: ${ENV_FILE}" >&2
  exit 1
fi
cd "${APP_DIR}"
set -a
source "${ENV_FILE}"
set +a
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
export DEMO_CLOUD_ENABLED=1
export DEMO_CLOUD_PROVIDER=gemini
export DEMO_CLOUD_MODEL=gemini-3.6-flash
export DEMO_GEMINI_THINKING_LEVEL=low
export DEMO_CLOUD_HISTORY_TURNS=1
export DEMO_TTS_ENABLED=1
export DEMO_TTS_PROVIDER=omnivoice
export DEMO_OMNIVOICE_BASE_URL="${DEMO_OMNIVOICE_BASE_URL:-http://127.0.0.1:8120}"
export DEMO_OMNIVOICE_FORCE_SYNTHESIS=1
export DEMO_VOICE_CACHE_DIR="${BENCH_DIR}/voice_cache/p${GPU_PERCENT}"
export DEMO_STATIC_AUDIO_MANIFEST="${DEMO_STATIC_AUDIO_MANIFEST:-${APP_DIR}/outputs/demo/voice/manifest.json}"

corpus=(
  "${BENCH_DIR}/corpus/01_1.000s_news.wav"
  "${BENCH_DIR}/corpus/02_3.000s_news.wav"
  "${BENCH_DIR}/corpus/03_5.000s_news.wav"
  "${BENCH_DIR}/corpus/04_7.000s_news.wav"
  "${BENCH_DIR}/corpus/05_7.500s_news.wav"
)
if [[ "${RAGENT_CLOUD_BENCH_CORPUS:-full}" == compact ]]; then
  corpus=("${corpus[0]}" "${corpus[2]}" "${corpus[4]}")
elif [[ "${RAGENT_CLOUD_BENCH_CORPUS:-full}" != full ]]; then
  echo "RAGENT_CLOUD_BENCH_CORPUS must be full or compact" >&2
  exit 2
fi

exec "${PYTHON}" scripts/benchmark_jetson_cloud_path.py \
  --gpu-limit "${GPU_PERCENT}" \
  --runs "${RUNS}" \
  --warmup-cycles 1 \
  --variants-root "${BENCH_DIR}/variants" \
  --raw-jsonl "${RESULT_DIR}/runs.jsonl" \
  --summary-json "${RESULT_DIR}/summary.json" \
  "${corpus[@]}"
