#!/usr/bin/env bash
set -euo pipefail

# Run the complete edge backend on Jetson. The browser reaches this loopback
# listener through demo/connect_jetson.sh; only dynamic cloud replies cross the
# reverse TTS forward. Fixed tool feedback stays in the Jetson audio bundle.
readonly APP_DIR="${RAGENT_JETSON_APP_DIR:-/home/trinq3/ragent}"
readonly PYTHON="${RAGENT_JETSON_PYTHON:-/home/trinq3/venvs/qwen_asr_bench/bin/python}"
readonly MODEL_DIR="${RAGENT_MODEL_DIR:-/home/trinq3/models/stcc}"

if [ ! -x "${PYTHON}" ]; then
  echo "Jetson Python is not executable: ${PYTHON}" >&2
  exit 1
fi
if [ ! -f "${APP_DIR}/demo/backend/app.py" ]; then
  echo "RAGENT deployment is missing under: ${APP_DIR}" >&2
  exit 1
fi
if [ ! -f "${MODEL_DIR}/tools_openai.json" ]; then
  echo "Standalone STCC checkpoint is missing: ${MODEL_DIR}" >&2
  exit 1
fi

export PYTHONPATH="${APP_DIR}/src:${APP_DIR}:${PYTHONPATH:-}"
export WEBTEST_HOST="${WEBTEST_HOST:-127.0.0.1}"
export WEBTEST_PORT="${WEBTEST_PORT:-8010}"
export WEBTEST_MODEL_MODE=vllm
export WEBTEST_MODEL_DIR="${WEBTEST_MODEL_DIR:-${MODEL_DIR}}"
export WEBTEST_VLLM_BASE_URL="${WEBTEST_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export WEBTEST_VLLM_MODEL="${WEBTEST_VLLM_MODEL:-stcc}"
export WEBTEST_MAX_NEW_TOKENS="${WEBTEST_MAX_NEW_TOKENS:-128}"
export WEBTEST_VAD="${WEBTEST_VAD:-omnivad}"
export WEBTEST_ENHANCER="${WEBTEST_ENHANCER:-fastenhancer_s}"
export WEBTEST_FASTENHANCER_S_MODEL="${WEBTEST_FASTENHANCER_S_MODEL:-${RAGENT_FASTENHANCER_MODEL:-${APP_DIR}/outputs/denoise/models/fastenhancer_s_dns.onnx}}"
export WEBTEST_CAPTURE_DIR="${WEBTEST_CAPTURE_DIR:-${APP_DIR}/outputs/demo/captures}"
export DEMO_VOICE_CACHE_DIR="${DEMO_VOICE_CACHE_DIR:-${APP_DIR}/outputs/demo/voice}"
export DEMO_STATIC_AUDIO_MANIFEST="${DEMO_STATIC_AUDIO_MANIFEST:-${APP_DIR}/outputs/demo/voice/manifest.json}"
export DEMO_TTS_ENABLED="${DEMO_TTS_ENABLED:-1}"
export DEMO_TTS_PROVIDER="${DEMO_TTS_PROVIDER:-omnivoice}"
export DEMO_OMNIVOICE_BASE_URL="${DEMO_OMNIVOICE_BASE_URL:-http://127.0.0.1:8120}"
export DEMO_CLOUD_ENABLED="${DEMO_CLOUD_ENABLED:-1}"
export DEMO_CLOUD_PROVIDER="${DEMO_CLOUD_PROVIDER:-gemini}"
export DEMO_CLOUD_MODEL="${DEMO_CLOUD_MODEL:-gemini-3.6-flash}"
export DEMO_GEMINI_THINKING_LEVEL="${DEMO_GEMINI_THINKING_LEVEL:-low}"

cd "${APP_DIR}"
exec "${PYTHON}" -m uvicorn demo.backend.app:app \
  --host "${WEBTEST_HOST}" \
  --port "${WEBTEST_PORT}"
