#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WEBTEST_MODEL_MODE=vllm
export WEBTEST_VLLM_BASE_URL="${WEBTEST_VLLM_BASE_URL:-http://127.0.0.1:8100/v1}"
export WEBTEST_VLLM_MODEL="${WEBTEST_VLLM_MODEL:-stcc}"

exec "$DEMO_DIR/run.sh" "$@"
