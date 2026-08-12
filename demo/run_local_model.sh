#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WEBTEST_MODEL_MODE=local
export WEBTEST_MODEL_DIR="${WEBTEST_MODEL_DIR:-$DEMO_DIR/../../stc/outputs/models/route_v1_best_step1250_hf}"

exec "$DEMO_DIR/run.sh" "$@"
