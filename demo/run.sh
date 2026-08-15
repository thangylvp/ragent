#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$DEMO_DIR/.." && pwd)"
PYTHON="${RAGENT_DEMO_PYTHON:-python3}"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"
cd "$REPO_DIR"

exec "$PYTHON" -m uvicorn demo.backend.app:app \
  --host "${WEBTEST_HOST:-127.0.0.1}" \
  --port "${WEBTEST_PORT:-8010}" "$@"
