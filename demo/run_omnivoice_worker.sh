#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$DEMO_DIR/.." && pwd)"
PYTHON="${RAGENT_TTS_PYTHON:-python3}"

cd "$REPO_DIR"
exec "$PYTHON" -m uvicorn demo.tts_worker:app \
  --host "${DEMO_OMNIVOICE_HOST:-0.0.0.0}" \
  --port "${DEMO_OMNIVOICE_PORT:-8120}"
