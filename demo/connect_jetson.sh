#!/usr/bin/env bash
set -euo pipefail

# Keep the browser and microphone on the laptop while forwarding every demo
# API/WebSocket request to the Jetson.  The reverse forward lets the Jetson
# harness call the laptop-resident OmniVoice worker without exposing that
# service outside the SSH connection.
readonly SSH_TARGET="${RAGENT_JETSON_SSH_TARGET:-jetson}"
readonly LOCAL_WEB_PORT="${RAGENT_LOCAL_WEB_PORT:-8011}"
readonly JETSON_WEB_PORT="${RAGENT_JETSON_WEB_PORT:-8010}"
readonly LAPTOP_TTS_PORT="${RAGENT_LAPTOP_TTS_PORT:-8120}"
readonly JETSON_TTS_PORT="${RAGENT_JETSON_TTS_PORT:-8120}"

for port in \
  "${LOCAL_WEB_PORT}" \
  "${JETSON_WEB_PORT}" \
  "${LAPTOP_TTS_PORT}" \
  "${JETSON_TTS_PORT}"; do
  if ! [[ "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
    echo "All tunnel ports must be integers from 1 to 65535; got: ${port}" >&2
    exit 2
  fi
done

printf 'RAGENT browser: http://127.0.0.1:%s\n' "${LOCAL_WEB_PORT}"
printf 'Forwarding browser -> %s:127.0.0.1:%s\n' \
  "${SSH_TARGET}" "${JETSON_WEB_PORT}"
printf 'Forwarding Jetson TTS -> laptop:127.0.0.1:%s\n' \
  "${LAPTOP_TTS_PORT}"
printf 'Keep this command running while using the demo. Press Ctrl-C to stop.\n'

exec ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${LOCAL_WEB_PORT}:127.0.0.1:${JETSON_WEB_PORT}" \
  -R "127.0.0.1:${JETSON_TTS_PORT}:127.0.0.1:${LAPTOP_TTS_PORT}" \
  "${SSH_TARGET}"
