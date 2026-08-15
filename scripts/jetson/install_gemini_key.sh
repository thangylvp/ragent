#!/usr/bin/env bash
set -euo pipefail

readonly TARGET="${RAGENT_JETSON_ENV:-/home/trinq3/ragent/.env}"
readonly TARGET_DIR="$(dirname "${TARGET}")"

if [ ! -d "${TARGET_DIR}" ]; then
  echo "Jetson demo directory does not exist: ${TARGET_DIR}" >&2
  exit 1
fi
if [ -e "${TARGET}" ] && { [ ! -f "${TARGET}" ] || [ -L "${TARGET}" ]; }; then
  echo "Refusing to replace a non-regular environment target: ${TARGET}" >&2
  exit 1
fi

printf 'Gemini API key: ' >&2
IFS= read -r -s gemini_key
printf '\n' >&2
case "${gemini_key}" in
  AIza*) ;;
  *) echo "Invalid Gemini API key format" >&2; exit 1 ;;
esac
if [ "${#gemini_key}" -lt 30 ]; then
  echo "Gemini API key is unexpectedly short" >&2
  exit 1
fi

staging="$(mktemp "${TARGET_DIR}/.env.gemini.XXXXXX")"
readonly staging
trap 'rm -f -- "${staging}"' EXIT
chmod 600 "${staging}"
printf '%s\n' \
  "GEMINI_API_KEY=${gemini_key}" \
  "DEMO_CLOUD_ENABLED=1" \
  "DEMO_CLOUD_PROVIDER=gemini" \
  "DEMO_CLOUD_MODEL=gemini-3.6-flash" \
  "DEMO_GEMINI_THINKING_LEVEL=low" \
  > "${staging}"
mv -f -- "${staging}" "${TARGET}"
trap - EXIT

printf 'Gemini demo environment installed: %s (mode %s)\n' \
  "${TARGET}" "$(stat -c '%a' "${TARGET}")"
