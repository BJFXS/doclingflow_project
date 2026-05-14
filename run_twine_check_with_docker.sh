#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-doclingflow2-twine-check}"
STATE_FILE="${SCRIPT_DIR}/outputs/reports/.last_twine_check_container"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found in PATH."
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/outputs/reports"

if [[ -f "${STATE_FILE}" ]]; then
  LAST_CONTAINER="$(cat "${STATE_FILE}")"
  if [[ -n "${LAST_CONTAINER}" ]]; then
    docker rm -f "${LAST_CONTAINER}" >/dev/null 2>&1 || true
  fi
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
CONTAINER_NAME="${CONTAINER_PREFIX}-${TIMESTAMP}"
printf '%s' "${CONTAINER_NAME}" > "${STATE_FILE}"
trap 'docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true' EXIT

docker run --name "${CONTAINER_NAME}" \
  -v "${SCRIPT_DIR}:/app" \
  -w /app \
  python:3.12-slim \
  bash -lc "rm -rf dist build *.egg-info && python -m pip install --upgrade pip setuptools wheel build twine && python -m build && twine check dist/*"
