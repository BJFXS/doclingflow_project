#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-doclingflow2:latest}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-doclingflow2-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
STATE_FILE="${OUTPUT_DIR}/reports/.last_test_container"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found in PATH."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/reports" "${OUTPUT_DIR}/logs"

if [[ -f "${STATE_FILE}" ]]; then
  LAST_CONTAINER="$(cat "${STATE_FILE}")"
  if [[ -n "${LAST_CONTAINER}" ]]; then
    docker rm -f "${LAST_CONTAINER}" >/dev/null 2>&1 || true
  fi
fi

docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
CONTAINER_NAME="${CONTAINER_PREFIX}-${TIMESTAMP}"
printf '%s' "${CONTAINER_NAME}" > "${STATE_FILE}"
trap 'docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true' EXIT

docker run --name "${CONTAINER_NAME}" \
  -v "${SCRIPT_DIR}:/app" \
  -w /app \
  "${IMAGE_NAME}" \
  bash -lc "python -m doclingflow --help && python -m unittest discover -s tests"
