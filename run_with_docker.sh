#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-${SCRIPT_DIR}/test_docs}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/outputs}"
IMAGE_NAME="${IMAGE_NAME:-doclingflow2:latest}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-doclingflow2-run}"
DEFAULT_MEMORY_LIMIT_MB="${DEFAULT_MEMORY_LIMIT_MB:-12288}"
STATE_FILE="${OUTPUT_DIR}/reports/.last_container"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found in PATH."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/markdown" "${OUTPUT_DIR}/images" "${OUTPUT_DIR}/reports" "${OUTPUT_DIR}/logs"

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

docker run --rm \
  --name "${CONTAINER_NAME}" \
  -e TEST_DOCS_DIR=/data/input \
  -e OUTPUTS_DIR=/data/output \
  -e DEFAULT_MEMORY_LIMIT_MB="${DEFAULT_MEMORY_LIMIT_MB}" \
  -v "${INPUT_DIR}:/data/input:ro" \
  -v "${OUTPUT_DIR}:/data/output" \
  "${IMAGE_NAME}" \
  python -m doclingflow batch /data/input -o /data/output
