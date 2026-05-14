#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_PREFIX="${BUILD_PREFIX:-doclingflow2-dist-build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-doclingflow2-dist-install}"
BUILD_STATE_FILE="${SCRIPT_DIR}/outputs/reports/.last_dist_build_container"
INSTALL_STATE_FILE="${SCRIPT_DIR}/outputs/reports/.last_dist_install_container"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found in PATH."
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/outputs/reports"

if [[ -f "${BUILD_STATE_FILE}" ]]; then
  LAST_BUILD_CONTAINER="$(cat "${BUILD_STATE_FILE}")"
  if [[ -n "${LAST_BUILD_CONTAINER}" ]]; then
    docker rm -f "${LAST_BUILD_CONTAINER}" >/dev/null 2>&1 || true
  fi
fi

if [[ -f "${INSTALL_STATE_FILE}" ]]; then
  LAST_INSTALL_CONTAINER="$(cat "${INSTALL_STATE_FILE}")"
  if [[ -n "${LAST_INSTALL_CONTAINER}" ]]; then
    docker rm -f "${LAST_INSTALL_CONTAINER}" >/dev/null 2>&1 || true
  fi
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BUILD_CONTAINER="${BUILD_PREFIX}-${TIMESTAMP}"
INSTALL_CONTAINER="${INSTALL_PREFIX}-${TIMESTAMP}"
printf '%s' "${BUILD_CONTAINER}" > "${BUILD_STATE_FILE}"
printf '%s' "${INSTALL_CONTAINER}" > "${INSTALL_STATE_FILE}"
trap 'docker rm -f "${BUILD_CONTAINER}" >/dev/null 2>&1 || true; docker rm -f "${INSTALL_CONTAINER}" >/dev/null 2>&1 || true' EXIT

docker run --name "${BUILD_CONTAINER}" \
  -v "${SCRIPT_DIR}:/app" \
  -w /app \
  python:3.12-slim \
  bash -lc "rm -rf dist build *.egg-info && python -m pip install --upgrade pip setuptools wheel build && python -m build"

docker run --name "${INSTALL_CONTAINER}" \
  -v "${SCRIPT_DIR}:/app" \
  -w /app \
  python:3.12-slim \
  bash -lc 'python -m pip install --upgrade pip setuptools wheel && WHEEL=$(find dist -maxdepth 1 -name "doclingflow-*.whl" | head -n 1) && test -n "$WHEEL" && pip install "$WHEEL" && doclingflow --help'
