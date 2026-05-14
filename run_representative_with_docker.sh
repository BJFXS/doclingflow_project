#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_INPUT_DIR="${SCRIPT_DIR}/outputs/representative_input"
TMP_OUTPUT_DIR="${SCRIPT_DIR}/outputs/representative_output"

rm -rf "${TMP_INPUT_DIR}" "${TMP_OUTPUT_DIR}"
mkdir -p \
  "${TMP_INPUT_DIR}/PPTX" \
  "${TMP_INPUT_DIR}/html" \
  "${TMP_INPUT_DIR}/office" \
  "${TMP_INPUT_DIR}/pdf_plain" \
  "${TMP_INPUT_DIR}/pdf_scan" \
  "${TMP_INPUT_DIR}/pdf_two_column"

cp "${SCRIPT_DIR}/test_docs/PPTX/pptx1.pptx" "${TMP_INPUT_DIR}/PPTX/"
cp "${SCRIPT_DIR}/test_docs/html/html_01.html" "${TMP_INPUT_DIR}/html/"
cp "${SCRIPT_DIR}/test_docs/office/office_01.docx" "${TMP_INPUT_DIR}/office/"
cp "${SCRIPT_DIR}/test_docs/pdf_plain/plain_01.pdf" "${TMP_INPUT_DIR}/pdf_plain/"
cp "${SCRIPT_DIR}/test_docs/pdf_scan/scan_1.pdf" "${TMP_INPUT_DIR}/pdf_scan/"
cp "${SCRIPT_DIR}/test_docs/pdf_two_column/paper_01.pdf" "${TMP_INPUT_DIR}/pdf_two_column/"

"${SCRIPT_DIR}/run_with_docker.sh" "${TMP_INPUT_DIR}" "${TMP_OUTPUT_DIR}"
