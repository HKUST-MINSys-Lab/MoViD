#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-output/demo_har_live}"
"${SCRIPT_DIR}/run_demo_with_har.sh" "${1:-dataset/output_raw.mp4}" "${2:-${OUTPUT_DIR}}"
