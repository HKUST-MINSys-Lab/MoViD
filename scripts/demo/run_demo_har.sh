#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VIDEO="${1:-${VIDEO:-dataset/output_raw.mp4}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-output/demo_har_output_raw}}"

export VIDEO
export OUTPUT_DIR

"${SCRIPT_DIR}/run_demo_with_har.sh" "${VIDEO}" "${OUTPUT_DIR}"
