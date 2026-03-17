#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_demo_with_har.sh" "${1:-dataset/output_raw.mp4}" "${2:-output/demo_har_output_raw}"
