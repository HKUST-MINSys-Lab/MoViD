#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

activate_conda_movid
enter_repo_root

INPUT_PATH="${1:-examples}"
OUTPUT_DIR="${2:-output/demo_batch}"
MOVID_CHECKPOINT="${MOVID_CHECKPOINT:-$(default_movid_checkpoint)}"

require_file "${MOVID_CHECKPOINT}" "MoViD checkpoint"
mkdir -p "${OUTPUT_DIR}"

if [ -f "${INPUT_PATH}" ]; then
    python demo.py \
        --video "${INPUT_PATH}" \
        --output_pth "${OUTPUT_DIR}" \
        --checkpoint "${MOVID_CHECKPOINT}" \
        --visualize \
        --estimate_local_only
    exit 0
fi

mapfile -t VIDEOS < <(find "${INPUT_PATH}" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | sort)

if [ "${#VIDEOS[@]}" -eq 0 ]; then
    echo "No videos found in ${INPUT_PATH}"
    exit 1
fi

for video in "${VIDEOS[@]}"; do
    echo "Processing $(basename "${video}")"
    python demo.py \
        --video "${video}" \
        --output_pth "${OUTPUT_DIR}" \
        --checkpoint "${MOVID_CHECKPOINT}" \
        --visualize \
        --estimate_local_only
done
