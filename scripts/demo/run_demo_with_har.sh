#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

activate_conda_movid
enter_repo_root

VIDEO="${1:-${VIDEO:-examples/demo_video.mp4}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-output/demo_har}}"
MOVID_CHECKPOINT="${MOVID_CHECKPOINT:-$(default_movid_checkpoint)}"
ACTION_CONFIG="${ACTION_CONFIG:-$(default_action_config)}"
ACTION_CHECKPOINT="${ACTION_CHECKPOINT:-$(default_action_checkpoint)}"
ACTION_LABEL_MAP="${ACTION_LABEL_MAP:-$(default_action_label_map)}"

require_file "${MOVID_CHECKPOINT}" "MoViD checkpoint"
require_file "${VIDEO}" "Input video"

ACTION_ARGS=()
if [ -f "${ACTION_CONFIG}" ] && [ -f "${ACTION_CHECKPOINT}" ] && [ -f "${ACTION_LABEL_MAP}" ]; then
    ACTION_ARGS=(
        --action_config "${ACTION_CONFIG}"
        --action_checkpoint "${ACTION_CHECKPOINT}"
        --action_label_map "${ACTION_LABEL_MAP}"
    )
else
    echo "Action recognition assets are incomplete."
    echo "Run: python tools/action/download_stgcn_model.py"
    echo "Or set ACTION_CONFIG / ACTION_CHECKPOINT / ACTION_LABEL_MAP manually."
    echo "Continuing without action recognition."
fi

echo "======================================"
echo "Running MoViD demo"
echo "======================================"
echo "Video: ${VIDEO}"
echo "Output: ${OUTPUT_DIR}"
echo "Checkpoint: ${MOVID_CHECKPOINT}"
echo "======================================"
echo ""

python demo.py \
    --video "${VIDEO}" \
    --output_pth "${OUTPUT_DIR}" \
    --checkpoint "${MOVID_CHECKPOINT}" \
    --save_pkl \
    --visualize \
    "${ACTION_ARGS[@]}"

echo ""
echo "Done! Results saved to: ${OUTPUT_DIR}"
