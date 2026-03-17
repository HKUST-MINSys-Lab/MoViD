#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

activate_conda_movid
enter_repo_root

ASD_DIR="${ASD_DIR:-$(repo_root)/ASD}"
OUTPUT_DIR="${OUTPUT_DIR:-${ASD_DIR}/output}"
MOVID_CHECKPOINT="${MOVID_CHECKPOINT:-$(default_movid_checkpoint)}"
ACTION_CONFIG="${ACTION_CONFIG:-$(default_action_config)}"
ACTION_CHECKPOINT="${ACTION_CHECKPOINT:-$(default_action_checkpoint)}"
ACTION_LABEL_MAP="${ACTION_LABEL_MAP:-$(default_action_label_map)}"
DISABLE_VISUALIZE="${DISABLE_VISUALIZE:-false}"
USE_SKELETON_ONLY="${USE_SKELETON_ONLY:-false}"
EXTRA_DEMO_ARGS="${EXTRA_DEMO_ARGS:-}"

require_file "${MOVID_CHECKPOINT}" "MoViD checkpoint"

ACTION_ARGS=()
if [ -f "${ACTION_CONFIG}" ] && [ -f "${ACTION_CHECKPOINT}" ] && [ -f "${ACTION_LABEL_MAP}" ]; then
    ACTION_ARGS=(
        --action_config "${ACTION_CONFIG}"
        --action_checkpoint "${ACTION_CHECKPOINT}"
        --action_label_map "${ACTION_LABEL_MAP}"
    )
else
    echo "Action recognition assets are missing, continuing with MoViD only."
fi

mkdir -p "${OUTPUT_DIR}"

mapfile -t VIDEOS < <(find "${ASD_DIR}" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | sort)

if [ "${#VIDEOS[@]}" -eq 0 ]; then
    echo "No videos found in ${ASD_DIR}"
    exit 1
fi

for i in "${!VIDEOS[@]}"; do
    video="${VIDEOS[$i]}"
    video_name="$(basename "${video}")"
    echo "[${i}/${#VIDEOS[@]}] Processing ${video_name}"

    VIS_ARGS=()
    if [ "${DISABLE_VISUALIZE}" != "true" ]; then
        VIS_ARGS+=(--visualize)
        if [ "${USE_SKELETON_ONLY}" = "true" ]; then
            VIS_ARGS+=(--skeleton_only)
        fi
    fi

    EXTRA_ARGS=()
    if [ -n "${EXTRA_DEMO_ARGS}" ]; then
        # shellcheck disable=SC2206
        EXTRA_ARGS=(${EXTRA_DEMO_ARGS})
    fi

    python demo.py \
        --video "${video}" \
        --output_pth "${OUTPUT_DIR}" \
        --checkpoint "${MOVID_CHECKPOINT}" \
        --save_pkl \
        "${VIS_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" \
        "${ACTION_ARGS[@]}"
done
