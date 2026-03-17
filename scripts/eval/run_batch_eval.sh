#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

activate_conda_wham
enter_repo_root

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <gt_checkpoint> <pred_checkpoint> <folder1> [folder2 ...]"
    exit 1
fi

GT_CHECKPOINT="$1"
PRED_CHECKPOINT="$2"
shift 2

require_file "${GT_CHECKPOINT}" "Ground-truth checkpoint"
require_file "${PRED_CHECKPOINT}" "Prediction checkpoint"

OUTPUT_BASE="${OUTPUT_BASE:-output/batch_eval}"
CALIB="${CALIB:-}"

CMD=(
    python batch_eval.py
    --gt_checkpoint "${GT_CHECKPOINT}"
    --pred_checkpoint "${PRED_CHECKPOINT}"
    --output_base "${OUTPUT_BASE}"
    --folders
)

for folder in "$@"; do
    CMD+=("${folder}")
done

if [ -n "${CALIB}" ]; then
    CMD+=(--calib "${CALIB}")
fi

if [ "${VISUALIZE:-false}" = "true" ]; then
    CMD+=(--visualize)
fi

if [ "${RUN_SMPLIFY:-false}" = "true" ]; then
    CMD+=(--run_smplify)
fi

if [ "${ESTIMATE_LOCAL_ONLY:-false}" = "true" ]; then
    CMD+=(--estimate_local_only)
fi

"${CMD[@]}"
