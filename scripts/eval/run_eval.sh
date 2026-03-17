#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

activate_conda_wham
enter_repo_root

TARGET="${1:-3dpw}"
CHECKPOINT="${2:-$(default_wham_checkpoint)}"
CFG="${CFG:-configs/yamls/demo.yaml}"

require_file "${CHECKPOINT}" "Evaluation checkpoint"

case "${TARGET}" in
    3dpw)
        python -m lib.eval.evaluate_3dpw --cfg "${CFG}" TRAIN.CHECKPOINT "${CHECKPOINT}"
        ;;
    rich)
        python -m lib.eval.evaluate_rich --cfg "${CFG}" TRAIN.CHECKPOINT "${CHECKPOINT}"
        ;;
    emdb1)
        python -m lib.eval.evaluate_emdb --cfg "${CFG}" --eval-split 1 TRAIN.CHECKPOINT "${CHECKPOINT}"
        ;;
    emdb2)
        python -m lib.eval.evaluate_emdb --cfg "${CFG}" --eval-split 2 TRAIN.CHECKPOINT "${CHECKPOINT}"
        ;;
    *)
        echo "Usage: $0 <3dpw|rich|emdb1|emdb2> [checkpoint]"
        exit 1
        ;;
esac
