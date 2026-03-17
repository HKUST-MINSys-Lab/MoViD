#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

activate_conda_wham
enter_repo_root

TRAIN_CFG="${TRAIN_CFG:-configs/yamls/stage2.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-checkpoints/wham_stage1.tar.pth}"

require_file "${TRAIN_CFG}" "Training config"
require_file "${INIT_CHECKPOINT}" "Stage 1 checkpoint"

mkdir -p logs
LOG_FILE="logs/stage2_train_$(date +%Y%m%d_%H%M%S).log"

nohup python train.py \
    --cfg "${TRAIN_CFG}" \
    TRAIN.CHECKPOINT "${INIT_CHECKPOINT}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > logs/stage2_train.pid

echo "Training started in background."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
