#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

activate_conda_wham
enter_repo_root

BASE_DIR="${BASE_DIR:-}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-15}"
LOG_FILE="logs/preprocess_humman_$(date +%Y%m%d_%H%M%S).log"

if [ -z "${BASE_DIR}" ]; then
    echo "Please set BASE_DIR to your HuMMan root directory."
    exit 1
fi

mkdir -p logs

for view in $(seq 0 9); do
    echo "Running HuMMan preprocessing for view ${view}" | tee -a "${LOG_FILE}"
    python tools/data/preprocess_HuMMan.py \
        --base_dir "${BASE_DIR}" \
        --view "${view}" \
        --split "${SPLIT}" \
        --batch_size "${BATCH_SIZE}" 2>&1 | tee -a "${LOG_FILE}"
done
