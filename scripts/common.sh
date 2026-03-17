#!/usr/bin/env bash

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"

repo_root() {
    printf '%s\n' "${ROOT_DIR}"
}

enter_repo_root() {
    cd "${ROOT_DIR}"
}

activate_conda_wham() {
    local candidate

    if ! command -v conda >/dev/null 2>&1; then
        for candidate in \
            "${HOME}/miniconda3/etc/profile.d/conda.sh" \
            "${HOME}/anaconda3/etc/profile.d/conda.sh"
        do
            if [ -f "${candidate}" ]; then
                # shellcheck disable=SC1090
                source "${candidate}"
                break
            fi
        done
    fi

    if command -v conda >/dev/null 2>&1; then
        conda activate "${CONDA_ENV_NAME:-wham}" >/dev/null 2>&1 || true
    fi
}

default_wham_checkpoint() {
    local path
    for path in \
        "${ROOT_DIR}/checkpoints/wham_vit_w_3dpw.pth.tar" \
        "${ROOT_DIR}/checkpoints/wham_stage1.tar.pth"
    do
        if [ -f "${path}" ]; then
            printf '%s\n' "${path}"
            return 0
        fi
    done

    printf '%s\n' "${ROOT_DIR}/checkpoints/wham_vit_w_3dpw.pth.tar"
}

default_action_config() {
    printf '%s\n' "${ROOT_DIR}/models/action_recognition/stgcn++_ntu60_xsub_3d_config.py"
}

default_action_label_map() {
    printf '%s\n' "${ROOT_DIR}/models/action_recognition/stgcn++_ntu60_xsub_3d_labels.txt"
}

default_action_checkpoint() {
    local path
    for path in \
        "${ROOT_DIR}/models/action_recognition/stgcn++_ntu60_xsub_3d.pth" \
        "${ROOT_DIR}/checkpoints/action_recognition/stgcn++_ntu60_xsub_3d.pth"
    do
        if [ -f "${path}" ]; then
            printf '%s\n' "${path}"
            return 0
        fi
    done

    printf '%s\n' "${ROOT_DIR}/models/action_recognition/stgcn++_ntu60_xsub_3d.pth"
}

require_file() {
    local path="$1"
    local label="${2:-Required file}"

    if [ ! -f "${path}" ]; then
        echo "${label} not found: ${path}" >&2
        return 1
    fi
}
