#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENABLE_MOTIONGPT="${ENABLE_MOTIONGPT:-false}"
if [ "${ENABLE_MOTIONGPT}" = "true" ]; then
    export EXTRA_DEMO_ARGS="--motiongpt"
else
    export EXTRA_DEMO_ARGS=""
fi

"${SCRIPT_DIR}/run_asd_demo.sh"
