#!/usr/bin/env bash

# Fix torchvision version compatibility issues
# PyTorch 2.8.0 requires torchvision 0.23.0

echo "Upgrading torchvision to a compatible version..."
echo "Current version: PyTorch 2.8.0, torchvision 0.12.0"
echo "Target version: torchvision 0.23.0"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

activate_conda_wham

# Upgrade torchvision
pip install --upgrade torchvision==0.23.0

echo ""
echo "Upgrade complete. Please run the script again."
