#!/usr/bin/env bash

# Install pyskl into the existing wham conda environment

echo "======================================"
echo "Install pyskl into the wham conda environment"
echo "======================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

activate_conda_wham
enter_repo_root

# Check the Python version
echo "Python version:"
python --version

# Change to the installation directory
INSTALL_DIR="${PYSKL_HOME:-${HOME}}"
cd "$INSTALL_DIR"

# Check whether the pyskl directory already exists
if [ -d "pyskl" ]; then
    echo "Found an existing pyskl directory: $INSTALL_DIR/pyskl"
    echo "Use the existing directory..."
    cd pyskl
    git pull || echo "Git pull failed; continue using the existing code"
else
    echo "Cloning the pyskl repository..."
    git clone https://github.com/kennymckormick/pyskl.git
    cd pyskl
fi

echo ""
echo "Installing pyskl dependencies..."
echo "======================================"

# Install the base dependencies
pip install -q mmcv-full mmengine mmdet mmpose

# Install pyskl (editable mode)
echo "Installing pyskl..."
pip install -e .

# Verify the installation
echo ""
echo "======================================"
echo "Verifying the installation..."
python -c "import pyskl; print('✓ pyskl installed successfully!')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✓ pyskl installation completed!"
    echo "======================================"
    echo ""
    echo "You can now run demo.py for action recognition:"
    echo "  bash scripts/demo/run_demo_har_simple.sh"
    echo ""
else
    echo ""
    echo "======================================"
    echo "✗ pyskl installation failed. Please check the error output"
    echo "======================================"
    echo ""
    echo "Common issues:"
    echo "1. If mmcv-full installation fails, you can try:"
    echo "   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html"
    echo ""
    echo "2. If you run into CUDA version issues, check the PyTorch version:"
    echo "   python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
    echo ""
fi
