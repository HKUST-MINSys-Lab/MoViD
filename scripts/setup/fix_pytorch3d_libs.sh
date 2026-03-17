#!/usr/bin/env bash

# Fix PyTorch3D CUDA library compatibility issue
# PyTorch3D 0.7.2 expects libtorch_cuda_cu.so, but newer PyTorch versions
# merged it into libtorch_cuda.so. This script creates the necessary symlink.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

activate_conda_movid

# Path to torch lib directory
TORCH_LIB_DIR="${CONDA_PREFIX}/lib/python3.9/site-packages/torch/lib"

if [ ! -d "$TORCH_LIB_DIR" ]; then
    echo "Error: Torch lib directory not found: $TORCH_LIB_DIR"
    exit 1
fi

cd "$TORCH_LIB_DIR"

# Check if libtorch_cuda.so exists
if [ ! -f "libtorch_cuda.so" ]; then
    echo "Error: libtorch_cuda.so not found in $TORCH_LIB_DIR"
    exit 1
fi

# Create symlink if it doesn't exist
if [ ! -f "libtorch_cuda_cu.so" ] && [ ! -L "libtorch_cuda_cu.so" ]; then
    echo "Creating symlink: libtorch_cuda_cu.so -> libtorch_cuda.so"
    sudo ln -s libtorch_cuda.so libtorch_cuda_cu.so
    echo "✓ Symlink created successfully"
elif [ -L "libtorch_cuda_cu.so" ]; then
    echo "✓ Symlink already exists"
    ls -la libtorch_cuda_cu.so
else
    echo "Warning: libtorch_cuda_cu.so already exists as a regular file"
    ls -la libtorch_cuda_cu.so
fi

# Also check for libtorch_cuda_cpp.so (sometimes needed)
if [ ! -f "libtorch_cuda_cpp.so" ] && [ ! -L "libtorch_cuda_cpp.so" ]; then
    echo "Creating symlink: libtorch_cuda_cpp.so -> libtorch_cuda.so"
    sudo ln -s libtorch_cuda.so libtorch_cuda_cpp.so
    echo "✓ Symlink created successfully"
fi

echo ""
echo "=========================================="
echo "Fix applied! You can now use PyTorch3D."
echo "Make sure to set LD_LIBRARY_PATH when running:"
echo "export LD_LIBRARY_PATH=\"${TORCH_LIB_DIR}:\${LD_LIBRARY_PATH}\""
echo "=========================================="
