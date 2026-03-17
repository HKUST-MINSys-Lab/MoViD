#!/usr/bin/env bash

# MoViD Environment Installation Script
# Based on docs/INSTALL.md

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

enter_repo_root

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MoViD Environment Installation${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}Warning: conda not found in PATH${NC}"
    echo -e "${YELLOW}Please make sure conda is installed and in your PATH${NC}"
    echo -e "${YELLOW}Or activate conda base environment first:${NC}"
    echo -e "${YELLOW}  source ~/anaconda3/bin/activate${NC}"
    echo -e "${YELLOW}  # or${NC}"
    echo -e "${YELLOW}  source ~/miniconda3/bin/activate${NC}"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " continue_choice
    if [[ ! "$continue_choice" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Create Conda environment
echo -e "${GREEN}Step 1: Creating Conda environment 'movid' with Python 3.9...${NC}"
if command -v conda &> /dev/null; then
    # Check if environment already exists
    if conda env list | grep -q "^movid "; then
        echo -e "${YELLOW}Environment 'movid' already exists.${NC}"
        read -p "Do you want to remove and recreate it? (y/N): " recreate_choice
        if [[ "$recreate_choice" =~ ^[Yy]$ ]]; then
            conda env remove -n movid -y
            conda create -n movid python=3.9 -y
        else
            echo -e "${YELLOW}Using existing environment.${NC}"
        fi
    else
        conda create -n movid python=3.9 -y
    fi
    
    echo -e "${GREEN}Activating conda environment 'movid'...${NC}"
    echo -e "${YELLOW}Please run: conda activate movid${NC}"
    echo -e "${YELLOW}Then continue with the installation.${NC}"
    echo ""
    
    # Note: We can't activate conda in a script, so we'll provide instructions
    echo -e "${GREEN}Next steps (run these commands after activating the environment):${NC}"
    echo ""
    echo "conda activate movid"
    echo ""
    echo "# Install PyTorch libraries"
    echo "conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y"
    echo ""
    echo "# Install PyTorch3D dependencies (optional)"
    echo "conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y"
    echo "pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html"
    echo ""
    echo "# Install MoViD dependencies"
    echo "pip install -r $(repo_root)/requirements.txt"
    echo ""
    echo "# Install ViTPose"
    echo "pip install -v -e $(repo_root)/third-party/ViTPose"
    echo ""
    echo "# Install DPVO"
    echo "cd $(repo_root)/third-party/DPVO"
    echo "wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    echo "unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip"
    echo "conda install pytorch-scatter=2.0.9 -c rusty1s -y"
    echo "conda install cudatoolkit-dev=11.3.1 -c conda-forge -y"
    echo ""
    echo "# Check GCC version"
    echo "gcc --version"
    echo "# If GCC > 10, install gxx=9.5:"
    echo "conda install -c conda-forge gxx=9.5 -y"
    echo ""
    echo "pip install ."
    echo "cd $(repo_root)"
    echo ""
    
else
    echo -e "${RED}Error: conda command not found${NC}"
    echo -e "${YELLOW}Please install Anaconda or Miniconda first:${NC}"
    echo -e "${YELLOW}  https://www.anaconda.com/products/distribution${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation script completed!${NC}"
echo -e "${GREEN}========================================${NC}"
