#!/usr/bin/env bash

# Setup NTU dataset files
# Handles files that are already downloaded or need to be downloaded

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

enter_repo_root

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DATASET_DIR="dataset/NTU"
NTU60_DIR="${DATASET_DIR}/NTU60"
NTU120_DIR="${DATASET_DIR}/NTU120"

mkdir -p "${NTU60_DIR}"
mkdir -p "${NTU120_DIR}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}NTU Dataset Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Expected files:${NC}"
echo "  - NTU60:  nturgbd_skeletons_s001_to_s017.zip"
echo "  - NTU120: nturgbd_skeletons_s018_to_s032.zip"
echo ""

# Check for existing files
ntu60_file="${NTU60_DIR}/nturgbd_skeletons_s001_to_s017.zip"
ntu120_file="${NTU120_DIR}/nturgbd_skeletons_s018_to_s032.zip"

if [ -f "$ntu60_file" ]; then
    echo -e "${GREEN}✓ Found: $(basename $ntu60_file)${NC}"
    ntu60_ok=true
else
    echo -e "${YELLOW}Missing: $(basename $ntu60_file)${NC}"
    ntu60_ok=false
fi

if [ -f "$ntu120_file" ]; then
    echo -e "${GREEN}✓ Found: $(basename $ntu120_file)${NC}"
    ntu120_ok=true
else
    echo -e "${YELLOW}Missing: $(basename $ntu120_file)${NC}"
    ntu120_ok=false
fi

echo ""

# If files are missing, ask user what to do
if [ "$ntu60_ok" = false ] || [ "$ntu120_ok" = false ]; then
    echo "What would you like to do?"
    echo "1) I have the files on another machine - provide download URLs"
    echo "2) I have the files locally - provide file paths to copy"
    echo "3) Skip for now"
    read -p "Enter choice [1/2/3]: " choice
    
    if [ "$choice" = "1" ]; then
        # Download from URLs
        if [ "$ntu60_ok" = false ]; then
            read -p "Enter download URL for NTU60 (s001_to_s017): " url_60
            if [ ! -z "$url_60" ]; then
                echo "Downloading NTU60..."
                wget --continue --progress=bar:force:noscroll -O "$ntu60_file" "$url_60" || \
                curl -L --progress-bar -o "$ntu60_file" "$url_60"
            fi
        fi
        
        if [ "$ntu120_ok" = false ]; then
            read -p "Enter download URL for NTU120 (s018_to_s032): " url_120
            if [ ! -z "$url_120" ]; then
                echo "Downloading NTU120..."
                wget --continue --progress=bar:force:noscroll -O "$ntu120_file" "$url_120" || \
                curl -L --progress-bar -o "$ntu120_file" "$url_120"
            fi
        fi
    
    elif [ "$choice" = "2" ]; then
        # Copy from local paths
        if [ "$ntu60_ok" = false ]; then
            read -p "Enter path to nturgbd_skeletons_s001_to_s017.zip: " path_60
            if [ ! -z "$path_60" ] && [ -f "$path_60" ]; then
                cp "$path_60" "$ntu60_file"
                echo "✓ Copied NTU60 file"
            fi
        fi
        
        if [ "$ntu120_ok" = false ]; then
            read -p "Enter path to nturgbd_skeletons_s018_to_s032.zip: " path_120
            if [ ! -z "$path_120" ] && [ -f "$path_120" ]; then
                cp "$path_120" "$ntu120_file"
                echo "✓ Copied NTU120 file"
            fi
        fi
    fi
fi

# Extract files
extract_zip() {
    local zip_file=$1
    local output_dir=$2
    local name=$(basename "$zip_file")
    
    if [ ! -f "$zip_file" ]; then
        return 1
    fi
    
    echo -e "${YELLOW}Extracting ${name}...${NC}"
    unzip -q -o "$zip_file" -d "$output_dir" && \
    echo -e "${GREEN}✓ Extracted ${name}${NC}"
}

# Check if extraction is needed
if [ -f "$ntu60_file" ]; then
    if [ ! -d "${NTU60_DIR}/nturgb+d_skeletons" ] && [ ! -d "${NTU60_DIR}/nturgbd_skeletons_s001_to_s017" ]; then
        read -p "Extract NTU60? [Y/n]: " extract
        if [[ ! "$extract" =~ ^[Nn]$ ]]; then
            extract_zip "$ntu60_file" "${NTU60_DIR}"
        fi
    else
        echo -e "${GREEN}✓ NTU60 already extracted${NC}"
    fi
fi

if [ -f "$ntu120_file" ]; then
    if [ ! -d "${NTU120_DIR}/nturgb+d_skeletons120" ] && [ ! -d "${NTU120_DIR}/nturgbd_skeletons_s018_to_s032" ]; then
        read -p "Extract NTU120? [Y/n]: " extract
        if [[ ! "$extract" =~ ^[Nn]$ ]]; then
            extract_zip "$ntu120_file" "${NTU120_DIR}"
        fi
    else
        echo -e "${GREEN}✓ NTU120 already extracted${NC}"
    fi
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Dataset locations:"
echo "  NTU60:  ${NTU60_DIR}/"
echo "  NTU120: ${NTU120_DIR}/"
echo ""
