# NTU RGB+D Dataset Download Guide

This guide explains how to download NTU60 and NTU120 datasets for the MoViD project.

## Overview

- **NTU RGB+D 60 (NTU60)**: Contains 60 action classes with skeleton, RGB, and depth data
- **NTU RGB+D 120 (NTU120)**: Extended version with 120 action classes

## Official Website

**Important**: You must register and get approval from the official website:
- Website: https://rose1.ntu.edu.sg/dataset/actionRecognition/
- Registration is required for academic/research use
- You will receive download links after approval

## Download Methods

### Method 1: Use the included setup helper

```bash
bash scripts/setup/setup_ntu_files.sh
```

The helper script creates the expected directory structure, lets you copy or download the zip files, and can extract them in place.

### Method 2: Manual Download

1. **Register at the official website**
   - Visit: https://rose1.ntu.edu.sg/dataset/actionRecognition/
   - Complete the registration form
   - Wait for approval (usually takes a few days)

2. **Download the skeleton data files**
   - For NTU60: Download `nturgbd_skeletons_s001_to_s017.zip` (or similar)
   - For NTU120: Download `nturgbd_skeletons_s018_to_s032.zip` (or similar)

3. **Place files in the correct directories**
   ```bash
   # Create directories
   mkdir -p dataset/NTU/NTU60
   mkdir -p dataset/NTU/NTU120
   
   # Move downloaded files
   mv nturgbd_skeletons_s001_to_s017.zip dataset/NTU/NTU60/
   mv nturgbd_skeletons_s018_to_s032.zip dataset/NTU/NTU120/
   ```

4. **Extract the files**
   ```bash
   bash scripts/setup/setup_ntu_files.sh
   ```

## Directory Structure

After downloading and extracting, your directory structure should look like:

```
dataset/
└── NTU/
    ├── NTU60/
    │   └── nturgb+d_skeletons/  (or similar)
    │       ├── S001C001P001R001A001.skeleton
    │       ├── S001C001P001R001A002.skeleton
    │       └── ...
    └── NTU120/
        └── nturgb+d_skeletons120/  (or similar)
            ├── S018C001P001R001A001.skeleton
            ├── S018C001P001R001A002.skeleton
            └── ...
```

## Dataset Information

### NTU60
- **Actions**: 60 classes
- **Subjects**: 40
- **Camera views**: 80 (2 cameras × 40 setups)
- **Total samples**: ~56,000

### NTU120
- **Actions**: 120 classes (includes all 60 from NTU60)
- **Subjects**: 106
- **Camera views**: 212 (2 cameras × 106 setups)
- **Total samples**: ~114,000

## Notes

- The skeleton data files are typically large (several GB)
- Download may take a long time depending on your internet connection
- Make sure you have enough disk space (recommended: at least 50GB free)
- RGB and depth videos are optional and much larger - only download if needed
- The datasets are for academic/research use only

## Troubleshooting

### Download fails
- Check your internet connection
- Verify the download URL is correct
- Try using a different download tool (wget, curl, or browser)

### Extraction fails
- Ensure you have enough disk space
- Check if the zip file is corrupted (try re-downloading)
- Verify the zip file is complete

### Permission denied
- Make sure you have write permissions in the dataset directory
- Try running with appropriate permissions: `sudo` (if needed)

## References

- Official NTU RGB+D Dataset: https://rose1.ntu.edu.sg/dataset/actionRecognition/
- Paper: "NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding" (CVPR 2019)
