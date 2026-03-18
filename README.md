# MoViD

MoViD is a research codebase for monocular 3D human motion reconstruction and activity understanding. This GitHub repository is organized as the camera-ready code release: it keeps the maintained training pipeline, full-video inference pipeline, streaming inference pipeline, API wrapper, and edge-side inference project in a single public tree.

## Camera-Ready Release

Included in this repository:

- full training code under `train.py`, `lib/`, `configs/`, and `scripts/train/`
- full inference code under `demo.py`, `movid_api.py`, and `scripts/demo/`
- evaluation code under `batch_eval.py`, `lib/eval/`, and `scripts/eval/`
- edge-side inference under `MoViD_edge/`
- installation notes, dataset notes, and usage guides under `docs/`
- bundled third-party dependencies under `third-party/`

Intentionally excluded from GitHub:

- datasets and parsed dataset dumps
- pretrained checkpoints and downloaded model weights
- example outputs, logs, and local experiment folders
- large binary artifacts such as videos, archives, and TensorRT engines

Use [docs/CAMERA_READY.md](docs/CAMERA_READY.md) for the full step-by-step code-release guide.

## Start Here

### 1. Clone the repository

```bash
git clone <your-movid-repo-url> --recursive
cd MoViD
```

### 2. Install the environment

See [docs/INSTALL.md](docs/INSTALL.md) for the detailed setup, or start with:

```bash
bash scripts/setup/install_environment.sh
```

If you want action recognition support:

```bash
bash scripts/setup/install_pyskl.sh
python tools/action/download_stgcn_model.py
```

### 3. Download demo assets and checkpoints

```bash
bash scripts/setup/fetch_demo_data.sh
```

This script downloads demo videos, default checkpoints, and required body-model assets. It will ask for your SMPL / SMPLify credentials.

### 4. Run offline inference

```bash
python demo.py \
  --video examples/demo_video.mp4 \
  --output_pth output/demo \
  --visualize
```

Or use the helper wrapper with action recognition:

```bash
bash scripts/demo/run_demo_with_har.sh examples/demo_video.mp4 output/demo_har
```

### 5. Run stream inference

```bash
python demo.py \
  --video examples/demo_video.mp4 \
  --mode stream \
  --stream_window_size 10 \
  --output_pth output/stream \
  --visualize
```

### 6. Launch training

```bash
python train.py --cfg configs/yamls/stage2.yaml
```

If your environment blocks multi-worker dataloaders, use:

```bash
python train.py --cfg configs/yamls/stage2.yaml NUM_WORKERS 0
```

### 7. Run evaluation

```bash
bash scripts/eval/run_eval.sh 3dpw checkpoints/movid_vit_w_3dpw.pth.tar
```

### 8. Run edge inference

Offline edge inference:

```bash
python MoViD_edge/demo.py \
  --video examples/demo_video.mp4 \
  --output_pth output/edge_demo \
  --visualize
```

Real-time / streaming edge inference:

```bash
python MoViD_edge/real_time.py \
  --video realsense \
  --output_pth output/edge_rt \
  --visualize \
  --max_frames 1000
```

## Main Entrypoints

- `train.py`: root training entrypoint
- `demo.py`: root full-video and stream inference entrypoint
- `batch_eval.py`: batch evaluation utility
- `movid_api.py`: compatibility API / CLI wrapper
- `MoViD_edge/demo.py`: edge offline inference entrypoint
- `MoViD_edge/real_time.py`: edge real-time streaming entrypoint

## Repository Layout

```text
.
|-- configs/
|-- docs/
|   |-- guides/
|   |-- API.md
|   |-- CAMERA_READY.md
|   |-- DATASET.md
|   `-- INSTALL.md
|-- lib/
|-- models/
|   `-- action_recognition/
|-- scripts/
|   |-- data/
|   |-- demo/
|   |-- eval/
|   |-- setup/
|   `-- train/
|-- tools/
|   |-- action/
|   |-- data/
|   `-- eval/
|-- MoViD_edge/
|-- third-party/
|-- batch_eval.py
|-- demo.py
|-- movid_api.py
`-- train.py
```

## Documentation

- [Camera-Ready Guide](docs/CAMERA_READY.md)
- [Installation](docs/INSTALL.md)
- [Dataset Preparation](docs/DATASET.md)
- [Python API](docs/API.md)
- [Action Recognition Guide](docs/guides/action-recognition.md)
- [HAR Quick Start](docs/guides/quick-start-har.md)
- [MoViD Edge Guide](MoViD_edge/README.md)

## What Stays Out of GitHub

The `.gitignore` excludes local-only or large artifacts such as:

- `dataset/`
- `checkpoints/`
- `logs/`
- `output/`
- `final_version/`
- `.local/`
- downloaded videos and archives

If you already have local weights, place them under `checkpoints/`, or set environment variables such as `MOVID_CHECKPOINT` and `ACTION_CHECKPOINT` when running the helper scripts.

Some non-core analysis, research, and one-off data-conversion helpers can be kept under `.local/retired_tools/` if you want local-only archives without publishing them to GitHub.

## Third-Party Components

MoViD keeps its upstream motion-reconstruction dependencies, DPVO and ViTPose, under `third-party/`. Their original licenses and attribution remain with those components, while the training, inference, and documentation organization in this repository is centered on MoViD itself.
