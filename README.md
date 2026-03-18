# MoViD

MoViD is a research codebase for monocular 3D human motion reconstruction and activity understanding, extended with:

- action recognition on reconstructed NTU-style 3D joints
- MotionGPT-based motion-to-text prediction
- dataset preprocessing and batch evaluation utilities

This repository has been cleaned up for GitHub so that source code, configs, docs, and reusable scripts stay in version control, while datasets, videos, logs, checkpoints, and other large local artifacts stay out.

## Highlights

- `demo.py`: main inference entrypoint
- `train.py`: model training entrypoint
- `batch_eval.py`: batch evaluation utility
- `lib/`: core models, data loaders, evaluation, and visualization
- `configs/`: runtime and training configs
- `scripts/`: organized helper scripts for setup, demo, evaluation, training, and data prep
- `tools/`: active action, data, and evaluation utilities
- `docs/`: installation notes, dataset notes, and extra guides

## Repository Layout

```text
.
|-- configs/
|-- docs/
|   |-- guides/
|-- archive/
|   `-- retired_tools/
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
|-- third-party/
|-- batch_eval.py
|-- demo.py
|-- train.py
`-- movid_api.py
```

## Quick Start

### 1. Install the environment

See [docs/INSTALL.md](docs/INSTALL.md) for the base environment setup, then use:

```bash
bash scripts/setup/install_environment.sh
```

If you want action recognition support:

```bash
bash scripts/setup/install_pyskl.sh
python tools/action/download_stgcn_model.py
```

### 2. Fetch demo assets

```bash
bash scripts/setup/fetch_demo_data.sh
```

### 3. Train the model

Stage-2 training:

```bash
python train.py --cfg configs/yamls/stage2.yaml
```

If you are running in a constrained environment where multi-worker dataloaders are blocked, use:

```bash
python train.py --cfg configs/yamls/stage2.yaml NUM_WORKERS 0
```

### 4. Evaluate a checkpoint

Use the helper script:

```bash
bash scripts/eval/run_eval.sh
```

Or run the batch evaluator directly:

```bash
python batch_eval.py \
  --folders <sequence_dir_1> <sequence_dir_2> \
  --output_base output/batch_eval \
  --gt_checkpoint <ground-truth-checkpoint> \
  --pred_checkpoint <prediction-checkpoint>
```

### 5. Run inference

MoViD supports several inference modes from the main repository and the edge subproject.

Offline full-video inference:

```bash
python demo.py --video <input-video.mp4> --output_pth output/demo --visualize
```

Stream inference with `network.stream_inference()`:

```bash
python demo.py --video <input-video.mp4> --mode stream --stream_window_size 10 --output_pth output/stream --visualize
```

Demo with action recognition:

```bash
bash scripts/demo/run_demo_with_har.sh <input-video.mp4> output/demo_har
```

Simple API-style CLI wrapper:

```bash
python movid_api.py --video <input-video.mp4> --output_dir output/api_demo --visualize
```

Edge offline inference:

```bash
python MoViD_edge/demo.py --video <input-video.mp4> --output_pth output/edge_demo --visualize
```

Edge real-time / streaming inference:

```bash
python MoViD_edge/real_time.py --video realsense --output_pth output/edge_rt --visualize --max_frames 1000
```

If action-recognition assets are missing, the HAR helper script will fall back to base reconstruction-only inference.

## Common Workflows

- Stage-2 training: [train.py](train.py) or [scripts/train/run_stage2_train.sh](scripts/train/run_stage2_train.sh)
- Offline full-video inference: [demo.py](demo.py)
- Stream inference: [demo.py](demo.py)
- API / wrapper inference: [movid_api.py](movid_api.py)
- Batch evaluation: [batch_eval.py](batch_eval.py) or [scripts/eval/run_eval.sh](scripts/eval/run_eval.sh)
- Demo with action recognition: [scripts/demo/run_demo_with_har.sh](scripts/demo/run_demo_with_har.sh)
- Batch demo over a folder: [scripts/demo/run_demo.sh](scripts/demo/run_demo.sh)
- Edge offline inference: [MoViD_edge/demo.py](MoViD_edge/demo.py)
- Edge real-time inference: [MoViD_edge/real_time.py](MoViD_edge/real_time.py)
- HuMMan preprocessing loop: [scripts/data/run_all_views.sh](scripts/data/run_all_views.sh)

## Extra Guides

- [Action Recognition Guide](docs/guides/action-recognition.md)
- [HAR Quick Start](docs/guides/quick-start-har.md)
- [Install pyskl](docs/guides/install-pyskl.md)
- [NTU Transfer Notes](docs/guides/ntu-transfer.md)

## What Stays Out of GitHub

The `.gitignore` now excludes:

- datasets and downloaded examples
- logs and output folders
- checkpoints and pretrained weights
- videos and other large binary artifacts
- temporary HTML/cookie/debug files
- local backup scripts under `.local/`

If you already have local weights, place them under `checkpoints/` or use environment variables such as `MOVID_CHECKPOINT` and `ACTION_CHECKPOINT` when running the helper scripts.

Some non-core analysis, research, and one-off data-conversion helpers have been moved into `archive/retired_tools/` so the active tree stays focused on training and full inference workflows while the older scripts remain available in version control.

## Third-Party Components

MoViD keeps its upstream motion-reconstruction component, DPVO, and ViTPose under `third-party/`. Their original licenses and attribution remain with those components, while the repository structure, entrypoints, and workflow documentation in this repo are organized around MoViD itself.
