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
- `tools/`: action, data, eval, analysis, and research utilities
- `docs/`: installation notes, dataset notes, and extra guides

## Repository Layout

```text
.
|-- configs/
|-- docs/
|   |-- guides/
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
|   |-- analysis/
|   |-- data/
|   |-- eval/
|   `-- research/
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

### 3. Run a demo

```bash
bash scripts/demo/run_demo_with_har.sh <input-video.mp4> output/demo_har
```

If action-recognition assets are missing, the script will fall back to base reconstruction-only inference.

## Common Workflows

- Demo with action recognition: [scripts/demo/run_demo_with_har.sh](scripts/demo/run_demo_with_har.sh)
- Batch demo over a folder: [scripts/demo/run_demo.sh](scripts/demo/run_demo.sh)
- Stage-2 training: [scripts/train/run_stage2_train.sh](scripts/train/run_stage2_train.sh)
- Evaluation: [scripts/eval/run_eval.sh](scripts/eval/run_eval.sh)
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

## Third-Party Components

MoViD keeps its upstream motion-reconstruction component, DPVO, and ViTPose under `third-party/`. Their original licenses and attribution remain with those components, while the repository structure, entrypoints, and workflow documentation in this repo are organized around MoViD itself.
