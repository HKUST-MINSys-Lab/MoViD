# Camera-Ready Code Release

This document describes what is included in the public MoViD code release and how to run the code step by step, from environment setup to training, evaluation, and inference.

## 1. What Is Included

The camera-ready GitHub release includes:

- maintained source code for training, evaluation, and inference
- the edge-side inference project under `MoViD_edge/`
- configs, scripts, and documentation needed to launch the main workflows
- bundled third-party submodules under `third-party/`

The release intentionally does not include:

- datasets
- pretrained checkpoints
- local logs and outputs
- large videos, archives, and other generated artifacts

## 2. Recommended Reading Order

If you are new to the project, use this order:

1. [Installation](INSTALL.md)
2. [Dataset Preparation](DATASET.md) if you want to train or evaluate
3. This document for the execution flow
4. [Python API](API.md) if you want to call MoViD from your own code
5. [MoViD Edge README](../MoViD_edge/README.md) if you only need edge inference

## 3. Repository Entry Points

- `train.py`: train the root MoViD model
- `demo.py`: run offline full-video inference or stream inference from the main repository
- `batch_eval.py`: run batch evaluation utilities
- `movid_api.py`: lightweight wrapper around the main inference entrypoint
- `MoViD_edge/demo.py`: edge-side offline inference
- `MoViD_edge/real_time.py`: edge-side real-time streaming inference

## 4. Step-by-Step Workflow

### Step 1. Clone the repository with submodules

```bash
git clone <your-movid-repo-url> --recursive
cd MoViD
```

### Step 2. Install the environment

Use the detailed instructions in [INSTALL.md](INSTALL.md), or start from the helper script:

```bash
bash scripts/setup/install_environment.sh
```

This repository expects a Python 3.9 environment and installs PyTorch, ViTPose, DPVO, and the MoViD Python dependencies.

If you also want action recognition support:

```bash
bash scripts/setup/install_pyskl.sh
python tools/action/download_stgcn_model.py
```

### Step 3. Download demo assets and default checkpoints

```bash
bash scripts/setup/fetch_demo_data.sh
```

This script downloads:

- SMPL body-model assets into `dataset/body_models/`
- default MoViD checkpoints into `checkpoints/`
- demo videos into `examples/`

### Step 4. Run a minimal offline inference test

```bash
python demo.py \
  --video examples/demo_video.mp4 \
  --output_pth output/demo \
  --visualize
```

This is the recommended first smoke test for the root repository.

### Step 5. Run stream inference from the root repository

```bash
python demo.py \
  --video examples/demo_video.mp4 \
  --mode stream \
  --stream_window_size 10 \
  --output_pth output/stream \
  --visualize
```

This path uses `network.stream_inference()` and is useful if you want frame-by-frame behavior while staying in the main repository.

### Step 6. Run the compatibility API wrapper

```bash
python movid_api.py \
  --video examples/demo_video.mp4 \
  --output_dir output/api_demo \
  --visualize
```

You can also import it from Python; see [API.md](API.md).

### Step 7. Prepare datasets for training or evaluation

Follow [DATASET.md](DATASET.md) to prepare:

- training datasets under `dataset/parsed_data/`
- detection results under `dataset/detection_results/`
- evaluation datasets such as 3DPW, RICH, and EMDB

If you only want to run demos or inference, you can skip this step.

### Step 8. Launch training

The standard stage-2 training command is:

```bash
python train.py --cfg configs/yamls/stage2.yaml
```

If you are working in an environment where multi-worker dataloaders are not allowed, use:

```bash
python train.py --cfg configs/yamls/stage2.yaml NUM_WORKERS 0
```

There is also a background helper script:

```bash
bash scripts/train/run_stage2_train.sh
```

### Step 9. Run evaluation

Evaluate a checkpoint on a supported benchmark:

```bash
bash scripts/eval/run_eval.sh 3dpw checkpoints/movid_vit_w_3dpw.pth.tar
```

Supported targets in the helper script:

- `3dpw`
- `rich`
- `emdb1`
- `emdb2`

For batch-style evaluation over a set of folders, use:

```bash
python batch_eval.py \
  --folders <sequence_dir_1> <sequence_dir_2> \
  --output_base output/batch_eval \
  --gt_checkpoint <ground-truth-checkpoint> \
  --pred_checkpoint <prediction-checkpoint>
```

### Step 10. Run edge-side offline inference

```bash
python MoViD_edge/demo.py \
  --video examples/demo_video.mp4 \
  --output_pth output/edge_demo \
  --visualize
```

This is the simplest entrypoint if you want to test the edge project on a recorded video.

### Step 11. Run edge-side real-time streaming inference

```bash
python MoViD_edge/real_time.py \
  --video realsense \
  --output_pth output/edge_rt \
  --visualize \
  --max_frames 1000
```

You can also feed a recorded video through the same streaming pipeline:

```bash
python MoViD_edge/real_time.py \
  --video examples/demo_video.mp4 \
  --output_pth output/edge_rt_video \
  --visualize \
  --max_frames 1000
```

To enable flip-eval streaming:

```bash
python MoViD_edge/real_time.py \
  --video realsense \
  --output_pth output/edge_rt_flip \
  --visualize \
  --flip_eval \
  --flip_select all \
  --max_frames 1000
```

## 5. Which Pipeline Should I Use?

Use the root repository if you need:

- training
- evaluation
- full offline inference
- root-level stream inference
- API integration

Use `MoViD_edge/` if you need:

- edge deployment
- camera / RealSense streaming
- lightweight edge-side offline inference
- flip-eval real-time inference

## 6. Files and Folders You Should Expect Locally

After setup, a typical local workspace will contain:

```text
MoViD/
|-- checkpoints/
|-- dataset/
|   |-- body_models/
|   `-- parsed_data/
|-- examples/
|-- output/
`-- logs/
```

Most of these are local runtime assets and are intentionally ignored by Git.

## 7. Notes for Reproducibility

- Use the same config path each time, for example `configs/yamls/stage2.yaml` for training.
- Keep your checkpoints under `checkpoints/` unless you override them explicitly.
- If action-recognition assets are missing, the helper script `scripts/demo/run_demo_with_har.sh` falls back to MoViD-only inference.
- `MoViD_edge/` is an inference-only subproject; the full training and evaluation pipeline stays at the repository root.
