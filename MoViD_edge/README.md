# MoViD Edge

`MoViD_edge` is the edge-side inference project for MoViD. It lives inside the main MoViD repository, and the active maintained entrypoints are `real_time.py` and `demo.py`.

For the full camera-ready release flow, including installation, root training, evaluation, and all inference modes, start from [docs/CAMERA_READY.md](../docs/CAMERA_READY.md).

## Active Entry Points

- `real_time.py`: recommended edge runtime; supports camera/video input, action recognition, and flip evaluation
- `demo.py`: offline / recorded-video inference entrypoint

Support modules are now organized under:

- [tools/inference](tools/README.md): the active reusable inference helper used by `real_time.py`
- [tools/model](tools/README.md): action-model download and TensorRT conversion tools
- [docs/guides](docs/guides): active edge-specific usage notes

## Quick Start

### 1. Install dependencies

See [docs/INSTALL.md](docs/INSTALL.md) for the base MoViD edge environment.

### 2. Download action-recognition assets if needed

```bash
python3 tools/model/download_action_model.py stgcn_ntu60_xsub_3d
```

### 3. Run edge inference

`MoViD_edge` keeps the edge-side inference entrypoints only. The active modes are:

- offline recorded-video inference with `demo.py`
- real-time streaming inference with `real_time.py`
- flip-eval enhanced streaming inference with `real_time.py --flip_eval`

Recommended order:

1. Run `demo.py` on a recorded video first
2. Run `real_time.py` on a recorded video through the streaming pipeline
3. Switch `real_time.py --video realsense` for live deployment

Recorded video with `demo.py`:

```bash
python3 demo.py --video <input-video.mp4> --visualize
```

RealSense / edge camera with `real_time.py`:

```bash
python3 real_time.py --video realsense --visualize --max_frames 1000
```

Recorded video through the real-time streaming pipeline:

```bash
python3 real_time.py --video <input-video.mp4> --visualize --max_frames 1000
```

Streaming inference with flip evaluation:

```bash
python3 real_time.py --video realsense --visualize --flip_eval --flip_select all --max_frames 1000
```

## Repository Layout

```text
.
|-- configs/
|-- docs/
|   `-- guides/
|-- lib/
|   `-- data/
|-- models/
|   `-- action_recognition/
|-- tools/
|   |-- inference/
|   `-- model/
|-- demo.py
`-- real_time.py
```

## Guides

- [Action Recognition](docs/guides/action-recognition.md)

## Notes

- Large artifacts such as videos, model weights, checkpoints, TensorRT engines, and outputs are intentionally ignored by Git.
- Retired entrypoints and experiments have been moved out of the active tree into `.local/retired_*`.
- Upstream third-party components remain under their original licenses; the maintained edge workflow in this directory is organized as part of MoViD.
