# Quick Start: HAR Demo

## 1. Download STGCN++ assets

```bash
python tools/action/download_stgcn_model.py
```

This stores the action-recognition config, checkpoint, and label map under `models/action_recognition/`.

## 2. Run the demo

```bash
bash scripts/demo/run_demo_with_har.sh examples/demo_video.mp4 output/demo_har
```

You can also point it at your own video:

```bash
bash scripts/demo/run_demo_with_har.sh /path/to/video.mp4 output/my_demo
```

## 3. Expected outputs

- `wham_output.pkl`
- `action_recognition_summary.pkl` when HAR assets are available
- visualization videos when `--visualize` is enabled

If the action-recognition checkpoint or label map is missing, the wrapper script automatically falls back to WHAM-only inference.
