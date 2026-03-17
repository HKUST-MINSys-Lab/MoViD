## Python API

Finish the basic installation first ([Installation](INSTALL.md) or [Docker](DOCKER.md)).

`MoViD_edge` does not currently maintain a separate stable Python API wrapper. The supported entrypoints are the CLI programs:

- `python3 demo.py --video <input-video.mp4> --visualize`
- `python3 real_time.py --video realsense --visualize --max_frames 1000`

If you need module-level integration, start from:

- `demo.py` and `real_time.py` for end-to-end runtime flow
- `tools/inference/optimized_streaming.py` for reusable streaming inference logic
