# Action Recognition on Edge

`real_time.py` supports optional action recognition during edge inference.

## 1. Download the model assets

```bash
python3 tools/model/download_action_model.py posec3d_ntu60_xsub
python3 tools/model/download_action_model.py stgcn_ntu60_xsub_3d
```

The files are stored under `models/action_recognition/`.

## 2. Run edge inference with action recognition

```bash
python3 real_time.py --video realsense --visualize --max_frames 1000 \
    --action_config models/action_recognition/stgcn_ntu60_xsub_3d_config.py \
    --action_checkpoint models/action_recognition/stgcn_ntu60_xsub_3d.pth \
    --action_label_map models/action_recognition/stgcn_ntu60_xsub_3d_labels.txt
```

Or use the demo wrapper:

```bash
python3 real_time.py --video examples/demo_video.mp4 --output_pth output/action_demo --visualize \
    --action_config models/action_recognition/stgcn_ntu60_xsub_3d_config.py \
    --action_checkpoint models/action_recognition/stgcn_ntu60_xsub_3d.pth \
    --action_label_map models/action_recognition/stgcn_ntu60_xsub_3d_labels.txt
```

## Model Options

- `posec3d_ntu60_xsub`: 17-keypoint model
- `stgcn_ntu60_xsub_3d`: 25-keypoint NTU model, better suited for 3D skeleton input
- `stgcn++_ntu60_xsub_3d`: alternative STGCN++ model

## Related Files

- `real_time.py`
- `lib/action_recognition.py`
- `lib/action_recognition_trt.py`
- `tools/model/download_action_model.py`
- `tools/model/convert_action_model_to_tensorrt.py`
