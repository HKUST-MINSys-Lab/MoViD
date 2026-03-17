# Action Recognition Guide

MoViD can extract NTU-style 3D joints from MoViD output and run STGCN++ on top of those joints.

## Relevant Files

- `lib/action_recognition.py`: online action-recognition wrapper
- `tools/action/prepare_action_data.py`: convert MoViD outputs into training-ready skeleton data
- `tools/action/finetune_stgcn.py`: finetune STGCN++ on custom skeleton sequences
- `tools/action/download_stgcn_model.py`: fetch pretrained STGCN++ assets

## Prepare Training Data

```bash
python tools/action/prepare_action_data.py \
    --movid_output /path/to/movid_outputs \
    --output /path/to/action_data.pkl \
    --config configs/yamls/stage2.yaml \
    --checkpoint /path/to/movid_checkpoint.pth.tar \
    --window_size 100 \
    --device cuda:0
```

## Finetune STGCN++

```bash
python tools/action/finetune_stgcn.py \
    --config models/action_recognition/stgcn++_ntu60_xsub_3d_config.py \
    --checkpoint /path/to/pretrained_stgcnpp.pth \
    --data /path/to/train_data.pkl \
    --val_data /path/to/val_data.pkl \
    --label_map /path/to/labels.txt \
    --work_dir work_dirs/finetune_stgcn \
    --epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --device cuda:0
```

## Demo Inference

```bash
bash scripts/demo/run_demo_with_har.sh examples/demo_video.mp4 output/demo_har
```

## Notes

- STGCN++ normally expects 100-frame windows.
- Downloaded model weights are intentionally ignored by Git.
- If you use a custom checkpoint, pass it with `ACTION_CHECKPOINT=/path/to/model.pth`.
