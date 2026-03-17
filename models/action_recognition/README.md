Action-recognition configs live here.

Large assets are intentionally not versioned:

- `stgcn++_ntu60_xsub_3d.pth`
- `stgcn++_ntu60_xsub_3d_labels.txt`

To download them locally, run:

```bash
python tools/action/download_stgcn_model.py
```

Helper scripts will also look for the checkpoint in `checkpoints/action_recognition/`.
