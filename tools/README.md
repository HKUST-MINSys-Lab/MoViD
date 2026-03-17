# Tools

This directory contains non-core utilities that support the main MoViD entrypoints.

- `action/`: action-recognition asset download, data preparation, and STGCN++ finetuning
- `analysis/`: analysis and visualization helpers
- `data/`: dataset preprocessing and local conversion scripts
- `eval/`: evaluation helpers that are not part of the main `batch_eval.py` entrypoint
- `research/`: experimental modules and prototype ideas

The main runtime entrypoints remain at the repository root:

- `demo.py`
- `train.py`
- `batch_eval.py`
- `wham_api.py` (compatibility API entrypoint)
