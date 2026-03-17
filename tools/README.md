# Tools

This directory contains non-core utilities that support the main MoViD entrypoints.

- `action/`: action-recognition asset download, data preparation, and STGCN++ finetuning
- `data/`: active dataset preprocessing scripts used by the main project
- `eval/`: evaluation helpers that are not part of the main `batch_eval.py` entrypoint

Archived helpers that are no longer part of the active tree now live under `archive/retired_tools/`, including:

- retired analysis and visualization scripts
- retired experimental / research prototypes
- retired one-off dataset conversion scripts such as `tools/data/process_*`

The main runtime entrypoints remain at the repository root:

- `demo.py`
- `train.py`
- `batch_eval.py`
- `movid_api.py` (compatibility API entrypoint)
