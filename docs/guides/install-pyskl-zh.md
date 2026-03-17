# pyskl Installation Guide

## Method 1: Run the script directly

```bash
bash scripts/setup/install_pyskl.sh
```

## Method 2: Install manually

```bash
conda activate movid
cd ~
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
pip install mmcv-full mmengine mmdet mmpose
pip install -e .
```

## After installation

```bash
python tools/action/download_stgcn_model.py
bash scripts/demo/run_demo_har_simple.sh
```

If `mmcv-full` installation fails, switch to a wheel that matches your current CUDA / PyTorch setup.
