# Install pyskl

## Option 1: Use the helper script

```bash
bash scripts/setup/install_pyskl.sh
```

## Option 2: Manual install

```bash
conda activate movid
cd ~
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
pip install mmcv-full mmengine mmdet mmpose
pip install -e .
```

## Verify

```bash
python -c "import pyskl; print('pyskl installed successfully')"
```

If `mmcv-full` fails, install the wheel that matches your CUDA and PyTorch versions.
