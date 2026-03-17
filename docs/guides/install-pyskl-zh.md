# pyskl 安装说明

## 方式 1：直接使用脚本

```bash
bash scripts/setup/install_pyskl.sh
```

## 方式 2：手动安装

```bash
conda activate wham
cd ~
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
pip install mmcv-full mmengine mmdet mmpose
pip install -e .
```

## 安装完成后

```bash
python tools/action/download_stgcn_model.py
bash scripts/demo/run_demo_har_simple.sh
```

如果 `mmcv-full` 安装失败，需要换成与你当前 CUDA / PyTorch 匹配的 wheel。
