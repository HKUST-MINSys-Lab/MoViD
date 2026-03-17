#!/usr/bin/env bash

# 在现有的wham conda环境中安装pyskl

echo "======================================"
echo "安装 pyskl 到 wham conda 环境"
echo "======================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

activate_conda_wham
enter_repo_root

# 检查Python版本
echo "Python版本:"
python --version

# 切换到安装目录
INSTALL_DIR="${PYSKL_HOME:-${HOME}}"
cd "$INSTALL_DIR"

# 检查是否已经存在pyskl目录
if [ -d "pyskl" ]; then
    echo "发现已存在的pyskl目录: $INSTALL_DIR/pyskl"
    echo "使用现有目录..."
    cd pyskl
    git pull || echo "Git pull失败，继续使用现有代码"
else
    echo "克隆pyskl仓库..."
    git clone https://github.com/kennymckormick/pyskl.git
    cd pyskl
fi

echo ""
echo "安装pyskl依赖..."
echo "======================================"

# 安装基础依赖
pip install -q mmcv-full mmengine mmdet mmpose

# 安装pyskl（可编辑模式）
echo "安装pyskl..."
pip install -e .

# 验证安装
echo ""
echo "======================================"
echo "验证安装..."
python -c "import pyskl; print('✓ pyskl安装成功!')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✓ pyskl安装完成！"
    echo "======================================"
    echo ""
    echo "现在可以运行demo.py进行action recognition了："
    echo "  bash scripts/demo/run_demo_har_simple.sh"
    echo ""
else
    echo ""
    echo "======================================"
    echo "✗ pyskl安装失败，请检查错误信息"
    echo "======================================"
    echo ""
    echo "常见问题："
    echo "1. 如果mmcv-full安装失败，可以尝试："
    echo "   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html"
    echo ""
    echo "2. 如果遇到CUDA版本问题，检查PyTorch版本："
    echo "   python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
    echo ""
fi
