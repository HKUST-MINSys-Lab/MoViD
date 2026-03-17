#!/usr/bin/env bash

# 修复 torchvision 版本兼容性问题
# PyTorch 2.8.0 需要 torchvision 0.23.0

echo "正在升级 torchvision 到兼容版本..."
echo "当前版本: PyTorch 2.8.0, torchvision 0.12.0"
echo "目标版本: torchvision 0.23.0"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

activate_conda_wham

# 升级 torchvision
pip install --upgrade torchvision==0.23.0

echo ""
echo "升级完成！请重新运行脚本。"
