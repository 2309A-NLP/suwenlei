#!/bin/bash
# 家庭记账本智能体 - 启动脚本（Linux/WSL）

echo "=============================================="
echo "家庭记账本智能体 - 启动脚本"
echo "=============================================="

# 切换到项目目录
cd "$(dirname "$0")"

# 检查 Python 环境
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "错误：未找到 Python，请先安装 Python 3.8+"
    exit 1
fi

echo "使用 Python: $($PYTHON --version)"

# 检查依赖
if ! $PYTHON -c "import flask" 2>/dev/null; then
    echo "正在安装依赖..."
    $PYTHON -m pip install -r requirements.txt
fi

# 启动 Web 服务
echo ""
echo "正在启动 Web 服务..."
echo "访问地址：http://localhost:5000"
echo "按 Ctrl+C 停止服务"
echo ""

# 源码位于 研发/ 目录，切换过去再启动（同目录导入 + templates 才能正常工作）
cd ../研发
$PYTHON web_server.py
