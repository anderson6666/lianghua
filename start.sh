#!/bin/bash
echo "================================================"
echo "    量化预测软件 - LiangHua v1.0"
echo "================================================"
echo ""

echo "正在检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误：未找到 Python，请先安装 Python 3.10+"
    exit 1
fi

echo "正在检查并安装依赖..."
pip3 install -r requirements.txt -q

echo ""
echo "正在启动应用..."
echo "浏览器将自动打开 http://localhost:8501"
echo ""

open http://localhost:8501 2>/dev/null || xdg-open http://localhost:8501 2>/dev/null || echo "请手动打开 http://localhost:8501"
streamlit run app.py