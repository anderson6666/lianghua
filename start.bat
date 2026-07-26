@echo off
chcp 65001 >nul
echo ============================================
echo     量化预测软件 - LiangHua v1.0
echo ============================================
echo.

echo 正在检查 Python 环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未找到 Python，请先安装 Python 3.10+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

echo 正在检查并安装依赖...
pip install -r requirements.txt -q

echo.
echo 正在启动应用...
echo 浏览器将自动打开 http://localhost:8501
echo.

start http://localhost:8501
streamlit run app.py

pause