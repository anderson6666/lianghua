"""
AI量化预测系统 - 启动入口
运行方式：python main.py
或：streamlit run main.py
"""
import os
import sys
import importlib

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    modules = ["data", "indicators", "backtest", "predict", "news", "agnes_agent", "optimize"]
    for mod in modules:
        try:
            importlib.reload(sys.modules[mod]) if mod in sys.modules else __import__(mod)
        except ImportError as e:
            print(f"导入模块 {mod} 失败: {e}")
            sys.exit(1)

    print("✅ 所有模块加载成功")

    os.system("streamlit run app.py --server.headless true --server.port 8501")
