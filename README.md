# LiangHua - 股票量化预测软件

全免费开源的股票量化预测软件，支持 A股/美股/港股/加密货币，包含机器学习趋势预测、策略回测、AI分析报告等功能。

## ✨ 功能特性

- 📊 **多市场支持**：A股（baostock）
- 📈 **技术指标分析**：MA、MACD、RSI、布林带、KDJ 等
- 🤖 **机器学习预测**：多模型集成（XGBoost + LightGBM + RandomForest）+ 三分类策略
- 📉 **策略回测**：4种经典策略 + 自适应策略推荐 + 回测结果自动记录
- 📝 **AI分析师报告**：Agnes AI 生成自然语言分析报告
- 🔄 **自我优化**：置信度过滤、超参数优化（Optuna）、特征选择

## 🚀 快速开始

### 方法一：一键启动（推荐）

**Windows**：
```bash
双击 start.bat
```

**Linux/Mac**：
```bash
chmod +x start.sh
./start.sh
```

### 方法二：手动启动

```bash
# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run app.py
```

### 访问地址
浏览器打开 `http://localhost:8501`

## 📦 依赖

```
streamlit>=1.30.0
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.18.0
scikit-learn>=1.2.0
akshare>=1.12.0
yfinance>=0.2.40
baostock>=0.8.8
xgboost>=2.0.0
lightgbm>=4.0.0
optuna>=3.0.0
```

## 🎯 使用说明

1. **选择市场**：A股、美股、港股、加密货币
2. **输入代码**：
   - A股：`600519`（贵州茅台）、`000001`（平安银行）
   - 美股：`AAPL`、`TSLA`、`MSFT`
   - 港股：`00700`（腾讯）、`09988`（阿里）
   - 加密货币：`BTC-USD`、`ETH-USD`
3. **选择日期范围**
4. **点击「开始分析」**



## 📁 项目结构

```
LiangHua/
├── app.py              # Streamlit 主界面
├── data.py             # 数据获取模块
├── indicators.py       # 技术指标计算
├── backtest.py         # 策略回测引擎
├── predict.py          # 机器学习预测模块
├── agnes_agent.py      # Agnes AI 调用接口
├── optimize.py         # 自我优化模块
├── main.py             # 启动入口
├── start.bat           # Windows 一键启动
├── start.sh            # Linux/Mac 一键启动
├── requirements.txt    # 依赖清单
├── .gitignore          # Git 忽略文件
└── .streamlit/
    └── secrets.toml.example  # API Key 配置模板
```

## ⚠️ 免责声明

本软件仅用于技术学习与研究，所有分析、预测、回测结果均为历史数据统计与概率参考，**不构成任何投资建议**。金融市场无法被准确预测，据此投资风险自负。

## 📄 开源协议

MIT License

