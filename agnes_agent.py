"""
Agnes AI 代理模块 - 生成自然语言报告
Agnes AI 是新加坡 Sapiens AI 推出的免费 AI 服务，支持 OpenAI 兼容接口。
官网：https://platform.agnes-ai.com
Base URL: https://apihub.agnes-ai.com/v1
免费模型: Agnes-2.0-Flash (1M上下文)
"""
import json
import os
import time

import pandas as pd

import requests
import streamlit as st


AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"
AGNES_MODEL = "Agnes-2.0-Flash"


def get_agnes_api_key() -> str:
    """获取 Agnes API Key（优先从环境变量，其次从 Streamlit Secrets）"""
    key = os.environ.get("AGNES_API_KEY")
    if not key:
        try:
            key = st.secrets.get("agnes", {}).get("api_key", "")
        except Exception:
            key = ""
    return key


def call_agnes(messages: list, api_key: str = None) -> str:
    """调用 Agnes AI 生成文本"""
    if not api_key:
        api_key = get_agnes_api_key()

    if not api_key:
        raise ValueError("请设置 AGNES_API_KEY 环境变量或在 Streamlit Secrets 中配置。")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "LiangHua-Stock-Analyzer/1.0",
    }

    payload = {
        "model": AGNES_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{AGNES_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            
            if resp.status_code == 503:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    raise RuntimeError("Agnes AI 服务暂时不可用（503），请稍后重试。")
            
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise RuntimeError(f"Agnes API 调用失败: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Agnes API 处理失败: {str(e)}")


def generate_daily_report(
    symbol: str, market: str, df: pd.DataFrame, signals: dict,
    ml_result: dict, bt_result: dict, api_key: str = None
) -> str:
    """生成综合每日报告"""
    if df.empty:
        return "数据不足，无法生成报告。"

    last_date = df.index[-1].date()
    prev_date = df.index[-2].date() if len(df) > 1 else last_date

    last_price = df["close"].iloc[-1]
    prev_price = df["close"].iloc[-2] if len(df) > 1 else last_price
    daily_change = ((last_price - prev_price) / prev_price * 100).round(2)

    signal_summary = "\n".join([
        f"- {name}: {text} ({'🟢看多' if tag == 'bull' else '🔴看空' if tag == 'bear' else '⚪中性'})"
        for name, (text, tag) in signals.items()
    ])

    bt_text = ""
    if bt_result and "metrics" in bt_result:
        m = bt_result["metrics"]
        bt_text = f"""
回测绩效：
- 累计收益率：{m.get('累计收益率', 0) * 100:.2f}%
- 年化收益率：{m.get('年化收益率', 0) * 100:.2f}%
- 夏普比率：{m.get('夏普比率', 0):.2f}
- 最大回撤：{m.get('最大回撤', 0) * 100:.2f}%
"""

    prompt = f"""
你是一位专业的股票量化分析师。请基于以下数据，生成一份详细的市场分析报告。

【基本信息】
- 股票代码：{symbol}
- 市场：{market}
- 报告日期：{last_date}
- 最新收盘价：{last_price}
- 当日涨跌幅：{daily_change}%

【技术指标信号】
{signal_summary}

【策略回测】
{bt_text}

请按照以下结构生成报告：

1. 📊 今日局势总结
   - 价格走势分析
   - 关键技术指标解读

2. 🔍 深层洞察
   - 当前市场结构判断
   - 技术形态分析

3. 💡 策略建议
   - 基于技术指标的操作建议
   - 策略配置建议

4. ⚠️ 风险提示
   - 潜在风险因素
   - 注意事项

要求：
- 语言专业但易懂，适合普通投资者阅读
- 明确区分"事实数据"与"预测判断"
- 避免绝对化表述，使用概率性语言
- 不要超过1500字
"""

    messages = [
        {"role": "system", "content": "你是一位专业的股票量化分析师，擅长将复杂的金融数据转化为清晰易懂的投资洞察。你的分析基于数据驱动，保持客观理性，同时具备深度的市场理解能力。"},
        {"role": "user", "content": prompt},
    ]

    return call_agnes(messages, api_key)


def generate_optimization_report(history: list, api_key: str = None) -> str:
    """生成策略优化报告"""
    if not history:
        return "无历史数据，无法生成优化报告。"

    prompt = f"""
你是一位专业的量化策略优化专家。请分析以下回测历史数据，提出改进建议。

【回测历史记录】
{json.dumps(history, ensure_ascii=False, indent=2)}

请分析：
1. 当前策略的表现趋势（收益、风险、稳定性）
2. 存在的问题与改进空间
3. 参数调整建议
4. 策略适配性分析（不同市场环境下的表现差异）

要求：专业、具体、可执行。
"""

    messages = [
        {"role": "system", "content": "你是一位专业的量化策略优化专家，擅长通过数据分析发现策略改进机会。"},
        {"role": "user", "content": prompt},
    ]

    return call_agnes(messages, api_key)
