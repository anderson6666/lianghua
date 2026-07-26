"""
自我优化模块 - 回测反馈驱动策略调整
功能：
1. 保存回测历史记录
2. 自动评估策略表现
3. 基于历史表现优化参数
4. 动态调整策略配置
"""
import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from backtest import STRATEGIES, run_backtest


HISTORY_FILE = "backtest_history.json"


def load_history() -> List[Dict]:
    """加载回测历史记录"""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history: List[Dict]):
    """保存回测历史记录"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def record_backtest(symbol: str, market: str, strategy: str,
                    metrics: Dict, date_range: str):
    """记录单次回测结果"""
    history = load_history()
    record = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "market": market,
        "strategy": strategy,
        "date_range": date_range,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    history.append(record)
    if len(history) > 100:
        history = history[-100:]
    save_history(history)


def evaluate_strategy(strategy: str, history: List[Dict]) -> Dict:
    """评估策略历史表现"""
    records = [r for r in history if r["strategy"] == strategy]
    if not records:
        return {"count": 0, "avg_return": 0, "avg_sharpe": 0, "best": None, "worst": None}

    returns = [r["metrics"]["累计收益率"] for r in records]
    sharpes = [r["metrics"]["夏普比率"] for r in records]
    max_dds = [r["metrics"]["最大回撤"] for r in records]

    return {
        "count": len(records),
        "avg_return": round(np.mean(returns) * 100, 2),
        "avg_sharpe": round(np.mean(sharpes), 2),
        "avg_max_dd": round(np.mean(max_dds) * 100, 2),
        "best_return": round(max(returns) * 100, 2),
        "worst_return": round(min(returns) * 100, 2),
        "volatility": round(np.std(returns) * 100, 2),
    }


def find_best_strategy(history: List[Dict], market: str = None) -> str:
    """根据历史回测找到最优策略"""
    records = history
    if market:
        records = [r for r in records if r["market"] == market]

    strategy_scores = {}
    for record in records:
        strat = record["strategy"]
        sharpe = record["metrics"]["夏普比率"]
        return_ = record["metrics"]["累计收益率"]
        dd = abs(record["metrics"]["最大回撤"])
        score = sharpe * 0.5 + return_ * 0.3 - dd * 0.2
        if strat not in strategy_scores:
            strategy_scores[strat] = []
        strategy_scores[strat].append(score)

    if not strategy_scores:
        return STRATEGIES[0]

    avg_scores = {k: np.mean(v) for k, v in strategy_scores.items()}
    return max(avg_scores, key=avg_scores.get)


def optimize_parameters(df: pd.DataFrame, strategy: str,
                        param_grid: Dict = None) -> Dict:
    """参数优化 - 网格搜索"""
    if param_grid is None:
        param_grid = {
            "fee": [0.0001, 0.0003, 0.0005],
        }

    best_params = None
    best_score = float("-inf")

    for fee in param_grid["fee"]:
        try:
            bt = run_backtest(df, strategy, fee=fee, init_capital=100000)
            m = bt["metrics"]
            score = m["夏普比率"] * 0.5 + m["累计收益率"] * 0.3 - abs(m["最大回撤"]) * 0.2
            if score > best_score:
                best_score = score
                best_params = {"fee": fee}
        except Exception:
            continue

    return best_params or {"fee": 0.0003}


def adaptive_strategy_switch(df: pd.DataFrame) -> str:
    """自适应策略选择 - 根据市场状态推荐策略"""
    volatility = df["close"].pct_change().std() * np.sqrt(252)
    trend_strength = abs(df["close"].pct_change().mean() * 252)

    if volatility > 0.3:
        return "布林带突破"
    elif trend_strength > 0.15:
        return "双均线交叉(MA5/MA20)"
    else:
        return "RSI超买超卖"


def generate_self_report(history: List[Dict]) -> str:
    """生成自我分析报告"""
    if not history:
        return "暂无回测历史数据。"

    report = ["## 🤖 量化系统自我分析报告\n"]

    recent = history[-5:] if len(history) >= 5 else history
    report.append("### 最近5次回测记录\n")
    for i, r in enumerate(recent, 1):
        report.append(f"{i}. {r['symbol']}({r['market']}) - {r['strategy']}")
        m = r["metrics"]
        report.append(f"   收益率: {m['累计收益率']*100:.2f}% | 夏普: {m['夏普比率']:.2f} | 回撤: {m['最大回撤']*100:.2f}%")
        report.append(f"   时间: {r['date_range']}")
        report.append("")

    report.append("### 策略表现评估\n")
    for strat in STRATEGIES:
        eval_result = evaluate_strategy(strat, history)
        if eval_result["count"] > 0:
            report.append(f"- **{strat}**: 测试{eval_result['count']}次 | 平均收益{eval_result['avg_return']}% | 夏普{eval_result['avg_sharpe']}")

    best_strat = find_best_strategy(history)
    report.append(f"\n### 当前最优策略\n推荐策略: **{best_strat}**\n")

    report.append("### 系统健康度\n")
    avg_sharpe = np.mean([r["metrics"]["夏普比率"] for r in history])
    avg_return = np.mean([r["metrics"]["累计收益率"] for r in history])
    if avg_sharpe > 0.5 and avg_return > 0:
        report.append("✅ 系统状态: 健康 - 策略表现良好")
    elif avg_sharpe > 0:
        report.append("⚠️ 系统状态: 一般 - 需要优化")
    else:
        report.append("❌ 系统状态: 不佳 - 建议调整策略")

    return "\n".join(report)
