"""
策略与回测引擎
提供若干经典策略，生成买卖信号并进行向量化回测，输出净值曲线与绩效指标。
"""
import numpy as np
import pandas as pd

from indicators import add_indicators


STRATEGIES = ["双均线交叉(MA5/MA20)", "MACD金叉死叉", "RSI超买超卖", "布林带突破"]


def generate_signals(df: pd.DataFrame, strategy: str) -> pd.Series:
    """
    返回持仓信号 Series：1 = 持有多头，0 = 空仓。
    信号在当日收盘产生，次日开盘（这里近似为次日收盘）生效以避免未来函数。
    """
    d = add_indicators(df)
    pos = pd.Series(0, index=d.index, dtype=float)

    if strategy == "双均线交叉(MA5/MA20)":
        pos[d["MA5"] > d["MA20"]] = 1

    elif strategy == "MACD金叉死叉":
        pos[d["DIF"] > d["DEA"]] = 1

    elif strategy == "RSI超买超卖":
        # RSI 上穿 30 买入，下穿 70 卖出，中间保持
        state = 0
        vals = []
        for r in d["RSI"]:
            if np.isnan(r):
                vals.append(0)
                continue
            if r <= 30:
                state = 1
            elif r >= 70:
                state = 0
            vals.append(state)
        pos = pd.Series(vals, index=d.index, dtype=float)

    elif strategy == "布林带突破":
        # 收盘价上穿中轨买入，下穿中轨卖出
        pos[d["close"] > d["BOLL_MID"]] = 1

    else:
        raise ValueError(f"未知策略: {strategy}")

    # 信号延迟一天生效，避免用当日信号交易当日价格（未来函数）
    return pos.shift(1).fillna(0)


def run_backtest(df: pd.DataFrame, strategy: str, fee: float = 0.0003,
                 init_capital: float = 100000.0) -> dict:
    """
    向量化回测。
    fee: 单边手续费率（含滑点近似）。
    返回包含净值曲线、绩效指标、交易信号的字典。
    """
    d = df.copy()
    pos = generate_signals(d, strategy)
    ret = d["close"].pct_change().fillna(0)

    # 策略每日收益 = 持仓 * 当日收益
    strat_ret = pos * ret
    # 换手时扣手续费
    trades = pos.diff().abs().fillna(0)
    strat_ret = strat_ret - trades * fee

    equity = (1 + strat_ret).cumprod() * init_capital
    bench = (1 + ret).cumprod() * init_capital

    metrics = _metrics(strat_ret, equity)
    metrics["交易次数"] = int((trades > 0).sum())

    return {
        "equity": equity,
        "benchmark": bench,
        "position": pos,
        "strat_ret": strat_ret,
        "metrics": metrics,
    }


def _metrics(strat_ret: pd.Series, equity: pd.Series) -> dict:
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    days = len(strat_ret)
    ann_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    max_dd = drawdown.min()

    # 胜率
    win_days = (strat_ret > 0).sum()
    trade_days = (strat_ret != 0).sum()
    win_rate = win_days / trade_days if trade_days > 0 else 0

    return {
        "累计收益率": total_return,
        "年化收益率": ann_return,
        "年化波动率": ann_vol,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
        "日胜率": win_rate,
    }
