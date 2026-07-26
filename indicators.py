"""
技术指标模块
纯 pandas/numpy 实现，无需 TA-Lib 等需编译的依赖。
"""
import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    hist = (dif - dea) * 2
    return dif, dea, hist


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3):
    low_min = df["low"].rolling(n).min()
    high_max = df["high"].rolling(n).max()
    rsv = (df["close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在原始 OHLCV 上追加常用技术指标列。"""
    out = df.copy()
    close = out["close"]
    out["MA5"] = sma(close, 5)
    out["MA10"] = sma(close, 10)
    out["MA20"] = sma(close, 20)
    out["MA60"] = sma(close, 60)
    out["DIF"], out["DEA"], out["MACD"] = macd(close)
    out["RSI"] = rsi(close)
    out["BOLL_UP"], out["BOLL_MID"], out["BOLL_LOW"] = bollinger(close)
    out["K"], out["D"], out["J"] = kdj(out)
    return out


def latest_signals(df: pd.DataFrame) -> dict:
    """
    基于最新一行数据给出简易信号提示。
    返回 {指标名: (状态文字, 'bull'/'bear'/'neutral')}
    """
    d = add_indicators(df)
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) > 1 else last
    sig = {}

    # 均线多空
    if last["MA5"] > last["MA20"]:
        sig["均线(MA5/MA20)"] = ("短期均线在上，偏多", "bull")
    else:
        sig["均线(MA5/MA20)"] = ("短期均线在下，偏空", "bear")

    # MACD 金叉/死叉
    if prev["DIF"] <= prev["DEA"] and last["DIF"] > last["DEA"]:
        sig["MACD"] = ("金叉，看多信号", "bull")
    elif prev["DIF"] >= prev["DEA"] and last["DIF"] < last["DEA"]:
        sig["MACD"] = ("死叉，看空信号", "bear")
    elif last["DIF"] > last["DEA"]:
        sig["MACD"] = ("DIF在DEA上方，偏多", "bull")
    else:
        sig["MACD"] = ("DIF在DEA下方，偏空", "bear")

    # RSI 超买超卖
    r = last["RSI"]
    if r >= 70:
        sig["RSI"] = (f"{r:.1f}，超买，警惕回调", "bear")
    elif r <= 30:
        sig["RSI"] = (f"{r:.1f}，超卖，可能反弹", "bull")
    else:
        sig["RSI"] = (f"{r:.1f}，中性区间", "neutral")

    # 布林带位置
    if last["close"] >= last["BOLL_UP"]:
        sig["布林带"] = ("触及上轨，短期偏强/超买", "bear")
    elif last["close"] <= last["BOLL_LOW"]:
        sig["布林带"] = ("触及下轨，短期偏弱/超卖", "bull")
    else:
        sig["布林带"] = ("运行于通道内，中性", "neutral")

    # KDJ
    if last["K"] > last["D"]:
        sig["KDJ"] = ("K在D上方，偏多", "bull")
    else:
        sig["KDJ"] = ("K在D下方，偏空", "bear")

    return sig
