"""
数据获取模块（仅支持A股）
数据源：
  - A股：baostock（首选，免费无需注册）、akshare（备用）
"""
import random
import time

import pandas as pd


def _retry_on_failure(max_retries=3, delay=2, backoff_factor=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (backoff_factor ** attempt) + random.uniform(0.5, 1.5)
                        time.sleep(sleep_time)
            raise last_error
        return wrapper
    return decorator


@_retry_on_failure(max_retries=3, delay=3)
def get_stock_data(market: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    统一数据获取入口（仅支持A股）。
    返回标准化 DataFrame，列：date, open, high, low, close, volume（date 为索引）。

    market: 'A股'
    symbol: 代码。如 '600519'
    start/end: 'YYYY-MM-DD'
    """
    market = market.strip()
    symbol = symbol.strip()

    if market != "A股":
        raise ValueError("本系统仅支持A股市场")

    # 优先使用baostock（免费无需注册），akshare备用
    df = _get_baostock_ashare(symbol, start, end)
    if df.empty:
        df = _get_ashare(symbol, start, end)

    if df is None or df.empty:
        raise ValueError("未获取到数据，请检查代码或日期范围是否正确。")
    df = df.sort_index()
    return df


def get_macro_data(market: str, start: str, end: str) -> pd.DataFrame:
    """获取宏观因子数据"""
    return pd.DataFrame()


def get_fund_flow(market: str, date: str = None) -> dict:
    """获取资金流向数据"""
    return {}


@_retry_on_failure(max_retries=3, delay=5)
def _get_baostock_ashare(symbol: str, start: str, end: str) -> pd.DataFrame:
    """baostock A股数据（免费无需注册）"""
    import baostock as bs
    import importlib
    
    try:
        # 清理可能残留的坏连接
        try:
            bs.logout()
        except Exception:
            pass
        
        # 强制重新加载模块，重置socket状态
        importlib.reload(bs)
        
        lg = bs.login()
        if lg.error_code != "0":
            try:
                bs.logout()
            except Exception:
                pass
            return pd.DataFrame()
        
        rs = bs.query_history_k_data_plus(
            f"sh.{symbol}" if symbol.startswith("6") else f"sz.{symbol}",
            "date,open,high,low,close,volume",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2",
        )
        
        if rs.error_code != "0":
            try:
                bs.logout()
            except Exception:
                pass
            return pd.DataFrame()

        # 标准baostock数据读取模式
        data_list = []
        while rs.next():
            try:
                row = rs.get_row_data()
                if row:
                    data_list.append(row)
            except Exception:
                break
        
        try:
            bs.logout()
        except Exception:
            pass
        
        if not data_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        return df.astype(float)
    except Exception as e:
        try:
            bs.logout()
        except Exception:
            pass
        return pd.DataFrame()


@_retry_on_failure(max_retries=2, delay=3)
def _get_ashare(symbol: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    try:
        s = start.replace("-", "")
        e = end.replace("-", "")
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily", start_date=s, end_date=e, adjust="qfq"
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
        return df.astype(float)
    except Exception:
        return pd.DataFrame()