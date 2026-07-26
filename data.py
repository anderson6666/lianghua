"""
数据获取模块
支持市场：A股、美股、港股、加密货币
数据源：
  - A股：baostock（首选，免费无需注册）、akshare（备用）、yfinance（备用）
  - 港股：akshare、yfinance
  - 美股/加密：yfinance
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
    统一数据获取入口。
    返回标准化 DataFrame，列：date, open, high, low, close, volume（date 为索引）。

    market: 'A股' | '美股' | '港股' | '加密货币'
    symbol: 代码。A股如 '600519'；美股如 'AAPL'；港股如 '00700'；加密如 'BTC-USD'
    start/end: 'YYYY-MM-DD'
    """
    market = market.strip()
    symbol = symbol.strip()
    
    if market == "A股":
        df = _get_baostock_ashare(symbol, start, end)
        if df.empty:
            df = _get_ashare(symbol, start, end)
        if df.empty:
            df = _get_yf(f"{symbol}.SS", start, end)
    elif market == "港股":
        df = _get_hk(symbol, start, end)
        if df.empty:
            df = _get_yf(f"{symbol}.HK", start, end)
    elif market in ("美股", "加密货币"):
        df = _get_yf(symbol, start, end)
    else:
        raise ValueError(f"不支持的市场类型: {market}")

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


MARKET_INDEX_MAP = {
    "A股": "000300",
    "美股": "^GSPC",
    "港股": "^HSI",
    "加密货币": "BTC-USD",
}


def get_market_index_data(market: str, start: str, end: str) -> pd.DataFrame:
    """获取大盘指数数据，用于特征增强"""
    if market not in MARKET_INDEX_MAP:
        return pd.DataFrame()
    
    index_symbol = MARKET_INDEX_MAP[market]
    
    try:
        if market == "A股":
            return _get_baostock_ashare(index_symbol, start, end)
        else:
            return _get_yf(index_symbol, start, end)
    except Exception:
        return pd.DataFrame()


@_retry_on_failure(max_retries=2, delay=3)
def _get_baostock_ashare(symbol: str, start: str, end: str) -> pd.DataFrame:
    """baostock A股数据（免费无需注册）"""
    import baostock as bs
    try:
        lg = bs.login()
        if lg.error_code != "0":
            return pd.DataFrame()
        
        rs = bs.query_history_k_data_plus(
            f"sh.{symbol}" if symbol.startswith("6") else f"sz.{symbol}",
            "date,open,high,low,close,volume",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2",
        )
        
        if rs.error_code != "0" or rs.next() == 0:
            bs.logout()
            return pd.DataFrame()
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        bs.logout()
        
        df = pd.DataFrame(data_list, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        return df.astype(float)
    except Exception:
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


@_retry_on_failure(max_retries=2, delay=3)
def _get_hk(symbol: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    try:
        s = start.replace("-", "")
        e = end.replace("-", "")
        df = ak.stock_hk_hist(
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


@_retry_on_failure(max_retries=3, delay=60)
def _get_yf(symbol: str, start: str, end: str) -> pd.DataFrame:
    """美股/港股/加密货币数据获取"""
    import requests
    import time
    
    time.sleep(2)
    
    try:
        start_ts = int(pd.to_datetime(start).timestamp())
        end_ts = int(pd.to_datetime(end).timestamp()) + 86400
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
        
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        if df.empty:
            return pd.DataFrame()
        
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        keep = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]]
        return df.astype(float)
    except Exception:
        return pd.DataFrame()