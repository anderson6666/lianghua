"""
新闻数据模块 - 获取财经新闻与市场舆情
支持：A股/美股/港股财经新闻、梁文峰相关资讯、市场情绪分析
数据源：akshare（免费）、新浪财经（公开接口）
"""
import json
import re
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st


@st.cache_data(ttl=3600)
def get_stock_news(symbol: str, market: str, count: int = 20) -> pd.DataFrame:
    """获取股票相关新闻列表"""
    try:
        if market in ("A股", "港股"):
            return _get_akshare_news(symbol, count)
        elif market == "美股":
            return _get_us_news(symbol, count)
        else:
            return _get_crypto_news(symbol, count)
    except Exception:
        return pd.DataFrame(columns=["title", "time", "summary", "source"])


def _get_akshare_news(symbol: str, count: int) -> pd.DataFrame:
    import akshare as ak
    try:
        df = ak.stock_news_em(symbol=symbol, count=count)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"标题": "title", "时间": "time", "摘要": "summary", "来源": "source"})
        df = df[["title", "time", "summary", "source"]].head(count)
        return df
    except Exception:
        return pd.DataFrame()


def _get_us_news(symbol: str, count: int) -> pd.DataFrame:
    base_url = "https://api.marketaux.com/v1/news/all"
    params = {
        "symbols": symbol,
        "filter_entities": "true",
        "limit": count,
        "language": "en",
    }
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            articles = []
            for item in data.get("data", []):
                articles.append({
                    "title": item.get("title", ""),
                    "time": item.get("published_at", ""),
                    "summary": item.get("description", ""),
                    "source": item.get("source", {}).get("name", ""),
                })
            return pd.DataFrame(articles)
    except Exception:
        pass
    return pd.DataFrame()


def _get_crypto_news(symbol: str, count: int) -> pd.DataFrame:
    coin = symbol.replace("-USD", "")
    try:
        url = f"https://newsapi.org/v2/everything?q={coin}&language=en&sortBy=publishedAt"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            articles = []
            for item in data.get("articles", []):
                articles.append({
                    "title": item.get("title", ""),
                    "time": item.get("publishedAt", ""),
                    "summary": item.get("description", ""),
                    "source": item.get("source", {}).get("name", ""),
                })
            return pd.DataFrame(articles).head(count)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=7200)
def get_liang_wenfeng_news(count: int = 10) -> pd.DataFrame:
    """获取梁文峰相关新闻（量化巨头动向）"""
    keywords = ["梁文峰", "梁文锋", "幻方量化", "DeepSeek"]
    all_news = []
    try:
        import akshare as ak
        for kw in keywords[:2]:
            try:
                df = ak.news_sina(keyword=kw)
                if df is not None and not df.empty:
                    df["keyword"] = kw
                    all_news.append(df)
            except Exception:
                continue
        if all_news:
            combined = pd.concat(all_news, ignore_index=True)
            combined = combined.drop_duplicates(subset=["title"])
            combined = combined.sort_values("time", ascending=False).head(count)
            return combined[["title", "time", "summary", "source"]]
    except Exception:
        pass
    return pd.DataFrame()


def analyze_sentiment(text: str) -> float:
    """简单的文本情绪分析 -1(负面) ~ 1(正面)"""
    positive_words = [
        "涨", "升", "利好", "突破", "创新高", "反弹", "增持", "买入",
        "盈利", "增长", "强劲", "超预期", "利好", "看好", "机会", "上涨",
        "positive", "bullish", "up", "gain", "profit", "strong",
    ]
    negative_words = [
        "跌", "降", "利空", "下跌", "暴跌", "破位", "减持", "卖出",
        "亏损", "下滑", "疲软", "不及预期", "利空", "看空", "风险", "暴跌",
        "negative", "bearish", "down", "loss", "weak", "drop",
    ]
    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w.lower() in text_lower)
    neg_count = sum(1 for w in negative_words if w.lower() in text_lower)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def get_market_sentiment(symbol: str, market: str) -> dict:
    """获取综合市场情绪"""
    news_df = get_stock_news(symbol, market, count=15)
    if news_df.empty:
        return {"score": 0.0, "count": 0, "pos_count": 0, "neg_count": 0}

    scores = []
    for _, row in news_df.iterrows():
        text = f"{row['title']} {row['summary']}"
        scores.append(analyze_sentiment(text))

    return {
        "score": round(sum(scores) / len(scores), 2),
        "count": len(scores),
        "pos_count": sum(1 for s in scores if s > 0.1),
        "neg_count": sum(1 for s in scores if s < -0.1),
        "news": news_df.head(5).to_dict(orient="records"),
    }
