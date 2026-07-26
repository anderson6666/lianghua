"""
股票量化预测软件 - Streamlit 主界面 (升级增强版)
全免费：数据源 akshare / yfinance，AI 引擎 Agnes（免费版）。

核心特性：
1. 多市场支持（A股/美股/港股/加密）
2. 实时新闻与市场情绪分析
3. 量化巨头动向追踪（梁文峰/幻方量化）
4. 机器学习趋势预测（结合情绪特征）
5. 策略回测与自我优化
6. Agnes AI 生成自然语言分析报告
7. 系统自我认知与健康度评估

免责声明：本软件仅用于技术学习与研究，所有分析、预测、回测结果均为
历史数据统计与概率参考，不构成任何投资建议。据此投资风险自负。
"""
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data import get_stock_data, get_fund_flow
from indicators import add_indicators, latest_signals
from backtest import run_backtest, STRATEGIES
from predict import train_predict
from agnes_agent import generate_daily_report, get_agnes_api_key
from optimize import (
    load_history, record_backtest, generate_self_report,
    evaluate_strategy, find_best_strategy, optimize_parameters,
    adaptive_strategy_switch,
)

st.set_page_config(page_title="AI量化预测系统", layout="wide", page_icon="🤖")

# 初始化 session_state
if "df" not in st.session_state:
    st.session_state.df = None
if "loaded" not in st.session_state:
    st.session_state.loaded = False

# ---------------- Agnes API Key 配置 ----------------
st.sidebar.header("🔑 Agnes AI 配置")
agnes_key = st.sidebar.text_input(
    "Agnes API Key",
    value=get_agnes_api_key(),
    help="注册 https://platform.agnes-ai.com 获取免费 API Key",
)
st.sidebar.caption(
    "Agnes AI 提供免费文本模型（1M上下文，20 RPM），用于生成自然语言分析报告。"
)
if agnes_key:
    st.sidebar.success("✅ API Key 已配置")
else:
    st.sidebar.info("未配置 API Key 将无法生成 AI 报告")

# ---------------- 侧边栏：参数 ----------------
st.sidebar.divider()
st.sidebar.title("📈 参数设置")

MARKET_EXAMPLES = {
    "A股": "600519",
}

market = st.sidebar.selectbox("市场", list(MARKET_EXAMPLES.keys()))

# 网络提示
st.sidebar.info("🌐 A股数据使用国内网络（baostock），无需代理")

symbol = st.sidebar.text_input(
    "代码", value=MARKET_EXAMPLES[market],
    help=f"示例：{MARKET_EXAMPLES[market]}",
)

col_d1, col_d2 = st.sidebar.columns(2)
default_start = date.today() - timedelta(days=365 * 2)
start = col_d1.date_input("开始日期", value=default_start)
end = col_d2.date_input("结束日期", value=date.today())

strategy = st.sidebar.selectbox("回测策略", STRATEGIES)
fee = st.sidebar.number_input("单边手续费率", value=0.0003, format="%.4f", step=0.0001)
init_capital = st.sidebar.number_input("初始资金", value=100000, step=10000)

run = st.sidebar.button("🚀 开始分析", use_container_width=True, type="primary")

st.sidebar.caption(
    "数据源：baostock（A股，国内网络）、akshare（A股备用）。"
)

# ---------------- 主区域 ----------------
st.title("🤖 AI量化预测系统")
st.caption(
    "机器学习预测 · 策略回测 · AI报告 | 全免费开源"
)
st.warning(
    "⚠️ 免责声明：本软件所有结果均为历史数据统计与概率参考，"
    "**不构成投资建议**。金融市场无法被准确预测，据此投资风险自负。"
)
st.info(
    "🌐 **网络提示**：A股数据使用国内网络（baostock），无需代理；"
    "美股/港股/加密货币需使用国外网络，且 Yahoo Finance 有请求频率限制。"
)


def _load(market, symbol, start, end):
    return get_stock_data(market, symbol, str(start), str(end))


def plot_kline(df: pd.DataFrame):
    d = add_indicators(df)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
        subplot_titles=("K线 + 均线 + 布林带", "成交量", "MACD"),
    )
    fig.add_trace(
        go.Candlestick(
            x=d.index, open=d["open"], high=d["high"], low=d["low"],
            close=d["close"], name="K线",
            increasing_line_color="red", decreasing_line_color="green",
        ),
        row=1, col=1,
    )
    for ma, color in [("MA5", "orange"), ("MA20", "blue"), ("MA60", "purple")]:
        fig.add_trace(go.Scatter(x=d.index, y=d[ma], name=ma,
                                 line=dict(width=1, color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["BOLL_UP"], name="BOLL上轨",
                             line=dict(width=1, dash="dot", color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["BOLL_LOW"], name="BOLL下轨",
                             line=dict(width=1, dash="dot", color="gray")), row=1, col=1)

    vol_colors = ["red" if c >= o else "green"
                  for c, o in zip(d["close"], d["open"])]
    fig.add_trace(go.Bar(x=d.index, y=d["volume"], name="成交量",
                         marker_color=vol_colors), row=2, col=1)

    fig.add_trace(go.Bar(x=d.index, y=d["MACD"], name="MACD柱"), row=3, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["DIF"], name="DIF",
                             line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["DEA"], name="DEA",
                             line=dict(width=1)), row=3, col=1)

    fig.update_layout(height=700, xaxis_rangeslider_visible=False,
                      hovermode="x unified", margin=dict(t=40, b=20))
    return fig


if run:
    try:
        with st.spinner("正在获取数据..."):
            st.session_state.df = get_stock_data(market, symbol, str(start), str(end))
            st.session_state.loaded = True
        st.success(f"已获取 {len(st.session_state.df)} 条数据（{st.session_state.df.index[0].date()} ~ {st.session_state.df.index[-1].date()}）")
    except Exception as e:
        st.error(f"数据获取失败：{e}")
        st.session_state.loaded = False
        st.stop()

if st.session_state.loaded and st.session_state.df is not None:
    df = st.session_state.df

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 行情图表", "🔔 指标信号", "🤖 趋势预测", "📉 策略回测", "📝 AI分析师报告"]
    )

    # --- 行情图表 ---
    with tab1:
        st.plotly_chart(plot_kline(df), use_container_width=True)

    # --- 指标信号 ---
    with tab2:
        st.subheader("当前技术指标信号")
        signals = latest_signals(df)
        cols = st.columns(3)
        icon = {"bull": "🟢 偏多", "bear": "🔴 偏空", "neutral": "⚪ 中性"}
        for i, (name, (text, tag)) in enumerate(signals.items()):
            with cols[i % 3]:
                st.metric(name, icon[tag], help=text)
                st.caption(text)
        bull = sum(1 for _, t in signals.values() if t == "bull")
        bear = sum(1 for _, t in signals.values() if t == "bear")
        st.divider()
        st.info(f"综合：偏多信号 {bull} 个，偏空信号 {bear} 个。")

        fund_flow = get_fund_flow(market)
        if fund_flow:
            st.subheader("资金流向")
            ff_cols = st.columns(4)
            names = ["北向资金", "南向资金", "主力净流入", "散户净流入"]
            for col, name in zip(ff_cols, names):
                if name in fund_flow:
                    val = fund_flow[name]
                    col.metric(name, f"{val / 1e8:.2f}亿")

    # --- 趋势预测 ---
    with tab3:
        st.subheader("机器学习次日趋势预测（多模型集成 + 三分类）")
        
        col_conf, col_opt = st.columns(2)
        confidence_threshold = col_conf.slider("置信度阈值", 0.5, 0.8, 0.6, 0.05,
            help="只在模型置信度超过此阈值时输出方向预测，否则标记为'观望'")
        optimize = col_opt.checkbox("启用超参数优化（Optuna）", False,
            help="使用贝叶斯优化搜索最优模型参数，训练时间会更长")
        
        try:
            with st.spinner("正在训练模型..." + ("（含超参数优化，可能需要几分钟）" if optimize else "")):
                res = train_predict(df,
                                   confidence_threshold=confidence_threshold,
                                   optimize=optimize)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("预测次日方向", res["direction"])
            c2.metric("测试集准确率", f"{res['test_acc']*100:.1f}%")
            c3.metric("高置信度准确率", f"{res['high_conf_acc']*100:.1f}%" if not np.isnan(res['high_conf_acc']) else "N/A")
            
            c4, c5, c6 = st.columns(3)
            c4.metric("盈亏比", f"{res['profit_loss_ratio']:.2f}")
            c5.metric("有效预测占比", f"{res['effective_ratio']*100:.1f}%")
            c6.metric("置信度阈值", f"{res['confidence_threshold']*100:.0f}%")
            
            st.caption(f"训练样本数：{res['n_samples']} | 特征数：{res['n_features']} | 模型：{res['model_used']}")

            st.write("**三分类概率分布**")
            prob_df = pd.DataFrame({
                "概率": [res["proba_up"], res["proba_down"], res["proba_flat"]]
            }, index=["上涨", "下跌", "震荡"])
            st.bar_chart(prob_df)

            st.write("**特征重要性**")
            st.bar_chart(res["importance"].head(20))

            st.info("三分类策略：涨>1% / 震荡 / 跌>1%。预测基于技术指标+大盘特征，使用多模型集成学习，概率经过校准处理。")
            st.warning("⚠️ 盈亏比（平均盈利/平均亏损）是比准确率更重要的指标。高准确率但低盈亏比的策略可能不赚钱。")
            st.warning("⚠️ 超参数优化可能导致过拟合，历史表现不代表未来。建议先用默认参数测试。")
        except Exception as e:
            st.error(f"预测失败：{e}")

    # --- 策略回测 ---
    with tab4:
        st.subheader(f"策略回测：{strategy}")
        try:
            bt = run_backtest(df, strategy, fee=fee, init_capital=float(init_capital))
            m = bt["metrics"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("累计收益率", f"{m['累计收益率']*100:.2f}%")
            c2.metric("年化收益率", f"{m['年化收益率']*100:.2f}%")
            c3.metric("夏普比率", f"{m['夏普比率']:.2f}")
            c4.metric("最大回撤", f"{m['最大回撤']*100:.2f}%")
            c5, c6, c7 = st.columns(3)
            c5.metric("年化波动率", f"{m['年化波动率']*100:.2f}%")
            c6.metric("日胜率", f"{m['日胜率']*100:.1f}%")
            c7.metric("交易次数", f"{m['交易次数']}")

            eq = pd.DataFrame({
                "策略净值": bt["equity"],
                "买入持有(基准)": bt["benchmark"],
            })
            st.line_chart(eq)

            record_backtest(
                symbol, market, strategy, m,
                f"{start} ~ {end}",
            )
            st.success("✅ 回测结果已记录，用于系统自我优化")

            st.divider()
            st.subheader("🎯 自适应策略推荐")
            best_strat = adaptive_strategy_switch(df)
            st.info(f"基于当前市场状态，推荐策略：**{best_strat}**")

        except Exception as e:
            st.error(f"回测失败：{e}")

    # --- AI分析师报告 ---
    with tab5:
        st.subheader("🤖 Agnes AI 分析师报告")

        if not agnes_key:
            st.warning("请在左侧配置 Agnes API Key 以生成 AI 报告")
            st.info("免费注册地址：https://platform.agnes-ai.com")
        else:
            signals = latest_signals(df)
            ml_result = {}
            bt_result = {}

            try:
                ml_result = train_predict(df)
            except Exception:
                pass

            try:
                bt_result = run_backtest(df, strategy, fee=fee, init_capital=float(init_capital))
            except Exception:
                pass

            if st.button("📝 生成分析报告", type="primary"):
                with st.spinner("Agnes AI 正在分析并生成报告..."):
                    try:
                        report = generate_daily_report(
                            symbol, market, df, signals,
                            ml_result, bt_result, api_key=agnes_key,
                        )
                        st.markdown(report)

                        with st.expander("💾 保存报告"):
                            st.download_button(
                                "下载报告",
                                report,
                                file_name=f"ai_report_{symbol}_{date.today()}.md",
                                mime="text/markdown",
                            )
                    except Exception as e:
                        st.error(f"生成报告失败：{e}")

        st.divider()
        st.subheader("🔍 系统自我分析报告")
        history = load_history()
        self_report = generate_self_report(history)
        st.markdown(self_report)

else:
    st.info("👈 在左侧设置参数后，点击「开始分析」。")
    st.markdown(
        """
        ### 系统功能概览

        **📊 行情图表**：K线、均线、布林带、成交量、MACD 多子图联动

        **🔔 指标信号**：MA / MACD / RSI / 布林带 / KDJ 实时多空提示 + 资金流向

        **🤖 趋势预测**：随机森林机器学习预测次日涨跌概率

        **📉 策略回测**：4 种经典策略 + 自适应策略推荐 + 回测结果自动记录

        **📝 AI分析师报告**：Agnes AI 生成自然语言分析报告 + 系统自我分析

        ### 代码示例
        | 市场 | 示例代码 |
        |------|----------|
        | A股 | `600519`（贵州茅台）、`000001`（平安银行） |
        | 美股 | `AAPL`、`TSLA`、`MSFT` |
        | 港股 | `00700`（腾讯）、`09988`（阿里） |
        | 加密货币 | `BTC-USD`、`ETH-USD` |

        ### Agnes AI 配置
        1. 访问 https://platform.agnes-ai.com 注册账号
        2. 创建 API Key
        3. 在左侧输入框粘贴 Key
        4. 即可生成 AI 分析报告（免费版 20 RPM）
        """
    )
