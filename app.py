import datetime as dt
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from indicators import (
    compute_bollinger,
    compute_ema,
    compute_kdj,
    compute_macd,
    compute_rsi,
    compute_sma,
    get_indicator_formulas,
)
from strategies import build_signals
from backtest import BacktestConfig, backtest_long_only
from broker import PaperBroker, FutuBroker, HuataiXTPBroker


@st.cache_data(ttl=3600)
def load_daily_data(symbol: str, start_date: dt.date, end_date: dt.date, adjust: Optional[str]) -> pd.DataFrame:
    import akshare as ak
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_str, end_date=end_str, adjust=adjust)
    if df is None or df.empty:
        return pd.DataFrame()
    rename_map = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"}
    df = df.rename(columns=rename_map)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
    return df[keep].sort_values("date").reset_index(drop=True)


def build_price_panel(
    data: pd.DataFrame,
    overlay_cols: List[str],
    buy_marks: List[Tuple[pd.Timestamp, float]],
    sell_marks: List[Tuple[pd.Timestamp, float]],
) -> go.Figure:
    rows = 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[1.0])
    candle = go.Candlestick(
        x=data["date"], open=data["open"], high=data["high"], low=data["low"], close=data["close"],
        name="K线", increasing_line_color="#e74c3c", decreasing_line_color="#2ecc71"
    )
    fig.add_trace(candle, row=1, col=1)
    for col in overlay_cols:
        if col in data.columns:
            fig.add_trace(go.Scatter(x=data["date"], y=data[col], name=col.upper(), mode="lines"), row=1, col=1)
    if buy_marks:
        fig.add_trace(go.Scatter(
            x=[d for d, _ in buy_marks], y=[p for _, p in buy_marks],
            mode="markers", name="买入", marker=dict(symbol="triangle-up", size=10, color="#e67e22")
        ), row=1, col=1)
    if sell_marks:
        fig.add_trace(go.Scatter(
            x=[d for d, _ in sell_marks], y=[p for _, p in sell_marks],
            mode="markers", name="卖出", marker=dict(symbol="triangle-down", size=10, color="#2980b9")
        ), row=1, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def main() -> None:
    st.set_page_config(page_title="A股量化系统 · 策略与回测", layout="wide")
    st.title("A股量化系统 · 策略、指标、信号、回测与实盘")

    with st.sidebar:
        st.header("数据")
        mode = st.radio("模式", ["回测", "实盘"], index=0)
        code = st.text_input("股票代码（仅数字）", value="600519")
        adjust = st.selectbox("复权方式", ["不复权", "前复权", "后复权"], index=1)
        adjust_map = {"不复权": None, "前复权": "qfq", "后复权": "hfq"}
        today = dt.date.today()
        start = st.date_input("开始日期", value=today - dt.timedelta(days=365 * 5))
        end = st.date_input("结束日期", value=today)

        st.header("叠加指标（价格面板）")
        use_sma = st.checkbox("SMA 均线", value=True)
        sma_a = st.number_input("SMA1", value=5, min_value=1, step=1)
        sma_b = st.number_input("SMA2", value=20, min_value=1, step=1)
        use_ema = st.checkbox("EMA 均线", value=False)
        ema_a = st.number_input("EMA1", value=12, min_value=1, step=1)
        ema_b = st.number_input("EMA2", value=26, min_value=1, step=1)
        use_boll = st.checkbox("布林带 BOLL", value=True)
        boll_n = st.number_input("BOLL 窗口", value=20, min_value=1, step=1)
        boll_k = st.number_input("BOLL k", value=2.0, min_value=0.5, step=0.5, format="%.1f")

        if mode == "回测":
            st.header("策略与参数")
            strat = st.selectbox("策略", ["SMA金叉死叉", "MACD金叉死叉", "RSI超买超卖反转", "BOLL突破"], index=0)
            if strat == "SMA金叉死叉":
                s_sw = st.number_input("短均线", value=10, min_value=1, step=1)
                s_lw = st.number_input("长均线", value=20, min_value=1, step=1)
                params = {"short_window": s_sw, "long_window": s_lw}
            elif strat == "MACD金叉死叉":
                s_fast = st.number_input("快线", value=12, min_value=1, step=1)
                s_slow = st.number_input("慢线", value=26, min_value=1, step=1)
                s_sig = st.number_input("Signal", value=9, min_value=1, step=1)
                params = {"fast": s_fast, "slow": s_slow, "signal": s_sig}
            elif strat == "RSI超买超卖反转":
                s_p = st.number_input("周期", value=14, min_value=1, step=1)
                s_lt = st.number_input("低阈值", value=30.0, step=0.5)
                s_ht = st.number_input("高阈值", value=70.0, step=0.5)
                params = {"period": s_p, "low_th": s_lt, "high_th": s_ht}
            else:  # BOLL突破
                s_w = st.number_input("窗口", value=20, min_value=1, step=1)
                s_k = st.number_input("倍数k", value=2.0, min_value=0.5, step=0.5, format="%.1f")
                params = {"window": s_w, "k": s_k}

            st.header("回测设置（仅多头全仓）")
            init_cap = st.number_input("初始资金", value=100000.0, min_value=1000.0, step=1000.0)
            fee = st.number_input("手续费(单边，比例)", value=0.0005, min_value=0.0, step=0.0001, format="%.4f")
            slip = st.number_input("滑点(比例近似)", value=0.0005, min_value=0.0, step=0.0001, format="%.4f")
        else:
            st.header("实盘经纪商")
            broker_kind = st.selectbox("选择经纪商", ["模拟撮合（本地）", "Futu OpenAPI", "华泰 XTP（需要券商开通）"], index=0)
            if "broker_kind" not in st.session_state or st.session_state["broker_kind"] != broker_kind:
                if broker_kind == "模拟撮合（本地）":
                    st.session_state["broker"] = PaperBroker()
                    st.session_state["broker"].connect()
                elif broker_kind == "Futu OpenAPI":
                    st.session_state["broker"] = FutuBroker()
                    st.session_state["broker"].connect()
                else:
                    st.session_state["broker"] = None  # 待用户填写参数后连接
                st.session_state["broker_kind"] = broker_kind
            if broker_kind == "Futu OpenAPI":
                st.caption("需安装并登录 Futu OpenD，A股账户方可交易。此处为占位实现。")
            if broker_kind == "华泰 XTP（需要券商开通）":
                st.caption("需向华泰申请开通 XTP 柜台权限并安装 SDK。以下为连接参数：")
                xtp_user = st.text_input("资金账号", value="")
                xtp_pwd = st.text_input("交易密码", value="", type="password")
                xtp_client_id = st.number_input("ClientID", value=1, min_value=1, step=1)
                colx1, colx2 = st.columns(2)
                with colx1:
                    xtp_td_ip = st.text_input("交易服务器IP", value="127.0.0.1")
                    xtp_td_port = st.number_input("交易端口", value=6001, step=1)
                with colx2:
                    xtp_qd_ip = st.text_input("行情服务器IP", value="127.0.0.1")
                    xtp_qd_port = st.number_input("行情端口", value=6002, step=1)
                xtp_protocol = st.selectbox("协议", ["TCP", "UDP"], index=0)
                xtp_env = st.selectbox("环境", ["仿真", "实盘"], index=0)
                if st.button("连接华泰 XTP"):
                    st.session_state["broker"] = HuataiXTPBroker(
                        user_id=xtp_user,
                        password=xtp_pwd,
                        client_id=int(xtp_client_id),
                        td_ip=xtp_td_ip,
                        td_port=int(xtp_td_port),
                        qd_ip=xtp_qd_ip,
                        qd_port=int(xtp_qd_port),
                        protocol=xtp_protocol,
                        env="SIM" if xtp_env == "仿真" else "REAL",
                    )
                    ok = st.session_state["broker"].connect()
                    st.toast("已连接（占位）" if ok else "连接失败，请检查参数与SDK安装", icon="✅" if ok else "⚠️")

    with st.spinner("加载数据中..."):
        df = load_daily_data(code, start, end, adjust_map[adjust])
    if df.empty:
        st.warning("未获取到数据，请检查代码或日期范围。")
        st.stop()

    # 叠加指标
    overlay_cols: List[str] = []
    if use_sma:
        df[f"ma_{sma_a}"] = compute_sma(df["close"], sma_a)
        df[f"ma_{sma_b}"] = compute_sma(df["close"], sma_b)
        overlay_cols += [f"ma_{sma_a}", f"ma_{sma_b}"]
    if use_ema:
        df[f"ema_{ema_a}"] = compute_ema(df["close"], ema_a)
        df[f"ema_{ema_b}"] = compute_ema(df["close"], ema_b)
        overlay_cols += [f"ema_{ema_a}", f"ema_{ema_b}"]
    if use_boll:
        boll = compute_bollinger(df["close"], boll_n, boll_k)
        df = df.join(boll)
        overlay_cols += ["boll_upper", "boll_mid", "boll_lower"]

    if mode == "回测":
        # 策略信号与回测
        sig_df = build_signals(df, strat, params)
        cfg = BacktestConfig(initial_capital=init_cap, commission_rate=fee, slippage_rate=slip)
        bt = backtest_long_only(df, sig_df["action"], cfg)

        # 上方价格+标注
        fig = build_price_panel(df, overlay_cols, bt["marks"]["buy"], bt["marks"]["sell"])
        st.plotly_chart(fig, use_container_width=True)

        # 绩效指标
        st.subheader("绩效指标")
        m = bt["metrics"]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("最终资产", f"{m['final_value']:.2f}")
        col2.metric("总收益", f"{m['total_return']*100:.2f}%")
        col3.metric("年化", f"{m['annual_return']*100:.2f}%")
        col4.metric("夏普", f"{m['sharpe']:.2f}")
        col5.metric("最大回撤", f"{m['max_drawdown']*100:.2f}%")

        # 权益曲线
        st.subheader("权益曲线")
        curve: pd.DataFrame = bt["curve"]
        curve_fig = go.Figure(data=[go.Scatter(x=curve["date"], y=curve["equity"], name="Equity")])
        curve_fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(curve_fig, use_container_width=True)

        # 交易明细与导出
        st.subheader("交易明细")
        trades: pd.DataFrame = bt["trades"]
        if trades.empty:
            st.info("无交易。")
        else:
            st.dataframe(trades, use_container_width=True)
            st.download_button("下载交易CSV", data=trades.to_csv(index=False).encode("utf-8-sig"), file_name=f"{code}_trades.csv")

        # 数据导出
        st.subheader("信号与数据导出")
        merged = df.copy()
        merged = merged.join(sig_df)
        st.download_button("下载信号CSV", data=merged.to_csv(index=False).encode("utf-8-sig"), file_name=f"{code}_signals.csv")
    else:
        # 实盘界面
        broker = st.session_state.get("broker", PaperBroker())
        quote = broker.get_quote(code)
        last_px = quote["last"] if quote else None
        st.subheader("实盘 · 报价")
        colq1, colq2, colq3 = st.columns(3)
        colq1.metric("最新价", f"{last_px:.2f}" if last_px is not None else "-")
        colq2.metric("买一", f"{quote['bid1']:.2f}" if quote and quote.get("bid1") else "-")
        colq3.metric("卖一", f"{quote['ask1']:.2f}" if quote and quote.get("ask1") else "-")
        st.button("刷新报价")

        # 上方价格图（无交易标注）
        fig = build_price_panel(df, overlay_cols, [], [])
        st.plotly_chart(fig, use_container_width=True)

        # 下单
        st.subheader("下单")
        with st.form("place_order_form", clear_on_submit=True):
            side = st.selectbox("方向", ["buy", "sell"])
            qty = st.number_input("数量(股)", value=100, min_value=1, step=1)
            price_default = float(last_px) if last_px is not None else float(df["close"].iloc[-1])
            price = st.number_input("限价", value=price_default, step=0.01, format="%.2f")
            submitted = st.form_submit_button("提交订单")
        if submitted:
            oid = broker.place_order(code, side, price, int(qty), order_type="limit")
            st.success(f"已提交订单: {oid}")

        # 订单与持仓
        st.subheader("订单列表")
        orders_df = broker.list_orders()
        if orders_df.empty:
            st.info("暂无订单")
        else:
            st.dataframe(orders_df, use_container_width=True)
            st.download_button("下载订单CSV", data=orders_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{code}_orders.csv")

        st.subheader("持仓")
        pos_df = broker.list_positions()
        if pos_df.empty:
            st.info("暂无持仓")
        else:
            st.dataframe(pos_df, use_container_width=True)

    # 指标公式
    st.subheader("指标公式")
    formulas = get_indicator_formulas()
    with st.expander("SMA（简单移动平均）", expanded=False):
        st.latex(formulas["SMA"])
    with st.expander("EMA（指数移动平均）", expanded=False):
        st.latex(formulas["EMA"])
    with st.expander("MACD", expanded=False):
        st.latex(formulas["MACD"])
    with st.expander("RSI", expanded=False):
        st.latex(formulas["RSI"])
    with st.expander("BOLL（布林带）", expanded=False):
        st.latex(formulas["BOLL"])


if __name__ == "__main__":
    main()


