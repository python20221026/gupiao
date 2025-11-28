from typing import Dict, Optional

import pandas as pd

from indicators import (
    compute_bollinger,
    compute_macd,
    compute_rsi,
    compute_sma,
)


def sma_crossover_signals(
    data: pd.DataFrame, short_window: int = 10, long_window: int = 20
) -> pd.DataFrame:
    df = data.copy()
    df["ma_short"] = compute_sma(df["close"], short_window)
    df["ma_long"] = compute_sma(df["close"], long_window)
    prev_diff = (df["ma_short"] - df["ma_long"]).shift(1)
    curr_diff = df["ma_short"] - df["ma_long"]
    cross_up = (prev_diff <= 0) & (curr_diff > 0)
    cross_dn = (prev_diff >= 0) & (curr_diff < 0)
    actions = pd.Series(index=df.index, dtype="object")
    actions.loc[cross_up] = "buy"
    actions.loc[cross_dn] = "sell"
    return pd.DataFrame({"action": actions})


def macd_cross_signals(
    data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    df = data.copy()
    macd_df = compute_macd(df["close"], fast, slow, signal)
    df = df.join(macd_df)
    prev_diff = (df["macd"] - df["macd_signal"]).shift(1)
    curr_diff = df["macd"] - df["macd_signal"]
    cross_up = (prev_diff <= 0) & (curr_diff > 0)
    cross_dn = (prev_diff >= 0) & (curr_diff < 0)
    actions = pd.Series(index=df.index, dtype="object")
    actions.loc[cross_up] = "buy"
    actions.loc[cross_dn] = "sell"
    return pd.DataFrame({"action": actions})


def rsi_rebound_signals(
    data: pd.DataFrame, period: int = 14, low_th: float = 30.0, high_th: float = 70.0
) -> pd.DataFrame:
    df = data.copy()
    df["rsi"] = compute_rsi(df["close"], period)
    # 超卖反弹向上穿越 low_th 买入；超买回落向下穿越 high_th 卖出
    buy_cross = (df["rsi"].shift(1) < low_th) & (df["rsi"] >= low_th)
    sell_cross = (df["rsi"].shift(1) > high_th) & (df["rsi"] <= high_th)
    actions = pd.Series(index=df.index, dtype="object")
    actions.loc[buy_cross] = "buy"
    actions.loc[sell_cross] = "sell"
    return pd.DataFrame({"action": actions})


def boll_breakout_signals(
    data: pd.DataFrame, window: int = 20, k: float = 2.0
) -> pd.DataFrame:
    df = data.copy()
    boll = compute_bollinger(df["close"], window, k)
    # 避免与外层已存在的 BOLL 列重名冲突：不合并，直接使用临时上轨/下轨序列
    upper = boll["boll_upper"]
    lower = boll["boll_lower"]
    buy_cross = (df["close"].shift(1) <= upper.shift(1)) & (df["close"] > upper)
    sell_cross = (df["close"].shift(1) >= lower.shift(1)) & (df["close"] < lower)
    actions = pd.Series(index=df.index, dtype="object")
    actions.loc[buy_cross] = "buy"
    actions.loc[sell_cross] = "sell"
    return pd.DataFrame({"action": actions})


def build_signals(
    data: pd.DataFrame,
    kind: str,
    params: Optional[Dict] = None,
) -> pd.DataFrame:
    params = params or {}
    if kind == "SMA金叉死叉":
        sw = int(params.get("short_window", 10))
        lw = int(params.get("long_window", 20))
        return sma_crossover_signals(data, sw, lw)
    if kind == "MACD金叉死叉":
        f = int(params.get("fast", 12))
        s = int(params.get("slow", 26))
        sig = int(params.get("signal", 9))
        return macd_cross_signals(data, f, s, sig)
    if kind == "RSI超买超卖反转":
        p = int(params.get("period", 14))
        lt = float(params.get("low_th", 30.0))
        ht = float(params.get("high_th", 70.0))
        return rsi_rebound_signals(data, p, lt, ht)
    if kind == "BOLL突破":
        w = int(params.get("window", 20))
        k = float(params.get("k", 2.0))
        return boll_breakout_signals(data, w, k)
    return pd.DataFrame({"action": pd.Series(index=data.index, dtype="object")})


