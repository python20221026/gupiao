import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive")
    return series.rolling(window=window, min_periods=1).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    if span <= 0:
        raise ValueError("span must be positive")
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": hist,
        }
    )


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if period <= 0:
        raise ValueError("period must be positive")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    # 使用 Wilder 平滑
    avg_gain = avg_gain.shift(1).ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = avg_loss.shift(1).ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0.0)


def compute_kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    k_period: int = 3,
    d_period: int = 3,
) -> pd.DataFrame:
    lowest_low = low.rolling(window=n, min_periods=1).min()
    highest_high = high.rolling(window=n, min_periods=1).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / k_period, adjust=False).mean()
    d = k.ewm(alpha=1 / d_period, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"kdj_k": k, "kdj_d": d, "kdj_j": j})


def compute_bollinger(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    mid = compute_sma(close, window)
    std = close.rolling(window=window, min_periods=1).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"boll_mid": mid, "boll_upper": upper, "boll_lower": lower})


def get_indicator_formulas() -> dict:
    """
    返回用于展示的 LaTeX 公式字典
    """
    return {
        "SMA": r"SMA_t = \\frac{1}{n} \\sum_{i=0}^{n-1} Close_{t-i}",
        "EMA": r"EMA_t = \\alpha \\cdot Close_t + (1-\\alpha) \\cdot EMA_{t-1},\\ \\alpha=\\frac{2}{n+1}",
        "MACD": (
            r"MACD = EMA_{fast}(Close) - EMA_{slow}(Close) \\quad;\\quad "
            r"Signal = EMA_{signal}(MACD) \\quad;\\quad "
            r"Hist = MACD - Signal"
        ),
        "RSI": (
            r"RSI = 100 - \\frac{100}{1 + RS},\\ RS = \\frac{AvgGain}{AvgLoss}"
        ),
        "KDJ": (
            r"RSV_t = \\frac{Close_t - LL_n}{HH_n - LL_n} \\times 100 \\quad;\\quad "
            r"K = EMA(RSV, k) \\quad;\\quad D = EMA(K, d) \\quad;\\quad J = 3K - 2D"
        ),
        "BOLL": (
            r"Middle = SMA_n(Close) \\quad;\\quad "
            r"Upper = Middle + k\\cdot\\sigma_n \\quad;\\quad "
            r"Lower = Middle - k\\cdot\\sigma_n"
        ),
    }


