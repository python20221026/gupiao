import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


TradeSide = Literal["buy", "sell"]


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission_rate: float = 0.0005  # 单边万5
    slippage_rate: float = 0.0005    # 用比例近似滑点
    trade_on: Literal["next_open"] = "next_open"
    allow_rebuy_same_day: bool = False


def _execution_price(row: pd.Series, trade_on: str) -> float:
    if trade_on == "next_open":
        return float(row["open"])
    return float(row["open"])


def align_exec_actions(actions: pd.Series, method: str = "shift1") -> pd.Series:
    """
    将信号生成日对齐到执行日（下一交易日），以避免未来函数。
    """
    if method == "shift1":
        return actions.shift(1)
    return actions.shift(1)


def backtest_long_only(
    ohlcv: pd.DataFrame,
    actions: pd.Series,
    config: Optional[BacktestConfig] = None,
) -> Dict[str, object]:
    """
    简单双均线/指标信号回测（只做多），全仓进出。
    输入：
      - ohlcv: 包含 date, open, high, low, close 的 DataFrame，date 为升序
      - actions: index 与 ohlcv 对齐，值为 'buy'/'sell'/nan（生成日），将被对齐到下一交易日执行
      - config: 回测配置
    返回：
      - dict 包含绩效、权益曲线、交易明细、标注点
    """
    if config is None:
        config = BacktestConfig()

    data = ohlcv.copy()
    data = data.reset_index(drop=True)

    exec_actions = align_exec_actions(actions)
    data["exec_action"] = exec_actions.values

    cash = config.initial_capital
    shares = 0.0
    equity_list: List[float] = []
    cash_list: List[float] = []
    pos_list: List[float] = []
    trade_records: List[Dict[str, object]] = []
    buy_marks: List[Tuple[pd.Timestamp, float]] = []
    sell_marks: List[Tuple[pd.Timestamp, float]] = []

    for i in range(len(data)):
        row = data.iloc[i]
        date = row["date"]
        close_price = float(row["close"])

        action = row.get("exec_action", np.nan)
        # 先执行交易（开盘价）
        if isinstance(action, str):
            px = _execution_price(row, config.trade_on)
            if action == "buy" and shares <= 1e-12:
                cost_px = px * (1.0 + config.commission_rate + config.slippage_rate)
                shares = math.floor(cash / cost_px)
                if shares > 0:
                    cost = shares * cost_px
                    cash -= cost
                    trade_records.append({
                        "date": date,
                        "side": "buy",
                        "price": px,
                        "shares": shares,
                        "amount": cost,
                    })
                    buy_marks.append((pd.to_datetime(date), px))
            elif action == "sell" and shares > 0:
                proceeds_px = px * (1.0 - config.commission_rate - config.slippage_rate)
                proceeds = shares * proceeds_px
                trade_records.append({
                    "date": date,
                    "side": "sell",
                    "price": px,
                    "shares": shares,
                    "amount": proceeds,
                })
                cash += proceeds
                shares = 0.0

        equity = cash + shares * close_price
        equity_list.append(equity)
        cash_list.append(cash)
        pos_list.append(shares)

    result = pd.DataFrame({
        "date": data["date"],
        "equity": equity_list,
        "cash": cash_list,
        "position": pos_list,
        "close": data["close"],
    })
    result["ret"] = result["equity"].pct_change().fillna(0.0)

    # 绩效指标
    total_return = result["equity"].iloc[-1] / config.initial_capital - 1.0
    daily_ret = result["ret"]
    ann_factor = 252
    avg = daily_ret.mean()
    vol = daily_ret.std(ddof=0)
    sharpe = (avg / vol * np.sqrt(ann_factor)) if vol > 0 else 0.0
    cummax = result["equity"].cummax()
    drawdown = result["equity"] / cummax - 1.0
    max_dd = drawdown.min()
    n_days = max(1, len(result))
    ann_return = (1 + total_return) ** (ann_factor / n_days) - 1.0

    trades_df = pd.DataFrame(trade_records)
    marks = {
        "buy": buy_marks,
        "sell": sell_marks,
    }

    metrics = {
        "initial_capital": config.initial_capital,
        "final_value": float(result["equity"].iloc[-1]),
        "total_return": float(total_return),
        "annual_return": float(ann_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "num_trades": int(len(trades_df)) if not trades_df.empty else 0,
    }

    return {
        "metrics": metrics,
        "curve": result,
        "trades": trades_df,
        "marks": marks,
    }


