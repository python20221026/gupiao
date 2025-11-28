from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


class BrokerBase:
    def connect(self) -> bool:
        raise NotImplementedError

    def is_connected(self) -> bool:
        raise NotImplementedError

    def get_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        raise NotImplementedError

    def place_order(self, symbol: str, side: str, price: float, quantity: int, order_type: str = "limit") -> str:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def list_orders(self) -> pd.DataFrame:
        raise NotImplementedError

    def list_positions(self) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class PaperBroker(BrokerBase):
    name: str = "PaperBroker"
    connected: bool = True
    _orders: List[Dict[str, object]] = field(default_factory=list)
    _positions: Dict[str, int] = field(default_factory=dict)
    _cash: float = 1_000_000.0

    def connect(self) -> bool:
        self.connected = True
        return True

    def is_connected(self) -> bool:
        return self.connected

    def get_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        try:
            import akshare as ak
            spot = ak.stock_zh_a_spot_em()
            if spot is None or spot.empty:
                return None
            row = spot[spot["代码"] == symbol]
            if row.empty:
                return None
            r = row.iloc[0]
            price = float(r.get("最新价", r.get("现价", 0.0)) or 0.0)
            bid1 = float(r.get("买一", price) or price)
            ask1 = float(r.get("卖一", price) or price)
            return {"last": price, "bid1": bid1, "ask1": ask1}
        except Exception:
            return None

    def place_order(self, symbol: str, side: str, price: float, quantity: int, order_type: str = "limit") -> str:
        ts = pd.Timestamp.now()
        order_id = f"paper_{int(time.time()*1000)}"
        # 简化撮合：立即按给定价格成交
        filled = True
        cost = price * quantity
        if side == "buy":
            if self._cash >= cost:
                self._cash -= cost
                self._positions[symbol] = self._positions.get(symbol, 0) + quantity
            else:
                filled = False
        elif side == "sell":
            if self._positions.get(symbol, 0) >= quantity:
                self._cash += cost
                self._positions[symbol] = self._positions.get(symbol, 0) - quantity
            else:
                filled = False
        self._orders.append({
            "order_id": order_id,
            "time": ts,
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": quantity,
            "status": "filled" if filled else "rejected",
            "type": order_type,
        })
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        for o in self._orders:
            if o["order_id"] == order_id and o["status"] == "new":
                o["status"] = "cancelled"
                return True
        return False

    def list_orders(self) -> pd.DataFrame:
        return pd.DataFrame(self._orders)

    def list_positions(self) -> pd.DataFrame:
        rows = []
        for sym, qty in self._positions.items():
            rows.append({"symbol": sym, "quantity": qty})
        return pd.DataFrame(rows)


class FutuBroker(BrokerBase):
    """
    Futu OpenAPI 占位实现：
      - 需安装客户端 OpenD 并登录交易账户
      - pip install futu-api
    这里保留接口与最小连接示例，完整下单逻辑可按需扩展。
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 11111):
        self.host = host
        self.port = port
        self._quote_ctx = None
        self._trade_ctx = None

    def connect(self) -> bool:
        try:
            from futu import OpenQuoteContext, OpenUSTradeContext, OpenHKTradeContext, OpenCNTradeContext
            # A股对应内地市场交易上下文（需有权限与登录）
            self._quote_ctx = OpenQuoteContext(host=self.host, port=self.port)
            # 具体券商账户环境初始化请替换为对应交易市场：
            self._trade_ctx = OpenCNTradeContext(host=self.host, port=self.port)
            return True
        except Exception:
            return False

    def is_connected(self) -> bool:
        return self._quote_ctx is not None

    def get_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        try:
            if self._quote_ctx is None:
                return None
            # Futu 代码格式与交易所前缀不同，需自行转换 symbol -> "SH.600519"/"SZ.000001"
            # 这里不做转换，仅返回 None，作为占位
            return None
        except Exception:
            return None

    def place_order(self, symbol: str, side: str, price: float, quantity: int, order_type: str = "limit") -> str:
        raise NotImplementedError("请根据 Futu OpenAPI 的交易接口完善下单实现。")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("请根据 Futu OpenAPI 的交易接口完善撤单实现。")

    def list_orders(self) -> pd.DataFrame:
        return pd.DataFrame()

    def list_positions(self) -> pd.DataFrame:
        return pd.DataFrame()


class HuataiXTPBroker(BrokerBase):
    """
    华泰证券 XTP 接口（占位实现）
    使用须知：
      - 需向华泰证券申请开通 XTP 柜台权限（仿真/实盘）
      - 安装华泰提供的 XTP SDK（含 Python 封装），通常为本地 DLL/SO + Python 包（非 PyPI 分发）
      - 获取交易/行情服务器地址与端口、资金账号、交易密码、ClientID、协议（TCP/UDP）等
    说明：
      - 此处先提供接口骨架，connect 成功与下单需完成对接华泰 SDK 的具体调用
      - 为不阻塞使用，报价默认回退使用 akshare 实时快照
    """

    def __init__(
        self,
        user_id: str,
        password: str,
        client_id: int,
        td_ip: str,
        td_port: int,
        qd_ip: str,
        qd_port: int,
        protocol: str = "TCP",
        env: str = "SIM",  # SIM/REAL
    ) -> None:
        self.user_id = user_id
        self.password = password
        self.client_id = client_id
        self.td_ip = td_ip
        self.td_port = td_port
        self.qd_ip = qd_ip
        self.qd_port = qd_port
        self.protocol = protocol
        self.env = env
        self._connected = False
        self._orders: List[Dict[str, object]] = []

    def connect(self) -> bool:
        try:
            # 占位：尝试导入 XTP SDK（名称视安装而定，常见为 xtp 或厂商定制包名）
            # 实际需要：初始化交易与行情连接、登录资金账号、设置回调等
            import importlib  # noqa
            # _xtp = importlib.import_module("xtp")  # 若本机已安装可启用
            self._connected = True  # 仅占位
        except Exception:
            self._connected = False
        return self._connected

    def is_connected(self) -> bool:
        return self._connected

    def get_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        # 占位：回退到 akshare 实时快照
        try:
            import akshare as ak
            spot = ak.stock_zh_a_spot_em()
            if spot is None or spot.empty:
                return None
            row = spot[spot["代码"] == symbol]
            if row.empty:
                return None
            r = row.iloc[0]
            price = float(r.get("最新价", r.get("现价", 0.0)) or 0.0)
            bid1 = float(r.get("买一", price) or price)
            ask1 = float(r.get("卖一", price) or price)
            return {"last": price, "bid1": bid1, "ask1": ask1}
        except Exception:
            return None

    def place_order(self, symbol: str, side: str, price: float, quantity: int, order_type: str = "limit") -> str:
        # TODO: 使用 XTP 交易接口发送委托（占位）
        raise NotImplementedError("请接入华泰 XTP SDK 的下单接口后启用。")

    def cancel_order(self, order_id: str) -> bool:
        # TODO: 使用 XTP 交易接口撤单（占位）
        raise NotImplementedError("请接入华泰 XTP SDK 的撤单接口后启用。")

    def list_orders(self) -> pd.DataFrame:
        return pd.DataFrame(self._orders)

    def list_positions(self) -> pd.DataFrame:
        # TODO: 通过 XTP 查询持仓（占位）
        return pd.DataFrame()

