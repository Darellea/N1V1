"""
core/execution/order_processor.py

Handles order processing, position tracking, and related calculations.
"""

import logging
from decimal import Decimal
from typing import Any, Dict, Optional

from core.types.order_types import Order, OrderStatus, OrderType
from risk.risk_manager import _safe_quantize
from utils.logger import get_trade_logger
from utils.time import now_ms, to_ms

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class OrderProcessor:
    """Handles order processing, position tracking, and related calculations."""

    def __init__(self) -> None:
        """Initialize the OrderProcessor."""
        self.open_orders: Dict[str, Order] = {}
        self.closed_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_count: int = 0

    def parse_order_response(self, response: Dict) -> Order:
        """
        Parse exchange order response into our Order dataclass.

        This parser is tolerant to varying exchange response shapes. It attempts
        to extract common keys with sensible defaults so downstream code doesn't
        crash when fields are missing or named differently.
        """
        if not isinstance(response, dict):
            raise ValueError("Invalid order response")

        # Helper to safely fetch nested/alias keys
        def _first(*keys, default=None):
            for k in keys:
                v = response.get(k)
                if v is not None:
                    return v
            return default

        id_val = _first("id", "orderId", "clientOrderId", default="")
        symbol = _first("symbol", "market", "market_id", default="")
        type_raw = _first("type", "order_type", default="market")
        status_raw = _first("status", "state", "status_code", default="open")
        amount_raw = _first("amount", "size", "filled", default=0)
        filled_raw = _first("filled", "filled", "executed", default=0)
        remaining_raw = _first("remaining", "remaining_amount", default=0)
        price_raw = _first("price", "cost", default=None)
        cost_raw = _first("cost", "executed_value", default=0)
        fee_raw = _first("fee", "fees", default=None)
        timestamp_raw = _first("timestamp", "datetime", default=0)

        # Normalize timestamp to epoch milliseconds consistently across the codebase.
        # Accepts seconds, milliseconds, ISO strings, datetime objects, etc.
        ts_ms = to_ms(timestamp_raw)
        if ts_ms is None:
            ts_ms = now_ms()

        params = response.get("params") or response.get("info") or {}

        # Normalize enums with safe fallbacks
        try:
            otype = OrderType(type_raw) if type_raw is not None else OrderType.MARKET
        except Exception:
            # Try to coerce strings like "limit" / "market"
            try:
                otype = OrderType(str(type_raw).lower())
            except Exception:
                otype = OrderType.MARKET

        try:
            status = (
                OrderStatus(status_raw) if status_raw is not None else OrderStatus.OPEN
            )
        except Exception:
            try:
                status = OrderStatus(str(status_raw).lower())
            except Exception:
                status = OrderStatus.OPEN

        # Parse numeric fields defensively
        try:
            amount = Decimal(str(amount_raw))
        except Exception:
            try:
                amount = Decimal(str(filled_raw))
            except Exception:
                amount = Decimal(0)

        try:
            filled = Decimal(str(filled_raw))
        except Exception:
            filled = Decimal(0)

        try:
            remaining = Decimal(str(remaining_raw))
        except Exception:
            remaining = Decimal(0)

        try:
            price = Decimal(str(price_raw)) if price_raw is not None else None
        except Exception:
            price = None

        try:
            cost = Decimal(str(cost_raw)) if cost_raw is not None else Decimal(0)
        except Exception:
            cost = Decimal(0)

        # Build Order dataclass safely
        return Order(
            id=str(id_val),
            symbol=symbol,
            type=otype,
            side=response.get("side") or response.get("direction") or "",
            amount=amount,
            price=price,
            status=status,
            filled=filled,
            remaining=remaining,
            cost=cost,
            fee=fee_raw,
            timestamp=ts_ms,
            params=params,
        )

    async def process_order(self, order: Order) -> Dict[str, Any]:
        """
        Process an order after execution (live or simulated).

        Args:
            order: The executed Order dataclass instance.

        Returns:
            Dictionary with processed order details including PnL.
        """
        self.trade_count += 1

        # Store the order
        if order.status == OrderStatus.FILLED:
            self.closed_orders[order.id] = order
            if order.id in self.open_orders:
                del self.open_orders[order.id]

            # Calculate PnL before updating positions (for sell orders)
            pnl = self._calculate_pnl(order) if order.side == "sell" else None

            # Update position tracking
            self._update_positions(order)
        else:
            self.open_orders[order.id] = order
            # No PnL for non-filled orders
            pnl = None

        # Optionally compute dynamic take-profit when requested via order.params
        take_profit_val: Optional[float] = None
        try:
            params = order.params or {}
            dynamic_tp = params.get("dynamic_tp") or params.get("dynamic_take_profit")
            # prefer explicit stop_loss from params; fallback to trailing_stop on position
            stop_loss_param = params.get("stop_loss", None)
            if stop_loss_param is None:
                # get position's trailing stop if any (useful for backtest/paper)
                pos = self.positions.get(order.symbol, {})
                stop_loss_param = pos.get("trailing_stop", None)

            if dynamic_tp and order.price and stop_loss_param is not None:
                entry = Decimal(str(order.price))
                stop = Decimal(str(stop_loss_param))
                # Determine direction and risk
                if order.side == "buy":
                    risk = entry - stop
                    # trend_strength can be passed via params (0..1); otherwise try to estimate
                    try:
                        if params.get("trend_strength") is not None:
                            trend_strength = float(params.get("trend_strength") or 0.0)
                        else:
                            # attempt to estimate trend strength from market data (live mode)
                            trend_strength = await self._estimate_trend_strength(
                                order.symbol
                            )
                    except Exception:
                        trend_strength = 0.0
                    trend_strength = max(0.0, min(1.0, float(trend_strength)))
                    multiplier = float(params.get("risk_reward_ratio", 2.0)) * (
                        1.0 + trend_strength
                    )
                    tp_dec = entry + Decimal(str(multiplier)) * Decimal(str(risk))
                    take_profit_val = float(_safe_quantize(tp_dec))
                else:
                    # short
                    risk = stop - entry
                    try:
                        if params.get("trend_strength") is not None:
                            trend_strength = float(params.get("trend_strength") or 0.0)
                        else:
                            # attempt to estimate trend strength (magnitude) for short
                            trend_strength = await self._estimate_trend_strength(
                                order.symbol
                            )
                    except Exception:
                        trend_strength = 0.0
                    trend_strength = max(0.0, min(1.0, float(trend_strength)))
                    multiplier = float(params.get("risk_reward_ratio", 2.0)) * (
                        1.0 + trend_strength
                    )
                    tp_dec = entry - Decimal(str(multiplier)) * Decimal(str(risk))
                    take_profit_val = float(_safe_quantize(tp_dec))
        except Exception:
            # If dynamic TP fails, leave take_profit_val as None
            take_profit_val = None

        result = {
            "id": order.id,
            "symbol": order.symbol,
            "type": order.type.value,
            "side": order.side,
            "amount": float(order.amount),
            "filled": float(order.filled),
            "price": float(order.price) if order.price else None,
            "status": order.status.value,
            "cost": float(order.cost),
            "fee": order.fee,
            "timestamp": order.timestamp,
            "pnl": pnl,
        }
        if take_profit_val is not None:
            result["take_profit"] = take_profit_val

        return result

    def _update_positions(self, order: Order) -> None:
        """Update position tracking based on filled orders."""
        position = self.positions.get(
            order.symbol,
            {
                "amount": Decimal(0),
                "entry_price": Decimal(0),
                "entry_cost": Decimal(0),
                "trailing_stop": None,
            },
        )

        if order.side == "buy":
            new_amount = position["amount"] + order.filled
            new_cost = position["entry_cost"] + order.cost
            # Determine trailing stop value from order (if provided) or keep existing
            trailing = position.get("trailing_stop")
            if getattr(order, "trailing_stop", None) is not None:
                trailing = order.trailing_stop
            elif order.params and isinstance(order.params, dict):
                # support params-based trailing_stop price key
                ts_param = order.params.get("trailing_stop")
                if isinstance(ts_param, dict) and ts_param.get("price"):
                    trailing = Decimal(str(ts_param.get("price")))
            position.update(
                {
                    "amount": new_amount,
                    "entry_price": new_cost / new_amount
                    if new_amount > 0
                    else Decimal(0),
                    "entry_cost": new_cost,
                    "trailing_stop": trailing,
                }
            )
            self.positions[order.symbol] = position
        else:
            # For sell orders, reduce position proportionally
            position["amount"] -= order.filled
            # Reduce entry cost proportionally to the amount sold
            position["entry_cost"] -= position["entry_price"] * order.filled
            if position["amount"] <= 0:
                del self.positions[order.symbol]
            else:
                self.positions[order.symbol] = position

    def _calculate_pnl(self, order: Order) -> Optional[float]:
        """Calculate PnL for a sell order."""
        if order.side != "sell" or order.symbol not in self.positions:
            return None

        position = self.positions[order.symbol]
        entry_value = position["entry_price"] * order.filled
        exit_value = order.price * order.filled
        gross_pnl = exit_value - entry_value
        fee = Decimal(order.fee["cost"]) if order.fee else Decimal(0)
        net_pnl = gross_pnl - fee

        return float(net_pnl)

    async def _estimate_trend_strength(
        self, symbol: str, timeframe: str = "1h", lookback: int = 20
    ) -> float:
        """
        Estimate trend strength for a symbol using recent OHLCV data.

        Returns a float in [0.0, 1.0] where higher means stronger trend in the
        direction of the recent price movement. For live mode this will attempt
        to fetch OHLCV from the exchange; otherwise returns 0.0.

        This is a lightweight heuristic: (last_price - first_price) / first_price,
        clamped to [-1,1] and returned as absolute value in [0,1].
        """
        # This method requires exchange access, so it's kept as a stub
        # The actual implementation would need access to an exchange instance
        return 0.0

    def get_active_order_count(self) -> int:
        """Get count of active/open orders."""
        return len(self.open_orders)

    def get_open_position_count(self) -> int:
        """Get count of open positions."""
        return len(self.positions)
