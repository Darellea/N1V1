"""
core/execution/backtest_executor.py

Handles backtest order execution and simulation.
"""

import logging
from decimal import Decimal
from typing import Dict, Any, Union
from datetime import datetime
from utils.time import now_ms

from core.types.order_types import Order, OrderStatus
from utils.logger import get_trade_logger
from utils.adapter import signal_to_dict

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class BacktestOrderExecutor:
    """Handles backtest order execution and simulation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the BacktestOrderExecutor.

        Args:
            config: Configuration dictionary with backtest settings
        """
        self.config = config
        self.trade_count: int = 0

    async def execute_backtest_order(self, signal: Any) -> Order:
        """
        Simulate order execution for backtesting.

        Args:
            signal: Object providing order details (must have timestamp, symbol, etc.).

        Returns:
            Order object with execution details. The Order.timestamp is always stored
            as an integer representing milliseconds since epoch (e.g., 1234567890000).
        """
        # Determine side
        side = getattr(signal, 'side', None)
        if side is None:
            from core.contracts import SignalType
            if signal.signal_type == SignalType.ENTRY_LONG:
                side = "buy"
            elif signal.signal_type == SignalType.ENTRY_SHORT:
                side = "sell"
            else:
                side = "buy"

        # Backtest orders are similar to paper trading but with historical data
        executed_price = signal.price  # For backtest, we use the exact price
        fee = self._calculate_fee(signal)

        # Normalize timestamp to milliseconds (int)
        timestamp_ms = self._normalize_timestamp_to_ms(signal.timestamp)

        order = Order(
            id=f"backtest_{self.trade_count}",
            symbol=signal.symbol,
            type=signal.order_type,
            side=side,
            amount=Decimal(signal.amount),
            price=Decimal(executed_price),
            status=OrderStatus.FILLED,
            filled=Decimal(signal.amount),
            remaining=Decimal(0),
            cost=Decimal(signal.amount) * Decimal(executed_price),
            params=getattr(signal, "params", {}) or ({"stop_loss": getattr(signal, "stop_loss", None)}),
            fee={"cost": float(fee), "currency": self.config["base_currency"]},
            trailing_stop=(
                Decimal(str(signal.trailing_stop.get("price")))
                if getattr(signal, "trailing_stop", None)
                and isinstance(signal.trailing_stop, dict)
                and signal.trailing_stop.get("price")
                else None
            ),
            timestamp=timestamp_ms,
        )

        self.trade_count += 1
        return order

    def _normalize_timestamp_to_ms(self, timestamp: Union[int, datetime]) -> int:
        """
        Normalize a timestamp to milliseconds (int).

        Args:
            timestamp: Timestamp as int (milliseconds) or datetime object

        Returns:
            Timestamp in milliseconds as int
        """
        if isinstance(timestamp, int):
            # Already in milliseconds
            return timestamp
        elif isinstance(timestamp, datetime):
            # Convert datetime to milliseconds
            return int(timestamp.timestamp() * 1000)
        else:
            # Fallback: try to convert using utils.time.to_ms
            try:
                from utils.time import to_ms
                result = to_ms(timestamp)
                return result if result is not None else 0
            except Exception:
                logger.warning(f"Failed to normalize timestamp {timestamp}, using 0")
                return 0

    def _calculate_fee(self, signal: Any) -> Decimal:
        """Calculate trading fee based on config.

        Args:
            signal: Object providing an 'amount' attribute or key.

        Returns:
            Decimal fee amount.
        """
        fee_rate = Decimal(self.config["trade_fee"])
        amt = getattr(
            signal, "amount", signal.get("amount") if isinstance(signal, dict) else 0
        )
        return Decimal(amt) * fee_rate
