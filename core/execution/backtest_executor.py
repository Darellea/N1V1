"""
core/execution/backtest_executor.py

Handles backtest order execution and simulation.
"""

import logging
from decimal import Decimal
from typing import Dict, Any
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

    async def execute_backtest_order(self, signal: Any) -> Dict[str, Any]:
        """
        Simulate order execution for backtesting.

        Args:
            signal: Object providing order details.

        Returns:
            Dictionary containing backtest order execution details.
        """
        # Backtest orders are similar to paper trading but with historical data
        executed_price = signal.price  # For backtest, we use the exact price
        fee = self._calculate_fee(signal)

        order = Order(
            id=f"backtest_{self.trade_count}",
            symbol=signal.symbol,
            type=signal.order_type,
            side=signal.side,
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
            timestamp=signal.timestamp,
        )

        self.trade_count += 1
        return order

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
