"""
Base Executor

Abstract base class for all order execution strategies.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from decimal import Decimal

from core.contracts import TradingSignal
from core.types.order_types import Order, OrderStatus

logger = logging.getLogger(__name__)


class BaseExecutor(ABC):
    """
    Abstract base class for order execution strategies.

    This class defines the interface that all execution strategies must implement,
    providing a consistent way to execute orders with different algorithms.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the executor.

        Args:
            config: Configuration dictionary for the executor
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def execute_order(self, signal: TradingSignal) -> List[Order]:
        """
        Execute an order using the specific execution strategy.

        Args:
            signal: Trading signal to execute

        Returns:
            List of executed orders
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            True if cancellation was successful
        """
        pass

    async def split_order(self, amount: Decimal, parts: int) -> List[Decimal]:
        """
        Split an order amount into multiple parts.

        Args:
            amount: Total order amount
            parts: Number of parts to split into

        Returns:
            List of split amounts
        """
        if parts <= 1:
            return [amount]

        # Calculate base amount per part
        base_amount = amount / parts

        # Distribute amounts to avoid rounding issues
        amounts = []
        remaining = amount

        for i in range(parts - 1):
            part_amount = round(base_amount, 8)  # Round to 8 decimal places
            amounts.append(Decimal(str(part_amount)))
            remaining -= part_amount

        # Last part gets the remainder
        amounts.append(remaining)

        return amounts

    def _validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate a trading signal.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid
        """
        if not hasattr(signal, 'symbol') or not signal.symbol:
            self.logger.error("Signal missing symbol")
            return False

        if not hasattr(signal, 'amount') or signal.amount <= 0:
            self.logger.error("Signal has invalid amount")
            return False

        return True

    def _create_child_order(self, parent_signal: TradingSignal, amount: Decimal,
                           order_id: str, part_number: int) -> TradingSignal:
        """
        Create a child order from a parent signal.

        Args:
            parent_signal: Original trading signal
            amount: Amount for this child order
            order_id: Unique order ID
            part_number: Part number for logging

        Returns:
            Child trading signal
        """
        # Create a copy of the parent signal
        child_signal = TradingSignal(
            strategy_id=f"{parent_signal.strategy_id}_part_{part_number}",
            symbol=parent_signal.symbol,
            signal_type=parent_signal.signal_type,
            signal_strength=parent_signal.signal_strength,
            order_type=parent_signal.order_type,
            amount=amount,
            price=parent_signal.price,
            current_price=parent_signal.current_price,
            stop_loss=parent_signal.stop_loss,
            take_profit=parent_signal.take_profit,
            timestamp=parent_signal.timestamp,
            trailing_stop=parent_signal.trailing_stop,
            metadata=parent_signal.metadata or {}
        )

        # Add parent order information to metadata
        if child_signal.metadata is None:
            child_signal.metadata = {}

        child_signal.metadata.update({
            'parent_order_id': getattr(parent_signal, 'order_id', None),
            'part_number': part_number,
            'total_parts': getattr(parent_signal, 'total_parts', 1),
            'execution_strategy': self.__class__.__name__
        })

        return child_signal

    async def _wait_delay(self, delay_seconds: float) -> None:
        """
        Wait for a specified delay.

        Args:
            delay_seconds: Delay in seconds
        """
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
