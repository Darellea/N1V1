"""
Smart Order Executor

Executes large orders by splitting them into smaller parts with delays.
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional
from decimal import Decimal

from .base_executor import BaseExecutor
from core.contracts import TradingSignal
from core.types.order_types import Order, OrderStatus, OrderType
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class SmartOrderExecutor(BaseExecutor):
    """
    Smart order executor that splits large orders into smaller parts.

    This executor automatically detects large orders and splits them into
    multiple smaller orders with configurable delays between executions.
    """

    def __init__(self, config: Dict[str, Any], exchange_api=None):
        """
        Initialize the smart order executor.

        Args:
            config: Configuration dictionary
            exchange_api: Exchange API instance for order execution
        """
        super().__init__(config)
        self.exchange_api = exchange_api

        # Configuration
        self.split_threshold = Decimal(str(config.get("split_threshold", 5000)))
        self.max_parts = config.get("max_parts", 5)
        self.delay_seconds = config.get("delay_seconds", 2.0)
        self.fallback_mode = config.get("fallback_mode", "market")

        self.logger.info(f"SmartOrderExecutor initialized: threshold={self.split_threshold}, "
                        f"max_parts={self.max_parts}, delay={self.delay_seconds}s")

    async def execute_order(self, signal: TradingSignal) -> List[Order]:
        """
        Execute an order using smart splitting logic.

        Args:
            signal: Trading signal to execute

        Returns:
            List of executed orders
        """
        if not self._validate_signal(signal):
            return []

        # Determine if order needs splitting
        order_value = self._calculate_order_value(signal)

        if order_value > self.split_threshold:
            # Split the order
            return await self._execute_split_order(signal)
        else:
            # Execute as single order
            return await self._execute_single_order(signal)

    async def _execute_split_order(self, signal: TradingSignal) -> List[Order]:
        """
        Execute a split order.

        Args:
            signal: Trading signal to split and execute

        Returns:
            List of executed orders
        """
        # Calculate number of parts
        order_value = self._calculate_order_value(signal)
        parts = min(self.max_parts, max(2, int(order_value / self.split_threshold)))

        # Split the amount
        split_amounts = await self.split_order(signal.amount, parts)

        self.logger.info(f"Smart split order triggered: {signal.symbol} "
                        f"size={signal.amount}, value={order_value}, "
                        f"parts={parts}, threshold={self.split_threshold}")

        trade_logger.trade("Smart Order Split", {
            "symbol": signal.symbol,
            "original_amount": float(signal.amount),
            "order_value": float(order_value),
            "parts": parts,
            "split_threshold": float(self.split_threshold)
        })

        # Execute each part with delay
        executed_orders = []
        parent_order_id = str(uuid.uuid4())

        for i, amount in enumerate(split_amounts):
            try:
                # Create child order
                child_signal = self._create_child_order(signal, amount, parent_order_id, i + 1)
                child_signal.metadata['total_parts'] = parts

                # Execute child order
                order = await self._execute_single_order(child_signal)
                if order:
                    executed_orders.extend(order)

                # Wait between orders (except for the last one)
                if i < len(split_amounts) - 1:
                    await self._wait_delay(self.delay_seconds)

            except Exception as e:
                self.logger.error(f"Failed to execute part {i+1}/{parts}: {e}")
                # Continue with remaining parts or cancel all?
                # For now, continue but log the error

        self.logger.info(f"Smart split order completed: {len(executed_orders)}/{parts} parts executed")
        return executed_orders

    async def _execute_single_order(self, signal: TradingSignal) -> List[Order]:
        """
        Execute a single order.

        Args:
            signal: Trading signal to execute

        Returns:
            List containing the executed order
        """
        try:
            if self.exchange_api:
                # Execute via exchange API
                order_response = await self._place_order_via_api(signal)
                order = self._parse_order_response(order_response)
            else:
                # Create mock order for testing/paper trading
                order = self._create_mock_order(signal)

            return [order] if order else []

        except Exception as e:
            self.logger.error(f"Failed to execute order: {e}")
            return []

    async def _place_order_via_api(self, signal: TradingSignal) -> Dict[str, Any]:
        """
        Place order via exchange API.

        Args:
            signal: Trading signal

        Returns:
            Order response from exchange
        """
        if not self.exchange_api:
            raise ValueError("Exchange API not configured")

        # Determine order parameters
        side = "buy" if signal.signal_type.value == "ENTRY_LONG" else "sell"
        order_type = signal.order_type.value if signal.order_type else "market"

        params = {
            'symbol': signal.symbol,
            'type': order_type,
            'side': side,
            'amount': float(signal.amount)
        }

        if signal.price and order_type == "limit":
            params['price'] = float(signal.price)

        # Add metadata
        if signal.metadata:
            params['metadata'] = signal.metadata

        # Place the order
        response = await self.exchange_api.create_order(**params)
        return response

    def _parse_order_response(self, response: Dict[str, Any]) -> Order:
        """
        Parse exchange order response.

        Args:
            response: Raw order response

        Returns:
            Parsed Order object
        """
        # This would be similar to the existing order parsing logic
        # For now, return a basic order
        return Order(
            id=str(response.get('id', '')),
            symbol=response.get('symbol', ''),
            type=OrderType(response.get('type', 'market')),
            side=response.get('side', ''),
            amount=Decimal(str(response.get('amount', 0))),
            price=Decimal(str(response.get('price', 0))) if response.get('price') else None,
            status=OrderStatus(response.get('status', 'open')),
            timestamp=response.get('timestamp', 0),
            filled=Decimal(str(response.get('filled', 0))),
            remaining=Decimal(str(response.get('remaining', 0))),
            cost=Decimal(str(response.get('cost', 0))),
            fee=response.get('fee')
        )

    def _create_mock_order(self, signal: TradingSignal) -> Order:
        """
        Create a mock order for testing/paper trading.

        Args:
            signal: Trading signal

        Returns:
            Mock Order object
        """
        return Order(
            id=str(uuid.uuid4()),
            symbol=signal.symbol,
            type=signal.order_type or OrderType.MARKET,
            side="buy" if signal.signal_type.value == "ENTRY_LONG" else "sell",
            amount=signal.amount,
            price=signal.price,
            status=OrderStatus.FILLED,  # Assume filled for mock
            timestamp=signal.timestamp,
            filled=signal.amount,
            remaining=Decimal(0),
            cost=signal.amount * (signal.price or Decimal(1)),
            fee={'cost': Decimal(0), 'currency': 'USD'}
        )

    def _calculate_order_value(self, signal: TradingSignal) -> Decimal:
        """
        Calculate the approximate value of an order.

        Args:
            signal: Trading signal

        Returns:
            Order value
        """
        # Use current price or signal price
        price = signal.current_price or signal.price or Decimal(1)
        return signal.amount * price

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            True if cancellation was successful
        """
        try:
            if self.exchange_api:
                await self.exchange_api.cancel_order(order_id)
            self.logger.info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
