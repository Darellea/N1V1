"""
TWAP Executor

Executes orders using Time-Weighted Average Price strategy.
"""

import logging
import asyncio
import uuid
import time
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime, timedelta

from .base_executor import BaseExecutor
from core.contracts import TradingSignal
from core.types.order_types import Order, OrderStatus, OrderType
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class TWAPExecutor(BaseExecutor):
    """
    Time-Weighted Average Price executor.

    Executes orders by splitting them into equal parts over a specified time period.
    This reduces market impact and provides better average execution prices.
    """

    def __init__(self, config: Dict[str, Any], exchange_api=None):
        """
        Initialize the TWAP executor.

        Args:
            config: Configuration dictionary
            exchange_api: Exchange API instance for order execution
        """
        super().__init__(config)
        self.exchange_api = exchange_api

        # Configuration
        self.duration_minutes = config.get("duration_minutes", 30)
        self.parts = config.get("parts", 10)
        self.fallback_mode = config.get("fallback_mode", "market")

        # Calculate timing
        self.total_duration_seconds = self.duration_minutes * 60
        self.interval_seconds = self.total_duration_seconds / self.parts

        self.logger.info(f"TWAPExecutor initialized: duration={self.duration_minutes}min, "
                        f"parts={self.parts}, interval={self.interval_seconds:.1f}s")

    async def execute_order(self, signal: TradingSignal) -> List[Order]:
        """
        Execute an order using TWAP strategy.

        Args:
            signal: Trading signal to execute

        Returns:
            List of executed orders
        """
        if not self._validate_signal(signal):
            return []

        self.logger.info(f"Executing TWAP order: {signal.symbol} {signal.amount} "
                        f"over {self.duration_minutes} minutes in {self.parts} parts")

        trade_logger.trade("TWAP Order Start", {
            "symbol": signal.symbol,
            "amount": float(signal.amount),
            "duration_minutes": self.duration_minutes,
            "parts": self.parts,
            "interval_seconds": self.interval_seconds
        })

        # Split the order into equal parts
        split_amounts = await self.split_order(signal.amount, self.parts)

        # Execute parts with time intervals
        executed_orders = []
        start_time = time.time()
        parent_order_id = str(uuid.uuid4())

        for i, amount in enumerate(split_amounts):
            try:
                # Calculate delay until next execution
                if i > 0:
                    elapsed = time.time() - start_time
                    expected_time = i * self.interval_seconds
                    delay = max(0, expected_time - elapsed)
                    await self._wait_delay(delay)

                # Create child order
                child_signal = self._create_child_order(signal, amount, parent_order_id, i + 1)
                child_signal.metadata.update({
                    'total_parts': self.parts,
                    'part_interval': self.interval_seconds,
                    'total_duration': self.total_duration_seconds
                })

                # Execute child order
                order = await self._execute_single_order(child_signal)
                if order:
                    executed_orders.extend(order)

                # Log progress
                progress = (i + 1) / self.parts * 100
                self.logger.debug(f"TWAP progress: {i+1}/{self.parts} parts ({progress:.1f}%)")

            except Exception as e:
                self.logger.error(f"Failed to execute TWAP part {i+1}/{self.parts}: {e}")
                # Continue with remaining parts

        # Log completion
        completion_time = time.time() - start_time
        self.logger.info(f"TWAP order completed: {len(executed_orders)}/{self.parts} parts "
                        f"in {completion_time:.1f}s")

        trade_logger.trade("TWAP Order Complete", {
            "symbol": signal.symbol,
            "parts_executed": len(executed_orders),
            "total_parts": self.parts,
            "duration_actual": completion_time,
            "duration_expected": self.total_duration_seconds
        })

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
            self.logger.error(f"Failed to execute TWAP order part: {e}")
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

        # Add TWAP metadata
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
            status=OrderStatus.FILLED,
            timestamp=signal.timestamp,
            filled=signal.amount,
            remaining=Decimal(0),
            cost=signal.amount * (signal.price or Decimal(1)),
            fee={'cost': Decimal(0), 'currency': 'USD'}
        )

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
            self.logger.info(f"Cancelled TWAP order: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel TWAP order {order_id}: {e}")
            return False

    async def get_execution_schedule(self) -> List[Dict[str, Any]]:
        """
        Get the execution schedule for the TWAP order.

        Returns:
            List of execution times and amounts
        """
        schedule = []
        split_amounts = await self.split_order(Decimal(1), self.parts)  # Use 1 as base for percentages

        for i, amount in enumerate(split_amounts):
            execution_time = i * self.interval_seconds
            schedule.append({
                'part': i + 1,
                'amount': float(amount),
                'execution_time_seconds': execution_time,
                'cumulative_time': execution_time
            })

        return schedule
