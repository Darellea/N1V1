"""
VWAP Executor

Executes orders using Volume-Weighted Average Price strategy.
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


class VWAPExecutor(BaseExecutor):
    """
    Volume-Weighted Average Price executor.

    Executes orders by weighting execution towards periods of higher liquidity
    based on historical volume profiles.
    """

    def __init__(self, config: Dict[str, Any], exchange_api=None):
        """
        Initialize the VWAP executor.

        Args:
            config: Configuration dictionary
            exchange_api: Exchange API instance for order execution
        """
        super().__init__(config)
        self.exchange_api = exchange_api

        # Configuration
        self.lookback_minutes = config.get("lookback_minutes", 60)
        self.parts = config.get("parts", 10)
        self.fallback_mode = config.get("fallback_mode", "market")

        self.logger.info(f"VWAPExecutor initialized: lookback={self.lookback_minutes}min, "
                        f"parts={self.parts}")

    async def execute_order(self, signal: TradingSignal) -> List[Order]:
        """
        Execute an order using VWAP strategy.

        Args:
            signal: Trading signal to execute

        Returns:
            List of executed orders
        """
        if not self._validate_signal(signal):
            return []

        self.logger.info(f"Executing VWAP order: {signal.symbol} {signal.amount} "
                        f"using {self.lookback_minutes}min volume profile")

        trade_logger.trade("VWAP Order Start", {
            "symbol": signal.symbol,
            "amount": float(signal.amount),
            "lookback_minutes": self.lookback_minutes,
            "parts": self.parts
        })

        # Get volume profile and calculate execution weights
        volume_profile = await self._get_volume_profile(signal.symbol)
        execution_weights = self._calculate_execution_weights(volume_profile)

        # Split order based on volume weights
        split_amounts = self._split_by_volume_weights(signal.amount, execution_weights)

        # Execute parts according to volume profile
        executed_orders = []
        start_time = time.time()
        parent_order_id = str(uuid.uuid4())

        for i, amount in enumerate(split_amounts):
            try:
                # Wait for next high-volume period if needed
                if i > 0:
                    await self._wait_for_volume_period(volume_profile, i)

                # Create child order
                child_signal = self._create_child_order(signal, amount, parent_order_id, i + 1)
                child_signal.metadata.update({
                    'total_parts': self.parts,
                    'volume_weight': float(execution_weights[i]),
                    'execution_period': i
                })

                # Execute child order
                order = await self._execute_single_order(child_signal)
                if order:
                    executed_orders.extend(order)

                # Log progress
                progress = (i + 1) / self.parts * 100
                self.logger.debug(f"VWAP progress: {i+1}/{self.parts} parts ({progress:.1f}%)")

            except Exception as e:
                self.logger.error(f"Failed to execute VWAP part {i+1}/{self.parts}: {e}")
                # Continue with remaining parts

        # Log completion
        completion_time = time.time() - start_time
        self.logger.info(f"VWAP order completed: {len(executed_orders)}/{self.parts} parts "
                        f"in {completion_time:.1f}s")

        trade_logger.trade("VWAP Order Complete", {
            "symbol": signal.symbol,
            "parts_executed": len(executed_orders),
            "total_parts": self.parts,
            "duration_actual": completion_time
        })

        return executed_orders

    async def _get_volume_profile(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get historical volume profile for the symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of volume data by time period
        """
        # In a real implementation, this would fetch historical volume data
        # For now, create a mock volume profile
        profile = []

        # Simulate volume profile over the lookback period
        # Higher volume during certain hours (e.g., market open/close)
        for i in range(self.parts):
            hour = (datetime.now().hour + i) % 24

            # Simulate higher volume during active hours
            if 9 <= hour <= 16:  # Typical market hours
                base_volume = 100
            else:
                base_volume = 30

            # Add some randomness
            volume = base_volume * (0.5 + 0.5 * (i % 3) / 2)  # Vary by period

            profile.append({
                'period': i,
                'volume': volume,
                'hour': hour,
                'is_high_volume': volume > 60
            })

        return profile

    def _calculate_execution_weights(self, volume_profile: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate execution weights based on volume profile.

        Args:
            volume_profile: Historical volume data

        Returns:
            List of execution weights (0-1)
        """
        if not volume_profile:
            # Equal weights if no profile available
            return [1.0 / self.parts] * self.parts

        # Extract volumes
        volumes = [period['volume'] for period in volume_profile]

        # Calculate weights proportional to volume
        total_volume = sum(volumes)
        if total_volume == 0:
            return [1.0 / self.parts] * self.parts

        weights = [volume / total_volume for volume in volumes]

        # Normalize to ensure they sum to 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        return normalized_weights

    def _split_by_volume_weights(self, total_amount: Decimal, weights: List[float]) -> List[Decimal]:
        """
        Split order amount based on volume weights.

        Args:
            total_amount: Total order amount
            weights: Execution weights

        Returns:
            List of split amounts
        """
        amounts = []
        remaining = total_amount

        for i, weight in enumerate(weights):
            if i == len(weights) - 1:
                # Last part gets the remainder
                amounts.append(remaining)
            else:
                amount = total_amount * Decimal(str(weight))
                amount = round(amount, 8)  # Round to 8 decimal places
                amounts.append(amount)
                remaining -= amount

        return amounts

    async def _wait_for_volume_period(self, volume_profile: List[Dict[str, Any]], period_index: int) -> None:
        """
        Wait for the next high-volume period if needed.

        Args:
            volume_profile: Volume profile data
            period_index: Current period index
        """
        if period_index >= len(volume_profile):
            return

        current_period = volume_profile[period_index]

        # If current period is high volume, no need to wait
        if current_period.get('is_high_volume', False):
            return

        # Find next high volume period
        next_high_volume = None
        for i in range(period_index + 1, len(volume_profile)):
            if volume_profile[i].get('is_high_volume', False):
                next_high_volume = volume_profile[i]
                break

        if next_high_volume:
            # Wait for high volume period (simulate time-based waiting)
            wait_seconds = 60  # Wait 1 minute for next period
            self.logger.debug(f"Waiting {wait_seconds}s for high volume period")
            await self._wait_delay(wait_seconds)

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
            self.logger.error(f"Failed to execute VWAP order part: {e}")
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

        # Add VWAP metadata
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
            self.logger.info(f"Cancelled VWAP order: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel VWAP order {order_id}: {e}")
            return False

    def get_volume_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Get volume profile analysis for the symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Volume profile analysis
        """
        # This would analyze historical volume patterns
        # For now, return mock analysis
        return {
            "symbol": symbol,
            "lookback_minutes": self.lookback_minutes,
            "high_volume_periods": [9, 10, 11, 14, 15, 16],  # Example hours
            "average_volume": 75.0,
            "peak_volume": 120.0,
            "volume_distribution": "normal"
        }
