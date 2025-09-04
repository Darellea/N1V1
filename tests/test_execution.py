"""
Unit tests for advanced execution strategies.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
import pandas as pd

from core.execution import (
    BaseExecutor,
    SmartOrderExecutor,
    TWAPExecutor,
    VWAPExecutor,
    DCAExecutor
)
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types.order_types import Order, OrderStatus, OrderType


class TestBaseExecutor:
    """Test BaseExecutor abstract class."""

    def test_abstract_methods(self):
        """Test that BaseExecutor defines abstract methods."""
        config = {"test": "config"}

        # Should raise TypeError when trying to instantiate abstract class
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseExecutor(config)

    @pytest.mark.asyncio
    async def test_split_order(self):
        """Test order splitting logic."""
        config = {}
        executor = SmartOrderExecutor(config)

        # Test normal splitting
        amounts = await executor.split_order(Decimal(10), 3)
        assert len(amounts) == 3
        assert sum(amounts) == Decimal(10)
        assert all(amount > 0 for amount in amounts)

        # Test single part
        amounts = await executor.split_order(Decimal(10), 1)
        assert len(amounts) == 1
        assert amounts[0] == Decimal(10)

    def test_validate_signal(self):
        """Test signal validation."""
        config = {}
        executor = SmartOrderExecutor(config)

        # Valid signal
        valid_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal(1),
            timestamp=1234567890
        )
        assert executor._validate_signal(valid_signal)

        # Invalid signal - no symbol
        invalid_signal = TradingSignal(
            strategy_id="test",
            symbol="",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal(1),
            timestamp=1234567890
        )
        assert not executor._validate_signal(invalid_signal)

        # Invalid signal - zero amount
        invalid_signal2 = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal(0),
            timestamp=1234567890
        )
        assert not executor._validate_signal(invalid_signal2)


class TestSmartOrderExecutor:
    """Test SmartOrderExecutor."""

    def test_initialization(self):
        """Test SmartOrderExecutor initialization."""
        config = {
            "split_threshold": 5000,
            "max_parts": 5,
            "delay_seconds": 2
        }
        executor = SmartOrderExecutor(config)
        assert executor.split_threshold == Decimal(5000)
        assert executor.max_parts == 5
        assert executor.delay_seconds == 2.0

    def test_calculate_order_value(self):
        """Test order value calculation."""
        config = {"split_threshold": 5000}
        executor = SmartOrderExecutor(config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal(10),
            price=Decimal(50000),
            timestamp=1234567890
        )

        value = executor._calculate_order_value(signal)
        assert value == Decimal(500000)  # 10 * 50000

    @pytest.mark.asyncio
    async def test_execute_small_order(self):
        """Test execution of small orders (no splitting)."""
        config = {"split_threshold": 5000}
        executor = SmartOrderExecutor(config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal(1),
            price=Decimal(1000),  # Lower price so value < threshold
            timestamp=1234567890
        )

        # Mock the single order execution
        with patch.object(executor, '_execute_single_order', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [Order(
                id="test_order",
                symbol="BTC/USDT",
                type=OrderType.MARKET,
                side="buy",
                amount=Decimal(1),
                status=OrderStatus.FILLED,
                timestamp=1234567890
            )]

            orders = await executor.execute_order(signal)

            assert len(orders) == 1
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_large_order(self):
        """Test execution of large orders (with splitting)."""
        config = {
            "split_threshold": 1000,
            "max_parts": 3,
            "delay_seconds": 0.1  # Fast for testing
        }
        executor = SmartOrderExecutor(config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal(5),  # Large amount
            price=Decimal(3000),  # High price to exceed threshold
            timestamp=1234567890
        )

        # Mock the single order execution
        with patch.object(executor, '_execute_single_order', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [Order(
                id="test_order",
                symbol="BTC/USDT",
                type=OrderType.MARKET,
                side="buy",
                amount=Decimal(1),
                status=OrderStatus.FILLED,
                timestamp=1234567890
            )]

            orders = await executor.execute_order(signal)

            # Should have been split into parts
            assert len(orders) >= 1
            # Verify multiple calls to execute_single_order
            assert mock_execute.call_count > 1


class TestTWAPExecutor:
    """Test TWAPExecutor."""

    def test_initialization(self):
        """Test TWAPExecutor initialization."""
        config = {
            "duration_minutes": 30,
            "parts": 10
        }
        executor = TWAPExecutor(config)
        assert executor.duration_minutes == 30
        assert executor.parts == 10
        assert executor.total_duration_seconds == 1800  # 30 * 60
        assert executor.interval_seconds == 180  # 1800 / 10

    @pytest.mark.asyncio
    async def test_execute_order(self):
        """Test TWAP order execution."""
        config = {
            "duration_minutes": 1,  # Short for testing
            "parts": 3
        }
        executor = TWAPExecutor(config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal(3),
            timestamp=1234567890
        )

        # Mock the single order execution
        with patch.object(executor, '_execute_single_order', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [Order(
                id="test_order",
                symbol="BTC/USDT",
                type=OrderType.MARKET,
                side="buy",
                amount=Decimal(1),
                status=OrderStatus.FILLED,
                timestamp=1234567890
            )]

            orders = await executor.execute_order(signal)

            # Should execute all parts
            assert len(orders) == 3
            assert mock_execute.call_count == 3

    @pytest.mark.asyncio
    async def test_get_execution_schedule(self):
        """Test execution schedule generation."""
        config = {"duration_minutes": 10, "parts": 5}
        executor = TWAPExecutor(config)

        schedule = await executor.get_execution_schedule()

        assert len(schedule) == 5
        assert schedule[0]['part'] == 1
        assert schedule[4]['part'] == 5

        # Check timing
        assert schedule[0]['execution_time_seconds'] == 0
        assert schedule[1]['execution_time_seconds'] == 120  # 10min * 60s / 5 parts


class TestVWAPExecutor:
    """Test VWAPExecutor."""

    def test_initialization(self):
        """Test VWAPExecutor initialization."""
        config = {
            "lookback_minutes": 60,
            "parts": 10
        }
        executor = VWAPExecutor(config)
        assert executor.lookback_minutes == 60
        assert executor.parts == 10

    def test_calculate_execution_weights(self):
        """Test execution weight calculation."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        volume_profile = [
            {'period': 0, 'volume': 100},
            {'period': 1, 'volume': 200},
            {'period': 2, 'volume': 150}
        ]

        weights = executor._calculate_execution_weights(volume_profile)

        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 0.001  # Should sum to 1
        assert weights[1] > weights[0]  # Higher volume should have higher weight

    def test_split_by_volume_weights(self):
        """Test order splitting by volume weights."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        total_amount = Decimal(100)
        weights = [0.2, 0.5, 0.3]

        amounts = executor._split_by_volume_weights(total_amount, weights)

        assert len(amounts) == 3
        assert sum(amounts) == total_amount
        assert amounts[1] > amounts[0]  # Higher weight should result in larger amount


class TestDCAExecutor:
    """Test DCAExecutor."""

    def test_initialization(self):
        """Test DCAExecutor initialization."""
        config = {
            "interval_minutes": 60,
            "parts": 5
        }
        executor = DCAExecutor(config)
        assert executor.interval_minutes == 60
        assert executor.parts == 5
        assert executor.interval_seconds == 3600  # 60 * 60

    @pytest.mark.asyncio
    async def test_start_dca_session(self):
        """Test starting a DCA session."""
        config = {
            "interval_minutes": 1,  # Short for testing
            "parts": 3
        }
        executor = DCAExecutor(config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal(3),
            timestamp=1234567890
        )

        # Mock the single order execution
        with patch.object(executor, '_execute_single_order', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [Order(
                id="test_order",
                symbol="BTC/USDT",
                type=OrderType.MARKET,
                side="buy",
                amount=Decimal(1),
                status=OrderStatus.FILLED,
                timestamp=1234567890
            )]

            orders = await executor.execute_order(signal)

            # Should execute first part immediately
            assert len(orders) == 1
            assert mock_execute.call_count == 1

            # Should have created a session
            assert len(executor.active_sessions) == 1

    @pytest.mark.asyncio
    async def test_continue_dca_session(self):
        """Test continuing a DCA session."""
        config = {
            "interval_minutes": 1,  # Short for testing
            "parts": 3
        }
        executor = DCAExecutor(config)

        # Create a mock session
        session_id = "test_session"
        executor.active_sessions[session_id] = {
            'session_id': session_id,
            'symbol': 'BTC/USDT',
            'total_amount': Decimal(3),
            'remaining_amount': Decimal(3),
            'parts': 3,
            'executed_parts': 0,
            'split_amounts': [Decimal(1), Decimal(1), Decimal(1)],
            'start_time': time.time() - 70,  # Started 70 seconds ago
            'next_execution': time.time() - 10,  # Should execute now
            'executed_orders': [],
            'parent_signal': TradingSignal(
                strategy_id="test",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.WEAK,
                order_type=OrderType.MARKET,
                amount=Decimal(3),
                timestamp=1234567890
            )
        }

        # Create continuation signal with session ID
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal(1),
            timestamp=1234567890,
            metadata={'dca_session_id': session_id}
        )

        # Mock the single order execution
        with patch.object(executor, '_execute_single_order', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [Order(
                id="test_order",
                symbol="BTC/USDT",
                type=OrderType.MARKET,
                side="buy",
                amount=Decimal(1),
                status=OrderStatus.FILLED,
                timestamp=1234567890
            )]

            orders = await executor.execute_order(signal)

            # Should execute next part
            assert len(orders) == 1
            assert mock_execute.call_count == 1

    def test_get_dca_sessions(self):
        """Test getting DCA session information."""
        config = {"interval_minutes": 60, "parts": 3}
        executor = DCAExecutor(config)

        # Add a mock session
        session_id = "test_session"
        executor.active_sessions[session_id] = {
            'session_id': session_id,
            'symbol': 'BTC/USDT',
            'total_amount': Decimal(3),
            'remaining_amount': Decimal(2),
            'parts': 3,
            'executed_parts': 1,
            'split_amounts': [Decimal(1), Decimal(1), Decimal(1)],
            'start_time': time.time(),
            'next_execution': time.time() + 3600,
            'executed_orders': ['order1'],
            'parent_signal': TradingSignal(
                strategy_id="test",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.WEAK,
                order_type=OrderType.MARKET,
                amount=Decimal(3),
                timestamp=1234567890
            )
        }

        sessions = executor.get_dca_sessions()

        assert len(sessions) == 1
        assert sessions[0]['session_id'] == session_id
        assert sessions[0]['symbol'] == 'BTC/USDT'
        assert sessions[0]['executed_parts'] == 1
        assert sessions[0]['progress'] == 1/3

    def test_get_session_status(self):
        """Test getting detailed session status."""
        config = {"interval_minutes": 60, "parts": 3}
        executor = DCAExecutor(config)

        session_id = "test_session"
        executor.active_sessions[session_id] = {
            'session_id': session_id,
            'symbol': 'BTC/USDT',
            'total_amount': Decimal(3),
            'remaining_amount': Decimal(2),
            'parts': 3,
            'executed_parts': 1,
            'split_amounts': [Decimal(1), Decimal(1), Decimal(1)],
            'start_time': time.time(),
            'next_execution': time.time() + 3600,
            'executed_orders': ['order1'],
            'parent_signal': TradingSignal(
                strategy_id="test",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.WEAK,
                order_type=OrderType.MARKET,
                amount=Decimal(3),
                timestamp=1234567890
            )
        }

        status = executor.get_session_status(session_id)

        assert status is not None
        assert status['session_id'] == session_id
        assert status['progress_percentage'] == 33.33333333333333  # 1/3
        assert status['seconds_until_next'] > 0


if __name__ == "__main__":
    pytest.main([__file__])
