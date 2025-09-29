"""
Unit tests for advanced execution strategies.
"""

import time
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.contracts import SignalStrength, SignalType, TradingSignal
from core.execution import (
    BaseExecutor,
    DCAExecutor,
    SmartOrderExecutor,
    TWAPExecutor,
    VWAPExecutor,
)
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
            timestamp=1234567890,
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
            timestamp=1234567890,
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
            timestamp=1234567890,
        )
        assert not executor._validate_signal(invalid_signal2)


class TestSmartOrderExecutor:
    """Test SmartOrderExecutor."""

    def test_initialization(self):
        """Test SmartOrderExecutor initialization."""
        config = {"split_threshold": 5000, "max_parts": 5, "delay_seconds": 2}
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
            timestamp=1234567890,
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
            timestamp=1234567890,
        )

        # Mock the single order execution
        with patch.object(
            executor, "_execute_single_order", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = [
                Order(
                    id="test_order",
                    symbol="BTC/USDT",
                    type=OrderType.MARKET,
                    side="buy",
                    amount=Decimal(1),
                    status=OrderStatus.FILLED,
                    timestamp=1234567890,
                )
            ]

            orders = await executor.execute_order(signal)

            assert len(orders) == 1
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_large_order(self):
        """Test execution of large orders (with splitting)."""
        config = {
            "split_threshold": 1000,
            "max_parts": 3,
            "delay_seconds": 0.1,  # Fast for testing
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
            timestamp=1234567890,
        )

        # Mock the single order execution
        with patch.object(
            executor, "_execute_single_order", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = [
                Order(
                    id="test_order",
                    symbol="BTC/USDT",
                    type=OrderType.MARKET,
                    side="buy",
                    amount=Decimal(1),
                    status=OrderStatus.FILLED,
                    timestamp=1234567890,
                )
            ]

            orders = await executor.execute_order(signal)

            # Should have been split into parts
            assert len(orders) >= 1
            # Verify multiple calls to execute_single_order
            assert mock_execute.call_count > 1


class TestTWAPExecutor:
    """Test TWAPExecutor."""

    def test_initialization(self):
        """Test TWAPExecutor initialization."""
        config = {"duration_minutes": 30, "parts": 10}
        executor = TWAPExecutor(config)
        assert executor.duration_minutes == 30
        assert executor.parts == 10
        assert executor.total_duration_seconds == 1800  # 30 * 60
        assert executor.interval_seconds == 180  # 1800 / 10

    @pytest.mark.asyncio
    async def test_execute_order(self):
        """Test TWAP order execution."""
        config = {"duration_minutes": 1, "parts": 3}  # Short for testing
        executor = TWAPExecutor(config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal(3),
            timestamp=1234567890,
        )

        # Mock the single order execution
        with patch.object(
            executor, "_execute_single_order", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = [
                Order(
                    id="test_order",
                    symbol="BTC/USDT",
                    type=OrderType.MARKET,
                    side="buy",
                    amount=Decimal(1),
                    status=OrderStatus.FILLED,
                    timestamp=1234567890,
                )
            ]

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
        assert schedule[0]["part"] == 1
        assert schedule[4]["part"] == 5

        # Check timing
        assert schedule[0]["execution_time_seconds"] == 0
        assert schedule[1]["execution_time_seconds"] == 120  # 10min * 60s / 5 parts


class TestVWAPExecutor:
    """Test VWAPExecutor."""

    def test_initialization(self):
        """Test VWAPExecutor initialization."""
        config = {"lookback_minutes": 60, "parts": 10}
        executor = VWAPExecutor(config)
        assert executor.lookback_minutes == 60
        assert executor.parts == 10

    def test_initialization_with_exchange_api(self):
        """Test VWAPExecutor initialization with exchange API."""
        config = {"lookback_minutes": 30, "parts": 5}
        mock_api = Mock()
        executor = VWAPExecutor(config, exchange_api=mock_api)
        assert executor.exchange_api is mock_api
        assert executor.lookback_minutes == 30
        assert executor.parts == 5

    @pytest.mark.asyncio
    async def test_get_volume_profile(self):
        """Test volume profile generation."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        profile = await executor._get_volume_profile("BTC/USDT")

        assert len(profile) == 3
        assert all("period" in period for period in profile)
        assert all("volume" in period for period in profile)
        assert all("hour" in period for period in profile)
        assert all("is_high_volume" in period for period in profile)

    def test_calculate_execution_weights(self):
        """Test execution weight calculation."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        volume_profile = [
            {"period": 0, "volume": 100},
            {"period": 1, "volume": 200},
            {"period": 2, "volume": 150},
        ]

        weights = executor._calculate_execution_weights(volume_profile)

        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 0.001  # Should sum to 1
        assert weights[1] > weights[0]  # Higher volume should have higher weight

    def test_calculate_execution_weights_empty_profile(self):
        """Test execution weight calculation with empty profile."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        weights = executor._calculate_execution_weights([])

        assert len(weights) == 3
        assert all(w == 1.0 / 3 for w in weights)  # Equal weights

    def test_calculate_execution_weights_zero_volume(self):
        """Test execution weight calculation with zero volume."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        volume_profile = [
            {"period": 0, "volume": 0},
            {"period": 1, "volume": 0},
            {"period": 2, "volume": 0},
        ]

        weights = executor._calculate_execution_weights(volume_profile)

        assert len(weights) == 3
        assert all(w == 1.0 / 3 for w in weights)  # Equal weights

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

    def test_split_by_volume_weights_single_part(self):
        """Test order splitting with single part."""
        config = {"lookback_minutes": 60, "parts": 1}
        executor = VWAPExecutor(config)

        total_amount = Decimal(100)
        weights = [1.0]

        amounts = executor._split_by_volume_weights(total_amount, weights)

        assert len(amounts) == 1
        assert amounts[0] == total_amount

    @pytest.mark.asyncio
    async def test_wait_for_volume_period_high_volume(self):
        """Test waiting for volume period when current period is high volume."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        volume_profile = [
            {"period": 0, "volume": 100, "is_high_volume": True},
            {"period": 1, "volume": 50, "is_high_volume": False},
            {"period": 2, "volume": 80, "is_high_volume": True},
        ]

        # Should not wait when current period is high volume
        with patch.object(executor, "_wait_delay") as mock_wait:
            await executor._wait_for_volume_period(volume_profile, 0)
            mock_wait.assert_not_called()

    @pytest.mark.asyncio
    async def test_wait_for_volume_period_low_volume(self):
        """Test waiting for volume period when current period is low volume."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        volume_profile = [
            {"period": 0, "volume": 30, "is_high_volume": False},
            {"period": 1, "volume": 50, "is_high_volume": False},
            {"period": 2, "volume": 80, "is_high_volume": True},
        ]

        # Should wait when current period is low volume and next is high volume
        with patch.object(executor, "_wait_delay") as mock_wait:
            await executor._wait_for_volume_period(volume_profile, 0)
            mock_wait.assert_called_once_with(60)  # Wait 60 seconds

    @pytest.mark.asyncio
    async def test_wait_for_volume_period_no_high_volume(self):
        """Test waiting for volume period when no high volume period exists."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        volume_profile = [
            {"period": 0, "volume": 30, "is_high_volume": False},
            {"period": 1, "volume": 40, "is_high_volume": False},
            {"period": 2, "volume": 50, "is_high_volume": False},
        ]

        # Should not wait when no high volume period exists
        with patch.object(executor, "_wait_delay") as mock_wait:
            await executor._wait_for_volume_period(volume_profile, 0)
            mock_wait.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_single_order_with_api(self):
        """Test single order execution with exchange API."""
        config = {"lookback_minutes": 60, "parts": 3}
        mock_api = AsyncMock()
        mock_api.create_order.return_value = {
            "id": "12345",
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "buy",
            "amount": 1.0,
            "status": "filled",
        }
        executor = VWAPExecutor(config, exchange_api=mock_api)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal(1),
            timestamp=1234567890,
        )

        orders = await executor._execute_single_order(signal)

        assert len(orders) == 1
        assert orders[0].id == "12345"
        assert orders[0].symbol == "BTC/USDT"
        mock_api.create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_single_order_without_api(self):
        """Test single order execution without exchange API."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal(1),
            timestamp=1234567890,
        )

        orders = await executor._execute_single_order(signal)

        assert len(orders) == 1
        assert orders[0].symbol == "BTC/USDT"
        assert orders[0].status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_place_order_via_api_buy_order(self):
        """Test placing buy order via API."""
        config = {"lookback_minutes": 60, "parts": 3}
        mock_api = AsyncMock()
        mock_api.create_order.return_value = {"id": "12345"}
        executor = VWAPExecutor(config, exchange_api=mock_api)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal(1),
            timestamp=1234567890,
        )

        response = await executor._place_order_via_api(signal)

        assert response == {"id": "12345"}
        mock_api.create_order.assert_called_once()
        call_args = mock_api.create_order.call_args
        assert call_args[1]["symbol"] == "BTC/USDT"
        assert call_args[1]["side"] == "buy"
        assert call_args[1]["amount"] == 1.0

    @pytest.mark.asyncio
    async def test_place_order_via_api_sell_order(self):
        """Test placing sell order via API."""
        config = {"lookback_minutes": 60, "parts": 3}
        mock_api = AsyncMock()
        mock_api.create_order.return_value = {"id": "12345"}
        executor = VWAPExecutor(config, exchange_api=mock_api)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.EXIT_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.LIMIT,
            amount=Decimal(1),
            price=Decimal(50000),
            timestamp=1234567890,
        )

        response = await executor._place_order_via_api(signal)

        assert response == {"id": "12345"}
        mock_api.create_order.assert_called_once()
        call_args = mock_api.create_order.call_args
        assert call_args[1]["symbol"] == "BTC/USDT"
        assert call_args[1]["side"] == "sell"
        assert call_args[1]["type"] == "limit"
        assert call_args[1]["price"] == 50000.0

    @pytest.mark.asyncio
    async def test_place_order_via_api_no_exchange_api(self):
        """Test placing order via API without exchange API configured."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal(1),
            timestamp=1234567890,
        )

        with pytest.raises(ValueError, match="Exchange API not configured"):
            await executor._place_order_via_api(signal)

    def test_parse_order_response_complete(self):
        """Test parsing complete order response."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        response = {
            "id": "12345",
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "buy",
            "amount": 1.5,
            "price": 50000.0,
            "status": "filled",
            "timestamp": 1234567890,
            "filled": 1.5,
            "remaining": 0.0,
            "cost": 75000.0,
            "fee": {"cost": 7.5, "currency": "USDT"},
        }

        order = executor._parse_order_response(response)

        assert order.id == "12345"
        assert order.symbol == "BTC/USDT"
        assert order.type == OrderType.MARKET
        assert order.side == "buy"
        assert order.amount == Decimal("1.5")
        assert order.price == Decimal("50000.0")
        assert order.status == OrderStatus.FILLED
        assert order.filled == Decimal("1.5")
        assert order.cost == Decimal("75000.0")
        assert order.fee == {"cost": 7.5, "currency": "USDT"}

    def test_parse_order_response_minimal(self):
        """Test parsing minimal order response."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        response = {
            "id": "12345",
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "buy",
            "amount": 1.0,
        }

        order = executor._parse_order_response(response)

        assert order.id == "12345"
        assert order.symbol == "BTC/USDT"
        assert order.type == OrderType.MARKET
        assert order.side == "buy"
        assert order.amount == Decimal("1.0")
        assert order.price is None
        assert order.status == OrderStatus.OPEN
        assert order.filled == Decimal("0")
        assert order.cost == Decimal("0")

    def test_create_mock_order_market_buy(self):
        """Test creating mock market buy order."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal(1),
            price=Decimal(50000),
            timestamp=1234567890,
        )

        order = executor._create_mock_order(signal)

        assert order.symbol == "BTC/USDT"
        assert order.type == OrderType.MARKET
        assert order.side == "buy"
        assert order.amount == Decimal(1)
        assert order.price == Decimal(50000)
        assert order.status == OrderStatus.FILLED
        assert order.filled == Decimal(1)
        assert order.remaining == Decimal(0)
        assert order.cost == Decimal(50000)
        assert order.fee == {"cost": Decimal(0), "currency": "USD"}

    def test_create_mock_order_limit_sell(self):
        """Test creating mock limit sell order."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.EXIT_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.LIMIT,
            amount=Decimal(0.5),
            price=Decimal(51000),
            timestamp=1234567890,
        )

        order = executor._create_mock_order(signal)

        assert order.symbol == "BTC/USDT"
        assert order.type == OrderType.LIMIT
        assert order.side == "sell"
        assert order.amount == Decimal(0.5)
        assert order.price == Decimal(51000)
        assert order.status == OrderStatus.FILLED
        assert order.cost == Decimal(25500)  # 0.5 * 51000

    def test_create_mock_order_no_price(self):
        """Test creating mock order without price."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal(1),
            timestamp=1234567890,
        )

        order = executor._create_mock_order(signal)

        assert order.price is None
        assert order.cost == Decimal(1)  # amount * 1 when no price

    @pytest.mark.asyncio
    async def test_cancel_order_with_api(self):
        """Test canceling order with exchange API."""
        config = {"lookback_minutes": 60, "parts": 3}
        mock_api = AsyncMock()
        mock_api.cancel_order.return_value = True
        executor = VWAPExecutor(config, exchange_api=mock_api)

        result = await executor.cancel_order("12345")

        assert result is True
        mock_api.cancel_order.assert_called_once_with("12345")

    @pytest.mark.asyncio
    async def test_cancel_order_without_api(self):
        """Test canceling order without exchange API."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        result = await executor.cancel_order("12345")

        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_api_failure(self):
        """Test canceling order when API fails."""
        config = {"lookback_minutes": 60, "parts": 3}
        mock_api = AsyncMock()
        mock_api.cancel_order.side_effect = Exception("API Error")
        executor = VWAPExecutor(config, exchange_api=mock_api)

        result = await executor.cancel_order("12345")

        assert result is False
        mock_api.cancel_order.assert_called_once_with("12345")

    def test_get_volume_profile_analysis(self):
        """Test getting volume profile analysis."""
        config = {"lookback_minutes": 120, "parts": 5}
        executor = VWAPExecutor(config)

        analysis = executor.get_volume_profile("BTC/USDT")

        assert analysis["symbol"] == "BTC/USDT"
        assert analysis["lookback_minutes"] == 120
        assert "high_volume_periods" in analysis
        assert "average_volume" in analysis
        assert "peak_volume" in analysis
        assert "volume_distribution" in analysis

    @pytest.mark.asyncio
    async def test_execute_order_invalid_signal(self):
        """Test executing order with invalid signal."""
        config = {"lookback_minutes": 60, "parts": 3}
        executor = VWAPExecutor(config)

        invalid_signal = TradingSignal(
            strategy_id="test",
            symbol="",  # Invalid
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal(1),
            timestamp=1234567890,
        )

        orders = await executor.execute_order(invalid_signal)

        assert orders == []

    @pytest.mark.asyncio
    async def test_execute_order_with_execution_failure(self):
        """Test executing order when single order execution fails."""
        config = {"lookback_minutes": 60, "parts": 2}
        executor = VWAPExecutor(config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal(2),
            timestamp=1234567890,
        )

        # Mock _execute_single_order to fail on first call, succeed on second
        call_count = 0

        async def mock_execute_single_order(signal):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Execution failed")
            return [
                Order(
                    id=f"order_{call_count}",
                    symbol="BTC/USDT",
                    type=OrderType.MARKET,
                    side="buy",
                    amount=Decimal(1),
                    status=OrderStatus.FILLED,
                    timestamp=1234567890,
                )
            ]

        with patch.object(
            executor, "_execute_single_order", side_effect=mock_execute_single_order
        ):
            orders = await executor.execute_order(signal)

        # Should have executed second part despite first failure
        assert len(orders) == 1
        assert orders[0].id == "order_2"


class TestDCAExecutor:
    """Test DCAExecutor."""

    def test_initialization(self):
        """Test DCAExecutor initialization."""
        config = {"interval_minutes": 60, "parts": 5}
        executor = DCAExecutor(config)
        assert executor.interval_minutes == 60
        assert executor.parts == 5
        assert executor.interval_seconds == 3600  # 60 * 60

    @pytest.mark.asyncio
    async def test_start_dca_session(self):
        """Test starting a DCA session."""
        config = {"interval_minutes": 1, "parts": 3}  # Short for testing
        executor = DCAExecutor(config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal(3),
            timestamp=1234567890,
        )

        # Mock the single order execution
        with patch.object(
            executor, "_execute_single_order", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = [
                Order(
                    id="test_order",
                    symbol="BTC/USDT",
                    type=OrderType.MARKET,
                    side="buy",
                    amount=Decimal(1),
                    status=OrderStatus.FILLED,
                    timestamp=1234567890,
                )
            ]

            orders = await executor.execute_order(signal)

            # Should execute first part immediately
            assert len(orders) == 1
            assert mock_execute.call_count == 1

            # Should have created a session
            assert len(executor.active_sessions) == 1

    @pytest.mark.asyncio
    async def test_continue_dca_session(self):
        """Test continuing a DCA session."""
        config = {"interval_minutes": 1, "parts": 3}  # Short for testing
        executor = DCAExecutor(config)

        # Create a mock session
        session_id = "test_session"
        executor.active_sessions[session_id] = {
            "session_id": session_id,
            "symbol": "BTC/USDT",
            "total_amount": Decimal(3),
            "remaining_amount": Decimal(3),
            "parts": 3,
            "executed_parts": 0,
            "split_amounts": [Decimal(1), Decimal(1), Decimal(1)],
            "start_time": time.time() - 70,  # Started 70 seconds ago
            "next_execution": time.time() - 10,  # Should execute now
            "executed_orders": [],
            "parent_signal": TradingSignal(
                strategy_id="test",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.WEAK,
                order_type=OrderType.MARKET,
                amount=Decimal(3),
                timestamp=1234567890,
            ),
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
            metadata={"dca_session_id": session_id},
        )

        # Mock the single order execution
        with patch.object(
            executor, "_execute_single_order", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = [
                Order(
                    id="test_order",
                    symbol="BTC/USDT",
                    type=OrderType.MARKET,
                    side="buy",
                    amount=Decimal(1),
                    status=OrderStatus.FILLED,
                    timestamp=1234567890,
                )
            ]

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
            "session_id": session_id,
            "symbol": "BTC/USDT",
            "total_amount": Decimal(3),
            "remaining_amount": Decimal(2),
            "parts": 3,
            "executed_parts": 1,
            "split_amounts": [Decimal(1), Decimal(1), Decimal(1)],
            "start_time": time.time(),
            "next_execution": time.time() + 3600,
            "executed_orders": ["order1"],
            "parent_signal": TradingSignal(
                strategy_id="test",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.WEAK,
                order_type=OrderType.MARKET,
                amount=Decimal(3),
                timestamp=1234567890,
            ),
        }

        sessions = executor.get_dca_sessions()

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == session_id
        assert sessions[0]["symbol"] == "BTC/USDT"
        assert sessions[0]["executed_parts"] == 1
        assert sessions[0]["progress"] == 1 / 3

    def test_get_session_status(self):
        """Test getting detailed session status."""
        config = {"interval_minutes": 60, "parts": 3}
        executor = DCAExecutor(config)

        session_id = "test_session"
        executor.active_sessions[session_id] = {
            "session_id": session_id,
            "symbol": "BTC/USDT",
            "total_amount": Decimal(3),
            "remaining_amount": Decimal(2),
            "parts": 3,
            "executed_parts": 1,
            "split_amounts": [Decimal(1), Decimal(1), Decimal(1)],
            "start_time": time.time(),
            "next_execution": time.time() + 3600,
            "executed_orders": ["order1"],
            "parent_signal": TradingSignal(
                strategy_id="test",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.WEAK,
                order_type=OrderType.MARKET,
                amount=Decimal(3),
                timestamp=1234567890,
            ),
        }

        status = executor.get_session_status(session_id)

        assert status is not None
        assert status["session_id"] == session_id
        assert status["progress_percentage"] == 33.33333333333333  # 1/3
        assert status["seconds_until_next"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
