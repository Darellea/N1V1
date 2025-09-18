import pytest
import asyncio
from decimal import Decimal
from unittest.mock import MagicMock
from core.execution.backtest_executor import BacktestOrderExecutor
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types.order_types import OrderType, OrderStatus


class TestBacktestOrderExecutor:
    """Test cases for BacktestOrderExecutor functionality."""

    @pytest.fixture
    def config(self):
        """Basic config for backtest executor."""
        return {
            "base_currency": "USDT",
            "trade_fee": "0.001",  # 0.1% fee
        }

    @pytest.fixture
    def backtest_executor(self, config):
        """Create a fresh BacktestOrderExecutor instance."""
        return BacktestOrderExecutor(config)

    @pytest.fixture
    def sample_signal(self):
        """Create a sample trading signal for testing."""
        return TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=1234567890000,
        )

    def test_initialization(self, config, backtest_executor):
        """Test BacktestOrderExecutor initialization."""
        assert backtest_executor.config == config
        assert backtest_executor.trade_count == 0

    @pytest.mark.asyncio
    async def test_execute_backtest_order_buy_signal(self, backtest_executor, sample_signal):
        """Test executing a buy order in backtest mode."""
        result = await backtest_executor.execute_backtest_order(sample_signal)

        assert result.id == "backtest_0"
        assert result.symbol == "BTC/USDT"
        assert result.type == OrderType.MARKET
        assert result.side == "buy"
        assert result.amount == Decimal("1.0")
        assert result.price == Decimal("50000")
        assert result.status == OrderStatus.FILLED
        assert result.filled == Decimal("1.0")
        assert result.remaining == Decimal("0")
        assert result.cost == Decimal("50000")  # 1.0 * 50000
        assert result.timestamp == 1234567890000
        assert backtest_executor.trade_count == 1

    @pytest.mark.asyncio
    async def test_execute_backtest_order_sell_signal(self, backtest_executor):
        """Test executing a sell order in backtest mode."""
        sell_signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="ETH/USDT",
            signal_type=SignalType.ENTRY_SHORT,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("2.0"),
            price=Decimal("3000"),
            timestamp=1234567900000,
        )

        result = await backtest_executor.execute_backtest_order(sell_signal)

        assert result.id == "backtest_0"
        assert result.symbol == "ETH/USDT"
        assert result.side == "sell"
        assert result.amount == Decimal("2.0")
        assert result.price == Decimal("3000")
        assert result.cost == Decimal("6000")  # 2.0 * 3000
        assert backtest_executor.trade_count == 1

    @pytest.mark.asyncio
    async def test_execute_backtest_order_with_explicit_side(self, backtest_executor):
        """Test executing order with explicit side attribute."""
        signal_with_side = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("0.5"),
            price=Decimal("40000"),
            timestamp=1234567890000,
        )
        # Set explicit side
        signal_with_side.side = "sell"

        result = await backtest_executor.execute_backtest_order(signal_with_side)

        assert result.side == "sell"
        assert result.amount == Decimal("0.5")
        assert result.price == Decimal("40000")
        assert result.cost == Decimal("20000")  # 0.5 * 40000

    @pytest.mark.asyncio
    async def test_execute_backtest_order_with_fee_calculation(self, backtest_executor):
        """Test that fees are calculated and included in the order."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=1234567890000,
        )

        result = await backtest_executor.execute_backtest_order(signal)

        # Fee should be 1.0 * 0.001 = 0.001
        assert result.fee == {"cost": 0.001, "currency": "USDT"}

    @pytest.mark.asyncio
    async def test_execute_backtest_order_with_params(self, backtest_executor):
        """Test executing order with additional parameters."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=1234567890000,
        )
        signal.params = {"test_param": "value", "stop_loss": "49000"}

        result = await backtest_executor.execute_backtest_order(signal)

        assert result.params == {"test_param": "value", "stop_loss": "49000"}

    @pytest.mark.asyncio
    async def test_execute_backtest_order_with_trailing_stop(self, backtest_executor):
        """Test executing order with trailing stop configuration."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=1234567890000,
        )
        signal.trailing_stop = {"price": "49500"}

        result = await backtest_executor.execute_backtest_order(signal)

        assert result.trailing_stop == Decimal("49500")

    @pytest.mark.asyncio
    async def test_execute_backtest_order_with_invalid_trailing_stop(self, backtest_executor):
        """Test executing order with invalid trailing stop (should be None)."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=1234567890000,
        )
        signal.trailing_stop = {"invalid": "structure"}  # Missing 'price' key

        result = await backtest_executor.execute_backtest_order(signal)

        assert result.trailing_stop is None

    @pytest.mark.asyncio
    async def test_execute_backtest_order_multiple_orders(self, backtest_executor):
        """Test executing multiple orders and trade count increment."""
        signal1 = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=1234567890000,
        )

        signal2 = TradingSignal(
            strategy_id="test_strategy",
            symbol="ETH/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("2.0"),
            price=Decimal("3000"),
            timestamp=1234567900000,
        )

        # Execute first order
        result1 = await backtest_executor.execute_backtest_order(signal1)
        assert result1.id == "backtest_0"
        assert backtest_executor.trade_count == 1

        # Execute second order
        result2 = await backtest_executor.execute_backtest_order(signal2)
        assert result2.id == "backtest_1"
        assert backtest_executor.trade_count == 2

    @pytest.mark.asyncio
    async def test_execute_backtest_order_different_order_types(self, backtest_executor):
        """Test executing orders with different order types."""
        limit_signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=1234567890000,
        )

        result = await backtest_executor.execute_backtest_order(limit_signal)

        assert result.type == OrderType.LIMIT
        assert result.price == Decimal("50000")

    def test_calculate_fee_with_signal_object(self, backtest_executor):
        """Test fee calculation with signal object."""
        signal = MagicMock()
        signal.amount = Decimal("1.0")

        fee = backtest_executor._calculate_fee(signal)

        # 1.0 * 0.001 = 0.001
        assert fee == Decimal("0.001")

    def test_calculate_fee_with_dict_signal(self, backtest_executor):
        """Test fee calculation with dict signal."""
        signal = {"amount": Decimal("2.0")}

        fee = backtest_executor._calculate_fee(signal)

        # 2.0 * 0.001 = 0.002
        assert fee == Decimal("0.002")

    def test_calculate_fee_with_missing_amount(self, backtest_executor):
        """Test fee calculation when amount is missing."""
        signal = MagicMock()
        # Remove the amount attribute entirely so getattr returns the default
        del signal.amount
        signal.get = MagicMock(return_value=None)

        fee = backtest_executor._calculate_fee(signal)

        # Should default to 0
        assert fee == Decimal("0")

    def test_calculate_fee_with_zero_amount(self, backtest_executor):
        """Test fee calculation with zero amount."""
        signal = MagicMock()
        signal.amount = Decimal("0")

        fee = backtest_executor._calculate_fee(signal)

        assert fee == Decimal("0")

    def test_calculate_fee_with_large_amount(self, backtest_executor):
        """Test fee calculation with large amount."""
        signal = MagicMock()
        signal.amount = Decimal("1000.0")

        fee = backtest_executor._calculate_fee(signal)

        # 1000.0 * 0.001 = 1.0
        assert fee == Decimal("1.0")

    def test_calculate_fee_with_different_fee_rate(self):
        """Test fee calculation with different fee rate in config."""
        config = {
            "base_currency": "USDT",
            "trade_fee": "0.002",  # 0.2% fee
        }
        executor = BacktestOrderExecutor(config)

        signal = MagicMock()
        signal.amount = Decimal("1.0")

        fee = executor._calculate_fee(signal)

        # 1.0 * 0.002 = 0.002
        assert fee == Decimal("0.002")

    @pytest.mark.asyncio
    async def test_execute_backtest_order_fallback_signal_type(self, backtest_executor):
        """Test fallback to buy side when signal type is not ENTRY_LONG or ENTRY_SHORT."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type="unknown_type",  # Not ENTRY_LONG or ENTRY_SHORT
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=1234567890000,
        )

        result = await backtest_executor.execute_backtest_order(signal)

        # Should default to "buy" side
        assert result.side == "buy"

    @pytest.mark.asyncio
    async def test_execute_backtest_order_with_none_params(self, backtest_executor):
        """Test executing order when signal has None params."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=1234567890000,
        )
        signal.params = None
        # Set stop_loss to test the fallback behavior
        signal.stop_loss = "49000"

        result = await backtest_executor.execute_backtest_order(signal)

        # When params is None, it falls back to {"stop_loss": signal.stop_loss}
        assert result.params == {"stop_loss": "49000"}

    @pytest.mark.asyncio
    async def test_execute_backtest_order_with_none_trailing_stop(self, backtest_executor):
        """Test executing order when signal has None trailing_stop."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=1234567890000,
        )
        signal.trailing_stop = None

        result = await backtest_executor.execute_backtest_order(signal)

        assert result.trailing_stop is None

    def test_backtest_executor_trade_count_persistence(self, backtest_executor):
        """Test that trade count persists across operations."""
        initial_count = backtest_executor.trade_count
        assert initial_count == 0

        # Simulate incrementing trade count (normally done in execute_backtest_order)
        backtest_executor.trade_count += 1
        assert backtest_executor.trade_count == 1

        backtest_executor.trade_count += 1
        assert backtest_executor.trade_count == 2

    def test_config_immutability(self, config, backtest_executor):
        """Test that the config is properly stored."""
        assert backtest_executor.config is config

        # Verify config contains expected keys
        assert "base_currency" in backtest_executor.config
        assert "trade_fee" in backtest_executor.config
        assert backtest_executor.config["base_currency"] == "USDT"
        assert backtest_executor.config["trade_fee"] == "0.001"
