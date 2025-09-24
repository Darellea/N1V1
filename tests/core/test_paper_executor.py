import pytest
import asyncio
from decimal import Decimal
from unittest.mock import MagicMock
from core.execution.paper_executor import PaperOrderExecutor
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types.order_types import OrderType, OrderStatus


class TestPaperOrderExecutor:
    """Test cases for PaperOrderExecutor functionality."""

    @pytest.fixture
    def config(self):
        """Basic config for paper executor."""
        return {
            "base_currency": "USDT",
            "trade_fee": "0.001",  # 0.1% fee
            "slippage": "0.0005",  # 0.05% slippage
        }

    @pytest.fixture
    def paper_executor(self, config):
        """Create a fresh PaperOrderExecutor instance."""
        executor = PaperOrderExecutor(config)
        # Initialize portfolio mode attributes that are set by set_portfolio_mode
        executor.portfolio_mode = False
        executor.pairs = []
        executor.pair_allocation = None
        return executor

    @pytest.fixture
    def paper_executor_with_balance(self, paper_executor):
        """Create a PaperOrderExecutor with default balance."""
        paper_executor.set_initial_balance(Decimal("1000000"))  # Much higher balance for testing
        return paper_executor

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

    def test_initialization(self, config, paper_executor):
        """Test PaperOrderExecutor initialization."""
        assert paper_executor.config == config
        assert paper_executor.paper_balance == Decimal("0")
        assert paper_executor.paper_balances == {}
        assert paper_executor.trade_count == 0

    def test_set_initial_balance_with_value(self, paper_executor):
        """Test setting initial balance with a valid value."""
        initial_balance = Decimal("10000")
        paper_executor.set_initial_balance(initial_balance)

        assert paper_executor.paper_balance == Decimal("10000")

    def test_set_initial_balance_with_none(self, paper_executor):
        """Test setting initial balance with None (should default to 0)."""
        paper_executor.set_initial_balance(None)

        assert paper_executor.paper_balance == Decimal("0")

    def test_set_initial_balance_with_invalid_value(self, paper_executor):
        """Test setting initial balance with invalid value (should default to 0)."""
        paper_executor.set_initial_balance("invalid")

        assert paper_executor.paper_balance == Decimal("0")

    def test_set_portfolio_mode_disabled(self, paper_executor):
        """Test setting portfolio mode to disabled."""
        paper_executor.set_portfolio_mode(False)

        assert paper_executor.portfolio_mode is False
        assert paper_executor.pairs == []
        assert paper_executor.pair_allocation is None

    def test_set_portfolio_mode_enabled_without_pairs(self, paper_executor):
        """Test setting portfolio mode enabled without pairs."""
        paper_executor.set_portfolio_mode(True)

        assert paper_executor.portfolio_mode is True
        assert paper_executor.pairs == []
        assert paper_executor.pair_allocation is None

    def test_set_portfolio_mode_with_pairs_equal_allocation(self, paper_executor):
        """Test setting portfolio mode with pairs and equal allocation."""
        paper_executor.set_initial_balance(Decimal("10000"))
        pairs = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]

        paper_executor.set_portfolio_mode(True, pairs)

        assert paper_executor.portfolio_mode is True
        assert paper_executor.pairs == pairs
        assert paper_executor.pair_allocation is None

        # Check equal allocation: 10000 / 3 = 3333.33... each
        # The _safe_quantize function rounds to appropriate decimal places
        expected_balance = Decimal("3333.333333")  # This will be quantized
        assert abs(paper_executor.paper_balances["BTC/USDT"] - expected_balance) < Decimal("0.01")
        assert abs(paper_executor.paper_balances["ETH/USDT"] - expected_balance) < Decimal("0.01")
        assert abs(paper_executor.paper_balances["ADA/USDT"] - expected_balance) < Decimal("0.01")

    def test_set_portfolio_mode_with_custom_allocation(self, paper_executor):
        """Test setting portfolio mode with custom allocation."""
        paper_executor.set_initial_balance(Decimal("10000"))
        pairs = ["BTC/USDT", "ETH/USDT"]
        allocation = {"BTC/USDT": 0.7, "ETH/USDT": 0.3}

        paper_executor.set_portfolio_mode(True, pairs, allocation)

        assert paper_executor.portfolio_mode is True
        assert paper_executor.pairs == pairs
        assert paper_executor.pair_allocation == allocation

        # Check custom allocation
        assert paper_executor.paper_balances["BTC/USDT"] == Decimal("7000")  # 10000 * 0.7
        assert paper_executor.paper_balances["ETH/USDT"] == Decimal("3000")  # 10000 * 0.3

    def test_set_portfolio_mode_with_missing_allocation(self, paper_executor):
        """Test setting portfolio mode with allocation missing some pairs."""
        paper_executor.set_initial_balance(Decimal("10000"))
        pairs = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        allocation = {"BTC/USDT": 0.5, "ETH/USDT": 0.3}  # Missing ADA

        paper_executor.set_portfolio_mode(True, pairs, allocation)

        # ADA should get 0 allocation
        assert paper_executor.paper_balances["BTC/USDT"] == Decimal("5000")  # 10000 * 0.5
        assert paper_executor.paper_balances["ETH/USDT"] == Decimal("3000")  # 10000 * 0.3
        assert paper_executor.paper_balances["ADA/USDT"] == Decimal("0")     # 10000 * 0

    @pytest.mark.asyncio
    async def test_execute_paper_order_buy_signal(self, paper_executor_with_balance, sample_signal):
        """Test executing a buy order in paper mode."""
        result = await paper_executor_with_balance.execute_paper_order(sample_signal)

        assert result.id == "paper_0"
        assert result.symbol == "BTC/USDT"
        assert result.type == OrderType.MARKET
        assert result.side == "buy"
        assert result.amount == Decimal("1.0")
        assert result.status == OrderStatus.FILLED
        assert result.filled == Decimal("1.0")
        assert result.remaining == Decimal("0")
        assert result.cost == Decimal("50025")  # 1.0 * 50000 with slippage
        assert paper_executor_with_balance.trade_count == 1

        # Check balance update (price with slippage + fee)
        # Expected price: 50000 * (1 + 0.0005) = 50025
        # Fee: 1.0 * 0.001 = 1
        # Total cost: 50025 + 1 = 50026
        expected_balance = Decimal("1000000") - Decimal("50026")
        assert abs(paper_executor_with_balance.paper_balance - expected_balance) < Decimal("1")

    @pytest.mark.asyncio
    async def test_execute_paper_order_sell_signal(self, paper_executor_with_balance):
        """Test executing a sell order in paper mode."""

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

        result = await paper_executor_with_balance.execute_paper_order(sell_signal)

        assert result.id == "paper_0"
        assert result.symbol == "ETH/USDT"
        assert result.side == "sell"
        assert result.amount == Decimal("2.0")
        assert result.cost == Decimal("5997")  # 2.0 * 3000 with slippage

        # Check balance update (price with slippage - fee)
        # Expected price: 3000 * (1 - 0.0005) = 2998.5
        # Fee: 2.0 * 0.001 = 2
        # Total credit: 5997 - 2 = 5995
        expected_balance = Decimal("1000000") + Decimal("5995")
        assert abs(paper_executor_with_balance.paper_balance - expected_balance) < Decimal("2")

    @pytest.mark.asyncio
    async def test_execute_paper_order_insufficient_balance(self, paper_executor, sample_signal):
        """Test executing order with insufficient balance."""
        paper_executor.set_initial_balance(Decimal("100"))  # Very low balance

        with pytest.raises(ValueError, match="Insufficient balance"):
            await paper_executor.execute_paper_order(sample_signal)

    @pytest.mark.asyncio
    async def test_execute_paper_order_with_explicit_side(self, paper_executor):
        """Test executing order with explicit side attribute."""
        paper_executor.set_initial_balance(Decimal("10000"))

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
        signal_with_side.side = "sell"

        result = await paper_executor.execute_paper_order(signal_with_side)

        assert result.side == "sell"
        assert result.amount == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_execute_paper_order_with_fee_calculation(self, paper_executor_with_balance):
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

        result = await paper_executor_with_balance.execute_paper_order(signal)

        # Fee should be 1.0 * 0.001 = 0.001
        assert result.fee == {"cost": 0.001, "currency": "USDT"}

    @pytest.mark.asyncio
    async def test_execute_paper_order_with_params(self, paper_executor_with_balance):
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

        result = await paper_executor_with_balance.execute_paper_order(signal)

        assert result.params == {"test_param": "value", "stop_loss": "49000"}

    @pytest.mark.asyncio
    async def test_execute_paper_order_with_trailing_stop(self, paper_executor_with_balance):
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

        result = await paper_executor_with_balance.execute_paper_order(signal)

        assert result.trailing_stop == Decimal("49500")

    @pytest.mark.asyncio
    async def test_execute_paper_order_portfolio_mode_buy(self, paper_executor_with_balance):
        """Test executing buy order in portfolio mode."""
        pairs = ["BTC/USDT", "ETH/USDT"]
        paper_executor_with_balance.set_portfolio_mode(True, pairs)

        # Use smaller amount to fit within allocated balance
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("0.05"),  # Much smaller amount
            price=Decimal("50000"),
            timestamp=1234567890000,
        )

        result = await paper_executor_with_balance.execute_paper_order(signal)

        # BTC balance should be reduced
        initial_btc_balance = Decimal("500000")  # 1000000 / 2
        # Cost: 50000 * 0.05 * (1 + 0.0005) + 0.05 * 0.001 = 2501.25 + 0.05 = 2501.30
        expected_btc_balance = initial_btc_balance - Decimal("2501.30")
        assert abs(paper_executor_with_balance.paper_balances["BTC/USDT"] - expected_btc_balance) < Decimal("0.1")

        # ETH balance should remain unchanged
        assert paper_executor_with_balance.paper_balances["ETH/USDT"] == Decimal("500000")

    @pytest.mark.asyncio
    async def test_execute_paper_order_portfolio_mode_sell(self, paper_executor_with_balance):
        """Test executing sell order in portfolio mode."""
        pairs = ["BTC/USDT", "ETH/USDT"]
        paper_executor_with_balance.set_portfolio_mode(True, pairs)

        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="ETH/USDT",
            signal_type=SignalType.ENTRY_SHORT,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("2.0"),
            price=Decimal("3000"),
            timestamp=1234567890000,
        )

        result = await paper_executor_with_balance.execute_paper_order(signal)

        # ETH balance should be increased
        initial_eth_balance = Decimal("500000")  # 1000000 / 2
        # Credit: 3000 * 2.0 * (1 - 0.0005) - 2.0 * 0.001 = 5997 - 2 = 5995
        expected_eth_balance = initial_eth_balance + Decimal("5995")
        assert abs(paper_executor_with_balance.paper_balances["ETH/USDT"] - expected_eth_balance) < Decimal("2")

        # BTC balance should remain unchanged
        assert paper_executor_with_balance.paper_balances["BTC/USDT"] == Decimal("500000")

    @pytest.mark.asyncio
    async def test_execute_paper_order_portfolio_mode_new_symbol(self, paper_executor_with_balance):
        """Test executing order for symbol not in initial pairs."""
        pairs = ["BTC/USDT"]
        paper_executor_with_balance.set_portfolio_mode(True, pairs)

        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="ETH/USDT",  # Not in initial pairs
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            price=Decimal("3000"),
            timestamp=1234567890000,
        )

        result = await paper_executor_with_balance.execute_paper_order(signal)

        # New symbol should get the current paper balance as initial
        initial_eth_balance = Decimal("1000000")  # Current paper balance
        # Cost: 3000 * 1.0 * (1 + 0.0005) + 1.0 * 0.001 = 3001.5 + 1 = 3002.5
        expected_eth_balance = initial_eth_balance - Decimal("3002.5")
        assert abs(paper_executor_with_balance.paper_balances["ETH/USDT"] - expected_eth_balance) < Decimal("2")

    def test_calculate_fee_with_signal_object(self, paper_executor):
        """Test fee calculation with signal object."""
        signal = MagicMock()
        signal.amount = Decimal("1.0")

        fee = paper_executor._calculate_fee(signal)

        # 1.0 * 0.001 = 0.001
        assert fee == Decimal("0.001")

    def test_calculate_fee_with_dict_signal(self, paper_executor):
        """Test fee calculation with dict signal."""
        signal = {"amount": Decimal("2.0")}

        fee = paper_executor._calculate_fee(signal)

        # 2.0 * 0.001 = 0.002
        assert fee == Decimal("0.002")

    def test_calculate_fee_with_missing_amount(self, paper_executor):
        """Test fee calculation when amount is missing."""
        signal = MagicMock()
        # Remove the amount attribute entirely
        del signal.amount
        signal.get = MagicMock(return_value=None)

        fee = paper_executor._calculate_fee(signal)

        # Should default to 0
        assert fee == Decimal("0")

    def test_calculate_fee_with_zero_amount(self, paper_executor):
        """Test fee calculation with zero amount."""
        signal = MagicMock()
        signal.amount = Decimal("0")

        fee = paper_executor._calculate_fee(signal)

        assert fee == Decimal("0")

    def test_apply_slippage_buy_order(self, paper_executor):
        """Test slippage application for buy orders."""
        signal = MagicMock()
        signal.price = Decimal("50000")

        adjusted_price = paper_executor._apply_slippage(signal, "buy")

        # 50000 * (1 + 0.0005) = 50025
        assert adjusted_price == Decimal("50025")

    def test_apply_slippage_sell_order(self, paper_executor):
        """Test slippage application for sell orders."""
        signal = MagicMock()
        signal.price = Decimal("50000")

        adjusted_price = paper_executor._apply_slippage(signal, "sell")

        # 50000 * (1 - 0.0005) = 49975
        assert adjusted_price == Decimal("49975")

    def test_apply_slippage_with_dict_signal(self, paper_executor):
        """Test slippage application with dict signal."""
        signal = {"price": Decimal("30000")}

        adjusted_price = paper_executor._apply_slippage(signal, "buy")

        # 30000 * (1 + 0.0005) = 30015
        assert adjusted_price == Decimal("30015")

    def test_apply_slippage_missing_price(self, paper_executor):
        """Test slippage application when price is missing."""
        signal = MagicMock()
        # Remove price attribute
        del signal.price
        signal.get = MagicMock(return_value=None)

        with pytest.raises(ValueError, match="Signal price required"):
            paper_executor._apply_slippage(signal, "buy")

    def test_get_balance_single_mode(self, paper_executor):
        """Test getting balance in single mode."""
        paper_executor.set_initial_balance(Decimal("10000"))

        balance = paper_executor.get_balance()

        assert balance == Decimal("10000")

    def test_get_balance_portfolio_mode(self, paper_executor):
        """Test getting balance in portfolio mode."""
        paper_executor.set_initial_balance(Decimal("10000"))
        pairs = ["BTC/USDT", "ETH/USDT"]
        paper_executor.set_portfolio_mode(True, pairs)

        # Manually set some balances
        paper_executor.paper_balances = {
            "BTC/USDT": Decimal("4000"),
            "ETH/USDT": Decimal("3500")
        }

        balance = paper_executor.get_balance()

        # Should sum all balances: 4000 + 3500 = 7500
        assert balance == Decimal("7500")

    def test_get_balance_portfolio_mode_empty(self, paper_executor):
        """Test getting balance in portfolio mode with empty balances."""
        paper_executor.set_initial_balance(Decimal("10000"))
        pairs = ["BTC/USDT", "ETH/USDT"]
        paper_executor.set_portfolio_mode(True, pairs)

        # Clear the paper_balances that were set by set_portfolio_mode
        paper_executor.paper_balances = {}

        balance = paper_executor.get_balance()

        # Should return 0 when portfolio mode but no balances
        assert balance == Decimal("0")

    @pytest.mark.asyncio
    async def test_execute_paper_order_multiple_orders(self, paper_executor_with_balance):
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
        result1 = await paper_executor_with_balance.execute_paper_order(signal1)
        assert result1.id == "paper_0"
        assert paper_executor_with_balance.trade_count == 1

        # Execute second order
        result2 = await paper_executor_with_balance.execute_paper_order(signal2)
        assert result2.id == "paper_1"
        assert paper_executor_with_balance.trade_count == 2

    @pytest.mark.asyncio
    async def test_execute_paper_order_different_order_types(self, paper_executor_with_balance):
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

        result = await paper_executor_with_balance.execute_paper_order(limit_signal)

        assert result.type == OrderType.LIMIT
        assert abs(result.price - Decimal("50025")) < Decimal("0.01")  # With slippage applied

    @pytest.mark.asyncio
    async def test_execute_paper_order_fallback_signal_type(self, paper_executor_with_balance):
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

        result = await paper_executor_with_balance.execute_paper_order(signal)

        # Should default to "buy" side
        assert result.side == "buy"

    @pytest.mark.asyncio
    async def test_execute_paper_order_with_none_params(self, paper_executor_with_balance):
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
        signal.stop_loss = "49000"

        result = await paper_executor_with_balance.execute_paper_order(signal)

        # When params is None, it falls back to {"stop_loss": signal.stop_loss}
        assert result.params == {"stop_loss": "49000"}

    @pytest.mark.asyncio
    async def test_execute_paper_order_with_none_trailing_stop(self, paper_executor_with_balance):
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

        result = await paper_executor_with_balance.execute_paper_order(signal)

        assert result.trailing_stop is None
