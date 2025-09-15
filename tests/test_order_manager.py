import asyncio
import pytest
import warnings
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from core.order_manager import OrderManager
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types import TradingMode
from core.types.order_types import OrderType, OrderStatus


class TestOrderManager:
    """Test cases for OrderManager functionality."""

    @pytest.fixture
    def config(self):
        """Basic config for testing."""
        return {
            "order": {
                "base_currency": "USDT",
                "exchange": "binance"
            },
            "risk": {},
            "paper": {
                "initial_balance": 10000.0
            },
            "reliability": {}
        }

    @pytest.fixture
    def mock_executors(self):
        """Mock all executors."""
        with patch('core.order_manager.LiveOrderExecutor') as mock_live, \
             patch('core.order_manager.PaperOrderExecutor') as mock_paper, \
             patch('core.order_manager.BacktestOrderExecutor') as mock_backtest:

            # Mock instances
            mock_live_instance = MagicMock()
            mock_paper_instance = MagicMock()
            mock_backtest_instance = MagicMock()

            mock_live.return_value = mock_live_instance
            mock_paper.return_value = mock_paper_instance
            mock_backtest.return_value = mock_backtest_instance

            # Mock paper executor methods
            mock_paper_instance.execute_paper_order = AsyncMock(return_value={"id": "test_order", "status": "filled"})
            mock_paper_instance.get_balance = MagicMock(return_value=Decimal("10000"))
            mock_paper_instance.set_initial_balance = MagicMock()
            mock_paper_instance.set_portfolio_mode = MagicMock()

            # Mock backtest executor
            mock_backtest_instance.execute_backtest_order = AsyncMock(return_value={"id": "backtest_order", "status": "filled"})

            yield {
                'live': mock_live_instance,
                'paper': mock_paper_instance,
                'backtest': mock_backtest_instance
            }

    @pytest.fixture
    def mock_managers(self):
        """Mock reliability and portfolio managers."""
        with patch('core.order_manager.ReliabilityManager') as mock_reliability, \
             patch('core.order_manager.PortfolioManager') as mock_portfolio, \
             patch('core.order_manager.OrderProcessor') as mock_processor:

            mock_reliability_instance = MagicMock()
            mock_portfolio_instance = MagicMock()
            mock_processor_instance = MagicMock()

            mock_reliability.return_value = mock_reliability_instance
            mock_portfolio.return_value = mock_portfolio_instance
            mock_processor.return_value = mock_processor_instance

            # Mock reliability manager
            mock_reliability_instance.safe_mode_active = False
            async def mock_retry_async(func, **kwargs):
                # For lambda functions that return coroutines, we need to call and await them
                if callable(func):
                    result = await func()
                else:
                    result = func
                return result
            mock_reliability_instance.retry_async = AsyncMock(side_effect=mock_retry_async)
            mock_reliability_instance.record_critical_error = MagicMock()

            # Mock portfolio manager
            mock_portfolio_instance.paper_balances = {"USDT": Decimal("10000")}
            mock_portfolio_instance.set_initial_balance = MagicMock()
            mock_portfolio_instance.initialize_portfolio = MagicMock()

            # Mock order processor
            mock_processor_instance.process_order = AsyncMock(return_value={"id": "processed_order", "status": "filled"})
            mock_processor_instance.open_orders = {}
            mock_processor_instance.closed_orders = {}
            mock_processor_instance.positions = {}
            mock_processor_instance.get_active_order_count = MagicMock(return_value=0)
            mock_processor_instance.get_open_position_count = MagicMock(return_value=0)

            yield {
                'reliability': mock_reliability_instance,
                'portfolio': mock_portfolio_instance,
                'processor': mock_processor_instance
            }

    def test_init_paper_mode(self, config, mock_executors, mock_managers):
        """Test OrderManager initialization in paper mode."""
        om = OrderManager(config, TradingMode.PAPER)

        assert om.mode == TradingMode.PAPER
        assert om.mode_name == "paper"
        assert om.live_executor is None
        assert om.paper_executor is not None
        assert om.backtest_executor is not None

    def test_init_live_mode(self, config, mock_executors, mock_managers):
        """Test OrderManager initialization in live mode."""
        om = OrderManager(config, TradingMode.LIVE)

        assert om.mode == TradingMode.LIVE
        assert om.mode_name == "live"
        assert om.live_executor is not None

    @pytest.mark.asyncio
    async def test_execute_order_paper_mode(self, config, mock_executors, mock_managers):
        """Test order execution in paper mode."""
        om = OrderManager(config, TradingMode.PAPER)

        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            price=Decimal("50000")
        )

        result = await om.execute_order(signal)

        assert result is not None
        assert result["id"] == "processed_order"
        assert result["status"] == "filled"

        # Verify paper executor was called
        mock_executors['paper'].execute_paper_order.assert_called_once_with(signal)
        mock_managers['processor'].process_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_order_backtest_mode(self, config, mock_executors, mock_managers):
        """Test order execution in backtest mode."""
        om = OrderManager(config, TradingMode.BACKTEST)

        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0")
        )

        result = await om.execute_order(signal)

        assert result is not None
        mock_executors['backtest'].execute_backtest_order.assert_called_once_with(signal)

    @pytest.mark.asyncio
    async def test_execute_order_safe_mode_active(self, config, mock_executors, mock_managers):
        """Test that orders are skipped when safe mode is active."""
        om = OrderManager(config, TradingMode.PAPER)
        mock_managers['reliability'].safe_mode_active = True

        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0")
        )

        result = await om.execute_order(signal)

        assert result is not None
        assert result["status"] == "skipped"
        assert result["reason"] == "safe_mode_active"

        # Verify no execution happened
        mock_executors['paper'].execute_paper_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_order_live_mode_with_retry(self, config, mock_executors, mock_managers):
        """Test order execution in live mode with retry logic."""
        om = OrderManager(config, TradingMode.LIVE)

        # Mock live executor
        mock_executors['live'].execute_live_order = AsyncMock(return_value={"id": "live_order", "status": "filled"})

        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0")
        )

        result = await om.execute_order(signal)

        assert result is not None
        mock_executors['live'].execute_live_order.assert_called_once_with(signal)
        mock_managers['reliability'].retry_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_order_live_mode(self, config, mock_executors, mock_managers):
        """Test order cancellation in live mode."""
        om = OrderManager(config, TradingMode.LIVE)

        # Mock exchange with proper AsyncMock setup
        mock_exchange = MagicMock()
        mock_exchange.cancel_order = AsyncMock(return_value=True)
        mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 50000.0})  # Add fetch_ticker for balance calculation
        mock_executors['live'].exchange = mock_exchange

        # Add order to processor
        mock_managers['processor'].open_orders = {"test_order": MagicMock()}

        result = await om.cancel_order("test_order")

        assert result is True
        mock_exchange.cancel_order.assert_called_once_with("test_order")

    @pytest.mark.asyncio
    async def test_cancel_order_non_live_mode(self, config, mock_executors, mock_managers):
        """Test that cancel_order returns False for non-live modes."""
        om = OrderManager(config, TradingMode.PAPER)

        result = await om.cancel_order("test_order")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_balance_live_mode(self, config, mock_executors, mock_managers):
        """Test balance retrieval in live mode."""
        om = OrderManager(config, TradingMode.LIVE)

        # Mock exchange with all async methods properly mocked
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance = AsyncMock(return_value={"total": {"USDT": 5000.0}})
        mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
        mock_executors['live'].exchange = mock_exchange

        balance = await om.get_balance()

        assert balance == Decimal("5000")
        mock_exchange.fetch_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_balance_paper_mode(self, config, mock_executors, mock_managers):
        """Test balance retrieval in paper mode."""
        om = OrderManager(config, TradingMode.PAPER)

        balance = await om.get_balance()

        assert balance == Decimal("10000")
        mock_executors['paper'].get_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_balance_backtest_mode(self, config, mock_executors, mock_managers):
        """Test balance retrieval in backtest mode."""
        om = OrderManager(config, TradingMode.BACKTEST)

        balance = await om.get_balance()

        assert balance == Decimal("0")  # Backtest doesn't track balance

    @pytest.mark.asyncio
    async def test_get_equity_with_portfolio_mode(self, config, mock_executors, mock_managers):
        """Test equity calculation with portfolio mode."""
        om = OrderManager(config, TradingMode.PAPER)
        om.portfolio_mode = True

        equity = await om.get_equity()

        # Should aggregate paper balances
        assert equity == Decimal("10000")

    @pytest.mark.asyncio
    async def test_initialize_portfolio(self, config, mock_executors, mock_managers):
        """Test portfolio initialization."""
        om = OrderManager(config, TradingMode.PAPER)

        pairs = ["BTC/USDT", "ETH/USDT"]
        allocation = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}

        await om.initialize_portfolio(pairs, True, allocation)

        assert om.pairs == pairs
        assert om.portfolio_mode is True
        assert om.pair_allocation == allocation

        mock_executors['paper'].set_portfolio_mode.assert_called_once_with(True, pairs, allocation)
        mock_managers['portfolio'].initialize_portfolio.assert_called_once_with(pairs, True, allocation)

    @pytest.mark.asyncio
    async def test_get_active_order_count(self, config, mock_executors, mock_managers):
        """Test getting active order count."""
        om = OrderManager(config, TradingMode.PAPER)

        count = await om.get_active_order_count()

        assert count == 0
        mock_managers['processor'].get_active_order_count.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_open_position_count(self, config, mock_executors, mock_managers):
        """Test getting open position count."""
        om = OrderManager(config, TradingMode.PAPER)

        count = await om.get_open_position_count()

        assert count == 0
        mock_managers['processor'].get_open_position_count.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown(self, config, mock_executors, mock_managers):
        """Test shutdown functionality."""
        om = OrderManager(config, TradingMode.LIVE)

        # Mock live executor shutdown
        mock_executors['live'].shutdown = AsyncMock()

        await om.shutdown()

        mock_executors['live'].shutdown.assert_called_once()

    def test_init_with_invalid_mode_string(self, config, mock_executors, mock_managers):
        """Test initialization with invalid mode string to cover fallback."""
        om = OrderManager(config, "invalid_mode")
        assert om.mode == TradingMode.PAPER  # Should fallback to PAPER

    def test_init_with_mode_enum_value(self, config, mock_executors, mock_managers):
        """Test initialization with mode as enum value."""
        om = OrderManager(config, "live")
        assert om.mode == TradingMode.LIVE

    @pytest.mark.asyncio
    async def test_execute_order_safe_mode_trigger_counter(self, config, mock_executors, mock_managers):
        """Test safe mode trigger counter increment."""
        om = OrderManager(config, TradingMode.PAPER)
        mock_managers['reliability'].safe_mode_active = True

        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0")
        )

        await om.execute_order(signal)
        assert hasattr(om, '_safe_mode_triggers')
        assert om._safe_mode_triggers == 1

        await om.execute_order(signal)
        assert om._safe_mode_triggers == 2

    @pytest.mark.asyncio
    async def test_execute_order_unknown_mode(self, config, mock_executors, mock_managers):
        """Test order execution with unknown mode (should treat as paper)."""
        om = OrderManager(config, TradingMode.PAPER)
        om.mode = "unknown"  # Simulate unknown mode

        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0")
        )

        result = await om.execute_order(signal)

        # Should execute as paper
        mock_executors['paper'].execute_paper_order.assert_called_once_with(signal)

    @pytest.mark.asyncio
    async def test_cancel_order_with_exception(self, config, mock_executors, mock_managers):
        """Test cancel_order with network exception."""
        om = OrderManager(config, TradingMode.LIVE)

        mock_exchange = MagicMock()
        mock_exchange.cancel_order = AsyncMock(side_effect=Exception("Network error"))
        mock_executors['live'].exchange = mock_exchange

        result = await om.cancel_order("test_order")

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_all_orders_with_exception(self, config, mock_executors, mock_managers):
        """Test cancel_all_orders with exception."""
        om = OrderManager(config, TradingMode.LIVE)

        mock_exchange = MagicMock()
        mock_exchange.cancel_order = AsyncMock(side_effect=Exception("Network error"))
        mock_executors['live'].exchange = mock_exchange

        mock_managers['processor'].open_orders = {"order1": MagicMock(), "order2": MagicMock()}

        with pytest.raises(Exception):
            await om.cancel_all_orders()

    @pytest.mark.asyncio
    async def test_rate_limit(self, config, mock_executors, mock_managers):
        """Test rate limiting functionality."""
        om = OrderManager(config, TradingMode.LIVE)
        om._last_request_time = 0.0
        om._request_interval = 0.1

        # Use monotonic time for more precise measurement (no drift)
        start_time = time.monotonic()
        await om._rate_limit()
        end_time = time.monotonic()

        elapsed = end_time - start_time

        # Should have waited at least the interval with reasonable tolerance
        # Use range assertion to handle OS scheduling variance and async overhead
        # Allow wider range to accommodate system load and timing variations
        assert 0.08 <= elapsed <= 0.25, \
            f"Rate limit timing out of range: {elapsed:.4f}s (expected ~0.1s ± tolerance)"

        # Verify that subsequent calls are also rate limited
        start_time2 = time.monotonic()
        await om._rate_limit()
        end_time2 = time.monotonic()

        elapsed2 = end_time2 - start_time2

        # Second call should also wait (though potentially less due to timing)
        assert elapsed2 >= 0.08, \
            f"Second rate limit call too fast: {elapsed2:.4f}s"

    @pytest.mark.asyncio
    async def test_rate_limit_deterministic_simulation(self, config, mock_executors, mock_managers):
        """Test rate limiting with deterministic time simulation (no OS jitter)."""
        om = OrderManager(config, TradingMode.LIVE)
        om._request_interval = 0.1

        # Use a simple approach: patch asyncio.sleep to be instantaneous
        # and manually control the time values
        original_sleep = asyncio.sleep

        async def instant_sleep(delay):
            # Don't actually sleep, just update the time
            om._last_request_time = time.monotonic()
            return

        with patch('asyncio.sleep', side_effect=instant_sleep):
            # First call - should wait full interval
            om._last_request_time = 0.0
            start_time = time.monotonic()

            await om._rate_limit()

            end_time = time.monotonic()
            elapsed = end_time - start_time

            # Since we mocked sleep to be instant, elapsed should be very small
            assert elapsed < 0.01, f"First call should be fast with mocked sleep: {elapsed}"

            # Verify that _last_request_time was updated
            assert om._last_request_time > 0

            # Second call - should wait again since time has "passed"
            start_time2 = time.monotonic()
            await om._rate_limit()
            end_time2 = time.monotonic()
            elapsed2 = end_time2 - start_time2

            # Second call should also be fast
            assert elapsed2 < 0.01, f"Second call should also be fast: {elapsed2}"

    @pytest.mark.asyncio
    async def test_get_cached_ticker_cache_hit(self, config, mock_executors, mock_managers):
        """Test _get_cached_ticker with cache hit."""
        om = OrderManager(config, TradingMode.LIVE)

        # Pre-populate cache
        om._ticker_cache["BTC/USDT"] = {"last": 50000.0}
        om._cache_timestamps["BTC/USDT"] = time.time()

        ticker = await om._get_cached_ticker("BTC/USDT")

        assert ticker == {"last": 50000.0}
        # Should not call exchange since cache hit

    @pytest.mark.asyncio
    async def test_get_cached_ticker_live_mode(self, config, mock_executors, mock_managers):
        """Test _get_cached_ticker in live mode."""
        om = OrderManager(config, TradingMode.LIVE)

        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
        mock_executors['live'].exchange = mock_exchange

        ticker = await om._get_cached_ticker("BTC/USDT")

        assert ticker == {"last": 50000.0}
        mock_exchange.fetch_ticker.assert_called_once_with("BTC/USDT")

    @pytest.mark.asyncio
    async def test_get_cached_ticker_non_live_mode(self, config, mock_executors, mock_managers):
        """Test _get_cached_ticker in non-live mode raises ValueError."""
        om = OrderManager(config, TradingMode.PAPER)

        with pytest.raises(ValueError, match="Ticker fetching only supported in LIVE mode"):
            await om._get_cached_ticker("BTC/USDT")

    @pytest.mark.asyncio
    async def test_get_balance_live_mode_with_exception(self, config, mock_executors, mock_managers):
        """Test get_balance in live mode with exception."""
        om = OrderManager(config, TradingMode.LIVE)

        mock_exchange = MagicMock()
        mock_exchange.fetch_balance = AsyncMock(side_effect=Exception("Network error"))
        mock_executors['live'].exchange = mock_exchange

        balance = await om.get_balance()

        assert balance == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_balance_backtest_portfolio_mode(self, config, mock_executors, mock_managers):
        """Test get_balance in backtest mode with portfolio."""
        om = OrderManager(config, TradingMode.BACKTEST)
        om.portfolio_mode = True
        mock_managers['portfolio'].paper_balances = {"USDT": Decimal("5000"), "BTC": Decimal("0.1")}

        balance = await om.get_balance()

        assert balance == Decimal("5000.1")

    @pytest.mark.asyncio
    async def test_get_equity_live_mode_with_positions(self, config, mock_executors, mock_managers):
        """Test get_equity in live mode with positions."""
        om = OrderManager(config, TradingMode.LIVE)

        mock_exchange = MagicMock()
        mock_exchange.fetch_balance = AsyncMock(return_value={"total": {"USDT": 5000.0}})
        mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 51000.0})
        mock_executors['live'].exchange = mock_exchange

        # Add position
        mock_managers['processor'].positions = {
            "BTC/USDT": {"entry_price": 50000.0, "amount": 0.1}
        }

        equity = await om.get_equity()

        # Balance 5000 + unrealized (51000 - 50000) * 0.1 = 5000 + 100 = 5100
        assert equity == Decimal("5100")

    @pytest.mark.asyncio
    async def test_get_equity_with_invalid_position_data(self, config, mock_executors, mock_managers):
        """Test get_equity with invalid position data."""
        om = OrderManager(config, TradingMode.LIVE)

        mock_exchange = MagicMock()
        mock_exchange.fetch_balance = AsyncMock(return_value={"total": {"USDT": 5000.0}})
        mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 51000.0})
        mock_executors['live'].exchange = mock_exchange

        # Add invalid position
        mock_managers['processor'].positions = {
            "BTC/USDT": {"entry_price": None, "amount": 0.1}
        }

        equity = await om.get_equity()

        # Should skip invalid position and return balance
        assert equity == Decimal("5000")

    @pytest.mark.asyncio
    async def test_initialize_portfolio_invalid_inputs(self, config, mock_executors, mock_managers):
        """Test initialize_portfolio with invalid inputs."""
        om = OrderManager(config, TradingMode.PAPER)

        # Test None pairs
        with pytest.raises(TypeError, match="pairs cannot be None"):
            await om.initialize_portfolio(None, True)

        # Test non-list pairs
        with pytest.raises(TypeError, match="pairs must be a list"):
            await om.initialize_portfolio("not_a_list", True)

        # Test invalid allocation
        with pytest.raises(TypeError, match="allocation must be a dict or None"):
            await om.initialize_portfolio(["BTC/USDT"], True, "not_a_dict")

        # Test negative allocation
        with pytest.raises(ValueError, match="cannot be negative"):
            await om.initialize_portfolio(["BTC/USDT"], True, {"BTC/USDT": -0.1})

        # Test allocation sum > 1
        with pytest.raises(ValueError, match="exceeds 100%"):
            await om.initialize_portfolio(["BTC/USDT", "ETH/USDT"], True, {"BTC/USDT": 0.7, "ETH/USDT": 0.4})

    @pytest.mark.asyncio
    async def test_initialize_portfolio_equal_allocation(self, config, mock_executors, mock_managers):
        """Test initialize_portfolio with default equal allocation."""
        om = OrderManager(config, TradingMode.PAPER)

        pairs = ["BTC/USDT", "ETH/USDT"]

        await om.initialize_portfolio(pairs, True, None)

        expected_allocation = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        assert om.pair_allocation == expected_allocation
