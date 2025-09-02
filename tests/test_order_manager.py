import asyncio
import pytest
import warnings
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
            mock_reliability_instance.retry_async = AsyncMock(side_effect=lambda func, **kwargs: func())
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
