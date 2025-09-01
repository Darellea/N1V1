import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.bot_engine import BotEngine, BotState
from core.types import TradingMode


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "environment": {"mode": "paper"},
        "trading": {
            "portfolio_mode": False,
            "symbol": "BTC/USDT",
            "initial_balance": 1000.0
        },
        "exchange": {
            "markets": ["BTC/USDT"],
            "base_currency": "USDT"
        },
        "strategies": {
            "active_strategies": [],
            "strategy_config": {}
        },
        "notifications": {
            "discord": {
                "enabled": False
            }
        },
        "monitoring": {
            "terminal_display": False,
            "update_interval": 1.0
        },
        "risk_management": {},
        "backtesting": {"timeframe": "1h"}
    }


@pytest.fixture
def mock_data_fetcher():
    """Mock DataFetcher."""
    fetcher = MagicMock()
    fetcher.initialize = AsyncMock()
    fetcher.get_historical_data = AsyncMock(return_value=MagicMock())
    fetcher.shutdown = AsyncMock()
    return fetcher


@pytest.fixture
def mock_order_manager():
    """Mock OrderManager."""
    manager = MagicMock()
    manager.initialize_portfolio = AsyncMock()
    manager.get_balance = AsyncMock(return_value=1000.0)
    manager.get_equity = AsyncMock(return_value=1000.0)
    manager.get_active_order_count = AsyncMock(return_value=0)
    manager.get_open_position_count = AsyncMock(return_value=0)
    manager.cancel_all_orders = AsyncMock()
    manager.shutdown = AsyncMock()
    return manager


@pytest.fixture
def mock_risk_manager():
    """Mock RiskManager."""
    return MagicMock()


@pytest.fixture
def mock_signal_router():
    """Mock SignalRouter."""
    return MagicMock()


@pytest.fixture
def mock_notifier():
    """Mock DiscordNotifier."""
    notifier = MagicMock()
    notifier.initialize = AsyncMock()
    notifier.send_alert = AsyncMock()
    notifier.shutdown = AsyncMock()
    return notifier


class TestBotEngine:
    """Test cases for BotEngine functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_config, mock_data_fetcher, mock_order_manager, mock_risk_manager, mock_signal_router):
        """Test basic initialization of BotEngine."""
        with patch('core.bot_engine.DataFetcher', return_value=mock_data_fetcher), \
             patch('core.bot_engine.OrderManager', return_value=mock_order_manager), \
             patch('core.bot_engine.RiskManager', return_value=mock_risk_manager), \
             patch('core.bot_engine.SignalRouter', return_value=mock_signal_router), \
             patch('core.bot_engine.DiscordNotifier', return_value=None):

            engine = BotEngine(mock_config)
            await engine.initialize()

            assert engine.mode == TradingMode.PAPER
            assert engine.pairs == ["BTC/USDT"]
            assert engine.data_fetcher == mock_data_fetcher
            assert engine.order_manager == mock_order_manager
            assert engine.risk_manager == mock_risk_manager
            assert engine.signal_router == mock_signal_router

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_config):
        """Test graceful shutdown of BotEngine."""
        engine = BotEngine(mock_config)
        engine.task_manager = MagicMock()
        engine.task_manager.cancel_all = AsyncMock()
        engine._shutdown_hooks = [AsyncMock()]

        await engine.shutdown()

        assert engine.state.running is False
        engine.task_manager.cancel_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_emergency_shutdown(self, mock_config, mock_notifier, mock_order_manager):
        """Test emergency shutdown procedures."""
        with patch('core.bot_engine.DiscordNotifier', return_value=mock_notifier), \
             patch('core.bot_engine.OrderManager', return_value=mock_order_manager):

            engine = BotEngine(mock_config)
            engine.notifier = mock_notifier
            engine.order_manager = mock_order_manager
            engine.task_manager = MagicMock()
            engine.task_manager.cancel_all = AsyncMock()
            engine._shutdown_hooks = []

            await engine._emergency_shutdown()

            mock_notifier.send_alert.assert_called_once_with("🚨 EMERGENCY SHUTDOWN TRIGGERED!")
            mock_order_manager.cancel_all_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_trading_cycle(self, mock_config, mock_data_fetcher, mock_order_manager, mock_risk_manager, mock_signal_router):
        """Test a single trading cycle."""
        with patch('core.bot_engine.DataFetcher', return_value=mock_data_fetcher), \
             patch('core.bot_engine.OrderManager', return_value=mock_order_manager), \
             patch('core.bot_engine.RiskManager', return_value=mock_risk_manager), \
             patch('core.bot_engine.SignalRouter', return_value=mock_signal_router):

            engine = BotEngine(mock_config)
            engine.data_fetcher = mock_data_fetcher
            engine.order_manager = mock_order_manager
            engine.risk_manager = mock_risk_manager
            engine.signal_router = mock_signal_router
            engine.strategies = []
            engine.task_manager = MagicMock()
            engine.pairs = ["BTC/USDT"]  # Set pairs

            # Disable safe mode
            mock_order_manager.safe_mode_active = False
            mock_risk_manager.block_signals = False
            mock_signal_router.block_signals = False

            # Mock market data
            mock_data_fetcher.get_historical_data.return_value = MagicMock()

            await engine._trading_cycle()

            mock_data_fetcher.get_historical_data.assert_called_once()
            mock_risk_manager.evaluate_signal.assert_not_called()  # No signals

    def test_update_performance_metrics(self, mock_config):
        """Test performance metrics update."""
        engine = BotEngine(mock_config)
        engine.state.equity = 1000.0
        engine.performance_stats["equity_history"] = [1000.0]

        engine._update_performance_metrics(50.0)

        assert engine.performance_stats["total_pnl"] == 50.0
        assert engine.performance_stats["wins"] == 1
        assert engine.performance_stats["losses"] == 0

    @pytest.mark.asyncio
    async def test_record_trade_equity(self, mock_config, mock_order_manager):
        """Test recording trade equity."""
        engine = BotEngine(mock_config)
        engine.order_manager = mock_order_manager
        mock_order_manager.get_equity.return_value = 1050.0

        order_result = {"id": "test_trade", "pnl": 50.0}

        await engine.record_trade_equity(order_result)

        assert len(engine.performance_stats["equity_progression"]) == 1
        record = engine.performance_stats["equity_progression"][0]
        assert record["trade_id"] == "test_trade"
        assert record["equity"] == 1050.0
        assert record["pnl"] == 50.0

    def test_update_display(self, mock_config):
        """Test display update."""
        engine = BotEngine(mock_config)
        engine.live_display = MagicMock()
        engine.state.balance = 1000.0
        engine.state.equity = 1000.0
        engine.state.active_orders = 0
        engine.state.open_positions = 0

        engine._update_display()

        engine.live_display.update.assert_called_once()

    def test_print_status_table(self, mock_config, capsys):
        """Test printing status table."""
        engine = BotEngine(mock_config)
        engine.state.balance = 1000.0
        engine.state.equity = 1000.0
        engine.state.active_orders = 0
        engine.state.open_positions = 0

        engine.print_status_table()

        captured = capsys.readouterr()
        assert "Trading Bot Status" in captured.out
        assert "PAPER" in captured.out

    @pytest.mark.asyncio
    async def test_check_global_safe_mode(self, mock_config, mock_order_manager, mock_risk_manager, mock_signal_router):
        """Test global safe mode checking."""
        engine = BotEngine(mock_config)
        engine.order_manager = mock_order_manager
        engine.risk_manager = mock_risk_manager
        engine.signal_router = mock_signal_router

        # Set safe mode flags
        mock_order_manager.safe_mode_active = True
        mock_risk_manager.block_signals = False
        mock_signal_router.block_signals = False

        await engine._check_global_safe_mode()

        assert engine.global_safe_mode is True

    @pytest.mark.asyncio
    async def test_run_main_loop(self, mock_config):
        """Test main run loop."""
        engine = BotEngine(mock_config)
        engine.state.running = True
        engine.task_manager = MagicMock()
        engine.live_display = MagicMock()  # Enable display update

        # Mock to stop after one cycle
        with patch.object(engine, '_trading_cycle', new_callable=AsyncMock) as mock_cycle, \
             patch.object(engine, '_update_display') as mock_update, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

            async def stop_after_cycle():
                engine.state.running = False

            mock_cycle.side_effect = stop_after_cycle

            await engine.run()

            mock_cycle.assert_called_once()
            mock_update.assert_called_once()
            mock_sleep.assert_called_once_with(1.0)
