import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.bot_engine import BotEngine, BotState
from core.types import TradingMode


# Note: Using shared fixtures from conftest.py for consistency


class TestBotEngine:
    """Test cases for BotEngine functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self, test_config, mock_data_fetcher, mock_order_manager, mock_risk_manager, mock_signal_router):
        """Test basic initialization of BotEngine."""
        with patch('core.bot_engine.DataFetcher', return_value=mock_data_fetcher), \
             patch('core.bot_engine.OrderManager', return_value=mock_order_manager), \
             patch('core.bot_engine.RiskManager', return_value=mock_risk_manager), \
             patch('core.bot_engine.SignalRouter', return_value=mock_signal_router), \
             patch('core.bot_engine.DiscordNotifier', return_value=None):

            engine = BotEngine(test_config)
            await engine.initialize()

            assert engine.mode == TradingMode.BACKTEST
            assert engine.pairs == ["BTC/USDT", "ETH/USDT"]
            assert engine.data_fetcher == mock_data_fetcher
            assert engine.order_manager == mock_order_manager
            assert engine.risk_manager == mock_risk_manager
            assert engine.signal_router == mock_signal_router

    @pytest.mark.asyncio
    async def test_shutdown(self, test_config):
        """Test graceful shutdown of BotEngine."""
        engine = BotEngine(test_config)
        engine.task_manager = MagicMock()
        engine.task_manager.cancel_all = AsyncMock()
        engine._shutdown_hooks = [AsyncMock()]

        await engine.shutdown()

        assert engine.state.running is False
        engine.task_manager.cancel_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_emergency_shutdown(self, test_config, mock_notifier, mock_order_manager):
        """Test emergency shutdown procedures."""
        with patch('core.bot_engine.DiscordNotifier', return_value=mock_notifier), \
             patch('core.bot_engine.OrderManager', return_value=mock_order_manager):

            engine = BotEngine(test_config)
            engine.notifier = mock_notifier
            engine.order_manager = mock_order_manager
            engine.task_manager = MagicMock()
            engine.task_manager.cancel_all = AsyncMock()
            engine._shutdown_hooks = []

            await engine._emergency_shutdown()

            mock_notifier.send_alert.assert_called_once_with("🚨 EMERGENCY SHUTDOWN TRIGGERED!")
            mock_order_manager.cancel_all_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_trading_cycle(self, test_config, mock_data_fetcher, mock_order_manager, mock_risk_manager, mock_signal_router):
        """Test a single trading cycle."""
        with patch('core.bot_engine.DataFetcher', return_value=mock_data_fetcher), \
             patch('core.bot_engine.OrderManager', return_value=mock_order_manager), \
             patch('core.bot_engine.RiskManager', return_value=mock_risk_manager), \
             patch('core.bot_engine.SignalRouter', return_value=mock_signal_router):

            engine = BotEngine(test_config)
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

    def test_update_performance_metrics(self, test_config):
        """Test performance metrics update."""
        engine = BotEngine(test_config)
        engine.state.equity = 1000.0
        engine.performance_stats["equity_history"] = [1000.0]

        engine._update_performance_metrics(50.0)

        assert engine.performance_stats["total_pnl"] == 50.0
        assert engine.performance_stats["wins"] == 1
        assert engine.performance_stats["losses"] == 0

    @pytest.mark.asyncio
    async def test_record_trade_equity(self, test_config, mock_order_manager):
        """Test recording trade equity."""
        engine = BotEngine(test_config)
        engine.order_manager = mock_order_manager
        mock_order_manager.get_equity.return_value = 1050.0

        order_result = {"id": "test_trade", "pnl": 50.0}

        await engine.record_trade_equity(order_result)

        assert len(engine.performance_stats["equity_progression"]) == 1
        record = engine.performance_stats["equity_progression"][0]
        assert record["trade_id"] == "test_trade"
        assert record["equity"] == 1050.0
        assert record["pnl"] == 50.0

    def test_update_display(self, test_config):
        """Test display update."""
        engine = BotEngine(test_config)
        engine.live_display = MagicMock()
        engine.state.balance = 1000.0
        engine.state.equity = 1000.0
        engine.state.active_orders = 0
        engine.state.open_positions = 0

        engine._update_display()

        # Verify update was called once
        engine.live_display.update.assert_called_once()

        # Verify the correct state data was passed
        call_args = engine.live_display.update.call_args[0][0]  # Get the first positional argument
        assert call_args["balance"] == 1000.0
        assert call_args["equity"] == 1000.0
        assert call_args["active_orders"] == 0
        assert call_args["open_positions"] == 0
        assert "performance_metrics" in call_args
        assert call_args["performance_metrics"] == engine.performance_stats

    def test_update_display_none_case(self, test_config):
        """Test display update when live_display is None."""
        engine = BotEngine(test_config)
        engine.live_display = None  # Explicitly set to None
        engine.state.balance = 1000.0
        engine.state.equity = 1000.0
        engine.state.active_orders = 0
        engine.state.open_positions = 0

        # This should not raise any exception and should do nothing
        engine._update_display()

    def test_update_display_exception_propagation(self, test_config):
        """Test that exceptions from live_display.update are allowed to propagate."""
        engine = BotEngine(test_config)
        engine.live_display = MagicMock()
        engine.live_display.update.side_effect = ValueError("Test exception")
        engine.state.balance = 1000.0
        engine.state.equity = 1000.0
        engine.state.active_orders = 0
        engine.state.open_positions = 0

        # The exception should propagate and not be caught
        with pytest.raises(ValueError, match="Test exception"):
            engine._update_display()

    def test_print_status_table(self, test_config, capsys):
        """Test printing status table."""
        engine = BotEngine(test_config)
        engine.state.balance = 1000.0
        engine.state.equity = 1000.0
        engine.state.active_orders = 0
        engine.state.open_positions = 0

        engine.print_status_table()

        captured = capsys.readouterr()
        assert "Trading Bot Status" in captured.out
        assert "BACKTEST" in captured.out

    @pytest.mark.asyncio
    async def test_check_global_safe_mode(self, test_config, mock_order_manager, mock_risk_manager, mock_signal_router):
        """Test global safe mode checking."""
        engine = BotEngine(test_config)
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
    async def test_run_main_loop(self, test_config):
        """Test main run loop."""
        engine = BotEngine(test_config)
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

    @pytest.mark.parametrize("invalid_balance,expected_default", [
        ("invalid", 10000.0),
        (None, 10000.0),
        (-100, 10000.0),
        (0, 10000.0),
    ])
    @pytest.mark.asyncio
    async def test_initialization_with_invalid_balance(self, invalid_balance, expected_default):
        """Test initialization with various invalid balance values."""
        config = {
            "environment": {"mode": "paper"},
            "trading": {
                "portfolio_mode": False,
                "symbol": "BTC/USDT",
                "initial_balance": invalid_balance
            },
            "exchange": {"markets": ["BTC/USDT"], "base_currency": "USDT"},
            "strategies": {"active_strategies": [], "strategy_config": {}},
            "notifications": {"discord": {"enabled": False}},
            "monitoring": {"terminal_display": False, "update_interval": 1.0},
            "risk_management": {},
            "backtesting": {"timeframe": "1h"}
        }
        with patch('core.bot_engine.DataFetcher'), \
             patch('core.bot_engine.OrderManager'), \
             patch('core.bot_engine.RiskManager'), \
             patch('core.bot_engine.SignalRouter'), \
             patch('core.bot_engine.DiscordNotifier', return_value=None):

            engine = BotEngine(config)
            assert engine.starting_balance == expected_default

    @pytest.mark.parametrize("portfolio_mode,discord_enabled,terminal_display,expected_pairs", [
        (True, False, False, ["BTC/USDT", "ETH/USDT"]),
        (False, True, False, ["BTC/USDT"]),
        (False, False, True, ["BTC/USDT"]),
    ])
    @pytest.mark.asyncio
    async def test_initialization_configurations(self, mock_data_fetcher, mock_order_manager,
                                                mock_risk_manager, mock_signal_router, mock_notifier,
                                                portfolio_mode, discord_enabled, terminal_display, expected_pairs):
        """Test initialization with different configuration combinations."""
        config = {
            "environment": {"mode": "paper"},
            "trading": {
                "portfolio_mode": portfolio_mode,
                "symbol": "BTC/USDT",
                "initial_balance": 1000.0,
                "pair_allocation": {"BTC/USDT": 0.5, "ETH/USDT": 0.5} if portfolio_mode else {}
            },
            "exchange": {
                "markets": expected_pairs,
                "base_currency": "USDT"
            },
            "strategies": {"active_strategies": [], "strategy_config": {}},
            "notifications": {"discord": {"enabled": discord_enabled}},
            "monitoring": {"terminal_display": terminal_display, "update_interval": 1.0},
            "risk_management": {},
            "backtesting": {"timeframe": "1h"}
        }

        with patch('core.bot_engine.DataFetcher', return_value=mock_data_fetcher), \
             patch('core.bot_engine.OrderManager', return_value=mock_order_manager), \
             patch('core.bot_engine.RiskManager', return_value=mock_risk_manager), \
             patch('core.bot_engine.SignalRouter', return_value=mock_signal_router), \
             patch('core.bot_engine.DiscordNotifier', return_value=mock_notifier if discord_enabled else None):

            engine = BotEngine(config)
            await engine.initialize()

            assert engine.portfolio_mode == portfolio_mode
            assert engine.pairs == expected_pairs
            assert engine.live_display is None  # Since rich is removed

            if portfolio_mode:
                mock_order_manager.initialize_portfolio.assert_called_once()
            if discord_enabled:
                assert engine.notifier == mock_notifier
                mock_notifier.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_strategies_with_active_strategies(self, mock_data_fetcher, mock_order_manager, mock_risk_manager, mock_signal_router):
        """Test initializing strategies with active strategies."""
        config = {
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
                "active_strategies": ["test_strategy"],
                "strategy_config": {"test_strategy": {}}
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

        mock_strategy = MagicMock()
        mock_strategy.initialize = AsyncMock()
        mock_strategy_class = MagicMock(return_value=mock_strategy)

        with patch('core.bot_engine.DataFetcher', return_value=mock_data_fetcher), \
             patch('core.bot_engine.OrderManager', return_value=mock_order_manager), \
             patch('core.bot_engine.RiskManager', return_value=mock_risk_manager), \
             patch('core.bot_engine.SignalRouter', return_value=mock_signal_router), \
             patch('core.bot_engine.DiscordNotifier', return_value=None), \
             patch('core.bot_engine.STRATEGY_MAP', {"test_strategy": mock_strategy_class}):

            engine = BotEngine(config)
            await engine.initialize()

            assert len(engine.strategies) == 1
            assert engine.strategies[0] == mock_strategy
            mock_strategy.initialize.assert_called_once_with(mock_data_fetcher)

    @pytest.mark.asyncio
    async def test_initialize_strategies_with_unknown_strategy(self, mock_data_fetcher, mock_order_manager, mock_risk_manager, mock_signal_router):
        """Test initializing strategies with unknown strategy name."""
        config = {
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
                "active_strategies": ["unknown_strategy"],
                "strategy_config": {"unknown_strategy": {}}
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

        with patch('core.bot_engine.DataFetcher', return_value=mock_data_fetcher), \
             patch('core.bot_engine.OrderManager', return_value=mock_order_manager), \
             patch('core.bot_engine.RiskManager', return_value=mock_risk_manager), \
             patch('core.bot_engine.SignalRouter', return_value=mock_signal_router), \
             patch('core.bot_engine.DiscordNotifier', return_value=None), \
             patch('core.bot_engine.STRATEGY_MAP', {}), \
             patch('core.bot_engine.logger') as mock_logger:

            engine = BotEngine(config)
            await engine.initialize()

            assert len(engine.strategies) == 0
            mock_logger.warning.assert_called_once_with("Strategy not found: unknown_strategy")

    @pytest.mark.asyncio
    async def test_run_with_paused_state(self, test_config):
        """Test run loop when bot is paused."""
        engine = BotEngine(test_config)
        engine.state.running = True
        engine.state.paused = True
        engine.task_manager = MagicMock()

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

            async def stop_after_pause(*args, **kwargs):
                engine.state.running = False

            mock_sleep.side_effect = stop_after_pause

            await engine.run()

            mock_sleep.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_run_with_exception(self, test_config):
        """Test run loop with exception in trading cycle."""
        engine = BotEngine(test_config)
        engine.state.running = True
        engine.task_manager = MagicMock()
        engine.live_display = None

        with patch.object(engine, '_trading_cycle', side_effect=Exception("Test exception")), \
             patch.object(engine, '_emergency_shutdown', new_callable=AsyncMock) as mock_emergency, \
             patch('core.bot_engine.logger') as mock_logger:

            await engine.run()

            mock_emergency.assert_called_once()
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_trading_cycle_with_portfolio_mode(self, test_config, mock_data_fetcher, mock_order_manager, mock_risk_manager, mock_signal_router):
        """Test trading cycle in portfolio mode."""
        config = test_config.copy()
        config["trading"]["portfolio_mode"] = True
        config["exchange"]["markets"] = ["BTC/USDT", "ETH/USDT"]

        with patch('core.bot_engine.DataFetcher', return_value=mock_data_fetcher), \
             patch('core.bot_engine.OrderManager', return_value=mock_order_manager), \
             patch('core.bot_engine.RiskManager', return_value=mock_risk_manager), \
             patch('core.bot_engine.SignalRouter', return_value=mock_signal_router):

            engine = BotEngine(config)
            engine.data_fetcher = mock_data_fetcher
            engine.order_manager = mock_order_manager
            engine.risk_manager = mock_risk_manager
            engine.signal_router = mock_signal_router
            engine.strategies = []
            engine.task_manager = MagicMock()
            engine.pairs = ["BTC/USDT", "ETH/USDT"]
            engine.portfolio_mode = True

            # Disable safe mode
            mock_order_manager.safe_mode_active = False
            mock_risk_manager.block_signals = False
            mock_signal_router.block_signals = False

            # Mock realtime data
            mock_data_fetcher.get_realtime_data = AsyncMock(return_value={"BTC/USDT": MagicMock(), "ETH/USDT": MagicMock()})

            await engine._trading_cycle()

            mock_data_fetcher.get_realtime_data.assert_called_once_with(["BTC/USDT", "ETH/USDT"])

    @pytest.mark.asyncio
    async def test_trading_cycle_with_safe_mode_active(self, test_config, mock_data_fetcher, mock_order_manager, mock_risk_manager, mock_signal_router, mock_notifier):
        """Test trading cycle when global safe mode is active."""
        # Enable Discord notifications in config
        config = test_config.copy()
        config["notifications"]["discord"]["enabled"] = True

        with patch('core.bot_engine.DataFetcher', return_value=mock_data_fetcher), \
             patch('core.bot_engine.OrderManager', return_value=mock_order_manager), \
             patch('core.bot_engine.RiskManager', return_value=mock_risk_manager), \
             patch('core.bot_engine.SignalRouter', return_value=mock_signal_router), \
             patch('core.bot_engine.DiscordNotifier', return_value=mock_notifier):

            engine = BotEngine(config)
            engine.data_fetcher = mock_data_fetcher
            engine.order_manager = mock_order_manager
            engine.risk_manager = mock_risk_manager
            engine.signal_router = mock_signal_router
            engine.notifier = mock_notifier
            engine.strategies = []
            engine.task_manager = MagicMock()
            engine.pairs = ["BTC/USDT"]

            # Enable safe mode
            mock_order_manager.safe_mode_active = True
            mock_risk_manager.block_signals = False
            mock_signal_router.block_signals = False

            await engine._trading_cycle()

            # Should not proceed to trading
            mock_notifier.send_alert.assert_called_once_with("Bot entering SAFE MODE: suspending new trades.")

    @pytest.mark.asyncio
    async def test_record_trade_equity_backtest_mode(self, test_config, mock_order_manager):
        """Test recording trade equity in backtest mode."""
        config = test_config.copy()
        config["environment"]["mode"] = "backtest"

        engine = BotEngine(config)
        engine.mode = TradingMode.BACKTEST
        engine.order_manager = mock_order_manager
        engine.starting_balance = 1000.0
        engine.performance_stats["total_pnl"] = 50.0
        mock_order_manager.get_equity.return_value = 0.0  # Simulate no tracking

        order_result = {"id": "test_trade", "pnl": 50.0}

        await engine.record_trade_equity(order_result)

        record = engine.performance_stats["equity_progression"][0]
        assert record["equity"] == 1050.0  # starting_balance + total_pnl

    @pytest.mark.asyncio
    async def test_check_global_safe_mode_activation(self, test_config, mock_order_manager, mock_risk_manager, mock_signal_router):
        """Test global safe mode activation."""
        engine = BotEngine(test_config)
        engine.order_manager = mock_order_manager
        engine.risk_manager = mock_risk_manager
        engine.signal_router = mock_signal_router

        # Initially not safe
        mock_order_manager.safe_mode_active = False
        mock_risk_manager.block_signals = False
        mock_signal_router.block_signals = False

        await engine._check_global_safe_mode()
        assert engine.global_safe_mode is False

        # Activate safe mode
        mock_order_manager.safe_mode_active = True
        await engine._check_global_safe_mode()
        assert engine.global_safe_mode is True

        # Deactivate
        mock_order_manager.safe_mode_active = False
        await engine._check_global_safe_mode()
        assert engine.global_safe_mode is False

    def test_log_status(self, test_config):
        """Test logging bot status."""
        engine = BotEngine(test_config)
        engine.state.balance = 1000.0
        engine.state.equity = 1000.0
        engine.state.active_orders = 0
        engine.state.open_positions = 0

        with patch('core.bot_engine.logger') as mock_logger:
            engine._log_status()

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "Bot Status" in call_args
            assert "BACKTEST" in call_args

    def test_print_status_table_exception(self, test_config, capsys):
        """Test print_status_table with exception."""
        engine = BotEngine(test_config)
        engine.state.balance = "invalid"  # Invalid type to trigger exception

        engine.print_status_table()

        captured = capsys.readouterr()
        # Should still print something, but handle exception
        assert "Trading Bot Status" in captured.out
