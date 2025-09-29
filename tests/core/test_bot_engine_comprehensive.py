"""
Comprehensive tests for BotEngine - the main trading bot engine.

Tests initialization, trading cycles, state management, performance tracking,
integration with components, error handling, and edge cases.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
import pytest

from core.bot_engine import BotEngine, TradingMode, now_ms
from core.contracts import SignalStrength, SignalType, TradingSignal
from core.order_manager import OrderManager
from core.signal_router import SignalRouter
from core.task_manager import TaskManager
from core.timeframe_manager import TimeframeManager
from data.data_fetcher import DataFetcher
from notifier.discord_bot import DiscordNotifier
from risk.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy


class TestBotEngineInitialization:
    """Test BotEngine initialization and configuration."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return {
            "environment": {"mode": "paper"},
            "trading": {
                "portfolio_mode": False,
                "symbol": "BTC/USDT",
                "initial_balance": 10000.0,
                "pair_allocation": {"BTC/USDT": 0.6, "ETH/USDT": 0.4},
            },
            "exchange": {"markets": ["BTC/USDT"], "base_currency": "USDT"},
            "strategies": {
                "active_strategies": ["TestStrategy"],
                "strategy_config": {"TestStrategy": {"param1": "value1"}},
            },
            "notifications": {"discord": {"enabled": False}},
            "monitoring": {"update_interval": 1.0, "terminal_display": False},
            "risk_management": {},
            "backtesting": {"timeframe": "1h"},
            "cache": {"enabled": False},
            "multi_timeframe": {},
        }

    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        return {
            "data_fetcher": Mock(spec=DataFetcher),
            "order_manager": Mock(spec=OrderManager),
            "risk_manager": Mock(spec=RiskManager),
            "signal_router": Mock(spec=SignalRouter),
            "timeframe_manager": Mock(spec=TimeframeManager),
            "notifier": Mock(spec=DiscordNotifier),
            "task_manager": Mock(spec=TaskManager),
        }

    def test_initialization_with_minimal_config(self, mock_config):
        """Test BotEngine initialization with minimal configuration."""
        engine = BotEngine(mock_config)

        assert engine.mode == TradingMode.PAPER
        assert engine.portfolio_mode == False
        assert engine.pairs == ["BTC/USDT"]
        assert engine.starting_balance == 10000.0
        assert engine.state.running == True
        assert engine.state.paused == False
        assert engine.global_safe_mode == False

    def test_initialization_portfolio_mode(self, mock_config):
        """Test BotEngine initialization in portfolio mode."""
        mock_config["trading"]["portfolio_mode"] = True
        mock_config["exchange"]["markets"] = ["BTC/USDT", "ETH/USDT"]

        engine = BotEngine(mock_config)

        assert engine.portfolio_mode == True
        assert set(engine.pairs) == {"BTC/USDT", "ETH/USDT"}

    def test_initialization_with_invalid_balance(self, mock_config):
        """Test initialization with invalid balance values."""
        mock_config["trading"]["initial_balance"] = "invalid"

        engine = BotEngine(mock_config)

        # Should fallback to default
        assert engine.starting_balance == 1000.0

    def test_initialization_with_empty_markets(self, mock_config):
        """Test initialization when markets list is empty."""
        mock_config["exchange"]["markets"] = []

        engine = BotEngine(mock_config)

        # Should use configured symbol
        assert engine.pairs == ["BTC/USDT"]

    @pytest.mark.asyncio
    async def test_determine_trading_pairs_portfolio_mode(self, mock_config):
        """Test trading pairs determination in portfolio mode."""
        mock_config["trading"]["portfolio_mode"] = True
        mock_config["exchange"]["markets"] = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]

        engine = BotEngine(mock_config)
        engine._determine_trading_pairs()

        assert set(engine.pairs) == {"BTC/USDT", "ETH/USDT", "ADA/USDT"}

    @pytest.mark.asyncio
    async def test_determine_trading_pairs_single_mode(self, mock_config):
        """Test trading pairs determination in single-pair mode."""
        mock_config["trading"]["portfolio_mode"] = False
        mock_config["exchange"]["markets"] = ["BTC/USDT", "ETH/USDT"]

        engine = BotEngine(mock_config)
        engine._determine_trading_pairs()

        assert engine.pairs == ["BTC/USDT"]

    @pytest.mark.asyncio
    async def test_initialize_cache_enabled(self, mock_config):
        """Test cache initialization when enabled."""
        mock_config["cache"]["enabled"] = True

        with patch("core.bot_engine.initialize_cache") as mock_init_cache, patch(
            "core.bot_engine.close_cache"
        ) as mock_close_cache:
            mock_init_cache.return_value = True

            engine = BotEngine(mock_config)
            await engine._initialize_cache()

            mock_init_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_cache_disabled(self, mock_config):
        """Test cache initialization when disabled."""
        mock_config["cache"]["enabled"] = False

        with patch("core.bot_engine.initialize_cache") as mock_init_cache:
            engine = BotEngine(mock_config)
            await engine._initialize_cache()

            mock_init_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_core_modules(self, mock_config, mock_components):
        """Test core modules initialization."""
        with patch(
            "core.bot_engine.DataFetcher", return_value=mock_components["data_fetcher"]
        ), patch(
            "core.bot_engine.OrderManager",
            return_value=mock_components["order_manager"],
        ), patch(
            "core.bot_engine.RiskManager", return_value=mock_components["risk_manager"]
        ), patch(
            "core.bot_engine.SignalRouter",
            return_value=mock_components["signal_router"],
        ), patch(
            "core.bot_engine.TimeframeManager",
            return_value=mock_components["timeframe_manager"],
        ):
            mock_components["data_fetcher"].initialize = AsyncMock()
            mock_components["timeframe_manager"].initialize = AsyncMock()
            mock_components["order_manager"].initialize_portfolio = AsyncMock()

            engine = BotEngine(mock_config)
            await engine._initialize_core_modules()

            assert engine.data_fetcher == mock_components["data_fetcher"]
            assert engine.order_manager == mock_components["order_manager"]
            assert engine.risk_manager == mock_components["risk_manager"]
            assert engine.signal_router == mock_components["signal_router"]
            assert engine.timeframe_manager == mock_components["timeframe_manager"]

            mock_components["data_fetcher"].initialize.assert_called_once()
            mock_components["timeframe_manager"].initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_strategies_success(self, mock_config):
        """Test successful strategy initialization."""
        mock_strategy = Mock(spec=BaseStrategy)
        mock_strategy.initialize = AsyncMock()

        with patch(
            "core.bot_engine.STRATEGY_MAP",
            {"TestStrategy": Mock(return_value=mock_strategy)},
        ):
            engine = BotEngine(mock_config)
            await engine._initialize_strategies()

            assert len(engine.strategies) == 1
            assert engine.strategies[0] == mock_strategy
            mock_strategy.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_strategies_unknown_strategy(self, mock_config):
        """Test initialization with unknown strategy."""
        with patch("core.bot_engine.STRATEGY_MAP", {}), patch(
            "core.bot_engine.logger"
        ) as mock_logger:
            engine = BotEngine(mock_config)
            await engine._initialize_strategies()

            assert len(engine.strategies) == 0
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_notifications_enabled(self, mock_config, mock_components):
        """Test notification system initialization when enabled."""
        mock_config["notifications"]["discord"]["enabled"] = True

        with patch(
            "core.bot_engine.DiscordNotifier", return_value=mock_components["notifier"]
        ):
            mock_components["notifier"].initialize = AsyncMock()

            engine = BotEngine(mock_config)
            await engine._initialize_notifications()

            assert engine.notifier == mock_components["notifier"]
            mock_components["notifier"].initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_notifications_disabled(self, mock_config):
        """Test notification system initialization when disabled."""
        mock_config["notifications"]["discord"]["enabled"] = False

        with patch("core.bot_engine.DiscordNotifier") as mock_notifier_class:
            engine = BotEngine(mock_config)
            await engine._initialize_notifications()

            assert engine.notifier is None
            mock_notifier_class.assert_not_called()

    def test_initialize_display_disabled(self, mock_config):
        """Test display initialization when disabled."""
        mock_config["monitoring"]["terminal_display"] = False

        engine = BotEngine(mock_config)
        engine._initialize_display()

        assert engine.live_display is None

    @pytest.mark.parametrize(
        "invalid_balance,expected_default",
        [
            ("invalid", 1000.0),
            (None, 1000.0),
            (-100, 1000.0),
            (0, 1000.0),
        ],
    )
    @pytest.mark.asyncio
    async def test_initialization_with_invalid_balance(
        self, invalid_balance, expected_default
    ):
        """Test initialization with various invalid balance values."""
        config = {
            "environment": {"mode": "paper"},
            "trading": {
                "portfolio_mode": False,
                "symbol": "BTC/USDT",
                "initial_balance": invalid_balance,
            },
            "exchange": {"markets": ["BTC/USDT"], "base_currency": "USDT"},
            "strategies": {"active_strategies": [], "strategy_config": {}},
            "notifications": {"discord": {"enabled": False}},
            "monitoring": {"terminal_display": False, "update_interval": 1.0},
            "risk_management": {},
            "backtesting": {"timeframe": "1h"},
        }
        with patch("core.bot_engine.DataFetcher"), patch(
            "core.bot_engine.OrderManager"
        ), patch("core.bot_engine.RiskManager"), patch(
            "core.bot_engine.SignalRouter"
        ), patch(
            "core.bot_engine.DiscordNotifier", return_value=None
        ):
            engine = BotEngine(config)
            assert engine.starting_balance == expected_default

    @pytest.mark.asyncio
    async def test_initialization_discord_enabled(self, mock_config, mock_components):
        """Test initialization with Discord notifications enabled."""
        mock_config["notifications"]["discord"]["enabled"] = True

        with patch(
            "core.bot_engine.DataFetcher", return_value=mock_components["data_fetcher"]
        ), patch(
            "core.bot_engine.OrderManager",
            return_value=mock_components["order_manager"],
        ), patch(
            "core.bot_engine.RiskManager", return_value=mock_components["risk_manager"]
        ), patch(
            "core.bot_engine.SignalRouter",
            return_value=mock_components["signal_router"],
        ), patch(
            "core.bot_engine.DiscordNotifier", return_value=mock_components["notifier"]
        ):
            engine = BotEngine(mock_config)
            await engine.initialize()

            assert engine.notifier == mock_components["notifier"]
            mock_components["notifier"].initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_strategies_with_active_strategies(
        self, mock_config, mock_components
    ):
        """Test initializing strategies with active strategies."""
        mock_config["strategies"]["active_strategies"] = ["test_strategy"]
        mock_config["strategies"]["strategy_config"] = {"test_strategy": {}}

        mock_strategy = Mock(spec=BaseStrategy)
        mock_strategy.initialize = AsyncMock()

        with patch(
            "core.bot_engine.DataFetcher", return_value=mock_components["data_fetcher"]
        ), patch(
            "core.bot_engine.OrderManager",
            return_value=mock_components["order_manager"],
        ), patch(
            "core.bot_engine.RiskManager", return_value=mock_components["risk_manager"]
        ), patch(
            "core.bot_engine.SignalRouter",
            return_value=mock_components["signal_router"],
        ), patch(
            "core.bot_engine.DiscordNotifier", return_value=None
        ), patch(
            "core.bot_engine.STRATEGY_MAP",
            {"test_strategy": Mock(return_value=mock_strategy)},
        ):
            engine = BotEngine(mock_config)
            await engine.initialize()

            assert len(engine.strategies) == 1
            assert engine.strategies[0] == mock_strategy
            mock_strategy.initialize.assert_called_once_with(
                mock_components["data_fetcher"]
            )

    @pytest.mark.asyncio
    async def test_initialize_strategies_with_unknown_strategy(
        self, mock_config, mock_components
    ):
        """Test initializing strategies with unknown strategy name."""
        mock_config["strategies"]["active_strategies"] = ["unknown_strategy"]
        mock_config["strategies"]["strategy_config"] = {"unknown_strategy": {}}

        with patch(
            "core.bot_engine.DataFetcher", return_value=mock_components["data_fetcher"]
        ), patch(
            "core.bot_engine.OrderManager",
            return_value=mock_components["order_manager"],
        ), patch(
            "core.bot_engine.RiskManager", return_value=mock_components["risk_manager"]
        ), patch(
            "core.bot_engine.SignalRouter",
            return_value=mock_components["signal_router"],
        ), patch(
            "core.bot_engine.DiscordNotifier", return_value=None
        ), patch(
            "core.bot_engine.STRATEGY_MAP", {}
        ), patch(
            "core.bot_engine.logger"
        ) as mock_logger:
            engine = BotEngine(mock_config)
            await engine.initialize()

            assert len(engine.strategies) == 0
            mock_logger.warning.assert_called_once_with(
                "Strategy not found: unknown_strategy"
            )


class TestBotEngineTradingCycle:
    """Test BotEngine trading cycle functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "environment": {"mode": "paper"},
            "trading": {
                "portfolio_mode": False,
                "symbol": "BTC/USDT",
                "initial_balance": 10000.0,
            },
            "exchange": {"markets": ["BTC/USDT"], "base_currency": "USDT"},
            "strategies": {"active_strategies": [], "strategy_config": {}},
            "notifications": {"discord": {"enabled": False}},
            "monitoring": {"update_interval": 1.0},
            "risk_management": {},
            "backtesting": {"timeframe": "1h"},
        }

    @pytest.fixture
    def mock_engine(self, mock_config):
        """Create mock engine with initialized components."""
        engine = BotEngine(mock_config)

        # Mock components
        engine.data_fetcher = Mock()
        engine.order_manager = Mock()
        engine.risk_manager = Mock()
        engine.signal_router = Mock()
        engine.strategies = []
        engine.task_manager = Mock()

        # Mock component methods
        engine.data_fetcher.get_historical_data = AsyncMock()
        engine.order_manager.safe_mode_active = False
        engine.risk_manager.block_signals = False
        engine.signal_router.block_signals = False

        return engine

    @pytest.mark.asyncio
    async def test_fetch_market_data_single_timeframe(self, mock_engine):
        """Test market data fetching for single timeframe."""
        mock_engine.portfolio_mode = False
        mock_engine.pairs = ["BTC/USDT"]

        # Mock data fetcher response
        mock_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )
        mock_engine.data_fetcher.get_historical_data.return_value = mock_data

        result = await mock_engine._fetch_market_data()

        assert "BTC/USDT" in result
        mock_engine.data_fetcher.get_historical_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_market_data_portfolio_mode(self, mock_engine):
        """Test market data fetching in portfolio mode."""
        mock_engine.portfolio_mode = True
        mock_engine.pairs = ["BTC/USDT", "ETH/USDT"]

        # Mock realtime data response
        mock_data = {
            "BTC/USDT": {"price": 50000, "volume": 100},
            "ETH/USDT": {"price": 3000, "volume": 200},
        }
        mock_engine.data_fetcher.get_realtime_data = AsyncMock(return_value=mock_data)

        result = await mock_engine._fetch_market_data()

        assert "BTC/USDT" in result
        assert "ETH/USDT" in result
        mock_engine.data_fetcher.get_realtime_data.assert_called_once_with(
            ["BTC/USDT", "ETH/USDT"]
        )

    @pytest.mark.asyncio
    async def test_fetch_market_data_with_cache(self, mock_engine):
        """Test market data fetching with cache."""
        mock_engine.market_data_cache = {"BTC/USDT": "cached_data"}
        mock_engine.cache_timestamp = time.time()
        mock_engine.cache_ttl = 100  # Long TTL

        with patch(
            "core.bot_engine.time.time", return_value=mock_engine.cache_timestamp + 10
        ):
            result = await mock_engine._fetch_market_data()

            assert result == {"BTC/USDT": "cached_data"}
            # Should not call data fetcher
            mock_engine.data_fetcher.get_historical_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_trading_cycle_with_portfolio_mode(
        self,
        mock_config,
        mock_data_fetcher,
        mock_order_manager,
        mock_risk_manager,
        mock_signal_router,
    ):
        """Test trading cycle in portfolio mode."""
        config = mock_config.copy()
        config["trading"]["portfolio_mode"] = True
        config["exchange"]["markets"] = ["BTC/USDT", "ETH/USDT"]

        with patch(
            "core.bot_engine.DataFetcher", return_value=mock_data_fetcher
        ), patch(
            "core.bot_engine.OrderManager", return_value=mock_order_manager
        ), patch(
            "core.bot_engine.RiskManager", return_value=mock_risk_manager
        ), patch(
            "core.bot_engine.SignalRouter", return_value=mock_signal_router
        ):
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
            mock_data_fetcher.get_realtime_data = AsyncMock(
                return_value={"BTC/USDT": MagicMock(), "ETH/USDT": MagicMock()}
            )

            await engine._trading_cycle()

            mock_data_fetcher.get_realtime_data.assert_called_once_with(
                ["BTC/USDT", "ETH/USDT"]
            )

    @pytest.mark.asyncio
    async def test_trading_cycle_with_safe_mode_active(
        self,
        mock_config,
        mock_data_fetcher,
        mock_order_manager,
        mock_risk_manager,
        mock_signal_router,
        mock_notifier,
    ):
        """Test trading cycle when global safe mode is active."""
        # Enable Discord notifications in config
        config = mock_config.copy()
        config["notifications"]["discord"]["enabled"] = True

        with patch(
            "core.bot_engine.DataFetcher", return_value=mock_data_fetcher
        ), patch(
            "core.bot_engine.OrderManager", return_value=mock_order_manager
        ), patch(
            "core.bot_engine.RiskManager", return_value=mock_risk_manager
        ), patch(
            "core.bot_engine.SignalRouter", return_value=mock_signal_router
        ), patch(
            "core.bot_engine.DiscordNotifier", return_value=mock_notifier
        ):
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
            mock_notifier.send_alert.assert_called_once_with(
                "Bot entering SAFE MODE: suspending new trades."
            )

    @pytest.mark.asyncio
    async def test_check_safe_mode_conditions_normal(self, mock_engine):
        """Test safe mode check under normal conditions."""
        mock_engine.order_manager.safe_mode_active = False
        mock_engine.risk_manager.block_signals = False
        mock_engine.signal_router.block_signals = False

        result = await mock_engine._check_safe_mode_conditions()

        assert result == False
        assert mock_engine.global_safe_mode == False

    @pytest.mark.asyncio
    async def test_check_safe_mode_conditions_triggered(self, mock_engine):
        """Test safe mode check when conditions are triggered."""
        mock_engine.order_manager.safe_mode_active = True
        mock_engine.risk_manager.block_signals = False
        mock_engine.signal_router.block_signals = False

        with patch("core.bot_engine.logger") as mock_logger:
            result = await mock_engine._check_safe_mode_conditions()

            assert result == True
            assert mock_engine.global_safe_mode == True
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_generate_signals_from_strategies(self, mock_engine):
        """Test signal generation from strategies."""
        # Create mock strategy
        mock_strategy = Mock()
        mock_strategy.generate_signals = AsyncMock(
            return_value=[
                TradingSignal(
                    "BTC/USDT", SignalType.BUY, SignalStrength.MODERATE, 50000, 1000
                )
            ]
        )

        mock_engine.strategies = [mock_strategy]
        mock_engine.pairs = ["BTC/USDT"]

        market_data = {"BTC/USDT": pd.DataFrame({"close": [50000]})}

        signals = await mock_engine._generate_signals(market_data)

        assert len(signals) == 1
        mock_strategy.generate_signals.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_risk_approved(self, mock_engine):
        """Test risk evaluation with approved signals."""
        mock_signal = TradingSignal(
            "BTC/USDT", SignalType.BUY, SignalStrength.MODERATE, 50000, 1000
        )
        mock_engine.risk_manager.evaluate_signal = AsyncMock(return_value=True)

        signals = [mock_signal]
        market_data = {"BTC/USDT": pd.DataFrame({"close": [50000]})}

        approved_signals = await mock_engine._evaluate_risk(signals, market_data)

        assert len(approved_signals) == 1
        assert approved_signals[0] == mock_signal
        mock_engine.risk_manager.evaluate_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_risk_rejected(self, mock_engine):
        """Test risk evaluation with rejected signals."""
        mock_signal = TradingSignal(
            "BTC/USDT", SignalType.BUY, SignalStrength.MODERATE, 50000, 1000
        )
        mock_engine.risk_manager.evaluate_signal = AsyncMock(return_value=False)

        signals = [mock_signal]
        market_data = {"BTC/USDT": pd.DataFrame({"close": [50000]})}

        approved_signals = await mock_engine._evaluate_risk(signals, market_data)

        assert len(approved_signals) == 0
        mock_engine.risk_manager.evaluate_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_orders_success(self, mock_engine):
        """Test successful order execution."""
        mock_signal = TradingSignal(
            "BTC/USDT", SignalType.BUY, SignalStrength.MODERATE, 50000, 1000
        )
        mock_engine.order_manager.execute_order = AsyncMock(return_value={"pnl": 50.0})

        approved_signals = [mock_signal]

        await mock_engine._execute_orders(approved_signals)

        mock_engine.order_manager.execute_order.assert_called_once_with(mock_signal)

    @pytest.mark.asyncio
    async def test_update_state(self, mock_engine):
        """Test bot state updates."""
        mock_engine.order_manager.get_balance = AsyncMock(return_value=9500.0)
        mock_engine.order_manager.get_equity = AsyncMock(return_value=9600.0)
        mock_engine.order_manager.get_active_order_count = AsyncMock(return_value=2)
        mock_engine.order_manager.get_open_position_count = AsyncMock(return_value=1)

        await mock_engine._update_state()

        assert mock_engine.state.balance == 9500.0
        assert mock_engine.state.equity == 9600.0
        assert mock_engine.state.active_orders == 2
        assert mock_engine.state.open_positions == 1


class TestBotEnginePerformanceTracking:
    """Test performance tracking functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "environment": {"mode": "paper"},
            "trading": {
                "portfolio_mode": False,
                "symbol": "BTC/USDT",
                "initial_balance": 10000.0,
            },
            "exchange": {"markets": ["BTC/USDT"], "base_currency": "USDT"},
            "strategies": {"active_strategies": [], "strategy_config": {}},
            "notifications": {"discord": {"enabled": False}},
            "monitoring": {"update_interval": 1.0},
            "risk_management": {},
            "backtesting": {"timeframe": "1h"},
        }

    @pytest.fixture
    def mock_engine(self, mock_config):
        """Create mock engine."""
        engine = BotEngine(mock_config)
        engine.starting_balance = 10000.0

        # Initialize order_manager for tests that need it
        engine.order_manager = Mock()
        engine.order_manager.get_equity = AsyncMock(return_value=10000.0)
        engine.order_manager.get_balance = AsyncMock(return_value=10000.0)

        return engine

    def test_update_performance_metrics_win(self, mock_engine):
        """Test performance metrics update for winning trade."""
        mock_engine._update_performance_metrics(100.0)

        assert mock_engine.performance_stats["total_pnl"] == 100.0
        assert mock_engine.performance_stats["wins"] == 1
        assert mock_engine.performance_stats["losses"] == 0
        assert mock_engine.performance_stats["win_rate"] == 1.0

    def test_update_performance_metrics_loss(self, mock_engine):
        """Test performance metrics update for losing trade."""
        mock_engine._update_performance_metrics(-50.0)

        assert mock_engine.performance_stats["total_pnl"] == -50.0
        assert mock_engine.performance_stats["wins"] == 0
        assert mock_engine.performance_stats["losses"] == 1
        assert mock_engine.performance_stats["win_rate"] == 0.0

    def test_update_performance_metrics_multiple_trades(self, mock_engine):
        """Test performance metrics update for multiple trades."""
        mock_engine._update_performance_metrics(100.0)  # Win
        mock_engine._update_performance_metrics(-50.0)  # Loss
        mock_engine._update_performance_metrics(75.0)  # Win

        assert mock_engine.performance_stats["total_pnl"] == 125.0
        assert mock_engine.performance_stats["wins"] == 2
        assert mock_engine.performance_stats["losses"] == 1
        assert mock_engine.performance_stats["win_rate"] == 2 / 3

    def test_calculate_max_drawdown(self, mock_engine):
        """Test maximum drawdown calculation."""
        # Simulate equity progression
        mock_engine.state.equity = 9500.0  # -5%
        mock_engine._update_equity_history()

        mock_engine.state.equity = 9200.0  # -8%
        mock_engine._update_equity_history()

        mock_engine.state.equity = 9800.0  # -2%
        mock_engine._update_equity_history()

        mock_engine._calculate_max_drawdown()

        # Max drawdown should be from peak (10000) to trough (9200) = 8%
        assert mock_engine.performance_stats["max_drawdown"] == 0.08

    def test_calculate_sharpe_ratio(self, mock_engine):
        """Test Sharpe ratio calculation."""
        # Add some returns history
        mock_engine.performance_stats["returns_history"] = [
            0.01,
            0.005,
            -0.003,
            0.008,
            0.002,
        ]

        mock_engine._calculate_sharpe_ratio()

        # Sharpe ratio should be calculated
        assert "sharpe_ratio" in mock_engine.performance_stats
        assert isinstance(mock_engine.performance_stats["sharpe_ratio"], float)

    @pytest.mark.asyncio
    async def test_record_trade_equity_success(self, mock_engine):
        """Test successful trade equity recording."""
        mock_engine.order_manager.get_equity = AsyncMock(return_value=10100.0)

        order_result = {"id": "test_trade_123", "pnl": 100.0, "symbol": "BTC/USDT"}

        await mock_engine.record_trade_equity(order_result)

        assert len(mock_engine.performance_stats["equity_progression"]) == 1

        record = mock_engine.performance_stats["equity_progression"][0]
        assert record["trade_id"] == "test_trade_123"
        assert record["equity"] == 10100.0
        assert record["pnl"] == 100.0
        assert record["symbol"] == "BTC/USDT"
        assert "cumulative_return" in record

    @pytest.mark.asyncio
    async def test_record_trade_equity_missing_equity(self, mock_engine):
        """Test trade equity recording when equity is unavailable."""
        mock_engine.order_manager.get_equity = AsyncMock(return_value=0.0)

        order_result = {"id": "test_trade", "pnl": 50.0}

        await mock_engine.record_trade_equity(order_result)

        record = mock_engine.performance_stats["equity_progression"][0]
        # Should calculate equity from starting balance + total pnl
        assert record["equity"] == 10050.0  # 10000 + 50

    @pytest.mark.asyncio
    async def test_record_trade_equity_backtest_mode(
        self, mock_config, mock_order_manager
    ):
        """Test recording trade equity in backtest mode."""
        config = mock_config.copy()
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
    async def test_check_global_safe_mode_activation(
        self, mock_config, mock_order_manager, mock_risk_manager, mock_signal_router
    ):
        """Test global safe mode activation."""
        engine = BotEngine(mock_config)
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

    def test_calculate_cumulative_return(self, mock_engine):
        """Test cumulative return calculation."""
        mock_engine.starting_balance = 10000.0

        # Test positive return
        return_value = mock_engine._calculate_cumulative_return(11000.0)
        assert return_value == 0.1  # 10% return

        # Test negative return
        return_value = mock_engine._calculate_cumulative_return(9500.0)
        assert return_value == -0.05  # -5% return

        # Test zero return
        return_value = mock_engine._calculate_cumulative_return(10000.0)
        assert return_value == 0.0

    def test_create_equity_record_complete(self, mock_engine):
        """Test creation of complete equity record."""
        order_result = {
            "id": "trade_123",
            "timestamp": 1640995200000,  # Example timestamp
            "symbol": "BTC/USDT",
            "pnl": 150.0,
        }

        record = mock_engine._create_equity_record(order_result, 10150.0, 0.015)

        assert record["trade_id"] == "trade_123"
        assert record["timestamp"] == 1640995200000
        assert record["symbol"] == "BTC/USDT"
        assert record["equity"] == 10150.0
        assert record["pnl"] == 150.0
        assert record["cumulative_return"] == 0.015

    def test_create_equity_record_missing_fields(self, mock_engine):
        """Test creation of equity record with missing fields."""
        order_result = {}  # Empty order result

        record = mock_engine._create_equity_record(order_result, 10000.0, 0.0)

        # Should generate trade_id and timestamp
        assert "trade_id" in record
        assert "timestamp" in record
        assert record["equity"] == 10000.0
        assert record["pnl"] is None
        assert record["symbol"] is None


class TestBotEngineStateManagement:
    """Test BotEngine state management."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "environment": {"mode": "paper"},
            "trading": {
                "portfolio_mode": False,
                "symbol": "BTC/USDT",
                "initial_balance": 10000.0,
            },
            "exchange": {"markets": ["BTC/USDT"], "base_currency": "USDT"},
            "strategies": {"active_strategies": [], "strategy_config": {}},
            "notifications": {"discord": {"enabled": False}},
            "monitoring": {"update_interval": 1.0},
            "risk_management": {},
            "backtesting": {"timeframe": "1h"},
        }

    @pytest.fixture
    def mock_engine(self, mock_config):
        """Create mock engine."""
        engine = BotEngine(mock_config)
        return engine

    @pytest.mark.asyncio
    async def test_run_main_loop_normal_operation(self, mock_engine):
        """Test main run loop under normal operation."""
        mock_engine.state.running = True

        with patch.object(
            mock_engine, "_trading_cycle", new_callable=AsyncMock
        ) as mock_cycle, patch.object(
            mock_engine, "_update_display"
        ) as mock_update, patch(
            "asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            # Stop after first cycle
            async def stop_after_cycle():
                mock_engine.state.running = False

            mock_cycle.side_effect = stop_after_cycle

            await mock_engine.run()

            mock_cycle.assert_called_once()
            mock_update.assert_called_once()
            mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_run_main_loop_paused_state(self, mock_engine):
        """Test main run loop when bot is paused."""
        mock_engine.state.running = True
        mock_engine.state.paused = True

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Stop after a few sleep cycles
            async def stop_after_sleeps(*args, **kwargs):
                mock_engine.state.running = False

            mock_sleep.side_effect = stop_after_sleeps

            await mock_engine.run()

            # Should have called sleep multiple times while paused
            assert mock_sleep.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_main_loop_with_exception(self, mock_engine):
        """Test main run loop exception handling."""
        mock_engine.state.running = True

        with patch.object(
            mock_engine, "_trading_cycle", side_effect=Exception("Test error")
        ), patch.object(
            mock_engine, "_emergency_shutdown", new_callable=AsyncMock
        ) as mock_emergency, patch(
            "core.bot_engine.logger"
        ) as mock_logger:
            await mock_engine.run()

            mock_emergency.assert_called_once()
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_run_with_paused_state(self, mock_config):
        """Test run loop when bot is paused."""
        engine = BotEngine(mock_config)
        engine.state.running = True
        engine.state.paused = True
        engine.task_manager = MagicMock()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

            async def stop_after_pause(*args, **kwargs):
                engine.state.running = False

            mock_sleep.side_effect = stop_after_pause

            await engine.run()

            mock_sleep.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_run_with_exception(self, mock_config):
        """Test run loop with exception in trading cycle."""
        engine = BotEngine(mock_config)
        engine.state.running = True
        engine.task_manager = MagicMock()
        engine.live_display = None

        async def fake_emergency_shutdown():
            engine.state.running = False

        with patch.object(
            engine, "_trading_cycle", side_effect=Exception("Test exception")
        ), patch.object(
            engine, "_emergency_shutdown", side_effect=fake_emergency_shutdown
        ) as mock_emergency, patch(
            "core.bot_engine.logger"
        ) as mock_logger:
            await engine.run()

            mock_emergency.assert_called_once()
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_emergency_shutdown_complete(self, mock_engine):
        """Test complete emergency shutdown procedure."""
        mock_engine.notifier = Mock()
        mock_engine.order_manager = Mock()

        mock_engine.notifier.send_alert = AsyncMock()
        mock_engine.order_manager.cancel_all_orders = AsyncMock()

        await mock_engine._emergency_shutdown()

        mock_engine.notifier.send_alert.assert_called_once()
        mock_engine.order_manager.cancel_all_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_hooks(self, mock_engine):
        """Test shutdown with cleanup hooks."""
        mock_hook1 = AsyncMock()
        mock_hook2 = AsyncMock()

        mock_engine._shutdown_hooks = [mock_hook1, mock_hook2]
        mock_engine.task_manager = Mock()
        mock_engine.task_manager.cancel_all = AsyncMock()

        await mock_engine.shutdown()

        mock_hook1.assert_called_once()
        mock_hook2.assert_called_once()
        mock_engine.task_manager.cancel_all.assert_called_once()
        assert mock_engine.state.running == False

    def test_log_status_normal(self, mock_engine):
        """Test status logging under normal conditions."""
        mock_engine.state.balance = 10000.0
        mock_engine.state.equity = 10100.0
        mock_engine.state.active_orders = 2
        mock_engine.state.open_positions = 1

        with patch("core.bot_engine.logger") as mock_logger:
            mock_engine._log_status()

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]

            assert "Bot Status" in call_args
            assert "PAPER" in call_args
            assert "10000.00" in call_args
            assert "10100.00" in call_args

    def test_print_status_table_normal(self, mock_engine, capsys):
        """Test status table printing."""
        mock_engine.state.balance = 10000.0
        mock_engine.state.equity = 10100.0
        mock_engine.state.active_orders = 2
        mock_engine.state.open_positions = 1
        mock_engine.performance_stats = {"total_pnl": 100.0, "win_rate": 0.75}

        mock_engine.print_status_table()

        captured = capsys.readouterr()
        assert "Trading Bot Status" in captured.out
        assert "PAPER" in captured.out
        assert "100.00" in captured.out  # Total PnL
        assert "75.00%" in captured.out  # Win Rate


class TestBotEngineIntegration:
    """Test BotEngine integration with other components."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "environment": {"mode": "paper"},
            "trading": {
                "portfolio_mode": False,
                "symbol": "BTC/USDT",
                "initial_balance": 10000.0,
            },
            "exchange": {"markets": ["BTC/USDT"], "base_currency": "USDT"},
            "strategies": {"active_strategies": [], "strategy_config": {}},
            "notifications": {"discord": {"enabled": False}},
            "monitoring": {"update_interval": 1.0},
            "risk_management": {},
            "backtesting": {"timeframe": "1h"},
        }

    @pytest.fixture
    def mock_engine(self, mock_config):
        """Create mock engine with components."""
        engine = BotEngine(mock_config)

        # Mock components
        engine.data_fetcher = Mock()
        engine.order_manager = Mock()
        engine.risk_manager = Mock()
        engine.signal_router = Mock()
        engine.strategies = []
        engine.task_manager = Mock()

        return engine

    @pytest.mark.asyncio
    async def test_process_binary_integration_enabled(self, mock_engine):
        """Test binary integration processing when enabled."""
        with patch("core.bot_engine.get_binary_integration") as mock_get_integration:
            mock_integration = Mock()
            mock_integration.enabled = True
            mock_integration.process_market_data = AsyncMock(
                return_value=Mock(should_trade=True, reasoning="Test decision")
            )
            mock_get_integration.return_value = mock_integration

            market_data = {"BTC/USDT": pd.DataFrame({"close": [50000]})}

            decisions = await mock_engine._process_binary_integration(market_data)

            assert len(decisions) == 1
            mock_integration.process_market_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_binary_integration_disabled(self, mock_engine):
        """Test binary integration processing when disabled."""
        with patch("core.bot_engine.get_binary_integration") as mock_get_integration:
            mock_integration = Mock()
            mock_integration.enabled = False
            mock_get_integration.return_value = mock_integration

            market_data = {"BTC/USDT": pd.DataFrame({"close": [50000]})}

            decisions = await mock_engine._process_binary_integration(market_data)

            assert decisions == []

    @pytest.mark.asyncio
    async def test_execute_integrated_decisions(self, mock_engine):
        """Test execution of integrated trading decisions."""
        mock_engine.order_manager.execute_order = AsyncMock(return_value={"pnl": 100.0})

        decision_data = {
            "symbol": "BTC/USDT",
            "decision": Mock(
                should_trade=True,
                binary_probability=0.8,
                selected_strategy=Mock(__name__="TestStrategy"),
                direction="long",
                position_size=1000.0,
                stop_loss=49000.0,
                take_profit=52000.0,
                reasoning="Test decision",
            ),
            "market_data": pd.DataFrame({"close": [50000]}),
        }

        integrated_decisions = [decision_data]

        await mock_engine._execute_integrated_decisions(integrated_decisions)

        mock_engine.order_manager.execute_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_global_safe_mode_transitions(self, mock_engine):
        """Test global safe mode state transitions."""
        # Start with normal state
        mock_engine.order_manager.safe_mode_active = False
        mock_engine.risk_manager.block_signals = False
        mock_engine.signal_router.block_signals = False

        await mock_engine._check_global_safe_mode()
        assert mock_engine.global_safe_mode == False

        # Trigger safe mode
        mock_engine.order_manager.safe_mode_active = True

        await mock_engine._check_global_safe_mode()
        assert mock_engine.global_safe_mode == True

        # Clear safe mode
        mock_engine.order_manager.safe_mode_active = False

        await mock_engine._check_global_safe_mode()
        assert mock_engine.global_safe_mode == False


class TestBotEngineErrorHandling:
    """Test BotEngine error handling and resilience."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "environment": {"mode": "paper"},
            "trading": {
                "portfolio_mode": False,
                "symbol": "BTC/USDT",
                "initial_balance": 10000.0,
            },
            "exchange": {"markets": ["BTC/USDT"], "base_currency": "USDT"},
            "strategies": {"active_strategies": [], "strategy_config": {}},
            "notifications": {"discord": {"enabled": False}},
            "monitoring": {"update_interval": 1.0},
            "risk_management": {},
            "backtesting": {"timeframe": "1h"},
        }

    @pytest.fixture
    def mock_engine(self, mock_config):
        """Create mock engine."""
        engine = BotEngine(mock_config)
        return engine

    @pytest.mark.asyncio
    async def test_trading_cycle_with_data_fetch_error(self, mock_engine):
        """Test trading cycle resilience to data fetch errors."""
        mock_engine.data_fetcher = Mock()
        mock_engine.data_fetcher.get_historical_data = AsyncMock(
            side_effect=Exception("Data fetch failed")
        )

        # Should not crash, should handle gracefully
        await mock_engine._trading_cycle()

        # Engine should still be in valid state
        assert mock_engine.state.running == True

    @pytest.mark.asyncio
    async def test_signal_generation_with_strategy_error(self, mock_engine):
        """Test signal generation resilience to strategy errors."""
        mock_strategy = Mock()
        mock_strategy.generate_signals = AsyncMock(
            side_effect=Exception("Strategy failed")
        )

        mock_engine.strategies = [mock_strategy]
        mock_engine.pairs = ["BTC/USDT"]

        market_data = {"BTC/USDT": pd.DataFrame({"close": [50000]})}

        # Should not crash, should handle gracefully
        signals = await mock_engine._generate_signals(market_data)

        # Should return empty signals on error
        assert signals == []

    @pytest.mark.asyncio
    async def test_risk_evaluation_with_component_error(self, mock_engine):
        """Test risk evaluation resilience to component errors."""
        mock_engine.risk_manager.evaluate_signal = AsyncMock(
            side_effect=Exception("Risk evaluation failed")
        )

        signals = [Mock()]
        market_data = {"BTC/USDT": pd.DataFrame({"close": [50000]})}

        # Should not crash, should handle gracefully
        approved_signals = await mock_engine._evaluate_risk(signals, market_data)

        # Should return empty approved signals on error
        assert approved_signals == []

    @pytest.mark.asyncio
    async def test_order_execution_with_component_error(self, mock_engine):
        """Test order execution resilience to component errors."""
        mock_engine.order_manager.execute_order = AsyncMock(
            side_effect=Exception("Order execution failed")
        )

        signals = [Mock()]

        # Should not crash, should handle gracefully
        await mock_engine._execute_orders(signals)

        # Should continue execution despite error
        assert mock_engine.state.running == True

    def test_performance_calculation_with_invalid_data(self, mock_engine):
        """Test performance calculation with invalid data."""
        # Add invalid equity values
        mock_engine.state.equity = "invalid"

        # Should not crash, should handle gracefully
        mock_engine._update_performance_metrics(100.0)

        # Performance stats should still be updated
        assert "total_pnl" in mock_engine.performance_stats

    @pytest.mark.asyncio
    async def test_state_update_with_component_errors(self, mock_engine):
        """Test state update resilience to component errors."""
        mock_engine.order_manager.get_balance = AsyncMock(
            side_effect=Exception("Balance fetch failed")
        )
        mock_engine.order_manager.get_equity = AsyncMock(
            return_value=10000.0
        )  # This should work
        mock_engine.order_manager.get_active_order_count = AsyncMock(return_value=0)
        mock_engine.order_manager.get_open_position_count = AsyncMock(return_value=0)

        # Should not crash, should handle gracefully
        await mock_engine._update_state()

        # State should be updated with available data
        assert mock_engine.state.equity == 10000.0


class TestBotEngineEdgeCases:
    """Test BotEngine edge cases and boundary conditions."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "environment": {"mode": "paper"},
            "trading": {
                "portfolio_mode": False,
                "symbol": "BTC/USDT",
                "initial_balance": 10000.0,
            },
            "exchange": {"markets": ["BTC/USDT"], "base_currency": "USDT"},
            "strategies": {"active_strategies": [], "strategy_config": {}},
            "notifications": {"discord": {"enabled": False}},
            "monitoring": {"update_interval": 1.0},
            "risk_management": {},
            "backtesting": {"timeframe": "1h"},
        }

    @pytest.fixture
    def mock_engine(self, mock_config):
        """Create mock engine."""
        engine = BotEngine(mock_config)
        return engine

    def test_initialization_with_extreme_values(self, mock_config):
        """Test initialization with extreme configuration values."""
        mock_config["trading"]["initial_balance"] = 999999999.99
        mock_config["monitoring"]["update_interval"] = 0.1

        engine = BotEngine(mock_config)

        assert engine.starting_balance == 999999999.99
        assert engine.config["monitoring"]["update_interval"] == 0.1

    @pytest.mark.asyncio
    async def test_trading_cycle_with_empty_market_data(self, mock_engine):
        """Test trading cycle with empty market data."""
        mock_engine.data_fetcher = Mock()
        mock_engine.data_fetcher.get_historical_data = AsyncMock(
            return_value=pd.DataFrame()
        )

        # Should handle empty data gracefully
        await mock_engine._trading_cycle()

        assert mock_engine.state.running == True

    @pytest.mark.asyncio
    async def test_trading_cycle_with_none_market_data(self, mock_engine):
        """Test trading cycle with None market data."""
        mock_engine.data_fetcher = Mock()
        mock_engine.data_fetcher.get_historical_data = AsyncMock(return_value=None)

        # Should handle None data gracefully
        await mock_engine._trading_cycle()

        assert mock_engine.state.running == True

    def test_performance_calculation_with_zero_division(self, mock_engine):
        """Test performance calculation avoiding zero division."""
        mock_engine.starting_balance = 0.0  # Invalid balance

        # Should handle zero division gracefully
        return_value = mock_engine._calculate_cumulative_return(100.0)

        assert return_value == 0.0

    def test_display_update_with_none_display(self, mock_engine):
        """Test display update when display is None."""
        mock_engine.live_display = None

        # Should not crash
        mock_engine._update_display()

        assert mock_engine.live_display is None

    @pytest.mark.asyncio
    async def test_concurrent_trading_cycles(self, mock_engine):
        """Test concurrent trading cycle execution."""
        mock_engine.data_fetcher = Mock()
        mock_engine.data_fetcher.get_historical_data = AsyncMock(
            return_value=pd.DataFrame({"close": [50000]})
        )

        # Run multiple concurrent trading cycles
        tasks = [mock_engine._trading_cycle() for _ in range(5)]
        await asyncio.gather(*tasks)

        # Should handle concurrency without issues
        assert mock_engine.state.running == True

    def test_status_logging_with_invalid_values(self, mock_engine):
        """Test status logging with invalid values."""
        mock_engine.state.balance = None
        mock_engine.state.equity = float("inf")

        # Should handle invalid values gracefully
        with patch("core.bot_engine.logger") as mock_logger:
            mock_engine._log_status()

            # Should still attempt to log
            mock_logger.info.assert_called_once()


class TestBotEngineUtilityFunctions:
    """Test BotEngine utility functions."""

    def test_now_ms_function(self):
        """Test now_ms timestamp function."""
        timestamp = now_ms()

        assert isinstance(timestamp, int)
        assert timestamp > 0

        # Should be current time in milliseconds
        import time

        current_time_ms = int(time.time() * 1000)
        assert abs(timestamp - current_time_ms) < 1000  # Within 1 second

    def test_extract_multi_tf_data_with_valid_data(self):
        """Test multi-timeframe data extraction with valid data."""
        engine = BotEngine({})

        market_data = {
            "BTC/USDT": {
                "single_timeframe": pd.DataFrame({"close": [50000]}),
                "multi_timeframe": {
                    "1h": pd.DataFrame({"close": [50000]}),
                    "4h": pd.DataFrame({"close": [50100]}),
                },
            }
        }

        result = engine._extract_multi_tf_data(market_data, "BTC/USDT")

        assert result is not None
        assert "1h" in result
        assert "4h" in result

    def test_extract_multi_tf_data_with_missing_data(self):
        """Test multi-timeframe data extraction with missing data."""
        engine = BotEngine({})

        market_data = {"BTC/USDT": pd.DataFrame({"close": [50000]})}

        result = engine._extract_multi_tf_data(market_data, "BTC/USDT")

        assert result is None

    def test_extract_multi_tf_data_with_invalid_symbol(self):
        """Test multi-timeframe data extraction with invalid symbol."""
        engine = BotEngine({})

        market_data = {"BTC/USDT": pd.DataFrame({"close": [50000]})}

        result = engine._extract_multi_tf_data(market_data, "INVALID")

        assert result is None

    def test_combine_market_data_function(self):
        """Test market data combination function."""
        engine = BotEngine({})

        market_data = {"BTC/USDT": pd.DataFrame({"close": [50000]})}
        multi_timeframe_data = {
            "BTC/USDT": {
                "1h": pd.DataFrame({"close": [50000]}),
                "4h": pd.DataFrame({"close": [50100]}),
            }
        }

        result = engine._combine_market_data(market_data, multi_timeframe_data)

        assert "BTC/USDT" in result
        assert "single_timeframe" in result["BTC/USDT"]
        assert "multi_timeframe" in result["BTC/USDT"]

    def test_initialize_performance_stats(self, mock_config):
        """Test performance stats initialization."""
        engine = BotEngine(mock_config)

        # Clear existing stats
        engine.performance_stats = {}

        engine._initialize_performance_stats()

        assert "total_pnl" in engine.performance_stats
        assert "equity_history" in engine.performance_stats
        assert "returns_history" in engine.performance_stats
        assert "wins" in engine.performance_stats
        assert "losses" in engine.performance_stats
        assert "equity_progression" in engine.performance_stats

    def test_update_pnl(self, mock_config):
        """Test PnL update."""
        engine = BotEngine(mock_config)
        engine.performance_stats["total_pnl"] = 100.0

        engine._update_pnl(50.0)
        assert engine.performance_stats["total_pnl"] == 150.0

    def test_update_win_loss_counts(self, mock_config):
        """Test win/loss count updates."""
        engine = BotEngine(mock_config)

        # Test win
        engine._update_win_loss_counts(50.0)
        assert engine.performance_stats["wins"] == 1
        assert engine.performance_stats["losses"] == 0

        # Test loss
        engine._update_win_loss_counts(-25.0)
        assert engine.performance_stats["wins"] == 1
        assert engine.performance_stats["losses"] == 1

        # Test zero (should not count)
        engine._update_win_loss_counts(0.0)
        assert engine.performance_stats["wins"] == 1
        assert engine.performance_stats["losses"] == 1

    def test_log_status(self, mock_config):
        """Test logging bot status."""
        engine = BotEngine(mock_config)
        engine.state.balance = 1000.0
        engine.state.equity = 1000.0
        engine.state.active_orders = 0
        engine.state.open_positions = 0

        with patch("core.bot_engine.logger") as mock_logger:
            engine._log_status()

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "Bot Status" in call_args
            assert "PAPER" in call_args

    def test_print_status_table_exception(self, mock_config):
        """Test print_status_table with exception."""
        engine = BotEngine(mock_config)
        engine.state.balance = "invalid"  # Invalid type to trigger exception

        with patch("core.bot_engine.logger") as mock_logger:
            engine.print_status_table()

            # Should still log something, but handle exception
            mock_logger.exception.assert_called_with("Failed to log status table")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
