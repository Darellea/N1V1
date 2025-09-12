"""
Integration Tests for Binary Model Integration

Tests the complete flow: market data → binary model → strategy selector → risk manager → order executor
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from core.binary_model_integration import (
    BinaryModelIntegration,
    BinaryModelResult,
    StrategySelectionResult,
    IntegratedTradingDecision,
    get_binary_integration
)
from core.bot_engine import BotEngine
from ml.trainer import CalibratedModel
from strategies.regime.strategy_selector import StrategySelector
from strategies.regime.market_regime import MarketRegime
from risk.risk_manager import RiskManager
from core.contracts import TradingSignal, SignalType, SignalStrength


class TestBinaryModelIntegration:
    """Test the binary model integration component."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        np.random.seed(42)

        data = {
            'open': 100 + np.random.normal(0, 2, 100).cumsum(),
            'high': 105 + np.random.normal(0, 2, 100).cumsum(),
            'low': 95 + np.random.normal(0, 2, 100).cumsum(),
            'close': 100 + np.random.normal(0, 1, 100).cumsum(),
            'volume': np.random.uniform(1000, 5000, 100)
        }

        df = pd.DataFrame(data, index=dates)
        # Ensure high >= close >= low >= open (basic OHLC integrity)
        for i in range(len(df)):
            df.loc[df.index[i], 'high'] = max(df.loc[df.index[i], ['open', 'high', 'low', 'close']])
            df.loc[df.index[i], 'low'] = min(df.loc[df.index[i], ['open', 'high', 'low', 'close']])

        return df

    @pytest.fixture
    def mock_binary_model(self):
        """Create a mock calibrated binary model."""
        model = Mock(spec=CalibratedModel)
        model.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70% probability of trade
        return model

    @pytest.fixture
    def mock_risk_manager(self):
        """Create a mock risk manager."""
        risk_manager = Mock(spec=RiskManager)
        risk_manager.evaluate_signal = AsyncMock(return_value=True)
        risk_manager.calculate_position_size = AsyncMock(return_value=1000.0)
        risk_manager.calculate_dynamic_stop_loss = AsyncMock(return_value=95.0)
        risk_manager.calculate_take_profit = AsyncMock(return_value=110.0)
        return risk_manager

    @pytest.fixture
    def binary_integration_config(self):
        """Configuration for binary integration."""
        return {
            "binary_integration": {
                "enabled": True,
                "threshold": 0.6,
                "min_confidence": 0.5,
                "require_regime_confirmation": True,
                "use_adaptive_position_sizing": True
            }
        }

    @pytest.fixture
    async def binary_integration(self, binary_integration_config, mock_binary_model, mock_risk_manager):
        """Create binary integration instance."""
        # Create mock strategy selector
        mock_strategy_selector = Mock()
        mock_strategy_selector.select_strategy.return_value = Mock(__name__="TestStrategy")

        # Initialize with mock strategy selector injected
        integration = BinaryModelIntegration(binary_integration_config, strategy_selector=mock_strategy_selector)

        # Initialize with mocks (don't overwrite the injected strategy selector)
        integration.binary_model = mock_binary_model
        integration.risk_manager = mock_risk_manager

        return integration

    @pytest.mark.asyncio
    async def test_binary_model_prediction_trade(self, binary_integration, sample_market_data):
        """Test binary model prediction that results in a trade decision."""
        # Process market data
        decision = await binary_integration.process_market_data(sample_market_data, "BTC/USDT")

        # Verify decision
        assert decision.should_trade == True
        assert decision.binary_probability == 0.7
        assert decision.direction in ["long", "short"]
        assert decision.selected_strategy is not None
        assert decision.position_size > 0
        assert decision.stop_loss is not None
        assert decision.take_profit is not None

    @pytest.mark.asyncio
    async def test_binary_model_prediction_skip(self, binary_integration, sample_market_data, mock_binary_model):
        """Test binary model prediction that results in skip decision."""
        # Mock low probability
        mock_binary_model.predict_proba.return_value = np.array([[0.8, 0.2]])  # 20% probability

        decision = await binary_integration.process_market_data(sample_market_data, "BTC/USDT")

        # Verify skip decision
        assert decision.should_trade == False
        assert decision.binary_probability == 0.2
        assert "Binary model suggests skip" in decision.reasoning

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, binary_integration):
        """Test handling of insufficient market data."""
        # Empty dataframe
        empty_df = pd.DataFrame()

        decision = await binary_integration.process_market_data(empty_df, "BTC/USDT")

        assert decision.should_trade == False
        assert "Binary integration disabled" in decision.reasoning

    @pytest.mark.asyncio
    async def test_risk_manager_rejection(self, binary_integration, sample_market_data, mock_risk_manager):
        """Test when risk manager rejects the trade."""
        # Mock risk rejection
        mock_risk_manager.evaluate_signal.return_value = False

        decision = await binary_integration.process_market_data(sample_market_data, "BTC/USDT")

        assert decision.should_trade == False
        assert "Risk: Rejected" in decision.reasoning

    @pytest.mark.asyncio
    async def test_strategy_selection_failure(self, binary_integration, sample_market_data):
        """Test handling of strategy selection failure."""
        # Mock strategy selector failure
        binary_integration.strategy_selector.select_strategy.return_value = None

        decision = await binary_integration.process_market_data(sample_market_data, "BTC/USDT")

        assert decision.should_trade == False
        assert decision.selected_strategy is None

    @pytest.mark.asyncio
    async def test_feature_extraction(self, binary_integration, sample_market_data):
        """Test feature extraction from market data."""
        features = binary_integration._extract_features(sample_market_data)

        # Verify expected features are present
        expected_features = ['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']
        for feature in expected_features:
            assert feature in features
            assert not pd.isna(features[feature])

    @pytest.mark.asyncio
    async def test_integration_disabled(self, binary_integration_config):
        """Test behavior when binary integration is disabled."""
        config = binary_integration_config.copy()
        config["binary_integration"]["enabled"] = False

        integration = BinaryModelIntegration(config)
        decision = await integration.process_market_data(pd.DataFrame(), "BTC/USDT")

        assert decision.should_trade == False
        assert "Binary integration disabled" in decision.reasoning


class TestBotEngineIntegration:
    """Test the complete integration in BotEngine."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for BotEngine."""
        return {
            "environment": {"mode": "backtest"},
            "trading": {
                "portfolio_mode": False,
                "symbol": "BTC/USDT",
                "initial_balance": 10000.0
            },
            "exchange": {
                "base_currency": "USDT",
                "markets": ["BTC/USDT"]
            },
            "binary_integration": {
                "enabled": True,
                "threshold": 0.6,
                "min_confidence": 0.5
            },
            "strategies": {
                "active_strategies": ["TestStrategy"],
                "strategy_config": {
                    "TestStrategy": {}
                }
            },
            "risk_management": {},
            "notifications": {"discord": {"enabled": False}},
            "monitoring": {"update_interval": 1.0}
        }

    @pytest.fixture
    def sample_market_data_dict(self):
        """Create sample market data dictionary."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='1H')
        np.random.seed(42)

        data = {
            'open': 100 + np.random.normal(0, 1, 50),
            'high': 105 + np.random.normal(0, 1, 50),
            'low': 95 + np.random.normal(0, 1, 50),
            'close': 100 + np.random.normal(0, 0.5, 50),
            'volume': np.random.uniform(1000, 5000, 50)
        }

        df = pd.DataFrame(data, index=dates)
        return {"BTC/USDT": df}

    @pytest.mark.asyncio
    async def test_trading_cycle_with_binary_integration(self, mock_config, sample_market_data_dict):
        """Test complete trading cycle with binary integration enabled."""
        with patch('core.bot_engine.get_binary_integration') as mock_get_integration, \
             patch('core.bot_engine.DataFetcher') as mock_data_fetcher, \
             patch('core.bot_engine.TimeframeManager') as mock_tf_manager, \
             patch('core.bot_engine.RiskManager') as mock_risk_manager, \
             patch('core.bot_engine.OrderManager') as mock_order_manager, \
             patch('core.bot_engine.SignalRouter') as mock_signal_router, \
             patch('core.bot_engine.DiscordNotifier') as mock_notifier, \
             patch('core.bot_engine.TaskManager') as mock_task_manager:

            # Mock binary integration
            mock_integration = Mock()
            mock_integration.enabled = True
            mock_integration.process_market_data = AsyncMock(return_value=Mock(
                should_trade=True,
                binary_probability=0.7,
                selected_strategy=Mock(__name__="TestStrategy"),
                direction="long",
                position_size=1000.0,
                stop_loss=95.0,
                take_profit=110.0,
                reasoning="Test decision"
            ))
            mock_get_integration.return_value = mock_integration

            # Mock other components
            mock_data_fetcher.return_value.initialize = AsyncMock()
            mock_tf_manager.return_value.initialize = AsyncMock()
            mock_risk_manager.return_value = Mock()
            mock_order_manager.return_value = Mock()
            mock_order_manager.return_value.initialize = AsyncMock()
            mock_order_manager.return_value.configure_portfolio = AsyncMock()
            mock_signal_router.return_value = Mock()
            mock_notifier.return_value = Mock()
            mock_task_manager.return_value = Mock()

            # Create bot engine
            bot = BotEngine(mock_config)

            # Mock the trading cycle components
            bot._fetch_market_data = AsyncMock(return_value=sample_market_data_dict)
            bot._check_safe_mode_conditions = AsyncMock(return_value=False)
            bot._process_binary_integration = AsyncMock(return_value=[{
                'symbol': 'BTC/USDT',
                'decision': Mock(
                    should_trade=True,
                    binary_probability=0.7,
                    selected_strategy=Mock(__name__="TestStrategy"),
                    direction="long",
                    position_size=1000.0,
                    stop_loss=95.0,
                    take_profit=110.0,
                    reasoning="Test decision"
                ),
                'market_data': sample_market_data_dict['BTC/USDT']
            }])
            bot._execute_integrated_decisions = AsyncMock()
            bot._update_state = AsyncMock()

            # Run trading cycle
            await bot._trading_cycle()

            # Verify binary integration was called
            bot._process_binary_integration.assert_called_once_with(sample_market_data_dict)
            bot._execute_integrated_decisions.assert_called_once()

    @pytest.mark.asyncio
    async def test_trading_cycle_fallback_to_legacy(self, mock_config, sample_market_data_dict):
        """Test fallback to legacy signal generation when binary integration fails."""
        with patch('core.bot_engine.get_binary_integration') as mock_get_integration, \
             patch('core.bot_engine.DataFetcher') as mock_data_fetcher, \
             patch('core.bot_engine.TimeframeManager') as mock_tf_manager, \
             patch('core.bot_engine.RiskManager') as mock_risk_manager, \
             patch('core.bot_engine.OrderManager') as mock_order_manager, \
             patch('core.bot_engine.SignalRouter') as mock_signal_router, \
             patch('core.bot_engine.DiscordNotifier') as mock_notifier, \
             patch('core.bot_engine.TaskManager') as mock_task_manager:

            # Mock binary integration to return empty (disabled/failed)
            mock_integration = Mock()
            mock_integration.enabled = False
            mock_get_integration.return_value = mock_integration

            # Mock other components
            mock_data_fetcher.return_value.initialize = AsyncMock()
            mock_tf_manager.return_value.initialize = AsyncMock()
            mock_risk_manager.return_value = Mock()
            mock_order_manager.return_value = Mock()
            mock_order_manager.return_value.initialize = AsyncMock()
            mock_order_manager.return_value.configure_portfolio = AsyncMock()
            mock_signal_router.return_value = Mock()
            mock_notifier.return_value = Mock()
            mock_task_manager.return_value = Mock()

            # Create bot engine
            bot = BotEngine(mock_config)

            # Mock the trading cycle components
            bot._fetch_market_data = AsyncMock(return_value=sample_market_data_dict)
            bot._check_safe_mode_conditions = AsyncMock(return_value=False)
            bot._generate_signals = AsyncMock(return_value=[])
            bot._evaluate_risk = AsyncMock(return_value=[])
            bot._execute_orders = AsyncMock()
            bot._update_state = AsyncMock()

            # Run trading cycle
            await bot._trading_cycle()

            # Verify legacy path was taken
            bot._generate_signals.assert_called_once_with(sample_market_data_dict)
            bot._evaluate_risk.assert_called_once()
            bot._execute_orders.assert_called_once()


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_complete_flow_simulation(self):
        """Simulate the complete flow from market data to order execution."""
        # This would be a comprehensive integration test that mocks all components
        # and verifies the complete data flow through the system

        # Create mock market data with sufficient rows for feature extraction
        dates = pd.date_range(start='2023-01-01', periods=50, freq='1H')
        np.random.seed(42)

        data = {
            'open': 100 + np.random.normal(0, 1, 50),
            'high': 105 + np.random.normal(0, 1, 50),
            'low': 95 + np.random.normal(0, 1, 50),
            'close': 100 + np.random.normal(0, 0.5, 50),
            'volume': np.random.uniform(1000, 5000, 50)
        }

        market_data = pd.DataFrame(data, index=dates)

        # Mock binary model
        binary_model = Mock()
        binary_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        # Mock risk manager
        risk_manager = Mock()
        risk_manager.evaluate_signal = AsyncMock(return_value=True)
        risk_manager.calculate_position_size = AsyncMock(return_value=1000.0)
        risk_manager.calculate_dynamic_stop_loss = AsyncMock(return_value=95.0)
        risk_manager.calculate_take_profit = AsyncMock(return_value=110.0)

        # Create integration
        config = {
            "binary_integration": {
                "enabled": True,
                "threshold": 0.6,
                "min_confidence": 0.5
            }
        }

        # Create mock strategy selector
        mock_strategy_selector = Mock()
        mock_strategy = Mock()
        mock_strategy.__name__ = "TestStrategy"
        mock_strategy_selector.select_strategy.return_value = mock_strategy

        integration = BinaryModelIntegration(config, strategy_selector=mock_strategy_selector)
        integration.binary_model = binary_model
        integration.risk_manager = risk_manager

        # Process market data
        decision = await integration.process_market_data(market_data, "BTC/USDT")

        # Verify complete decision
        assert decision.should_trade == True
        assert decision.binary_probability == 0.7
        assert decision.selected_strategy == mock_strategy
        assert decision.direction in ["long", "short"]
        assert decision.position_size == 1000.0
        assert decision.stop_loss == 95.0
        assert decision.take_profit == 110.0
        assert "Binary:" in decision.reasoning
        assert "Strategy:" in decision.reasoning
        assert "Risk:" in decision.reasoning

    @pytest.mark.asyncio
    async def test_error_handling_in_integration(self):
        """Test error handling throughout the integration pipeline."""
        # Create integration with invalid configuration
        config = {
            "binary_integration": {
                "enabled": True,
                "threshold": 2.0,  # Invalid threshold > 1.0
                "min_confidence": 0.5
            }
        }

        integration = BinaryModelIntegration(config)

        # Test with empty market data
        decision = await integration.process_market_data(pd.DataFrame(), "BTC/USDT")

        # Should handle gracefully and return neutral decision
        assert decision.should_trade == False
        assert "Binary integration disabled" in decision.reasoning

    @pytest.mark.asyncio
    async def test_multi_symbol_processing(self):
        """Test processing multiple symbols simultaneously."""
        # Create sample data for multiple symbols with sufficient rows
        symbols_data = {}
        dates = pd.date_range(start='2023-01-01', periods=50, freq='1H')
        np.random.seed(42)

        for i, symbol in enumerate(["BTC/USDT", "ETH/USDT", "ADA/USDT"]):
            data = {
                'open': 100 + i*10 + np.random.normal(0, 1, 50),
                'high': 105 + i*10 + np.random.normal(0, 1, 50),
                'low': 95 + i*10 + np.random.normal(0, 1, 50),
                'close': 100 + i*10 + np.random.normal(0, 0.5, 50),
                'volume': np.random.uniform(1000, 5000, 50)
            }
            symbols_data[symbol] = pd.DataFrame(data, index=dates)

        # Mock components
        binary_model = Mock()
        binary_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        risk_manager = Mock()
        risk_manager.evaluate_signal = AsyncMock(return_value=True)
        risk_manager.calculate_position_size = AsyncMock(return_value=1000.0)
        risk_manager.calculate_dynamic_stop_loss = AsyncMock(return_value=95.0)
        risk_manager.calculate_take_profit = AsyncMock(return_value=110.0)

        # Create integration
        config = {
            "binary_integration": {
                "enabled": True,
                "threshold": 0.6,
                "min_confidence": 0.5
            }
        }

        # Create mock strategy selector
        mock_strategy_selector = Mock()
        mock_strategy = Mock()
        mock_strategy.__name__ = "TestStrategy"
        mock_strategy_selector.select_strategy.return_value = mock_strategy

        integration = BinaryModelIntegration(config, strategy_selector=mock_strategy_selector)
        integration.binary_model = binary_model
        integration.risk_manager = risk_manager

        # Process each symbol
        decisions = []
        for symbol, data in symbols_data.items():
            decision = await integration.process_market_data(data, symbol)
            decisions.append((symbol, decision))

        # Verify all symbols were processed
        assert len(decisions) == 3
        for symbol, decision in decisions:
            assert decision.should_trade == True
            assert decision.binary_probability == 0.7
            assert symbol in ["BTC/USDT", "ETH/USDT", "ADA/USDT"]


if __name__ == "__main__":
    pytest.main([__file__])
