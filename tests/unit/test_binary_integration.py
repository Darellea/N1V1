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
        dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
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
        dates = pd.date_range(start='2023-01-01', periods=50, freq='h')
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
        dates = pd.date_range(start='2023-01-01', periods=50, freq='h')
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
        dates = pd.date_range(start='2023-01-01', periods=50, freq='h')
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


class TestBinaryModelIntegrationEdgeCases:
    """Test edge cases and error handling in binary model integration."""

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

    @pytest.mark.asyncio
    async def test_binary_model_prediction_with_exact_threshold(self, mock_binary_model, mock_risk_manager, binary_integration_config):
        """Test binary model prediction with probability exactly at threshold."""
        # Mock exact threshold probability
        mock_binary_model.predict_proba.return_value = np.array([[0.4, 0.6]])  # Exactly 60% threshold

        mock_strategy_selector = Mock()
        mock_strategy = Mock()
        mock_strategy.__name__ = "TestStrategy"
        mock_strategy_selector.select_strategy.return_value = mock_strategy

        integration = BinaryModelIntegration(binary_integration_config, strategy_selector=mock_strategy_selector)
        integration.binary_model = mock_binary_model
        integration.risk_manager = mock_risk_manager

        # Create minimal market data
        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        result = await integration._predict_binary_model(market_data, "BTC/USDT")

        assert result.should_trade == True  # Should trade at exact threshold
        assert result.probability == 0.6
        assert result.threshold == 0.6

    @pytest.mark.asyncio
    async def test_binary_model_prediction_with_low_confidence(self, mock_binary_model, mock_risk_manager, binary_integration_config):
        """Test binary model prediction with low confidence."""
        # Mock high probability but low confidence
        mock_binary_model.predict_proba.return_value = np.array([[0.45, 0.55]])  # 55% probability

        mock_strategy_selector = Mock()
        mock_strategy = Mock()
        mock_strategy.__name__ = "TestStrategy"
        mock_strategy_selector.select_strategy.return_value = mock_strategy

        integration = BinaryModelIntegration(binary_integration_config, strategy_selector=mock_strategy_selector)
        integration.binary_model = mock_binary_model
        integration.risk_manager = mock_risk_manager

        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        result = await integration._predict_binary_model(market_data, "BTC/USDT")

        # Should not trade due to low confidence
        assert result.should_trade == False
        assert result.probability == 0.55
        assert result.confidence < 0.5  # Low confidence

    @pytest.mark.asyncio
    async def test_feature_extraction_with_empty_dataframe(self):
        """Test feature extraction with empty dataframe."""
        integration = BinaryModelIntegration({})

        features = integration._extract_features(pd.DataFrame())

        assert features == {}

    @pytest.mark.asyncio
    async def test_feature_extraction_with_insufficient_data(self):
        """Test feature extraction with insufficient data."""
        integration = BinaryModelIntegration({})

        # DataFrame with only 2 rows (insufficient for most calculations)
        market_data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [95, 96],
            'close': [102, 103],
            'volume': [1000, 1100]
        })

        features = integration._extract_features(market_data)

        # Should return features with fallback values
        assert isinstance(features, dict)
        assert 'RSI' in features
        assert 'MACD' in features

    @pytest.mark.asyncio
    async def test_feature_extraction_with_nan_values(self):
        """Test feature extraction handles NaN values properly."""
        integration = BinaryModelIntegration({})

        # DataFrame with NaN values
        market_data = pd.DataFrame({
            'open': [100, np.nan, 102],
            'high': [105, 106, np.nan],
            'low': [95, np.nan, 97],
            'close': [102, 103, 104],
            'volume': [1000, np.nan, 1200]
        })

        features = integration._extract_features(market_data)

        # Should handle NaN values and return valid features
        assert isinstance(features, dict)
        for key, value in features.items():
            assert not pd.isna(value), f"Feature {key} should not be NaN"

    @pytest.mark.asyncio
    async def test_strategy_selection_with_unknown_regime(self):
        """Test strategy selection with unknown regime."""
        integration = BinaryModelIntegration({})

        # Mock regime detector to return unknown regime
        with patch('core.binary_model_integration.get_market_regime_detector') as mock_get_detector:
            mock_detector = Mock()
            mock_regime_result = Mock()
            mock_regime_result.regime_name = "UNKNOWN"
            mock_regime_result.confidence_score = 0.0
            mock_detector.detect_enhanced_regime.return_value = mock_regime_result
            mock_get_detector.return_value = mock_detector

            # Mock strategy selector
            mock_strategy_selector = Mock()
            mock_strategy = Mock()
            mock_strategy.__name__ = "FallbackStrategy"
            mock_strategy_selector.select_strategy.return_value = mock_strategy

            integration.strategy_selector = mock_strategy_selector

            market_data = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [102, 103, 104],
                'volume': [1000, 1100, 1200]
            })

            binary_result = BinaryModelResult(
                should_trade=True,
                probability=0.7,
                confidence=0.8,
                threshold=0.6,
                features={},
                timestamp=datetime.now()
            )

            result = await integration._select_strategy(market_data, binary_result)

            assert result.selected_strategy == mock_strategy
            assert result.regime == "UNKNOWN"
            assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_risk_validation_with_rejection(self, mock_risk_manager):
        """Test risk validation when risk manager rejects."""
        integration = BinaryModelIntegration({})
        integration.risk_manager = mock_risk_manager

        # Mock risk rejection
        mock_risk_manager.evaluate_signal.return_value = False

        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        strategy_result = StrategySelectionResult(
            selected_strategy=Mock(__name__="TestStrategy"),
            direction="long",
            regime="BULLISH",
            confidence=0.8,
            reasoning="Test",
            risk_multiplier=1.0
        )

        result = await integration._validate_risk(market_data, strategy_result, "BTC/USDT")

        assert result["approved"] == False
        assert result["position_size"] == 0.0
        assert result["stop_loss"] is None
        assert result["take_profit"] is None

    @pytest.mark.asyncio
    async def test_risk_validation_with_exception(self):
        """Test risk validation when exceptions occur."""
        integration = BinaryModelIntegration({})

        # Mock risk manager to raise exception
        mock_risk_manager = Mock()
        mock_risk_manager.evaluate_signal = AsyncMock(side_effect=Exception("Risk evaluation failed"))
        integration.risk_manager = mock_risk_manager

        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        strategy_result = StrategySelectionResult(
            selected_strategy=Mock(__name__="TestStrategy"),
            direction="long",
            regime="BULLISH",
            confidence=0.8,
            reasoning="Test",
            risk_multiplier=1.0
        )

        result = await integration._validate_risk(market_data, strategy_result, "BTC/USDT")

        # Should return safe defaults on exception
        assert result["approved"] == False
        assert result["position_size"] == 0.0

    @pytest.mark.asyncio
    async def test_create_integrated_decision_with_all_components(self):
        """Test creating integrated decision with all components working."""
        integration = BinaryModelIntegration({})

        binary_result = BinaryModelResult(
            should_trade=True,
            probability=0.8,
            confidence=0.9,
            threshold=0.6,
            features={"RSI": 65.0},
            timestamp=datetime.now()
        )

        strategy_result = StrategySelectionResult(
            selected_strategy=Mock(__name__="TestStrategy"),
            direction="long",
            regime="BULLISH",
            confidence=0.8,
            reasoning="Strong bullish signal",
            risk_multiplier=1.2
        )

        risk_result = {
            "approved": True,
            "position_size": 1500.0,
            "stop_loss": 98.0,
            "take_profit": 115.0,
            "risk_score": 1.2
        }

        decision = await integration._create_integrated_decision(
            binary_result, strategy_result, risk_result, "BTC/USDT"
        )

        assert decision.should_trade == True
        assert decision.binary_probability == 0.8
        assert decision.direction == "long"
        assert decision.position_size == 1500.0
        assert decision.stop_loss == 98.0
        assert decision.take_profit == 115.0
        assert "Binary:" in decision.reasoning
        assert "Strategy:" in decision.reasoning
        assert "Regime:" in decision.reasoning
        assert "Risk:" in decision.reasoning

    @pytest.mark.asyncio
    async def test_create_integrated_decision_with_rejection(self):
        """Test creating integrated decision when components reject."""
        integration = BinaryModelIntegration({})

        binary_result = BinaryModelResult(
            should_trade=False,
            probability=0.3,
            confidence=0.2,
            threshold=0.6,
            features={},
            timestamp=datetime.now()
        )

        strategy_result = StrategySelectionResult(
            selected_strategy=None,
            direction="neutral",
            regime="UNKNOWN",
            confidence=0.0,
            reasoning="No strategy selected",
            risk_multiplier=1.0
        )

        risk_result = {
            "approved": False,
            "position_size": 0.0,
            "stop_loss": None,
            "take_profit": None,
            "risk_score": 1.0
        }

        decision = await integration._create_integrated_decision(
            binary_result, strategy_result, risk_result, "BTC/USDT"
        )

        assert decision.should_trade == False
        assert decision.selected_strategy is None
        assert decision.position_size == 0.0

    @pytest.mark.asyncio
    async def test_process_market_data_with_disabled_integration(self):
        """Test processing market data when integration is disabled."""
        config = {
            "binary_integration": {
                "enabled": False,
                "threshold": 0.6,
                "min_confidence": 0.5
            }
        }

        integration = BinaryModelIntegration(config)

        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        decision = await integration.process_market_data(market_data, "BTC/USDT")

        assert decision.should_trade == False
        assert "Binary integration disabled" in decision.reasoning

    @pytest.mark.asyncio
    async def test_process_market_data_without_binary_model(self):
        """Test processing market data without binary model initialized."""
        config = {
            "binary_integration": {
                "enabled": True,
                "threshold": 0.6,
                "min_confidence": 0.5
            }
        }

        integration = BinaryModelIntegration(config)
        # Don't initialize binary_model

        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        decision = await integration.process_market_data(market_data, "BTC/USDT")

        assert decision.should_trade == False
        assert "Binary integration disabled" in decision.reasoning


class TestTechnicalIndicators:
    """Test technical indicator calculations."""

    def test_calculate_rsi_with_valid_data(self):
        """Test RSI calculation with valid data."""
        integration = BinaryModelIntegration({})

        # Create test data with known RSI
        series = pd.Series([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65])

        rsi = integration._calculate_rsi(series)

        assert isinstance(rsi, float)
        assert 0 <= rsi <= 100

    def test_calculate_rsi_with_insufficient_data(self):
        """Test RSI calculation with insufficient data."""
        integration = BinaryModelIntegration({})

        series = pd.Series([50, 51])  # Less than period (14)

        rsi = integration._calculate_rsi(series)

        assert rsi == 50.0  # Should return fallback value

    def test_calculate_macd_with_valid_data(self):
        """Test MACD calculation with valid data."""
        integration = BinaryModelIntegration({})

        series = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                           111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                           121, 122, 123, 124, 125, 126, 127, 128, 129, 130])

        macd = integration._calculate_macd(series)

        assert isinstance(macd, float)
        assert not pd.isna(macd)

    def test_calculate_atr_with_valid_data(self):
        """Test ATR calculation with valid data."""
        integration = BinaryModelIntegration({})

        df = pd.DataFrame({
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
        })

        atr = integration._calculate_atr(df)

        assert isinstance(atr, float)
        assert atr >= 0

    def test_calculate_stoch_rsi_with_valid_data(self):
        """Test Stochastic RSI calculation with valid data."""
        integration = BinaryModelIntegration({})

        series = pd.Series([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                           66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80])

        stoch_rsi = integration._calculate_stoch_rsi(series)

        assert isinstance(stoch_rsi, float)
        assert 0 <= stoch_rsi <= 1

    def test_calculate_trend_strength_with_valid_data(self):
        """Test trend strength calculation with valid data."""
        integration = BinaryModelIntegration({})

        series = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                           111, 112, 113, 114, 115, 116, 117, 118, 119, 120])

        trend_strength = integration._calculate_trend_strength(series)

        assert isinstance(trend_strength, float)


class TestGlobalIntegrationFunctions:
    """Test global integration utility functions."""

    def test_get_binary_integration_singleton(self):
        """Test that get_binary_integration returns singleton."""
        with patch('core.binary_model_integration.get_config') as mock_get_config:
            mock_get_config.return_value = {"binary_integration": {"enabled": False}}

            integration1 = get_binary_integration()
            integration2 = get_binary_integration()

            assert integration1 is integration2

    @pytest.mark.asyncio
    async def test_integrate_binary_model_convenience_function(self):
        """Test integrate_binary_model convenience function."""
        with patch('core.binary_model_integration.get_binary_integration') as mock_get_integration:
            mock_integration = Mock()
            mock_integration.process_market_data = AsyncMock(return_value=Mock(should_trade=True))
            mock_get_integration.return_value = mock_integration

            market_data = pd.DataFrame({'close': [100, 101, 102]})
            result = await integrate_binary_model(market_data, "BTC/USDT")

            assert result.should_trade == True
            mock_integration.process_market_data.assert_called_once()


class TestConfigurationHandling:
    """Test configuration handling and validation."""

    def test_initialization_with_missing_config(self):
        """Test initialization with missing configuration."""
        integration = BinaryModelIntegration({})

        assert integration.enabled == False  # Default when missing
        assert integration.binary_threshold == 0.6
        assert integration.min_confidence == 0.5

    def test_initialization_with_partial_config(self):
        """Test initialization with partial configuration."""
        config = {
            "binary_integration": {
                "enabled": True,
                "threshold": 0.7
                # Missing min_confidence
            }
        }

        integration = BinaryModelIntegration(config)

        assert integration.enabled == True
        assert integration.binary_threshold == 0.7
        assert integration.min_confidence == 0.5  # Default value

    def test_initialization_with_invalid_config_values(self):
        """Test initialization with invalid configuration values."""
        config = {
            "binary_integration": {
                "enabled": True,
                "threshold": 1.5,  # Invalid: > 1.0
                "min_confidence": -0.1  # Invalid: < 0
            }
        }

        integration = BinaryModelIntegration(config)

        # Should accept invalid values (no validation in constructor)
        assert integration.binary_threshold == 1.5
        assert integration.min_confidence == -0.1


class TestErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.mark.asyncio
    async def test_binary_model_prediction_error_recovery(self):
        """Test error recovery in binary model prediction."""
        integration = BinaryModelIntegration({})

        # Mock binary model to raise exception
        mock_model = Mock()
        mock_model.predict_proba.side_effect = Exception("Model prediction failed")
        integration.binary_model = mock_model

        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        result = await integration._predict_binary_model(market_data, "BTC/USDT")

        # Should return safe error result
        assert result.should_trade == False
        assert result.probability == 0.0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_strategy_selection_error_recovery(self):
        """Test error recovery in strategy selection."""
        integration = BinaryModelIntegration({})

        # Mock strategy selector to raise exception
        with patch('core.binary_model_integration.get_market_regime_detector', side_effect=Exception("Regime detection failed")):
            market_data = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [102, 103, 104],
                'volume': [1000, 1100, 1200]
            })

            binary_result = BinaryModelResult(
                should_trade=True,
                probability=0.7,
                confidence=0.8,
                threshold=0.6,
                features={},
                timestamp=datetime.now()
            )

            result = await integration._select_strategy(market_data, binary_result)

            # Should return safe error result
            assert result.selected_strategy is None
            assert result.direction == "neutral"
            assert result.regime == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_complete_pipeline_error_recovery(self):
        """Test error recovery in complete pipeline."""
        config = {
            "binary_integration": {
                "enabled": True,
                "threshold": 0.6,
                "min_confidence": 0.5
            }
        }

        integration = BinaryModelIntegration(config)

        # Mock all components to fail
        mock_model = Mock()
        mock_model.predict_proba.side_effect = Exception("Model failed")
        integration.binary_model = mock_model

        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        decision = await integration.process_market_data(market_data, "BTC/USDT")

        # Should handle all errors gracefully
        assert decision.should_trade == False
        assert "Integration error" in decision.reasoning


class TestPerformanceAndScalability:
    """Test performance and scalability aspects."""

    @pytest.mark.asyncio
    async def test_concurrent_market_data_processing(self):
        """Test concurrent processing of multiple market data streams."""
        config = {
            "binary_integration": {
                "enabled": True,
                "threshold": 0.6,
                "min_confidence": 0.5
            }
        }

        # Create mock components
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        mock_risk_manager = Mock()
        mock_risk_manager.evaluate_signal = AsyncMock(return_value=True)
        mock_risk_manager.calculate_position_size = AsyncMock(return_value=1000.0)
        mock_risk_manager.calculate_dynamic_stop_loss = AsyncMock(return_value=95.0)
        mock_risk_manager.calculate_take_profit = AsyncMock(return_value=110.0)

        mock_strategy_selector = Mock()
        mock_strategy = Mock()
        mock_strategy.__name__ = "TestStrategy"
        mock_strategy_selector.select_strategy.return_value = mock_strategy

        # Create multiple integration instances
        integrations = []
        for i in range(5):
            integration = BinaryModelIntegration(config, strategy_selector=mock_strategy_selector)
            integration.binary_model = mock_model
            integration.risk_manager = mock_risk_manager
            integrations.append(integration)

        # Create test market data
        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        # Process concurrently
        tasks = [
            integration.process_market_data(market_data, f"BTC/USDT_{i}")
            for i, integration in enumerate(integrations)
        ]

        results = await asyncio.gather(*tasks)

        # Verify all results
        assert len(results) == 5
        for result in results:
            assert result.should_trade == True
            assert result.binary_probability == 0.7

    def test_memory_usage_with_large_datasets(self):
        """Test memory usage with large datasets."""
        integration = BinaryModelIntegration({})

        # Create large market data
        large_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 10000),
            'high': np.random.normal(105, 5, 10000),
            'low': np.random.normal(95, 5, 10000),
            'close': np.random.normal(100, 5, 10000),
            'volume': np.random.uniform(1000, 5000, 10000)
        })

        # Should handle large datasets without memory issues
        features = integration._extract_features(large_data)

        assert isinstance(features, dict)
        assert len(features) > 0


class TestLoggingAndMonitoring:
    """Test logging and monitoring functionality."""

    @pytest.mark.asyncio
    async def test_binary_prediction_logging(self):
        """Test that binary predictions are logged."""
        integration = BinaryModelIntegration({})

        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        integration.binary_model = mock_model

        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        with patch('core.binary_model_integration.trade_logger') as mock_trade_logger:
            await integration._predict_binary_model(market_data, "BTC/USDT")

            mock_trade_logger.log_binary_prediction.assert_called_once()

    @pytest.mark.asyncio
    async def test_decision_outcome_logging(self):
        """Test that decision outcomes are logged."""
        integration = BinaryModelIntegration({})

        binary_result = BinaryModelResult(
            should_trade=True,
            probability=0.7,
            confidence=0.8,
            threshold=0.6,
            features={},
            timestamp=datetime.now()
        )

        strategy_result = StrategySelectionResult(
            selected_strategy=Mock(__name__="TestStrategy"),
            direction="long",
            regime="BULLISH",
            confidence=0.8,
            reasoning="Test",
            risk_multiplier=1.0
        )

        risk_result = {
            "approved": True,
            "position_size": 1000.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "risk_score": 1.0
        }

        with patch('core.binary_model_integration.trade_logger') as mock_trade_logger:
            await integration._create_integrated_decision(
                binary_result, strategy_result, risk_result, "BTC/USDT"
            )

            mock_trade_logger.log_binary_decision.assert_called_once()


class TestDataValidation:
    """Test data validation and sanitization."""

    @pytest.mark.asyncio
    async def test_market_data_validation(self):
        """Test market data validation."""
        integration = BinaryModelIntegration({})

        # Test with invalid OHLC data (high < close)
        invalid_data = pd.DataFrame({
            'open': [100],
            'high': [99],  # High < close, invalid
            'low': [95],
            'close': [102],
            'volume': [1000]
        })

        # Should still process but may produce NaN features
        features = integration._extract_features(invalid_data)

        assert isinstance(features, dict)

    @pytest.mark.asyncio
    async def test_empty_symbol_handling(self):
        """Test handling of empty or invalid symbols."""
        config = {
            "binary_integration": {
                "enabled": True,
                "threshold": 0.6,
                "min_confidence": 0.5
            }
        }

        integration = BinaryModelIntegration(config)

        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        # Test with empty symbol
        decision = await integration.process_market_data(market_data, "")

        assert decision.should_trade == False

    @pytest.mark.asyncio
    async def test_none_symbol_handling(self):
        """Test handling of None symbol."""
        config = {
            "binary_integration": {
                "enabled": True,
                "threshold": 0.6,
                "min_confidence": 0.5
            }
        }

        integration = BinaryModelIntegration(config)

        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        # Test with None symbol
        decision = await integration.process_market_data(market_data, None)

        assert decision.should_trade == False


if __name__ == "__main__":
    pytest.main([__file__])
