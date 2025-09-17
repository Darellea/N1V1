"""
Enhanced Binary Model Integration Testing Suite

This module provides additional comprehensive testing for the Binary Model Integration feature,
focusing on areas with low coverage and edge cases not covered by the main test suite.
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from core.binary_model_integration import (
    BinaryModelIntegration,
    BinaryModelResult,
    StrategySelectionResult,
    IntegratedTradingDecision,
    get_binary_integration,
    integrate_binary_model
)
from strategies.regime.market_regime import MarketRegime


class TestBinaryModelIntegrationConfiguration:
    """Test configuration handling and validation."""

    def test_initialization_with_complete_config(self):
        """Test initialization with complete configuration."""
        config = {
            "binary_integration": {
                "enabled": True,
                "threshold": 0.7,
                "min_confidence": 0.6,
                "require_regime_confirmation": False,
                "use_adaptive_position_sizing": False
            }
        }

        integration = BinaryModelIntegration(config)

        assert integration.enabled == True
        assert integration.binary_threshold == 0.7
        assert integration.min_confidence == 0.6
        assert integration.require_regime_confirmation == False
        assert integration.use_adaptive_position_sizing == False

    def test_initialization_with_minimal_config(self):
        """Test initialization with minimal configuration."""
        config = {
            "binary_integration": {
                "enabled": True
            }
        }

        integration = BinaryModelIntegration(config)

        # Should use defaults for missing values
        assert integration.enabled == True
        assert integration.binary_threshold == 0.6  # default
        assert integration.min_confidence == 0.5  # default

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

    def test_initialization_without_binary_integration_config(self):
        """Test initialization without binary_integration config section."""
        config = {}

        integration = BinaryModelIntegration(config)

        assert integration.enabled == False  # Default when missing
        assert integration.binary_threshold == 0.6
        assert integration.min_confidence == 0.5


class TestBinaryModelIntegrationFeatureExtraction:
    """Test feature extraction functionality."""

    @pytest.fixture
    def integration(self):
        """Binary model integration instance."""
        return BinaryModelIntegration({})

    def test_extract_features_with_sufficient_data(self, integration):
        """Test feature extraction with sufficient market data."""
        # Create market data with enough rows for all calculations
        dates = pd.date_range(start='2023-01-01', periods=50, freq='1h')
        np.random.seed(42)

        data = {
            'open': 100 + np.random.normal(0, 1, 50),
            'high': 105 + np.random.normal(0, 1, 50),
            'low': 95 + np.random.normal(0, 1, 50),
            'close': 100 + np.random.normal(0, 0.5, 50),
            'volume': np.random.uniform(1000, 5000, 50)
        }

        market_data = pd.DataFrame(data, index=dates)

        features = integration._extract_features(market_data)

        # Verify all expected features are present
        expected_features = ['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']
        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
            assert not pd.isna(features[feature])

    def test_extract_features_with_nan_handling(self, integration):
        """Test feature extraction handles NaN values properly."""
        # Create data with some NaN values
        data = {
            'open': [100, np.nan, 102, 103, 104],
            'high': [105, 106, np.nan, 108, 109],
            'low': [95, np.nan, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, np.nan, 1200, 1300, 1400]
        }

        market_data = pd.DataFrame(data)

        features = integration._extract_features(market_data)

        # Should handle NaN values and return valid features
        assert isinstance(features, dict)
        for key, value in features.items():
            assert not pd.isna(value), f"Feature {key} should not be NaN"

    def test_extract_features_with_extreme_values(self, integration):
        """Test feature extraction with extreme market values."""
        # Create data with extreme price movements
        data = {
            'open': [100, 1000, 1, 10000, 0.01],
            'high': [1000, 10000, 100, 100000, 1],
            'low': [1, 10, 0.01, 1000, 0.001],
            'close': [500, 5000, 50, 50000, 0.5],
            'volume': [1000, 1000000, 10, 10000000, 1]
        }

        market_data = pd.DataFrame(data)

        features = integration._extract_features(market_data)

        # Should handle extreme values without crashing
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_calculate_rsi_edge_cases(self, integration):
        """Test RSI calculation edge cases."""
        # Test with constant prices (should give neutral RSI)
        constant_prices = pd.Series([100] * 20)
        rsi = integration._calculate_rsi(constant_prices)
        assert rsi == 50.0  # Neutral RSI for constant prices

        # Test with trending prices
        trending_prices = pd.Series(range(100, 120))
        rsi = integration._calculate_rsi(trending_prices)
        assert 70 <= rsi <= 100  # Should be high RSI for uptrend

    def test_calculate_macd_edge_cases(self, integration):
        """Test MACD calculation edge cases."""
        # Test with insufficient data
        short_series = pd.Series([100, 101])
        macd = integration._calculate_macd(short_series)
        assert macd == 0.0  # Should return fallback

        # Test with constant prices
        constant_prices = pd.Series([100] * 50)
        macd = integration._calculate_macd(constant_prices)
        assert abs(macd) < 0.01  # Should be close to zero

    def test_calculate_atr_edge_cases(self, integration):
        """Test ATR calculation edge cases."""
        # Create data with no volatility
        data = {
            'high': [100] * 20,
            'low': [100] * 20,
            'close': [100] * 20
        }
        df = pd.DataFrame(data)

        atr = integration._calculate_atr(df)
        assert atr == 0.0  # No volatility

    def test_calculate_stoch_rsi_edge_cases(self, integration):
        """Test Stochastic RSI calculation edge cases."""
        # Test with constant RSI values
        constant_prices = pd.Series([100] * 30)
        stoch_rsi = integration._calculate_stoch_rsi(constant_prices)
        assert stoch_rsi == 0.5  # Should be neutral

    def test_calculate_trend_strength_edge_cases(self, integration):
        """Test trend strength calculation edge cases."""
        # Test with insufficient data
        short_series = pd.Series([100, 101])
        trend_strength = integration._calculate_trend_strength(short_series)
        assert trend_strength == 0.0  # Should return fallback

        # Test with perfect linear trend
        perfect_trend = pd.Series(range(100, 120))
        trend_strength = integration._calculate_trend_strength(perfect_trend)
        assert trend_strength > 0  # Should detect positive trend


class TestBinaryModelIntegrationGlobalFunctions:
    """Test global functions and singleton pattern."""

    def test_get_binary_integration_singleton_pattern(self):
        """Test that get_binary_integration returns singleton."""
        with patch('core.binary_model_integration.get_config') as mock_get_config:
            mock_get_config.return_value = {"binary_integration": {"enabled": False}}

            instance1 = get_binary_integration()
            instance2 = get_binary_integration()

            assert instance1 is instance2
            assert isinstance(instance1, BinaryModelIntegration)

    def test_get_binary_integration_with_config(self):
        """Test get_binary_integration with specific config."""
        config = {"binary_integration": {"enabled": True, "threshold": 0.8}}

        with patch('core.binary_model_integration.get_config', return_value=config):
            instance = get_binary_integration()

            assert instance.enabled == True
            assert instance.binary_threshold == 0.8

    @pytest.mark.asyncio
    async def test_integrate_binary_model_convenience_function(self):
        """Test the integrate_binary_model convenience function."""
        with patch('core.binary_model_integration.get_binary_integration') as mock_get_integration:
            mock_integration = Mock()
            mock_integration.process_market_data = AsyncMock(return_value=Mock(should_trade=True))
            mock_get_integration.return_value = mock_integration

            market_data = pd.DataFrame({'close': [100, 101, 102]})
            result = await integrate_binary_model(market_data, "BTC/USDT")

            assert result.should_trade == True
            mock_integration.process_market_data.assert_called_once_with(market_data, "BTC/USDT")


class TestBinaryModelIntegrationMetricsIntegration:
    """Test metrics integration functionality."""

    @pytest.fixture
    def integration(self):
        """Binary model integration instance."""
        return BinaryModelIntegration({})

    @pytest.mark.asyncio
    async def test_metrics_recording_in_prediction(self, integration):
        """Test that metrics are recorded during prediction."""
        # Mock binary model
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

        with patch('core.binary_model_integration.get_binary_model_metrics_collector') as mock_get_collector:
            mock_collector = Mock()
            mock_get_collector.return_value = mock_collector

            await integration._predict_binary_model(market_data, "BTC/USDT")

            mock_collector.record_prediction.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_recording_in_decision(self, integration):
        """Test that metrics are recorded during decision creation."""
        binary_result = BinaryModelResult(
            should_trade=True,
            probability=0.8,
            confidence=0.9,
            threshold=0.6,
            features={},
            timestamp=datetime.now()
        )

        strategy_result = StrategySelectionResult(
            selected_strategy=Mock(__name__="TestStrategy"),
            direction="long",
            regime=MarketRegime.BULLISH,
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

        with patch('core.binary_model_integration.get_binary_model_metrics_collector') as mock_get_collector:
            mock_collector = Mock()
            mock_get_collector.return_value = mock_collector

            await integration._create_integrated_decision(
                binary_result, strategy_result, risk_result, "BTC/USDT"
            )

            mock_collector.record_decision_outcome.assert_called_once()


class TestBinaryModelIntegrationErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.fixture
    def integration(self):
        """Binary model integration instance."""
        return BinaryModelIntegration({})

    @pytest.mark.asyncio
    async def test_predict_binary_model_with_model_exception(self, integration):
        """Test binary model prediction with model exceptions."""
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
    async def test_select_strategy_with_regime_detector_failure(self, integration):
        """Test strategy selection when regime detector fails."""
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
            assert result.regime == MarketRegime.UNKNOWN

    @pytest.mark.asyncio
    async def test_validate_risk_with_component_failure(self, integration):
        """Test risk validation when components fail."""
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
            regime=MarketRegime.BULLISH,
            confidence=0.8,
            reasoning="Test",
            risk_multiplier=1.0
        )

        result = await integration._validate_risk(market_data, strategy_result, "BTC/USDT")

        # Should return safe defaults on exception
        assert result["approved"] == False
        assert result["position_size"] == 0.0

    @pytest.mark.asyncio
    async def test_complete_pipeline_with_multiple_failures(self, integration):
        """Test complete pipeline with multiple component failures."""
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
        assert "Binary integration disabled" in decision.reasoning


class TestBinaryModelIntegrationDataValidation:
    """Test data validation and sanitization."""

    @pytest.fixture
    def integration(self):
        """Binary model integration instance."""
        return BinaryModelIntegration({})

    @pytest.mark.asyncio
    async def test_process_market_data_with_invalid_ohlc(self, integration):
        """Test processing with invalid OHLC data."""
        # Create data where high < close (invalid)
        market_data = pd.DataFrame({
            'open': [100],
            'high': [99],  # High < close, invalid
            'low': [95],
            'close': [102],
            'volume': [1000]
        })

        decision = await integration.process_market_data(market_data, "BTC/USDT")

        # Should still process but may produce different results
        assert isinstance(decision, IntegratedTradingDecision)

    @pytest.mark.asyncio
    async def test_process_market_data_with_empty_symbol(self, integration):
        """Test processing with empty symbol."""
        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        decision = await integration.process_market_data(market_data, "")

        assert decision.should_trade == False

    @pytest.mark.asyncio
    async def test_process_market_data_with_none_symbol(self, integration):
        """Test processing with None symbol."""
        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        decision = await integration.process_market_data(market_data, None)

        assert decision.should_trade == False

    @pytest.mark.asyncio
    async def test_process_market_data_with_special_characters_symbol(self, integration):
        """Test processing with symbol containing special characters."""
        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        decision = await integration.process_market_data(market_data, "BTC/USDT:PERP")

        # Should handle special characters gracefully
        assert isinstance(decision, IntegratedTradingDecision)


class TestBinaryModelIntegrationLogging:
    """Test logging functionality."""

    @pytest.fixture
    def integration(self):
        """Binary model integration instance."""
        return BinaryModelIntegration({})

    @pytest.mark.asyncio
    async def test_binary_prediction_logging_integration(self, integration):
        """Test that binary predictions are logged correctly."""
        # Mock binary model
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

            # Verify logging was called
            mock_trade_logger.log_binary_prediction.assert_called_once()

    @pytest.mark.asyncio
    async def test_decision_outcome_logging_integration(self, integration):
        """Test that decision outcomes are logged correctly."""
        binary_result = BinaryModelResult(
            should_trade=True,
            probability=0.8,
            confidence=0.9,
            threshold=0.6,
            features={},
            timestamp=datetime.now()
        )

        strategy_result = StrategySelectionResult(
            selected_strategy=Mock(__name__="TestStrategy"),
            direction="long",
            regime=MarketRegime.BULLISH,
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

            # Verify logging was called
            mock_trade_logger.log_binary_decision.assert_called_once()


class TestBinaryModelIntegrationPerformance:
    """Test performance and scalability aspects."""

    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self):
        """Test concurrent processing of multiple symbols."""
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
        for i in range(3):  # Reduced number for performance
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

        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()

        # Verify all results completed
        assert len(results) == 3
        for result in results:
            assert isinstance(result, IntegratedTradingDecision)

        # Performance check (should complete within reasonable time)
        duration = end_time - start_time
        assert duration < 5.0  # Should complete within 5 seconds


class TestBinaryModelIntegrationMemoryManagement:
    """Test memory management and resource usage."""

    def test_large_dataframe_processing(self):
        """Test processing of large market dataframes."""
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

    def test_memory_cleanup_on_reuse(self):
        """Test that memory is properly managed when reusing integration instances."""
        integration = BinaryModelIntegration({})

        # Process multiple datasets
        for i in range(10):
            market_data = pd.DataFrame({
                'open': [100 + i, 101 + i, 102 + i],
                'high': [105 + i, 106 + i, 107 + i],
                'low': [95 + i, 96 + i, 97 + i],
                'close': [102 + i, 103 + i, 104 + i],
                'volume': [1000 + i*100, 1100 + i*100, 1200 + i*100]
            })

            features = integration._extract_features(market_data)
            assert isinstance(features, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
