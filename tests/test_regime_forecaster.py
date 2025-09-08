"""
Unit and integration tests for Predictive Regime Forecasting feature.

Tests cover:
- Regime forecaster model validation and training
- Feature engineering pipeline
- Prediction accuracy and confidence calibration
- Integration with strategy selector
- Performance benchmarking
- Edge cases and error handling
- Model versioning and updates
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import tempfile
import os

from strategies.regime.regime_forecaster import RegimeForecaster
from strategies.regime.market_regime import MarketRegimeDetector
from strategies.regime.strategy_selector import StrategySelector
from core.diagnostics import HealthStatus


class TestRegimeForecasterInitialization:
    """Test RegimeForecaster initialization and configuration."""

    def test_initialization_with_config(self, test_config, temp_dir):
        """Test forecaster initialization with configuration."""
        config = test_config.get("regime_forecasting", {})
        config["model_path"] = temp_dir

        forecaster = RegimeForecaster(config)

        assert forecaster.config == config
        assert forecaster.model_path == temp_dir
        assert forecaster.forecast_horizon == config.get("forecast_horizon", 24)
        assert forecaster.confidence_threshold == config.get("confidence_threshold", 0.7)
        assert forecaster.is_initialized is False

    @pytest.mark.asyncio
    async def test_async_initialization(self, test_config, temp_dir):
        """Test asynchronous initialization."""
        config = test_config.get("regime_forecasting", {})
        config["model_path"] = temp_dir

        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        assert forecaster.is_initialized is True
        # Model should be loaded or created
        assert hasattr(forecaster, 'model')

    def test_initialization_without_config(self):
        """Test initialization with default configuration."""
        forecaster = RegimeForecaster({})

        assert forecaster.forecast_horizon == 24
        assert forecaster.confidence_threshold == 0.7
        assert forecaster.is_initialized is False


class TestFeatureEngineering:
    """Test feature engineering pipeline."""

    def test_feature_engineering_basic(self, generate_regime_data):
        """Test basic feature engineering."""
        # Generate bull market data
        data = generate_regime_data("bull_market", n_points=100)

        forecaster = RegimeForecaster({})

        # Test feature extraction
        features = forecaster._extract_features(data)

        assert isinstance(features, dict)
        assert len(features) > 0

        # Should have basic features
        expected_features = ['returns', 'volatility', 'trend_strength', 'volume_trend']
        for feature in expected_features:
            assert feature in features or any(feature in key for key in features.keys())

    def test_feature_engineering_different_regimes(self, generate_regime_data):
        """Test feature engineering across different market regimes."""
        regimes = ["bull_market", "bear_market", "sideways", "high_volatility"]

        forecaster = RegimeForecaster({})

        for regime in regimes:
            data = generate_regime_data(regime, n_points=50)

            features = forecaster._extract_features(data)

            assert isinstance(features, dict)
            assert len(features) > 0

            # Features should be numeric
            for key, value in features.items():
                assert isinstance(value, (int, float, np.number))

    def test_feature_engineering_edge_cases(self):
        """Test feature engineering with edge cases."""
        forecaster = RegimeForecaster({})

        # Empty data
        empty_data = pd.DataFrame()
        features = forecaster._extract_features(empty_data)
        assert isinstance(features, dict)  # Should handle gracefully

        # Single row data
        single_row = pd.DataFrame({
            'open': [50000.0],
            'high': [51000.0],
            'low': [49000.0],
            'close': [50500.0],
            'volume': [1000.0]
        })
        features = forecaster._extract_features(single_row)
        assert isinstance(features, dict)

        # Data with NaN values
        nan_data = pd.DataFrame({
            'open': [50000.0, np.nan, 51000.0],
            'high': [51000.0, 52000.0, np.nan],
            'low': [49000.0, 48000.0, 50000.0],
            'close': [50500.0, 49500.0, 50500.0],
            'volume': [1000.0, np.nan, 1200.0]
        })
        features = forecaster._extract_features(nan_data)
        assert isinstance(features, dict)

    def test_technical_indicators_calculation(self, generate_regime_data):
        """Test technical indicators calculation."""
        data = generate_regime_data("bull_market", n_points=100)

        forecaster = RegimeForecaster({})

        # Test RSI calculation
        rsi = forecaster._calculate_rsi(data['close'], period=14)
        assert len(rsi) == len(data)
        assert all(0 <= r <= 100 for r in rsi.dropna())

        # Test moving averages
        sma_20 = forecaster._calculate_sma(data['close'], period=20)
        sma_50 = forecaster._calculate_sma(data['close'], period=50)
        assert len(sma_20) == len(data)
        assert len(sma_50) == len(data)

        # Test Bollinger Bands
        bb_upper, bb_middle, bb_lower = forecaster._calculate_bollinger_bands(data['close'])
        assert len(bb_upper) == len(data)
        assert len(bb_middle) == len(data)
        assert len(bb_lower) == len(data)

        # Upper should be above middle, middle above lower
        valid_bb = bb_upper.dropna() > bb_middle.dropna()
        assert all(valid_bb[bb_middle.dropna() > bb_lower.dropna()])


class TestModelTraining:
    """Test model training and validation."""

    def test_model_initialization(self, test_config, temp_dir):
        """Test model initialization."""
        config = test_config.get("regime_forecasting", {})
        config["model_path"] = temp_dir

        forecaster = RegimeForecaster(config)

        # Model should be initialized
        assert forecaster.model is not None

    def test_training_data_preparation(self, generate_regime_data):
        """Test training data preparation."""
        # Generate training data for multiple regimes
        regimes = ["bull_market", "bear_market", "sideways"]
        training_data = []

        for regime in regimes:
            data = generate_regime_data(regime, n_points=200)
            training_data.append((data, regime))

        forecaster = RegimeForecaster({})

        X, y = forecaster._prepare_training_data(training_data)

        assert X.shape[0] > 0
        assert y.shape[0] > 0
        assert X.shape[0] == y.shape[0]

        # Should have features
        assert X.shape[1] > 0

        # Labels should be valid regime names
        unique_labels = np.unique(y)
        for label in unique_labels:
            assert label in regimes

    @pytest.mark.asyncio
    async def test_model_training(self, generate_regime_data, temp_dir):
        """Test model training process."""
        # Generate training data
        training_data = []
        regimes = ["bull_market", "bear_market", "sideways"]

        for regime in regimes:
            data = generate_regime_data(regime, n_points=100)
            training_data.append((data, regime))

        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Train model
        await forecaster._train_model(training_data)

        # Model should be trained
        assert forecaster.is_trained is True

    @pytest.mark.asyncio
    async def test_model_persistence(self, generate_regime_data, temp_dir):
        """Test model saving and loading."""
        # Generate training data
        training_data = []
        regimes = ["bull_market", "bear_market"]

        for regime in regimes:
            data = generate_regime_data(regime, n_points=50)
            training_data.append((data, regime))

        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Train and save model
        await forecaster._train_model(training_data)
        await forecaster._save_model()

        # Create new forecaster and load model
        forecaster2 = RegimeForecaster(config)
        await forecaster2.initialize()

        # Should load the trained model
        assert forecaster2.is_trained is True


class TestPrediction:
    """Test prediction functionality."""

    @pytest.mark.asyncio
    async def test_regime_prediction(self, generate_regime_data, temp_dir):
        """Test regime prediction."""
        # Generate training data
        training_data = []
        regimes = ["bull_market", "bear_market", "sideways"]

        for regime in regimes:
            data = generate_regime_data(regime, n_points=100)
            training_data.append((data, regime))

        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Train model
        await forecaster._train_model(training_data)

        # Test prediction on training data
        for data, expected_regime in training_data:
            prediction = await forecaster.predict_regime(data)

            assert isinstance(prediction, dict)
            assert 'predicted_regime' in prediction
            assert 'confidence' in prediction
            assert prediction['predicted_regime'] in regimes
            assert 0.0 <= prediction['confidence'] <= 1.0

    @pytest.mark.asyncio
    async def test_prediction_confidence_calibration(self, generate_regime_data, temp_dir):
        """Test prediction confidence calibration."""
        # Generate clear bull market data
        bull_data = generate_regime_data("bull_market", n_points=200)

        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Train with mixed data
        training_data = [
            (generate_regime_data("bull_market", 100), "bull_market"),
            (generate_regime_data("bear_market", 100), "bear_market"),
            (generate_regime_data("sideways", 100), "sideways")
        ]
        await forecaster._train_model(training_data)

        # Predict on clear bull market data
        prediction = await forecaster.predict_regime(bull_data)

        # Should have high confidence for clear regime
        assert prediction['confidence'] > 0.5

    @pytest.mark.asyncio
    async def test_prediction_edge_cases(self, temp_dir):
        """Test prediction with edge cases."""
        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Empty data
        empty_data = pd.DataFrame()
        prediction = await forecaster.predict_regime(empty_data)
        assert isinstance(prediction, dict)

        # Single row data
        single_row = pd.DataFrame({
            'open': [50000.0],
            'high': [51000.0],
            'low': [49000.0],
            'close': [50500.0],
            'volume': [1000.0]
        })
        prediction = await forecaster.predict_regime(single_row)
        assert isinstance(prediction, dict)

    @pytest.mark.asyncio
    async def test_forecast_horizon(self, generate_regime_data, temp_dir):
        """Test forecast horizon functionality."""
        data = generate_regime_data("bull_market", n_points=100)

        config = {"model_path": temp_dir, "forecast_horizon": 48}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        prediction = await forecaster.predict_regime(data)

        assert 'forecast_horizon' in prediction
        assert prediction['forecast_horizon'] == 48


class TestIntegrationWithStrategySelector:
    """Test integration with Strategy Selector."""

    @pytest.mark.asyncio
    async def test_strategy_selector_integration(self, generate_regime_data, temp_dir):
        """Test integration with strategy selector."""
        # Setup forecaster
        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Train with some data
        training_data = [
            (generate_regime_data("bull_market", 50), "bull_market"),
            (generate_regime_data("bear_market", 50), "bear_market")
        ]
        await forecaster._train_model(training_data)

        # Setup strategy selector
        selector = StrategySelector({})

        # Mock market data
        market_data = generate_regime_data("bull_market", 50)

        # Get regime forecast
        forecast = await forecaster.predict_regime(market_data)

        # Strategy selector should use forecast
        # (This would be tested in integration tests with actual strategies)
        assert forecast is not None
        assert forecast['predicted_regime'] in ['bull_market', 'bear_market', 'sideways']

    @pytest.mark.asyncio
    async def test_forecast_driven_strategy_selection(self, generate_regime_data, temp_dir):
        """Test forecast-driven strategy selection."""
        # Setup forecaster
        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Train model
        training_data = [
            (generate_regime_data("bull_market", 50), "bull_market"),
            (generate_regime_data("bear_market", 50), "bear_market"),
            (generate_regime_data("sideways", 50), "sideways")
        ]
        await forecaster._train_model(training_data)

        # Test different market conditions
        test_cases = [
            ("bull_market", "bull_market"),
            ("bear_market", "bear_market"),
            ("sideways", "sideways")
        ]

        for regime_type, expected_regime in test_cases:
            test_data = generate_regime_data(regime_type, 30)
            forecast = await forecaster.predict_regime(test_data)

            # Forecast should match expected regime for clear cases
            assert forecast['predicted_regime'] == expected_regime
            assert forecast['confidence'] > 0.5


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_prediction_performance(self, generate_regime_data, temp_dir, performance_timer):
        """Test prediction performance."""
        data = generate_regime_data("bull_market", n_points=100)

        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Train model
        training_data = [(generate_regime_data("bull_market", 50), "bull_market")]
        await forecaster._train_model(training_data)

        # Measure prediction time
        performance_timer.start()
        prediction = await forecaster.predict_regime(data)
        performance_timer.stop()

        # Should be fast (< 50ms as per requirements)
        duration_ms = performance_timer.duration_ms()
        assert duration_ms < 50
        assert prediction is not None

    @pytest.mark.asyncio
    async def test_memory_usage(self, generate_regime_data, temp_dir, memory_monitor):
        """Test memory usage during prediction."""
        data = generate_regime_data("bull_market", n_points=200)

        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        memory_monitor.start()

        # Make prediction
        prediction = await forecaster.predict_regime(data)

        memory_delta = memory_monitor.get_memory_delta()

        # Memory usage should be reasonable
        assert memory_delta < 50  # Less than 50MB increase
        assert prediction is not None

    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, generate_regime_data, temp_dir):
        """Test concurrent prediction handling."""
        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Train model
        training_data = [(generate_regime_data("bull_market", 50), "bull_market")]
        await forecaster._train_model(training_data)

        # Generate multiple prediction tasks
        tasks = []
        for i in range(5):
            data = generate_regime_data("bull_market", 30)
            task = forecaster.predict_regime(data)
            tasks.append(task)

        # Execute concurrently
        predictions = await asyncio.gather(*tasks)

        # All predictions should succeed
        assert len(predictions) == 5
        assert all(isinstance(p, dict) for p in predictions)
        assert all('predicted_regime' in p for p in predictions)


class TestModelUpdates:
    """Test model update and versioning."""

    @pytest.mark.asyncio
    async def test_model_update(self, generate_regime_data, temp_dir):
        """Test model update with new data."""
        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Initial training
        initial_data = [(generate_regime_data("bull_market", 50), "bull_market")]
        await forecaster._train_model(initial_data)

        initial_accuracy = forecaster.get_forecast_accuracy()

        # Update with more data
        update_data = [
            (generate_regime_data("bull_market", 30), "bull_market"),
            (generate_regime_data("bear_market", 30), "bear_market")
        ]
        await forecaster.update_model(update_data)

        updated_accuracy = forecaster.get_forecast_accuracy()

        # Accuracy should be tracked
        assert isinstance(initial_accuracy, (int, float))
        assert isinstance(updated_accuracy, (int, float))

    @pytest.mark.asyncio
    async def test_model_versioning(self, generate_regime_data, temp_dir):
        """Test model versioning."""
        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Train initial model
        training_data = [(generate_regime_data("bull_market", 50), "bull_market")]
        await forecaster._train_model(training_data)
        await forecaster._save_model()

        # Check model file exists
        model_files = list(Path(temp_dir).glob("*.pkl"))
        assert len(model_files) > 0

        # Create new forecaster and verify it loads the model
        forecaster2 = RegimeForecaster(config)
        await forecaster2.initialize()

        assert forecaster2.is_trained is True


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_prediction_with_corrupted_data(self, temp_dir):
        """Test prediction with corrupted data."""
        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Data with extreme values
        corrupted_data = pd.DataFrame({
            'open': [50000.0, 1e10, 50000.0],  # Extreme value
            'high': [51000.0, 52000.0, 51000.0],
            'low': [49000.0, 48000.0, 49000.0],
            'close': [50500.0, 49500.0, 50500.0],
            'volume': [1000.0, 1e15, 1200.0]  # Extreme volume
        })

        # Should handle gracefully
        prediction = await forecaster.predict_regime(corrupted_data)
        assert isinstance(prediction, dict)

    @pytest.mark.asyncio
    async def test_model_loading_failure(self, temp_dir):
        """Test handling of model loading failures."""
        config = {"model_path": "/nonexistent/path"}
        forecaster = RegimeForecaster(config)

        # Should handle missing model gracefully
        await forecaster.initialize()
        assert forecaster.is_initialized is True

    @pytest.mark.asyncio
    async def test_empty_training_data(self, temp_dir):
        """Test handling of empty training data."""
        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Empty training data
        empty_training = []

        # Should handle gracefully
        await forecaster._train_model(empty_training)
        assert forecaster.is_trained is False

    @pytest.mark.asyncio
    async def test_insufficient_data_for_training(self, temp_dir):
        """Test handling of insufficient data for training."""
        config = {"model_path": temp_dir}
        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        # Very small training dataset
        small_training = [
            (pd.DataFrame({'open': [50000.0], 'high': [51000.0],
                          'low': [49000.0], 'close': [50500.0], 'volume': [1000.0]}),
             "bull_market")
        ]

        # Should handle gracefully
        await forecaster._train_model(small_training)
        # Training may succeed or fail depending on implementation
        # but should not crash


class TestHealthMonitoring:
    """Test health monitoring integration."""

    @pytest.mark.asyncio
    async def test_health_check_integration(self, test_config, temp_dir):
        """Test integration with health monitoring system."""
        from core.diagnostics import get_diagnostics_manager

        config = test_config.get("regime_forecasting", {})
        config["model_path"] = temp_dir

        forecaster = RegimeForecaster(config)
        await forecaster.initialize()

        diagnostics = get_diagnostics_manager()

        # Register health check
        async def check_regime_forecaster():
            try:
                is_trained = forecaster.is_trained
                model_age = forecaster.get_model_age_hours()
                last_accuracy = forecaster.get_forecast_accuracy()

                status = HealthStatus.HEALTHY if is_trained else HealthStatus.DEGRADED

                return {
                    'component': 'regime_forecaster',
                    'status': status,
                    'latency_ms': 10.0,
                    'message': f'Forecaster healthy: trained={is_trained}, accuracy={last_accuracy:.2f}',
                    'details': {
                        'is_trained': is_trained,
                        'model_age_hours': model_age,
                        'last_accuracy': last_accuracy
                    }
                }
            except Exception as e:
                return {
                    'component': 'regime_forecaster',
                    'status': HealthStatus.CRITICAL,
                    'message': f'Health check failed: {str(e)}',
                    'details': {'error': str(e)}
                }

        diagnostics.register_health_check('regime_forecaster', check_regime_forecaster)

        # Run health check
        state = await diagnostics.run_health_check()

        # Should have forecaster health data
        assert 'regime_forecaster' in state.component_statuses
        rf_status = state.component_statuses['regime_forecaster']

        assert rf_status.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert 'Forecaster healthy:' in rf_status.message
