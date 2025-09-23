"""
Unit tests for predictive models.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from predictive_models import (
    PricePredictor,
    VolatilityPredictor,
    VolumePredictor,
    PredictiveModelManager,
    PredictionContext
)


class TestPredictionContext:
    """Test PredictionContext dataclass."""

    def test_valid_prediction_context(self):
        """Test creating a valid PredictionContext."""
        context = PredictionContext(
            price_direction="up",
            volatility="low",
            volume_surge=True,
            confidence=0.8
        )
        assert context.price_direction == "up"
        assert context.volatility == "low"
        assert context.volume_surge is True
        assert context.confidence == 0.8

    def test_invalid_price_direction(self):
        """Test invalid price direction raises ValueError."""
        with pytest.raises(ValueError):
            PredictionContext(price_direction="invalid")

    def test_invalid_volatility(self):
        """Test invalid volatility raises ValueError."""
        with pytest.raises(ValueError):
            PredictionContext(volatility="invalid")

    def test_invalid_confidence(self):
        """Test invalid confidence raises ValueError."""
        with pytest.raises(ValueError):
            PredictionContext(confidence=1.5)

    def test_to_dict(self):
        """Test converting PredictionContext to dict."""
        context = PredictionContext(
            price_direction="up",
            volatility="high",
            volume_surge=False,
            confidence=0.7
        )
        data = context.to_dict()
        assert data["price_direction"] == "up"
        assert data["volatility"] == "high"
        assert data["volume_surge"] is False
        assert data["confidence"] == 0.7

    def test_from_dict(self):
        """Test creating PredictionContext from dict."""
        data = {
            "price_direction": "down",
            "volatility": "low",
            "volume_surge": True,
            "confidence": 0.6
        }
        context = PredictionContext.from_dict(data)
        assert context.price_direction == "down"
        assert context.volatility == "low"
        assert context.volume_surge is True
        assert context.confidence == 0.6


class TestPricePredictor:
    """Test PricePredictor class."""

    def test_initialization(self):
        """Test PricePredictor initialization."""
        config = {
            "type": "lightgbm",
            "confidence_threshold": 0.6,
            "lookback": 50
        }
        predictor = PricePredictor(config)
        assert predictor.model_type == "lightgbm"
        assert predictor.confidence_threshold == 0.6
        assert predictor.lookback == 50

    def test_create_features(self):
        """Test feature creation."""
        config = {"type": "lightgbm"}
        predictor = PricePredictor(config)

        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        data = {
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }
        df = pd.DataFrame(data)

        features = predictor._create_features(df)
        assert not features.empty
        assert len(features.columns) > 0

    def test_predict_without_model(self):
        """Test prediction without trained model returns neutral."""
        config = {"type": "lightgbm"}
        predictor = PricePredictor(config)

        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })

        direction, confidence = predictor.predict(df)
        assert direction == "neutral"
        assert confidence == 0.5


class TestVolatilityPredictor:
    """Test VolatilityPredictor class."""

    def test_initialization(self):
        """Test VolatilityPredictor initialization."""
        config = {
            "type": "garch",
            "forecast_horizon": 5,
            "threshold": 0.02
        }
        predictor = VolatilityPredictor(config)
        assert predictor.model_type == "garch"
        assert predictor.forecast_horizon == 5
        assert predictor.threshold == 0.02

    def test_create_features(self):
        """Test feature creation."""
        config = {"type": "garch"}
        predictor = VolatilityPredictor(config)

        # Create sample data
        data = {
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }
        df = pd.DataFrame(data)

        features = predictor._create_features(df)
        assert not features.empty
        assert len(features.columns) > 0

    @patch('predictive_models.volatility_predictor.ARCH_AVAILABLE', False)
    def test_predict_garch_unavailable(self):
        """Test GARCH prediction when arch library is not available."""
        config = {"type": "garch"}
        predictor = VolatilityPredictor(config)

        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105]
        })

        regime, confidence = predictor.predict(df)
        assert regime == "low"
        assert confidence == 0.5


class TestVolumePredictor:
    """Test VolumePredictor class."""

    def test_initialization(self):
        """Test VolumePredictor initialization."""
        config = {
            "type": "zscore",
            "threshold": 2.5,
            "lookback": 50
        }
        predictor = VolumePredictor(config)
        assert predictor.model_type == "zscore"
        assert predictor.threshold == 2.5
        assert predictor.lookback == 50

    def test_create_features(self):
        """Test feature creation."""
        config = {"type": "zscore"}
        predictor = VolumePredictor(config)

        # Create sample data
        data = {
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }
        df = pd.DataFrame(data)

        features = predictor._create_features(df)
        assert not features.empty
        assert len(features.columns) > 0

    def test_predict_zscore(self):
        """Test z-score based volume surge detection."""
        config = {"type": "zscore", "threshold": 2.0, "lookback": 20}
        predictor = VolumePredictor(config)

        # Create data with a volume spike
        volumes = [1000] * 19 + [3000]  # Last volume is 3x the mean
        data = {
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [95] * 20,
            'close': [103] * 20,
            'volume': volumes
        }
        df = pd.DataFrame(data)

        is_surge, confidence = predictor.predict(df)
        assert is_surge is True
        assert confidence > 0.5


class TestPredictiveModelManager:
    """Test PredictiveModelManager class."""

    def test_initialization_disabled(self):
        """Test initialization when disabled."""
        config = {"enabled": False}
        manager = PredictiveModelManager(config)
        assert not manager.enabled
        assert manager.price_predictor is None

    def test_initialization_enabled(self):
        """Test initialization when enabled."""
        config = {
            "enabled": True,
            "models": {
                "price_direction": {"type": "lightgbm"},
                "volatility": {"type": "garch"},
                "volume_surge": {"type": "zscore"}
            }
        }
        manager = PredictiveModelManager(config)
        assert manager.enabled
        assert manager.price_predictor is not None
        assert manager.volatility_predictor is not None
        assert manager.volume_predictor is not None

    def test_predict_disabled(self):
        """Test prediction when disabled."""
        config = {"enabled": False}
        manager = PredictiveModelManager(config)

        df = pd.DataFrame({'close': [100, 101, 102]})
        context = manager.predict(df)

        assert context.price_direction == "neutral"
        assert context.volatility == "low"
        assert context.volume_surge is False

    def test_should_allow_signal_disabled(self):
        """Test signal allowance when disabled."""
        config = {"enabled": False}
        manager = PredictiveModelManager(config)

        context = PredictionContext()
        allowed = manager.should_allow_signal("BUY", context)
        assert allowed is True

    def test_should_allow_signal_price_filter(self):
        """Test signal filtering based on price direction."""
        config = {
            "enabled": True,
            "models": {
                "price_direction": {"enabled": True, "confidence_threshold": 0.6}
            }
        }
        manager = PredictiveModelManager(config)

        # Mock price predictor
        manager.price_predictor = Mock()
        manager.price_predictor.predict.return_value = ("down", 0.8)

        # Test BUY signal when price predicts down
        context = PredictionContext(price_direction="down", price_confidence=0.8)
        allowed = manager.should_allow_signal("BUY", context)
        assert allowed is False

        # Test SELL signal when price predicts down
        allowed = manager.should_allow_signal("SELL", context)
        assert allowed is True

    def test_get_model_status(self):
        """Test getting model status."""
        config = {
            "enabled": True,
            "models": {
                "price_direction": {"type": "lightgbm"},
                "volatility": {"type": "garch"},
                "volume_surge": {"type": "zscore"}
            }
        }
        manager = PredictiveModelManager(config)
        status = manager.get_model_status()

        assert status["enabled"] is True
        assert "price_direction" in status["models"]
        assert "volatility" in status["models"]
        assert "volume_surge" in status["models"]


if __name__ == "__main__":
    pytest.main([__file__])
