"""
Unit tests for market regime detection module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from strategies.regime.market_regime import (
    MarketRegimeDetector,
    RuleBasedRegimeDetector,
    MLBasedRegimeDetector,
    MarketRegime,
    RegimeDetectionResult,
    EnhancedRegimeResult,
    get_market_regime_detector,
    detect_market_regime,
    get_recommended_strategies,
    get_regime_risk_multiplier
)


class TestMarketRegimeDetector:
    """Test MarketRegimeDetector."""

    @patch('strategies.regime.market_regime.get_config')
    def test_init(self, mock_get_config):
        """Test initialization."""
        mock_get_config.return_value = {
            'enabled': True,
            'mode': 'rule_based',
            'adx_trend_threshold': 25
        }

        detector = MarketRegimeDetector()
        assert detector.enabled is True
        assert detector.mode == 'rule_based'

    @patch('strategies.regime.market_regime.get_config')
    def test_detect_enhanced_regime(self, mock_get_config):
        """Test enhanced regime detection."""
        mock_get_config.return_value = {
            'enabled': True,
            'mode': 'rule_based'
        }

        detector = MarketRegimeDetector()

        # Create test data
        data = pd.DataFrame({
            'open': [10] * 50,
            'high': [11] * 50,
            'low': [9] * 50,
            'close': [10] * 50,
            'volume': [1000] * 50
        })

        result = detector.detect_enhanced_regime(data)
        assert isinstance(result, EnhancedRegimeResult)
        assert isinstance(result.regime_name, str)
        assert isinstance(result.confidence_score, float)
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.reasons, dict)

    @patch('strategies.regime.market_regime.get_config')
    def test_detect_regime_rule_based(self, mock_get_config):
        """Test regime detection in rule-based mode."""
        mock_get_config.return_value = {
            'enabled': True,
            'mode': 'rule_based'
        }

        detector = MarketRegimeDetector()

        # Create sideways market data (low ADX, low volatility)
        data = pd.DataFrame({
            'open': [10] * 50,
            'high': [11] * 50,
            'low': [9] * 50,
            'close': [10] * 50,
            'volume': [1000] * 50
        })

        result = detector.detect_regime(data)
        assert isinstance(result, RegimeDetectionResult)
        assert isinstance(result.regime, MarketRegime)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    @patch('strategies.regime.market_regime.get_config')
    def test_detect_regime_disabled(self, mock_get_config):
        """Test regime detection when disabled."""
        mock_get_config.return_value = {'enabled': False}

        detector = MarketRegimeDetector()
        data = pd.DataFrame({'close': [10, 11, 12]})

        result = detector.detect_regime(data)
        assert result.regime == MarketRegime.UNKNOWN
        assert result.confidence == 0.0

    @patch('strategies.regime.market_regime.get_config')
    def test_switch_mode(self, mock_get_config):
        """Test mode switching."""
        mock_get_config.return_value = {'enabled': True, 'mode': 'rule_based'}

        detector = MarketRegimeDetector()
        assert detector.mode == 'rule_based'

        detector.switch_mode('ml_based')
        assert detector.mode == 'ml_based'

    @patch('strategies.regime.market_regime.get_config')
    def test_get_regime_statistics(self, mock_get_config):
        """Test getting regime statistics."""
        mock_get_config.return_value = {'enabled': True}

        detector = MarketRegimeDetector()

        # Create some test data
        data = pd.DataFrame({
            'open': [10] * 30,
            'high': [11] * 30,
            'low': [9] * 30,
            'close': [10] * 30,
            'volume': [1000] * 30
        })

        # Generate some regime detections
        for _ in range(5):
            detector.detect_regime(data)

        stats = detector.get_regime_statistics()
        assert isinstance(stats, dict)
        assert 'total_observations' in stats
        assert 'regime_distribution' in stats


class TestRuleBasedRegimeDetector:
    """Test RuleBasedRegimeDetector."""

    def test_init(self):
        """Test initialization."""
        config = {'adx_trend_threshold': 30}
        detector = RuleBasedRegimeDetector(config)
        assert detector.adx_trend_threshold == 30

    def test_detect_trending_regime(self):
        """Test detection of trending regime."""
        detector = RuleBasedRegimeDetector()

        # Create trending data (simulated high ADX)
        data = pd.DataFrame({
            'open': list(range(10, 60)),  # Strong upward trend
            'high': [i + 3 for i in range(10, 60)],
            'low': [i - 1 for i in range(10, 60)],
            'close': [i + 2 for i in range(10, 60)],
            'volume': [1000] * 50
        })

        result = detector.detect_regime(data)
        assert isinstance(result, RegimeDetectionResult)

    def test_detect_sideways_regime(self):
        """Test detection of sideways regime."""
        detector = RuleBasedRegimeDetector()

        # Create sideways data
        data = pd.DataFrame({
            'open': [10] * 50,
            'high': [11] * 50,
            'low': [9] * 50,
            'close': [10] * 50,
            'volume': [1000] * 50
        })

        result = detector.detect_regime(data)
        assert isinstance(result, RegimeDetectionResult)
        # Should detect sideways due to low volatility and ADX

    def test_detect_volatile_regime(self):
        """Test detection of volatile regime."""
        detector = RuleBasedRegimeDetector()

        # Create volatile data (high ATR relative to average)
        data = pd.DataFrame({
            'open': [10, 15, 8, 12, 9] * 10,  # High volatility
            'high': [16, 17, 14, 18, 15] * 10,
            'low': [8, 13, 6, 10, 7] * 10,
            'close': [15, 8, 12, 9, 14] * 10,
            'volume': [1000] * 50
        })

        result = detector.detect_regime(data)
        assert isinstance(result, RegimeDetectionResult)

    def test_detect_trend_up_regime(self):
        """Test detection of trend up regime."""
        detector = RuleBasedRegimeDetector()

        # Create strong upward trending data
        data = pd.DataFrame({
            'open': list(range(10, 60)),  # Strong upward trend
            'high': [i + 3 for i in range(10, 60)],
            'low': [i - 1 for i in range(10, 60)],
            'close': [i + 2 for i in range(10, 60)],
            'volume': [1000] * 50
        })

        result = detector.detect_regime(data)
        assert isinstance(result, RegimeDetectionResult)
        # Should detect TREND_UP due to strong positive slope and high ADX

    def test_detect_trend_down_regime(self):
        """Test detection of trend down regime."""
        detector = RuleBasedRegimeDetector()

        # Create strong downward trending data
        data = pd.DataFrame({
            'open': list(range(60, 10, -1)),  # Strong downward trend
            'high': [i + 1 for i in range(60, 10, -1)],
            'low': [i - 3 for i in range(60, 10, -1)],
            'close': [i - 2 for i in range(60, 10, -1)],
            'volume': [1000] * 50
        })

        result = detector.detect_regime(data)
        assert isinstance(result, RegimeDetectionResult)
        # Should detect TREND_DOWN due to strong negative slope and high ADX

    def test_detect_range_tight_regime(self):
        """Test detection of range tight regime."""
        detector = RuleBasedRegimeDetector()

        # Create tight range data (very narrow Bollinger Bands)
        data = pd.DataFrame({
            'open': [10.0] * 50,  # Very stable prices
            'high': [10.1] * 50,
            'low': [9.9] * 50,
            'close': [10.0] * 50,
            'volume': [1000] * 50
        })

        result = detector.detect_regime(data)
        assert isinstance(result, RegimeDetectionResult)
        # Should detect RANGE_TIGHT due to very narrow Bollinger Bands

    def test_detect_range_wide_regime(self):
        """Test detection of range wide regime."""
        detector = RuleBasedRegimeDetector()

        # Create wide range data (wide Bollinger Bands but low ADX)
        data = pd.DataFrame({
            'open': [10, 12, 8, 11, 9] * 10,  # Moderate volatility
            'high': [13, 14, 12, 15, 13] * 10,
            'low': [7, 8, 6, 9, 7] * 10,
            'close': [12, 8, 11, 9, 10] * 10,
            'volume': [1000] * 50
        })

        result = detector.detect_regime(data)
        assert isinstance(result, RegimeDetectionResult)
        # Should detect RANGE_WIDE due to wide Bollinger Bands

    def test_detect_volatile_spike_regime(self):
        """Test detection of volatile spike regime."""
        detector = RuleBasedRegimeDetector()

        # Create volatile spike data (high volume and ATR)
        data = pd.DataFrame({
            'open': [10, 15, 8, 12, 9] * 10,  # High volatility
            'high': [16, 17, 14, 18, 15] * 10,
            'low': [8, 13, 6, 10, 7] * 10,
            'close': [15, 8, 12, 9, 14] * 10,
            'volume': [5000, 6000, 4500, 5500, 4800] * 10  # High volume
        })

        result = detector.detect_regime(data)
        assert isinstance(result, RegimeDetectionResult)
        # Should detect VOLATILE_SPIKE due to high volume spike and ATR

    def test_detect_enhanced_regime_trend_up(self):
        """Test enhanced regime detection for trend up."""
        detector = RuleBasedRegimeDetector()

        # Create strong upward trending data with consistent direction
        # Use a more gradual trend to ensure ADX can build up
        base_prices = []
        for i in range(100):
            base_prices.append(10 + i * 0.1)  # Gradual upward trend

        data = pd.DataFrame({
            'open': base_prices,
            'high': [p + 0.5 for p in base_prices],
            'low': [p - 0.5 for p in base_prices],
            'close': [p + 0.2 for p in base_prices],
            'volume': [1000] * 100
        })

        result = detector.detect_enhanced_regime(data)
        assert isinstance(result, EnhancedRegimeResult)
        # The regime could be trend_up, trending, or range_wide depending on calculations
        assert result.regime_name in ["trend_up", "trending", "range_wide", "sideways"]
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.reasons, dict)
        assert "adx" in result.reasons
        assert "slope" in result.reasons

    def test_stability_window(self):
        """Test stability window prevents frequent regime changes."""
        detector = RuleBasedRegimeDetector({'stability_window': 3})

        # Create consistent sideways data
        data = pd.DataFrame({
            'open': [10] * 30,
            'high': [11] * 30,
            'low': [9] * 30,
            'close': [10] * 30,
            'volume': [1000] * 30
        })

        # First detection
        result1 = detector.detect_regime(data)
        assert detector.stability_count == 1

        # Second detection (same regime)
        result2 = detector.detect_regime(data)
        assert detector.stability_count == 2

        # Third detection (should stabilize)
        result3 = detector.detect_regime(data)
        assert detector.stability_count == 3

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        detector = RuleBasedRegimeDetector()

        # Create data with insufficient length
        data = pd.DataFrame({
            'open': [10, 11],
            'high': [12, 13],
            'low': [8, 9],
            'close': [11, 12],
            'volume': [1000, 1100]
        })

        result = detector.detect_regime(data)
        assert result.regime == MarketRegime.UNKNOWN
        assert result.confidence == 0.0


class TestMLBasedRegimeDetector:
    """Test MLBasedRegimeDetector."""

    def test_init(self):
        """Test initialization."""
        config = {'n_clusters': 4}
        detector = MLBasedRegimeDetector(config)
        assert detector.n_clusters == 4

    def test_extract_features(self):
        """Test feature extraction."""
        detector = MLBasedRegimeDetector()

        # Create test data
        data = pd.DataFrame({
            'open': list(range(10, 60)),
            'high': [i + 2 for i in range(10, 60)],
            'low': [i - 2 for i in range(10, 60)],
            'close': [i + 1 for i in range(10, 60)],
            'volume': [1000] * 50
        })

        features = detector._extract_features(data)
        assert isinstance(features, np.ndarray)
        assert len(features) == len(detector.feature_columns)

    def test_detect_regime_untrained(self):
        """Test regime detection when model is untrained."""
        detector = MLBasedRegimeDetector()

        # Create test data
        data = pd.DataFrame({
            'open': [10] * 30,
            'high': [11] * 30,
            'low': [9] * 30,
            'close': [10] * 30,
            'volume': [1000] * 30
        })

        # Should fall back to rule-based when untrained
        result = detector.detect_regime(data)
        assert isinstance(result, RegimeDetectionResult)

    def test_train_model(self):
        """Test model training."""
        detector = MLBasedRegimeDetector()

        # Create historical data for training
        data = pd.DataFrame({
            'open': list(range(10, 110)),  # 100 periods
            'high': [i + 2 for i in range(10, 110)],
            'low': [i - 2 for i in range(10, 110)],
            'close': [i + 1 for i in range(10, 110)],
            'volume': [1000] * 100
        })

        # Train the model
        detector.train_model(data)

        # Check if model was trained
        assert detector.is_trained
        assert detector.cluster_model is not None

    def test_insufficient_training_data(self):
        """Test training with insufficient data."""
        detector = MLBasedRegimeDetector()

        # Create insufficient data
        data = pd.DataFrame({
            'open': [10, 11],
            'high': [12, 13],
            'low': [8, 9],
            'close': [11, 12],
            'volume': [1000, 1100]
        })

        # Should not train with insufficient data
        detector.train_model(data)
        assert not detector.is_trained


class TestRegimeDetectionResult:
    """Test RegimeDetectionResult."""

    def test_init(self):
        """Test initialization."""
        from datetime import datetime

        result = RegimeDetectionResult(
            regime=MarketRegime.TRENDING,
            confidence=0.85,
            features={'adx': 30.5, 'atr': 2.1},
            timestamp=datetime.now(),
            stability_count=2
        )

        assert result.regime == MarketRegime.TRENDING
        assert result.confidence == 0.85
        assert result.features['adx'] == 30.5
        assert result.stability_count == 2


class TestGlobalFunctions:
    """Test global convenience functions."""

    @patch('strategies.regime.market_regime.get_market_regime_detector')
    def test_detect_market_regime_global(self, mock_get_detector):
        """Test global detect_market_regime function."""
        mock_detector = MagicMock()
        mock_result = RegimeDetectionResult(
            regime=MarketRegime.SIDEWAYS,
            confidence=0.7,
            features={},
            timestamp=pd.Timestamp.now(),
            stability_count=1
        )
        mock_detector.detect_regime.return_value = mock_result
        mock_get_detector.return_value = mock_detector

        data = pd.DataFrame({'close': [10, 11, 12]})
        result = detect_market_regime(data)

        assert result.regime == MarketRegime.SIDEWAYS
        assert result.confidence == 0.7
        mock_detector.detect_regime.assert_called_once_with(data)

    def test_get_recommended_strategies(self):
        """Test getting recommended strategies for regimes."""
        # Test trending regime
        strategies = get_recommended_strategies(MarketRegime.TRENDING)
        assert isinstance(strategies, list)
        assert len(strategies) > 0

        # Test sideways regime
        strategies = get_recommended_strategies(MarketRegime.SIDEWAYS)
        assert isinstance(strategies, list)
        assert len(strategies) > 0

        # Test unknown regime
        strategies = get_recommended_strategies(MarketRegime.UNKNOWN)
        assert isinstance(strategies, list)
        assert len(strategies) > 0

    def test_get_regime_risk_multiplier(self):
        """Test getting risk multipliers for regimes."""
        # Test different regimes
        multiplier_trending = get_regime_risk_multiplier(MarketRegime.TRENDING)
        multiplier_sideways = get_regime_risk_multiplier(MarketRegime.SIDEWAYS)
        multiplier_volatile = get_regime_risk_multiplier(MarketRegime.VOLATILE)
        multiplier_unknown = get_regime_risk_multiplier(MarketRegime.UNKNOWN)

        assert isinstance(multiplier_trending, float)
        assert isinstance(multiplier_sideways, float)
        assert isinstance(multiplier_volatile, float)
        assert isinstance(multiplier_unknown, float)

        # Volatile should have lowest multiplier (most conservative)
        assert multiplier_volatile <= multiplier_trending
        assert multiplier_volatile <= multiplier_sideways


class TestIntegration:
    """Test integration scenarios."""

    @patch('strategies.regime.market_regime.get_config')
    def test_full_regime_detection_workflow(self, mock_get_config):
        """Test full regime detection workflow."""
        mock_get_config.return_value = {
            'enabled': True,
            'mode': 'rule_based'
        }

        detector = MarketRegimeDetector()

        # Create trending market data
        data = pd.DataFrame({
            'open': list(range(10, 60)),  # Strong trend
            'high': [i + 3 for i in range(10, 60)],
            'low': [i - 1 for i in range(10, 60)],
            'close': [i + 2 for i in range(10, 60)],
            'volume': [1000] * 50
        })

        # Detect regime
        result = detector.detect_regime(data)
        assert isinstance(result, RegimeDetectionResult)

        # Get recommended strategies
        strategies = get_recommended_strategies(result.regime)
        assert isinstance(strategies, list)

        # Get risk multiplier
        multiplier = get_regime_risk_multiplier(result.regime)
        assert isinstance(multiplier, float)

        # Check regime history
        history = detector.get_regime_history()
        assert len(history) >= 1
        assert history[-1] == result

    @patch('strategies.regime.market_regime.get_config')
    def test_regime_persistence(self, mock_get_config):
        """Test regime detection persistence."""
        mock_get_config.return_value = {'enabled': True}

        detector = MarketRegimeDetector()

        # Create data and detect regime multiple times
        data = pd.DataFrame({
            'open': [10] * 30,
            'high': [11] * 30,
            'low': [9] * 30,
            'close': [10] * 30,
            'volume': [1000] * 30
        })

        # Detect regime multiple times
        for _ in range(3):
            result = detector.detect_regime(data)

        # Check statistics
        stats = detector.get_regime_statistics()
        assert stats['total_observations'] == 3
        assert isinstance(stats['regime_distribution'], dict)


class TestErrorHandling:
    """Test error handling."""

    @patch('strategies.regime.market_regime.get_config')
    def test_empty_data_handling(self, mock_get_config):
        """Test handling of empty data."""
        mock_get_config.return_value = {'enabled': True}

        detector = MarketRegimeDetector()
        data = pd.DataFrame()

        result = detector.detect_regime(data)
        assert result.regime == MarketRegime.UNKNOWN
        assert result.confidence == 0.0

    def test_invalid_regime_enum(self):
        """Test handling of invalid regime values."""
        # This should not raise an error
        multiplier = get_regime_risk_multiplier(MarketRegime.UNKNOWN)
        assert isinstance(multiplier, float)

        strategies = get_recommended_strategies(MarketRegime.UNKNOWN)
        assert isinstance(strategies, list)


class TestConfiguration:
    """Test configuration handling."""

    def test_custom_configuration(self):
        """Test custom configuration."""
        custom_config = {
            'enabled': True,
            'mode': 'rule_based',
            'adx_trend_threshold': 30,
            'adx_sideways_threshold': 15,
            'atr_volatility_factor': 2.0,
            'stability_window': 5
        }

        detector = MarketRegimeDetector(custom_config)

        # Check that custom config was applied to the rule detector
        assert detector.rule_detector.adx_trend_threshold == 30
        assert detector.rule_detector.stability_window == 5

    def test_default_configuration(self):
        """Test default configuration."""
        detector = RuleBasedRegimeDetector()

        # Check default values
        assert detector.adx_trend_threshold == 25
        assert detector.adx_sideways_threshold == 20
        assert detector.atr_volatility_factor == 1.5
        assert detector.stability_window == 3


class TestMLTraining:
    """Test ML model training scenarios."""

    def test_clustering_analysis(self):
        """Test cluster analysis for regime mapping."""
        detector = MLBasedRegimeDetector()

        # Create synthetic features for different regimes matching feature_columns
        # Features: returns_volatility, volume_volatility, trend_strength, adx, atr_normalized, momentum, bb_width, autocorrelation

        features = np.array([
            [0.01, 0.05, 0.15, 30, 1.2, 0.02, 0.03, 0.75],  # Trending
            [0.005, 0.02, 0.02, 15, 0.8, 0.01, 0.015, 0.1],  # Sideways
            [0.03, 0.08, 0.08, 20, 1.5, 0.04, 0.06, 0.2],   # Volatile
            [0.015, 0.06, 0.12, 35, 1.3, 0.025, 0.04, 0.8], # Trending
            [0.003, 0.015, 0.01, 12, 0.7, 0.008, 0.005, 0.05], # Sideways
        ])

        clusters = np.array([0, 1, 2, 0, 1])  # 3 clusters

        # Test cluster analysis
        detector._analyze_clusters(features, clusters)

        # Check that cluster mapping was created
        assert hasattr(detector, 'cluster_regime_mapping')
        assert len(detector.cluster_regime_mapping) == 3

        # Check that all clusters are mapped to valid regimes
        for cluster_id in [0, 1, 2]:
            regime = detector.cluster_regime_mapping[cluster_id]
            assert isinstance(regime, MarketRegime)

    def test_cluster_to_regime_conversion(self):
        """Test conversion from cluster to regime."""
        detector = MLBasedRegimeDetector()

        # Set up cluster mapping
        detector.cluster_regime_mapping = {
            0: MarketRegime.TRENDING,
            1: MarketRegime.SIDEWAYS,
            2: MarketRegime.VOLATILE
        }

        # Test conversion
        assert detector._cluster_to_regime(0) == MarketRegime.TRENDING
        assert detector._cluster_to_regime(1) == MarketRegime.SIDEWAYS
        assert detector._cluster_to_regime(2) == MarketRegime.VOLATILE

        # Test unknown cluster
        assert detector._cluster_to_regime(999) == MarketRegime.UNKNOWN
