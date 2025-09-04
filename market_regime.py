"""
Market Regime Detection Module

This module provides sophisticated market regime classification to help trading strategies
adapt to different market conditions. It supports both rule-based and ML-based detection
methods with configurable parameters and stability windows.

Supported Regimes:
- TRENDING: Strong directional movement (ADX > threshold)
- SIDEWAYS: Range-bound, low volatility (ADX < threshold, low ATR)
- VOLATILE: High volatility breakout conditions (high ATR relative to average)

Detection Methods:
- Rule-based: Uses ADX, ATR, and trend indicators
- ML-based: K-means clustering on volatility and momentum features
- HMM-based: Hidden Markov Models for probabilistic regime transitions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from abc import ABC, abstractmethod
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

from indicators import calculate_adx, calculate_atr
from utils.config_loader import get_config

logger = logging.getLogger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING = "trending"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class RegimeDetectionResult:
    """Result of regime detection with metadata."""
    regime: MarketRegime
    confidence: float
    features: Dict[str, float]
    timestamp: datetime
    stability_count: int
    previous_regime: Optional[MarketRegime] = None


class RegimeDetector(ABC):
    """Abstract base class for regime detection methods."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.stability_window = self.config.get('stability_window', 3)
        self.regime_history: List[RegimeDetectionResult] = []
        self.current_regime: Optional[MarketRegime] = None
        self.stability_count = 0

    @abstractmethod
    def detect_regime(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """Detect current market regime from data."""
        pass

    def _check_stability(self, new_regime: MarketRegime) -> MarketRegime:
        """Apply stability window to prevent regime switching too frequently."""
        if self.current_regime is None:
            self.current_regime = new_regime
            self.stability_count = 1
            return new_regime

        if new_regime == self.current_regime:
            self.stability_count += 1
        else:
            self.stability_count = 1

        # Only switch regime if we've seen the new regime for stability_window periods
        if self.stability_count >= self.stability_window:
            if new_regime != self.current_regime:
                logger.info(f"Regime changed: {self.current_regime.value} -> {new_regime.value}")
                self.current_regime = new_regime
            return new_regime
        else:
            # Stick with current regime
            return self.current_regime

    def get_regime_history(self, limit: Optional[int] = None) -> List[RegimeDetectionResult]:
        """Get historical regime detection results."""
        if limit:
            return self.regime_history[-limit:]
        return self.regime_history

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime detection performance."""
        if not self.regime_history:
            return {}

        regimes = [r.regime for r in self.regime_history]
        regime_counts = pd.Series([r.value for r in regimes]).value_counts()

        transitions = 0
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                transitions += 1

        return {
            'total_observations': len(regimes),
            'regime_distribution': regime_counts.to_dict(),
            'transitions': transitions,
            'avg_stability': np.mean([r.stability_count for r in self.regime_history]),
            'current_regime': self.current_regime.value if self.current_regime else None,
            'stability_count': self.stability_count
        }


class RuleBasedRegimeDetector(RegimeDetector):
    """Rule-based regime detection using technical indicators."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Set attributes from config, with defaults
        self.adx_trend_threshold = self.config.get('adx_trend_threshold', 25)
        self.adx_sideways_threshold = self.config.get('adx_sideways_threshold', 20)
        self.atr_volatility_factor = self.config.get('atr_volatility_factor', 1.5)
        self.atr_period = self.config.get('atr_period', 14)
        self.adx_period = self.config.get('adx_period', 14)

    def detect_regime(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """Detect regime using rule-based approach."""
        if data.empty or len(data) < max(self.adx_period, self.atr_period):
            return RegimeDetectionResult(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                features={},
                timestamp=datetime.now(),
                stability_count=0
            )

        try:
            # Calculate indicators
            adx_series = calculate_adx(data, period=self.adx_period)
            atr_series = calculate_atr(data, period=self.atr_period)

            # Ensure we have pandas Series and extract scalar values safely
            if isinstance(adx_series, (list, tuple)):
                current_adx = float(adx_series[-1]) if adx_series else 0.0
            else:
                current_adx = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0.0

            if isinstance(atr_series, (list, tuple)):
                current_atr = float(atr_series[-1]) if atr_series else 0.0
            else:
                current_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

            # Calculate ATR moving average for volatility comparison
            if isinstance(atr_series, (list, tuple)):
                atr_ma = pd.Series(atr_series).rolling(window=self.atr_period).mean()
                current_atr_ma = float(atr_ma.iloc[-1]) if not pd.isna(atr_ma.iloc[-1]) else current_atr
            else:
                atr_ma = atr_series.rolling(window=self.atr_period).mean()
                current_atr_ma = float(atr_ma.iloc[-1]) if not pd.isna(atr_ma.iloc[-1]) else current_atr

            # Calculate slope of closing prices (trend direction and strength)
            if len(data) >= 20:
                # Simple slope calculation: price change over time
                slope = (data['close'].iloc[-1] - data['close'].iloc[0]) / len(data)
                slope_normalized = slope / data['close'].iloc[0]  # Normalize by starting price
            else:
                slope_normalized = 0.0

            # Calculate trend strength (price change over period)
            if len(data) >= 20:
                price_change = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
                trend_strength = abs(price_change)
            else:
                trend_strength = 0.0

            # Calculate volatility (standard deviation of returns)
            returns = data['close'].pct_change().dropna()
            if len(returns) >= 10:
                volatility = returns.tail(10).std()
            else:
                volatility = 0.0

            # Calculate Bollinger Band width for low volatility detection
            bb_width = 0.0
            if len(data) >= 20:
                try:
                    from indicators import calculate_bollinger_bands
                    bb_data = calculate_bollinger_bands(data, period=20, std_dev=2)
                    if bb_data is not None and isinstance(bb_data, dict):
                        if 'upper' in bb_data and 'lower' in bb_data and 'middle' in bb_data:
                            upper_band = bb_data['upper']
                            lower_band = bb_data['lower']
                            middle_band = bb_data['middle']

                            # Extract scalar values safely
                            if hasattr(upper_band, 'iloc'):
                                upper_val = float(upper_band.iloc[-1])
                                lower_val = float(lower_band.iloc[-1])
                                middle_val = float(middle_band.iloc[-1])
                            else:
                                upper_val = float(upper_band[-1]) if isinstance(upper_band, (list, tuple)) else float(upper_band)
                                lower_val = float(lower_band[-1]) if isinstance(lower_band, (list, tuple)) else float(lower_band)
                                middle_val = float(middle_band[-1]) if isinstance(middle_band, (list, tuple)) else float(middle_band)

                            bb_width = (upper_val - lower_val) / middle_val if middle_val > 0 else 0.0
                except Exception as bb_error:
                    logger.debug(f"Bollinger Band calculation failed: {bb_error}")
                    bb_width = 0.0

            # Rule-based classification with improved trend detection
            features = {
                'adx': current_adx,
                'atr': current_atr,
                'atr_ma': current_atr_ma,
                'slope': slope_normalized,
                'trend_strength': trend_strength,
                'volatility': volatility,
                'bb_width': bb_width
            }

            # Determine regime with improved logic using slope + trend_strength as alternative to ADX
            slope_threshold = 0.05  # Minimum slope for trending detection
            trend_strength_threshold = 0.3  # Minimum trend strength for confirmation

            # Primary condition: ADX-based trending detection
            if current_adx > self.adx_trend_threshold:
                if slope_normalized > 0.01:
                    # Positive slope + high ADX = trending up
                    regime = MarketRegime.TRENDING
                    confidence = min((abs(slope_normalized) * 2.0 + current_adx / 50.0) / 3.0, 1.0)
                elif slope_normalized < -0.01:
                    # Negative slope + high ADX = trending down (still TRENDING for strategy selection)
                    regime = MarketRegime.TRENDING
                    confidence = min((abs(slope_normalized) * 2.0 + current_adx / 50.0) / 3.0, 1.0)
                else:
                    # High ADX but no clear slope direction
                    regime = MarketRegime.TRENDING
                    confidence = min(current_adx / 50.0, 1.0)

            # Secondary condition: Slope + trend strength based trending detection (when ADX is flat)
            elif abs(slope_normalized) > slope_threshold and trend_strength > trend_strength_threshold:
                if slope_normalized > 0:
                    # Strong positive slope + trend strength = trending up
                    regime = MarketRegime.TRENDING
                    confidence = min((abs(slope_normalized) * 2.0 + trend_strength) / 3.0, 1.0)
                else:
                    # Strong negative slope + trend strength = trending down (still TRENDING for strategy selection)
                    regime = MarketRegime.TRENDING
                    confidence = min((abs(slope_normalized) * 2.0 + trend_strength) / 3.0, 1.0)

            # Volatility-based regimes
            elif current_atr > (current_atr_ma * self.atr_volatility_factor):
                # High volatility relative to average
                regime = MarketRegime.VOLATILE
                confidence = min(current_atr / current_atr_ma, 2.0) / 2.0  # Normalize to 0-1

            # Low volatility sideways detection
            elif bb_width > 0 and bb_width < 0.02:
                # Very narrow Bollinger Bands = low volatility sideways
                regime = MarketRegime.SIDEWAYS
                confidence = 0.9
            elif current_adx < self.adx_sideways_threshold and volatility < 0.02:
                # Low ADX and low volatility = sideways
                regime = MarketRegime.SIDEWAYS
                confidence = 1.0 - (current_adx / self.adx_sideways_threshold)

            # Default fallback
            else:
                # Default to sideways if unclear
                regime = MarketRegime.SIDEWAYS
                confidence = 0.5

            # Apply stability window
            stable_regime = self._check_stability(regime)

            result = RegimeDetectionResult(
                regime=stable_regime,
                confidence=confidence,
                features=features,
                timestamp=datetime.now(),
                stability_count=self.stability_count,
                previous_regime=self.current_regime
            )

            self.regime_history.append(result)
            return result

        except Exception as e:
            logger.warning(f"Error in rule-based regime detection: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return RegimeDetectionResult(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                features={},
                timestamp=datetime.now(),
                stability_count=0
            )


class MLBasedRegimeDetector(RegimeDetector):
    """ML-based regime detection using clustering and statistical methods."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.ml_method = self.config.get('ml_method', 'clustering')  # 'clustering' or 'hmm'
        self.n_clusters = self.config.get('n_clusters', 3)
        self.lookback_window = self.config.get('lookback_window', 50)
        self.feature_columns = self.config.get('feature_columns', [
            'returns_volatility', 'volume_volatility', 'trend_strength',
            'adx', 'atr_normalized', 'momentum'
        ])

        # ML components
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.is_trained = False

        # Training data storage
        self.training_features: List[np.ndarray] = []
        self.training_labels: List[int] = []

    def detect_regime(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """Detect regime using ML-based approach."""
        if data.empty or len(data) < self.lookback_window:
            return RegimeDetectionResult(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                features={},
                timestamp=datetime.now(),
                stability_count=0
            )

        try:
            # Extract features
            features = self._extract_features(data)

            if not self.is_trained:
                # Use rule-based fallback until trained
                rule_detector = RuleBasedRegimeDetector(self.config)
                return rule_detector.detect_regime(data)

            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Predict cluster
            if self.ml_method == 'clustering':
                cluster = self.cluster_model.predict(features_scaled)[0]
                regime = self._cluster_to_regime(cluster)
                confidence = self._calculate_cluster_confidence(features_scaled, cluster)
            else:
                # HMM or other ML method
                regime = MarketRegime.UNKNOWN
                confidence = 0.0

            # Apply stability window
            stable_regime = self._check_stability(regime)

            result = RegimeDetectionResult(
                regime=stable_regime,
                confidence=confidence,
                features=dict(zip(self.feature_columns, features)),
                timestamp=datetime.now(),
                stability_count=self.stability_count,
                previous_regime=self.current_regime
            )

            self.regime_history.append(result)
            return result

        except Exception as e:
            logger.warning(f"Error in ML-based regime detection: {e}")
            return RegimeDetectionResult(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                features={},
                timestamp=datetime.now(),
                stability_count=0
            )

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for ML model."""
        # Calculate basic indicators
        adx = calculate_adx(data, period=14)
        atr = calculate_atr(data, period=14)

        # Calculate returns inline
        if len(data) > 1:
            returns = data['close'].pct_change().dropna()
        else:
            returns = pd.Series([], dtype=float)

        # Feature calculations
        features = {}

        # Returns volatility (standard deviation of returns)
        if len(returns) >= 10:
            features['returns_volatility'] = returns.tail(10).std()
        else:
            features['returns_volatility'] = 0

        # Volume volatility
        if 'volume' in data.columns and len(data) >= 10:
            volume_returns = data['volume'].pct_change().dropna()
            features['volume_volatility'] = volume_returns.tail(10).std()
        else:
            features['volume_volatility'] = 0

        # Trend strength (ADX)
        features['adx'] = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0

        # Normalized ATR
        if len(atr) >= 14:
            atr_ma = atr.rolling(14).mean()
            features['atr_normalized'] = atr.iloc[-1] / atr_ma.iloc[-1] if atr_ma.iloc[-1] > 0 else 0
        else:
            features['atr_normalized'] = 0

        # Momentum (recent price change)
        if len(data) >= 5:
            features['momentum'] = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
        else:
            features['momentum'] = 0

        # Trend strength (longer term)
        if len(data) >= 20:
            features['trend_strength'] = abs((data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20])
        else:
            features['trend_strength'] = 0

        # Convert to array in correct order
        feature_array = np.array([features[col] for col in self.feature_columns])
        return feature_array

    def train_model(self, historical_data: pd.DataFrame, labels: Optional[List[int]] = None):
        """Train the ML model on historical data."""
        if historical_data.empty or len(historical_data) < self.lookback_window:
            logger.warning("Insufficient data for ML training")
            return

        try:
            # Extract features from historical data
            features_list = []

            # Use sliding window to extract features
            for i in range(self.lookback_window, len(historical_data)):
                window_data = historical_data.iloc[i-self.lookback_window:i]
                features = self._extract_features(window_data)
                features_list.append(features)

            if not features_list:
                logger.warning("No features extracted for training")
                return

            features_array = np.array(features_list)

            # Fit scaler
            self.scaler.fit(features_array)

            # Scale features
            features_scaled = self.scaler.transform(features_array)

            if self.ml_method == 'clustering':
                # Use K-means clustering
                self.cluster_model = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=42,
                    n_init=10
                )
                clusters = self.cluster_model.fit_predict(features_scaled)

                # Analyze clusters to map to regimes
                self._analyze_clusters(features_array, clusters)

            self.is_trained = True
            logger.info(f"ML model trained successfully with {len(features_list)} samples")

        except Exception as e:
            logger.error(f"Error training ML model: {e}")

    def _analyze_clusters(self, features: np.ndarray, clusters: np.ndarray):
        """Analyze clusters to map them to market regimes."""
        # Simple heuristic: analyze feature statistics per cluster
        cluster_stats = {}

        for cluster_id in range(self.n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_features = features[cluster_mask]

            if len(cluster_features) == 0:
                continue

            # Calculate average features for this cluster
            avg_features = np.mean(cluster_features, axis=0)
            feature_dict = dict(zip(self.feature_columns, avg_features))

            cluster_stats[cluster_id] = feature_dict

        # Map clusters to regimes based on feature values
        self.cluster_regime_mapping = {}

        for cluster_id, stats in cluster_stats.items():
            # Decision logic based on feature values
            if stats['adx'] > 25:  # High ADX = trending
                self.cluster_regime_mapping[cluster_id] = MarketRegime.TRENDING
            elif stats['atr_normalized'] > 1.2:  # High ATR = volatile
                self.cluster_regime_mapping[cluster_id] = MarketRegime.VOLATILE
            else:  # Default to sideways
                self.cluster_regime_mapping[cluster_id] = MarketRegime.SIDEWAYS

        logger.info(f"Cluster-regime mapping: {self.cluster_regime_mapping}")

    def _cluster_to_regime(self, cluster: int) -> MarketRegime:
        """Convert cluster ID to market regime."""
        return self.cluster_regime_mapping.get(cluster, MarketRegime.UNKNOWN)

    def _calculate_cluster_confidence(self, features_scaled: np.ndarray, cluster: int) -> float:
        """Calculate confidence in cluster assignment."""
        if self.cluster_model is None:
            return 0.0

        # Calculate distance to cluster center
        center = self.cluster_model.cluster_centers_[cluster]
        distance = np.linalg.norm(features_scaled - center)

        # Convert distance to confidence (closer = higher confidence)
        # This is a simple heuristic - could be improved
        confidence = max(0, 1 - distance / 2.0)
        return confidence


class MarketRegimeDetector:
    """
    Main market regime detector that combines multiple detection methods.
    Provides unified interface and handles method switching.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the market regime detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.enabled = self.config.get('enabled', True)
        self.mode = self.config.get('mode', 'rule_based')

        # Initialize detectors with full config
        self.rule_detector = RuleBasedRegimeDetector(self.config)
        self.ml_detector = MLBasedRegimeDetector(self.config)

        # Ensure rule detector gets the config values it needs
        if hasattr(self.rule_detector, 'adx_trend_threshold'):
            self.rule_detector.adx_trend_threshold = self.config.get('adx_trend_threshold', 25)
            self.rule_detector.adx_sideways_threshold = self.config.get('adx_sideways_threshold', 20)
            self.rule_detector.atr_volatility_factor = self.config.get('atr_volatility_factor', 1.5)
            self.rule_detector.stability_window = self.config.get('stability_window', 3)

        # Current detector
        self.current_detector = self.rule_detector if self.mode == 'rule_based' else self.ml_detector

        logger.info(f"MarketRegimeDetector initialized: mode={self.mode}, enabled={self.enabled}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'enabled': True,
            'mode': 'rule_based',
            'adx_trend_threshold': 25,
            'adx_sideways_threshold': 20,
            'atr_volatility_factor': 1.5,
            'atr_period': 14,
            'adx_period': 14,
            'stability_window': 3,
            'ml_method': 'clustering',
            'n_clusters': 3,
            'lookback_window': 50
        }

    def detect_regime(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """
        Detect current market regime.

        Args:
            data: OHLCV DataFrame

        Returns:
            RegimeDetectionResult with detection details
        """
        if not self.enabled or data.empty:
            return RegimeDetectionResult(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                features={},
                timestamp=datetime.now(),
                stability_count=0
            )

        return self.current_detector.detect_regime(data)

    def train_ml_model(self, historical_data: pd.DataFrame, labels: Optional[List[int]] = None):
        """
        Train the ML model on historical data.

        Args:
            historical_data: Historical OHLCV data
            labels: Optional regime labels for supervised learning
        """
        if self.mode == 'ml_based':
            self.ml_detector.train_model(historical_data, labels)
        else:
            logger.warning("ML training requested but detector is in rule-based mode")

    def switch_mode(self, mode: str):
        """
        Switch detection mode.

        Args:
            mode: 'rule_based' or 'ml_based'
        """
        if mode == 'rule_based':
            self.current_detector = self.rule_detector
            self.mode = 'rule_based'
        elif mode == 'ml_based':
            self.current_detector = self.ml_detector
            self.mode = 'ml_based'
        else:
            logger.warning(f"Unknown mode: {mode}")

        logger.info(f"Switched to {self.mode} detection mode")

    def get_regime_history(self, limit: Optional[int] = None) -> List[RegimeDetectionResult]:
        """Get historical regime detection results."""
        return self.current_detector.get_regime_history(limit)

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime detection."""
        return self.current_detector.get_regime_statistics()

    def save_model(self, path: str):
        """Save ML model to disk."""
        if self.mode == 'ml_based' and self.ml_detector.is_trained:
            model_data = {
                'scaler': self.ml_detector.scaler,
                'cluster_model': self.ml_detector.cluster_model,
                'cluster_regime_mapping': self.ml_detector.cluster_regime_mapping,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }

            Path(path).parent.mkdir(parents=True, exist_ok=True)
            # Note: This is a simplified save - in production you'd use joblib or pickle
            logger.info(f"Model save functionality would save to {path}")

    def load_model(self, path: str):
        """Load ML model from disk."""
        if not Path(path).exists():
            logger.warning(f"Model file not found: {path}")
            return

        # Note: This is a placeholder for model loading
        logger.info(f"Model load functionality would load from {path}")


# Global market regime detector instance
_market_regime_detector: Optional[MarketRegimeDetector] = None


def get_market_regime_detector() -> MarketRegimeDetector:
    """Get the global market regime detector instance."""
    global _market_regime_detector
    if _market_regime_detector is None:
        config = get_config('market_regime', {})
        _market_regime_detector = MarketRegimeDetector(config)
    return _market_regime_detector


def detect_market_regime(data: pd.DataFrame) -> RegimeDetectionResult:
    """
    Convenience function to detect market regime.

    Args:
        data: OHLCV DataFrame

    Returns:
        RegimeDetectionResult
    """
    detector = get_market_regime_detector()
    return detector.detect_regime(data)


# Strategy recommendations based on regime
REGIME_STRATEGY_MAPPING = {
    MarketRegime.TRENDING: ['EMACrossStrategy', 'MACDStrategy'],
    MarketRegime.SIDEWAYS: ['RSIStrategy', 'BollingerBandsStrategy'],
    MarketRegime.VOLATILE: ['BreakoutStrategy', 'MomentumStrategy'],
    MarketRegime.UNKNOWN: ['RSIStrategy']  # Default fallback
}


def get_recommended_strategies(regime: MarketRegime) -> List[str]:
    """
    Get recommended strategies for a given market regime.

    Args:
        regime: Detected market regime

    Returns:
        List of recommended strategy names
    """
    return REGIME_STRATEGY_MAPPING.get(regime, REGIME_STRATEGY_MAPPING[MarketRegime.UNKNOWN])


def get_regime_risk_multiplier(regime: MarketRegime) -> float:
    """
    Get risk multiplier based on market regime.

    Args:
        regime: Detected market regime

    Returns:
        Risk multiplier (higher = more conservative)
    """
    multipliers = {
        MarketRegime.TRENDING: 1.0,      # Normal risk
        MarketRegime.SIDEWAYS: 0.8,      # Slightly reduced risk
        MarketRegime.VOLATILE: 0.6,      # Significantly reduced risk
        MarketRegime.UNKNOWN: 0.7        # Conservative default
    }
    return multipliers.get(regime, 0.7)
