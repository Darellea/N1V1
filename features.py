"""
Feature Extraction Pipeline

This module provides a comprehensive feature extraction pipeline for trading signals.
It processes OHLCV data through technical indicators and prepares features for ML models.

Key features:
- Indicator calculation pipeline
- Feature normalization and scaling
- Lagged feature generation
- Feature validation and cleaning
- Support for multiple symbols and timeframes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging
from datetime import datetime, timedelta

from indicators import (
    calculate_all_indicators,
    get_indicator_names,
    validate_ohlcv_data
)

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Feature extraction pipeline for trading data.

    Handles the complete pipeline from raw OHLCV data to ML-ready features.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature extractor.

        Args:
            config: Configuration dictionary for feature extraction
        """
        self.config = config or self._get_default_config()
        self.scaler = None
        self.feature_columns = []
        self.is_fitted = False

        # Initialize scaler based on config
        self._init_scaler()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for feature extraction."""
        return {
            'indicator_params': {
                'rsi_period': 14,
                'ema_period': 20,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'bb_std_dev': 2.0,
                'atr_period': 14,
                'adx_period': 14
            },
            'scaling': {
                'method': 'standard',  # 'standard', 'minmax', 'robust', 'none'
                'feature_range': (0, 1)  # for minmax scaler
            },
            'lagged_features': {
                'enabled': True,
                'periods': [1, 2, 3, 5, 10]  # periods to lag
            },
            'price_features': {
                'returns': True,
                'log_returns': True,
                'price_changes': True
            },
            'volume_features': {
                'volume_sma': True,
                'volume_ratio': True
            },
            'validation': {
                'require_min_rows': 50,
                'handle_missing': 'drop',  # 'drop', 'fill', 'interpolate'
                'fill_method': 'ffill'
            }
        }

    def _init_scaler(self):
        """Initialize the feature scaler based on configuration."""
        scaling_config = self.config.get('scaling', {})
        method = scaling_config.get('method', 'standard')

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            feature_range = scaling_config.get('feature_range', (0, 1))
            self.scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'none':
            self.scaler = None
        else:
            logger.warning(f"Unknown scaling method '{method}', using StandardScaler")
            self.scaler = StandardScaler()

    def extract_features(self, data: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Extract features from OHLCV data.

        Args:
            data: DataFrame with OHLCV data
            fit_scaler: Whether to fit the scaler on this data

        Returns:
            DataFrame with extracted and processed features
        """
        # Handle empty data gracefully
        if data.empty:
            logger.warning("Input data is empty, returning empty feature DataFrame")
            return pd.DataFrame()

        if not validate_ohlcv_data(data):
            raise ValueError("Data must contain OHLCV columns: open, high, low, close, volume")

        if len(data) < self.config['validation']['require_min_rows']:
            logger.warning(f"Insufficient data rows: {len(data)} < {self.config['validation']['require_min_rows']}")
            return pd.DataFrame()

        try:
            # Step 1: Calculate technical indicators
            df_with_indicators = self._calculate_indicators(data)

            # Step 2: Add price-based features
            df_with_price_features = self._add_price_features(df_with_indicators)

            # Step 3: Add volume-based features
            df_with_volume_features = self._add_volume_features(df_with_price_features)

            # Step 4: Add lagged features
            df_with_lagged = self._add_lagged_features(df_with_volume_features)

            # Step 5: Clean and validate features
            df_clean = self._clean_features(df_with_lagged)

            # Step 6: Scale features
            df_scaled = self._scale_features(df_clean, fit_scaler)

            logger.info(f"Extracted {len(df_scaled.columns)} features from {len(data)} data points")
            return df_scaled

        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            return pd.DataFrame()

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        indicator_params = self.config.get('indicator_params', {})
        df = calculate_all_indicators(data, indicator_params)

        # Store feature columns for later use
        indicator_names = get_indicator_names()
        self.feature_columns.extend([col for col in indicator_names if col in df.columns])

        return df

    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        df = data.copy()
        price_config = self.config.get('price_features', {})

        if price_config.get('returns', True):
            df['returns'] = df['close'].pct_change()
            self.feature_columns.append('returns')

        if price_config.get('log_returns', True):
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            self.feature_columns.append('log_returns')

        if price_config.get('price_changes', True):
            df['price_change'] = df['close'] - df['close'].shift(1)
            self.feature_columns.append('price_change')

        # Add more price features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['body_size'] = (df['close'] - df['open']).abs()
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

        new_price_features = ['high_low_ratio', 'close_open_ratio', 'body_size', 'upper_shadow', 'lower_shadow']
        self.feature_columns.extend(new_price_features)

        return df

    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df = data.copy()
        volume_config = self.config.get('volume_features', {})

        if volume_config.get('volume_sma', True):
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            self.feature_columns.append('volume_sma_20')

        if volume_config.get('volume_ratio', True):
            df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
            self.feature_columns.append('volume_ratio')

        # Additional volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

        new_volume_features = ['volume_change', 'volume_ma_ratio']
        self.feature_columns.extend(new_volume_features)

        return df

    def _add_lagged_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for temporal patterns."""
        df = data.copy()
        lagged_config = self.config.get('lagged_features', {})

        if not lagged_config.get('enabled', True):
            return df

        periods = lagged_config.get('periods', [1, 2, 3])

        # Create lagged versions of key features
        key_features = ['close', 'rsi', 'ema', 'macd', 'bb_upper', 'bb_lower', 'atr', 'adx']

        for feature in key_features:
            if feature in df.columns:
                for period in periods:
                    lagged_col = f"{feature}_lag_{period}"
                    df[lagged_col] = df[feature].shift(period)
                    self.feature_columns.append(lagged_col)

        return df

    def _clean_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features."""
        df = data.copy()
        validation_config = self.config.get('validation', {})

        # Handle missing values
        handle_missing = validation_config.get('handle_missing', 'drop')

        if handle_missing == 'drop':
            # Drop rows with any NaN values
            initial_rows = len(df)
            df = df.dropna()
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with missing values")

        elif handle_missing == 'fill':
            # Fill missing values
            fill_method = validation_config.get('fill_method', 'ffill')
            if fill_method == 'ffill':
                df = df.ffill()
            elif fill_method == 'bfill':
                df = df.bfill()
            else:
                # Fill with specific value
                df = df.fillna(0)

        elif handle_missing == 'interpolate':
            df = df.interpolate(method='linear')

        # Remove duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]

        # Ensure we have some features left
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        if not feature_cols:
            logger.warning("No feature columns remaining after cleaning")
            return pd.DataFrame()

        self.feature_columns = feature_cols
        return df

    def _scale_features(self, data: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Scale features using the configured scaler."""
        if self.scaler is None:
            return data

        df = data.copy()
        feature_cols = [col for col in self.feature_columns if col in df.columns]

        if not feature_cols:
            return df

        try:
            if fit_scaler or not self.is_fitted:
                self.scaler.fit(df[feature_cols])
                self.is_fitted = True

            scaled_features = self.scaler.transform(df[feature_cols])
            scaled_df = pd.DataFrame(
                scaled_features,
                columns=feature_cols,
                index=df.index
            )

            # Keep non-feature columns unchanged
            non_feature_cols = [col for col in df.columns if col not in feature_cols]
            if non_feature_cols:
                scaled_df = pd.concat([scaled_df, df[non_feature_cols]], axis=1)

            return scaled_df

        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return data

    def get_feature_importance_template(self) -> Dict[str, float]:
        """
        Get a template for feature importance scores.

        Returns:
            Dictionary with feature names as keys and 0.0 as values
        """
        return {col: 0.0 for col in self.feature_columns}

    def get_feature_stats(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for extracted features.

        Args:
            data: DataFrame with features

        Returns:
            Dictionary with feature statistics
        """
        stats = {}
        feature_cols = [col for col in self.feature_columns if col in data.columns]

        for col in feature_cols:
            series = data[col].dropna()
            if len(series) > 0:
                stats[col] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'count': len(series)
                }

        return stats

    def save_scaler(self, path: str):
        """Save the fitted scaler to disk."""
        if self.scaler and self.is_fitted:
            import joblib
            joblib.dump(self.scaler, path)
            logger.info(f"Scaler saved to {path}")

    def load_scaler(self, path: str):
        """Load a fitted scaler from disk."""
        import joblib
        self.scaler = joblib.load(path)
        self.is_fitted = True
        logger.info(f"Scaler loaded from {path}")


def create_feature_pipeline(config: Optional[Dict[str, Any]] = None) -> FeatureExtractor:
    """
    Create a feature extraction pipeline with default or custom configuration.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured FeatureExtractor instance
    """
    return FeatureExtractor(config)


def extract_features_for_symbol(data: pd.DataFrame, symbol: str,
                               config: Optional[Dict[str, Any]] = None) -> Tuple[str, pd.DataFrame]:
    """
    Extract features for a specific symbol.

    Args:
        data: OHLCV data for the symbol
        symbol: Symbol name
        config: Feature extraction configuration

    Returns:
        Tuple of (symbol, features DataFrame)
    """
    extractor = FeatureExtractor(config)
    features = extractor.extract_features(data)
    return symbol, features


def batch_extract_features(data_dict: Dict[str, pd.DataFrame],
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
    """
    Extract features for multiple symbols in batch.

    Args:
        data_dict: Dictionary mapping symbols to their OHLCV data
        config: Feature extraction configuration

    Returns:
        Dictionary mapping symbols to their feature DataFrames
    """
    extractor = FeatureExtractor(config)
    results = {}

    for symbol, data in data_dict.items():
        try:
            features = extractor.extract_features(data, fit_scaler=False)  # Use same scaler
            if not features.empty:
                results[symbol] = features
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")

    return results
