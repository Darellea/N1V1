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
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.exceptions import NotFittedError
from scipy.stats import ks_2samp, entropy
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Feature extraction pipeline for trading data.

    Handles the complete pipeline from raw OHLCV data to ML-ready features.
    Uses dependency injection for indicator functions to reduce coupling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 calculate_all_indicators_func: Optional[Callable] = None,
                 get_indicator_names_func: Optional[Callable] = None,
                 validate_ohlcv_data_func: Optional[Callable] = None):
        """
        Initialize the feature extractor.

        Args:
            config: Configuration dictionary for feature extraction
            calculate_all_indicators_func: Function to calculate indicators (dependency injection)
            get_indicator_names_func: Function to get indicator names (dependency injection)
            validate_ohlcv_data_func: Function to validate OHLCV data (dependency injection)
        """
        self.config = config or self._get_default_config()
        self.scaler = None
        self.feature_columns = []
        self.is_fitted = False

        # Dependency injection for indicator functions to reduce coupling
        # If not provided, import default implementations (for backward compatibility)
        if calculate_all_indicators_func is None:
            from ml.indicators import calculate_all_indicators
            self.calculate_all_indicators = calculate_all_indicators
        else:
            self.calculate_all_indicators = calculate_all_indicators_func

        if get_indicator_names_func is None:
            from ml.indicators import get_indicator_names
            self.get_indicator_names = get_indicator_names
        else:
            self.get_indicator_names = get_indicator_names_func

        if validate_ohlcv_data_func is None:
            from ml.indicators import validate_ohlcv_data
            self.validate_ohlcv_data = validate_ohlcv_data
        else:
            self.validate_ohlcv_data = validate_ohlcv_data_func

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
                'periods': [1, 2, 3, 5, 10],  # periods to lag
                'key_features': ['close', 'rsi', 'ema', 'macd', 'bb_upper', 'bb_lower', 'atr', 'adx']  # features to create lags for
            },
            'price_features': {
                'returns': True,
                'log_returns': True,
                'price_changes': True
            },
            'volume_features': {
                'volume_sma': True,
                'volume_ratio': True,
                'window_size': 20  # Configurable window size for volume-based features
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

        if not self.validate_ohlcv_data(data):
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

        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error in feature extraction: {e}")
            raise
        except (KeyError, AttributeError) as e:
            logger.error(f"Data structure error in feature extraction: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in feature extraction: {e}")
            raise

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        indicator_params = self.config.get('indicator_params', {})
        df = self.calculate_all_indicators(data, indicator_params)

        # Store feature columns for later use
        indicator_names = self.get_indicator_names()
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
        """
        Add volume-based features.

        Uses a configurable window size for rolling calculations to allow tuning
        for different market conditions and strategies, improving feature flexibility.
        """
        df = data.copy()
        volume_config = self.config.get('volume_features', {})
        window_size = volume_config.get('window_size', 20)  # Configurable window size for volume features

        if volume_config.get('volume_sma', True):
            df[f'volume_sma_{window_size}'] = df['volume'].rolling(window=window_size).mean()
            self.feature_columns.append(f'volume_sma_{window_size}')

        if volume_config.get('volume_ratio', True):
            df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
            self.feature_columns.append('volume_ratio')

        # Additional volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=window_size).mean()

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

        # Create lagged versions of key features (configurable for flexibility)
        key_features = lagged_config.get('key_features', ['close', 'rsi', 'ema', 'macd', 'bb_upper', 'bb_lower', 'atr', 'adx'])

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

            # Check if scaler is fitted before transforming (prevents NotFittedError in production)
            if not self.is_fitted:
                raise RuntimeError("Scaler is not fitted. Please fit the scaler first by calling extract_features with fit_scaler=True or load a pre-fitted scaler.")

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

        except NotFittedError:
            raise RuntimeError("Scaler is not fitted. Please fit the scaler first by calling extract_features with fit_scaler=True or load a pre-fitted scaler.")
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

    def add_cross_asset_features(self, data: pd.DataFrame, asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Add cross-asset features (correlations, spreads).

        Args:
            data: Primary asset data
            asset_data: Dictionary of other asset data for cross-asset analysis

        Returns:
            DataFrame with cross-asset features added
        """
        df = data.copy()

        if not asset_data:
            return df

        # Rolling correlation features
        window_sizes = [20, 50, 100]
        for asset_name, asset_df in asset_data.items():
            if 'close' in asset_df.columns and len(asset_df) == len(df):
                for window in window_sizes:
                    if len(df) > window:
                        corr_col = f'corr_{asset_name}_{window}'
                        df[corr_col] = df['close'].rolling(window).corr(asset_df['close'])
                        self.feature_columns.append(corr_col)

                        # Spread features
                        spread_col = f'spread_{asset_name}'
                        df[spread_col] = df['close'] - asset_df['close']
                        self.feature_columns.append(spread_col)

                        # Relative strength
                        rel_strength_col = f'rel_strength_{asset_name}'
                        df[rel_strength_col] = df['close'] / asset_df['close']
                        self.feature_columns.append(rel_strength_col)

        return df

    def add_time_anchored_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-anchored features (rolling windows, volatility clusters).

        Args:
            data: Input data with datetime index

        Returns:
            DataFrame with time-anchored features added
        """
        df = data.copy()

        # Ensure we have datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
                df.index = pd.to_datetime(df.index)

        # Rolling volatility clusters
        vol_windows = [10, 20, 30, 60]
        for window in vol_windows:
            if len(df) > window:
                vol_col = f'volatility_{window}'
                df[vol_col] = df['close'].pct_change().rolling(window).std()
                self.feature_columns.append(vol_col)

                # Volatility ratio (current vs historical)
                vol_ratio_col = f'vol_ratio_{window}'
                df[vol_ratio_col] = df[vol_col] / df[vol_col].rolling(window*2).mean()
                self.feature_columns.append(vol_ratio_col)

        # Time-of-day features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour_of_day'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month_of_year'] = df.index.month
            df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

            time_features = ['hour_of_day', 'day_of_week', 'month_of_year', 'is_weekend']
            self.feature_columns.extend(time_features)

        # Rolling momentum features
        momentum_periods = [5, 10, 20]
        for period in momentum_periods:
            if len(df) > period:
                mom_col = f'momentum_{period}'
                df[mom_col] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
                self.feature_columns.append(mom_col)

        return df

    def detect_feature_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame,
                           threshold: float = 0.05) -> Dict[str, Dict[str, Any]]:
        """
        Detect feature drift using Kolmogorov-Smirnov tests and PSI.

        Args:
            reference_data: Reference (training) data
            current_data: Current (production) data
            threshold: Drift detection threshold

        Returns:
            Dictionary with drift detection results per feature
        """
        drift_results = {}

        common_features = set(reference_data.columns) & set(current_data.columns)
        common_features = [f for f in common_features if f in self.feature_columns]

        for feature in common_features:
            ref_values = reference_data[feature].dropna().values
            curr_values = current_data[feature].dropna().values

            if len(ref_values) == 0 or len(curr_values) == 0:
                drift_results[feature] = {
                    'drift_detected': False,
                    'ks_statistic': None,
                    'psi_score': None,
                    'reason': 'Insufficient data'
                }
                continue

            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_pvalue = ks_2samp(ref_values, curr_values)
                ks_drift = ks_pvalue < threshold
            except Exception as e:
                ks_stat, ks_pvalue, ks_drift = None, None, False

            # Population Stability Index (PSI)
            try:
                psi_score = self._calculate_psi(ref_values, curr_values)
                psi_drift = psi_score > 0.25  # Standard PSI threshold
            except Exception as e:
                psi_score, psi_drift = None, False

            # Overall drift decision
            drift_detected = (ks_drift if ks_drift is not None else False) or \
                           (psi_drift if psi_drift is not None else False)

            drift_results[feature] = {
                'drift_detected': drift_detected,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'psi_score': psi_score,
                'distribution_shift': self._detect_distribution_shift(ref_values, curr_values)
            }

        return drift_results

    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray,
                      bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for histogram

        Returns:
            PSI score
        """
        # Create histograms
        ref_hist, bin_edges = np.histogram(reference, bins=bins, density=True)
        curr_hist, _ = np.histogram(current, bins=bin_edges, density=True)

        # Avoid division by zero
        ref_hist = ref_hist + 1e-10
        curr_hist = curr_hist + 1e-10

        # Calculate PSI
        psi = np.sum((curr_hist - ref_hist) * np.log(curr_hist / ref_hist))

        return float(psi)

    def _detect_distribution_shift(self, reference: np.ndarray, current: np.ndarray) -> str:
        """
        Detect the type of distribution shift.

        Args:
            reference: Reference distribution
            current: Current distribution

        Returns:
            Type of shift detected
        """
        ref_mean, ref_std = np.mean(reference), np.std(reference)
        curr_mean, curr_std = np.mean(current), np.std(current)

        mean_shift = abs(curr_mean - ref_mean) / abs(ref_mean) if ref_mean != 0 else 0
        std_shift = abs(curr_std - ref_std) / abs(ref_std) if ref_std != 0 else 0

        if mean_shift > 0.1 and std_shift < 0.1:
            return "mean_shift"
        elif std_shift > 0.1 and mean_shift < 0.1:
            return "variance_shift"
        elif mean_shift > 0.1 and std_shift > 0.1:
            return "mean_and_variance_shift"
        else:
            return "no_significant_shift"

    def get_drift_report(self, drift_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive drift report.

        Args:
            drift_results: Results from detect_feature_drift

        Returns:
            Drift report summary
        """
        total_features = len(drift_results)
        drifted_features = sum(1 for result in drift_results.values() if result['drift_detected'])

        report = {
            'total_features': total_features,
            'drifted_features': drifted_features,
            'drift_percentage': drifted_features / total_features if total_features > 0 else 0,
            'most_drifted_features': self._get_most_drifted_features(drift_results),
            'drift_summary': {
                'ks_based_drift': sum(1 for r in drift_results.values()
                                    if r.get('ks_pvalue', 1.0) < 0.05),
                'psi_based_drift': sum(1 for r in drift_results.values()
                                     if r.get('psi_score', 0) > 0.25)
            }
        }

        return report

    def _get_most_drifted_features(self, drift_results: Dict[str, Dict[str, Any]],
                                 top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most drifted features.

        Args:
            drift_results: Drift detection results
            top_n: Number of top features to return

        Returns:
            List of most drifted features with their scores
        """
        feature_scores = []

        for feature, results in drift_results.items():
            # Combine KS statistic and PSI score for ranking
            ks_score = results.get('ks_statistic', 0) or 0
            psi_score = results.get('psi_score', 0) or 0
            combined_score = (ks_score + psi_score) / 2

            feature_scores.append({
                'feature': feature,
                'drift_score': combined_score,
                'drift_detected': results.get('drift_detected', False),
                'shift_type': results.get('distribution_shift', 'unknown')
            })

        # Sort by drift score descending
        feature_scores.sort(key=lambda x: x['drift_score'], reverse=True)

        return feature_scores[:top_n]


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
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error extracting features for {symbol}: {e}")
        except (KeyError, AttributeError) as e:
            logger.error(f"Data structure error extracting features for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error extracting features for {symbol}: {e}")

    return results


# Additional feature generation functions for testing compatibility

def generate_cross_asset_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate cross-asset features from multi-asset data.

    Args:
        data: DataFrame with multi-asset price data (columns like BTC_close, ETH_close, etc.)

    Returns:
        DataFrame with cross-asset features
    """
    if data.empty:
        return pd.DataFrame()

    # Extract asset names from column names (assuming format: ASSET_close)
    asset_columns = [col for col in data.columns if col.endswith('_close')]
    if len(asset_columns) < 2:
        raise ValueError("At least 2 assets required for cross-asset features")

    assets = [col.replace('_close', '') for col in asset_columns]
    features_df = data.copy()

    # Generate correlation features
    window_sizes = [7, 14, 30]
    for i, asset1 in enumerate(assets):
        for j, asset1 in enumerate(assets[i+1:], i+1):
            asset2 = assets[j]
            col1 = f"{asset1}_close"
            col2 = f"{asset2}_close"

            for window in window_sizes:
                if len(data) > window:
                    corr_col = f"{asset1}_{asset2}_correlation_{window}d"
                    features_df[corr_col] = data[col1].rolling(window).corr(data[col2])

                    # Spread features
                    spread_col = f"{asset1}_{asset2}_spread"
                    features_df[spread_col] = data[col1] - data[col2]

                    # Ratio features
                    ratio_col = f"{asset1}_{asset2}_ratio"
                    features_df[ratio_col] = data[col1] / data[col2]

    return features_df


def generate_time_anchored_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate time-anchored features from price data.

    Args:
        data: DataFrame with OHLCV data

    Returns:
        DataFrame with time-anchored features
    """
    if data.empty:
        return pd.DataFrame()

    if 'close' not in data.columns:
        raise ValueError("Data must contain 'close' column")

    features_df = data.copy()

    # Rolling volatility features
    vol_periods = [7, 14, 30]
    for period in vol_periods:
        if len(data) > period:
            vol_col = f"volatility_{period}d"
            features_df[vol_col] = data['close'].pct_change().rolling(period).std()

    # Momentum features
    momentum_periods = [7, 14, 30]
    for period in momentum_periods:
        if len(data) > period:
            mom_col = f"momentum_{period}d"
            features_df[mom_col] = (data['close'] - data['close'].shift(period)) / data['close'].shift(period)

    # Volume-based features if volume column exists
    if 'volume' in data.columns:
        vol_ma_periods = [7, 14, 30]
        for period in vol_ma_periods:
            if len(data) > period:
                vol_ma_col = f"volume_ma_{period}d"
                features_df[vol_ma_col] = data['volume'].rolling(period).mean()

                vol_std_col = f"volume_std_{period}d"
                features_df[vol_std_col] = data['volume'].rolling(period).std()

    return features_df


# Feature generation classes for testing

class CrossAssetFeatureGenerator:
    """Generator for cross-asset features."""

    def __init__(self):
        pass

    def generate_correlation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate correlation-based features."""
        if len(data.columns) < 2:
            raise ValueError("At least 2 assets required")

        asset_cols = [col for col in data.columns if col.endswith('_close')]
        if len(asset_cols) < 2:
            raise ValueError("At least 2 assets required")

        features_df = pd.DataFrame(index=data.index)

        for i, col1 in enumerate(asset_cols):
            for col2 in asset_cols[i+1:]:
                asset1 = col1.replace('_close', '')
                asset2 = col2.replace('_close', '')
                corr_col = f"{asset1}_{asset2}_correlation_7d"
                features_df[corr_col] = data[col1].rolling(7).corr(data[col2])

        return features_df

    def generate_spread_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate spread-based features."""
        asset_cols = [col for col in data.columns if col.endswith('_close')]
        if len(asset_cols) < 2:
            raise ValueError("At least 2 assets required")

        features_df = pd.DataFrame(index=data.index)

        for i, col1 in enumerate(asset_cols):
            for col2 in asset_cols[i+1:]:
                asset1 = col1.replace('_close', '')
                asset2 = col2.replace('_close', '')
                spread_col = f"{asset1}_{asset2}_spread"
                features_df[spread_col] = data[col1] - data[col2]

        return features_df

    def generate_ratio_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ratio-based features."""
        asset_cols = [col for col in data.columns if col.endswith('_close')]
        if len(asset_cols) < 2:
            raise ValueError("At least 2 assets required")

        features_df = pd.DataFrame(index=data.index)

        for i, col1 in enumerate(asset_cols):
            for col2 in asset_cols[i+1:]:
                asset1 = col1.replace('_close', '')
                asset2 = col2.replace('_close', '')
                ratio_col = f"{asset1}_{asset2}_ratio"
                features_df[ratio_col] = data[col1] / data[col2]

        return features_df


class TimeAnchoredFeatureGenerator:
    """Generator for time-anchored features."""

    def __init__(self):
        pass

    def generate_rolling_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling volatility features."""
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")

        if len(data) < 10:
            raise ValueError("Insufficient data")

        features_df = pd.DataFrame(index=data.index)

        periods = [7, 14, 30]
        for period in periods:
            if len(data) > period:
                vol_col = f"volatility_{period}d"
                features_df[vol_col] = data['close'].pct_change().rolling(period).std()

        return features_df

    def generate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum features."""
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")

        if len(data) < 10:
            raise ValueError("Insufficient data")

        features_df = pd.DataFrame(index=data.index)

        periods = [7, 14, 30]
        for period in periods:
            if len(data) > period:
                mom_col = f"momentum_{period}d"
                features_df[mom_col] = (data['close'] - data['close'].shift(period)) / data['close'].shift(period)

        return features_df

    def generate_volume_profile_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume profile features."""
        if 'volume' not in data.columns:
            raise ValueError("Data must contain 'volume' column")

        if len(data) < 10:
            raise ValueError("Insufficient data")

        features_df = pd.DataFrame(index=data.index)

        period = 7
        if len(data) > period:
            vol_ma_col = f"volume_ma_{period}d"
            features_df[vol_ma_col] = data['volume'].rolling(period).mean()

            vol_std_col = f"volume_std_{period}d"
            features_df[vol_std_col] = data['volume'].rolling(period).std()

            vol_zscore_col = f"volume_zscore_{period}d"
            features_df[vol_zscore_col] = (data['volume'] - features_df[vol_ma_col]) / features_df[vol_std_col]

        return features_df


class FeatureDriftDetector:
    """Detector for feature drift."""

    def __init__(self, method: str = 'ks'):
        """
        Initialize drift detector.

        Args:
            method: Drift detection method ('ks' or 'psi')
        """
        self.method = method
        if method not in ['ks', 'psi']:
            raise ValueError("Unsupported drift detection method")

    def detect_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect drift between reference and current data.

        Args:
            reference_data: Reference data
            current_data: Current data

        Returns:
            Dictionary of drift scores per feature
        """
        common_features = set(reference_data.columns) & set(current_data.columns)
        drift_scores = {}

        for feature in common_features:
            ref_vals = reference_data[feature].dropna().values
            curr_vals = current_data[feature].dropna().values

            if len(ref_vals) == 0 or len(curr_vals) == 0:
                drift_scores[feature] = 0.0
                continue

            if self.method == 'ks':
                from scipy.stats import ks_2samp
                try:
                    stat, _ = ks_2samp(ref_vals, curr_vals)
                    drift_scores[feature] = stat
                except:
                    drift_scores[feature] = 0.0
            elif self.method == 'psi':
                drift_scores[feature] = self._calculate_psi(ref_vals, curr_vals)

        return drift_scores

    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Population Stability Index."""
        # Simple PSI calculation
        ref_mean = np.mean(reference)
        curr_mean = np.mean(current)

        if ref_mean == 0:
            return 0.0

        return abs(curr_mean - ref_mean) / ref_mean


def detect_feature_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame,
                        method: str = 'ks') -> Tuple[bool, Dict[str, float]]:
    """
    Detect feature drift between reference and current data.

    Args:
        reference_data: Reference data
        current_data: Current data
        method: Drift detection method

    Returns:
        Tuple of (drift_detected, drift_scores)
    """
    detector = FeatureDriftDetector(method)
    scores = detector.detect_drift(reference_data, current_data)
    drift_detected = any(score > 0.05 for score in scores.values())  # Simple threshold
    return drift_detected, scores


def validate_features(data: pd.DataFrame, check_missing: bool = True,
                     check_ranges: bool = False, check_correlations: bool = False,
                     check_importance_stability: bool = False,
                     max_missing_ratio: float = 0.1, max_correlation: float = 0.95) -> Tuple[bool, List[str]]:
    """
    Validate features in a DataFrame.

    Args:
        data: Feature DataFrame to validate
        check_missing: Check for missing values
        check_ranges: Check for extreme values
        check_correlations: Check for high correlations
        check_importance_stability: Check feature importance stability
        max_missing_ratio: Maximum allowed missing ratio
        max_correlation: Maximum allowed correlation

    Returns:
        Tuple of (is_valid, issues_list)
    """
    issues = []
    is_valid = True

    if check_missing:
        for col in data.columns:
            missing_ratio = data[col].isna().mean()
            if missing_ratio > max_missing_ratio:
                issues.append(f"Column '{col}' has {missing_ratio:.2%} missing values")
                is_valid = False

    if check_ranges:
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                mean_val = data[col].mean()
                std_val = data[col].std()
                if std_val > 0:
                    extreme_count = ((data[col] - mean_val).abs() > 5 * std_val).sum()
                    if extreme_count > 0:
                        issues.append(f"Column '{col}' has {extreme_count} extreme values")
                        is_valid = False

    if check_correlations:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = abs(corr_matrix.iloc[i, j])
                    if corr > max_correlation:
                        issues.append(f"High correlation ({corr:.2f}) between '{numeric_cols[i]}' and '{numeric_cols[j]}'")
                        is_valid = False

    return is_valid, issues
