"""
Shared Indicators Cache Module

This module provides caching functionality for technical indicators to avoid
redundant calculations across multiple strategies and improve performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Tuple
from functools import lru_cache
import hashlib
import logging

logger = logging.getLogger(__name__)


class IndicatorsCache:
    """
    Cache for technical indicators to avoid redundant calculations.

    This class provides caching for commonly used indicators like RSI, Bollinger Bands,
    ATR, etc. The cache uses a combination of data hash and parameters to ensure
    data integrity and parameter-specific caching.
    """

    def __init__(self, max_cache_size: int = 100):
        """
        Initialize the indicators cache.

        Args:
            max_cache_size: Maximum number of cached results to store
        """
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, Any] = {}
        self.cache_order: list = []  # For LRU eviction

    def _get_cache_key(self, data: pd.DataFrame, indicator_name: str, **params) -> str:
        """
        Generate a unique cache key based on data content and parameters.

        Args:
            data: DataFrame containing market data
            indicator_name: Name of the indicator
            **params: Indicator parameters

        Returns:
            Unique cache key string
        """
        # Create a hash of the data (using last few rows and key columns)
        if len(data) > 100:
            # Use last 100 rows for hash to balance accuracy and performance
            data_sample = data.iloc[-100:]
        else:
            data_sample = data

        # Include key columns that affect indicator calculation
        key_columns = ['close', 'high', 'low', 'volume']
        available_columns = [col for col in key_columns if col in data_sample.columns]

        if available_columns:
            data_str = data_sample[available_columns].to_string()
        else:
            data_str = str(data_sample.values.tobytes())

        data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]

        # Include parameters in key
        params_str = str(sorted(params.items()))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        return f"{indicator_name}_{data_hash}_{params_hash}"

    def get(self, data: pd.DataFrame, indicator_name: str, **params) -> Optional[pd.Series]:
        """
        Retrieve cached indicator result if available.

        Args:
            data: DataFrame containing market data
            indicator_name: Name of the indicator
            **params: Indicator parameters

        Returns:
            Cached Series if available, None otherwise
        """
        cache_key = self._get_cache_key(data, indicator_name, **params)

        if cache_key in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(cache_key)
            self.cache_order.append(cache_key)
            logger.debug(f"Cache hit for {indicator_name}")
            return self.cache[cache_key]

        logger.debug(f"Cache miss for {indicator_name}")
        return None

    def put(self, data: pd.DataFrame, indicator_name: str, result: pd.Series, **params) -> None:
        """
        Store indicator result in cache.

        Args:
            data: DataFrame containing market data
            indicator_name: Name of the indicator
            result: Indicator result to cache
            **params: Indicator parameters
        """
        cache_key = self._get_cache_key(data, indicator_name, **params)

        # Evict oldest if cache is full
        if len(self.cache) >= self.max_cache_size:
            oldest_key = self.cache_order.pop(0)
            del self.cache[oldest_key]

        self.cache[cache_key] = result
        self.cache_order.append(cache_key)
        logger.debug(f"Cached {indicator_name} with key {cache_key}")

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self.cache_order.clear()
        logger.info("Indicators cache cleared")

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'cache_hit_ratio': 0.0  # Could be implemented with hit/miss counters
        }


# Global cache instance
indicators_cache = IndicatorsCache()


def cached_indicator(indicator_func):
    """
    Decorator to add caching to indicator functions.

    Args:
        indicator_func: Function to be cached

    Returns:
        Wrapped function with caching
    """
    def wrapper(data: pd.DataFrame, **params):
        # Try to get from cache first
        cached_result = indicators_cache.get(data, indicator_func.__name__, **params)
        if cached_result is not None:
            return cached_result

        # Calculate and cache result
        result = indicator_func(data, **params)
        indicators_cache.put(data, indicator_func.__name__, result, **params)
        return result

    return wrapper


# Vectorized indicator calculation functions with caching

@cached_indicator
def calculate_rsi_vectorized(data: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    Calculate RSI using vectorized operations for better performance.

    Args:
        data: DataFrame with OHLCV data
        period: RSI calculation period
        column: Column to use for calculation

    Returns:
        Series with RSI values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")

    if len(data) < period:
        return pd.Series([np.nan] * len(data), index=data.index)

    delta = data[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


@cached_indicator
def calculate_bollinger_bands_vectorized(data: pd.DataFrame, period: int = 20,
                                       std_dev: float = 2.0, column: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands using vectorized operations.

    Args:
        data: DataFrame with OHLCV data
        period: Moving average period
        std_dev: Standard deviation multiplier
        column: Column to use for calculation

    Returns:
        Tuple of (Upper Band, Middle Band/SMA, Lower Band)
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")

    if len(data) < period:
        nan_series = pd.Series([np.nan] * len(data), index=data.index)
        return nan_series, nan_series, nan_series

    sma = data[column].rolling(window=period).mean()
    std = data[column].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    return upper_band, sma, lower_band


@cached_indicator
def calculate_atr_vectorized(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate ATR using vectorized operations.

    Args:
        data: DataFrame with OHLCV data
        period: ATR calculation period

    Returns:
        Series with ATR values
    """
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    if len(data) < period:
        return pd.Series([np.nan] * len(data), index=data.index)

    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift(1)).abs()
    low_close = (data['low'] - data['close'].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def calculate_indicators_for_multi_symbol(data: pd.DataFrame, indicators_config: Dict[str, Dict]) -> pd.DataFrame:
    """
    Calculate multiple indicators for multi-symbol data using vectorized operations.

    This function processes all symbols at once using vectorized pandas/numpy operations
    instead of inefficient groupby operations, providing significant performance improvements
    for large datasets with many symbols.

    Args:
        data: DataFrame with symbol column and OHLCV data
        indicators_config: Dictionary specifying which indicators to calculate
                          Format: {'indicator_name': {'param1': value1, 'param2': value2}}

    Returns:
        DataFrame with calculated indicators

    Example:
        config = {
            'rsi': {'period': 14},
            'bb': {'period': 20, 'std_dev': 2.0},
            'atr': {'period': 14}
        }
        result = calculate_indicators_for_multi_symbol(data, config)
    """
    if data.empty:
        return data

    result_df = data.copy()

    # Process each indicator
    for indicator_name, params in indicators_config.items():
        try:
            if indicator_name == 'rsi':
                period = params.get('period', 14)
                rsi = calculate_rsi_vectorized(result_df, period=period)
                result_df['rsi'] = rsi

            elif indicator_name == 'bb':
                period = params.get('period', 20)
                std_dev = params.get('std_dev', 2.0)
                upper, middle, lower = calculate_bollinger_bands_vectorized(
                    result_df, period=period, std_dev=std_dev
                )
                result_df['bb_upper'] = upper
                result_df['bb_middle'] = middle
                result_df['bb_lower'] = lower

                # Calculate position within bands (vectorized)
                band_range = result_df['bb_upper'] - result_df['bb_lower']
                result_df['bb_position'] = np.where(
                    band_range == 0,
                    np.nan,
                    (result_df['close'] - result_df['bb_lower']) / band_range
                )

                # Calculate band width
                result_df['bb_width'] = band_range / result_df['bb_middle']

            elif indicator_name == 'atr':
                period = params.get('period', 14)
                result_df['atr'] = calculate_atr_vectorized(result_df, period=period)

            elif indicator_name == 'ema':
                period = params.get('period', 20)
                result_df['ema'] = result_df['close'].ewm(span=period, adjust=False).mean()

            elif indicator_name == 'sma':
                period = params.get('period', 20)
                result_df['sma'] = result_df['close'].rolling(window=period).mean()

            elif indicator_name == 'keltner':
                sma_period = params.get('sma_period', 20)
                atr_period = params.get('atr_period', 14)
                atr_multiplier = params.get('atr_multiplier', 2.0)

                # Calculate SMA for middle line
                result_df['keltner_middle'] = result_df['close'].rolling(window=sma_period).mean()

                # Calculate ATR for bands
                high_low = result_df['high'] - result_df['low']
                high_close = (result_df['high'] - result_df['close'].shift(1)).abs()
                low_close = (result_df['low'] - result_df['close'].shift(1)).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=atr_period).mean()

                # Calculate Keltner Channel bands
                result_df['keltner_upper'] = result_df['keltner_middle'] + (atr * atr_multiplier)
                result_df['keltner_lower'] = result_df['keltner_middle'] - (atr * atr_multiplier)

                # Calculate position within channel
                band_range = result_df['keltner_upper'] - result_df['keltner_lower']
                result_df['keltner_position'] = np.where(
                    band_range == 0,
                    np.nan,
                    (result_df['close'] - result_df['keltner_lower']) / band_range
                )

                # Calculate channel width
                result_df['keltner_width'] = band_range / result_df['keltner_middle']

            elif indicator_name == 'donchian':
                period = params.get('period', 20)

                # Calculate Donchian Channel using rolling operations
                result_df['donchian_high'] = result_df['high'].rolling(window=period).max()
                result_df['donchian_low'] = result_df['low'].rolling(window=period).min()
                result_df['donchian_mid'] = (result_df['donchian_high'] + result_df['donchian_low']) / 2

                # Calculate channel width
                result_df['donchian_width'] = result_df['donchian_high'] - result_df['donchian_low']
                # Avoid division by zero
                result_df['donchian_width_pct'] = np.where(
                    result_df['donchian_mid'] == 0,
                    np.nan,
                    result_df['donchian_width'] / result_df['donchian_mid']
                )

            logger.debug(f"Calculated {indicator_name} for multi-symbol data")

        except Exception as e:
            logger.warning(f"Failed to calculate {indicator_name}: {e}")

    return result_df
