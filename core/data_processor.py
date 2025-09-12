"""
DataProcessor - High-performance data processing component.

Implements vectorized operations, caching, and batch processing
for efficient market data analysis and signal generation.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache
import weakref

import numpy as np
import pandas as pd
from numba import jit
import hashlib

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    High-performance data processing engine with vectorized operations,
    caching, and batch processing capabilities.
    """

    def __init__(self, cache_size: int = 1000, cache_ttl: float = 300.0):
        """Initialize the DataProcessor.

        Args:
            cache_size: Maximum number of cached results
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl

        # Cache for computed indicators
        self._indicator_cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_access_times: Dict[str, float] = {}

        # Batch processing queues
        self._batch_queue: List[Tuple[str, Any]] = []
        self._batch_size = 10

        # Memory monitoring
        self._memory_usage = 0
        self._object_pool: Dict[str, List[Any]] = {}

        # Performance metrics
        self._operation_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, operation: str, data_hash: str, params: Dict[str, Any]) -> str:
        """Generate a unique cache key for an operation."""
        params_str = str(sorted(params.items()))
        key_str = f"{operation}:{data_hash}:{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_data_hash(self, data: pd.DataFrame) -> str:
        """Generate a hash of the data for cache keying."""
        if data.empty:
            return "empty"

        # Use a sample of the data for hashing to avoid performance issues
        sample = data.head(10).to_string()
        return hashlib.md5(sample.encode()).hexdigest()[:8]

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is still valid."""
        if cache_key not in self._cache_access_times:
            return False

        age = time.time() - self._cache_access_times[cache_key]
        return age < self.cache_ttl

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get a result from cache if available and valid."""
        if not self._is_cache_valid(cache_key):
            if cache_key in self._indicator_cache:
                del self._indicator_cache[cache_key]
                del self._cache_access_times[cache_key]
            return None

        self._cache_hits += 1
        self._cache_access_times[cache_key] = time.time()
        return self._indicator_cache[cache_key][0]

    def _cache_result(self, cache_key: str, result: Any):
        """Cache a computation result."""
        current_time = time.time()

        # Clean up expired entries if cache is full
        if len(self._indicator_cache) >= self.cache_size:
            expired_keys = [
                k for k, t in self._cache_access_times.items()
                if current_time - t > self.cache_ttl
            ]
            for k in expired_keys:
                if k in self._indicator_cache:
                    del self._indicator_cache[k]
                del self._cache_access_times[k]

        # If still full, remove oldest entries
        if len(self._indicator_cache) >= self.cache_size:
            oldest_keys = sorted(
                self._cache_access_times.keys(),
                key=lambda k: self._cache_access_times[k]
            )[:10]  # Remove 10 oldest

            for k in oldest_keys:
                if k in self._indicator_cache:
                    del self._indicator_cache[k]
                del self._cache_access_times[k]

        # Store the result
        self._indicator_cache[cache_key] = (result, current_time)
        self._cache_access_times[cache_key] = current_time
        self._cache_misses += 1

    @staticmethod
    @jit(nopython=True)
    def _vectorized_rsi(prices: np.ndarray, period: int) -> np.ndarray:
        """Vectorized RSI calculation using Numba for performance."""
        n = len(prices)
        rsi_values = np.full(n, np.nan)

        if n < period + 1:
            return rsi_values

        # Calculate price changes
        delta = np.zeros(n)
        delta[1:] = prices[1:] - prices[:-1]

        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Calculate initial averages
        avg_gain = np.mean(gains[1:period+1])
        avg_loss = np.mean(losses[1:period+1])

        if avg_loss == 0:
            rsi_values[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[period] = 100 - (100 / (1 + rs))

        # Calculate subsequent values using Wilder's smoothing
        for i in range(period + 1, n):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi_values[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100 - (100 / (1 + rs))

        return rsi_values

    def calculate_rsi_batch(self, data_dict: Dict[str, pd.DataFrame],
                           period: int = 14) -> Dict[str, pd.DataFrame]:
        """Calculate RSI for multiple symbols in batch."""
        results = {}

        for symbol, data in data_dict.items():
            if data.empty or len(data) < period:
                results[symbol] = data.copy()
                continue

            # Check cache first
            data_hash = self._get_data_hash(data)
            cache_key = self._get_cache_key('rsi', data_hash, {'period': period})
            cached_result = self._get_cached_result(cache_key)

            if cached_result is not None:
                results[symbol] = cached_result
                continue

            # Calculate RSI using vectorized approach
            prices = data['close'].values
            rsi_values = self._vectorized_rsi(prices, period)

            # Create result DataFrame
            result_df = data.copy()
            result_df['rsi'] = rsi_values

            # Cache the result
            self._cache_result(cache_key, result_df)
            results[symbol] = result_df

        return results

    @staticmethod
    @jit(nopython=True)
    def _vectorized_sma(values: np.ndarray, period: int) -> np.ndarray:
        """Vectorized Simple Moving Average calculation."""
        n = len(values)
        sma_values = np.full(n, np.nan)

        if n < period:
            return sma_values

        # Calculate cumulative sum for efficient moving average
        cumsum = np.cumsum(values)
        sma_values[period-1:] = (cumsum[period-1:] - np.roll(cumsum, period)[period-1:]) / period

        return sma_values

    @staticmethod
    @jit(nopython=True)
    def _vectorized_ema(values: np.ndarray, period: int) -> np.ndarray:
        """Vectorized Exponential Moving Average calculation."""
        n = len(values)
        ema_values = np.full(n, np.nan)

        if n < period:
            return ema_values

        # Calculate multiplier
        multiplier = 2 / (period + 1)

        # Calculate initial SMA
        ema_values[period-1] = np.mean(values[:period])

        # Calculate subsequent EMA values
        for i in range(period, n):
            ema_values[i] = (values[i] - ema_values[i-1]) * multiplier + ema_values[i-1]

        return ema_values

    def calculate_moving_averages_batch(self, data_dict: Dict[str, pd.DataFrame],
                                       periods: List[int] = None,
                                       ma_type: str = 'sma') -> Dict[str, pd.DataFrame]:
        """Calculate moving averages for multiple symbols in batch."""
        if periods is None:
            periods = [10, 20, 50]

        results = {}

        for symbol, data in data_dict.items():
            if data.empty:
                results[symbol] = data.copy()
                continue

            result_df = data.copy()

            for period in periods:
                if len(data) < period:
                    continue

                # Check cache
                data_hash = self._get_data_hash(data)
                cache_key = self._get_cache_key(f'{ma_type}_{period}', data_hash, {})
                cached_result = self._get_cached_result(cache_key)

                if cached_result is not None:
                    result_df[f'{ma_type}_{period}'] = cached_result
                    continue

                # Calculate moving average
                prices = data['close'].values

                if ma_type == 'sma':
                    ma_values = self._vectorized_sma(prices, period)
                elif ma_type == 'ema':
                    ma_values = self._vectorized_ema(prices, period)
                else:
                    continue

                result_df[f'{ma_type}_{period}'] = ma_values

                # Cache individual MA calculation
                self._cache_result(cache_key, ma_values)

            results[symbol] = result_df

        return results

    @staticmethod
    @jit(nopython=True)
    def _vectorized_bollinger_bands(prices: np.ndarray, period: int, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized Bollinger Bands calculation."""
        n = len(prices)
        upper = np.full(n, np.nan)
        middle = np.full(n, np.nan)
        lower = np.full(n, np.nan)

        if n < period:
            return upper, middle, lower

        for i in range(period - 1, n):
            window = prices[max(0, i - period + 1):i + 1]
            mean = np.mean(window)
            std = np.std(window)

            middle[i] = mean
            upper[i] = mean + (std_dev * std)
            lower[i] = mean - (std_dev * std)

        return upper, middle, lower

    def calculate_technical_indicators_batch(self, data_dict: Dict[str, pd.DataFrame],
                                           indicators: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Calculate multiple technical indicators in batch for performance."""
        if indicators is None:
            indicators = ['rsi', 'sma', 'bb']

        results = {}

        for symbol, data in data_dict.items():
            if data.empty:
                results[symbol] = data.copy()
                continue

            result_df = data.copy()

            # RSI
            if 'rsi' in indicators:
                rsi_data = self.calculate_rsi_batch({symbol: data}, period=14)
                if symbol in rsi_data and 'rsi' in rsi_data[symbol].columns:
                    result_df['rsi'] = rsi_data[symbol]['rsi']

            # Moving Averages
            if 'sma' in indicators:
                sma_data = self.calculate_moving_averages_batch({symbol: data},
                                                              periods=[10, 20, 50],
                                                              ma_type='sma')
                if symbol in sma_data:
                    for col in sma_data[symbol].columns:
                        if col.startswith('sma_'):
                            result_df[col] = sma_data[symbol][col]

            # Bollinger Bands
            if 'bb' in indicators and len(data) >= 20:
                prices = data['close'].values
                upper, middle, lower = self._vectorized_bollinger_bands(prices, 20)

                result_df['bb_upper'] = upper
                result_df['bb_middle'] = middle
                result_df['bb_lower'] = lower

            results[symbol] = result_df

        return results

    def add_binary_labels_batch(self, data_dict: Dict[str, pd.DataFrame],
                               horizon: int = 5, profit_threshold: float = 0.005,
                               include_fees: bool = True, fee_rate: float = 0.001) -> Dict[str, pd.DataFrame]:
        """
        Add binary labels to multiple datasets in batch.

        Args:
            data_dict: Dictionary of symbol to DataFrame
            horizon: Number of periods ahead to look for forward return
            profit_threshold: Minimum profit threshold after fees (fractional)
            include_fees: Whether to account for trading fees
            fee_rate: Trading fee rate (fractional)

        Returns:
            Dictionary with binary labels added
        """
        from ml.trainer import create_binary_labels

        results = {}

        for symbol, data in data_dict.items():
            if data.empty:
                results[symbol] = data.copy()
                continue

            try:
                # Add binary labels
                labeled_data = create_binary_labels(
                    df=data,
                    horizon=horizon,
                    profit_threshold=profit_threshold,
                    include_fees=include_fees,
                    fee_rate=fee_rate
                )
                results[symbol] = labeled_data
                logger.info(f"Added binary labels to {symbol} dataset")

            except Exception as e:
                logger.error(f"Failed to add binary labels to {symbol}: {e}")
                results[symbol] = data.copy()

        return results

    def batch_process_signals(self, data_dict: Dict[str, pd.DataFrame],
                            signal_functions: List[callable]) -> Dict[str, List[Any]]:
        """Process signals for multiple symbols in batch."""
        results = {}

        for symbol, data in data_dict.items():
            if data.empty:
                results[symbol] = []
                continue

            signals = []
            for signal_func in signal_functions:
                try:
                    symbol_signals = signal_func(symbol, data)
                    if symbol_signals:
                        signals.extend(symbol_signals)
                except Exception as e:
                    logger.exception(f"Error in signal function for {symbol}: {e}")

            results[symbol] = signals

        return results

    def get_object_from_pool(self, object_type: str, factory_func: callable, *args, **kwargs):
        """Get an object from the pool or create a new one."""
        if object_type not in self._object_pool:
            self._object_pool[object_type] = []

        pool = self._object_pool[object_type]

        # Try to find an available object
        for obj in pool:
            if hasattr(obj, '_in_use') and not obj._in_use:
                obj._in_use = True
                return obj

        # Create a new object
        try:
            obj = factory_func(*args, **kwargs)
            obj._in_use = True
            pool.append(obj)
            return obj
        except Exception as e:
            logger.exception(f"Failed to create object for pool: {e}")
            return None

    def return_object_to_pool(self, object_type: str, obj: Any):
        """Return an object to the pool."""
        if hasattr(obj, '_in_use'):
            obj._in_use = False

    def cleanup_pool(self, object_type: str = None):
        """Clean up object pools."""
        if object_type:
            if object_type in self._object_pool:
                # Clean up objects that have been in the pool too long
                pool = self._object_pool[object_type]
                active_objects = [obj for obj in pool if getattr(obj, '_in_use', False)]

                # Keep only active objects
                self._object_pool[object_type] = active_objects
        else:
            # Clean up all pools
            for pool_type in list(self._object_pool.keys()):
                self.cleanup_pool(pool_type)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the data processor."""
        total_operations = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_operations if total_operations > 0 else 0

        return {
            "cache_size": len(self._indicator_cache),
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_operations": total_operations,
            "object_pools": {k: len(v) for k, v in self._object_pool.items()},
            "memory_usage_estimate": self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of cached data."""
        # Rough estimation
        cache_memory = len(self._indicator_cache) * 1024  # ~1KB per cached item
        pool_memory = sum(len(pool) * 512 for pool in self._object_pool.values())  # ~512B per pooled object

        return cache_memory + pool_memory

    def clear_cache(self):
        """Clear all cached data."""
        self._indicator_cache.clear()
        self._cache_access_times.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Data processor cache cleared")

    async def lazy_load_data(self, symbol: str, data_fetcher, timeframe: str = '1h',
                           required_length: int = 100) -> Optional[pd.DataFrame]:
        """Lazily load data only when needed."""
        # Check if we have sufficient data in cache
        cache_key = f"lazy_{symbol}_{timeframe}_{required_length}"
        cached_data = self._get_cached_result(cache_key)

        if cached_data is not None and len(cached_data) >= required_length:
            return cached_data

        # Load data asynchronously
        try:
            data = await data_fetcher.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=max(required_length, 1000)  # Load extra for future use
            )

            if not data.empty and len(data) >= required_length:
                self._cache_result(cache_key, data)
                return data

        except Exception as e:
            logger.exception(f"Failed to lazy load data for {symbol}: {e}")

        return None


# Global data processor instance
_data_processor = None

def get_data_processor() -> DataProcessor:
    """Get the global data processor instance."""
    global _data_processor
    if _data_processor is None:
        _data_processor = DataProcessor()
    return _data_processor
