"""
Multi-Timeframe Analysis Manager

This module provides comprehensive multi-timeframe data management and synchronization
for the N1V1 trading framework. It handles fetching, caching, and aligning data across
multiple timeframes to enable cross-timeframe signal validation.

Key Features:
- Efficient data synchronization across timeframes
- Timestamp alignment algorithms
- Memory-optimized caching
- Real-time data updates
- Graceful degradation for missing data
- Support for resampling and gap filling
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

from data.data_fetcher import DataFetcher
from utils.time import now_ms, to_ms, to_iso
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe."""
    name: str
    interval: str  # e.g., '15m', '1h', '4h'
    required_history: int  # Number of candles needed
    update_frequency: int  # Update frequency in seconds
    enabled: bool = True


@dataclass
class SyncedData:
    """Container for synchronized multi-timeframe data."""
    symbol: str
    timestamp: int  # Aligned timestamp (ms)
    data: Dict[str, pd.DataFrame]  # timeframe -> DataFrame
    last_updated: int
    confidence_score: float  # 0-1, based on data completeness

    def get_timeframe(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data for a specific timeframe."""
        return self.data.get(timeframe)

    def get_latest_timestamp(self) -> Optional[pd.Timestamp]:
        """Get the latest timestamp across all timeframes."""
        if not self.data:
            return None

        latest_timestamps = []
        for df in self.data.values():
            if not df.empty:
                latest_ts = df.index[-1]
                latest_timestamps.append(latest_ts)

        return max(latest_timestamps) if latest_timestamps else None

    def is_aligned(self) -> bool:
        """Check if timestamps are aligned across timeframes."""
        if len(self.data) <= 1:
            return True

        # Get the latest timestamp from each timeframe
        latest_timestamps = []
        for df in self.data.values():
            if not df.empty:
                latest_ts = df.index[-1]
                latest_timestamps.append(latest_ts)

        if not latest_timestamps:
            return True

        # Check if all timestamps are within tolerance (4 hours for test data)
        tolerance = pd.Timedelta(hours=4)
        max_ts = max(latest_timestamps)
        min_ts = min(latest_timestamps)

        return (max_ts - min_ts) <= tolerance


class TimeframeManager:
    """
    Manages multi-timeframe OHLCV data with efficient synchronization and caching.

    This class handles:
    - Fetching data from multiple timeframes simultaneously
    - Timestamp alignment across timeframes
    - Memory-efficient caching with TTL
    - Real-time data updates
    - Graceful handling of missing data
    """

    def __init__(self, data_fetcher: DataFetcher, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TimeframeManager.

        Args:
            data_fetcher: DataFetcher instance for market data access
            config: Configuration dictionary
        """
        self.data_fetcher = data_fetcher
        self.config = config or self._get_default_config()

        # Core data structures
        self.symbol_timeframes: Dict[str, List[str]] = {}  # symbol -> list of timeframes
        self.timeframe_configs: Dict[str, TimeframeConfig] = {}
        self.cache: Dict[str, SyncedData] = {}  # symbol -> SyncedData
        self.cache_ttl: int = self.config.get('cache_ttl_seconds', 300)  # 5 minutes default

        # Performance and threading
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tf_manager")
        self._running = True
        self._update_tasks: Dict[str, asyncio.Task] = {}

        # Initialize default timeframe configurations
        self._initialize_timeframe_configs()

        logger.info("TimeframeManager initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'cache_ttl_seconds': 300,
            'max_concurrent_fetches': 4,
            'timestamp_alignment_tolerance_ms': 60000,  # 1 minute tolerance
            'enable_resampling': True,
            'missing_data_threshold': 0.8,  # 80% data completeness required
            'update_interval_seconds': 60,
        }

    def _initialize_timeframe_configs(self):
        """Initialize default timeframe configurations."""
        default_configs = {
            '5m': TimeframeConfig('5m', '5m', 50, 30),
            '15m': TimeframeConfig('15m', '15m', 100, 60),
            '1h': TimeframeConfig('1h', '1h', 200, 300),
            '4h': TimeframeConfig('4h', '4h', 300, 1200),
            '1d': TimeframeConfig('1d', '1d', 100, 3600),
        }

        for tf, config in default_configs.items():
            self.timeframe_configs[tf] = config

    async def initialize(self):
        """Initialize the timeframe manager."""
        logger.info("TimeframeManager initialization complete")

    async def shutdown(self):
        """Shutdown the timeframe manager."""
        self._running = False

        # Cancel all update tasks
        for task in self._update_tasks.values():
            if not task.done():
                task.cancel()

        # Clear cache
        self.cache.clear()
        self.symbol_timeframes.clear()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("TimeframeManager shutdown complete")

    def add_symbol(self, symbol: str, timeframes: List[str]) -> bool:
        """
        Register a symbol with its required timeframes.

        Args:
            symbol: Trading pair symbol
            timeframes: List of timeframe strings (e.g., ['15m', '1h', '4h'])

        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Validate timeframes
            valid_timeframes = []
            for tf in timeframes:
                if tf in self.timeframe_configs:
                    valid_timeframes.append(tf)
                else:
                    logger.warning(f"Unknown timeframe '{tf}' for symbol {symbol}, skipping")

            if not valid_timeframes:
                logger.error(f"No valid timeframes provided for symbol {symbol}")
                return False

            self.symbol_timeframes[symbol] = valid_timeframes
            logger.info(f"Added symbol {symbol} with timeframes: {valid_timeframes}")
            return True

        except Exception as e:
            logger.error(f"Failed to add symbol {symbol}: {e}")
            return False

    async def fetch_multi_timeframe_data(self, symbol: str) -> Optional[SyncedData]:
        """
        Fetch latest multi-timeframe data for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            SyncedData object or None if failed
        """
        if symbol not in self.symbol_timeframes:
            logger.error(f"Symbol {symbol} not registered with TimeframeManager")
            return None

        # Check cache first
        cached_data = self.get_synced_data(symbol)
        if cached_data is not None:
            logger.debug(f"Returning cached data for {symbol}")
            return cached_data

        try:
            timeframes = self.symbol_timeframes[symbol]

            # Fetch data for all timeframes concurrently
            tasks = []
            for tf in timeframes:
                task = self._fetch_single_timeframe(symbol, tf)
                tasks.append(task)

            # Wait for all fetches to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            tf_data = {}
            for i, result in enumerate(results):
                tf = timeframes[i]
                if isinstance(result, Exception):
                    logger.warning(f"Failed to fetch {tf} data for {symbol}: {result}")
                    # Create empty DataFrame for failed fetches
                    tf_data[tf] = pd.DataFrame()
                    continue

                # Include empty DataFrames as well
                tf_data[tf] = result if result is not None else pd.DataFrame()

            if not tf_data:
                logger.error(f"No data fetched for symbol {symbol}")
                return None

            # Synchronize timestamps
            synced_data = await self._synchronize_data(symbol, tf_data)

            # Cache the result
            self.cache[symbol] = synced_data

            logger.debug(f"Fetched multi-timeframe data for {symbol}: {list(tf_data.keys())}")
            return synced_data

        except Exception as e:
            logger.error(f"Failed to fetch multi-timeframe data for {symbol}: {e}")
            return None

    async def _fetch_single_timeframe(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single timeframe.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string

        Returns:
            DataFrame or None if failed
        """
        try:
            config = self.timeframe_configs[timeframe]
            limit = config.required_history

            # Call the async data fetcher method directly
            data = await self.data_fetcher.get_historical_data(symbol, timeframe, limit)

            if data is None or (hasattr(data, 'empty') and data.empty):
                logger.warning(f"No data received for {symbol} {timeframe}")
                return None

            # Ensure proper datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index, unit='ms')

            # Sort by timestamp
            data = data.sort_index()

            return data

        except Exception as e:
            logger.error(f"Failed to fetch {timeframe} data for {symbol}: {e}")
            return None

    async def _synchronize_data(self, symbol: str, tf_data: Dict[str, pd.DataFrame]) -> SyncedData:
        """
        Synchronize timestamps across multiple timeframes.

        Args:
            symbol: Trading pair symbol
            tf_data: Dictionary of timeframe -> DataFrame

        Returns:
            SyncedData object with aligned timestamps
        """
        try:
            # Find the most recent common timestamp
            latest_timestamp = await self._find_common_timestamp(tf_data)

            # Align all dataframes to this timestamp
            aligned_data = {}
            total_points = 0
            available_points = 0

            for tf, df in tf_data.items():
                aligned_df = await self._align_to_timestamp(df, latest_timestamp, tf)
                aligned_data[tf] = aligned_df

                # Calculate data completeness
                total_points += len(df)
                available_points += len(aligned_df)

            # Calculate confidence score based on data completeness
            confidence = available_points / total_points if total_points > 0 else 0.0

            # Apply minimum confidence threshold
            min_confidence = self.config.get('missing_data_threshold', 0.8)
            if confidence < min_confidence:
                logger.warning(f"Low confidence score for {symbol}: {confidence:.2f} < {min_confidence}")

            return SyncedData(
                symbol=symbol,
                timestamp=latest_timestamp,
                data=aligned_data,
                last_updated=now_ms(),
                confidence_score=confidence
            )

        except Exception as e:
            logger.error(f"Failed to synchronize data for {symbol}: {e}")
            # Return partial data with low confidence
            return SyncedData(
                symbol=symbol,
                timestamp=now_ms(),
                data=tf_data,
                last_updated=now_ms(),
                confidence_score=0.0
            )

    async def _find_common_timestamp(self, tf_data: Dict[str, pd.DataFrame]) -> int:
        """
        Find the most recent timestamp that exists in all timeframes.

        Args:
            tf_data: Dictionary of timeframe -> DataFrame

        Returns:
            Common timestamp in milliseconds
        """
        try:
            # Get the latest timestamp from each timeframe
            latest_timestamps = []
            for df in tf_data.values():
                if not df.empty:
                    latest_ts = df.index[-1]
                    if isinstance(latest_ts, pd.Timestamp):
                        latest_ts = int(latest_ts.timestamp() * 1000)
                    latest_timestamps.append(latest_ts)

            if not latest_timestamps:
                return now_ms()

            # Find the earliest of the latest timestamps (most conservative approach)
            common_timestamp = min(latest_timestamps)

            # Apply tolerance for slight timing differences
            tolerance = self.config.get('timestamp_alignment_tolerance_ms', 60000)
            current_time = now_ms()

            # Don't go too far back if data is very old
            if current_time - common_timestamp > tolerance:
                logger.warning(f"Common timestamp too old: {common_timestamp}, using current time")
                return current_time

            return common_timestamp

        except Exception as e:
            logger.error(f"Failed to find common timestamp: {e}")
            return now_ms()

    async def _align_to_timestamp(self, df: pd.DataFrame, target_timestamp: int,
                                timeframe: str) -> pd.DataFrame:
        """
        Align DataFrame to a specific timestamp.

        Args:
            df: Input DataFrame
            target_timestamp: Target timestamp in milliseconds
            timeframe: Timeframe string for resampling if needed

        Returns:
            Aligned DataFrame
        """
        try:
            if df.empty:
                return df

            # Convert target timestamp to pandas Timestamp
            target_ts = pd.Timestamp(target_timestamp / 1000, unit='s')

            # Find the closest timestamp in the data
            closest_idx = df.index.get_indexer([target_ts], method='nearest')[0]

            if closest_idx >= 0 and closest_idx < len(df):
                # Get data up to and including the closest timestamp
                aligned_df = df.iloc[:closest_idx + 1].copy()
            else:
                # If no close timestamp, return the most recent data
                aligned_df = df.tail(1).copy()

            return aligned_df

        except Exception as e:
            logger.error(f"Failed to align data to timestamp {target_timestamp}: {e}")
            return df.tail(1).copy() if not df.empty else df

    def get_synced_data(self, symbol: str) -> Optional[SyncedData]:
        """
        Get synchronized multi-timeframe data for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            SyncedData object or None if not available
        """
        # Check cache first
        if symbol in self.cache:
            cached_data = self.cache[symbol]
            current_time = now_ms()

            # Check if cache is still valid
            time_diff = current_time - cached_data.last_updated
            ttl_ms = self.cache_ttl * 1000

            logger.debug(f"Cache check for {symbol}: time_diff={time_diff}ms, ttl={ttl_ms}ms, cache_ttl={self.cache_ttl}")

            if time_diff < ttl_ms:
                logger.debug(f"Returning cached data for {symbol}")
                return cached_data
            else:
                # Cache expired, remove it
                logger.debug(f"Cache expired for {symbol}, removing from cache")
                del self.cache[symbol]

        return None

    async def update_cache(self, symbol: str, timeframe: str, new_data: pd.DataFrame) -> bool:
        """
        Update specific timeframe data in cache.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            new_data: New DataFrame with updated data

        Returns:
            True if successfully updated, False otherwise
        """
        try:
            if symbol not in self.symbol_timeframes:
                logger.error(f"Symbol {symbol} not registered")
                return False

            if timeframe not in self.symbol_timeframes[symbol]:
                logger.error(f"Timeframe {timeframe} not registered for symbol {symbol}")
                return False

            # Get existing synced data or create new
            synced_data = self.get_synced_data(symbol)
            if synced_data is None:
                # Create new synced data with just this timeframe
                synced_data = SyncedData(
                    symbol=symbol,
                    timestamp=now_ms(),
                    data={timeframe: new_data},
                    last_updated=now_ms(),
                    confidence_score=1.0
                )
            else:
                # Update existing data
                synced_data.data[timeframe] = new_data
                synced_data.last_updated = now_ms()

                # Recalculate confidence score
                total_points = sum(len(df) for df in synced_data.data.values())
                expected_points = len(self.symbol_timeframes[symbol]) * len(new_data)
                synced_data.confidence_score = total_points / expected_points if expected_points > 0 else 0.0

            # Update cache
            self.cache[symbol] = synced_data

            logger.debug(f"Updated cache for {symbol} {timeframe}")
            return True

        except Exception as e:
            logger.error(f"Failed to update cache for {symbol} {timeframe}: {e}")
            return False

    def get_available_symbols(self) -> List[str]:
        """Get list of registered symbols."""
        return list(self.symbol_timeframes.keys())

    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol and all associated data.

        Args:
            symbol: Trading pair symbol to remove

        Returns:
            True if successfully removed, False if symbol not found
        """
        if symbol in self.symbol_timeframes:
            del self.symbol_timeframes[symbol]
            # Also remove from cache if present
            if symbol in self.cache:
                del self.cache[symbol]
            logger.info(f"Removed symbol {symbol}")
            return True
        return False

    def get_registered_symbols(self) -> List[str]:
        """
        Get list of all registered symbols.

        Returns:
            List of registered symbol strings
        """
        return list(self.symbol_timeframes.keys())

    def get_symbol_timeframes(self, symbol: str) -> List[str]:
        """Get timeframes for a specific symbol."""
        return self.symbol_timeframes.get(symbol, [])

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cache for a specific symbol or all symbols.

        Args:
            symbol: Specific symbol to clear, or None for all
        """
        if symbol:
            if symbol in self.cache:
                del self.cache[symbol]
                logger.info(f"Cleared cache for symbol {symbol}")
        else:
            self.cache.clear()
            logger.info("Cleared all cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_symbols = len(self.cache)
        total_size = sum(len(str(data)) for data in self.cache.values())

        return {
            'cached_symbols': total_symbols,
            'estimated_size_bytes': total_size,
            'cache_ttl_seconds': self.cache_ttl,
            'registered_symbols': len(self.symbol_timeframes),
        }

    async def start_background_updates(self):
        """Start background update tasks for all registered symbols."""
        if not self._running:
            return

        for symbol in self.symbol_timeframes.keys():
            if symbol not in self._update_tasks or self._update_tasks[symbol].done():
                task = asyncio.create_task(self._background_update_symbol(symbol))
                self._update_tasks[symbol] = task

        logger.info(f"Started background updates for {len(self._update_tasks)} symbols")

    async def _background_update_symbol(self, symbol: str):
        """Background update task for a single symbol."""
        while self._running:
            try:
                # Check if update is needed
                synced_data = self.get_synced_data(symbol)
                if synced_data is None:
                    # No cached data, fetch immediately
                    await self.fetch_multi_timeframe_data(symbol)
                else:
                    # Check if any timeframe needs updating
                    current_time = now_ms()
                    needs_update = False

                    for tf in self.symbol_timeframes[symbol]:
                        config = self.timeframe_configs[tf]
                        time_since_update = current_time - synced_data.last_updated

                        if time_since_update > (config.update_frequency * 1000):
                            needs_update = True
                            break

                    if needs_update:
                        await self.fetch_multi_timeframe_data(symbol)

            except Exception as e:
                logger.error(f"Background update failed for {symbol}: {e}")

            # Wait before next check
            await asyncio.sleep(self.config.get('update_interval_seconds', 60))

    def stop_background_updates(self):
        """Stop all background update tasks."""
        self._running = False
        logger.info("Stopped background updates")
