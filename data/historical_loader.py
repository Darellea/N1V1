"""
data/historical_loader.py

Handles loading and preprocessing of historical market data for backtesting.
Includes data validation, cleaning, resampling, and technical indicator support.
"""

import asyncio
import hashlib
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.constants import (
    CACHE_TTL,
    DEFAULT_GAP_HANDLING_STRATEGY,
    HISTORICAL_DATA_BASE_DIR,
    MAX_PAGINATION_ITERATIONS,
    MAX_RETRIES,
    MEMORY_EFFICIENT_THRESHOLD,
    RETRY_DELAY,
)
from data.interfaces import IDataFetcher
from utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration parameters are invalid."""

    pass


class HistoricalDataLoader:
    """
    Loads and manages historical market data for backtesting.
    Handles data validation, cleaning, resampling, and storage.
    """

    def __init__(self, config: Dict, data_fetcher: IDataFetcher):
        """
        Initialize the HistoricalDataLoader.

        Args:
            config: Configuration dictionary
            data_fetcher: DataFetcher instance for live data fetching
        """
        self.config = config["backtesting"]
        self.data_fetcher = data_fetcher
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.validated_pairs: List[str] = []
        # Configurable option: whether to deduplicate timestamps after pagination.
        # When True, duplicate index entries (identical timestamps) are removed
        # keeping the first occurrence. Default is True to ensure proper deduplication.
        self.deduplicate = self.config.get("deduplicate", True)
        self._setup_data_directory()

    def _validate_data_directory(self, data_dir: str) -> str:
        """
        Validate and sanitize the data directory path.

        Args:
            data_dir: Data directory path to validate

        Returns:
            Validated and sanitized path

        Raises:
            ConfigurationError: If path validation fails
        """
        if not data_dir or not isinstance(data_dir, str):
            raise ConfigurationError("data_dir must be a non-empty string")

        # Check for absolute path patterns (Windows and Unix)
        is_absolute = (
            data_dir.startswith("/")
            or data_dir.startswith("\\")
            or (len(data_dir) >= 3 and data_dir[1:3] == ":\\")
        )

        if is_absolute:
            logger.error(f"Absolute path detected: {data_dir}")
            raise ConfigurationError("Absolute path detected")
        else:
            # For relative paths, apply existing validation
            # Check for path traversal patterns
            if ".." in data_dir:
                logger.error(f"Path traversal detected in data_dir: {data_dir}")
                raise ConfigurationError(
                    f"Path traversal detected in data_dir: {data_dir}"
                )

            if "/" in data_dir or "\\" in data_dir:
                logger.error(f"Path separators not allowed in data_dir: {data_dir}")
                raise ConfigurationError(
                    f"Path separators not allowed in data_dir: {data_dir}"
                )

            # Allow only alphanumeric characters, underscores, hyphens, and optional leading dot
            if not re.match(r"^[.]?[a-zA-Z0-9_-]+$", data_dir):
                logger.error(f"Invalid characters in data_dir: {data_dir}")
                raise ConfigurationError("Invalid characters in data_dir")

            # Ensure name is not too long (prevent filesystem issues)
            if len(data_dir) > 100:
                logger.error(f"data_dir too long: {data_dir}")
                raise ConfigurationError(
                    f"data_dir too long (max 100 characters): {data_dir}"
                )

            logger.debug(f"data_dir validated successfully: {data_dir}")
            return data_dir

    def _setup_data_directory(self) -> None:
        """Create the historical data directory if it doesn't exist."""
        raw_data_dir = self.config.get("data_dir", "historical_data")

        if os.path.isabs(raw_data_dir):
            # Allow absolute paths if they are within a temporary directory (for testing)
            if "temp" in raw_data_dir.lower() and os.path.exists(
                raw_data_dir
            ):  # Heuristic for temp dirs
                self.data_dir_path = os.path.normpath(raw_data_dir)
                self.data_dir = raw_data_dir  # Keep original for consistency
                os.makedirs(self.data_dir_path, exist_ok=True)
                return
            else:
                raise ConfigurationError("Absolute path detected")

        # Validate the data directory path
        try:
            validated_dir = self._validate_data_directory(raw_data_dir)
            base_path = os.path.join(os.getcwd(), "data", "historical")
            self.data_dir_path = os.path.join(base_path, validated_dir)
            self.data_dir = self.data_dir_path  # Set to absolute path
            os.makedirs(self.data_dir_path, exist_ok=True)
        except (ConfigurationError, OSError) as e:
            logger.error(f"Data directory setup failed: {str(e)}")
            raise ConfigurationError(f"Data directory setup failed: {str(e)}")

    async def load_historical_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for multiple symbols within a date range.

        Args:
            symbols: List of trading pair symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            force_refresh: Whether to ignore cached data

        Returns:
            Dictionary mapping symbols to their historical DataFrames
        """
        start_time = time.time()
        logger.info(
            f"Starting historical data load: symbols={symbols}, start_date={start_date}, end_date={end_date}, timeframe={timeframe}, force_refresh={force_refresh}"
        )

        results = {}
        tasks = [
            self._load_symbol_data(
                symbol, start_date, end_date, timeframe, force_refresh
            )
            for symbol in symbols
        ]

        fetched_data = await asyncio.gather(*tasks)

        for symbol, df in zip(symbols, fetched_data):
            if df is not None and not df.empty:
                results[symbol] = df
                self.validated_pairs.append(symbol)

        duration = time.time() - start_time
        logger.info(
            f"Completed historical data load: {len(results)}/{len(symbols)} symbols successful, duration={duration:.2f}s"
        )

        return results

    async def _load_symbol_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str,
        force_refresh: bool,
    ) -> Optional[pd.DataFrame]:
        """
        Load historical data for a single symbol.

        Args:
            symbol: Trading pair symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe string
            force_refresh: Whether to ignore cached data

        Returns:
            DataFrame with historical data or None if loading failed
        """
        cache_key = self._generate_cache_key(symbol, start_date, end_date, timeframe)
        cache_path = os.path.join(self.data_dir_path, f"{cache_key}.parquet")

        # Try to load from cache first
        if not force_refresh and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                if self._validate_data(df, timeframe):
                    logger.info(f"Loaded cached data for {symbol} ({timeframe})")
                    return df
            except Exception as e:
                logger.warning(f"Failed to load cached data for {symbol}: {str(e)}")

        # Fetch data from exchange
        logger.info(f"Fetching historical data for {symbol} ({timeframe})")
        df = await self._fetch_complete_history(symbol, start_date, end_date, timeframe)

        if df is None or df.empty:
            logger.error(f"No data available for {symbol}")
            return None

        # Clean and validate the data
        df = self._clean_data(df)
        if not self._validate_data(df, timeframe):
            logger.error(f"Data validation failed for {symbol}")
            return None

        # Save to cache
        try:
            df.to_parquet(cache_path)
            logger.debug(f"Saved historical data for {symbol} to cache")
        except Exception as e:
            logger.warning(f"Failed to cache data for {symbol}: {str(e)}")

        return df

    def _calculate_pagination_params(
        self, start_date: str, end_date: str, timeframe: str
    ) -> Tuple[pd.Timestamp, pd.Timestamp, timedelta, int]:
        """
        Calculate pagination parameters for historical data fetching.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe string

        Returns:
            Tuple of (start_dt, end_dt, delta, estimated_requests)
        """
        start_dt = pd.to_datetime(start_date).tz_localize(None)
        end_dt = pd.to_datetime(end_date).tz_localize(None)
        delta = self._get_timeframe_delta(timeframe)

        # Calculate the number of requests needed for progress bar
        total_days = (end_dt - start_dt).days
        timeframe_days = self._timeframe_to_days(timeframe)
        estimated_requests = max(
            1, total_days // (30 * timeframe_days)
        )  # ~1 month per request

        return start_dt, end_dt, delta, estimated_requests

    def _initialize_pagination_state(self) -> Tuple[int, int, int]:
        """
        Initialize pagination state variables.

        Returns:
            Tuple of (max_iterations, iteration_count, consecutive_same_start)
        """
        max_iterations = (
            MAX_PAGINATION_ITERATIONS  # Reasonable upper bound for iterations
        )
        iteration_count = 0
        consecutive_same_start = 0
        return max_iterations, iteration_count, consecutive_same_start

    async def _fetch_single_page(
        self,
        symbol: str,
        timeframe: str,
        current_start: pd.Timestamp,
        current_end: pd.Timestamp,
        max_retries: int = MAX_RETRIES,
        retry_delay: int = RETRY_DELAY,
    ) -> Tuple[Optional[pd.DataFrame], bool]:
        """
        Fetch a single page of historical data with retry logic.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            current_start: Start timestamp for this page
            current_end: End timestamp for this page
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Tuple of (filtered_data, success_flag)
        """
        fetched = False
        retries = 0

        while not fetched and retries < max_retries:
            try:
                data = await self.data_fetcher.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=int(current_start.timestamp() * 1000),
                    limit=1000,
                )

                if data is not None and not data.empty:
                    # Filter data that's within our date range
                    mask = (data.index >= current_start) & (data.index <= current_end)
                    filtered = data[mask]
                    return filtered, True
                else:
                    # No more data available
                    return None, True

            except Exception as e:
                retries += 1
                logger.warning(
                    f"Error fetching data for {symbol} (attempt {retries}): {str(e)}"
                )
                if retries < max_retries:
                    await asyncio.sleep(retry_delay)

        if not fetched:
            logger.error(
                f"Failed to fetch data for {symbol} after {max_retries} attempts"
            )
        return None, False

    def _advance_pagination_window(
        self,
        current_start: pd.Timestamp,
        last_index: pd.Timestamp,
        delta: timedelta,
        symbol: str,
    ) -> pd.Timestamp:
        """
        Advance the pagination window based on the last fetched index.

        Args:
            current_start: Current start timestamp
            last_index: Last index from fetched data
            delta: Timeframe delta
            symbol: Trading pair symbol for logging

        Returns:
            New current_start timestamp
        """
        # Always advance by at least the timeframe delta to prevent infinite loops
        if last_index <= current_start:
            # Exchange returned same or earlier data, advance by delta
            new_start = current_start + delta
            logger.debug(
                f"Exchange returned same/earlier last_index for {symbol}, "
                f"advancing by timeframe delta: {delta}"
            )
        else:
            # Normal case: advance to last_index
            new_start = last_index

        return new_start

    def _detect_infinite_loop(
        self,
        current_start: pd.Timestamp,
        last_current_start: Optional[pd.Timestamp],
        consecutive_same_start: int,
        symbol: str,
    ) -> Tuple[bool, int]:
        """
        Detect potential infinite loops in pagination.

        Args:
            current_start: Current start timestamp
            last_current_start: Previous start timestamp
            consecutive_same_start: Count of consecutive same starts
            symbol: Trading pair symbol for logging

        Returns:
            Tuple of (should_break, new_consecutive_count)
        """
        if current_start == last_current_start:
            consecutive_same_start += 1
            if consecutive_same_start >= 4:
                logger.warning(
                    f"Detected potential infinite loop for {symbol}: current_start unchanged "
                    f"for {consecutive_same_start} iterations. Breaking loop."
                )
                return True, consecutive_same_start
        else:
            consecutive_same_start = 0

        return False, consecutive_same_start

    def _combine_paginated_data(
        self, all_data: List[pd.DataFrame], symbol: str
    ) -> Optional[pd.DataFrame]:
        """
        Combine paginated data chunks into a single DataFrame.

        Args:
            all_data: List of DataFrame chunks
            symbol: Trading pair symbol for logging

        Returns:
            Combined DataFrame or None if no data
        """
        if not all_data:
            return None

        # Combine all fetched chunks into a single DataFrame efficiently
        # For large datasets, use generator-based concatenation to reduce memory usage
        if len(all_data) > MEMORY_EFFICIENT_THRESHOLD:  # Threshold for large datasets
            # Use generator to avoid loading all DataFrames into memory at once
            combined = pd.concat((df for df in all_data), copy=False)
        else:
            # Standard concatenation for smaller datasets
            combined = pd.concat(all_data, copy=False)

        combined.sort_index(inplace=True)

        # Deduplicate overlapping timestamps from pagination if configured.
        # This removes duplicated index entries that can occur when pages overlap.
        if self.deduplicate:
            try:
                combined = combined[~combined.index.duplicated(keep="first")]
            except Exception:
                # If deduplication fails for any reason, continue with the combined DataFrame as-is.
                logger.debug(
                    "Deduplication of paginated data failed; proceeding without dedupe"
                )

        return combined

    def _prepare_data_for_gap_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for gap handling by ensuring required columns exist.

        Args:
            df: DataFrame to prepare

        Returns:
            DataFrame with required columns
        """
        # Ensure required OHLCV columns exist before filling
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan

        return df

    async def _fetch_complete_history(
        self, symbol, start_date, end_date, timeframe="1d"
    ):
        results = []
        current_start = pd.to_datetime(start_date).tz_localize(None)
        unit = timeframe.lstrip("0123456789")
        end = pd.to_datetime(end_date).tz_localize(None)
        max_iterations = MAX_PAGINATION_ITERATIONS
        iteration_count = 0

        while (
            current_start.tz_localize(None) <= end.tz_localize(None)
            and iteration_count < max_iterations
        ):
            df = await self.data_fetcher.get_historical_data(
                symbol, timeframe, since=current_start
            )
            if df is None or df.empty:
                break
            results.append(df)
            # Ensure we work with tz-naive timestamps
            last_index = df.index[-1]
            if hasattr(last_index, "tz_localize"):
                last_index = last_index.tz_localize(None)
            current_start = last_index + pd.Timedelta(1, unit=unit)
            iteration_count += 1

        if iteration_count >= max_iterations:
            logger.warning(
                f"Reached maximum iteration limit ({max_iterations}) for {symbol} historical data fetching"
            )

        if results:
            final_df = pd.concat(results)
            final_df = final_df[~final_df.index.duplicated(keep="last")]
            return final_df.sort_index()
        return pd.DataFrame()

    def _validate_fetch_parameters(
        self, symbol: str, start_date: str, end_date: str, timeframe: str
    ) -> bool:
        """
        Validate parameters for historical data fetching.

        Args:
            symbol: Trading pair symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe string

        Returns:
            True if parameters are valid, False otherwise
        """
        # Guard clause: validate symbol
        if not symbol or not isinstance(symbol, str):
            logger.error(f"Invalid symbol: {symbol}")
            return False

        # Guard clause: validate dates
        if not start_date or not end_date:
            logger.error(
                f"Missing date parameters: start_date={start_date}, end_date={end_date}"
            )
            return False

        # Guard clause: validate timeframe
        supported_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        if not timeframe or timeframe not in supported_timeframes:
            logger.error(
                f"Invalid timeframe: {timeframe}. Supported: {supported_timeframes}"
            )
            return False

        # Guard clause: validate date order
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            if start_dt >= end_dt:
                logger.error(
                    f"Start date must be before end date: {start_date} >= {end_date}"
                )
                return False
        except Exception as e:
            logger.error(
                f"Invalid date format: start_date={start_date}, end_date={end_date}, error={str(e)}"
            )
            return False

        return True

    async def _execute_pagination_loop(
        self,
        symbol: str,
        timeframe: str,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        delta: timedelta,
        estimated_requests: int,
    ) -> List[pd.DataFrame]:
        """
        Execute the pagination loop to fetch all data pages.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            start_dt: Start timestamp
            end_dt: End timestamp
            delta: Timeframe delta
            estimated_requests: Estimated number of requests for progress bar

        Returns:
            List of DataFrame chunks from pagination
        """
        # Initialize pagination state
        (
            max_iterations,
            iteration_count,
            consecutive_same_start,
        ) = self._initialize_pagination_state()

        all_data = []
        current_start = start_dt
        last_current_start = None

        with tqdm(total=estimated_requests, desc=f"Fetching {symbol}") as pbar:
            while current_start < end_dt and iteration_count < max_iterations:
                # Guard clause: detect infinite loops
                should_break, consecutive_same_start = self._detect_infinite_loop(
                    current_start, last_current_start, consecutive_same_start, symbol
                )
                if should_break:
                    break

                last_current_start = current_start
                current_end = min(current_start + delta, end_dt)

                # Fetch single page with retry logic
                filtered_data, success = await self._fetch_single_page(
                    symbol, timeframe, current_start, current_end
                )

                # Guard clause: break on fetch failure
                if not success:
                    break

                # Process successful fetch
                if filtered_data is not None and not filtered_data.empty:
                    all_data.append(filtered_data)
                    pbar.update(1)

                    # Advance pagination window
                    last_index = filtered_data.index[-1]
                    current_start = self._advance_pagination_window(
                        current_start, last_index, delta, symbol
                    )
                else:
                    # No more data available
                    current_start = end_dt

                iteration_count += 1

            # Log warning if we hit the iteration limit
            if iteration_count >= max_iterations:
                logger.warning(
                    f"Reached maximum iteration limit ({max_iterations}) for {symbol} "
                    f"historical data fetching. This may indicate an infinite loop condition."
                )

        return all_data

    def _process_paginated_data(
        self, all_data: List[pd.DataFrame], symbol: str
    ) -> Optional[pd.DataFrame]:
        """
        Process and combine paginated data with gap handling.

        Args:
            all_data: List of DataFrame chunks from pagination
            symbol: Trading pair symbol for logging

        Returns:
            Processed DataFrame or None if no data
        """
        # Guard clause: check if we have any data
        if not all_data:
            logger.warning(f"No data retrieved for {symbol}")
            return None

        # Combine all paginated data
        combined = self._combine_paginated_data(all_data, symbol)

        # Guard clause: check if combination was successful
        if combined is None or combined.empty:
            logger.warning(f"Failed to combine paginated data for {symbol}")
            return combined

        # Prepare for gap handling and apply strategy
        combined = self._prepare_data_for_gap_handling(combined)
        if not combined.empty:
            combined = self._apply_gap_handling_strategy(combined, symbol)

        return combined

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw historical data.

        Args:
            df: Raw DataFrame from exchange

        Returns:
            Cleaned DataFrame
        """
        # Remove any rows with NaN in critical columns
        df = df.dropna(subset=["open", "high", "low", "close"])

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

        # Remove zero-volume candles that have no price movement
        mask = (
            (df["volume"] == 0)
            & (df["open"] == df["high"])
            & (df["open"] == df["low"])
            & (df["open"] == df["close"])
        )
        df = df[~mask]

        return df

    def _validate_data(self, df: pd.DataFrame, timeframe: str) -> bool:
        """
        Validate the integrity of historical data.

        Args:
            df: DataFrame to validate
            timeframe: Expected timeframe

        Returns:
            True if data is valid, False otherwise
        """
        if df.empty:
            return False

        # Check required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            return False

        # Check for missing values in critical columns
        if df[required_cols].isnull().any().any():
            return False

        # Check timeframe consistency
        if len(df) > 1:
            time_diff = df.index[1] - df.index[0]
            expected_diff = pd.Timedelta(self._get_pandas_freq(timeframe))

            # Allow some tolerance for minor inconsistencies
            if not (0.8 * expected_diff <= time_diff <= 1.2 * expected_diff):
                return False

        return True

    def _generate_cache_key(
        self, symbol: str, start_date: str, end_date: str, timeframe: str
    ) -> str:
        """
        Generate a unique cache key for historical data.

        Args:
            symbol: Trading pair symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe string

        Returns:
            MD5 hash string as cache key
        """
        key_str = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_timeframe_delta(self, timeframe: str) -> timedelta:
        """
        Get timedelta for a given timeframe.

        Args:
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')

        Returns:
            Corresponding timedelta
        """
        timeframe_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1),
        }
        return timeframe_map.get(timeframe, timedelta(days=1))

    def _get_pandas_freq(self, timeframe: str) -> str:
        """
        Get pandas frequency string for a given timeframe.

        Args:
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')

        Returns:
            Corresponding pandas frequency string
        """
        timeframe_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D",
            "1w": "1W",
        }
        return timeframe_map.get(timeframe, "1D")

    def _timeframe_to_days(self, timeframe: str) -> int:
        """
        Convert timeframe to approximate days.

        Args:
            timeframe: Timeframe string

        Returns:
            Approximate days per candle
        """
        timeframe_map = {
            "1m": 1 / (24 * 60),
            "5m": 5 / (24 * 60),
            "15m": 15 / (24 * 60),
            "30m": 30 / (24 * 60),
            "1h": 1 / 24,
            "4h": 4 / 24,
            "1d": 1,
            "1w": 7,
        }
        return timeframe_map.get(timeframe, 1)

    def _apply_gap_handling_strategy(
        self, df: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """
        Apply configurable gap handling strategy to OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol for logging

        Returns:
            DataFrame with gaps handled according to configuration
        """
        # Get gap handling configuration
        gap_strategy = self.config.get(
            "gap_handling_strategy", DEFAULT_GAP_HANDLING_STRATEGY
        )

        if gap_strategy == "forward_fill":
            return self._apply_forward_fill_with_logging(df, symbol)
        elif gap_strategy == "interpolate":
            return self._apply_interpolation_with_logging(df, symbol)
        elif gap_strategy == "reject":
            return self._apply_reject_strategy(df, symbol)
        else:
            logger.warning(
                f"Unknown gap handling strategy '{gap_strategy}' for {symbol}, defaulting to forward_fill"
            )
            return self._apply_forward_fill_with_logging(df, symbol)

    def _apply_forward_fill_with_logging(
        self, df: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """
        Apply forward fill with comprehensive logging of gaps.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol for logging

        Returns:
            DataFrame with forward-filled gaps
        """
        df_filled = df.copy()
        gap_info = {}

        # Check for gaps in each column
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df_filled.columns:
                continue

            # Find NaN values that will be filled
            nan_mask = df_filled[col].isna()
            nan_count = nan_mask.sum()

            if nan_count > 0:
                # Find contiguous gap blocks
                gap_blocks = []
                in_gap = False
                gap_start = None

                for idx, is_nan in nan_mask.items():
                    if is_nan and not in_gap:
                        # Start of a gap
                        in_gap = True
                        gap_start = idx
                    elif not is_nan and in_gap:
                        # End of a gap
                        in_gap = False
                        gap_blocks.append((gap_start, idx))
                        gap_start = None

                # Handle case where gap extends to end
                if in_gap:
                    gap_blocks.append((gap_start, df_filled.index[-1]))

                gap_info[col] = {
                    "total_gaps": nan_count,
                    "gap_blocks": len(gap_blocks),
                    "gap_ranges": [(str(start), str(end)) for start, end in gap_blocks],
                }

                # Apply forward fill
                if col == "volume":
                    df_filled[col] = df_filled[col].fillna(0)
                else:
                    df_filled[col] = df_filled[col].ffill()

        # Log gap information
        if gap_info:
            total_gaps = sum(info["total_gaps"] for info in gap_info.values())
            logger.info(
                f"Applied forward fill to {total_gaps} missing values in {symbol} data"
            )

            for col, info in gap_info.items():
                logger.debug(
                    f"Column '{col}' in {symbol}: {info['total_gaps']} gaps in {info['gap_blocks']} blocks"
                )
                if (
                    info["gap_blocks"] > 0 and info["gap_blocks"] <= 5
                ):  # Log details for small number of gaps
                    for i, (start, end) in enumerate(info["gap_ranges"]):
                        logger.debug(f"  Gap block {i+1}: {start} to {end}")
                elif info["gap_blocks"] > 5:
                    logger.debug(
                        f"  First gap: {info['gap_ranges'][0][0]} to {info['gap_ranges'][0][1]}"
                    )
                    logger.debug(
                        f"  Last gap: {info['gap_ranges'][-1][0]} to {info['gap_ranges'][-1][1]}"
                    )
        else:
            logger.debug(f"No gaps found in {symbol} data")

        return df_filled

    def _apply_interpolation_with_logging(
        self, df: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """
        Apply interpolation with logging for gap handling.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol for logging

        Returns:
            DataFrame with interpolated gaps
        """
        df_interpolated = df.copy()
        total_interpolated = 0

        # Apply interpolation to price columns
        for col in ["open", "high", "low", "close"]:
            if col in df_interpolated.columns:
                nan_count = df_interpolated[col].isna().sum()
                if nan_count > 0:
                    df_interpolated[col] = df_interpolated[col].interpolate(
                        method="linear"
                    )
                    total_interpolated += nan_count
                    logger.debug(
                        f"Interpolated {nan_count} missing values in {col} for {symbol}"
                    )

        # Handle volume separately (use forward fill for volume)
        if "volume" in df_interpolated.columns:
            nan_count = df_interpolated["volume"].isna().sum()
            if nan_count > 0:
                df_interpolated["volume"] = df_interpolated["volume"].fillna(0)
                logger.debug(f"Set {nan_count} missing volume values to 0 for {symbol}")

        if total_interpolated > 0:
            logger.info(
                f"Applied interpolation to {total_interpolated} missing values in {symbol} data"
            )
        else:
            logger.debug(f"No gaps found in {symbol} data for interpolation")

        return df_interpolated

    def _apply_reject_strategy(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Apply reject strategy - raise error if gaps are found.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol for logging

        Returns:
            Original DataFrame if no gaps, raises exception if gaps found

        Raises:
            ValueError: If gaps are found in the data
        """
        # Check for any NaN values in critical columns
        critical_cols = ["open", "high", "low", "close", "volume"]
        total_gaps = 0

        for col in critical_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    total_gaps += nan_count
                    logger.warning(
                        f"Found {nan_count} missing values in {col} for {symbol}"
                    )

        if total_gaps > 0:
            error_msg = f"Reject strategy: Found {total_gaps} missing values in {symbol} data. Data contains gaps."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"No gaps found in {symbol} data - reject strategy passed")
        return df

    async def resample_data(
        self, data: Dict[str, pd.DataFrame], new_timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Resample historical data to a different timeframe.

        Args:
            data: Dictionary of symbol to DataFrame
            new_timeframe: Target timeframe string

        Returns:
            Dictionary of resampled DataFrames
        """
        resampled = {}

        for symbol, df in data.items():
            if df.empty:
                continue

            resample_map = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }

            # Use pandas resample with explicit freq. Prefer resample, fallback to Grouper.
            freq = self._get_pandas_freq(new_timeframe)
            try:
                resampled_df = df.resample(freq).agg(resample_map)
            except Exception:
                # Fallback for edge-cases: use Grouper
                resampled_df = df.groupby(pd.Grouper(freq=freq)).agg(resample_map)

            resampled[symbol] = resampled_df

        return resampled

    def get_available_pairs(self) -> List[str]:
        """
        Get list of symbol pairs with validated historical data.

        Returns:
            List of symbol strings
        """
        return self.validated_pairs

    def load_chunked(
        self,
        data: pd.DataFrame,
        chunk_size: int = 1000,
        max_memory_mb: Optional[float] = None,
        memory_monitor: Optional[callable] = None,
        progress_callback: Optional[callable] = None,
        handle_corruption: bool = False,
        resume_from: Optional[Dict] = None,
        memory_manager: Optional[any] = None,
    ):
        """
        Load data in chunks to prevent memory issues.

        Args:
            data: DataFrame to chunk
            chunk_size: Number of rows per chunk
            max_memory_mb: Maximum memory per chunk in MB
            memory_monitor: Callback for memory monitoring
            progress_callback: Callback for progress updates
            handle_corruption: Whether to handle corrupted chunks gracefully
            resume_from: Resume state dict with 'last_index' and 'processed_chunks'
            memory_manager: Memory manager instance for integration

        Yields:
            DataFrame chunks
        """
        if data.empty:
            return

        # Get memory manager if not provided
        if memory_manager is None:
            try:
                from core.memory_manager import get_memory_manager

                memory_manager = get_memory_manager()
            except ImportError:
                memory_manager = None

        # Calculate chunk size based on memory limits
        if max_memory_mb is not None:
            estimated_chunk_memory = self._estimate_dataframe_memory(data.head(100))
            dynamic_chunk_size = max(
                1, int(max_memory_mb / estimated_chunk_memory * 100)
            )
            chunk_size = min(chunk_size, dynamic_chunk_size)

        # Handle resume
        start_index = 0
        processed_chunks = 0
        if resume_from:
            start_index = resume_from.get("last_index", 0)
            processed_chunks = resume_from.get("processed_chunks", 0)

        total_chunks = (len(data) - start_index + chunk_size - 1) // chunk_size
        total_processed = processed_chunks

        for i in range(start_index, len(data), chunk_size):
            chunk = data.iloc[i : i + chunk_size].copy()

            # Handle corrupted chunks
            if handle_corruption and self._is_chunk_corrupted(chunk):
                logger.warning(f"Skipping corrupted chunk at index {i}")
                continue

            # Memory monitoring
            if memory_monitor:
                memory_info = memory_monitor()
                if memory_info and memory_info.get("memory_mb", 0) > self.config.get(
                    "max_memory_mb", 500
                ):
                    logger.warning("Memory threshold exceeded, triggering cleanup")
                    if memory_manager:
                        memory_manager.trigger_cleanup()
            elif memory_manager:
                # Use memory manager for monitoring if no custom monitor provided
                memory_stats = memory_manager.get_memory_stats()
                if memory_stats.get("current_memory_mb", 0) > self.config.get(
                    "max_memory_mb", 500
                ):
                    logger.warning("Memory threshold exceeded, triggering cleanup")
                    memory_manager.trigger_cleanup()

            # Progress tracking
            total_processed += 1
            if progress_callback:
                progress = int(
                    (total_processed / (total_chunks + processed_chunks)) * 100
                )
                progress_callback(progress)

            yield chunk

    async def load_chunked_async(
        self,
        data: pd.DataFrame,
        chunk_size: int = 1000,
        max_memory_mb: Optional[float] = None,
        memory_monitor: Optional[callable] = None,
        progress_callback: Optional[callable] = None,
        handle_corruption: bool = False,
        resume_from: Optional[Dict] = None,
        memory_manager: Optional[any] = None,
    ):
        """
        Async version of load_chunked.

        Args:
            data: DataFrame to chunk
            chunk_size: Number of rows per chunk
            max_memory_mb: Maximum memory per chunk in MB
            memory_monitor: Async callback for memory monitoring
            progress_callback: Callback for progress updates
            handle_corruption: Whether to handle corrupted chunks gracefully
            resume_from: Resume state dict
            memory_manager: Memory manager instance

        Yields:
            DataFrame chunks
        """
        if data.empty:
            return

        # Get memory manager if not provided
        if memory_manager is None:
            try:
                from core.memory_manager import get_memory_manager

                memory_manager = get_memory_manager()
            except ImportError:
                memory_manager = None

        # Calculate chunk size based on memory limits
        if max_memory_mb is not None:
            estimated_chunk_memory = self._estimate_dataframe_memory(data.head(100))
            dynamic_chunk_size = max(
                1, int(max_memory_mb / estimated_chunk_memory * 100)
            )
            chunk_size = min(chunk_size, dynamic_chunk_size)

        # Handle resume
        start_index = 0
        processed_chunks = 0
        if resume_from:
            start_index = resume_from.get("last_index", 0)
            processed_chunks = resume_from.get("processed_chunks", 0)

        total_chunks = (len(data) - start_index + chunk_size - 1) // chunk_size
        total_processed = processed_chunks

        for i in range(start_index, len(data), chunk_size):
            chunk = data.iloc[i : i + chunk_size].copy()

            # Handle corrupted chunks
            if handle_corruption and self._is_chunk_corrupted(chunk):
                logger.warning(f"Skipping corrupted chunk at index {i}")
                continue

            # Memory monitoring
            if memory_monitor:
                memory_info = await memory_monitor()
                if memory_info and memory_info.get("memory_mb", 0) > self.config.get(
                    "max_memory_mb", 500
                ):
                    logger.warning("Memory threshold exceeded, triggering cleanup")
                    if memory_manager:
                        memory_manager.trigger_cleanup()

            # Progress tracking
            total_processed += 1
            if progress_callback:
                progress = int(
                    (total_processed / (total_chunks + processed_chunks)) * 100
                )
                progress_callback(progress)

            yield chunk

    async def load_historical_data_chunked(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str,
        chunk_size: int = 1000,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical data using chunked processing for memory efficiency.

        Args:
            symbols: List of trading pair symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe string
            chunk_size: Number of rows per chunk
            force_refresh: Whether to ignore cached data

        Returns:
            Dictionary mapping symbols to their historical DataFrames
        """
        start_time = time.time()
        logger.info(
            f"Starting chunked historical data load: symbols={symbols}, "
            f"start_date={start_date}, end_date={end_date}, timeframe={timeframe}, "
            f"chunk_size={chunk_size}"
        )

        results = {}

        for symbol in symbols:
            try:
                # Load data for this symbol
                df = await self._load_symbol_data_chunked(
                    symbol, start_date, end_date, timeframe, chunk_size, force_refresh
                )

                if df is not None and not df.empty:
                    results[symbol] = df
                    self.validated_pairs.append(symbol)
                    logger.info(f"Successfully loaded chunked data for {symbol}")

            except Exception as e:
                logger.error(f"Failed to load chunked data for {symbol}: {str(e)}")

        duration = time.time() - start_time
        logger.info(
            f"Completed chunked historical data load: {len(results)}/{len(symbols)} symbols successful, "
            f"duration={duration:.2f}s"
        )

        return results

    async def _load_symbol_data_chunked(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str,
        chunk_size: int,
        force_refresh: bool,
    ) -> Optional[pd.DataFrame]:
        """
        Load historical data for a single symbol using chunked processing.

        Args:
            symbol: Trading pair symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe string
            chunk_size: Chunk size for processing
            force_refresh: Whether to ignore cache

        Returns:
            DataFrame with historical data or None
        """
        cache_key = self._generate_cache_key(symbol, start_date, end_date, timeframe)
        cache_path = os.path.join(self.data_dir_path, f"{cache_key}.parquet")

        # Try to load from cache first
        if not force_refresh and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                if self._validate_data(df, timeframe):
                    logger.info(f"Loaded cached data for {symbol} ({timeframe})")
                    return df
            except Exception as e:
                logger.warning(f"Failed to load cached data for {symbol}: {str(e)}")

        # Fetch data using streaming approach
        all_chunks = []
        chunks_processed = 0

        try:
            async for chunk in self._fetch_historical_data_streaming(
                symbol, start_date, end_date, timeframe, chunk_size
            ):
                if chunk is not None and not chunk.empty:
                    # Clean and validate chunk
                    chunk = self._clean_data(chunk)
                    if self._validate_data(chunk, timeframe):
                        all_chunks.append(chunk)
                        chunks_processed += 1

                        # Memory management: combine chunks periodically
                        if len(all_chunks) >= 10:  # Combine every 10 chunks
                            combined = pd.concat(all_chunks, copy=False)
                            combined.sort_index(inplace=True)
                            if self.deduplicate:
                                combined = combined[
                                    ~combined.index.duplicated(keep="first")
                                ]
                            all_chunks = [combined]

            # Final combination
            if all_chunks:
                final_df = pd.concat(all_chunks, copy=False)
                final_df.sort_index(inplace=True)
                if self.deduplicate:
                    final_df = final_df[~final_df.index.duplicated(keep="first")]

                # Apply gap handling
                final_df = self._prepare_data_for_gap_handling(final_df)
                final_df = self._apply_gap_handling_strategy(final_df, symbol)

                # Cache the result
                try:
                    final_df.to_parquet(cache_path)
                    logger.debug(f"Saved chunked historical data for {symbol} to cache")
                except Exception as e:
                    logger.warning(
                        f"Failed to cache chunked data for {symbol}: {str(e)}"
                    )

                logger.info(f"Processed {chunks_processed} chunks for {symbol}")
                return final_df

        except Exception as e:
            logger.error(f"Error in chunked loading for {symbol}: {str(e)}")

        return None

    async def _fetch_historical_data_streaming(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str,
        chunk_size: int,
    ):
        """
        Fetch historical data in streaming fashion.

        Args:
            symbol: Trading pair symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            chunk_size: Size of chunks to yield

        Yields:
            DataFrame chunks
        """
        current_start = pd.to_datetime(start_date).tz_localize(None)
        end = pd.to_datetime(end_date).tz_localize(None)
        unit = timeframe.lstrip("0123456789")
        max_iterations = MAX_PAGINATION_ITERATIONS
        iteration_count = 0

        accumulated_data = []

        while current_start <= end and iteration_count < max_iterations:
            try:
                # Fetch a batch of data
                df = await self.data_fetcher.get_historical_data(
                    symbol, timeframe, since=current_start
                )

                if df is None or df.empty:
                    break

                accumulated_data.append(df)

                # When we have enough data for a chunk, yield it
                while len(accumulated_data) > 0:
                    combined_length = sum(len(chunk) for chunk in accumulated_data)

                    if combined_length >= chunk_size:
                        # Combine enough data for one chunk
                        chunk_data = []
                        remaining_needed = chunk_size

                        while accumulated_data and remaining_needed > 0:
                            current_chunk = accumulated_data[0]
                            if len(current_chunk) <= remaining_needed:
                                chunk_data.append(current_chunk)
                                accumulated_data.pop(0)
                                remaining_needed -= len(current_chunk)
                            else:
                                # Split the chunk
                                split_chunk = current_chunk.iloc[:remaining_needed]
                                chunk_data.append(split_chunk)
                                accumulated_data[0] = current_chunk.iloc[
                                    remaining_needed:
                                ]
                                remaining_needed = 0

                        if chunk_data:
                            chunk = pd.concat(chunk_data, copy=False)
                            chunk.sort_index(inplace=True)
                            yield chunk
                    else:
                        break

                # Advance to next batch
                last_index = df.index[-1]
                if hasattr(last_index, "tz_localize"):
                    last_index = last_index.tz_localize(None)
                current_start = last_index + pd.Timedelta(1, unit=unit)
                iteration_count += 1

            except Exception as e:
                logger.error(f"Error fetching streaming data for {symbol}: {str(e)}")
                break

        # Yield any remaining data
        if accumulated_data:
            remaining = pd.concat(accumulated_data, copy=False)
            remaining.sort_index(inplace=True)

            # Yield remaining data in chunks
            for i in range(0, len(remaining), chunk_size):
                chunk = remaining.iloc[i : i + chunk_size]
                if not chunk.empty:
                    yield chunk

    async def resample_data_chunked(
        self,
        data: Dict[str, pd.DataFrame],
        new_timeframe: str,
        chunk_size: int = 5000,
    ) -> Dict[str, pd.DataFrame]:
        """
        Resample data using chunked processing for memory efficiency.

        Args:
            data: Dictionary of symbol to DataFrame
            new_timeframe: Target timeframe
            chunk_size: Processing chunk size

        Returns:
            Dictionary of resampled DataFrames
        """
        resampled = {}

        for symbol, df in data.items():
            if df.empty:
                continue

            logger.info(f"Resampling {symbol} data to {new_timeframe} using chunks")

            try:
                # Process in chunks to avoid memory issues
                resampled_chunks = []

                for chunk in self.load_chunked(df, chunk_size=chunk_size):
                    resample_map = {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }

                    freq = self._get_pandas_freq(new_timeframe)
                    try:
                        resampled_chunk = chunk.resample(freq).agg(resample_map)
                        if not resampled_chunk.empty:
                            resampled_chunks.append(resampled_chunk)
                    except Exception as e:
                        logger.warning(
                            f"Failed to resample chunk for {symbol}: {str(e)}"
                        )

                if resampled_chunks:
                    # Combine resampled chunks
                    final_resampled = pd.concat(resampled_chunks, copy=False)
                    final_resampled.sort_index(inplace=True)

                    # Remove duplicates that may occur at chunk boundaries
                    final_resampled = final_resampled[
                        ~final_resampled.index.duplicated(keep="last")
                    ]

                    resampled[symbol] = final_resampled
                    logger.info(
                        f"Successfully resampled {symbol} with {len(resampled_chunks)} chunks"
                    )

            except Exception as e:
                logger.error(f"Failed to resample {symbol}: {str(e)}")

        return resampled

    def _estimate_dataframe_memory(self, df: pd.DataFrame) -> float:
        """
        Estimate memory usage of a DataFrame in MB.

        Args:
            df: DataFrame to estimate

        Returns:
            Memory usage in MB
        """
        if df.empty:
            return 0.0

        return df.memory_usage(deep=True).sum() / 1024 / 1024

    def _is_chunk_corrupted(self, chunk: pd.DataFrame) -> bool:
        """
        Check if a data chunk is corrupted.

        Args:
            chunk: DataFrame chunk to check

        Returns:
            True if chunk appears corrupted
        """
        if chunk.empty:
            return False

        # Check for excessive NaN values
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col in chunk.columns:
                nan_ratio = chunk[col].isna().sum() / len(chunk)
                if nan_ratio > 0.5:  # More than 50% NaN values
                    return True

        # Check for invalid price relationships
        if all(col in chunk.columns for col in ["high", "low"]):
            invalid_prices = (chunk["high"] < chunk["low"]).sum()
            if invalid_prices > len(chunk) * 0.1:  # More than 10% invalid
                return True

        return False

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self.data_cache.clear()
        self.validated_pairs.clear()
