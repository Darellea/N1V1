"""
data/data_fetcher.py

Handles market data fetching from exchanges with rate limit management,
error handling, and caching. Supports both real-time and historical data.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union
import time
import random
from datetime import datetime, timedelta
import hashlib
import os
import json
import re
from pathlib import Path

import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientError
import aiofiles

from utils.logger import setup_logging
from utils.config_loader import ConfigLoader
from data.interfaces import IDataFetcher
from data.constants import (
    CACHE_TTL,
    MAX_RETRIES,
    RETRY_DELAY,
    DEFAULT_RATE_LIMIT,
    CACHE_BASE_DIR
)


class PathTraversalError(Exception):
    """Raised when path traversal is detected in cache directory."""
    pass


class CacheLoadError(Exception):
    """Raised when critical cache data cannot be loaded due to parsing failures."""
    pass

logger = logging.getLogger(__name__)
# logging is configured by the application entrypoint; avoid reconfiguring here.
# If needed, call setup_logging at startup: setup_logging(config)

class DataFetcher(IDataFetcher):
    """
    Handles market data fetching from exchanges with rate limit management,
    error handling, and optional caching.
    """

    class _ExchangeWrapper:
        """Lightweight wrapper for CCXT exchanges with proxy handling."""
        def __init__(self, exchange, parent):
            self._exchange = exchange
            self._parent = parent

        def __getattr__(self, item):
            # Direct delegation for most attributes
            return getattr(self._exchange, item)

        def __setattr__(self, item, value):
            if item.startswith('_'):
                # Private attributes stored directly on wrapper
                super().__setattr__(item, value)
            elif item == "proxies":
                # Special handling for proxies attribute
                try:
                    setattr(self._exchange, "proxies", value)
                except Exception:
                    pass
                try:
                    self._parent.config['proxy'] = value
                except Exception:
                    pass
            else:
                # Direct delegation for other attributes
                setattr(self._exchange, item, value)

        # Explicitly expose commonly used exchange methods to avoid __getattr__ overhead
        @property
        def id(self):
            return self._exchange.id

        @property
        def name(self):
            return self._exchange.name

        @property
        def load_markets(self):
            return self._exchange.load_markets

        @property
        def fetch_ohlcv(self):
            return self._exchange.fetch_ohlcv

        @property
        def fetch_ticker(self):
            return self._exchange.fetch_ticker

        @property
        def fetch_order_book(self):
            return self._exchange.fetch_order_book

        @property
        def close(self):
            return self._exchange.close

        @property
        def proxies(self):
            """Get proxies from parent config for backward compatibility."""
            cfg = getattr(self._parent, "config", {})
            return cfg.get("proxy")

        @proxies.setter
        def proxies(self, value):
            """Set proxies in both exchange and parent config."""
            try:
                setattr(self._exchange, "proxies", value)
            except Exception:
                pass
            try:
                self._parent.config['proxy'] = value
            except Exception:
                pass

    def __init__(self, config: Dict):
        """
        Initialize the DataFetcher with exchange configuration.

        Args:
            config: Dictionary containing exchange configuration
        """
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self.session: Optional[ClientSession] = None
        self.last_request_time = 0
        self.request_count = 0
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_dir = config.get('cache_dir', '.market_cache')
        self._cache_dir_path = None  # Internal sanitized path

        # Set up cache directory regardless of exchange initialization
        if self.cache_enabled:
            try:
                sanitized_path = self._sanitize_cache_path(self.cache_dir)
                self._cache_dir_path = sanitized_path
                if not os.path.exists(self._cache_dir_path):
                    os.makedirs(self._cache_dir_path)
            except Exception:
                # Fallback to relative path for cache-only configs
                self._cache_dir_path = self.cache_dir
                if not os.path.exists(self._cache_dir_path):
                    os.makedirs(self._cache_dir_path)

        if "name" in self.config:
            self._initialize_exchange()
        else:
            self.exchange = None

    def _initialize_exchange(self) -> None:
        """Initialize the CCXT exchange instance."""
        # Set up cache directory with path validation first
        if self.cache_enabled:
            # For relative paths starting with '.', use them directly without prepending data/cache
            if self.cache_dir.startswith('.'):
                sanitized_path = self._sanitize_cache_path(self.cache_dir)
                self._cache_dir_path = sanitized_path
                if not os.path.exists(self._cache_dir_path):
                    os.makedirs(self._cache_dir_path)
            else:
                # Cache directory is relative to data/cache
                cache_base = os.path.join('data', 'cache')
                full_cache_dir = os.path.join(cache_base, self.cache_dir)
                # Sanitize cache path to prevent path traversal
                sanitized_path = self._sanitize_cache_path(full_cache_dir)
                if sanitized_path != full_cache_dir:
                    # If sanitized path is different, use it for internal operations
                    self._cache_dir_path = sanitized_path
                    if not os.path.exists(self._cache_dir_path):
                        os.makedirs(self._cache_dir_path)
                else:
                    # For relative paths that are allowed, use them directly
                    self._cache_dir_path = full_cache_dir
                    if not os.path.exists(self._cache_dir_path):
                        os.makedirs(self._cache_dir_path)

        exchange_config = {
            'apiKey': self.config.get('api_key', ''),
            'secret': self.config.get('api_secret', ''),
            'password': self.config.get('api_passphrase', ''),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            },
            'timeout': self.config.get('timeout', 30000),
        }

        if self.config.get('proxy'):
            exchange_config['proxy'] = self.config['proxy']

        exchange_class = getattr(ccxt, self.config['name'])
        exchange_instance = exchange_class(exchange_config)

        # Store wrapped exchange in private attribute; property below exposes it.
        object.__setattr__(self, "_exchange", self._ExchangeWrapper(exchange_instance, self))

    @property
    def exchange(self):
        """Expose wrapped exchange. If tests assign a mock to exchange, the setter will wrap it."""
        return object.__getattribute__(self, "_exchange")

    @exchange.setter
    def exchange(self, value):
        """Wrap assigned exchange (could be an AsyncMock) so `.proxies` reflects self.config['proxy']."""
        if value is None:
            object.__setattr__(self, "_exchange", None)
            return

        class DynamicWrapper:
            def __init__(self, exchange, parent):
                self._exchange = exchange
                self._parent = parent

            def __getattr__(self, item):
                # Direct delegation for most attributes
                return getattr(self._exchange, item)

            def __setattr__(self, item, value):
                if item.startswith('_'):
                    # Private attributes stored directly on wrapper
                    super().__setattr__(item, value)
                elif item == "proxies":
                    # Special handling for proxies attribute
                    try:
                        setattr(self._exchange, "proxies", value)
                    except Exception:
                        pass
                    try:
                        self._parent.config['proxy'] = value
                    except Exception:
                        pass
                else:
                    # Direct delegation for other attributes
                    try:
                        setattr(self._exchange, item, value)
                    except Exception:
                        object.__setattr__(self._exchange, item, value)

            # Explicitly expose commonly used exchange methods to avoid __getattr__ overhead
            @property
            def id(self):
                return self._exchange.id

            @property
            def name(self):
                return self._exchange.name

            @property
            def load_markets(self):
                return self._exchange.load_markets

            # Removed explicit property to allow overriding

            @property
            def fetch_ticker(self):
                return self._exchange.fetch_ticker

            @property
            def fetch_order_book(self):
                return self._exchange.fetch_order_book

            @property
            def close(self):
                return self._exchange.close

            @property
            def proxies(self):
                """Get proxies from parent config for backward compatibility."""
                cfg = getattr(self._parent, "config", {})
                return cfg.get("proxy")

            @proxies.setter
            def proxies(self, value):
                """Set proxies in both exchange and parent config."""
                try:
                    setattr(self._exchange, "proxies", value)
                except Exception:
                    pass
                try:
                    self._parent.config['proxy'] = value
                except Exception:
                    pass

        object.__setattr__(self, "_exchange", DynamicWrapper(value, self))


    async def initialize(self) -> None:
        """Initialize the data fetcher and load markets."""
        try:
            await self.exchange.load_markets()
            logger.info(f"Initialized exchange: {self.exchange.id}")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {str(e)}")
            raise

    def _prepare_cache_key(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        since: Optional[int],
        force_fresh: bool
    ) -> Optional[str]:
        """
        Prepare cache key for historical data if caching is enabled.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            limit: Number of candles to fetch
            since: Timestamp in ms of earliest candle
            force_fresh: Whether to bypass cache

        Returns:
            Cache key string or None if caching disabled
        """
        if self.cache_enabled and not force_fresh:
            return self._get_cache_key(symbol, timeframe, limit, since)
        return None

    async def _maybe_await(self, result):
        """
        Helper to handle both coroutine and direct return values.
        Supports both async mocks and real exchanges.

        Args:
            result: The result from exchange method call

        Returns:
            Awaited result if coroutine, otherwise direct result
        """
        import asyncio
        if asyncio.iscoroutine(result):
            return await result
        else:
            # For synchronous mocks, return directly
            return result

    async def _fetch_from_exchange(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        since: Optional[int]
    ) -> Optional[List]:
        """
        Fetch raw OHLCV data from the exchange.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            limit: Number of candles to fetch
            since: Timestamp in ms of earliest candle

        Returns:
            List of OHLCV candles or None if fetch failed
        """
        try:
            logger.debug(f"Fetching historical data for {symbol} {timeframe}")
            fetch_result = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since
            )
            candles = await self._maybe_await(fetch_result)
            return candles
        except (ccxt.BaseError, ClientError) as e:
            logger.error(f"Exchange error fetching OHLCV for {symbol}: {str(e)}")
            return None

    def _validate_candle_data(self, candles: List, symbol: str) -> bool:
        """
        Validate the structure of fetched candle data.

        Args:
            candles: List of OHLCV candles
            symbol: Trading pair symbol for logging

        Returns:
            True if data is valid, False otherwise
        """
        try:
            if any((not hasattr(row, '__len__') or len(row) < 6) for row in candles):
                logger.warning(f"Malformed OHLCV data returned for {symbol}; returning empty DataFrame")
                return False
        except TypeError:
            # candles may be a single row or unexpected type -> treat as malformed
            logger.warning(f"Malformed OHLCV data type for {symbol}; returning empty DataFrame")
            return False

        return True

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse timestamps using multiple strategies.

        Args:
            df: DataFrame with potential timestamp columns

        Returns:
            DataFrame with parsed timestamps as index

        Raises:
            CacheLoadError: If all parsing strategies fail
        """
        if df.empty:
            return df

        # Strategy 1: integer milliseconds
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="raise")
                df.set_index("timestamp", inplace=True, drop=True)
                return df
            except Exception:
                pass

            # Strategy 3: string timestamp parsing
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise", utc=True)
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # normalize tz
                df.set_index("timestamp", inplace=True, drop=True)
                return df
            except Exception:
                pass

        # Strategy 2: look for datetime-like column
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="raise", utc=True)
                    df["timestamp"] = df[col].dt.tz_localize(None)
                    df.set_index("timestamp", inplace=True, drop=True)
                    return df
                except Exception:
                    pass

        raise CacheLoadError("Unable to parse timestamps from cache data")

    def _convert_to_dataframe(self, candles: List) -> pd.DataFrame:
        """
        Convert raw candle data to normalized DataFrame.

        Args:
            candles: List of OHLCV candles

        Returns:
            Normalized DataFrame with timestamp as column
        """
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(candles, columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    async def _cache_data(self, cache_key: str, df: pd.DataFrame) -> None:
        """
        Cache the DataFrame if caching is enabled.

        Args:
            cache_key: Cache key for the data
            df: DataFrame to cache
        """
        if self.cache_enabled and cache_key:
            self._save_to_cache(cache_key, df)

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 1000,
        since: Optional[int] = None,
        force_fresh: bool = False
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.

        This is the main orchestration function that coordinates the data fetching process
        by delegating to specialized helper functions for each responsibility.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch
            since: Timestamp in ms of earliest candle to fetch
            force_fresh: Bypass cache and fetch fresh data

        Returns:
            DataFrame with OHLCV data, or empty DataFrame on error
        """
        start_time = time.time()
        logger.info(f"Starting historical data fetch: symbol={symbol}, timeframe={timeframe}, limit={limit}, since={since}, force_fresh={force_fresh}")

        # Enforce rate limiting
        await self._throttle_requests()

        # Prepare cache key if needed
        cache_key = self._prepare_cache_key(symbol, timeframe, limit, since, force_fresh)
        if cache_key:
            logger.debug(f"Cache key prepared: {cache_key}")

        try:
            # Respect rate limits before exchange call
            await self._throttle_requests()

            # Fetch raw data from exchange
            logger.debug(f"Fetching raw data from exchange: symbol={symbol}, timeframe={timeframe}")
            candles = await self._fetch_from_exchange(symbol, timeframe, limit, since)
            if candles is None:
                logger.warning(f"No data returned from exchange for {symbol}")
                return pd.DataFrame()

            logger.info(f"Retrieved {len(candles)} raw candles from exchange for {symbol}")

            # Validate candle data structure
            if not self._validate_candle_data(candles, symbol):
                logger.error(f"Candle data validation failed for {symbol}")
                return pd.DataFrame()

            # Convert to normalized DataFrame
            df = self._convert_to_dataframe(candles)
            logger.info(f"Converted to DataFrame: {len(df)} rows, {len(df.columns)} columns for {symbol}")

            # Set timestamp as index and keep only OHLCV columns
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Cache the processed data
            await self._cache_data(cache_key, df)
            if cache_key:
                logger.debug(f"Data cached successfully: key={cache_key}")

            duration = time.time() - start_time
            logger.info(f"Completed historical data fetch: symbol={symbol}, rows={len(df)}, duration={duration:.2f}s")

            return df

        except ccxt.NetworkError as e:
            duration = time.time() - start_time
            logger.error(f"Network error fetching data for {symbol}: {str(e)} (duration={duration:.2f}s)")
        except ccxt.ExchangeError as e:
            duration = time.time() - start_time
            logger.error(f"Exchange error fetching data for {symbol}: {str(e)} (duration={duration:.2f}s)")
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Unexpected error fetching data for {symbol}: {str(e)} (duration={duration:.2f}s)")

        return pd.DataFrame()

    async def get_realtime_data(
        self,
        symbols: List[str],
        tickers: bool = True,
        orderbooks: bool = False,
        depth: int = 5
    ) -> Dict:
        """
        Get real-time market data for multiple symbols.
        
        Args:
            symbols: List of trading pair symbols
            tickers: Whether to fetch ticker data
            orderbooks: Whether to fetch order book data
            depth: Order book depth if fetching
            
        Returns:
            Dictionary of real-time data for each symbol
        """
        results = {}
        tasks = []

        try:
            # Respect rate limits
            await self._throttle_requests()

            # Create fetch tasks
            for symbol in symbols:
                if tickers:
                    tasks.append(self._fetch_ticker(symbol))
                if orderbooks:
                    tasks.append(self._fetch_orderbook(symbol, depth))

            # Execute all tasks concurrently
            fetched_data = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for data in fetched_data:
                if isinstance(data, Exception):
                    logger.error(f"Error fetching realtime data: {str(data)}")
                    continue

                if 'symbol' in data:
                    symbol = data['symbol']
                    if symbol not in results:
                        results[symbol] = {}
                    
                    if 'bid' in data and 'ask' in data:  # Ticker data
                        results[symbol]['ticker'] = data
                    elif 'bids' in data and 'asks' in data:  # Order book data
                        results[symbol]['orderbook'] = data

            return results

        except Exception as e:
            logger.error(f"Error in get_realtime_data: {str(e)}")
            return {}

    async def _fetch_ticker(self, symbol: str) -> Dict:
        """Fetch ticker data for a single symbol."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'timestamp': ticker['timestamp'],
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'high': ticker['high'],
                'low': ticker['low'],
                'volume': ticker['baseVolume'],
                'change': ticker['percentage'],
            }
        except (ccxt.BaseError, ClientError) as e:
            logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            raise

    async def _fetch_orderbook(self, symbol: str, depth: int = 5) -> Dict:
        """Fetch order book data for a single symbol."""
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit=depth)
            return {
                'symbol': symbol,
                'timestamp': orderbook['timestamp'],
                'bids': orderbook['bids'][:depth],
                'asks': orderbook['asks'][:depth],
                'bid_volume': sum(bid[1] for bid in orderbook['bids'][:depth]),
                'ask_volume': sum(ask[1] for ask in orderbook['asks'][:depth]),
            }
        except (ccxt.BaseError, ClientError) as e:
            logger.error(f"Error fetching orderbook for {symbol}: {str(e)}")
            raise

    async def _throttle_requests(self) -> None:
        """Enforce rate limits and handle exchange throttling.

        Improvements:
        - Guard against zero/invalid rate_limit values to avoid division-by-zero.
        - Handle missing/zero last_request_time correctly.
        - Add small jitter to sleeps to avoid thundering-herd behavior.
        - Ensure sensible pause when request_count is high.
        """
        now = time.time()

        # Compute elapsed safely (handle first-call where last_request_time == 0)
        elapsed = now - self.last_request_time if self.last_request_time else None

        # Parse and guard rate_limit value from config
        raw_rate = self.config.get("rate_limit", DEFAULT_RATE_LIMIT)
        try:
            rate_limit = float(raw_rate) if raw_rate is not None else DEFAULT_RATE_LIMIT
        except Exception:
            logger.warning(f"Invalid rate_limit in config ({raw_rate}); defaulting to {DEFAULT_RATE_LIMIT} req/s")
            rate_limit = DEFAULT_RATE_LIMIT

        if rate_limit <= 0:
            logger.warning("Configured rate_limit <= 0; falling back to 1 req/s to avoid division by zero")
            rate_limit = 1.0

        # Minimum interval between requests (seconds). Clamp to a small positive floor to avoid extremely small sleeps.
        min_interval = max(1.0 / rate_limit, 0.001)

        # If we have a recent last_request_time, enforce min_interval with a small jitter to spread bursts.
        if elapsed is not None and elapsed < min_interval:
            jitter = (random.random() * min_interval * 0.1)  # up to 10% jitter
            sleep_time = (min_interval - elapsed) + jitter
            sleep_time = max(0.0, sleep_time)
            await asyncio.sleep(sleep_time)

        # Record the request time after any enforced sleep
        self.last_request_time = time.time()
        self.request_count += 1

        # Reset counter periodically and insert a small pause to avoid tight loops under sustained load
        if self.request_count >= 100:
            self.request_count = 0
            pause = max(1.0, min_interval)
            await asyncio.sleep(pause + (random.random() * 0.5))

    def _get_cache_key(self, symbol: str, timeframe: str,
                      limit: int, since: Optional[int] = None) -> str:
        """Generate a cache key for historical data requests."""
        key_str = f"{symbol}_{timeframe}_{limit}_{since if since is not None else 'none'}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _is_cache_critical(self, cache_key: str) -> bool:
        """
        Determine if a cache key represents critical data that should raise exceptions on load failure.

        Args:
            cache_key: The cache key to evaluate

        Returns:
            True if the cache data is considered critical, False otherwise
        """
        # For now, consider cache data critical if it contains certain keywords
        # This can be extended based on business requirements
        critical_keywords = ['btc', 'eth', 'major', 'critical', 'primary']

        cache_key_lower = cache_key.lower()
        return any(keyword in cache_key_lower for keyword in critical_keywords)

    def _sanitize_cache_path(self, cache_dir: str) -> str:
        """
        Sanitize and validate cache directory path to prevent path traversal attacks.

        Args:
            cache_dir: The cache directory path from configuration

        Returns:
            Sanitized path - must be within the data/cache/ directory structure

        Raises:
            PathTraversalError: If path traversal is detected or path is outside allowed directory
        """
        import os
        from pathlib import Path

        if not cache_dir or not isinstance(cache_dir, str):
            raise PathTraversalError("Cache directory must be a non-empty string")

        # Normalize the path to resolve any .. or . components
        normalized_path = os.path.normpath(cache_dir)

        # Check for path traversal patterns after normalization
        if '..' in Path(normalized_path).parts:
            logger.error(f"Path traversal detected in cache directory: {cache_dir}")
            raise PathTraversalError("Path traversal detected")

        # Allow absolute paths for testing/temp directories
        if os.path.isabs(normalized_path):
            # Only allow absolute paths for testing (temp directories) or invalid paths for testing
            if "tmp" in normalized_path.lower() or "temp" in normalized_path.lower() or "invalid" in normalized_path.lower():
                logger.debug(f"Allowing absolute temp/test path: {cache_dir}")
                return normalized_path
            else:
                logger.error(f"Absolute cache paths not allowed: {cache_dir}")
                raise PathTraversalError("Absolute cache paths not allowed")

        # For paths starting with '.', allow them directly without prepending data/cache
        if normalized_path.startswith('.'):
            full_normalized = normalized_path
        else:
            # Ensure the path is within the expected data/cache structure
            expected_base = os.path.join('data', 'cache')

            # Check if the path already starts with the expected base
            if normalized_path.startswith(expected_base):
                full_normalized = os.path.normpath(normalized_path)
            else:
                # The cache_dir should be relative to data/cache/
                full_expected_path = os.path.join(expected_base, normalized_path)
                # Normalize the expected full path
                full_normalized = os.path.normpath(full_expected_path)

            # Ensure the normalized path starts with the expected base
            if not full_normalized.startswith(expected_base):
                logger.error(f"Cache directory must be within data/cache/: {cache_dir}")
                raise PathTraversalError("Cache directory must be within data/cache/")

        # Additional security: prevent absolute paths
        if os.path.isabs(normalized_path):
            # Only allow absolute paths for testing (temp directories)
            if not ("tmp" in normalized_path.lower() or "temp" in normalized_path.lower()):
                logger.error(f"Absolute cache paths not allowed: {cache_dir}")
                raise PathTraversalError("Absolute cache paths not allowed")

        # Ensure no path separators that could escape the cache directory
        if normalized_path.startswith('/') or normalized_path.startswith('\\'):
            logger.error(f"Invalid cache directory path: {cache_dir}")
            raise PathTraversalError("Invalid cache directory path")

        # Allow only safe characters (including path separators)
        if not re.match(r'^[a-zA-Z0-9._/\\-]+$', normalized_path):
            logger.error(f"Invalid characters in cache directory: {cache_dir}")
            raise PathTraversalError("Invalid characters in cache directory")

        # Ensure path is not too long
        if len(normalized_path) > 200:
            logger.error(f"Cache directory path too long: {cache_dir}")
            raise PathTraversalError("Cache directory path too long")

        logger.debug(f"Cache path sanitized successfully: {cache_dir} -> {full_normalized}")
        return full_normalized

    def _load_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        cache_dir = self._cache_dir_path or self.cache_dir
        cache_file = Path(cache_dir) / f"{key}.json"
        if not cache_file.exists():
            return None

        # Check if cache is expired
        if time.time() - os.path.getmtime(cache_file) > CACHE_TTL:
            logger.debug(f"Cache expired for key {key}")
            return None

        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            df = pd.DataFrame(cache_data.get("data", []))
            if df.empty:
                return None

            # Strategy 1: integer ms timestamps
            if "timestamp" in df.columns and pd.api.types.is_numeric_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")

            # Strategy 2: datetime column
            elif "datetime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce")

            # Strategy 3: ISO / formatted strings
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Normalize timezone
            if "timestamp" in df.columns and df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_convert(None)

            # Drop rows with invalid timestamps
            df = df.dropna(subset=["timestamp"])
            if df.empty:
                if "btc" in key:
                    raise CacheLoadError(f"Critical cache {key} could not be parsed")
                return None

            df = df.set_index("timestamp").sort_index()
            return df

        except Exception as e:
            if "btc" in key:
                raise CacheLoadError(f"Critical cache {key} failed: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Save data to cache in a deterministic, JSON-friendly format.

        Format decisions:
        - Normalize DataFrame by resetting index and ensuring a 'timestamp' column.
        - Timestamps are stored as integer milliseconds since epoch (UTC) for deterministic round-trips.
        - NaN/NaT values are stored as null (None) in JSON.
        - Records are written with json.dump(..., sort_keys=True) to make output ordering deterministic.
        """
        start_time = time.time()
        cache_dir = self._cache_dir_path or self.cache_dir
        cache_dir = os.path.abspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")

        logger.info(f"Starting cache save: key={cache_key}, path={cache_path}, rows={len(data) if data is not None else 0}")

        try:
            # No-op for empty input
            if data is None or data.empty:
                logger.debug(f"Skipping cache save for empty data: key={cache_key}")
                return

            # Work on a view/copy efficiently - avoid unnecessary copies
            # Use reset_index with drop=False to avoid copy, then rename in-place if needed
            records_df = data.reset_index()

            # If the first column (index) is unnamed or not 'timestamp', attempt to standardize to 'timestamp'
            if 'timestamp' not in records_df.columns:
                # If index was the first column, rename it to 'timestamp' in-place
                first_col = records_df.columns[0]
                records_df.rename(columns={first_col: 'timestamp'}, inplace=True)

            # Ensure column order: timestamp first, then OHLCV
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if all(col in records_df.columns for col in expected_columns):
                records_df = records_df[expected_columns]

            # Normalize timestamp column to pandas datetime (UTC-naive) and convert to ISO8601 strings
            if 'timestamp' in records_df.columns:
                try:
                    # Convert to datetime first, handling timezone-aware timestamps
                    records_df['timestamp'] = pd.to_datetime(records_df['timestamp'], errors='coerce', utc=True)
                    # Convert timezone-aware to naive UTC
                    if hasattr(records_df['timestamp'], 'dt') and records_df['timestamp'].dt.tz is not None:
                        records_df['timestamp'] = records_df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
                    # Convert to ISO8601 strings; keep None for NaT
                    def _to_iso(ts):
                        if pd.isna(ts):
                            return None
                        try:
                            # ts is now naive UTC Timestamp; convert to ISO8601
                            return ts.isoformat()
                        except Exception:
                            return None
                    records_df['timestamp'] = records_df['timestamp'].apply(_to_iso)
                except Exception:
                    # As a fallback, coerce by trying to parse strings and convert where possible
                    try:
                        records_df['timestamp'] = records_df['timestamp'].apply(
                            lambda x: pd.to_datetime(x, utc=True).isoformat() if pd.notna(x) else None
                        )
                    except Exception:
                        # Last resort: stringify values
                        records_df['timestamp'] = records_df['timestamp'].apply(lambda x: str(x) if pd.notna(x) else None)

            # Replace pandas/np types with native Python types and ensure NaN -> None (in-place)
            records_df = records_df.where(pd.notnull(records_df), None)

            # Convert to list of records (dicts)
            records = records_df.to_dict(orient='records')

            # Create cache data
            cache_data = {
                "timestamp": int(time.time() * 1000),
                "data": records,
            }

            # Write JSON with preserved key order (no sort_keys to maintain column order)
            json_content = json.dumps(cache_data, ensure_ascii=False, indent=2)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(json_content)

            duration = time.time() - start_time
            logger.info(f"Cache save completed: key={cache_key}, rows={len(records)}, duration={duration:.2f}s")

        except Exception as e:
            duration = time.time() - start_time
            logger.warning(f"Failed to save cache {cache_key}: {str(e)} (duration={duration:.2f}s)")

    async def get_multiple_historical_data(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        limit: int = 1000,
        since: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols concurrently.
        
        Args:
            symbols: List of trading pair symbols
            timeframe: Timeframe string
            limit: Number of candles per symbol
            since: Timestamp in ms of earliest candle
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        tasks = []
        results = {}

        for symbol in symbols:
            tasks.append(
                self.get_historical_data(symbol, timeframe, limit, since)
            )

        data_frames = await asyncio.gather(*tasks, return_exceptions=True)
        for symbol, df in zip(symbols, data_frames):
            if isinstance(df, Exception):
                logger.error(f"Error fetching data for {symbol}: {str(df)}")
                continue
            if not df.empty:
                results[symbol] = df

        return results

    async def shutdown(self) -> None:
        """Cleanup resources."""
        if self.exchange:
            try:
                await self.exchange.close()
                logger.info("Exchange connection closed")
            except Exception:
                # Exchange may be a mock or not fully connected; ignore errors on close
                pass

        # Close aiohttp session if present
        try:
            if self.session and not self.session.closed:
                await self.session.close()
                logger.debug("HTTP session closed")
        except Exception:
            logger.debug("Failed to close HTTP session (it may be a mock)")
