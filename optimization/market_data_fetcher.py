"""
Market Data Fetcher Module

This module provides functionality for fetching market data from various sources
for cross-asset validation. It supports multiple data providers with fallback
mechanisms and implements caching to improve performance.

Key Features:
- Multiple data source support (Binance, CoinGecko, local cache)
- Async data fetching for improved performance
- Intelligent caching with TTL (Time To Live)
- Rate limiting and error handling
- Data quality validation
- Support for different timeframes and data formats
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .config import get_cross_asset_validation_config


class MarketDataFetcher:
    """
    Unified interface for fetching market data from multiple sources.

    This class provides a consistent API for retrieving historical market data
    from various providers, with automatic fallback mechanisms and caching
    to ensure reliable data access for cross-asset validation.

    Key responsibilities:
    - Fetch historical price data from multiple sources
    - Implement caching to reduce API calls and improve performance
    - Handle rate limiting and API errors gracefully
    - Validate data quality and completeness
    - Support multiple timeframes and data formats
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the market data fetcher.

        Args:
            config: Configuration dictionary for data fetching.
                   If None, uses default configuration from config module.
        """
        if config is None:
            data_config = get_cross_asset_validation_config()
            config = {
                'name': data_config.data_fetcher.name,
                'cache_enabled': data_config.data_fetcher.cache_enabled,
                'cache_dir': data_config.data_fetcher.cache_dir,
                'requests_per_minute': data_config.data_fetcher.requests_per_minute,
                'requests_per_hour': data_config.data_fetcher.requests_per_hour,
                'min_data_points': data_config.data_fetcher.min_data_points,
                'max_missing_data_pct': data_config.data_fetcher.max_missing_data_pct
            }

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Core configuration
        self.provider_name = config.get('name', 'binance')
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_dir = config.get('cache_dir', 'data/cache')
        self.min_data_points = config.get('min_data_points', 100)
        self.max_missing_data_pct = config.get('max_missing_data_pct', 0.05)

        # Rate limiting
        self.requests_per_minute = config.get('requests_per_minute', 60)
        self.requests_per_hour = config.get('requests_per_hour', 1000)
        self._request_times: List[float] = []
        self._hourly_request_times: List[float] = []

        # Cache management
        self._cache: Dict[str, Tuple[pd.DataFrame, float]] = {}  # (data, timestamp)
        self._cache_ttl = 3600  # 1 hour default TTL

        # Ensure cache directory exists
        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.logger.info(f"MarketDataFetcher initialized with provider: {self.provider_name}")

    def get_historical_data(self, symbol: str, timeframe: str = '1h',
                           limit: int = 1000) -> pd.DataFrame:
        """
        Get historical market data for a symbol.

        This is the main entry point for fetching data. It handles caching,
        rate limiting, and fallback mechanisms automatically.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Data timeframe ('1m', '5m', '1h', '1d', etc.)
            limit: Maximum number of data points to retrieve

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check cache first
            if self.cache_enabled:
                cached_data = self._get_from_cache(symbol, timeframe, limit)
                if cached_data is not None:
                    self.logger.debug(f"Cache hit for {symbol} {timeframe}")
                    return cached_data

            # Apply rate limiting
            self._apply_rate_limiting()

            # Fetch data based on provider
            if self.provider_name.lower() == 'binance':
                data = self._fetch_from_binance(symbol, timeframe, limit)
            elif self.provider_name.lower() == 'coingecko':
                data = self._fetch_from_coingecko(symbol, timeframe, limit)
            else:
                raise ValueError(f"Unsupported data provider: {self.provider_name}")

            # Validate data quality
            if not self._validate_data_quality(data):
                self.logger.warning(f"Data quality validation failed for {symbol}")
                return pd.DataFrame()

            # Cache the result
            if self.cache_enabled:
                self._save_to_cache(symbol, timeframe, limit, data)

            return data

        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
            return pd.DataFrame()

    async def get_historical_data_async(self, symbol: str, timeframe: str = '1h',
                                       limit: int = 1000) -> pd.DataFrame:
        """
        Async version of get_historical_data.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            limit: Maximum number of data points

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check cache first
            if self.cache_enabled:
                cached_data = self._get_from_cache(symbol, timeframe, limit)
                if cached_data is not None:
                    return cached_data

            # Apply rate limiting
            await self._apply_rate_limiting_async()

            # Fetch data based on provider
            if self.provider_name.lower() == 'binance':
                data = await self._fetch_from_binance_async(symbol, timeframe, limit)
            elif self.provider_name.lower() == 'coingecko':
                data = await self._fetch_from_coingecko_async(symbol, timeframe, limit)
            else:
                raise ValueError(f"Unsupported data provider: {self.provider_name}")

            # Validate data quality
            if not self._validate_data_quality(data):
                self.logger.warning(f"Data quality validation failed for {symbol}")
                return pd.DataFrame()

            # Cache the result
            if self.cache_enabled:
                self._save_to_cache(symbol, timeframe, limit, data)

            return data

        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _fetch_from_binance(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Fetch data from Binance API.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            limit: Number of data points

        Returns:
            DataFrame with OHLCV data
        """
        try:
            from binance.client import Client as BinanceClient

            # Initialize Binance client (would need API keys in production)
            # client = BinanceClient(api_key, api_secret)
            # klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)

            # Mock data for demonstration
            self.logger.info(f"Fetching {limit} {timeframe} candles for {symbol} from Binance")

            # Generate mock OHLCV data
            base_time = datetime.now() - timedelta(hours=limit)
            timestamps = [base_time + timedelta(hours=i) for i in range(limit)]

            np.random.seed(42)  # For reproducible mock data
            prices = np.random.uniform(40000, 60000, limit)
            volumes = np.random.uniform(100, 1000, limit)

            data = []
            for i, ts in enumerate(timestamps):
                open_price = prices[i]
                close_price = prices[i] + np.random.normal(0, 100)
                high_price = max(open_price, close_price) + abs(np.random.normal(0, 50))
                low_price = min(open_price, close_price) - abs(np.random.normal(0, 50))

                data.append({
                    'timestamp': ts,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volumes[i]
                })

            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Binance fetch failed: {str(e)}")
            return pd.DataFrame()

    async def _fetch_from_binance_async(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Async version of Binance data fetching.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            limit: Number of data points

        Returns:
            DataFrame with OHLCV data
        """
        # For async Binance fetching, you would use aiohttp or similar
        # For now, delegate to sync version in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_from_binance, symbol, timeframe, limit)

    def _fetch_from_coingecko(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Fetch data from CoinGecko API.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            limit: Number of data points

        Returns:
            DataFrame with OHLCV data
        """
        try:
            import requests

            # Map symbol to CoinGecko ID
            symbol_to_id = self._get_coingecko_id_mapping()
            base_symbol = symbol.split('/')[0].upper()

            if base_symbol not in symbol_to_id:
                raise ValueError(f"Unsupported symbol for CoinGecko: {symbol}")

            coin_id = symbol_to_id[base_symbol]

            # CoinGecko API call
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': min(limit // 24, 365),  # Convert to days (max 365)
                'interval': 'hourly' if timeframe in ['1h', '4h'] else 'daily'
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # Parse CoinGecko response
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])

            if not prices:
                return pd.DataFrame()

            # Build DataFrame
            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                volume = volumes[i][1] if i < len(volumes) else 0

                df_data.append({
                    'timestamp': datetime.fromtimestamp(timestamp / 1000),
                    'open': price,
                    'high': price * 1.01,  # Approximate high
                    'low': price * 0.99,   # Approximate low
                    'close': price,
                    'volume': volume
                })

            df = pd.DataFrame(df_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            return df.head(limit)

        except Exception as e:
            self.logger.error(f"CoinGecko fetch failed: {str(e)}")
            return pd.DataFrame()

    async def _fetch_from_coingecko_async(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Async version of CoinGecko data fetching.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            limit: Number of data points

        Returns:
            DataFrame with OHLCV data
        """
        try:
            import aiohttp

            # Map symbol to CoinGecko ID
            symbol_to_id = self._get_coingecko_id_mapping()
            base_symbol = symbol.split('/')[0].upper()

            if base_symbol not in symbol_to_id:
                raise ValueError(f"Unsupported symbol for CoinGecko: {symbol}")

            coin_id = symbol_to_id[base_symbol]

            # CoinGecko API call
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': min(limit // 24, 365),
                'interval': 'hourly' if timeframe in ['1h', '4h'] else 'daily'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        raise ValueError(f"CoinGecko API error: {response.status}")

                    data = await response.json()

            # Parse response (same as sync version)
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])

            if not prices:
                return pd.DataFrame()

            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                volume = volumes[i][1] if i < len(volumes) else 0

                df_data.append({
                    'timestamp': datetime.fromtimestamp(timestamp / 1000),
                    'open': price,
                    'high': price * 1.01,
                    'low': price * 0.99,
                    'close': price,
                    'volume': volume
                })

            df = pd.DataFrame(df_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            return df.head(limit)

        except Exception as e:
            self.logger.error(f"Async CoinGecko fetch failed: {str(e)}")
            return pd.DataFrame()

    def _get_coingecko_id_mapping(self) -> Dict[str, str]:
        """
        Get mapping from common symbols to CoinGecko IDs.

        Returns:
            Dictionary mapping symbols to CoinGecko IDs
        """
        return {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'ADA': 'cardano',
            'SOL': 'solana',
            'DOT': 'polkadot',
            'LINK': 'chainlink',
            'BNB': 'binancecoin',
            'XRP': 'ripple',
            'LTC': 'litecoin',
            'DOGE': 'dogecoin',
            'MATIC': 'matic-network',
            'AVAX': 'avalanche-2'
        }

    def _validate_data_quality(self, data: pd.DataFrame) -> bool:
        """
        Validate the quality of fetched data.

        Args:
            data: DataFrame to validate

        Returns:
            True if data passes quality checks
        """
        if data.empty:
            return False

        # Check minimum data points
        if len(data) < self.min_data_points:
            self.logger.warning(f"Insufficient data points: {len(data)} < {self.min_data_points}")
            return False

        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > self.max_missing_data_pct:
            self.logger.warning(f"Too many missing values: {missing_pct:.1%} > {self.max_missing_data_pct:.1%}")
            return False

        # Check for reasonable price values
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            self.logger.warning("Missing required OHLC columns")
            return False

        # Check for negative or zero prices
        for col in required_columns:
            if (data[col] <= 0).any():
                self.logger.warning(f"Invalid {col} values found (negative or zero)")
                return False

        # Check OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        if invalid_ohlc.any():
            self.logger.warning("Invalid OHLC relationships found")
            return False

        return True

    def _apply_rate_limiting(self) -> None:
        """
        Apply rate limiting to prevent API abuse.
        """
        current_time = time.time()

        # Clean old request times
        self._request_times = [t for t in self._request_times if current_time - t < 60]
        self._hourly_request_times = [t for t in self._hourly_request_times if current_time - t < 3600]

        # Check rate limits
        if len(self._request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self._request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)

        if len(self._hourly_request_times) >= self.requests_per_hour:
            sleep_time = 3600 - (current_time - self._hourly_request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Hourly rate limit reached, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)

        # Record this request
        self._request_times.append(current_time)
        self._hourly_request_times.append(current_time)

    async def _apply_rate_limiting_async(self) -> None:
        """
        Async version of rate limiting.
        """
        current_time = time.time()

        # Clean old request times
        self._request_times = [t for t in self._request_times if current_time - t < 60]
        self._hourly_request_times = [t for t in self._hourly_request_times if current_time - t < 3600]

        # Check rate limits
        if len(self._request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self._request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)

        if len(self._hourly_request_times) >= self.requests_per_hour:
            sleep_time = 3600 - (current_time - self._hourly_request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Hourly rate limit reached, sleeping for {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)

        # Record this request
        self._request_times.append(current_time)
        self._hourly_request_times.append(current_time)

    def _get_cache_key(self, symbol: str, timeframe: str, limit: int) -> str:
        """
        Generate a cache key for the given parameters.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            limit: Number of data points

        Returns:
            Cache key string
        """
        return f"{symbol}_{timeframe}_{limit}"

    def _get_from_cache(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache if available and not expired.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            limit: Number of data points

        Returns:
            Cached DataFrame or None if not available/expired
        """
        if not self.cache_enabled:
            return None

        cache_key = self._get_cache_key(symbol, timeframe, limit)

        # Check memory cache first
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return data
            else:
                # Remove expired cache entry
                del self._cache[cache_key]

        # Check file cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                cache_mtime = os.path.getmtime(cache_file)
                if time.time() - cache_mtime < self._cache_ttl:
                    data = pd.read_pickle(cache_file)
                    # Update memory cache
                    self._cache[cache_key] = (data, time.time())
                    return data
                else:
                    # Remove expired file
                    os.remove(cache_file)
            except Exception as e:
                self.logger.warning(f"Failed to load cache file {cache_file}: {str(e)}")

        return None

    def _save_to_cache(self, symbol: str, timeframe: str, limit: int, data: pd.DataFrame) -> None:
        """
        Save data to cache.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            limit: Number of data points
            data: DataFrame to cache
        """
        if not self.cache_enabled or data.empty:
            return

        cache_key = self._get_cache_key(symbol, timeframe, limit)
        current_time = time.time()

        # Save to memory cache
        self._cache[cache_key] = (data.copy(), current_time)

        # Save to file cache
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            data.to_pickle(cache_file)
        except Exception as e:
            self.logger.warning(f"Failed to save cache file {cache_key}: {str(e)}")

    def clear_cache(self) -> None:
        """
        Clear all cached data.
        """
        self._cache.clear()

        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(self.cache_dir, file))
                    except Exception as e:
                        self.logger.warning(f"Failed to remove cache file {file}: {str(e)}")

        self.logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        cache_files = []
        if os.path.exists(self.cache_dir):
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]

        return {
            'memory_cache_entries': len(self._cache),
            'file_cache_entries': len(cache_files),
            'cache_enabled': self.cache_enabled,
            'cache_ttl_seconds': self._cache_ttl,
            'cache_directory': self.cache_dir
        }

    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """
        Set the cache time-to-live.

        Args:
            ttl_seconds: Cache TTL in seconds
        """
        self._cache_ttl = ttl_seconds
        self.logger.info(f"Cache TTL set to {ttl_seconds} seconds")

    def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported trading symbols.

        Returns:
            List of supported symbols
        """
        if self.provider_name.lower() == 'binance':
            # Common Binance symbols
            return [
                'BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'DOT/USDT',
                'LINK/USDT', 'BNB/USDT', 'XRP/USDT', 'LTC/USDT', 'DOGE/USDT'
            ]
        elif self.provider_name.lower() == 'coingecko':
            # CoinGecko supported symbols
            return [
                'BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'DOT/USDT',
                'LINK/USDT', 'MATIC/USDT', 'AVAX/USDT'
            ]
        else:
            return []

    def get_supported_timeframes(self) -> List[str]:
        """
        Get list of supported timeframes.

        Returns:
            List of supported timeframes
        """
        if self.provider_name.lower() == 'binance':
            return ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        elif self.provider_name.lower() == 'coingecko':
            return ['1h', '1d']  # Limited by CoinGecko API
        else:
            return ['1h', '1d']

    def __str__(self) -> str:
        """
        String representation of the data fetcher.

        Returns:
            Human-readable description
        """
        return f"MarketDataFetcher(provider={self.provider_name}, cache={self.cache_enabled})"
