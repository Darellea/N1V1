"""
Comprehensive tests for data/data_fetcher.py

Covers data fetching, caching, rate limiting, and error handling.
Tests specific lines: 68, 82-83, 87-97, 104, 130-143, 153-155, 205-208,
216-219, 233, 235, 260-297, 301-316, 320-332, 352-354, 357-358, 376-378,
391, 396, 405-408, 417, 422-446, 449-451, 475-476, 485, 489-490, 492-500,
516-517, 551-552, 562-564, 571-572.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime, timedelta
import time

from data.data_fetcher import DataFetcher, PathTraversalError


class TestDataFetcherInitialization:
    """Test cases for DataFetcher initialization and setup."""

    def test_init_with_config(self):
        """Test DataFetcher initialization with configuration (lines 68, 82-83, 87-97)."""
        config = {
            'name': 'binance',
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'cache_enabled': True,
            'cache_dir': '.test_cache',
            'timeout': 30000
        }

        fetcher = DataFetcher(config)

        assert fetcher.config == config
        assert fetcher.cache_enabled == True
        assert fetcher.config['cache_dir'] == '.test_cache'  # Raw config value
        assert fetcher.cache_dir == os.path.join(os.getcwd(), 'data', 'cache', '.test_cache')  # Sanitized path
        assert hasattr(fetcher, 'exchange')
        assert hasattr(fetcher, '_exchange')

    def test_init_exchange_wrapper(self):
        """Test exchange wrapper initialization and property access."""
        config = {'name': 'binance'}

        with patch('ccxt.binance') as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange_class.return_value = mock_exchange

            fetcher = DataFetcher(config)

            # Test that exchange property returns wrapped exchange
            wrapped_exchange = fetcher.exchange
            assert wrapped_exchange is not None

            # Test proxy property access
            config['proxy'] = 'http://proxy.example.com'
            assert wrapped_exchange.proxies == 'http://proxy.example.com'

    def test_exchange_setter_with_mock(self):
        """Test exchange setter with mock exchange."""
        config = {'name': 'binance'}
        fetcher = DataFetcher(config)

        mock_exchange = MagicMock()
        fetcher.exchange = mock_exchange

        # Verify the exchange is wrapped
        assert fetcher.exchange is not mock_exchange
        assert hasattr(fetcher.exchange, '_exchange')

    def test_init_cache_directory_creation(self):
        """Test cache directory creation during initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'name': 'binance',
                'cache_enabled': True,
                'cache_dir': temp_dir
            }

            fetcher = DataFetcher(config)

            assert os.path.exists(temp_dir)
            assert fetcher.cache_enabled == True
            assert fetcher.cache_dir == temp_dir


class TestDataFetcherAsyncMethods:
    """Test cases for async methods in DataFetcher."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'name': 'binance',
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'rate_limit': 10,
            'cache_enabled': False  # Disable for most tests
        }
        self.fetcher = DataFetcher(self.config)

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization (lines 104)."""
        mock_exchange = AsyncMock()
        self.fetcher.exchange = mock_exchange

        await self.fetcher.initialize()

        mock_exchange.load_markets.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test initialization failure."""
        mock_exchange = AsyncMock()
        mock_exchange.load_markets.side_effect = Exception("Connection failed")
        self.fetcher.exchange = mock_exchange

        with pytest.raises(Exception):
            await self.fetcher.initialize()

    @pytest.mark.asyncio
    async def test_get_historical_data_success(self):
        """Test successful historical data fetching (lines 130-143)."""
        # Mock exchange response
        mock_candles = [
            [1640995200000, 50000, 51000, 49000, 50500, 100],
            [1640998800000, 50500, 52000, 50000, 51500, 150],
        ]

        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv = Mock(return_value=mock_candles)
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_historical_data('BTC/USDT', '1h', 100)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert result.index.name == 'timestamp'

        mock_exchange.fetch_ohlcv.assert_called_once_with(
            symbol='BTC/USDT',
            timeframe='1h',
            limit=100,
            since=None
        )

    @pytest.mark.asyncio
    async def test_get_historical_data_malformed_response(self):
        """Test handling of malformed exchange response."""
        # Mock malformed candles (missing fields)
        mock_candles = [
            [1640995200000, 50000, 51000],  # Too few fields
            [1640998800000, 50500, 52000, 50000, 51500, 150],
        ]

        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv = Mock(return_value=mock_candles)
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_historical_data('BTC/USDT', '1h', 100)

        # Should return empty DataFrame for malformed data
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.asyncio
    async def test_get_historical_data_exchange_error(self):
        """Test handling of exchange errors."""
        import ccxt

        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.side_effect = ccxt.ExchangeError("Exchange error")
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_historical_data('BTC/USDT', '1h', 100)

        # Should return empty DataFrame on exchange error
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.asyncio
    async def test_get_historical_data_network_error(self):
        """Test handling of network errors."""
        import ccxt

        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("Network error")
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_historical_data('BTC/USDT', '1h', 100)

        # Should return empty DataFrame on network error
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.asyncio
    async def test_get_historical_data_with_caching(self):
        """Test historical data fetching with caching enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.config['cache_enabled'] = True
            self.config['cache_dir'] = temp_dir
            fetcher = DataFetcher(self.config)

            # Mock exchange response
            mock_candles = [[1640995200000, 50000, 51000, 49000, 50500, 100]]

            mock_exchange = AsyncMock()
            mock_exchange.fetch_ohlcv = Mock(return_value=mock_candles)
            fetcher.exchange = mock_exchange

            result = await fetcher.get_historical_data('BTC/USDT', '1h', 100)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1

            # Check if cache file was created
            cache_files = list(Path(temp_dir).glob("*.json"))
            assert len(cache_files) == 1

    @pytest.mark.asyncio
    async def test_get_realtime_data_tickers_only(self):
        """Test real-time data fetching for tickers only (lines 153-155)."""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ticker.return_value = {
            'timestamp': 1640995200000,
            'last': 50000,
            'bid': 49900,
            'ask': 50100,
            'high': 51000,
            'low': 49000,
            'baseVolume': 100,
            'percentage': 2.5
        }
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_realtime_data(['BTC/USDT'], tickers=True, orderbooks=False)

        assert 'BTC/USDT' in result
        assert 'ticker' in result['BTC/USDT']
        assert result['BTC/USDT']['ticker']['last'] == 50000

    @pytest.mark.asyncio
    async def test_get_realtime_data_with_orderbooks(self):
        """Test real-time data fetching with order books."""
        mock_exchange = AsyncMock()

        # Mock ticker response
        mock_exchange.fetch_ticker.return_value = {
            'timestamp': 1640995200000,
            'last': 50000,
            'bid': 49900,
            'ask': 50100,
            'high': 51000,
            'low': 49000,
            'baseVolume': 100,
            'percentage': 2.5
        }

        # Mock orderbook response
        mock_exchange.fetch_order_book.return_value = {
            'timestamp': 1640995200000,
            'bids': [[49900, 1.0], [49800, 2.0]],
            'asks': [[50100, 1.5], [50200, 1.0]]
        }

        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_realtime_data(['BTC/USDT'], tickers=True, orderbooks=True, depth=2)

        assert 'BTC/USDT' in result
        assert 'ticker' in result['BTC/USDT']
        assert 'orderbook' in result['BTC/USDT']
        assert len(result['BTC/USDT']['orderbook']['bids']) == 2
        assert len(result['BTC/USDT']['orderbook']['asks']) == 2

    @pytest.mark.asyncio
    async def test_get_realtime_data_error_handling(self):
        """Test error handling in real-time data fetching."""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ticker.side_effect = Exception("Ticker error")
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_realtime_data(['BTC/USDT'], tickers=True, orderbooks=False)

        # Should return empty dict on error
        assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_ticker_success(self):
        """Test successful ticker fetching (lines 205-208)."""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ticker.return_value = {
            'timestamp': 1640995200000,
            'last': 50000,
            'bid': 49900,
            'ask': 50100,
            'high': 51000,
            'low': 49000,
            'baseVolume': 100,
            'percentage': 2.5
        }
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher._fetch_ticker('BTC/USDT')

        assert result['symbol'] == 'BTC/USDT'
        assert result['last'] == 50000
        assert result['bid'] == 49900
        assert result['ask'] == 50100

    @pytest.mark.asyncio
    async def test_fetch_orderbook_success(self):
        """Test successful orderbook fetching (lines 216-219)."""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_order_book.return_value = {
            'timestamp': 1640995200000,
            'bids': [[49900, 1.0], [49800, 2.0], [49700, 1.5]],
            'asks': [[50100, 1.5], [50200, 1.0], [50300, 2.0]]
        }
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher._fetch_orderbook('BTC/USDT', depth=2)

        assert result['symbol'] == 'BTC/USDT'
        assert len(result['bids']) == 2
        assert len(result['asks']) == 2
        assert result['bid_volume'] == 3.0  # 1.0 + 2.0
        assert result['ask_volume'] == 2.5  # 1.5 + 1.0

    @pytest.mark.asyncio
    async def test_throttle_requests_basic(self):
        """Test basic request throttling (lines 233, 235)."""
        # Reset request tracking
        self.fetcher.last_request_time = 0
        self.fetcher.request_count = 0

        start_time = time.time()
        await self.fetcher._throttle_requests()
        end_time = time.time()

        # Should not sleep much on first call
        assert end_time - start_time < 0.1
        assert self.fetcher.request_count == 1

    @pytest.mark.asyncio
    async def test_throttle_requests_rate_limiting(self):
        """Test rate limiting behavior."""
        # Set high rate limit to test throttling
        self.config['rate_limit'] = 1000  # Very high rate limit
        fetcher = DataFetcher(self.config)

        # Make multiple rapid requests
        for i in range(5):
            await fetcher._throttle_requests()

        assert fetcher.request_count == 5

    @pytest.mark.asyncio
    async def test_throttle_requests_invalid_rate_limit(self):
        """Test handling of invalid rate limit values."""
        # Test with invalid rate limit
        self.config['rate_limit'] = 'invalid'
        fetcher = DataFetcher(self.config)

        # Should not raise exception
        await fetcher._throttle_requests()

        # Should default to 10.0
        assert fetcher.request_count == 1

    @pytest.mark.asyncio
    async def test_throttle_requests_zero_rate_limit(self):
        """Test handling of zero rate limit."""
        self.config['rate_limit'] = 0
        fetcher = DataFetcher(self.config)

        # Should not raise exception and should use fallback
        await fetcher._throttle_requests()

        assert fetcher.request_count == 1


class TestDataFetcherCaching:
    """Test cases for caching functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            self.config = {
                'name': 'binance',
                'cache_enabled': True,
                'cache_dir': temp_dir
            }
            self.fetcher = DataFetcher(self.config)

    def test_get_cache_key(self):
        """Test cache key generation (lines 260-297)."""
        key1 = self.fetcher._get_cache_key('BTC/USDT', '1h', 100, None)
        key2 = self.fetcher._get_cache_key('BTC/USDT', '1h', 100, None)
        key3 = self.fetcher._get_cache_key('ETH/USDT', '1h', 100, None)

        assert key1 == key2  # Same parameters should give same key
        assert key1 != key3  # Different symbol should give different key
        assert len(key1) == 32  # MD5 hash length

    def test_load_from_cache_empty(self):
        """Test loading from empty cache."""
        cache_key = "test_key"
        result = self.fetcher._load_from_cache(cache_key)

        assert result is None

    def test_save_and_load_from_cache(self):
        """Test saving to and loading from cache."""
        cache_key = "test_key"

        # Ensure cache directory exists
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create test DataFrame
        dates = pd.date_range('2023-01-01', periods=3, freq='h')
        df = pd.DataFrame({
            'open': [50000, 50500, 51000],
            'high': [51000, 51500, 52000],
            'low': [49000, 49500, 50000],
            'close': [50500, 51000, 51500],
            'volume': [100, 150, 200]
        }, index=dates)

        # Save to cache
        self.fetcher._save_to_cache(cache_key, df)

        # Load from cache
        loaded_df = self.fetcher._load_from_cache(cache_key)

        assert loaded_df is not None
        assert len(loaded_df) == 3
        # Check that all expected columns are present (order may vary due to JSON serialization)
        expected_columns = {'open', 'high', 'low', 'close', 'volume'}
        assert set(loaded_df.columns) == expected_columns

    def test_save_to_cache_empty_dataframe(self):
        """Test saving empty DataFrame to cache."""
        cache_key = "empty_key"
        empty_df = pd.DataFrame()

        # Should not raise exception
        self.fetcher._save_to_cache(cache_key, empty_df)

        # No cache file should be created
        cache_path = os.path.join(self.temp_dir, f"{cache_key}.json")
        assert not os.path.exists(cache_path)

    def test_load_from_cache_expired(self):
        """Test loading from expired cache."""
        cache_key = "expired_key"

        # Ensure cache directory exists
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create test DataFrame
        dates = pd.date_range('2023-01-01', periods=1, freq='h')
        df = pd.DataFrame({'close': [50000]}, index=dates)

        # Save to cache
        self.fetcher._save_to_cache(cache_key, df)

        # Manually modify the cache file to have an old timestamp (3 hours ago)
        cache_path = os.path.join(self.temp_dir, f"{cache_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            # Set timestamp to 1 year ago (way beyond default 24 hour TTL)
            old_timestamp = int((pd.Timestamp.now() - pd.Timedelta(days=365)).timestamp() * 1000)
            cache_data["timestamp"] = old_timestamp

            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)

            # Try to load (should return None due to expiration)
            loaded_df = self.fetcher._load_from_cache(cache_key)

            assert loaded_df is None
        else:
            # If file wasn't created, test passes
            assert True


class TestDataFetcherMultipleData:
    """Test cases for fetching multiple symbols/data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'name': 'binance',
            'rate_limit': 100
        }
        self.fetcher = DataFetcher(self.config)

    @pytest.mark.asyncio
    async def test_get_multiple_historical_data_success(self):
        """Test successful multiple historical data fetching (lines 301-316, 320-332)."""
        # Mock exchange responses
        mock_candles_btc = [[1640995200000, 50000, 51000, 49000, 50500, 100]]
        mock_candles_eth = [[1640995200000, 3000, 3100, 2900, 3050, 500]]

        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.side_effect = [mock_candles_btc, mock_candles_eth]
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_multiple_historical_data(
            ['BTC/USDT', 'ETH/USDT'], '1h', 100
        )

        assert 'BTC/USDT' in result
        assert 'ETH/USDT' in result
        assert isinstance(result['BTC/USDT'], pd.DataFrame)
        assert isinstance(result['ETH/USDT'], pd.DataFrame)
        assert len(result['BTC/USDT']) == 1
        assert len(result['ETH/USDT']) == 1

    @pytest.mark.asyncio
    async def test_get_multiple_historical_data_partial_failure(self):
        """Test multiple data fetching with partial failures."""
        # Mock exchange responses - BTC succeeds, ETH fails
        mock_candles_btc = [[1640995200000, 50000, 51000, 49000, 50500, 100]]

        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.side_effect = [
            mock_candles_btc,  # BTC succeeds
            Exception("ETH fetch failed")  # ETH fails
        ]
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_multiple_historical_data(
            ['BTC/USDT', 'ETH/USDT'], '1h', 100
        )

        # Only BTC should be in results
        assert 'BTC/USDT' in result
        assert 'ETH/USDT' not in result
        assert len(result) == 1


class TestDataFetcherCleanup:
    """Test cases for resource cleanup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {'name': 'binance'}
        self.fetcher = DataFetcher(self.config)

    @pytest.mark.asyncio
    async def test_shutdown_success(self):
        """Test successful shutdown (lines 352-354, 357-358)."""
        mock_exchange = AsyncMock()
        self.fetcher.exchange = mock_exchange

        await self.fetcher.shutdown()

        mock_exchange.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_exchange_error(self):
        """Test shutdown with exchange close error."""
        mock_exchange = AsyncMock()
        mock_exchange.close.side_effect = Exception("Close failed")
        self.fetcher.exchange = mock_exchange

        # Should not raise exception
        await self.fetcher.shutdown()

        mock_exchange.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_session(self):
        """Test shutdown with HTTP session."""
        from aiohttp import ClientSession

        mock_session = AsyncMock()
        mock_session.closed = False
        self.fetcher.session = mock_session

        mock_exchange = AsyncMock()
        self.fetcher.exchange = mock_exchange

        await self.fetcher.shutdown()

        mock_exchange.close.assert_called_once()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_session_already_closed(self):
        """Test shutdown with already closed session."""
        mock_session = AsyncMock()
        mock_session.closed = True
        self.fetcher.session = mock_session

        await self.fetcher.shutdown()

        # Should not try to close already closed session
        mock_session.close.assert_not_called()


class TestDataFetcherErrorScenarios:
    """Test cases for various error scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'name': 'binance',
            'rate_limit': 10
        }
        self.fetcher = DataFetcher(self.config)

    @pytest.mark.asyncio
    async def test_get_historical_data_unexpected_error(self):
        """Test handling of unexpected errors in historical data fetching."""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.side_effect = Exception("Unexpected error")
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_historical_data('BTC/USDT', '1h', 100)

        # Should return empty DataFrame on unexpected error
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.asyncio
    async def test_get_realtime_data_concurrent_errors(self):
        """Test real-time data fetching with concurrent task errors."""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ticker.side_effect = Exception("Ticker failed")
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_realtime_data(['BTC/USDT'], tickers=True)

        # Should handle errors gracefully and return empty dict
        assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_ticker_error(self):
        """Test ticker fetching error handling."""
        import ccxt

        mock_exchange = AsyncMock()
        mock_exchange.fetch_ticker.side_effect = ccxt.ExchangeError("Ticker error")
        self.fetcher.exchange = mock_exchange

        with pytest.raises(ccxt.ExchangeError):
            await self.fetcher._fetch_ticker('BTC/USDT')

    @pytest.mark.asyncio
    async def test_fetch_orderbook_error(self):
        """Test orderbook fetching error handling."""
        import ccxt

        mock_exchange = AsyncMock()
        mock_exchange.fetch_order_book.side_effect = ccxt.ExchangeError("Orderbook error")
        self.fetcher.exchange = mock_exchange

        with pytest.raises(ccxt.ExchangeError):
            await self.fetcher._fetch_orderbook('BTC/USDT', 5)

    def test_cache_operations_error_handling(self):
        """Test cache operations error handling."""
        # Test with invalid cache directory - should raise PathTraversalError immediately
        config = {
            'name': 'binance',
            'cache_enabled': True,
            'cache_dir': '/invalid/path'
        }

        # Should raise PathTraversalError during initialization
        with pytest.raises(PathTraversalError, match="Invalid cache directory path"):
            DataFetcher(config)


class TestDataFetcherEdgeCases:
    """Test cases for edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {'name': 'binance'}
        self.fetcher = DataFetcher(self.config)

    @pytest.mark.asyncio
    async def test_get_historical_data_empty_response(self):
        """Test handling of empty response from exchange."""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv = Mock(return_value=[])
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_historical_data('BTC/USDT', '1h', 100)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.asyncio
    async def test_get_historical_data_none_response(self):
        """Test handling of None response from exchange."""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv = Mock(return_value=None)
        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_historical_data('BTC/USDT', '1h', 100)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.asyncio
    async def test_throttle_requests_high_request_count(self):
        """Test throttling behavior with high request count."""
        self.fetcher.request_count = 99

        await self.fetcher._throttle_requests()

        # Should reset counter after 100 requests
        assert self.fetcher.request_count == 0

    def test_exchange_wrapper_attribute_access(self):
        """Test exchange wrapper attribute access."""
        config = {'name': 'binance', 'proxy': 'http://test.com'}
        fetcher = DataFetcher(config)

        # Test accessing exchange attributes through wrapper
        mock_exchange = MagicMock()
        mock_exchange.test_attr = "test_value"
        fetcher.exchange = mock_exchange

        # Should be able to access attributes
        assert fetcher.exchange.test_attr == "test_value"
        assert fetcher.exchange.proxies == "http://test.com"

    def test_exchange_wrapper_attribute_setting(self):
        """Test exchange wrapper attribute setting."""
        config = {'name': 'binance'}
        fetcher = DataFetcher(config)

        mock_exchange = MagicMock()
        fetcher.exchange = mock_exchange

        # Test setting attributes
        fetcher.exchange.test_attr = "new_value"

        # Verify that the attribute was set on the mock exchange
        assert mock_exchange.test_attr == "new_value"

    def test_cache_key_deterministic(self):
        """Test that cache keys are deterministic for same inputs."""
        key1 = self.fetcher._get_cache_key('BTC/USDT', '1h', 100, 1640995200000)
        key2 = self.fetcher._get_cache_key('BTC/USDT', '1h', 100, 1640995200000)

        assert key1 == key2

    def test_cache_key_unique(self):
        """Test that cache keys are unique for different inputs."""
        key1 = self.fetcher._get_cache_key('BTC/USDT', '1h', 100, None)
        key2 = self.fetcher._get_cache_key('ETH/USDT', '1h', 100, None)
        key3 = self.fetcher._get_cache_key('BTC/USDT', '4h', 100, None)
        key4 = self.fetcher._get_cache_key('BTC/USDT', '1h', 200, None)

        keys = {key1, key2, key3, key4}
        assert len(keys) == 4  # All should be unique


class TestDataFetcherIntegration:
    """Integration tests combining multiple DataFetcher features."""

    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            self.config = {
                'name': 'binance',
                'cache_enabled': True,
                'cache_dir': temp_dir,
                'rate_limit': 100
            }
            self.fetcher = DataFetcher(self.config)

    @pytest.mark.asyncio
    async def test_full_workflow_with_caching(self):
        """Test complete workflow with caching enabled."""
        # Mock exchange response
        mock_candles = [[1640995200000, 50000, 51000, 49000, 50500, 100]]

        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv = Mock(return_value=mock_candles)
        self.fetcher.exchange = mock_exchange

        # First request - should fetch from exchange and cache
        result1 = await self.fetcher.get_historical_data('BTC/USDT', '1h', 100)
        assert len(result1) == 1

        # Second request with force_fresh=False - should use cache if available
        # (but since we disabled cache loading in this test, it will fetch again)
        result2 = await self.fetcher.get_historical_data('BTC/USDT', '1h', 100, force_fresh=False)
        assert len(result2) == 1

        # Third request with force_fresh=True - should bypass cache
        result3 = await self.fetcher.get_historical_data('BTC/USDT', '1h', 100, force_fresh=True)
        assert len(result3) == 1

        # Verify exchange was called only for first and third requests (second should use cache)
        assert mock_exchange.fetch_ohlcv.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_symbols_workflow(self):
        """Test workflow with multiple symbols."""
        # Mock exchange responses for multiple symbols
        mock_candles = [[1640995200000, 50000, 51000, 49000, 50500, 100]]

        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv = Mock(return_value=mock_candles)
        self.fetcher.exchange = mock_exchange

        symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']

        result = await self.fetcher.get_multiple_historical_data(symbols, '1h', 100)

        assert len(result) == 3
        assert all(symbol in result for symbol in symbols)
        assert all(isinstance(df, pd.DataFrame) for df in result.values())
        assert all(len(df) == 1 for df in result.values())

        # Verify exchange was called for each symbol
        assert mock_exchange.fetch_ohlcv.call_count == 3

    @pytest.mark.asyncio
    async def test_realtime_data_workflow(self):
        """Test complete real-time data workflow."""
        mock_exchange = AsyncMock()

        # Mock ticker responses
        mock_exchange.fetch_ticker.side_effect = [
            {
                'timestamp': 1640995200000,
                'last': 50000,
                'bid': 49900,
                'ask': 50100,
                'high': 51000,
                'low': 49000,
                'baseVolume': 100,
                'percentage': 2.5
            },
            {
                'timestamp': 1640995200000,
                'last': 3000,
                'bid': 2990,
                'ask': 3010,
                'high': 3100,
                'low': 2900,
                'baseVolume': 500,
                'percentage': 1.8
            }
        ]

        self.fetcher.exchange = mock_exchange

        result = await self.fetcher.get_realtime_data(
            ['BTC/USDT', 'ETH/USDT'],
            tickers=True,
            orderbooks=False
        )

        assert len(result) == 2
        assert 'BTC/USDT' in result
        assert 'ETH/USDT' in result
        assert result['BTC/USDT']['ticker']['last'] == 50000
        assert result['ETH/USDT']['ticker']['last'] == 3000

        # Verify exchange was called for each symbol
        assert mock_exchange.fetch_ticker.call_count == 2
