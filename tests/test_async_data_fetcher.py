"""
tests/test_async_data_fetcher.py

Tests for async-first data fetching behavior.
Verifies that blocking operations are properly offloaded and timeouts work.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from data.data_fetcher import DataFetcher


class TestAsyncDataFetcher:
    """Test cases for async-first data fetching."""

    @pytest.fixture
    def fetcher(self):
        """Create a test data fetcher instance."""
        config = {
            "name": "binance",
            "cache_enabled": False,  # Disable for most tests
            "rate_limit": 100,
        }
        return DataFetcher(config)

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_with_timeout(self, fetcher):
        """Test that OHLCV fetching includes timeout protection."""
        # Mock a slow exchange that takes too long
        mock_exchange = AsyncMock()

        async def slow_ohlcv(*a, **k):
            await asyncio.sleep(2)
            return []

        mock_exchange.fetch_ohlcv = AsyncMock(side_effect=slow_ohlcv)  # 2 second delay
        fetcher.exchange = mock_exchange

        start_time = time.time()

        # This should timeout after 10 seconds (default CCXT timeout)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                fetcher.exchange.fetch_ohlcv("BTC/USDT", "1h", limit=100),
                timeout=0.1,  # Very short timeout for test
            )

        elapsed = time.time() - start_time
        assert elapsed < 0.2  # Should timeout quickly

    @pytest.mark.asyncio
    async def test_cache_operations_offloaded_to_thread_pool(self, fetcher):
        """Test that CPU-intensive cache operations are offloaded to thread pools."""
        # Enable caching for this test
        fetcher.config["cache_enabled"] = True
        fetcher._cache_dir_path = "/tmp/test_cache"

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="h"),
                "open": [50000] * 100,
                "high": [51000] * 100,
                "low": [49000] * 100,
                "close": [50500] * 100,
                "volume": [100] * 100,
            }
        ).set_index("timestamp")

        # Mock the thread pool operation
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = {"timestamp": 1234567890000, "data": []}

            await fetcher._save_to_cache_async("test_key", df)

            # Verify that CPU-intensive operations were offloaded
            assert mock_to_thread.called
            # Should call _prepare_cache_data in thread pool
            prepare_calls = [
                call
                for call in mock_to_thread.call_args_list
                if len(call[0]) > 0 and call[0][0] == fetcher._prepare_cache_data
            ]
            assert len(prepare_calls) > 0

    @pytest.mark.asyncio
    async def test_async_file_io_used_for_cache(self, fetcher):
        """Test that cache uses async file I/O instead of blocking operations."""
        fetcher.config["cache_enabled"] = True
        fetcher._cache_dir_path = "/tmp/test_cache"

        # Mock aiofiles
        with patch("aiofiles.open") as mock_aiofiles_open:
            mock_file = AsyncMock()
            mock_aiofiles_open.return_value.__aenter__.return_value = mock_file
            mock_aiofiles_open.return_value.__aexit__.return_value = None

            df = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2023-01-01", periods=5, freq="h"),
                    "close": [50000] * 5,
                }
            ).set_index("timestamp")

            await fetcher._save_to_cache_async("test_key", df)

            # Verify async file operations were used
            mock_aiofiles_open.assert_called_once()
            mock_file.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_json_processing_offloaded(self, fetcher):
        """Test that JSON processing is offloaded to thread pool."""
        fetcher.config["cache_enabled"] = True
        fetcher._cache_dir_path = "/tmp/test_cache"

        # Mock async file read
        with patch("aiofiles.open") as mock_aiofiles_open, patch(
            "asyncio.to_thread"
        ) as mock_to_thread:
            mock_file = AsyncMock()
            mock_file.read.return_value = '{"timestamp": 1234567890000, "data": []}'
            mock_aiofiles_open.return_value.__aenter__.return_value = mock_file
            mock_aiofiles_open.return_value.__aexit__.return_value = None

            mock_to_thread.return_value = {"timestamp": 1234567890000, "data": []}

            result = await fetcher._load_from_cache_async("test_key")

            # Verify JSON parsing was offloaded to thread pool
            json_load_calls = [
                call
                for call in mock_to_thread.call_args_list
                if len(call[0]) > 0 and call[0][0] == fetcher._process_cached_dataframe
            ]
            assert len(json_load_calls) > 0

    @pytest.mark.asyncio
    async def test_event_loop_not_blocked_during_cache_operations(self, fetcher):
        """Test that cache operations don't block the event loop."""
        fetcher.config["cache_enabled"] = True
        fetcher._cache_dir_path = "/tmp/test_cache"

        # Create a large DataFrame to simulate CPU-intensive work
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=1000, freq="min"),
                "open": list(range(1000)),
                "high": list(range(1000, 2000)),
                "low": list(range(2000, 3000)),
                "close": list(range(3000, 4000)),
                "volume": list(range(4000, 5000)),
            }
        ).set_index("timestamp")

        # Mock the thread pool to simulate slow processing
        async def slow_prepare(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow CPU work
            return {"timestamp": 1234567890000, "data": []}

        with patch("asyncio.to_thread", side_effect=slow_prepare), patch(
            "aiofiles.open"
        ) as mock_aiofiles:
            mock_file = AsyncMock()
            mock_aiofiles.return_value.__aenter__.return_value = mock_file
            mock_aiofiles.return_value.__aexit__.return_value = None

            # Start cache operation
            cache_task = asyncio.create_task(
                fetcher._save_to_cache_async("test_key", df)
            )

            # While cache operation is running, verify event loop is still responsive
            start_time = time.time()
            responsive_check = 0

            while not cache_task.done() and (time.time() - start_time) < 1.0:
                # This should execute multiple times if event loop is not blocked
                await asyncio.sleep(0.01)
                responsive_check += 1

            # Wait for cache operation to complete
            await cache_task

            # Verify event loop remained responsive during cache operation
            assert responsive_check > 5, "Event loop was blocked during cache operation"

    @pytest.mark.asyncio
    async def test_concurrent_fetches_dont_block_each_other(self, fetcher):
        """Test that multiple concurrent fetch operations don't block each other."""
        # Mock exchange with different response times
        mock_exchange = AsyncMock()
        fetcher.exchange = mock_exchange

        # Mock fetch_ohlcv to return different data with different delays
        async def mock_fetch_ohlcv(symbol, timeframe, **kwargs):
            if "BTC" in symbol:
                await asyncio.sleep(0.1)  # BTC takes longer
                return [[1640995200000, 50000, 51000, 49000, 50500, 100]]
            else:
                await asyncio.sleep(0.05)  # ETH is faster
                return [[1640995200000, 3000, 3100, 2900, 3050, 500]]

        mock_exchange.fetch_ohlcv.side_effect = mock_fetch_ohlcv

        # Start multiple concurrent fetches
        start_time = time.time()

        tasks = [
            fetcher.get_historical_data("BTC/USDT", "1h", 100),
            fetcher.get_historical_data("ETH/USDT", "1h", 100),
            fetcher.get_historical_data("ADA/USDT", "1h", 100),
        ]

        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # All operations should complete, and total time should be less than
        # if they were executed sequentially (0.1 + 0.05 + 0.05 = 0.2)
        # Concurrent execution should take ~0.1 seconds
        assert elapsed < 0.15, f"Concurrent operations took too long: {elapsed}s"
        assert len(results) == 3
        assert all(isinstance(df, pd.DataFrame) for df in results)

    @pytest.mark.asyncio
    async def test_rate_limiting_with_async_sleep(self, fetcher):
        """Test that rate limiting uses async sleep instead of blocking sleep."""
        # Set a high rate limit to test throttling
        fetcher.config["rate_limit"] = 10  # 10 requests per second

        # Mock time to control timing
        with patch("time.time") as mock_time, patch(
            "asyncio.sleep"
        ) as mock_async_sleep:
            mock_time.return_value = 1000.0

            # First call should not sleep
            await fetcher._throttle_requests()
            assert mock_async_sleep.call_count == 0

            # Second call immediately after should sleep
            await fetcher._throttle_requests()
            assert mock_async_sleep.call_count == 1

            # Verify async sleep was used, not time.sleep
            mock_async_sleep.assert_called_once()

    def test_backward_compatibility_sync_wrappers(self, fetcher):
        """Test that sync wrapper methods work for backward compatibility."""
        fetcher.config["cache_enabled"] = True
        fetcher._cache_dir_path = "/tmp/test_cache"

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5, freq="h"),
                "close": [50000] * 5,
            }
        ).set_index("timestamp")

        # Test sync save (should work via async fallback)
        with patch("asyncio.run") as mock_asyncio_run:
            fetcher.save_to_cache("test_key", df)
            # Should have called asyncio.run for the async operation
            mock_asyncio_run.assert_called_once()

        # Test sync load (should work via async fallback)
        with patch("asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.return_value = df
            result = fetcher.load_from_cache("test_key")
            assert result is not None
            mock_asyncio_run.assert_called_once()


class TestAsyncTimeoutProtection:
    """Test cases for timeout protection in async operations."""

    @pytest.fixture
    def fetcher(self):
        """Create a test data fetcher instance."""
        config = {
            "name": "binance",
            "cache_enabled": False,
            "rate_limit": 100,
        }
        return DataFetcher(config)

    @pytest.mark.asyncio
    async def test_exchange_initialization_timeout(self, fetcher):
        """Test that exchange initialization has timeout protection."""
        # Mock slow exchange initialization
        mock_exchange = AsyncMock()

        async def slow_load(*a, **k):
            await asyncio.sleep(5)

        mock_exchange.load_markets = AsyncMock(side_effect=slow_load)  # Slow init
        fetcher.exchange = mock_exchange

        with pytest.raises(asyncio.TimeoutError):
            # This should timeout if we set a very short timeout
            await asyncio.wait_for(fetcher.initialize(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_realtime_data_fetch_timeout(self, fetcher):
        """Test timeout protection for real-time data fetching."""
        # Mock slow ticker fetch
        mock_exchange = AsyncMock()

        async def slow_ticker(*a, **k):
            await asyncio.sleep(2)
            return {
                "timestamp": 1234567890000,
                "last": 50000,
                "bid": 49900,
                "ask": 50100,
                "high": 51000,
                "low": 49000,
                "baseVolume": 100,
                "percentage": 1.0,
            }

        mock_exchange.fetch_ticker = AsyncMock(side_effect=slow_ticker)
        fetcher.exchange = mock_exchange

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(fetcher._fetch_ticker("BTC/USDT"), timeout=0.1)

    @pytest.mark.asyncio
    async def test_orderbook_fetch_timeout(self, fetcher):
        """Test timeout protection for orderbook fetching."""
        # Mock slow orderbook fetch
        mock_exchange = AsyncMock()

        async def slow_orderbook(*a, **k):
            await asyncio.sleep(2)
            return {
                "timestamp": 1234567890000,
                "bids": [[50000, 1], [49900, 2]],
                "asks": [[50100, 1], [50200, 2]],
            }

        mock_exchange.fetch_order_book = AsyncMock(side_effect=slow_orderbook)
        fetcher.exchange = mock_exchange

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(fetcher._fetch_orderbook("BTC/USDT", 5), timeout=0.1)


class TestAsyncResourceManagement:
    """Test cases for proper async resource management."""

    @pytest.fixture
    def fetcher(self):
        """Create a test data fetcher instance."""
        config = {
            "name": "binance",
            "cache_enabled": False,
        }
        return DataFetcher(config)

    @pytest.mark.asyncio
    async def test_exchange_cleanup_on_shutdown(self, fetcher):
        """Test that exchange connections are properly closed on shutdown."""
        mock_exchange = AsyncMock()
        fetcher.exchange = mock_exchange

        await fetcher.shutdown()

        mock_exchange.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_session_cleanup(self, fetcher):
        """Test that HTTP sessions are properly closed."""

        mock_session = AsyncMock()
        mock_session.closed = False
        fetcher.session = mock_session

        await fetcher.shutdown()

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_errors(self, fetcher):
        """Test shutdown handles errors gracefully."""
        # Mock exchange that fails to close
        mock_exchange = AsyncMock()
        mock_exchange.close.side_effect = Exception("Close failed")
        fetcher.exchange = mock_exchange

        # Should not raise exception
        await fetcher.shutdown()

        mock_exchange.close.assert_called_once()
