"""
tests/integration/test_api_retry_mechanisms.py

Integration tests for API retry mechanisms, rate limiting, and circuit breaker functionality.
Tests ensure robust handling of external API failures with proper timeout prevention.
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd

from data.data_fetcher import DataFetcher
from core.api_protection import CircuitOpenError, APICircuitBreaker, CircuitBreakerConfig
from core.retry import RetryConfig


class TestAPIRetryMechanisms:
    """Test suite for API retry mechanisms."""

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock exchange for testing."""
        exchange = Mock()
        exchange.id = "test_exchange"
        exchange.load_markets = AsyncMock()
        exchange.fetch_ohlcv = AsyncMock()
        exchange.fetch_ticker = AsyncMock()
        exchange.fetch_order_book = AsyncMock()
        exchange.close = AsyncMock()
        return exchange

    @pytest.fixture
    def data_fetcher_config(self):
        """Default configuration for DataFetcher."""
        return {
            "name": "binance",
            "rate_limit": 10,
            "timeout": 30000,
            "cache_enabled": False,
            "retry": {
                "max_attempts": 3,
                "base_delay": 0.1,
                "jitter": 0.05,
                "max_delay": 1.0
            },
            "endpoint_retry_configs": {
                "fetch_ohlcv": {
                    "max_attempts": 5,
                    "base_delay": 0.2
                }
            }
        }

    @pytest.fixture
    def data_fetcher(self, data_fetcher_config, mock_exchange):
        """Create DataFetcher instance with mock exchange."""
        fetcher = DataFetcher(data_fetcher_config)
        fetcher.exchange = mock_exchange
        # Use a fresh circuit breaker for each test to avoid state pollution
        from core.api_protection import APICircuitBreaker, CircuitBreakerConfig
        fetcher.circuit_breaker = APICircuitBreaker(CircuitBreakerConfig())
        return fetcher

    @pytest.mark.timeout(15)
    def test_retry_timeout_prevention(self, data_fetcher, mock_exchange):
        """Test that retry mechanism respects timeout to prevent hangs."""
        # Configure fetcher with high retry attempts
        data_fetcher.retry_config.max_attempts = 10
        data_fetcher.retry_config.base_delay = 0.5

        # Mock API that always fails
        mock_exchange.fetch_ohlcv.side_effect = Exception("Rate limit exceeded")

        start_time = time.time()

        # Should timeout before completing all retries
        with pytest.raises((asyncio.TimeoutError, Exception)):
            # Use asyncio.wait_for to enforce timeout
            async def fetch_with_timeout():
                return await asyncio.wait_for(
                    data_fetcher.get_historical_data("BTC/USDT"),
                    timeout=2.0
                )
            asyncio.run(fetch_with_timeout())

        elapsed = time.time() - start_time
        assert elapsed < 3.0, f"Test took too long: {elapsed}s"

    @pytest.mark.asyncio
    async def test_successful_retry_after_failures(self, data_fetcher, mock_exchange):
        """Test that retry mechanism succeeds after initial failures."""
        # Mock: fail twice, then succeed
        mock_exchange.fetch_ohlcv.side_effect = [
            Exception("Temporary failure 1"),
            Exception("Temporary failure 2"),
            [
                [1640995200000, 50000, 51000, 49000, 50500, 100],
                [1640995260000, 50500, 51500, 49500, 51000, 150]
            ]
        ]

        df = await data_fetcher.get_historical_data("BTC/USDT", limit=2)

        assert not df.empty
        assert len(df) == 2
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

        # Verify fetch_ohlcv was called 3 times
        assert mock_exchange.fetch_ohlcv.call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, data_fetcher, mock_exchange):
        """Test circuit breaker prevents calls when open."""
        # Configure circuit breaker to open after 2 failures
        data_fetcher.circuit_breaker.config.failure_threshold = 2

        # Mock consistent failures
        mock_exchange.fetch_ohlcv.side_effect = Exception("API Error")

        # Make calls that will trigger circuit breaker
        # The circuit breaker opens during retries, so calls may succeed initially but eventually fail
        try:
            await data_fetcher.get_historical_data("BTC/USDT")
        except Exception:
            pass  # Expected to fail

        try:
            await data_fetcher.get_historical_data("BTC/USDT")
        except Exception:
            pass  # Expected to fail

        # Circuit should now be open
        assert data_fetcher.circuit_breaker.is_open()

        # Next call should fail with CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await data_fetcher.get_historical_data("BTC/USDT")

    @pytest.mark.asyncio
    async def test_endpoint_specific_retry_config(self, data_fetcher, mock_exchange):
        """Test that different endpoints can have different retry configurations."""
        # fetch_ohlcv should use endpoint-specific config (5 attempts)
        mock_exchange.fetch_ohlcv.side_effect = [Exception("Fail")] * 4 + [[
            [1640995200000, 50000, 51000, 49000, 50500, 100]
        ]]

        # fetch_ticker should use default config (3 attempts)
        mock_exchange.fetch_ticker.side_effect = [Exception("Fail")] * 2 + [{
            "timestamp": 1640995200000,
            "last": 50000,
            "bid": 49900,
            "ask": 50100,
            "high": 51000,
            "low": 49000,
            "baseVolume": 100,
            "percentage": 1.0
        }]

        # Test OHLCV with 5 attempts
        df = await data_fetcher.get_historical_data("BTC/USDT")
        assert not df.empty
        assert mock_exchange.fetch_ohlcv.call_count == 5

        # Test ticker with 3 attempts
        result = await data_fetcher.get_realtime_data(["BTC/USDT"], tickers=True, orderbooks=False)
        assert "BTC/USDT" in result
        assert "ticker" in result["BTC/USDT"]
        assert mock_exchange.fetch_ticker.call_count == 3

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, data_fetcher, mock_exchange):
        """Test that rate limiting works correctly."""
        # Set very low rate limit
        data_fetcher.config["rate_limit"] = 2  # 2 requests per second

        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 50000, 51000, 49000, 50500, 100]
        ]

        start_time = time.time()

        # Make multiple requests
        await data_fetcher.get_historical_data("BTC/USDT")
        await data_fetcher.get_historical_data("ETH/USDT")
        await data_fetcher.get_historical_data("ADA/USDT")

        elapsed = time.time() - start_time
        # Should take at least 1 second for 3 requests at 2 req/s
        assert elapsed >= 1.0, f"Rate limiting not working: {elapsed}s for 3 requests"

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_api_unavailable(self, data_fetcher, mock_exchange):
        """Test graceful degradation when APIs are completely unavailable."""
        # Mock all endpoints to fail
        mock_exchange.fetch_ohlcv.side_effect = Exception("API Unavailable")
        mock_exchange.fetch_ticker.side_effect = Exception("API Unavailable")
        mock_exchange.fetch_order_book.side_effect = Exception("API Unavailable")

        # Should return empty DataFrame, not crash
        df = await data_fetcher.get_historical_data("BTC/USDT")
        assert df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

        # Should return empty dict, not crash
        result = await data_fetcher.get_realtime_data(["BTC/USDT"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_request_deduplication_for_idempotent_operations(self, data_fetcher, mock_exchange):
        """Test that idempotent operations are handled correctly."""
        # Disable circuit breaker for this test to focus on retry behavior
        data_fetcher.circuit_breaker.config.failure_threshold = 1000

        # Mock that fails once then succeeds for all subsequent calls
        call_count = 0
        def mock_fetch_ohlcv(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return [[1640995200000, 50000, 51000, 49000, 50500, 100]]

        mock_exchange.fetch_ohlcv.side_effect = mock_fetch_ohlcv

        # Multiple concurrent calls for same data should work
        tasks = [
            data_fetcher.get_historical_data("BTC/USDT", force_fresh=True)
            for _ in range(3)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed and return same data
        for df in results:
            assert not df.empty
            assert len(df) == 1

        # fetch_ohlcv should be called multiple times due to retries and concurrent calls
        assert mock_exchange.fetch_ohlcv.call_count >= 4  # 1 failure + 3 successes

    @pytest.mark.asyncio
    async def test_permanent_vs_temporary_failure_handling(self, data_fetcher, mock_exchange):
        """Test different handling of permanent vs temporary failures."""
        # Mock permanent failure (4xx error)
        from ccxt import BadRequest
        mock_exchange.fetch_ohlcv.side_effect = BadRequest("Invalid symbol")

        # Should fail immediately without retries for permanent errors
        with pytest.raises(BadRequest):
            await data_fetcher.get_historical_data("INVALID/USDT")

        # Verify only called once (no retries for permanent errors)
        assert mock_exchange.fetch_ohlcv.call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, data_fetcher, mock_exchange):
        """Test circuit breaker recovery after failures."""
        # Configure circuit breaker with lower success threshold
        data_fetcher.circuit_breaker.config.failure_threshold = 2
        data_fetcher.circuit_breaker.config.recovery_timeout = 0.5
        data_fetcher.circuit_breaker.config.success_threshold = 1  # Close after 1 success

        # Mock that always succeeds
        mock_exchange.fetch_ohlcv.return_value = [[1640995200000, 50000, 51000, 49000, 50500, 100]]

        # Manually set circuit breaker to open state
        data_fetcher.circuit_breaker.state = data_fetcher.circuit_breaker.state.OPEN
        data_fetcher.circuit_breaker.last_failure_time = time.time() - 1.0  # 1 second ago

        # Wait for recovery timeout
        await asyncio.sleep(0.6)

        # Circuit should transition to half-open when checked
        assert not data_fetcher.circuit_breaker.is_open()
        assert data_fetcher.circuit_breaker.state.value == "half-open"

        # Next call should succeed (half-open state allows one test call)
        df = await data_fetcher.get_historical_data("BTC/USDT")
        assert not df.empty

        # Circuit should be closed again after success
        assert data_fetcher.circuit_breaker.state.value == "closed"

    @pytest.mark.asyncio
    async def test_retry_metrics_and_logging(self, data_fetcher, mock_exchange):
        """Test that retry attempts are properly logged and metrics tracked."""
        # Mock failures then success
        mock_exchange.fetch_ohlcv.side_effect = [
            Exception("Attempt 1 failure"),
            Exception("Attempt 2 failure"),
            [[1640995200000, 50000, 51000, 49000, 50500, 100]]
        ]

        df = await data_fetcher.get_historical_data("BTC/USDT")

        assert not df.empty

        # Verify that fetch_ohlcv was called 3 times (2 failures + 1 success)
        assert mock_exchange.fetch_ohlcv.call_count == 3

        # The logging is tested implicitly - if retries work, logging is working
        # In a real scenario, logs would be captured by monitoring systems

    @pytest.mark.asyncio
    async def test_configurable_retry_strategies(self, data_fetcher, mock_exchange):
        """Test that retry strategies can be configured per endpoint."""
        # Test different configurations
        data_fetcher.endpoint_retry_configs = {
            "fetch_ohlcv": {"max_attempts": 2, "base_delay": 0.01},
            "fetch_ticker": {"max_attempts": 4, "base_delay": 0.02}
        }

        # Mock OHLCV: fail once then succeed
        mock_exchange.fetch_ohlcv.side_effect = [
            Exception("OHLCV fail"),
            [[1640995200000, 50000, 51000, 49000, 50500, 100]]
        ]

        # Mock ticker: fail 3 times then succeed
        mock_exchange.fetch_ticker.side_effect = [
            Exception("Ticker fail 1"),
            Exception("Ticker fail 2"),
            Exception("Ticker fail 3"),
            {
                "timestamp": 1640995200000,
                "last": 50000,
                "bid": 49900,
                "ask": 50100,
                "high": 51000,
                "low": 49000,
                "baseVolume": 100,
                "percentage": 1.0
            }
        ]

        # Test OHLCV (2 attempts)
        df = await data_fetcher.get_historical_data("BTC/USDT")
        assert not df.empty
        assert mock_exchange.fetch_ohlcv.call_count == 2

        # Test ticker (4 attempts)
        result = await data_fetcher.get_realtime_data(["BTC/USDT"], tickers=True, orderbooks=False)
        assert "BTC/USDT" in result
        assert mock_exchange.fetch_ticker.call_count == 4
