"""
Tests for structured logging and resource cleanup functionality.

This module tests the enhanced logging and resource management features
implemented in the core modules.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import logging
import json

from core.logging_utils import get_structured_logger, LogSensitivity, StructuredLogger
from core.cache import RedisCache
from core.metrics_endpoint import MetricsEndpoint
from core.interfaces import CacheConfig, MemoryConfig, EvictionPolicy


class TestStructuredLogging:
    """Test structured logging functionality."""

    def test_logger_initialization(self):
        """Test that structured logger initializes correctly."""
        logger = get_structured_logger("test_component", LogSensitivity.INFO)
        assert isinstance(logger, StructuredLogger)
        assert logger.sensitivity == LogSensitivity.INFO

    def test_logger_sensitivity_levels(self):
        """Test different sensitivity levels."""
        logger = get_structured_logger("test_component", LogSensitivity.DEBUG)

        # Test sensitivity change
        logger.set_sensitivity(LogSensitivity.SECURE)
        assert logger.sensitivity == LogSensitivity.SECURE

    def test_structured_message_formatting(self):
        """Test structured message formatting with metadata."""
        logger = StructuredLogger("test", LogSensitivity.INFO)

        # Test formatting with kwargs
        message = logger._format_structured_message(
            "Test message",
            component="test",
            operation="test_op",
            user_id="123"
        )

        assert "Test message" in message
        assert "component=test" in message
        assert "operation=test_op" in message
        assert "user_id=123" in message

    @patch('core.logging_utils.logging')
    def test_log_levels(self, mock_logging):
        """Test different log levels work correctly."""
        logger = StructuredLogger("test", LogSensitivity.INFO)
        mock_logger = Mock()
        logger.logger = mock_logger

        # Test info level
        logger.info("Test info", component="test")
        mock_logger.info.assert_called_once()

        # Test error level
        logger.error("Test error", component="test")
        mock_logger.error.assert_called_once()

        # Test debug level
        logger.debug("Test debug", component="test")
        mock_logger.debug.assert_called_once()


class TestCacheResourceManagement:
    """Test cache resource management with context managers."""

    @pytest.fixture
    def cache_config(self):
        """Create a test cache configuration."""
        return CacheConfig(
            host="localhost",
            port=6379,
            db=0,
            password=None,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            ttl_config={
                "default": 300,
                "market_ticker": 60,
                "ohlcv": 300,
                "account_balance": 30
            },
            max_cache_size=10000,
            eviction_policy=EvictionPolicy.LRU,
            memory_config=MemoryConfig(
                max_memory_mb=500.0,
                warning_memory_mb=400.0,
                cleanup_memory_mb=350.0,
                eviction_batch_size=100,
                memory_check_interval=60.0
            )
        )

    @patch('core.cache.redis')
    async def test_cache_context_manager(self, mock_redis, cache_config):
        """Test cache context manager functionality."""
        # Mock Redis client
        mock_client = AsyncMock()
        mock_redis.Redis.return_value = mock_client
        mock_client.ping.return_value = True

        cache = RedisCache(cache_config)

        # Test context manager
        async with cache:
            # Should have initialized
            assert cache._connected is True
            mock_client.ping.assert_called_once()

        # Should have closed
        assert cache._connected is False
        mock_client.close.assert_called_once()

    @patch('core.cache.redis')
    async def test_cache_initialization_failure(self, mock_redis, cache_config):
        """Test cache initialization failure handling."""
        # Mock Redis client to fail
        mock_client = AsyncMock()
        mock_redis.Redis.return_value = mock_client
        mock_client.ping.side_effect = Exception("Connection failed")

        cache = RedisCache(cache_config)

        # Should handle initialization failure gracefully
        success = await cache.initialize()
        assert success is False
        assert cache._connected is False

    @patch('core.cache.redis')
    async def test_cache_close_error_handling(self, mock_redis, cache_config):
        """Test cache close error handling."""
        # Mock Redis client
        mock_client = AsyncMock()
        mock_redis.Redis.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.close.side_effect = Exception("Close failed")

        cache = RedisCache(cache_config)
        await cache.initialize()

        # Should handle close error gracefully
        await cache.close()
        assert cache._connected is False


class TestMetricsEndpointResourceManagement:
    """Test metrics endpoint resource management."""

    @pytest.fixture
    def endpoint_config(self):
        """Create a test endpoint configuration."""
        return {
            "host": "localhost",
            "port": 9090,
            "path": "/metrics",
            "enable_auth": False,
            "enable_tls": False
        }

    async def test_endpoint_context_manager(self, endpoint_config):
        """Test endpoint context manager functionality."""
        endpoint = MetricsEndpoint(endpoint_config)

        # Test context manager
        async with endpoint:
            assert endpoint._running is True

        # Should have stopped
        assert endpoint._running is False

    async def test_endpoint_initialization(self, endpoint_config):
        """Test endpoint initialization."""
        endpoint = MetricsEndpoint(endpoint_config)

        assert endpoint.host == "localhost"
        assert endpoint.port == 9090
        assert endpoint.path == "/metrics"
        assert endpoint.enable_auth is False
        assert endpoint.enable_tls is False

    async def test_endpoint_stats(self, endpoint_config):
        """Test endpoint statistics."""
        endpoint = MetricsEndpoint(endpoint_config)

        stats = endpoint.get_stats()
        assert "running" in stats
        assert "host" in stats
        assert "port" in stats
        assert "request_count" in stats
        assert "error_count" in stats
        assert "avg_response_time" in stats


class TestResourceCleanupIntegration:
    """Test integration of resource cleanup across components."""

    async def test_multiple_context_managers(self):
        """Test using multiple context managers together."""
        # Create mock components
        cache_config = CacheConfig(
            host="localhost",
            port=6379,
            db=0,
            password=None,
            socket_timeout=5.0,
            socket_connect_timeout=5.0
        )

        endpoint_config = {
            "host": "localhost",
            "port": 9090,
            "enable_auth": False,
            "enable_tls": False
        }

        # This would be the pattern for using both together
        cache = RedisCache(cache_config)
        endpoint = MetricsEndpoint(endpoint_config)

        # Test that both can be created without errors
        assert cache is not None
        assert endpoint is not None

        # Note: Full integration test would require mocking Redis and aiohttp
        # but this tests the basic setup

    @patch('core.cache.redis')
    async def test_cache_context_manager_with_cleanup(self, mock_redis):
        """Test cache context manager with proper cleanup."""
        from core.cache import CacheContext

        # Mock Redis client
        mock_client = AsyncMock()
        mock_redis.Redis.return_value = mock_client
        mock_client.ping.return_value = True

        cache_config = CacheConfig(
            host="localhost",
            port=6379,
            db=0,
            password=None,
            socket_timeout=5.0,
            socket_connect_timeout=5.0
        )

        # Test context manager
        async with CacheContext(cache_config) as cache:
            assert cache is not None
            assert cache._connected is True

        # Verify cleanup was called
        mock_client.close.assert_called_once()

    async def test_endpoint_context_manager_with_cleanup(self):
        """Test endpoint context manager with proper cleanup."""
        endpoint_config = {
            "host": "localhost",
            "port": 9090,
            "enable_auth": False,
            "enable_tls": False
        }

        endpoint = MetricsEndpoint(endpoint_config)

        # Test context manager
        async with endpoint:
            assert endpoint._running is True

        # Should have stopped
        assert endpoint._running is False

    @patch('core.cache.redis')
    async def test_cache_global_instance_cleanup(self, mock_redis):
        """Test global cache instance cleanup on exit."""
        from core.cache import initialize_cache, close_cache, _cleanup_cache_on_exit

        # Mock Redis client
        mock_client = AsyncMock()
        mock_redis.Redis.return_value = mock_client
        mock_client.ping.return_value = True

        cache_config = {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "socket_timeout": 5.0,
            "socket_connect_timeout": 5.0
        }

        # Initialize cache
        success = initialize_cache(cache_config)
        assert success is True

        # Test cleanup function
        _cleanup_cache_on_exit()

        # Verify cleanup was called
        mock_client.close.assert_called_once()

    async def test_endpoint_global_instance_cleanup(self):
        """Test global endpoint instance cleanup on exit."""
        from core.metrics_endpoint import create_metrics_endpoint, _cleanup_endpoint_on_exit

        endpoint_config = {
            "host": "localhost",
            "port": 9090,
            "enable_auth": False,
            "enable_tls": False
        }

        # Create endpoint
        endpoint = create_metrics_endpoint(endpoint_config)
        assert endpoint is not None

        # Test cleanup function (this will try to stop the endpoint)
        await _cleanup_endpoint_on_exit()

        # Note: In a real scenario, this would stop the server
        # Here we just verify the function doesn't crash

    async def test_cleanup_on_startup_failure(self):
        """Test cleanup when startup fails."""
        # Test with invalid SSL configuration to trigger cleanup
        endpoint_config = {
            "host": "localhost",
            "port": 9090,
            "enable_auth": False,
            "enable_tls": True,
            "cert_file": "/nonexistent/cert.pem",
            "key_file": "/nonexistent/key.pem"
        }

        endpoint = MetricsEndpoint(endpoint_config)

        # Attempt to start (should fail due to missing cert files)
        with pytest.raises(FileNotFoundError):
            await endpoint.start()

        # Verify cleanup was performed
        assert endpoint._running is False
        assert endpoint.site is None
        assert endpoint.runner is None
        assert endpoint.app is None


if __name__ == "__main__":
    # Run basic tests
    import sys

    async def run_async_tests():
        """Run async tests."""
        print("Running async tests...")

        # Test structured logging
        logger = get_structured_logger("test", LogSensitivity.INFO)
        print("✓ Structured logger created")

        # Test cache context manager (with mocking)
        cache_config = CacheConfig(
            host="localhost",
            port=6379,
            db=0,
            password=None,
            socket_timeout=5.0,
            socket_connect_timeout=5.0
        )

        with patch('core.cache.redis') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.Redis.return_value = mock_client
            mock_client.ping.return_value = True

            cache = RedisCache(cache_config)

            async with cache:
                print("✓ Cache context manager entered")

            print("✓ Cache context manager exited")

        # Test endpoint context manager
        endpoint_config = {
            "host": "localhost",
            "port": 9090,
            "enable_auth": False,
            "enable_tls": False
        }

        endpoint = MetricsEndpoint(endpoint_config)

        async with endpoint:
            print("✓ Endpoint context manager entered")

        print("✓ Endpoint context manager exited")
        print("All basic tests passed!")

    # Run async tests
    asyncio.run(run_async_tests())
