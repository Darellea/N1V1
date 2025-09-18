"""
Comprehensive tests for Redis cache implementation.
Tests basic operations, specialized methods, batch operations,
cache invalidation, global functions, context manager, and error handling.
"""

import asyncio
import pytest
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from core.cache import (
    RedisCache,
    CacheConfig,
    EvictionPolicy,
    initialize_cache,
    get_cache,
    close_cache,
    CacheContext
)


class TestCacheBasicOperations:
    """Test basic cache operations (get, set, delete)."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.setex.return_value = True
        mock_client.delete.return_value = 1
        mock_client.ttl.return_value = 30
        return mock_client

    @pytest.fixture
    def cache_config(self):
        """Cache configuration for testing."""
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
                "ohlcv": 600,
                "account_balance": 30
            },
            eviction_policy=EvictionPolicy.TTL,
            max_cache_size=10000,
            memory_config={
                "max_memory_mb": 500.0,
                "warning_memory_mb": 400.0,
                "cleanup_memory_mb": 350.0,
                "eviction_batch_size": 100,
                "memory_check_interval": 60.0
            }
        )

    @pytest.fixture
    def cache_instance(self, cache_config, mock_redis):
        """Cache instance with mocked Redis client."""
        cache = RedisCache(cache_config)
        cache._redis_client = mock_redis
        cache._connected = True
        return cache

    @pytest.mark.asyncio
    async def test_initialization_success(self, cache_config, mock_redis):
        """Test successful cache initialization."""
        with patch('redis.asyncio.Redis', return_value=mock_redis):
            cache = RedisCache(cache_config)
            success = await cache.initialize()

            assert success == True
            assert cache._connected == True
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_failure(self, cache_config):
        """Test cache initialization failure."""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.ping.side_effect = Exception("Connection failed")
            mock_redis_class.return_value = mock_redis_instance

            cache = RedisCache(cache_config)
            success = await cache.initialize()

            assert success == False
            assert cache._connected == False

    @pytest.mark.asyncio
    async def test_get_existing_key(self, cache_instance, mock_redis):
        """Test getting an existing key."""
        test_data = {"symbol": "BTC/USDT", "price": 50000}
        mock_redis.get.return_value = json.dumps(test_data).encode('utf-8')

        result = await cache_instance.get("test:key")

        assert result == test_data
        mock_redis.get.assert_called_once_with("test:key")

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, cache_instance, mock_redis):
        """Test getting a nonexistent key."""
        mock_redis.get.return_value = None

        result = await cache_instance.get("nonexistent:key")

        assert result is None
        mock_redis.get.assert_called_once_with("nonexistent:key")

    @pytest.mark.asyncio
    async def test_get_with_invalid_json(self, cache_instance, mock_redis):
        """Test getting a key with invalid JSON."""
        mock_redis.get.return_value = b"invalid json"

        result = await cache_instance.get("test:key")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_success(self, cache_instance, mock_redis):
        """Test successful set operation."""
        test_data = {"symbol": "BTC/USDT", "price": 50000}

        result = await cache_instance.set("test:key", test_data, ttl=300)

        assert result == True
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "test:key"
        assert call_args[0][1] == 300
        # Verify JSON serialization
        deserialized = json.loads(call_args[0][2].decode('utf-8'))
        assert deserialized == test_data

    @pytest.mark.asyncio
    async def test_set_without_ttl(self, cache_instance, mock_redis):
        """Test set operation without explicit TTL."""
        test_data = {"symbol": "BTC/USDT", "price": 50000}

        result = await cache_instance.set("test:key", test_data)

        assert result == True
        # Should use default TTL
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 300  # default TTL

    @pytest.mark.asyncio
    async def test_set_failure(self, cache_instance, mock_redis):
        """Test set operation failure."""
        mock_redis.setex.side_effect = Exception("Redis error")

        result = await cache_instance.set("test:key", {"data": "test"})

        assert result == False

    @pytest.mark.asyncio
    async def test_delete_success(self, cache_instance, mock_redis):
        """Test successful delete operation."""
        mock_redis.delete.return_value = 1

        result = await cache_instance.delete("test:key")

        assert result == True
        mock_redis.delete.assert_called_once_with("test:key")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, cache_instance, mock_redis):
        """Test deleting a nonexistent key."""
        mock_redis.delete.return_value = 0

        result = await cache_instance.delete("nonexistent:key")

        assert result == False

    @pytest.mark.asyncio
    async def test_delete_failure(self, cache_instance, mock_redis):
        """Test delete operation failure."""
        mock_redis.delete.side_effect = Exception("Redis error")

        result = await cache_instance.delete("test:key")

        assert result == False

    def test_cache_key_generation(self, cache_instance):
        """Test cache key generation for different data types."""
        # Test OHLCV key
        key = cache_instance._get_cache_key("ohlcv", "BTC/USDT", timeframe="1h")
        assert key == "ohlcv:BTC/USDT:1h"

        # Test market ticker key
        key = cache_instance._get_cache_key("market_ticker", "ETH/USDT")
        assert key == "market_ticker:ETH/USDT"

        # Test order book key
        key = cache_instance._get_cache_key("order_book", "ADA/USDT", limit=100)
        assert key == "order_book:ADA/USDT:100"

        # Test simple key
        key = cache_instance._get_cache_key("account_balance", "user123")
        assert key == "account_balance:user123"

    def test_ttl_retrieval(self, cache_instance):
        """Test TTL retrieval for different data types."""
        # Test known data type
        ttl = cache_instance._get_ttl("market_ticker")
        assert ttl == 60

        # Test unknown data type (should use default)
        ttl = cache_instance._get_ttl("unknown_type")
        assert ttl == 300


class TestCacheSpecializedMethods:
    """Test specialized cache methods for market data."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.setex.return_value = True
        return mock_client

    @pytest.fixture
    def cache_config(self):
        """Cache configuration."""
        return CacheConfig(ttl_config={"default": 300, "market_ticker": 60, "ohlcv": 600, "account_balance": 30})

    @pytest.fixture
    def cache_instance(self, cache_config, mock_redis):
        """Cache instance with mocked Redis."""
        cache = RedisCache(cache_config)
        cache._redis_client = mock_redis
        cache._connected = True
        return cache

    @pytest.mark.asyncio
    async def test_get_market_ticker_cached(self, cache_instance, mock_redis):
        """Test getting cached market ticker."""
        ticker_data = {"symbol": "BTC/USDT", "price": 50000, "volume": 100}
        mock_redis.get.return_value = json.dumps(ticker_data).encode('utf-8')

        result = await cache_instance.get_market_ticker("BTC/USDT")

        assert result == ticker_data
        mock_redis.get.assert_called_once_with("market_ticker:BTC/USDT")

    @pytest.mark.asyncio
    async def test_get_market_ticker_not_cached(self, cache_instance, mock_redis):
        """Test getting market ticker when not cached."""
        mock_redis.get.return_value = None

        result = await cache_instance.get_market_ticker("BTC/USDT")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_market_ticker(self, cache_instance, mock_redis):
        """Test setting market ticker in cache."""
        ticker_data = {"symbol": "BTC/USDT", "price": 50000, "volume": 100}

        result = await cache_instance.set_market_ticker("BTC/USDT", ticker_data)

        assert result == True
        # Verify correct key and TTL
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "market_ticker:BTC/USDT"
        assert call_args[0][1] == 60  # market_ticker TTL

    @pytest.mark.asyncio
    async def test_get_ohlcv_cached(self, cache_instance, mock_redis):
        """Test getting cached OHLCV data."""
        ohlcv_data = [[1640995200000, 50000, 51000, 49000, 50500, 100]]
        mock_redis.get.return_value = json.dumps(ohlcv_data).encode('utf-8')

        result = await cache_instance.get_ohlcv("BTC/USDT", "1h")

        assert result == ohlcv_data
        mock_redis.get.assert_called_once_with("ohlcv:BTC/USDT:1h")

    @pytest.mark.asyncio
    async def test_set_ohlcv(self, cache_instance, mock_redis):
        """Test setting OHLCV data in cache."""
        ohlcv_data = [[1640995200000, 50000, 51000, 49000, 50500, 100]]

        result = await cache_instance.set_ohlcv("BTC/USDT", "1h", ohlcv_data)

        assert result == True
        # Verify correct key and TTL
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "ohlcv:BTC/USDT:1h"
        assert call_args[0][1] == 600  # ohlcv TTL

    @pytest.mark.asyncio
    async def test_get_account_balance_cached(self, cache_instance, mock_redis):
        """Test getting cached account balance."""
        balance_data = {"USDT": 10000.0, "BTC": 0.5}
        mock_redis.get.return_value = json.dumps(balance_data).encode('utf-8')

        result = await cache_instance.get_account_balance("user123")

        assert result == balance_data
        mock_redis.get.assert_called_once_with("account_balance:user123")

    @pytest.mark.asyncio
    async def test_set_account_balance(self, cache_instance, mock_redis):
        """Test setting account balance in cache."""
        balance_data = {"USDT": 10000.0, "BTC": 0.5}

        result = await cache_instance.set_account_balance("user123", balance_data)

        assert result == True
        # Verify correct key and TTL
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "account_balance:user123"
        assert call_args[0][1] == 30  # account_balance TTL


class TestCacheBatchOperations:
    """Test batch cache operations."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.mget.return_value = [b'{"price": 50000}', None, b'{"price": 3000}']
        mock_client.pipeline.return_value = mock_client
        mock_client.setex.return_value = True
        mock_client.execute.return_value = None
        return mock_client

    @pytest.fixture
    def cache_config(self):
        """Cache configuration."""
        return CacheConfig(ttl_config={"default": 300, "ohlcv": 600})

    @pytest.fixture
    def cache_instance(self, cache_config, mock_redis):
        """Cache instance with mocked Redis."""
        cache = RedisCache(cache_config)
        cache._redis_client = mock_redis
        cache._connected = True
        return cache

    @pytest.mark.asyncio
    async def test_get_multiple_ohlcv(self, cache_instance, mock_redis):
        """Test getting multiple OHLCV data."""
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        timeframe = "1h"

        result = await cache_instance.get_multiple_ohlcv(symbols, timeframe)

        assert len(result) == 3
        assert result["BTC/USDT"] == {"price": 50000}
        assert result["ETH/USDT"] is None
        assert result["ADA/USDT"] == {"price": 3000}

        # Verify mget was called with correct keys
        mock_redis.mget.assert_called_once()
        call_args = mock_redis.mget.call_args[0][0]
        expected_keys = ["ohlcv:BTC/USDT:1h", "ohlcv:ETH/USDT:1h", "ohlcv:ADA/USDT:1h"]
        assert call_args == expected_keys

    @pytest.mark.asyncio
    async def test_set_multiple_ohlcv_success(self, cache_instance, mock_redis):
        """Test setting multiple OHLCV data successfully."""
        data = {
            "BTC/USDT": [[1640995200000, 50000, 51000, 49000, 50500, 100]],
            "ETH/USDT": [[1640995200000, 3000, 3100, 2900, 3050, 200]]
        }

        result = await cache_instance.set_multiple_ohlcv(data, "1h")

        assert result["BTC/USDT"] == True
        assert result["ETH/USDT"] == True

        # Verify pipeline operations
        assert mock_redis.pipeline.called
        assert mock_redis.setex.call_count == 2

    @pytest.mark.asyncio
    async def test_set_multiple_ohlcv_failure(self, cache_instance, mock_redis):
        """Test setting multiple OHLCV data with failure."""
        mock_redis.execute.side_effect = Exception("Pipeline failed")

        data = {"BTC/USDT": [[1640995200000, 50000, 51000, 49000, 50500, 100]]}

        result = await cache_instance.set_multiple_ohlcv(data, "1h")

        assert result["BTC/USDT"] == False

    @pytest.mark.asyncio
    async def test_get_multiple_ohlcv_not_connected(self, cache_config):
        """Test getting multiple OHLCV when not connected."""
        cache = RedisCache(cache_config)
        cache._connected = False

        result = await cache.get_multiple_ohlcv(["BTC/USDT"], "1h")

        # Should return None for all symbols
        assert result["BTC/USDT"] is None


class TestCacheInvalidation:
    """Test cache invalidation operations."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.keys.return_value = ["ohlcv:BTC/USDT:1h", "market_ticker:BTC/USDT", "ohlcv:ETH/USDT:1h"]
        mock_client.delete.return_value = 3
        return mock_client

    @pytest.fixture
    def cache_config(self):
        """Cache configuration."""
        return CacheConfig()

    @pytest.fixture
    def cache_instance(self, cache_config, mock_redis):
        """Cache instance with mocked Redis."""
        cache = RedisCache(cache_config)
        cache._redis_client = mock_redis
        cache._connected = True
        return cache

    @pytest.mark.asyncio
    async def test_invalidate_symbol_data_all_types(self, cache_instance, mock_redis):
        """Test invalidating all data for a symbol."""
        result = await cache_instance.invalidate_symbol_data("BTC/USDT")

        assert result == 3  # Three keys deleted (including ETH key)
        mock_redis.keys.assert_called_once_with("*:BTC/USDT:*")
        # Verify delete was called with all matching keys
        mock_redis.delete.assert_called_once()
        delete_args = mock_redis.delete.call_args[0]
        assert len(delete_args) == 3

    @pytest.mark.asyncio
    async def test_invalidate_symbol_data_specific_types(self, cache_instance, mock_redis):
        """Test invalidating specific data types for a symbol."""
        # Mock keys to return only BTC/USDT keys
        mock_redis.keys.return_value = ["ohlcv:BTC/USDT:1h", "market_ticker:BTC/USDT"]

        result = await cache_instance.invalidate_symbol_data("BTC/USDT", ["ohlcv"])

        assert result == 1  # One key deleted
        mock_redis.keys.assert_called_once_with("*:BTC/USDT:*")
        mock_redis.delete.assert_called_once_with("ohlcv:BTC/USDT:1h")

    @pytest.mark.asyncio
    async def test_invalidate_symbol_data_no_matches(self, cache_instance, mock_redis):
        """Test invalidating data when no keys match."""
        mock_redis.keys.return_value = []

        result = await cache_instance.invalidate_symbol_data("UNKNOWN/USDT")

        assert result == 0
        mock_redis.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_all_success(self, cache_instance, mock_redis):
        """Test clearing all cache data."""
        mock_redis.flushdb.return_value = True

        result = await cache_instance.clear_all()

        assert result == True
        mock_redis.flushdb.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_all_failure(self, cache_instance, mock_redis):
        """Test clearing all cache data with failure."""
        mock_redis.flushdb.side_effect = Exception("Flush failed")

        result = await cache_instance.clear_all()

        assert result == False


class TestCacheGlobalFunctions:
    """Test global cache functions."""

    def test_initialize_cache_success(self):
        """Test successful cache initialization."""
        config = {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "eviction_policy": "lru",
            "max_cache_size": 5000
        }

        with patch('redis.asyncio.Redis') as mock_redis_class, \
             patch('asyncio.run') as mock_asyncio_run:

            mock_redis_instance = AsyncMock()
            mock_redis_instance.ping.return_value = True
            mock_redis_class.return_value = mock_redis_instance
            mock_asyncio_run.return_value = True

            success = initialize_cache(config)

            assert success == True

            # Verify cache instance was created
            cache = get_cache()
            assert cache is not None
            assert cache.config.eviction_policy == EvictionPolicy.LRU
            assert cache.config.max_cache_size == 5000

            # Clean up
            close_cache()

    def test_initialize_cache_already_exists(self):
        """Test cache initialization when already initialized."""
        config = {"host": "localhost", "port": 6379}

        with patch('redis.asyncio.Redis') as mock_redis_class, \
             patch('asyncio.run') as mock_asyncio_run:

            mock_redis_instance = AsyncMock()
            mock_redis_instance.ping.return_value = True
            mock_redis_class.return_value = mock_redis_instance
            mock_asyncio_run.return_value = True

            # First initialization
            success1 = initialize_cache(config)
            assert success1 == True

            # Second initialization should return True but not re-initialize
            success2 = initialize_cache(config)
            assert success2 == True

            # Clean up
            close_cache()

    def test_get_cache_none(self):
        """Test getting cache when not initialized."""
        # Ensure no cache is initialized
        close_cache()

        cache = get_cache()
        assert cache is None

    def test_close_cache_success(self):
        """Test successful cache closing."""
        config = {"host": "localhost", "port": 6379}

        with patch('redis.asyncio.Redis') as mock_redis_class, \
             patch('asyncio.run') as mock_asyncio_run, \
             patch('asyncio.run', side_effect=[True, None]) as mock_asyncio_run_close:

            mock_redis_instance = AsyncMock()
            mock_redis_instance.ping.return_value = True
            mock_redis_instance.close.return_value = None
            mock_redis_class.return_value = mock_redis_instance

            # Initialize
            initialize_cache(config)

            # Close
            close_cache()

            # Verify cache is None after closing
            cache = get_cache()
            assert cache is None


class TestCacheContextManager:
    """Test cache context manager."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        return mock_client

    @pytest.fixture
    def cache_config(self):
        """Cache configuration."""
        return {"host": "localhost", "port": 6379}

    @pytest.mark.asyncio
    async def test_context_manager_success(self, cache_config, mock_redis):
        """Test context manager with successful initialization."""
        with patch('redis.asyncio.Redis', return_value=mock_redis), \
             patch('core.cache.initialize_cache', return_value=True) as mock_init, \
             patch('core.cache.close_cache') as mock_close, \
             patch('core.cache.get_cache', return_value=None):

            async with CacheContext(cache_config) as cache:
                assert cache is not None

            mock_init.assert_called_once_with(cache_config)
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_initialization_failure(self, cache_config):
        """Test context manager when initialization fails."""
        with patch('core.cache.initialize_cache', return_value=False):
            with pytest.raises(RuntimeError, match="Failed to initialize cache"):
                async with CacheContext(cache_config) as cache:
                    pass

    @pytest.mark.asyncio
    async def test_context_manager_no_config(self):
        """Test context manager without config when cache not initialized."""
        with patch('core.cache.get_cache', return_value=None):
            with pytest.raises(RuntimeError, match="Cache not initialized"):
                async with CacheContext() as cache:
                    pass


class TestCacheErrorHandling:
    """Test cache error handling and edge cases."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        return mock_client

    @pytest.fixture
    def cache_config(self):
        """Cache configuration."""
        return CacheConfig()

    @pytest.fixture
    def cache_instance(self, cache_config, mock_redis):
        """Cache instance with mocked Redis."""
        cache = RedisCache(cache_config)
        cache._redis_client = mock_redis
        cache._connected = True
        return cache

    @pytest.mark.asyncio
    async def test_operations_when_not_connected(self, cache_config):
        """Test operations when cache is not connected."""
        cache = RedisCache(cache_config)
        cache._connected = False
        cache._redis_client = None

        # All operations should return appropriate failure values
        assert await cache.get("test") is None
        assert await cache.set("test", "value") == False
        assert await cache.delete("test") == False
        assert await cache.get_market_ticker("BTC/USDT") is None
        assert await cache.set_market_ticker("BTC/USDT", {}) == False
        assert await cache.clear_all() == False

    @pytest.mark.asyncio
    async def test_get_with_corrupted_data(self, cache_instance, mock_redis):
        """Test getting data with corrupted JSON."""
        mock_redis.get.return_value = b"corrupted json data {"

        result = await cache_instance.get("test:key")

        assert result is None

    @pytest.mark.asyncio
    async def test_batch_operations_with_partial_failures(self, cache_instance, mock_redis):
        """Test batch operations with partial failures."""
        # Mock mget to return mix of valid and invalid data
        mock_redis.mget.return_value = [
            json.dumps({"price": 50000}).encode('utf-8'),  # Valid
            b"invalid json",  # Invalid
            None  # Not found
        ]

        result = await cache_instance.get_multiple_ohlcv(["BTC/USDT", "ETH/USDT", "ADA/USDT"], "1h")

        assert "BTC/USDT" in result
        assert "ETH/USDT" in result
        assert "ADA/USDT" in result
        assert result["BTC/USDT"] == {"price": 50000}
        assert result["ETH/USDT"] is None  # Invalid JSON
        assert result["ADA/USDT"] is None  # Not found

    @pytest.mark.asyncio
    async def test_redis_connection_errors(self, cache_instance, mock_redis):
        """Test handling of Redis connection errors."""
        mock_redis.get.side_effect = Exception("Connection lost")

        result = await cache_instance.get("test:key")

        assert result is None

    @pytest.mark.asyncio
    async def test_invalidation_with_redis_errors(self, cache_instance, mock_redis):
        """Test cache invalidation with Redis errors."""
        mock_redis.keys.side_effect = Exception("Redis error")

        result = await cache_instance.invalidate_symbol_data("BTC/USDT")

        assert result == 0

    def test_config_validation(self):
        """Test configuration validation."""
        # Test with missing required fields
        invalid_config = CacheConfig(host="")  # Empty host

        # Should still create config, but with defaults
        assert invalid_config.host == ""
        assert invalid_config.port == 6379  # Default port

    @pytest.mark.asyncio
    async def test_async_context_manager(self, cache_instance):
        """Test async context manager methods."""
        # Test __aenter__
        await cache_instance.__aenter__()
        assert cache_instance._redis_client is not None

        # Test __aexit__
        await cache_instance.__aexit__(None, None, None)
        # Should not raise any exceptions


class TestCacheMemoryMonitoring:
    """Test memory monitoring functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.dbsize.return_value = 100
        mock_client.info.return_value = {"used_memory": 100000000, "used_memory_peak": 150000000}
        return mock_client

    @pytest.fixture
    def cache_config(self):
        """Cache configuration."""
        return CacheConfig()

    @pytest.fixture
    def cache_instance(self, cache_config, mock_redis):
        """Cache instance with mocked Redis."""
        cache = RedisCache(cache_config)
        cache._redis_client = mock_redis
        cache._connected = True
        return cache

    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, cache_instance):
        """Test memory usage monitoring."""
        with patch('psutil.Process') as mock_process:
            mock_memory_info = Mock()
            mock_memory_info.rss = 300 * 1024 * 1024  # 300MB
            mock_process.return_value.memory_info.return_value = mock_memory_info

            memory_mb = cache_instance._get_memory_usage()

            assert memory_mb == 300.0

    @pytest.mark.asyncio
    async def test_memory_thresholds(self, cache_instance):
        """Test memory threshold checking."""
        with patch.object(cache_instance, '_get_memory_usage') as mock_memory:
            # Normal memory
            mock_memory.return_value = 200.0
            status = cache_instance._check_memory_thresholds()
            assert not any(status.values())

            # High memory
            mock_memory.return_value = 600.0
            status = cache_instance._check_memory_thresholds()
            assert all(status.values())

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, cache_instance, mock_redis):
        """Test cache statistics retrieval."""
        with patch.object(cache_instance, '_get_memory_usage') as mock_memory:
            mock_memory.return_value = 250.0

            stats = await cache_instance.get_cache_stats()

            assert stats["cache_size"] == 100
            assert stats["memory_mb"] == 250.0
            assert stats["redis_memory_used"] == 95.367431640625  # 100MB in MB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
