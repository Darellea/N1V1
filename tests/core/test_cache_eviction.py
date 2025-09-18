"""
Unit tests for cache eviction policies and memory monitoring.
Tests LRU, TTL, and LFU eviction strategies, memory monitoring,
and thread safety.
"""

import asyncio
import pytest
import time
import threading
from unittest.mock import Mock, patch, AsyncMock
import psutil

from core.cache import (
    RedisCache,
    CacheConfig,
    EvictionPolicy,
    MemoryConfig,
    initialize_cache,
    get_cache,
    close_cache
)


class TestCacheEviction:
    """Test cache eviction policies and memory monitoring."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.setex.return_value = True
        mock_client.delete.return_value = 1
        mock_client.ttl.return_value = 30
        mock_client.keys.return_value = ["test:key1", "test:key2"]
        mock_client.dbsize.return_value = 100
        mock_client.object.return_value = [10, 20]  # idle times or frequencies
        mock_client.info.return_value = {"used_memory": 100000000, "used_memory_peak": 150000000}
        return mock_client

    @pytest.fixture
    def cache_config(self):
        """Cache configuration for testing."""
        return CacheConfig(
            eviction_policy=EvictionPolicy.TTL,
            max_cache_size=100,
            memory_config=MemoryConfig(
                max_memory_mb=500.0,
                warning_memory_mb=400.0,
                cleanup_memory_mb=350.0,
                eviction_batch_size=10
            )
        )

    @pytest.fixture
    def cache_instance(self, cache_config, mock_redis):
        """Cache instance with mocked Redis client."""
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
    async def test_memory_thresholds_check(self, cache_instance):
        """Test memory threshold checking."""
        with patch.object(cache_instance, '_get_memory_usage') as mock_get_memory:
            # Test normal memory usage
            mock_get_memory.return_value = 200.0
            status = cache_instance._check_memory_thresholds()
            assert not status["exceeds_max"]
            assert not status["exceeds_warning"]
            assert not status["exceeds_cleanup"]

            # Test warning threshold
            mock_get_memory.return_value = 400.0
            status = cache_instance._check_memory_thresholds()
            assert not status["exceeds_max"]
            assert status["exceeds_warning"]
            assert status["exceeds_cleanup"]

            # Test max threshold
            mock_get_memory.return_value = 600.0
            status = cache_instance._check_memory_thresholds()
            assert status["exceeds_max"]
            assert status["exceeds_warning"]
            assert status["exceeds_cleanup"]

    @pytest.mark.asyncio
    async def test_evict_expired_entries(self, cache_instance, mock_redis):
        """Test eviction of expired entries."""
        # Mock TTL responses - some expired (0), some valid (>0), some no TTL (-1)
        mock_redis.ttl.side_effect = [0, 30, -1, 0]  # Two expired entries

        evicted = await cache_instance._evict_expired_entries()

        # Should have called delete for expired keys
        mock_redis.delete.assert_called()
        assert evicted >= 0

    @pytest.mark.asyncio
    async def test_enforce_cache_limits_ttl_policy(self, cache_instance, mock_redis):
        """Test cache size enforcement with TTL policy."""
        mock_redis.dbsize.return_value = 150  # Over limit

        with patch.object(cache_instance, '_evict_oldest') as mock_evict:
            mock_evict.return_value = 10

            evicted = await cache_instance._enforce_cache_limits()

            mock_evict.assert_called_once_with(10)
            assert evicted == 10

    @pytest.mark.asyncio
    async def test_enforce_cache_limits_lru_policy(self, cache_instance, mock_redis):
        """Test cache size enforcement with LRU policy."""
        cache_instance.config.eviction_policy = EvictionPolicy.LRU
        mock_redis.dbsize.return_value = 150

        with patch.object(cache_instance, '_evict_lru') as mock_evict:
            mock_evict.return_value = 10

            evicted = await cache_instance._enforce_cache_limits()

            mock_evict.assert_called_once_with(10)
            assert evicted == 10

    @pytest.mark.asyncio
    async def test_enforce_cache_limits_lfu_policy(self, cache_instance, mock_redis):
        """Test cache size enforcement with LFU policy."""
        cache_instance.config.eviction_policy = EvictionPolicy.LFU
        mock_redis.dbsize.return_value = 150

        with patch.object(cache_instance, '_evict_lfu') as mock_evict:
            mock_evict.return_value = 10

            evicted = await cache_instance._enforce_cache_limits()

            mock_evict.assert_called_once_with(10)
            assert evicted == 10

    @pytest.mark.asyncio
    async def test_evict_lru(self, cache_instance, mock_redis):
        """Test LRU eviction logic."""
        mock_redis.object.return_value = [100, 50, 200]  # idle times
        mock_redis.keys.return_value = ["key1", "key2", "key3"]

        evicted = await cache_instance._evict_lru(2)

        mock_redis.delete.assert_called_once()
        assert evicted >= 0

    @pytest.mark.asyncio
    async def test_evict_lfu(self, cache_instance, mock_redis):
        """Test LFU eviction logic."""
        mock_redis.object.return_value = [10, 5, 15]  # access frequencies
        mock_redis.keys.return_value = ["key1", "key2", "key3"]

        evicted = await cache_instance._evict_lfu(2)

        mock_redis.delete.assert_called_once()
        assert evicted >= 0

    @pytest.mark.asyncio
    async def test_evict_oldest(self, cache_instance, mock_redis):
        """Test TTL-based oldest entry eviction."""
        mock_redis.ttl.return_value = [300, 60, 120]  # TTL values
        mock_redis.keys.return_value = ["key1", "key2", "key3"]

        evicted = await cache_instance._evict_oldest(2)

        mock_redis.delete.assert_called_once()
        assert evicted >= 0

    @pytest.mark.asyncio
    async def test_perform_maintenance_normal_conditions(self, cache_instance, mock_redis):
        """Test maintenance under normal conditions."""
        mock_redis.dbsize.return_value = 50  # Under limit

        with patch.object(cache_instance, '_get_memory_usage') as mock_memory:
            mock_memory.return_value = 200.0  # Normal memory

            result = await cache_instance.perform_maintenance()

            assert not result["maintenance_performed"]
            assert result["memory_mb"] == 200.0
            assert result["cache_size"] == 50

    @pytest.mark.asyncio
    async def test_perform_maintenance_high_memory(self, cache_instance, mock_redis):
        """Test maintenance under high memory conditions."""
        mock_redis.dbsize.return_value = 50

        with patch.object(cache_instance, '_get_memory_usage') as mock_memory, \
             patch.object(cache_instance, '_evict_expired_entries') as mock_expired, \
             patch.object(cache_instance, '_enforce_cache_limits') as mock_limits:

            mock_memory.return_value = 400.0  # High memory
            mock_expired.return_value = 5
            mock_limits.return_value = 3

            result = await cache_instance.perform_maintenance()

            assert result["maintenance_performed"]
            assert result["evicted_expired"] == 5
            assert result["evicted_limits"] == 3

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, cache_instance, mock_redis):
        """Test cache statistics retrieval."""
        mock_redis.dbsize.return_value = 75
        mock_redis.info.return_value = {"used_memory": 100000000, "used_memory_peak": 150000000}

        with patch.object(cache_instance, '_get_memory_usage') as mock_memory:
            mock_memory.return_value = 250.0

            stats = await cache_instance.get_cache_stats()

            assert stats["cache_size"] == 75
            assert stats["memory_mb"] == 250.0
            assert stats["eviction_policy"] == "ttl"
            assert stats["redis_memory_used"] == 95.367431640625  # 100MB in MB

    @pytest.mark.asyncio
    async def test_cache_not_connected(self, cache_instance):
        """Test behavior when cache is not connected."""
        cache_instance._connected = False
        cache_instance._redis_client = None

        result = await cache_instance.perform_maintenance()
        assert result == {"error": "Cache not connected"}

        stats = await cache_instance.get_cache_stats()
        assert stats == {"error": "Cache not connected"}

    def test_initialize_cache_with_eviction_config(self):
        """Test cache initialization with eviction configuration."""
        config = {
            "host": "localhost",
            "port": 6379,
            "eviction_policy": "lru",
            "max_cache_size": 5000,
            "memory_config": {
                "max_memory_mb": 800.0,
                "warning_memory_mb": 600.0,
                "cleanup_memory_mb": 500.0,
                "eviction_batch_size": 50
            }
        }

        # Mock the Redis client creation
        with patch('redis.asyncio.Redis') as mock_redis_class, \
             patch('asyncio.run') as mock_asyncio_run:

            mock_redis_instance = AsyncMock()
            mock_redis_instance.ping.return_value = True
            mock_redis_class.return_value = mock_redis_instance
            mock_asyncio_run.return_value = True

            success = initialize_cache(config)

            # Verify cache was initialized with correct config
            cache = get_cache()
            assert cache is not None
            assert cache.config.eviction_policy == EvictionPolicy.LRU
            assert cache.config.max_cache_size == 5000
            assert cache.config.memory_config.max_memory_mb == 800.0

            # Clean up
            close_cache()

    @pytest.mark.asyncio
    async def test_thread_safety(self, cache_instance):
        """Test thread safety of cache operations."""
        results = []
        errors = []

        async def worker(worker_id):
            """Worker function for concurrent operations."""
            try:
                for i in range(10):
                    key = f"test:{worker_id}:{i}"
                    await cache_instance.set(key, f"value_{i}")
                    value = await cache_instance.get(key)
                    results.append((worker_id, i, value))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Run multiple workers concurrently
        tasks = [worker(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify no errors occurred
        assert len(errors) == 0

        # Verify results were collected
        assert len(results) == 50  # 5 workers * 10 iterations

    @pytest.mark.asyncio
    async def test_eviction_under_load(self, cache_instance, mock_redis):
        """Test eviction behavior under load."""
        # Simulate high cache usage
        mock_redis.dbsize.return_value = 200  # Well over limit

        with patch.object(cache_instance, '_evict_oldest') as mock_evict:
            mock_evict.return_value = 20

            # Trigger maintenance
            result = await cache_instance.perform_maintenance()

            assert result["maintenance_performed"]
            mock_evict.assert_called()

    def test_memory_config_defaults(self):
        """Test memory configuration defaults."""
        config = MemoryConfig()

        assert config.max_memory_mb == 500.0
        assert config.warning_memory_mb == 400.0
        assert config.cleanup_memory_mb == 350.0
        assert config.eviction_batch_size == 100
        assert config.memory_check_interval == 60.0

    def test_cache_config_defaults(self):
        """Test cache configuration defaults."""
        config = CacheConfig()

        assert config.eviction_policy == EvictionPolicy.TTL
        assert config.max_cache_size == 10000
        assert config.memory_config is not None
        assert isinstance(config.memory_config, MemoryConfig)

    @pytest.mark.asyncio
    async def test_error_handling_in_eviction(self, cache_instance, mock_redis):
        """Test error handling in eviction operations."""
        # Simulate Redis errors
        mock_redis.keys.side_effect = Exception("Redis connection error")

        # Should not raise exception, should handle gracefully
        evicted = await cache_instance._evict_expired_entries()
        assert evicted == 0

        evicted = await cache_instance._enforce_cache_limits()
        assert evicted == 0


class TestCacheIntegration:
    """Integration tests for cache with memory manager."""

    def test_memory_manager_integration(self):
        """Test integration with memory manager."""
        from core.memory_manager import MemoryManager

        memory_manager = MemoryManager(enable_monitoring=False)

        # Mock cache instance
        mock_cache = Mock()
        mock_cache.perform_maintenance = AsyncMock(return_value={"maintenance_performed": True})

        # Integrate cache with memory manager
        memory_manager.integrate_cache_maintenance(mock_cache)

        # Verify callback was added
        assert len(memory_manager._cleanup_callbacks) > 0

    @pytest.mark.asyncio
    async def test_end_to_end_eviction_workflow(self, cache_instance, mock_redis):
        """Test complete eviction workflow."""
        # Set up scenario: high memory + large cache
        mock_redis.dbsize.return_value = 150

        with patch.object(cache_instance, '_get_memory_usage') as mock_memory, \
             patch.object(cache_instance, '_evict_expired_entries') as mock_expired, \
             patch.object(cache_instance, '_enforce_cache_limits') as mock_limits:

            mock_memory.return_value = 400.0  # High memory
            mock_expired.return_value = 5
            mock_limits.return_value = 15

            # Perform maintenance
            result = await cache_instance.perform_maintenance()

            # Verify workflow completed
            assert result["maintenance_performed"]
            assert result["evicted_expired"] == 5
            assert result["evicted_limits"] == 15
            assert result["memory_status"]["exceeds_warning"]


if __name__ == "__main__":
    pytest.main([__file__])
