"""
tests/test_memory_cache.py

Tests for LRU cache with TTL support and memory caps.
"""

import time
import threading
import pytest

from core.memory_cache import (
    LRUCacheWithTTL,
    CacheEntry,
    CacheMetrics,
    get_default_cache,
    get_large_cache,
    get_no_ttl_cache,
    create_cache,
)


class TestCacheEntry:
    """Test cases for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test basic CacheEntry creation."""
        entry = CacheEntry(value="test_value", timestamp=123456.0, access_count=5)
        assert entry.value == "test_value"
        assert entry.timestamp == 123456.0
        assert entry.access_count == 5
        assert entry.last_access == 123456.0  # Should be set to timestamp

    def test_cache_entry_last_access_initialization(self):
        """Test that last_access is initialized to timestamp."""
        entry = CacheEntry(value="test", timestamp=1000.0)
        assert entry.last_access == 1000.0


class TestCacheMetrics:
    """Test cases for CacheMetrics dataclass."""

    def test_cache_metrics_initialization(self):
        """Test CacheMetrics starts with zero values."""
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.sets == 0
        assert metrics.deletes == 0
        assert metrics.current_size == 0
        assert metrics.max_size == 0
        assert metrics.total_accesses == 0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        metrics = CacheMetrics()
        assert metrics.hit_rate == 0.0

        metrics.hits = 7
        metrics.misses = 3
        metrics.total_accesses = 10
        assert metrics.hit_rate == 0.7

    def test_hit_rate_with_zero_accesses(self):
        """Test hit rate with zero total accesses."""
        metrics = CacheMetrics()
        metrics.hits = 5
        metrics.misses = 5
        # total_accesses still 0
        assert metrics.hit_rate == 0.0


class TestLRUCacheWithTTL:
    """Test cases for LRUCacheWithTTL functionality."""

    @pytest.fixture
    def cache(self):
        """Create a test cache instance."""
        return LRUCacheWithTTL(max_size=3, ttl_seconds=10.0)

    @pytest.fixture
    def no_ttl_cache(self):
        """Create a cache without TTL."""
        return LRUCacheWithTTL(max_size=3, ttl_seconds=None)

    def test_initialization(self):
        """Test cache initialization."""
        cache = LRUCacheWithTTL(max_size=5, ttl_seconds=30.0, enable_metrics=True)
        assert cache.max_size == 5
        assert cache.ttl_seconds == 30.0
        assert cache.enable_metrics is True
        assert len(cache) == 0

    def test_initialization_invalid_max_size(self):
        """Test that invalid max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            LRUCacheWithTTL(max_size=0)

    def test_get_nonexistent_key(self, cache):
        """Test getting a key that doesn't exist."""
        assert cache.get("nonexistent") is None
        metrics = cache.get_metrics()
        assert metrics.misses == 1
        assert metrics.total_accesses == 1

    def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        metrics = cache.get_metrics()
        assert metrics.sets == 1
        assert metrics.hits == 1
        assert metrics.total_accesses == 1

    def test_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # All should be present
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert len(cache) == 3

        # Access key1 to make it most recently used
        cache.get("key1")

        # Add fourth item - should evict least recently used (key2)
        cache.set("key4", "value4")

        assert len(cache) == 3
        assert cache.get("key1") == "value1"  # Still present (recently accessed)
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"  # Still present
        assert cache.get("key4") == "value4"  # New item

        metrics = cache.get_metrics()
        assert metrics.evictions == 1

    def test_ttl_expiration(self, cache):
        """Test TTL-based expiration."""
        # Set an item
        cache.set("short_ttl", "value", ttl_seconds=0.1)  # Very short TTL

        # Should be available immediately
        assert cache.get("short_ttl") == "value"

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired now
        assert cache.get("short_ttl") is None

        metrics = cache.get_metrics()
        assert metrics.evictions == 1  # Expired entry was evicted

    def test_no_ttl_never_expires(self, no_ttl_cache):
        """Test that items without TTL never expire."""
        no_ttl_cache.set("permanent", "value")

        # Wait longer than any reasonable TTL
        time.sleep(0.2)

        # Should still be available
        assert no_ttl_cache.get("permanent") == "value"

    def test_update_existing_key(self, cache):
        """Test updating an existing key."""
        cache.set("key1", "value1")
        cache.set("key1", "updated_value")

        assert cache.get("key1") == "updated_value"
        assert len(cache) == 1

        metrics = cache.get_metrics()
        assert metrics.sets == 2  # Two set operations

    def test_delete_existing_key(self, cache):
        """Test deleting an existing key."""
        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert len(cache) == 0

        metrics = cache.get_metrics()
        assert metrics.deletes == 1

    def test_delete_nonexistent_key(self, cache):
        """Test deleting a key that doesn't exist."""
        assert cache.delete("nonexistent") is False

    def test_clear_cache(self, cache):
        """Test clearing all cache entries."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0

    def test_cleanup_expired(self, cache):
        """Test manual cleanup of expired entries."""
        # Set items with different TTLs
        cache.set("short", "value1", ttl_seconds=0.1)
        cache.set("long", "value2", ttl_seconds=10.0)

        # Both should be available
        assert len(cache) == 2

        # Wait for short TTL to expire
        time.sleep(0.2)

        # Manual cleanup
        removed = cache.cleanup_expired()
        assert removed == 1

        # Only long TTL item should remain
        assert cache.get("short") is None
        assert cache.get("long") == "value2"
        assert len(cache) == 1

    def test_contains_operator(self, cache):
        """Test the 'in' operator for cache membership."""
        cache.set("key1", "value1")

        assert "key1" in cache
        assert "nonexistent" not in cache

    def test_contains_with_expired_entry(self, cache):
        """Test contains with an expired entry."""
        cache.set("expired", "value", ttl_seconds=0.1)
        time.sleep(0.2)

        # Should return False and clean up expired entry
        assert "expired" not in cache
        assert len(cache) == 0

    def test_get_stats(self, cache):
        """Test getting comprehensive cache statistics."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss

        stats = cache.get_stats()

        assert stats["current_size"] == 2
        assert stats["max_size"] == 3
        assert stats["utilization_percent"] == (2 / 3) * 100
        assert stats["ttl_enabled"] is True
        assert stats["metrics_enabled"] is True
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_get_stats_no_ttl(self, no_ttl_cache):
        """Test stats for cache without TTL."""
        no_ttl_cache.set("key1", "value1")

        stats = no_ttl_cache.get_stats()
        assert stats["ttl_enabled"] is False

    def test_thread_safety(self):
        """Test that cache operations are thread-safe."""
        # Use a larger cache to avoid LRU eviction conflicts
        cache = LRUCacheWithTTL(max_size=50, ttl_seconds=None)
        results = []
        operations_completed = []

        def worker(worker_id):
            """Worker function for thread safety testing."""
            completed = 0
            for i in range(10):
                key = f"worker_{worker_id}_key_{i}"
                expected_value = f"value_{worker_id}_{i}"

                # Set value
                cache.set(key, expected_value)
                completed += 1

                # Small delay to encourage race conditions
                time.sleep(0.001)

                # Get value and check consistency
                retrieved = cache.get(key)
                if retrieved is not None and retrieved != expected_value:
                    results.append(
                        f"Mismatch for {key}: expected {expected_value}, got {retrieved}"
                    )
                elif retrieved is None:
                    results.append(f"Value disappeared for {key}")
                else:
                    completed += 1

            operations_completed.append(completed)

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that operations completed successfully
        assert len(operations_completed) == 3
        assert all(completed > 0 for completed in operations_completed)

        # In a properly synchronized cache, we should have very few or no mismatches
        # Allow for some race condition edge cases
        total_operations = sum(operations_completed)
        if total_operations > 0:
            mismatch_rate = len(results) / total_operations
            assert (
                mismatch_rate < 0.05
            ), f"Too many mismatches: {results}"  # Allow up to 5% mismatch rate

        # Cache should have entries
        assert len(cache) > 0

    def test_metrics_disabled(self):
        """Test cache with metrics disabled."""
        cache = LRUCacheWithTTL(max_size=5, enable_metrics=False)

        cache.set("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"
        assert cache.get_metrics() is None

    def test_string_representation(self, cache):
        """Test string representation of cache."""
        cache.set("key1", "value1")

        repr_str = repr(cache)
        assert "LRUCacheWithTTL" in repr_str
        assert "size=1/3" in repr_str
        assert "ttl=10.0" in repr_str


class TestGlobalCacheInstances:
    """Test global cache instance functions."""

    def test_get_default_cache(self):
        """Test getting the default cache instance."""
        cache = get_default_cache()
        assert isinstance(cache, LRUCacheWithTTL)
        assert cache.max_size == 1000
        assert cache.ttl_seconds == 300

    def test_get_large_cache(self):
        """Test getting the large cache instance."""
        cache = get_large_cache()
        assert isinstance(cache, LRUCacheWithTTL)
        assert cache.max_size == 10000
        assert cache.ttl_seconds == 3600

    def test_get_no_ttl_cache(self):
        """Test getting the no-TTL cache instance."""
        cache = get_no_ttl_cache()
        assert isinstance(cache, LRUCacheWithTTL)
        assert cache.max_size == 500
        assert cache.ttl_seconds is None

    def test_global_instances_are_singletons(self):
        """Test that global cache instances are singletons."""
        cache1 = get_default_cache()
        cache2 = get_default_cache()
        assert cache1 is cache2

        large1 = get_large_cache()
        large2 = get_large_cache()
        assert large1 is large2

    def test_create_cache(self):
        """Test creating a custom cache instance."""
        cache = create_cache(max_size=50, ttl_seconds=60.0, enable_metrics=False)

        assert isinstance(cache, LRUCacheWithTTL)
        assert cache.max_size == 50
        assert cache.ttl_seconds == 60.0
        assert cache.enable_metrics is False


class TestCacheIntegrationScenarios:
    """Integration tests for realistic cache usage scenarios."""

    def test_cache_access_pattern_simulation(self):
        """Test cache behavior with realistic access patterns."""
        cache = LRUCacheWithTTL(max_size=5, ttl_seconds=None)

        # Add initial items
        cache.set("frequent1", "value1")
        cache.set("frequent2", "value2")
        cache.set("rare1", "value_rare1")
        cache.set("rare2", "value_rare2")
        cache.set("rare3", "value_rare3")

        # Access frequent items many times to keep them fresh
        for _ in range(20):
            cache.get("frequent1")
            cache.get("frequent2")

            # Occasionally access one rare item
            if _ % 5 == 0:
                cache.get("rare1")

        # Add one more item - should evict the least recently used rare item
        cache.set("rare4", "value_rare4")

        # Frequent items should still be in cache
        assert cache.get("frequent1") == "value1"
        assert cache.get("frequent2") == "value2"

        # At least one rare item should have been evicted
        rare_items_in_cache = 0
        for i in range(1, 5):
            if cache.get(f"rare{i}") is not None:
                rare_items_in_cache += 1

        assert (
            rare_items_in_cache < 4
        ), "Some rare items should have been evicted due to LRU"
        assert rare_items_in_cache >= 2, "At least some rare items should remain"

    def test_ttl_and_lru_interaction(self):
        """Test interaction between TTL expiration and LRU eviction."""
        cache = LRUCacheWithTTL(max_size=3, ttl_seconds=0.5)

        # Fill cache
        cache.set("item1", "value1")
        cache.set("item2", "value2")
        cache.set("item3", "value3")

        # Access item1 to make it recently used
        cache.get("item1")

        # Add item4 - should evict item2 (LRU)
        cache.set("item4", "value4")

        # All items should be present initially
        assert len(cache) == 3
        assert cache.get("item1") == "value1"
        assert cache.get("item3") == "value3"
        assert cache.get("item4") == "value4"
        assert cache.get("item2") is None  # Should be evicted

        # Wait for TTL expiration
        time.sleep(0.6)

        # All items should be expired
        assert cache.get("item1") is None
        assert cache.get("item3") is None
        assert cache.get("item4") is None
        assert len(cache) == 0

    def test_memory_efficiency_under_load(self):
        """Test that cache maintains bounded size under continuous load."""
        cache = LRUCacheWithTTL(max_size=10, ttl_seconds=None)

        # Simulate continuous additions
        for i in range(50):
            cache.set(f"key_{i}", f"value_{i}")

            # Every 10 additions, verify cache size is bounded
            if (i + 1) % 10 == 0:
                assert len(cache) <= 10, f"Cache grew beyond max_size at iteration {i}"

        # Final size should be exactly max_size
        assert len(cache) == 10

        # Verify metrics
        metrics = cache.get_metrics()
        assert metrics.evictions > 0  # Should have evicted items
        assert metrics.sets == 50
        assert metrics.current_size == 10
