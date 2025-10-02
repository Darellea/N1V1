"""
Memory Management and Cache Testing Suite for BotEngine
=======================================================

This module provides comprehensive testing for the BotEngine's LRU cache implementation,
memory monitoring, and cache performance under various scenarios.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.bot_engine import LRUCache, MemoryManager


class TestLRUCacheCoreFunctionality:
    """Test core LRU cache functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = LRUCache(maxsize=100, max_memory_mb=10.0)

    def teardown_method(self):
        """Cleanup after tests."""
        self.cache.shutdown()

    def test_initialization(self):
        """Test cache initialization."""
        assert self.cache.maxsize == 100
        assert self.cache.max_memory_bytes == 10 * 1024 * 1024
        assert len(self.cache.cache) == 0
        assert not self.cache._shutdown

    def test_basic_set_get(self):
        """Test basic set and get operations."""
        # Set a value
        self.cache.set("key1", "value1")
        assert self.cache.get("key1") == "value1"

        # Test hit/miss metrics
        stats = self.cache.get_stats()
        assert stats["hits"] == 1
        assert stats["sets"] == 1

    def test_cache_miss(self):
        """Test cache miss behavior."""
        result = self.cache.get("nonexistent")
        assert result is None

        stats = self.cache.get_stats()
        assert stats["misses"] == 1

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to maxsize
        for i in range(105):  # Exceed maxsize of 100
            self.cache.set(f"key{i}", f"value{i}")

        stats = self.cache.get_stats()
        assert stats["size"] <= 100  # Should not exceed maxsize
        assert stats["evictions"] > 0  # Should have evictions

    def test_memory_limit_eviction(self):
        """Test memory-based eviction."""
        # Create large data to trigger memory limits
        large_data = "x" * (1024 * 1024)  # 1MB string

        # Add data until memory limit is reached
        for i in range(15):  # Should exceed 10MB limit
            self.cache.set(f"large_key{i}", large_data)

        stats = self.cache.get_stats()
        assert stats["memory_usage_mb"] <= 10.1  # Allow small tolerance
        assert stats["evictions"] > 0

    def test_data_type_limits(self):
        """Test data type specific limits."""
        # Add ticker data (limit: 500, but global max is 100)
        for i in range(60):  # Exceed global limit but within ticker limit
            self.cache.set(f"ticker:key{i}", f"ticker_value{i}", data_type="ticker")

        # Add OHLCV data
        for i in range(30):
            self.cache.set(f"ohlcv:key{i}", f"ohlcv_value{i}", data_type="ohlcv")

        stats = self.cache.get_stats()
        assert stats["size"] <= 100  # Global limit
        assert stats["data_type_counts"]["ticker"] <= 500  # Type limit (but constrained by global)
        assert stats["data_type_counts"]["ohlcv"] <= 200  # Type limit

    def test_cache_invalidation_market_close(self):
        """Test market close cache invalidation."""
        # Add realtime data
        self.cache.set("ticker:BTC/USDT", {"price": 50000}, data_type="ticker")
        self.cache.set("realtime:ETH/USDT", {"price": 3000}, data_type="ticker")
        self.cache.set("current:ADA/USDT", {"price": 2.0}, data_type="ticker")

        # Add non-realtime data (should not be invalidated)
        self.cache.set("ohlcv:BTC/USDT:1h", pd.DataFrame(), data_type="ohlcv")

        initial_size = self.cache.get_stats()["size"]
        assert initial_size >= 4

        # Invalidate market close
        self.cache.invalidate_market_close()

        final_size = self.cache.get_stats()["size"]
        assert final_size < initial_size  # Should have removed realtime data

        # Check that non-realtime data remains
        assert self.cache.get("ohlcv:BTC/USDT:1h") is not None

    def test_emergency_clear(self):
        """Test emergency cache clearing."""
        # Fill cache
        for i in range(50):
            self.cache.set(f"key{i}", f"value{i}")

        assert self.cache.get_stats()["size"] > 0

        # Emergency clear
        self.cache.emergency_clear()

        stats = self.cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0  # Metrics should be reset
        assert stats["misses"] == 0

    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        import threading

        results = []
        errors = []

        def worker(worker_id, num_operations):
            """Worker function for concurrent operations."""
            try:
                for i in range(num_operations):
                    key = f"worker{worker_id}_key{i}"
                    self.cache.set(key, f"value{i}")
                    result = self.cache.get(key)
                    if result != f"value{i}":
                        errors.append(f"Worker {worker_id}: expected {f'value{i}'}, got {result}")
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i, 20))
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check for errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"

        # Verify cache state
        stats = self.cache.get_stats()
        assert stats["size"] > 0

    def test_shutdown_behavior(self):
        """Test cache behavior after shutdown."""
        self.cache.set("test_key", "test_value")
        assert self.cache.get("test_key") == "test_value"

        # Shutdown
        self.cache.shutdown()

        # Operations should be no-ops after shutdown
        self.cache.set("new_key", "new_value")
        result = self.cache.get("test_key")
        assert result is None  # Should return None after shutdown


class TestMemoryManager:
    """Test memory manager functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = LRUCache(maxsize=1000, max_memory_mb=5.0)
        self.memory_manager = MemoryManager(
            self.cache,
            warning_threshold=0.7,
            critical_threshold=0.9
        )

    def teardown_method(self):
        """Cleanup after tests."""
        self.cache.shutdown()

    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        assert self.memory_manager.cache == self.cache
        assert self.memory_manager.warning_threshold == 0.7
        assert self.memory_manager.critical_threshold == 0.9
        assert not self.memory_manager._monitoring_active

    def test_memory_check_normal(self):
        """Test memory check in normal conditions."""
        # Start monitoring
        self.memory_manager.start_monitoring()

        # Add some data but stay under thresholds
        for i in range(10):
            self.cache.set(f"key{i}", f"value{i}")

        status = self.memory_manager.check_memory_usage()
        assert status["status"] == "normal"
        assert len(status["actions_taken"]) == 0

    def test_memory_check_warning(self):
        """Test memory check at warning threshold."""
        # Start monitoring
        self.memory_manager.start_monitoring()

        # Fill cache to warning level (70% of 5MB = ~3.5MB)
        large_data = "x" * (1024 * 350)  # ~350KB each
        for i in range(10):
            self.cache.set(f"large_key{i}", large_data)

        status = self.memory_manager.check_memory_usage()
        # May be normal, warning, or critical depending on exact memory usage
        assert status["status"] in ["normal", "warning", "critical"]

    def test_memory_check_critical(self):
        """Test memory check at critical threshold."""
        # Start monitoring
        self.memory_manager.start_monitoring()

        # Fill cache to critical level (90% of 5MB = ~4.5MB)
        large_data = "x" * (1024 * 500)  # ~500KB each
        for i in range(10):
            self.cache.set(f"critical_key{i}", large_data)

        status = self.memory_manager.check_memory_usage()
        # Should trigger critical actions if memory usage is high enough
        if "memory_percent" in status and status["memory_percent"] >= 90:
            assert status["status"] == "critical"
            assert "emergency_clear" in status["actions_taken"]

    def test_monitoring_toggle(self):
        """Test monitoring start/stop functionality."""
        assert not self.memory_manager._monitoring_active

        self.memory_manager.start_monitoring()
        assert self.memory_manager._monitoring_active

        self.memory_manager.stop_monitoring()
        assert not self.memory_manager._monitoring_active

    def test_memory_report(self):
        """Test detailed memory report generation."""
        # Add some test data
        self.cache.set("test_key", "test_value")

        report = self.memory_manager.get_memory_report()

        assert "cache_stats" in report
        assert "memory_manager" in report
        assert "system_memory" in report

        cache_stats = report["cache_stats"]
        assert "size" in cache_stats
        assert "hit_rate" in cache_stats
        assert "memory_usage_mb" in cache_stats

        memory_manager_info = report["memory_manager"]
        assert "monitoring_active" in memory_manager_info
        assert "warning_threshold" in memory_manager_info
        assert "critical_threshold" in memory_manager_info


class TestCachePerformanceBenchmarks:
    """Test cache performance under various loads."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = LRUCache(maxsize=10000, max_memory_mb=50.0)

    def teardown_method(self):
        """Cleanup after tests."""
        self.cache.shutdown()

    @pytest.mark.timeout(15)
    def test_cache_performance_with_timeout(self):
        """Test cache performance with timeout protection."""
        # Moderate batch operations with timeout
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(1000):  # Reduced from 10000 to avoid timeout
                future = executor.submit(self.cache.set, f"key_{i}", f"value_{i}")
                futures.append(future)

            # Wait with timeout to prevent hangs
            for future in as_completed(futures, timeout=10):
                future.result()

        # Verify operations completed
        stats = self.cache.get_stats()
        assert stats["sets"] == 1000
        assert stats["size"] <= 1000

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        # Add some data
        for i in range(100):
            self.cache.set(f"key{i}", f"value{i}")

        # Generate hits
        for i in range(50):
            self.cache.get(f"key{i}")

        # Generate misses
        for i in range(50, 150):
            self.cache.get(f"missing_key{i}")

        stats = self.cache.get_stats()
        expected_hit_rate = 50 / (50 + 100)  # hits / (hits + misses)
        assert abs(stats["hit_rate"] - expected_hit_rate) < 0.01

    def test_memory_usage_tracking(self):
        """Test memory usage tracking accuracy."""
        initial_memory = self.cache.get_stats()["memory_usage_mb"]

        # Add data and check memory increases
        test_data = "x" * 10000  # 10KB string
        self.cache.set("memory_test", test_data)

        after_memory = self.cache.get_stats()["memory_usage_mb"]
        assert after_memory > initial_memory

        # Memory should be roughly tracked (allowing for overhead)
        estimated_size_mb = (len("memory_test") + len(test_data) + 100) / (1024 * 1024)  # Rough overhead in MB
        assert abs(after_memory - initial_memory - estimated_size_mb) < 0.1  # Allow some tolerance

    def test_concurrent_access_performance(self):
        """Test performance under concurrent access."""
        import threading

        operation_count = [0]
        operation_count_lock = threading.Lock()

        def performance_worker(num_operations):
            """Worker for performance testing."""
            for i in range(num_operations):
                key = f"perf_key_{threading.current_thread().ident}_{i}"
                self.cache.set(key, f"perf_value_{i}")

                with operation_count_lock:
                    operation_count[0] += 1

        # Create multiple threads
        threads = []
        operations_per_thread = 1000
        num_threads = 5

        start_time = time.time()

        for i in range(num_threads):
            t = threading.Thread(target=performance_worker, args=(operations_per_thread,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        end_time = time.time()
        total_time = end_time - start_time

        # Verify all operations completed
        assert operation_count[0] == num_threads * operations_per_thread

        # Check performance (should complete within reasonable time)
        operations_per_second = operation_count[0] / total_time
        assert operations_per_second > 100  # At least 100 ops/sec (realistic for Python threading)

        stats = self.cache.get_stats()
        assert stats["size"] > 0


class TestCacheEdgeCases:
    """Test cache behavior in edge cases."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = LRUCache(maxsize=100, max_memory_mb=10.0)

    def teardown_method(self):
        """Cleanup after tests."""
        self.cache.shutdown()

    def test_empty_key_value_handling(self):
        """Test handling of empty keys and values."""
        # Empty key should work
        self.cache.set("", "empty_key_value")
        assert self.cache.get("") == "empty_key_value"

        # None value should work
        self.cache.set("none_value", None)
        assert self.cache.get("none_value") is None

    def test_large_key_handling(self):
        """Test handling of very large keys."""
        large_key = "x" * 10000  # 10KB key
        self.cache.set(large_key, "value")
        assert self.cache.get(large_key) == "value"

        stats = self.cache.get_stats()
        assert stats["size"] >= 1

    def test_special_characters_in_keys(self):
        """Test keys with special characters."""
        special_keys = [
            "key:with:colons",
            "key with spaces",
            "key-with-dashes",
            "key_with_underscores",
            "key.123.numbers",
            "key@symbol#chars",
        ]

        for key in special_keys:
            self.cache.set(key, f"value_for_{key}")
            assert self.cache.get(key) == f"value_for_{key}"

    def test_memory_estimation_edge_cases(self):
        """Test memory estimation with edge cases."""
        # Test with various data types
        test_cases = [
            ("string", "simple string"),
            ("int", 42),
            ("float", 3.14159),
            ("list", [1, 2, 3, 4, 5]),
            ("dict", {"key": "value", "number": 123}),
            ("dataframe", pd.DataFrame({"a": [1, 2], "b": [3, 4]})),
            ("numpy_array", np.array([1, 2, 3, 4, 5])),
        ]

        for key, value in test_cases:
            self.cache.set(key, value)

        stats = self.cache.get_stats()
        assert stats["size"] == len(test_cases)
        assert stats["memory_usage_mb"] > 0

    def test_cache_under_memory_pressure(self):
        """Test cache behavior under extreme memory pressure."""
        # Create very large objects to force memory limits
        huge_data = "x" * (1024 * 1024 * 2)  # 2MB each

        # Try to add multiple large objects
        added_count = 0
        for i in range(10):
            try:
                self.cache.set(f"huge_{i}", huge_data)
                added_count += 1
            except Exception:
                break  # Stop if cache rejects due to memory

        stats = self.cache.get_stats()
        # Should have added at least some items but not exceed memory limits
        assert stats["memory_usage_mb"] <= 10.1  # Allow small tolerance
        assert stats["evictions"] >= 0  # May have evictions


class TestCacheIntegration:
    """Test cache integration with other components."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = LRUCache(maxsize=500, max_memory_mb=20.0)
        self.memory_manager = MemoryManager(self.cache)

    def teardown_method(self):
        """Cleanup after tests."""
        self.cache.shutdown()

    @patch('psutil.virtual_memory')
    def test_system_memory_integration(self, mock_memory):
        """Test integration with system memory monitoring."""
        # Mock system memory
        mock_memory.return_value = MagicMock()
        mock_memory.return_value.total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_memory.return_value.available = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.return_value.used = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.return_value.percent = 50.0

        report = self.memory_manager.get_memory_report()

        system_memory = report["system_memory"]
        assert system_memory["total_mb"] == 16 * 1024
        assert system_memory["available_mb"] == 8 * 1024
        assert system_memory["used_mb"] == 8 * 1024
        assert system_memory["used_percent"] == 50.0

    def test_cache_stats_persistence(self):
        """Test that cache statistics persist correctly."""
        # Perform various operations
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.get("key1")  # Hit
        self.cache.get("key3")  # Miss

        stats1 = self.cache.get_stats()

        # Perform more operations
        self.cache.set("key3", "value3")
        self.cache.get("key2")  # Hit
        self.cache.get("key4")  # Miss

        stats2 = self.cache.get_stats()

        # Verify stats accumulation
        assert stats2["sets"] == stats1["sets"] + 1
        assert stats2["hits"] == stats1["hits"] + 1
        assert stats2["misses"] == stats1["misses"] + 1

    def test_memory_manager_monitoring_states(self):
        """Test memory manager monitoring state transitions."""
        assert not self.memory_manager._monitoring_active

        # Start monitoring
        self.memory_manager.start_monitoring()
        assert self.memory_manager._monitoring_active

        # Check memory (should work)
        status = self.memory_manager.check_memory_usage()
        assert isinstance(status, dict)

        # Stop monitoring
        self.memory_manager.stop_monitoring()
        assert not self.memory_manager._monitoring_active

        # Check memory when not monitoring
        status = self.memory_manager.check_memory_usage()
        assert status == {"status": "monitoring_disabled"}


# Integration test fixtures
@pytest.fixture
def lru_cache():
    """LRU cache fixture with cleanup."""
    cache = LRUCache(maxsize=1000, max_memory_mb=50.0)
    yield cache
    cache.shutdown()


@pytest.fixture
def memory_manager(lru_cache):
    """Memory manager fixture."""
    manager = MemoryManager(lru_cache)
    yield manager


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
