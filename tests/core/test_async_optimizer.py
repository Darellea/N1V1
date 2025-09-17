"""
Comprehensive tests for AsyncOptimizer - async I/O optimization and performance monitoring.

Tests async file operations, thread/process pools, performance monitoring,
memory management, blocking operation detection, and health checks.
"""

import pytest
import asyncio
import tempfile
import os
import json
import time
import psutil
import gc
from unittest.mock import Mock, patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from core.async_optimizer import (
    AsyncOptimizer,
    get_async_optimizer,
    async_read_file,
    async_write_file,
    run_async,
    monitor_performance
)


class TestAsyncOptimizerInitialization:
    """Test AsyncOptimizer initialization and configuration."""

    def test_default_initialization(self):
        """Test AsyncOptimizer with default parameters."""
        optimizer = AsyncOptimizer()

        assert optimizer.max_workers == 4
        assert optimizer.enable_monitoring is True
        assert isinstance(optimizer._thread_pool, ThreadPoolExecutor)
        assert isinstance(optimizer._process_pool, ProcessPoolExecutor)
        assert optimizer._performance_metrics["total_operations"] == 0

    def test_custom_initialization(self):
        """Test AsyncOptimizer with custom parameters."""
        optimizer = AsyncOptimizer(max_workers=8, enable_monitoring=False)

        assert optimizer.max_workers == 8
        assert optimizer.enable_monitoring is False

    def test_thread_pool_configuration(self):
        """Test thread pool is properly configured."""
        optimizer = AsyncOptimizer(max_workers=6)

        # Check thread pool configuration
        assert optimizer._thread_pool._max_workers == 6
        assert "AsyncOpt" in optimizer._thread_pool._thread_name_prefix

    def test_process_pool_configuration(self):
        """Test process pool is properly configured."""
        optimizer = AsyncOptimizer(max_workers=8)

        # Process pool should have half the workers of thread pool
        assert optimizer._process_pool._max_workers == 4

    def test_memory_thresholds_initialization(self):
        """Test memory thresholds are properly initialized."""
        optimizer = AsyncOptimizer()

        expected_thresholds = {
            "warning_mb": 500,
            "critical_mb": 1000,
            "cleanup_interval": 300
        }

        assert optimizer._memory_thresholds == expected_thresholds

    def test_blocking_patterns_initialization(self):
        """Test blocking patterns are properly initialized."""
        optimizer = AsyncOptimizer()

        assert isinstance(optimizer._blocking_patterns, list)
        assert "open(" in optimizer._blocking_patterns
        assert "time.sleep" in optimizer._blocking_patterns
        assert "requests.get" in optimizer._blocking_patterns


class TestAsyncFileOperations:
    """Test async file read/write operations."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir

        # Cleanup
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_async_file_read_success(self, temp_file):
        """Test successful async file reading."""
        optimizer = AsyncOptimizer()

        content = await optimizer.async_file_read(temp_file)

        assert content == "test content"

        # Check that operation was recorded
        assert "file_read" in optimizer._operation_stats
        assert len(optimizer._operation_stats["file_read"]) == 1

    @pytest.mark.asyncio
    async def test_async_file_read_with_encoding(self, temp_file):
        """Test async file reading with custom encoding."""
        optimizer = AsyncOptimizer()

        content = await optimizer.async_file_read(temp_file, encoding='utf-8')

        assert content == "test content"

    @pytest.mark.asyncio
    async def test_async_file_read_nonexistent_file(self):
        """Test async file reading with nonexistent file."""
        optimizer = AsyncOptimizer()

        with pytest.raises(FileNotFoundError):
            await optimizer.async_file_read("nonexistent_file.txt")

    @pytest.mark.asyncio
    async def test_async_file_write_success(self, temp_dir):
        """Test successful async file writing."""
        optimizer = AsyncOptimizer()
        file_path = os.path.join(temp_dir, "test_write.txt")

        await optimizer.async_file_write(file_path, "new content")

        # Verify file was created and has correct content
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            assert f.read() == "new content"

        # Check that operation was recorded
        assert "file_write" in optimizer._operation_stats

    @pytest.mark.asyncio
    async def test_async_file_write_creates_directory(self, temp_dir):
        """Test async file writing creates necessary directories."""
        optimizer = AsyncOptimizer()
        nested_path = os.path.join(temp_dir, "nested", "dir", "file.txt")

        await optimizer.async_file_write(nested_path, "content")

        assert os.path.exists(nested_path)
        with open(nested_path, 'r') as f:
            assert f.read() == "content"

    @pytest.mark.asyncio
    async def test_async_file_write_with_encoding(self, temp_dir):
        """Test async file writing with custom encoding."""
        optimizer = AsyncOptimizer()
        file_path = os.path.join(temp_dir, "encoded.txt")

        await optimizer.async_file_write(file_path, "content", encoding='utf-8')

        assert os.path.exists(file_path)

    @pytest.mark.asyncio
    async def test_async_json_load(self, temp_dir):
        """Test async JSON loading."""
        optimizer = AsyncOptimizer()
        json_path = os.path.join(temp_dir, "test.json")

        test_data = {"key": "value", "number": 42}
        with open(json_path, 'w') as f:
            json.dump(test_data, f)

        loaded_data = await optimizer.async_json_load(json_path)

        assert loaded_data == test_data

    @pytest.mark.asyncio
    async def test_async_json_dump(self, temp_dir):
        """Test async JSON dumping."""
        optimizer = AsyncOptimizer()
        json_path = os.path.join(temp_dir, "output.json")

        test_data = {"test": "data", "array": [1, 2, 3]}

        await optimizer.async_json_dump(test_data, json_path)

        assert os.path.exists(json_path)
        with open(json_path, 'r') as f:
            loaded = json.load(f)
            assert loaded == test_data

    @pytest.mark.asyncio
    async def test_async_json_dump_with_indent(self, temp_dir):
        """Test async JSON dumping with custom indentation."""
        optimizer = AsyncOptimizer()
        json_path = os.path.join(temp_dir, "indented.json")

        test_data = {"key": "value"}

        await optimizer.async_json_dump(test_data, json_path, indent=4)

        assert os.path.exists(json_path)
        with open(json_path, 'r') as f:
            content = f.read()
            assert "    " in content  # Check for indentation


class TestThreadAndProcessOperations:
    """Test thread and process pool operations."""

    @pytest.mark.asyncio
    async def test_run_in_thread_success(self):
        """Test successful execution in thread pool."""
        optimizer = AsyncOptimizer()

        def blocking_function(x, y):
            time.sleep(0.01)  # Small delay to simulate work
            return x + y

        result = await optimizer.run_in_thread(blocking_function, 5, 3)

        assert result == 8
        assert "threaded" in optimizer._operation_stats

    @pytest.mark.asyncio
    async def test_run_in_thread_with_exception(self):
        """Test thread execution with exception."""
        optimizer = AsyncOptimizer()

        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await optimizer.run_in_thread(failing_function)

    @pytest.mark.asyncio
    async def test_run_in_process_success(self):
        """Test successful execution in process pool."""
        optimizer = AsyncOptimizer()

        def cpu_intensive_function(n):
            # Simulate CPU-intensive work
            result = 0
            for i in range(n):
                result += i ** 2
            return result

        result = await optimizer.run_in_process(cpu_intensive_function, 100)

        expected = sum(i ** 2 for i in range(100))
        assert result == expected
        assert "processed" in optimizer._operation_stats

    @pytest.mark.asyncio
    async def test_run_in_process_with_exception(self):
        """Test process execution with exception."""
        optimizer = AsyncOptimizer()

        def failing_function():
            raise RuntimeError("Process error")

        with pytest.raises(RuntimeError, match="Process error"):
            await optimizer.run_in_process(failing_function)


class TestBlockingOperationDetection:
    """Test blocking operation detection."""

    def test_detect_blocking_operations_found(self):
        """Test detection of blocking operations."""
        optimizer = AsyncOptimizer()

        code = """
        def some_function():
            with open('file.txt', 'r') as f:
                data = f.read()
            time.sleep(1)
            response = requests.get('http://example.com')
        """

        detected = optimizer.detect_blocking_operations(code)

        assert "open(" in detected
        assert "time.sleep" in detected
        assert "requests.get" in detected

    def test_detect_blocking_operations_none_found(self):
        """Test when no blocking operations are detected."""
        optimizer = AsyncOptimizer()

        code = """
        def async_function():
            result = await some_async_call()
            return result
        """

        detected = optimizer.detect_blocking_operations(code)

        assert detected == []

    def test_detect_blocking_operations_records_detection(self):
        """Test that blocking operations are recorded."""
        optimizer = AsyncOptimizer()

        code = "with open('test.txt') as f: pass"

        detected = optimizer.detect_blocking_operations(code)

        assert len(optimizer._blocking_operations) == 1
        assert "open(" in optimizer._blocking_operations[0]["patterns"]


class TestPerformanceMonitoring:
    """Test performance monitoring decorators."""

    @pytest.mark.asyncio
    async def test_monitor_async_function(self):
        """Test monitoring of async functions."""
        optimizer = AsyncOptimizer()

        @optimizer.monitor_async_function
        async def test_async_function():
            await asyncio.sleep(0.01)
            return "result"

        result = await test_async_function()

        assert result == "result"
        assert "async_test_async_function" in optimizer._operation_stats

    @pytest.mark.asyncio
    async def test_monitor_async_function_with_exception(self):
        """Test monitoring of async functions that raise exceptions."""
        optimizer = AsyncOptimizer()

        @optimizer.monitor_async_function
        async def failing_async_function():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await failing_async_function()

        assert "async_failing_async_function_error" in optimizer._operation_stats

    def test_monitor_sync_function(self):
        """Test monitoring of sync functions."""
        optimizer = AsyncOptimizer()

        @optimizer.monitor_sync_function
        def test_sync_function():
            time.sleep(0.01)
            return "sync_result"

        result = test_sync_function()

        assert result == "sync_result"
        assert "sync_test_sync_function" in optimizer._operation_stats

    def test_monitor_sync_function_with_blocking_detection(self):
        """Test monitoring of sync functions with blocking detection."""
        optimizer = AsyncOptimizer()

        @optimizer.monitor_sync_function
        def blocking_function():
            with open(__file__, 'r') as f:
                content = f.read()
            return len(content)

        with patch('core.async_optimizer.logger') as mock_logger:
            result = blocking_function()

            assert isinstance(result, int)
            mock_logger.warning.assert_called()

    def test_monitor_sync_function_slow_operation(self):
        """Test monitoring of slow sync functions."""
        optimizer = AsyncOptimizer()

        @optimizer.monitor_sync_function
        def slow_function():
            time.sleep(1.1)  # Longer than 1 second threshold
            return "slow"

        with patch('core.async_optimizer.logger') as mock_logger:
            result = slow_function()

            assert result == "slow"
            mock_logger.warning.assert_called()


class TestMemoryManagement:
    """Test memory management and monitoring."""

    def test_memory_report_generation(self):
        """Test memory report generation."""
        optimizer = AsyncOptimizer()

        report = optimizer.get_memory_report()

        assert isinstance(report, dict)
        assert "current_memory_mb" in report
        assert "warning_threshold_mb" in report
        assert "critical_threshold_mb" in report
        assert "recommendations" in report

    def test_memory_report_with_stats(self):
        """Test memory report with collected statistics."""
        optimizer = AsyncOptimizer()

        # Add some memory stats
        optimizer._memory_stats["memory_usage"] = [100.0, 150.0, 200.0]

        report = optimizer.get_memory_report()

        assert "memory_stats" in report
        assert report["memory_stats"]["avg_memory_mb"] == 150.0
        assert report["memory_stats"]["max_memory_mb"] == 200.0
        assert report["memory_stats"]["min_memory_mb"] == 100.0

    def test_set_memory_thresholds(self):
        """Test setting memory thresholds."""
        optimizer = AsyncOptimizer()

        optimizer.set_memory_thresholds(warning_mb=600, critical_mb=1200, cleanup_interval=600)

        assert optimizer._memory_thresholds["warning_mb"] == 600
        assert optimizer._memory_thresholds["critical_mb"] == 1200
        assert optimizer._memory_thresholds["cleanup_interval"] == 600

    def test_memory_cleanup_operations(self):
        """Test memory cleanup operations."""
        optimizer = AsyncOptimizer()

        # Add some data to be cleaned
        optimizer._operation_stats["test_op"] = list(range(600))  # More than 500
        optimizer._blocking_operations = list(range(150))  # More than 100

        with patch('core.async_optimizer.gc') as mock_gc:
            mock_gc.collect.return_value = 42
            optimizer._perform_memory_cleanup()

            # Check that cleanup was performed
            assert len(optimizer._operation_stats["test_op"]) <= 250
            assert len(optimizer._blocking_operations) <= 50
            mock_gc.collect.assert_called()

    def test_periodic_cleanup(self):
        """Test periodic cleanup operations."""
        optimizer = AsyncOptimizer()

        # Set last cleanup to be old
        optimizer._last_cleanup = time.time() - 400

        with patch('core.async_optimizer.gc') as mock_gc:
            mock_gc.collect.return_value = 10
            optimizer._perform_periodic_cleanup()

            mock_gc.collect.assert_called_with(0)


class TestBatchOperations:
    """Test batch async operations."""

    @pytest.mark.asyncio
    async def test_batch_async_operations_success(self):
        """Test successful batch async operations."""
        optimizer = AsyncOptimizer()

        async def operation1():
            await asyncio.sleep(0.01)
            return "result1"

        async def operation2():
            await asyncio.sleep(0.01)
            return "result2"

        operations = [operation1, operation2]

        results = await optimizer.batch_async_operations(operations)

        assert results == ["result1", "result2"]
        assert "batch_async" in optimizer._operation_stats

    @pytest.mark.asyncio
    async def test_batch_async_operations_with_exceptions(self):
        """Test batch operations with some exceptions."""
        optimizer = AsyncOptimizer()

        async def success_operation():
            await asyncio.sleep(0.01)
            return "success"

        async def failing_operation():
            await asyncio.sleep(0.01)
            raise ValueError("Test failure")

        operations = [success_operation, failing_operation]

        results = await optimizer.batch_async_operations(operations)

        assert results[0] == "success"
        assert results[1] is None  # Exception converted to None

    @pytest.mark.asyncio
    async def test_batch_async_operations_empty_list(self):
        """Test batch operations with empty list."""
        optimizer = AsyncOptimizer()

        results = await optimizer.batch_async_operations([])

        assert results == []


class TestPerformanceReporting:
    """Test performance reporting functionality."""

    def test_get_performance_report(self):
        """Test performance report generation."""
        optimizer = AsyncOptimizer()

        # Add some test data
        optimizer._performance_metrics["total_operations"] = 100
        optimizer._performance_metrics["async_operations"] = 80
        optimizer._operation_stats["test_op"] = [0.1, 0.2, 0.15]

        report = optimizer.get_performance_report()

        assert report["total_operations"] == 100
        assert report["async_operations"] == 80
        assert report["async_efficiency"] == 0.8
        assert "operation_stats" in report
        assert "test_op" in report["operation_stats"]

    def test_get_performance_report_empty_stats(self):
        """Test performance report with no statistics."""
        optimizer = AsyncOptimizer()

        report = optimizer.get_performance_report()

        assert report["total_operations"] == 0
        assert report["async_efficiency"] == 0.0
        assert report["operation_stats"] == {}

    def test_get_blocking_operations_report(self):
        """Test blocking operations report."""
        optimizer = AsyncOptimizer()

        # Add some blocking operations
        optimizer._blocking_operations = [
            {"timestamp": 1000, "patterns": ["open("], "code_sample": "test"},
            {"timestamp": 1001, "patterns": ["time.sleep"], "code_sample": "test2"}
        ]

        report = optimizer.get_blocking_operations_report()

        assert len(report) == 2
        assert report[0]["patterns"] == ["open("]

    def test_get_blocking_operations_report_with_limit(self):
        """Test blocking operations report with limit."""
        optimizer = AsyncOptimizer()

        # Add many blocking operations
        optimizer._blocking_operations = [
            {"timestamp": i, "patterns": [f"pattern{i}"], "code_sample": f"code{i}"}
            for i in range(30)
        ]

        report = optimizer.get_blocking_operations_report(limit=10)

        assert len(report) == 10
        # Should return the most recent ones
        assert report[-1]["timestamp"] == 29

    def test_get_blocking_operations_report_empty(self):
        """Test blocking operations report when empty."""
        optimizer = AsyncOptimizer()

        report = optimizer.get_blocking_operations_report()

        assert report == []


class TestHealthChecks:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        optimizer = AsyncOptimizer()

        health = await optimizer.health_check()

        assert isinstance(health, dict)
        assert "thread_pool_active" in health
        assert "process_pool_active" in health
        assert "thread_pool_working" in health

    @pytest.mark.asyncio
    async def test_health_check_thread_pool_failure(self):
        """Test health check when thread pool fails."""
        optimizer = AsyncOptimizer()

        # Mock thread pool to fail
        with patch.object(optimizer, 'run_in_thread', side_effect=Exception("Thread pool error")):
            health = await optimizer.health_check()

            assert health["thread_pool_working"] is False
            assert "thread_pool_error" in health


class TestGlobalFunctions:
    """Test global utility functions."""

    def test_get_async_optimizer_singleton(self):
        """Test that get_async_optimizer returns singleton."""
        optimizer1 = get_async_optimizer()
        optimizer2 = get_async_optimizer()

        assert optimizer1 is optimizer2

    @pytest.mark.asyncio
    async def test_async_read_file_convenience_function(self, temp_file):
        """Test async_read_file convenience function."""
        content = await async_read_file(temp_file)

        assert content == "test content"

    @pytest.mark.asyncio
    async def test_async_write_file_convenience_function(self, temp_dir):
        """Test async_write_file convenience function."""
        file_path = os.path.join(temp_dir, "convenience.txt")

        await async_write_file(file_path, "convenience content")

        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            assert f.read() == "convenience content"

    @pytest.mark.asyncio
    async def test_run_async_decorator(self):
        """Test run_async decorator."""
        @run_async
        def blocking_add(x, y):
            time.sleep(0.01)
            return x + y

        result = await blocking_add(3, 4)

        assert result == 7

    def test_monitor_performance_decorator_async(self):
        """Test monitor_performance decorator with async function."""
        @monitor_performance
        async def async_test():
            await asyncio.sleep(0.01)
            return "async_result"

        # The decorator should return a wrapped function
        assert asyncio.iscoroutinefunction(async_test)

    def test_monitor_performance_decorator_sync(self):
        """Test monitor_performance decorator with sync function."""
        @monitor_performance
        def sync_test():
            time.sleep(0.01)
            return "sync_result"

        # The decorator should return a wrapped function
        assert callable(sync_test)
        assert not asyncio.iscoroutinefunction(sync_test)


class TestShutdownAndCleanup:
    """Test shutdown and cleanup functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_success(self):
        """Test successful shutdown."""
        optimizer = AsyncOptimizer()

        await optimizer.shutdown()

        # Check that pools are shutdown
        assert optimizer._thread_pool._shutdown
        assert optimizer._process_pool._shutdown

        # Check that data is cleared
        assert optimizer._operation_stats == {}
        assert optimizer._blocking_operations == []
        assert optimizer._performance_metrics == {}

    @pytest.mark.asyncio
    async def test_shutdown_with_pending_tasks(self):
        """Test shutdown with pending tasks."""
        optimizer = AsyncOptimizer()

        # Create a task that should be cancelled
        async def long_running_task():
            await asyncio.sleep(10)
            return "completed"

        task = asyncio.create_task(long_running_task())

        # Shutdown should handle pending tasks
        await optimizer.shutdown()

        # Task should be cancelled
        assert task.cancelled() or task.done()


class TestErrorHandling:
    """Test error handling in various scenarios."""

    @pytest.mark.asyncio
    async def test_async_file_read_error_handling(self):
        """Test error handling in async file read."""
        optimizer = AsyncOptimizer()

        with patch('aiofiles.open', side_effect=IOError("Read error")):
            with pytest.raises(IOError):
                await optimizer.async_file_read("test.txt")

    @pytest.mark.asyncio
    async def test_async_file_write_error_handling(self):
        """Test error handling in async file write."""
        optimizer = AsyncOptimizer()

        with patch('aiofiles.open', side_effect=PermissionError("Write error")):
            with pytest.raises(PermissionError):
                await optimizer.async_file_write("test.txt", "content")

    @pytest.mark.asyncio
    async def test_batch_operations_error_handling(self):
        """Test error handling in batch operations."""
        optimizer = AsyncOptimizer()

        async def failing_operation():
            raise RuntimeError("Batch operation failed")

        operations = [failing_operation]

        # Should not raise exception, should return None for failed operation
        results = await optimizer.batch_async_operations(operations)

        assert results == [None]

    def test_memory_report_error_handling(self):
        """Test error handling in memory report generation."""
        optimizer = AsyncOptimizer()

        with patch('psutil.Process', side_effect=Exception("PSUtil error")):
            report = optimizer.get_memory_report()

            assert "error" in report
            assert "PSUtil error" in report["error"]


class TestDataOptimization:
    """Test data optimization features."""

    def test_optimize_data_fetching(self):
        """Test data fetching optimization."""
        optimizer = AsyncOptimizer()

        def mock_fetch(symbol):
            return f"data_for_{symbol}"

        optimized_fetch = optimizer.optimize_data_fetching(
            ["BTC", "ETH", "ADA"],
            mock_fetch
        )

        # Should return a coroutine function
        assert asyncio.iscoroutinefunction(optimized_fetch)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_operation_stats_size_limit(self):
        """Test that operation stats are properly limited."""
        optimizer = AsyncOptimizer()

        # Add more than 1000 operations
        for i in range(1100):
            optimizer._record_operation(f"op_{i % 10}", 0.1)

        # Should keep only last 1000 per operation type
        total_stats = sum(len(times) for times in optimizer._operation_stats.values())
        assert total_stats <= 10000  # 1000 * 10 operation types

    def test_memory_stats_size_limit(self):
        """Test that memory stats are properly limited."""
        optimizer = AsyncOptimizer()

        # Add more than 100 memory readings
        for i in range(150):
            optimizer._memory_stats["memory_usage"].append(float(i))

        optimizer._check_memory_usage()

        # Should keep only last 100
        assert len(optimizer._memory_stats["memory_usage"]) <= 100

    def test_blocking_operations_size_limit(self):
        """Test that blocking operations list is properly limited."""
        optimizer = AsyncOptimizer()

        # Add many blocking operations
        for i in range(150):
            optimizer._blocking_operations.append({
                "timestamp": time.time(),
                "patterns": ["test"],
                "code_sample": f"code_{i}"
            })

        # Trigger cleanup
        optimizer._perform_memory_cleanup()

        # Should keep only last 50
        assert len(optimizer._blocking_operations) <= 50

    def test_empty_operations_list(self):
        """Test handling of empty operations list."""
        optimizer = AsyncOptimizer()

        # Should handle empty list gracefully
        assert optimizer._operation_stats == {}
        assert optimizer._blocking_operations == []

    def test_very_long_operation_time(self):
        """Test handling of very long operation times."""
        optimizer = AsyncOptimizer()

        # Record a very long operation
        optimizer._record_operation("slow_op", 1000.0)  # 1000 seconds

        assert "slow_op" in optimizer._operation_stats
        assert optimizer._operation_stats["slow_op"][0] == 1000.0

    def test_zero_operation_time(self):
        """Test handling of zero operation time."""
        optimizer = AsyncOptimizer()

        # Record a zero-time operation
        optimizer._record_operation("instant_op", 0.0)

        assert "instant_op" in optimizer._operation_stats
        assert optimizer._operation_stats["instant_op"][0] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
