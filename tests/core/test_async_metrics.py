"""
Async Performance Metrics Tests
===============================

Comprehensive test suite for async performance monitoring capabilities.
Tests async metric collection, concurrency, performance benchmarks, and edge cases.
"""

import asyncio
import json
import pytest
import statistics
import time
from pathlib import Path
from typing import Dict, Any, List, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import aiofiles

from core.performance_monitor import (
    RealTimePerformanceMonitor,
    PerformanceAlert,
    PerformanceBaseline,
    AnomalyDetectionResult,
    get_performance_monitor,
    create_performance_monitor,
)
from core.metrics_collector import get_metrics_collector


class AsyncPerformanceMonitor:
    """Test helper class for async metrics collection testing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.async_collectors: List[Callable] = []
        self.is_monitoring = False

    def add_async_collector(self, collector_func: Callable):
        """Add an async metric collector function."""
        self.async_collectors.append(collector_func)

    async def collect_async_metrics(self) -> Dict[str, float]:
        """Collect metrics from async collectors concurrently."""
        if not self.async_collectors:
            return {}

        tasks = [collector() for collector in self.async_collectors]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        metrics = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Async collector {i} failed: {result}")
                continue
            if isinstance(result, dict):
                metrics.update(result)

        return metrics

    async def start_monitoring(self):
        """Mock start monitoring."""
        self.is_monitoring = True

    async def stop_monitoring(self):
        """Mock stop monitoring."""
        self.is_monitoring = False

    def monitor_context(self):
        """Mock async context manager."""
        return self

    async def __aenter__(self):
        await self.start_monitoring()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_monitoring()


@pytest.fixture
def async_monitor():
    """Fixture for async performance monitor."""
    config = {
        "monitoring_interval": 1.0,  # Fast for testing
        "baseline_window": 60.0,     # 1 minute for testing
        "anomaly_detection": False,  # Disable for cleaner tests
        "alerting": False,
    }
    monitor = AsyncPerformanceMonitor(config)
    return monitor


@pytest.fixture
def real_monitor():
    """Fixture for real performance monitor."""
    config = {
        "monitoring_interval": 0.1,  # Very fast for testing
        "baseline_window": 10.0,     # Short window
        "anomaly_detection": False,
        "alerting": False,
    }
    monitor = RealTimePerformanceMonitor(config)
    return monitor


@pytest.fixture
def stress_monitor():
    """Fixture for stress testing monitor."""
    config = {
        "monitoring_interval": 0.1,
        "baseline_window": 5.0,
        "anomaly_detection": True,
        "alerting": True,
    }
    monitor = AsyncPerformanceMonitor(config)
    return monitor

@pytest.mark.asyncio
class TestAsyncMetricsCollection:
    """Test async metrics collection functionality."""

    @pytest.mark.timeout(30)
    async def test_async_metrics_collection_basic(self, async_monitor):
        """Test basic async metrics collection."""
        # Add a simple async collector
        async def simple_collector():
            await asyncio.sleep(0.1)  # Simulate async work
            return {"test_metric": 42.0, "async_collector_status": 1.0}

        async_monitor.add_async_collector(simple_collector)

        # Collect metrics
        metrics = await async_monitor.collect_async_metrics()

        assert "test_metric" in metrics
        assert metrics["test_metric"] == 42.0
        assert "async_collector_status" in metrics
        assert metrics["async_collector_status"] == 1.0

    @pytest.mark.timeout(30)
    async def test_concurrent_async_collectors(self, async_monitor):
        """Test concurrent execution of multiple async collectors."""
        async def slow_collector(name: str, value: float, delay: float):
            await asyncio.sleep(delay)
            return {f"{name}_metric": value, f"{name}_delay": delay}

        # Add multiple collectors with different delays
        async_monitor.add_async_collector(
            lambda: slow_collector("fast", 1.0, 0.1)
        )
        async_monitor.add_async_collector(
            lambda: slow_collector("medium", 2.0, 0.2)
        )
        async_monitor.add_async_collector(
            lambda: slow_collector("slow", 3.0, 0.3)
        )

        start_time = time.time()
        metrics = await async_monitor.collect_async_metrics()
        end_time = time.time()

        # All metrics should be collected
        assert "fast_metric" in metrics
        assert "medium_metric" in metrics
        assert "slow_metric" in metrics
        assert metrics["fast_metric"] == 1.0
        assert metrics["medium_metric"] == 2.0
        assert metrics["slow_metric"] == 3.0

        # Should complete in ~0.3s (max delay), not ~0.6s (sum of delays)
        execution_time = end_time - start_time
        assert execution_time < 0.5  # Allow some overhead

    @pytest.mark.timeout(30)
    async def test_async_collector_error_handling(self, async_monitor):
        """Test graceful handling of async collector failures."""
        async def failing_collector():
            await asyncio.sleep(0.1)
            raise ValueError("Simulated collector failure")

        async def working_collector():
            await asyncio.sleep(0.1)
            return {"working_metric": 100.0}

        async_monitor.add_async_collector(failing_collector)
        async_monitor.add_async_collector(working_collector)

        # Should not raise exception, but log the error
        metrics = await async_monitor.collect_async_metrics()

        # Working collector should still provide metrics
        assert "working_metric" in metrics
        assert metrics["working_metric"] == 100.0

    @pytest.mark.timeout(30)
    async def test_async_metrics_timeout_protection(self):
        """Test timeout protection for hanging async collectors."""
        config = {
            "monitoring_interval": 1.0,
            "baseline_window": 60.0,
            "anomaly_detection": False,
            "alerting": False,
        }
        monitor = AsyncPerformanceMonitor(config)

        # Create hanging collector
        async def hanging_collector():
            await asyncio.sleep(60)  # Would hang without timeout
            return {"hanging_metric": 1}

        monitor.add_async_collector(hanging_collector)

        # Should timeout, not hang
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(monitor.collect_async_metrics(), timeout=5.0)


@pytest.mark.asyncio
class TestConcurrentMetricsUpdates:
    """Test concurrent metrics updates and accuracy."""

    @pytest.mark.timeout(30)
    async def test_concurrent_metric_updates_accuracy(self, real_monitor):
        """Test that concurrent updates maintain metric accuracy."""
        update_count = 100
        concurrent_tasks = 10

        async def update_metrics(task_id: int):
            """Simulate concurrent metric updates."""
            for i in range(update_count):
                metrics = {
                    f"task_{task_id}_counter": float(i + 1),
                    f"task_{task_id}_value": float(task_id * 100 + i),
                    "_timestamp": time.time()
                }
                await real_monitor._update_baselines_with_metrics(metrics)
                await asyncio.sleep(0.001)  # Small delay to allow interleaving

        # Run concurrent updates
        tasks = [update_metrics(i) for i in range(concurrent_tasks)]
        await asyncio.gather(*tasks)

        # Verify all updates were recorded
        for task_id in range(concurrent_tasks):
            metric_name = f"task_{task_id}_counter"
            assert metric_name in real_monitor.performance_metrics

            values = [entry["value"] for entry in real_monitor.performance_metrics[metric_name]]
            # Each task updates its own metric, so each metric should have update_count values
            assert len(values) == update_count

            # Values should be in range [1, update_count] for this task
            task_values = [v for v in values if 1 <= v <= update_count]
            assert len(task_values) == update_count  # All values should be in range

    @pytest.mark.timeout(30)
    async def test_concurrent_baseline_calculations(self, real_monitor):
        """Test concurrent baseline calculations."""
        # Add some test data by simulating metrics collection
        metric_name = "test_concurrent_metric"
        current_time = time.time()

        # Add historical data directly to baseline_history (what _update_baselines reads)
        for i in range(50):
            timestamp = current_time - (50 - i) * 1.0  # 1 second intervals
            value = 100.0 + np.random.normal(0, 5)  # Base value with noise
            real_monitor.baseline_history[metric_name].append((timestamp, value))

        # Start monitoring to enable baseline calculations
        await real_monitor.start_monitoring()

        # Run multiple concurrent baseline updates
        tasks = [real_monitor._update_baselines() for _ in range(5)]
        await asyncio.gather(*tasks)

        # Baseline should be calculated correctly
        assert metric_name in real_monitor.baselines
        baseline = real_monitor.baselines[metric_name]

        assert baseline.sample_count >= 10  # Minimum samples
        assert baseline.mean > 90 and baseline.mean < 110  # Reasonable range
        assert baseline.std >= 0

        # Clean up
        await real_monitor.stop_monitoring()

    @pytest.mark.timeout(30)
    async def test_async_context_manager(self, real_monitor):
        """Test async context manager functionality."""
        async with real_monitor.monitor_context():
            assert real_monitor.is_monitoring

            # Perform some operations
            status = await real_monitor.get_performance_status()
            assert status["is_monitoring"] is True

        # Should be stopped after context exit
        assert not real_monitor.is_monitoring


@pytest.mark.asyncio
class TestPerformanceBenchmarks:
    """Test performance benchmarks and comparisons."""

    @pytest.mark.timeout(60)
    async def test_async_vs_sync_performance_comparison(self, async_monitor):
        """Compare performance of async vs sync metrics collection."""
        # Setup sync-style collection (simulated)
        async def sync_style_collection():
            """Simulate sync-style collection with asyncio.sleep(0) for fairness."""
            metrics = {}

            # Simulate sequential collection
            await asyncio.sleep(0.01)  # profiler
            metrics.update({"profiler_metric": 1.0})

            await asyncio.sleep(0.01)  # system metrics
            metrics.update({"system_metric": 2.0})

            await asyncio.sleep(0.01)  # custom metrics
            metrics.update({"custom_metric": 3.0})

            return metrics

        # Setup async collection
        async def async_collector_1():
            await asyncio.sleep(0.01)
            return {"async_profiler_metric": 1.0}

        async def async_collector_2():
            await asyncio.sleep(0.01)
            return {"async_system_metric": 2.0}

        async def async_collector_3():
            await asyncio.sleep(0.01)
            return {"async_custom_metric": 3.0}

        async_monitor.add_async_collector(async_collector_1)
        async_monitor.add_async_collector(async_collector_2)
        async_monitor.add_async_collector(async_collector_3)

        # Benchmark sync style
        sync_times = []
        for _ in range(10):
            start = time.time()
            await sync_style_collection()
            sync_times.append(time.time() - start)

        # Benchmark async style
        async_times = []
        for _ in range(10):
            start = time.time()
            await async_monitor.collect_async_metrics()
            async_times.append(time.time() - start)

        sync_avg = statistics.mean(sync_times)
        async_avg = statistics.mean(async_times)

        print(".4f")
        print(".4f")

        # Async should be faster (at least not significantly slower)
        assert async_avg <= sync_avg * 1.5  # Allow 50% overhead

    @pytest.mark.timeout(60)
    async def test_scalability_with_concurrent_collectors(self, async_monitor):
        """Test scalability as number of concurrent collectors increases."""
        async def scalable_collector(collector_id: int, base_delay: float = 0.01):
            await asyncio.sleep(base_delay)
            return {f"collector_{collector_id}_metric": float(collector_id)}

        scalability_results = {}

        for num_collectors in [1, 5, 10, 20]:
            # Clear previous collectors
            async_monitor.async_collectors.clear()

            # Add collectors
            for i in range(num_collectors):
                async_monitor.add_async_collector(
                    lambda cid=i: scalable_collector(cid)
                )

            # Benchmark
            times = []
            for _ in range(5):  # Multiple runs for averaging
                start = time.time()
                metrics = await async_monitor.collect_async_metrics()
                times.append(time.time() - start)

                # Verify all collectors ran
                assert len(metrics) == num_collectors

            avg_time = statistics.mean(times)
            scalability_results[num_collectors] = avg_time

            print(".4f")

        # Performance should degrade gracefully, not exponentially
        # Time for 20 collectors should be less than 4x time for 5 collectors
        if 20 in scalability_results and 5 in scalability_results:
            ratio = scalability_results[20] / scalability_results[5]
            assert ratio < 4.0, f"Performance degraded too much: {ratio:.2f}x slower"

    @pytest.mark.timeout(30)
    async def test_memory_efficiency_under_load(self, async_monitor):
        """Test memory efficiency during high-frequency async operations."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Add memory-intensive async collectors
        async def memory_intensive_collector():
            # Allocate some memory temporarily
            data = [i for i in range(10000)]  # ~40KB
            await asyncio.sleep(0.01)
            result = {"memory_test": sum(data) / len(data)}
            del data  # Explicit cleanup
            return result

        for _ in range(10):
            async_monitor.add_async_collector(memory_intensive_collector)

        # Run high-frequency collections
        for _ in range(50):
            await async_monitor.collect_async_metrics()
            await asyncio.sleep(0.01)  # Small delay between collections

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        memory_mb = memory_increase / (1024 * 1024)

        print(".2f")

        # Memory increase should be reasonable (< 50MB)
        assert memory_mb < 50, f"Excessive memory usage: {memory_mb:.2f}MB"


@pytest.mark.asyncio
class TestEdgeCasesAndStress:
    """Test edge cases and stress conditions."""

    @pytest.mark.timeout(60)
    async def test_graceful_degradation_under_system_stress(self, stress_monitor):
        """Test graceful degradation when async operations fail under stress."""
        failure_count = 0

        async def stress_collector():
            nonlocal failure_count
            try:
                # Simulate random failures under stress
                if np.random.random() < 0.3:  # 30% failure rate
                    raise ConnectionError("Simulated network failure")

                await asyncio.sleep(np.random.uniform(0.01, 0.1))
                return {"stress_metric": np.random.uniform(0, 100)}
            except Exception:
                failure_count += 1
                raise

        # Add multiple stressed collectors
        for _ in range(20):
            stress_monitor.add_async_collector(stress_collector)

        # Run under stress
        successful_runs = 0
        total_runs = 10

        for _ in range(total_runs):
            try:
                metrics = await asyncio.wait_for(
                    stress_monitor.collect_async_metrics(),
                    timeout=2.0  # Generous timeout
                )
                if metrics:  # Some metrics collected despite failures
                    successful_runs += 1
            except asyncio.TimeoutError:
                continue  # Expected under stress

        # Should have some successful runs despite failures
        success_rate = successful_runs / total_runs
        print(".2f")

        # At least 50% success rate even under stress
        assert success_rate >= 0.5, f"Too many failures under stress: {success_rate:.2f}"

    @pytest.mark.timeout(30)
    async def test_concurrent_monitoring_loops(self, stress_monitor):
        """Test multiple monitoring loops running concurrently."""
        config = {
            "monitoring_interval": 0.1,
            "baseline_window": 5.0,
            "anomaly_detection": True,
            "alerting": True,
        }
        monitor2 = AsyncPerformanceMonitor(config)

        try:
            # Start both monitors
            async with stress_monitor.monitor_context():
                async with monitor2.monitor_context():
                    # Both should be monitoring
                    assert stress_monitor.is_monitoring
                    assert monitor2.is_monitoring

                    # Run for a short period
                    await asyncio.sleep(2.0)

                    # Both should still be running
                    assert stress_monitor.is_monitoring
                    assert monitor2.is_monitoring

        finally:
            if monitor2.is_monitoring:
                await monitor2.stop_monitoring()

    @pytest.mark.timeout(30)
    async def test_async_file_operations_under_load(self):
        """Test async file operations under concurrent load."""
        # Create temporary data directory
        temp_dir = Path("temp_test_data")
        temp_dir.mkdir(exist_ok=True)

        try:
            # Simulate concurrent file operations
            async def file_operation_worker(worker_id: int):
                file_path = temp_dir / f"worker_{worker_id}.json"

                # Write data
                data = {"worker": worker_id, "timestamp": time.time(), "metrics": []}
                for i in range(10):
                    data["metrics"].append({"value": i, "time": time.time()})

                async with aiofiles.open(file_path, "w") as f:
                    await f.write(json.dumps(data, indent=2))

                # Read it back
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    read_data = json.loads(content)

                return read_data

            # Run multiple file operations concurrently
            tasks = [file_operation_worker(i) for i in range(10)]
            results = await asyncio.gather(*tasks)

            # Verify all operations succeeded
            assert len(results) == 10
            for i, result in enumerate(results):
                assert result["worker"] == i
                assert len(result["metrics"]) == 10

        finally:
            # Cleanup
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


@pytest.fixture
def global_monitor():
    """Fixture for global performance monitor."""
    monitor = get_performance_monitor()
    return monitor


@pytest.mark.asyncio
class TestIntegrationWithExistingSystem:
    """Test integration with existing performance monitoring system."""

    @pytest.mark.timeout(30)
    async def test_global_monitor_async_operations(self, global_monitor):
        """Test that global monitor supports async operations."""
        # Should be able to start monitoring asynchronously
        await global_monitor.start_monitoring()
        assert global_monitor.is_monitoring

        # Should be able to get status
        status = await global_monitor.get_performance_status()
        assert isinstance(status, dict)
        assert "is_monitoring" in status

        # Should be able to stop monitoring
        await global_monitor.stop_monitoring()
        assert not global_monitor.is_monitoring

    @pytest.mark.timeout(30)
    async def test_async_context_manager_global_instance(self, global_monitor):
        """Test async context manager with global instance."""
        async with global_monitor.monitor_context():
            assert global_monitor.is_monitoring

            # Should be able to perform async operations
            status = await global_monitor.get_performance_status()
            assert status["is_monitoring"] is True

        assert not global_monitor.is_monitoring


# Performance regression tests
@pytest.mark.asyncio
@pytest.mark.timeout(120)  # Longer timeout for performance tests
class TestPerformanceRegression:
    """Test for performance regressions in async operations."""

    async def test_async_collection_performance_regression(self):
        """Test that async collection performance doesn't regress."""
        monitor = AsyncPerformanceMonitor({
            "monitoring_interval": 0.1,
            "baseline_window": 10.0,
            "anomaly_detection": False,
            "alerting": False,
        })

        # Add realistic collectors
        async def realistic_collector(collector_id: int):
            # Simulate real work: some computation + async I/O
            data = [i ** 2 for i in range(100)]  # Computation
            await asyncio.sleep(0.001)  # Minimal async delay
            return {f"computed_metric_{collector_id}": sum(data) / len(data)}

        for i in range(5):
            monitor.add_async_collector(lambda cid=i: realistic_collector(cid))

        # Benchmark performance
        times = []
        for _ in range(20):
            start = time.time()
            metrics = await monitor.collect_async_metrics()
            times.append(time.time() - start)

            # Verify metrics collected
            assert len(metrics) == 5

        avg_time = statistics.mean(times)
        p95_time = np.percentile(times, 95)

        print(".4f")
        print(".4f")

        # Performance thresholds (adjust based on system capabilities)
        assert avg_time < 0.1, f"Average collection time too slow: {avg_time:.4f}s"
        assert p95_time < 0.2, f"P95 collection time too slow: {p95_time:.4f}s"

        if monitor.is_monitoring:
            await monitor.stop_monitoring()
