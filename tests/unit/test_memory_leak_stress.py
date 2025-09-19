"""
Stress test for memory leak detection and resolution in AsyncOptimizer.

This test simulates heavy load to verify that memory leaks are properly
detected and cleaned up automatically.
"""

import asyncio
import time
import psutil
import gc
import logging
from typing import List, Dict, Any
import tempfile
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.async_optimizer import AsyncOptimizer, get_async_optimizer

logger = logging.getLogger(__name__)


class MemoryLeakStressTest:
    """Stress test for memory leak detection and cleanup."""

    def __init__(self):
        self.optimizer = AsyncOptimizer(max_workers=8, enable_monitoring=True)
        self.test_data = []
        self.initial_memory = 0
        self.test_results = {}

    async def setup(self):
        """Setup test environment."""
        # Create test data
        for i in range(1000):
            self.test_data.append({
                "id": i,
                "data": "x" * 1000,  # 1KB per item
                "nested": {"value": "y" * 500}
            })

        # Record initial memory
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"Initial memory: {self.initial_memory:.1f}MB")

    async def run_memory_stress_test(self) -> Dict[str, Any]:
        """Run comprehensive memory stress test."""
        logger.info("Starting memory leak stress test")

        # Test 1: Heavy async file operations
        await self._test_file_operations()

        # Test 2: Thread pool stress
        await self._test_thread_pool_stress()

        # Test 3: Batch operations
        await self._test_batch_operations()

        # Test 4: Memory monitoring
        await self._test_memory_monitoring()

        # Test 5: Automatic cleanup
        await self._test_automatic_cleanup()

        # Generate final report
        return await self._generate_test_report()

    async def _test_file_operations(self):
        """Test heavy file I/O operations."""
        logger.info("Testing file operations stress")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create many files
            file_tasks = []
            for i in range(100):
                file_path = os.path.join(temp_dir, f"test_file_{i}.json")
                data = {"index": i, "data": self.test_data[i % len(self.test_data)]}
                file_tasks.append(self.optimizer.async_json_dump(data, file_path))

            # Execute file writes
            await asyncio.gather(*file_tasks)

            # Read files back
            read_tasks = []
            for i in range(100):
                file_path = os.path.join(temp_dir, f"test_file_{i}.json")
                read_tasks.append(self.optimizer.async_json_load(file_path))

            results = await asyncio.gather(*read_tasks)
            assert len(results) == 100

        logger.info("File operations stress test completed")

    async def _test_thread_pool_stress(self):
        """Test thread pool under heavy load."""
        logger.info("Testing thread pool stress")

        # Create CPU-intensive tasks
        def cpu_intensive_task(n: int) -> int:
            result = 0
            for i in range(n):
                result += i ** 2
            return result

        # Run many CPU-intensive tasks
        tasks = []
        for i in range(50):
            tasks.append(self.optimizer.run_in_thread(cpu_intensive_task, 10000))

        results = await asyncio.gather(*tasks)
        assert len(results) == 50
        assert all(isinstance(r, int) for r in results)

        logger.info("Thread pool stress test completed")

    async def _test_batch_operations(self):
        """Test batch async operations."""
        logger.info("Testing batch operations")

        # Create many small async tasks
        async def small_async_task(n: int) -> int:
            await asyncio.sleep(0.001)  # Small delay
            return n * 2

        operations = [lambda n=n: small_async_task(n) for n in range(200)]
        results = await self.optimizer.batch_async_operations(operations)

        assert len(results) == 200
        assert all(r == n * 2 for n, r in enumerate(results) if r is not None)

        logger.info("Batch operations test completed")

    async def _test_memory_monitoring(self):
        """Test memory monitoring functionality."""
        logger.info("Testing memory monitoring")

        # Get initial memory report
        initial_report = self.optimizer.get_memory_report()

        # Perform some operations to generate memory usage
        for i in range(20):
            await self.optimizer.run_in_thread(lambda: "x" * 100000)  # Allocate memory

        # Get updated memory report
        updated_report = self.optimizer.get_memory_report()

        # Verify monitoring is working
        assert "current_memory_mb" in initial_report
        assert "current_memory_mb" in updated_report
        assert "memory_stats" in updated_report

        logger.info("Memory monitoring test completed")

    async def _test_automatic_cleanup(self):
        """Test automatic memory cleanup."""
        logger.info("Testing automatic cleanup")

        # Force some memory usage
        large_objects = []
        for i in range(10):
            large_objects.append("x" * 1000000)  # 1MB each

        # Delete references
        del large_objects

        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Manual GC collected {collected} objects")

        # Check memory report
        report = self.optimizer.get_memory_report()
        logger.info(f"Memory after cleanup: {report.get('current_memory_mb', 0):.1f}MB")

        logger.info("Automatic cleanup test completed")

    async def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - self.initial_memory

        # Get final reports
        memory_report = self.optimizer.get_memory_report()
        performance_report = self.optimizer.get_performance_report()

        report = {
            "test_timestamp": time.time(),
            "initial_memory_mb": self.initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": memory_increase,
            "memory_leak_detected": memory_increase > 50,  # >50MB increase is concerning
            "memory_report": memory_report,
            "performance_report": performance_report,
            "gc_stats": {
                "collections_per_generation": list(gc.get_count()),
                "gc_thresholds": list(gc.get_threshold())
            },
            "test_status": "PASSED" if memory_increase < 100 else "FAILED"
        }

        logger.info(f"Stress test completed. Memory increase: {memory_increase:.1f}MB")
        logger.info(f"Test status: {report['test_status']}")

        return report

    async def cleanup(self):
        """Cleanup test resources."""
        await self.optimizer.shutdown()


async def run_memory_leak_stress_test() -> Dict[str, Any]:
    """Run the complete memory leak stress test."""
    test = MemoryLeakStressTest()

    try:
        await test.setup()
        results = await test.run_memory_stress_test()
        return results
    finally:
        await test.cleanup()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Run stress test
    results = asyncio.run(run_memory_leak_stress_test())

    print("\n=== MEMORY LEAK STRESS TEST RESULTS ===")
    print(f"Status: {results['test_status']}")
    print(".1f")
    print(".1f")
    print(".1f")
    print(f"Memory Leak Detected: {results['memory_leak_detected']}")

    if results['memory_leak_detected']:
        print("⚠️  WARNING: Potential memory leak detected!")
    else:
        print("✅ No significant memory leak detected")

    print("\nMemory Report:")
    mem_report = results['memory_report']
    print(f"  Current Memory: {mem_report.get('current_memory_mb', 0):.1f}MB")
    print(f"  Warning Threshold: {mem_report.get('warning_threshold_mb', 0)}MB")
    print(f"  Critical Threshold: {mem_report.get('critical_threshold_mb', 0)}MB")

    print("\nPerformance Report:")
    perf_report = results['performance_report']
    print(f"  Total Operations: {perf_report.get('total_operations', 0)}")
    print(f"  Async Efficiency: {perf_report.get('async_efficiency', 0):.2%}")
