"""
AsyncOptimizer - Async I/O optimization and performance monitoring component.

Identifies blocking I/O operations, replaces synchronous operations with async,
implements thread pools for CPU-bound operations, and adds performance monitoring.
"""

import asyncio
import logging
import time
import threading
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, Any, Optional, List, Callable, Awaitable
import aiofiles
import os
import json
from functools import wraps, partial

logger = logging.getLogger(__name__)


class AsyncOptimizer:
    """
    Comprehensive async I/O optimization system with thread pools,
    performance monitoring, and blocking operation detection.
    """

    def __init__(self, max_workers: int = 4, enable_monitoring: bool = True):
        """Initialize the AsyncOptimizer.

        Args:
            max_workers: Maximum number of worker threads
            enable_monitoring: Whether to enable performance monitoring
        """
        self.max_workers = max_workers
        self.enable_monitoring = enable_monitoring

        # Thread pools for different types of operations
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="AsyncOpt")
        self._process_pool = ProcessPoolExecutor(max_workers=max_workers // 2)

        # Performance monitoring
        self._operation_stats: Dict[str, List[float]] = {}
        self._blocking_operations: List[Dict[str, Any]] = []
        self._performance_metrics: Dict[str, Any] = {
            "total_operations": 0,
            "async_operations": 0,
            "threaded_operations": 0,
            "blocking_detected": 0,
            "avg_response_time": 0.0
        }

        # Memory monitoring
        self._memory_stats: Dict[str, List[float]] = {}
        self._memory_thresholds = {
            "warning_mb": 500,  # Warn at 500MB
            "critical_mb": 1000,  # Critical at 1GB
            "cleanup_interval": 300  # Cleanup every 5 minutes
        }
        self._last_cleanup = time.time()
        self._gc_threshold = gc.get_threshold()

        # Known blocking operations to monitor
        self._blocking_patterns = [
            "open(", "read(", "write(", "os.path", "os.stat",
            "json.load", "json.dump", "pickle.load", "pickle.dump",
            "time.sleep", "requests.get", "requests.post"
        ]

        logger.info(f"AsyncOptimizer initialized with {max_workers} workers")

    async def async_file_read(self, file_path: str, mode: str = 'r',
                            encoding: str = 'utf-8') -> str:
        """Asynchronously read a file.

        Args:
            file_path: Path to the file
            mode: File mode
            encoding: File encoding

        Returns:
            File contents as string
        """
        start_time = time.time()

        try:
            async with aiofiles.open(file_path, mode, encoding=encoding) as f:
                content = await f.read()

            self._record_operation("file_read", time.time() - start_time)
            return content

        except Exception as e:
            logger.exception(f"Error reading file {file_path}: {e}")
            raise

    async def async_file_write(self, file_path: str, content: str,
                             mode: str = 'w', encoding: str = 'utf-8') -> None:
        """Asynchronously write to a file.

        Args:
            file_path: Path to the file
            content: Content to write
            mode: File mode
            encoding: File encoding
        """
        start_time = time.time()

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            async with aiofiles.open(file_path, mode, encoding=encoding) as f:
                await f.write(content)

            self._record_operation("file_write", time.time() - start_time)

        except Exception as e:
            logger.exception(f"Error writing to file {file_path}: {e}")
            raise

    async def async_json_load(self, file_path: str) -> Dict[str, Any]:
        """Asynchronously load JSON from file.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON data
        """
        content = await self.async_file_read(file_path)
        return json.loads(content)

    async def async_json_dump(self, data: Dict[str, Any], file_path: str,
                            indent: int = 2) -> None:
        """Asynchronously dump data to JSON file.

        Args:
            data: Data to serialize
            file_path: Path to output file
            indent: JSON indentation
        """
        content = json.dumps(data, indent=indent, default=str)
        await self.async_file_write(file_path, content)

    def run_in_thread(self, func: Callable, *args, **kwargs) -> Awaitable:
        """Run a blocking function in a thread pool.

        Args:
            func: Function to run
            *args, **kwargs: Arguments for the function

        Returns:
            Awaitable that resolves to the function result
        """
        start_time = time.time()

        async def wrapper():
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._thread_pool,
                    partial(func, *args, **kwargs)
                )

                self._record_operation("threaded", time.time() - start_time)
                return result

            except Exception as e:
                logger.exception(f"Error in threaded operation {func.__name__}: {e}")
                raise

        return wrapper()

    def run_in_process(self, func: Callable, *args, **kwargs) -> Awaitable:
        """Run a CPU-intensive function in a process pool.

        Args:
            func: Function to run
            *args, **kwargs: Arguments for the function

        Returns:
            Awaitable that resolves to the function result
        """
        start_time = time.time()

        async def wrapper():
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._process_pool,
                    partial(func, *args, **kwargs)
                )

                self._record_operation("processed", time.time() - start_time)
                return result

            except Exception as e:
                logger.exception(f"Error in processed operation {func.__name__}: {e}")
                raise

        return wrapper()

    def detect_blocking_operations(self, code: str) -> List[str]:
        """Detect potentially blocking operations in code.

        Args:
            code: Code string to analyze

        Returns:
            List of detected blocking patterns
        """
        detected = []
        for pattern in self._blocking_patterns:
            if pattern in code:
                detected.append(pattern)

        if detected:
            self._blocking_operations.append({
                "timestamp": time.time(),
                "patterns": detected,
                "code_sample": code[:200] + "..." if len(code) > 200 else code
            })

        return detected

    def monitor_async_function(self, func: Callable) -> Callable:
        """Decorator to monitor async function performance.

        Args:
            func: Async function to monitor

        Returns:
            Monitored function
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                self._record_operation(f"async_{func.__name__}", execution_time)
                return result

            except Exception as e:
                execution_time = time.time() - start_time
                self._record_operation(f"async_{func.__name__}_error", execution_time)
                raise

        return wrapper

    def monitor_sync_function(self, func: Callable) -> Callable:
        """Decorator to monitor sync function and suggest async conversion.

        Args:
            func: Sync function to monitor

        Returns:
            Monitored function with blocking detection
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                # Check if this might be a blocking operation
                import inspect
                source = inspect.getsource(func)
                blocking_patterns = self.detect_blocking_operations(source)

                if blocking_patterns:
                    logger.warning(f"Detected potentially blocking patterns in {func.__name__}: {blocking_patterns}")

                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                self._record_operation(f"sync_{func.__name__}", execution_time)

                # Warn about slow synchronous operations
                if execution_time > 1.0:  # More than 1 second
                    logger.warning(f"Slow synchronous operation {func.__name__}: {execution_time:.2f}s")

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                self._record_operation(f"sync_{func.__name__}_error", execution_time)
                raise

        return wrapper

    def _record_operation(self, operation_type: str, execution_time: float):
        """Record operation statistics.

        Args:
            operation_type: Type of operation
            execution_time: Time taken to execute
        """
        if operation_type not in self._operation_stats:
            self._operation_stats[operation_type] = []

        self._operation_stats[operation_type].append(execution_time)

        # Keep only last 1000 measurements per operation type
        if len(self._operation_stats[operation_type]) > 1000:
            self._operation_stats[operation_type] = self._operation_stats[operation_type][-1000:]

        # Update global metrics
        self._performance_metrics["total_operations"] += 1

        if operation_type.startswith("async_"):
            self._performance_metrics["async_operations"] += 1
        elif operation_type in ["threaded", "processed"]:
            self._performance_metrics["threaded_operations"] += 1

        # Update average response time
        total_time = sum(sum(times) for times in self._operation_stats.values())
        total_count = sum(len(times) for times in self._operation_stats.values())

        if total_count > 0:
            self._performance_metrics["avg_response_time"] = total_time / total_count

        # Check memory usage and trigger cleanup if needed
        self._check_memory_usage()

    def _check_memory_usage(self):
        """Check current memory usage and trigger cleanup if thresholds exceeded."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            # Record memory usage
            if "memory_usage" not in self._memory_stats:
                self._memory_stats["memory_usage"] = []
            self._memory_stats["memory_usage"].append(memory_mb)

            # Keep only last 100 measurements
            if len(self._memory_stats["memory_usage"]) > 100:
                self._memory_stats["memory_usage"] = self._memory_stats["memory_usage"][-100:]

            # Check thresholds
            warning_threshold = self._memory_thresholds["warning_mb"]
            critical_threshold = self._memory_thresholds["critical_mb"]

            if memory_mb > critical_threshold:
                logger.warning(f"Critical memory usage: {memory_mb:.1f}MB (threshold: {critical_threshold}MB)")
                self._perform_memory_cleanup()
            elif memory_mb > warning_threshold:
                logger.info(f"High memory usage: {memory_mb:.1f}MB (threshold: {warning_threshold}MB)")

            # Periodic cleanup
            current_time = time.time()
            if current_time - self._last_cleanup > self._memory_thresholds["cleanup_interval"]:
                self._perform_periodic_cleanup()
                self._last_cleanup = current_time

        except Exception as e:
            logger.debug(f"Memory monitoring failed: {e}")

    def _perform_memory_cleanup(self):
        """Perform aggressive memory cleanup."""
        try:
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection collected {collected} objects")

            # Clear operation stats if too large
            for op_type in list(self._operation_stats.keys()):
                if len(self._operation_stats[op_type]) > 500:
                    # Keep only recent measurements
                    self._operation_stats[op_type] = self._operation_stats[op_type][-250:]

            # Clear blocking operations if too many
            if len(self._blocking_operations) > 100:
                self._blocking_operations = self._blocking_operations[-50:]

            logger.info("Memory cleanup completed")

        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

    def _perform_periodic_cleanup(self):
        """Perform periodic maintenance cleanup."""
        try:
            # Light garbage collection
            collected = gc.collect(0)  # Only collect generation 0
            if collected > 0:
                logger.debug(f"Periodic GC collected {collected} objects")

            # Clean up old memory stats
            cutoff_time = time.time() - 3600  # 1 hour ago
            for stat_type in list(self._memory_stats.keys()):
                # Remove old entries (assuming timestamps, but we don't have them)
                if len(self._memory_stats[stat_type]) > 50:
                    self._memory_stats[stat_type] = self._memory_stats[stat_type][-25:]

        except Exception as e:
            logger.debug(f"Periodic cleanup failed: {e}")

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report.

        Returns:
            Memory usage statistics and recommendations
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            report = {
                "current_memory_mb": memory_info.rss / 1024 / 1024,
                "virtual_memory_mb": memory_info.vms / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "warning_threshold_mb": self._memory_thresholds["warning_mb"],
                "critical_threshold_mb": self._memory_thresholds["critical_mb"],
                "cleanup_interval_seconds": self._memory_thresholds["cleanup_interval"],
                "last_cleanup_seconds_ago": time.time() - self._last_cleanup
            }

            # Add memory statistics
            if self._memory_stats.get("memory_usage"):
                memory_usage = self._memory_stats["memory_usage"]
                report["memory_stats"] = {
                    "avg_memory_mb": sum(memory_usage) / len(memory_usage),
                    "max_memory_mb": max(memory_usage),
                    "min_memory_mb": min(memory_usage),
                    "samples": len(memory_usage)
                }

            # Add GC statistics
            gc_stats = {}
            for i, count in enumerate(gc.get_count()):
                gc_stats[f"generation_{i}_collections"] = count
            report["gc_stats"] = gc_stats

            # Add recommendations
            current_mb = report["current_memory_mb"]
            if current_mb > self._memory_thresholds["critical_mb"]:
                report["recommendations"] = ["Immediate memory cleanup required", "Consider reducing concurrent operations"]
            elif current_mb > self._memory_thresholds["warning_mb"]:
                report["recommendations"] = ["Monitor memory usage closely", "Consider periodic cleanup"]
            else:
                report["recommendations"] = ["Memory usage normal"]

            return report

        except Exception as e:
            logger.error(f"Failed to generate memory report: {e}")
            return {"error": str(e)}

    def set_memory_thresholds(self, warning_mb: int = None, critical_mb: int = None,
                            cleanup_interval: int = None):
        """Set memory monitoring thresholds.

        Args:
            warning_mb: Warning threshold in MB
            critical_mb: Critical threshold in MB
            cleanup_interval: Cleanup interval in seconds
        """
        if warning_mb is not None:
            self._memory_thresholds["warning_mb"] = warning_mb
        if critical_mb is not None:
            self._memory_thresholds["critical_mb"] = critical_mb
        if cleanup_interval is not None:
            self._memory_thresholds["cleanup_interval"] = cleanup_interval

        logger.info(f"Memory thresholds updated: warning={self._memory_thresholds['warning_mb']}MB, "
                   f"critical={self._memory_thresholds['critical_mb']}MB, "
                   f"cleanup_interval={self._memory_thresholds['cleanup_interval']}s")

    async def batch_async_operations(self, operations: List[Callable]) -> List[Any]:
        """Execute multiple async operations in batch.

        Args:
            operations: List of async callables

        Returns:
            List of operation results
        """
        start_time = time.time()

        try:
            # Create tasks for all operations
            tasks = [op() for op in operations]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            execution_time = time.time() - start_time
            self._record_operation("batch_async", execution_time)

            # Handle exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.exception(f"Error in batch operation {i}: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            logger.exception(f"Error in batch async operations: {e}")
            raise

    def optimize_data_fetching(self, symbols: List[str], fetch_func: Callable) -> Callable:
        """Optimize data fetching with batching and caching.

        Args:
            symbols: List of symbols to fetch
            fetch_func: Function to fetch data for a symbol

        Returns:
            Optimized fetch function
        """
        async def optimized_fetch():
            # Batch fetch operations
            operations = [partial(fetch_func, symbol) for symbol in symbols]

            # Convert to async operations
            async_operations = [self.run_in_thread(op) for op in operations]

            # Execute in batch
            return await self.batch_async_operations(async_operations)

        return optimized_fetch

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report.

        Returns:
            Performance metrics and statistics
        """
        report = self._performance_metrics.copy()

        # Add operation-specific statistics
        operation_stats = {}
        for op_type, times in self._operation_stats.items():
            if times:
                operation_stats[op_type] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_time": sum(times)
                }

        report["operation_stats"] = operation_stats
        report["blocking_operations_detected"] = len(self._blocking_operations)

        # Calculate efficiency metrics
        async_ops = report.get("async_operations", 0)
        total_ops = report.get("total_operations", 0)

        if total_ops > 0:
            report["async_efficiency"] = async_ops / total_ops
        else:
            report["async_efficiency"] = 0.0

        return report

    def get_blocking_operations_report(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get report of detected blocking operations.

        Args:
            limit: Maximum number of operations to return

        Returns:
            List of blocking operation detections
        """
        return self._blocking_operations[-limit:] if self._blocking_operations else []

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on async optimization components.

        Returns:
            Health status of components
        """
        health = {
            "thread_pool_active": not self._thread_pool._shutdown,
            "process_pool_active": not getattr(self._process_pool, '_shutdown', False),
            "threads_alive": len(self._thread_pool._threads) if hasattr(self._thread_pool, '_threads') else 0,
            "processes_alive": self._process_pool._processes if hasattr(self._process_pool, '_processes') else 0,
            "queued_tasks": getattr(self._thread_pool, '_work_queue', None).qsize() if hasattr(self._thread_pool, '_work_queue') else 0
        }

        # Test thread pool with a simple operation
        try:
            test_result = await self.run_in_thread(lambda: 42)
            health["thread_pool_working"] = test_result == 42
        except Exception as e:
            health["thread_pool_working"] = False
            health["thread_pool_error"] = str(e)

        return health

    async def shutdown(self):
        """Shutdown the async optimizer and cleanup resources."""
        logger.info("Shutting down AsyncOptimizer")

        # Cancel any pending tasks in the event loop
        current_task = asyncio.current_task()
        if current_task:
            # Get all tasks and cancel those related to this optimizer
            all_tasks = asyncio.all_tasks()
            optimizer_tasks = [task for task in all_tasks
                             if hasattr(task, 'get_coro') and
                             'AsyncOpt' in str(task.get_coro())]

            for task in optimizer_tasks:
                if not task.done() and task != current_task:
                    task.cancel()

            # Wait for tasks to cancel with timeout
            if optimizer_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*optimizer_tasks, return_exceptions=True),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Some tasks did not cancel within timeout")

        # Shutdown thread pools with proper cleanup
        self._thread_pool.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)

        # Clear references to help GC
        self._operation_stats.clear()
        self._blocking_operations.clear()
        self._performance_metrics.clear()

        logger.info("AsyncOptimizer shutdown complete")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_thread_pool') and not self._thread_pool._shutdown:
                self._thread_pool.shutdown(wait=False)
            if hasattr(self, '_process_pool') and not self._process_pool._shutdown:
                self._process_pool.shutdown(wait=False)
        except:
            pass  # Ignore errors during cleanup


# Global async optimizer instance
_async_optimizer = None

def get_async_optimizer() -> AsyncOptimizer:
    """Get the global async optimizer instance."""
    global _async_optimizer
    if _async_optimizer is None:
        _async_optimizer = AsyncOptimizer()
    return _async_optimizer


# Utility functions for easy integration
async def async_read_file(file_path: str, encoding: str = 'utf-8') -> str:
    """Convenience function for async file reading."""
    optimizer = get_async_optimizer()
    return await optimizer.async_file_read(file_path, encoding=encoding)


async def async_write_file(file_path: str, content: str, encoding: str = 'utf-8') -> None:
    """Convenience function for async file writing."""
    optimizer = get_async_optimizer()
    return await optimizer.async_file_write(file_path, content, encoding=encoding)


def run_async(func: Callable) -> Callable:
    """Decorator to run a blocking function asynchronously in a thread pool."""
    optimizer = get_async_optimizer()

    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await optimizer.run_in_thread(func, *args, **kwargs)

    return wrapper


def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    optimizer = get_async_optimizer()

    if asyncio.iscoroutinefunction(func):
        return optimizer.monitor_async_function(func)
    else:
        return optimizer.monitor_sync_function(func)
