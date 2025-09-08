"""
AsyncOptimizer - Async I/O optimization and performance monitoring component.

Identifies blocking I/O operations, replaces synchronous operations with async,
implements thread pools for CPU-bound operations, and adds performance monitoring.
"""

import asyncio
import logging
import time
import threading
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
        self._process_pool = ProcessPoolExecutor(max_workers=max_workers // 2, max_tasks_per_child=50)

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
            "process_pool_active": not self._process_pool._shutdown,
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

        # Shutdown thread pools
        self._thread_pool.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)

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
