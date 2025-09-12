"""
Performance Profiler for Trading Framework
==========================================

This module provides comprehensive performance profiling capabilities for the trading framework,
including function-level timing, memory usage tracking, I/O operation monitoring, and garbage
collection impact analysis.

Key Features:
- Multi-level profiling (function, memory, I/O, GC)
- Real-time performance monitoring
- Statistical analysis and anomaly detection
- Hierarchical performance reports
- Low-overhead sampling profiler
"""

import time
import psutil
import gc
import threading
import functools
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import logging
import tracemalloc
import cProfile
import pstats
import io
from contextlib import contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics data."""
    function_name: str
    execution_time: float
    memory_usage: int
    memory_peak: int
    cpu_percent: float
    io_read_bytes: int
    io_write_bytes: int
    gc_collections: Dict[int, int]
    timestamp: float
    call_count: int = 1
    avg_execution_time: float = 0.0
    std_execution_time: float = 0.0


@dataclass
class ProfilingSession:
    """Represents a profiling session with collected metrics."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    metrics: List[PerformanceMetrics] = field(default_factory=list)
    memory_snapshots: List[Dict] = field(default_factory=list)
    gc_stats: List[Dict] = field(default_factory=list)
    io_stats: Dict[str, int] = field(default_factory=dict)


class PerformanceProfiler:
    """
    Advanced performance profiler for trading framework.

    Provides multi-level profiling capabilities including:
    - Function-level timing with statistical analysis
    - Memory usage tracking and leak detection
    - I/O operation monitoring
    - Garbage collection impact measurement
    - Real-time performance monitoring
    """

    def __init__(self, sampling_interval: float = 0.01, max_history: int = 1000):
        """
        Initialize the performance profiler.

        Args:
            sampling_interval: Time interval for sampling profiler (seconds)
            max_history: Maximum number of historical metrics to keep
        """
        self.sampling_interval = sampling_interval
        self.max_history = max_history

        # Profiling state
        self.is_profiling = True  # Always enabled for context manager usage
        self.current_session: Optional[ProfilingSession] = None
        self.sessions: Dict[str, ProfilingSession] = {}

        # Historical data
        self.metrics_history: deque = deque(maxlen=max_history)
        self.baseline_metrics: Dict[str, Dict] = {}

        # Monitoring threads
        self.monitoring_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Memory tracking
        self.memory_tracking = False
        self.memory_snapshots: List = []

        # Statistical baselines
        self.performance_baselines: Dict[str, Dict] = {}
        self.anomaly_thresholds: Dict[str, float] = {}

        # Locks for thread safety
        self._lock = threading.RLock()

        logger.info("Performance Profiler initialized")

    def start_profiling(self, session_id: str) -> str:
        """
        Start a new profiling session.

        Args:
            session_id: Unique identifier for the profiling session

        Returns:
            Session ID
        """
        with self._lock:
            if self.is_profiling:
                self.stop_profiling()

            self.current_session = ProfilingSession(
                session_id=session_id,
                start_time=time.time()
            )
            self.is_profiling = True

            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()

            # Start memory tracking if enabled
            if self.memory_tracking:
                tracemalloc.start()
                self.memory_snapshots = []

            logger.info(f"Started profiling session: {session_id}")
            return session_id

    def stop_profiling(self) -> Optional[ProfilingSession]:
        """
        Stop the current profiling session.

        Returns:
            Completed profiling session
        """
        with self._lock:
            if not self.is_profiling:
                return None

            self.is_profiling = False
            if self.current_session:
                self.current_session.end_time = time.time()
                session = self.current_session
                self.sessions[session.session_id] = session
                self.current_session = None

                # Stop memory tracking
                if self.memory_tracking:
                    tracemalloc.stop()

                # Stop monitoring thread
                if self.monitoring_thread and self.monitoring_thread.is_alive():
                    self.monitoring_thread.join(timeout=1.0)

                logger.info(f"Stopped profiling session: {session.session_id}")
                return session
            return None

    def _monitoring_loop(self):
        """Background monitoring loop for real-time metrics collection."""
        process = psutil.Process()
        last_io = process.io_counters()

        while self.is_profiling:
            try:
                current_time = time.time()

                # Collect system metrics
                cpu_percent = process.cpu_percent(interval=None)
                memory_info = process.memory_info()
                current_io = process.io_counters()

                # Calculate I/O deltas
                io_read_delta = current_io.read_bytes - last_io.read_bytes if last_io else 0
                io_write_delta = current_io.write_bytes - last_io.write_bytes if last_io else 0
                last_io = current_io

                # Collect GC stats
                gc_stats = {}
                for gen in range(3):
                    gc_stats[gen] = gc.get_count()[gen]

                # Store metrics in current session
                if self.current_session:
                    with self._lock:
                        # This would be populated by profiled functions
                        pass

                # Memory snapshot if tracking enabled
                if self.memory_tracking:
                    snapshot = tracemalloc.take_snapshot()
                    self.memory_snapshots.append(snapshot)

                time.sleep(self.sampling_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                break

    @contextmanager
    def profile_function(self, function_name: str):
        """
        Context manager for profiling a function.

        Args:
            function_name: Name of the function being profiled
        """
        if not self.is_profiling:
            yield
            return

        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss
        start_gc = gc.get_count()
        start_io = process.io_counters()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = process.memory_info().rss
            end_gc = gc.get_count()
            end_io = process.io_counters()

            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            memory_peak = max(start_memory, end_memory)

            io_read_delta = end_io.read_bytes - start_io.read_bytes
            io_write_delta = end_io.write_bytes - start_io.write_bytes

            gc_collections = {}
            for gen in range(3):
                # Ensure we don't get negative values
                gc_collections[gen] = max(0, end_gc[gen] - start_gc[gen])

            # Create metrics object
            metrics = PerformanceMetrics(
                function_name=function_name,
                execution_time=execution_time,
                memory_usage=memory_delta,
                memory_peak=memory_peak,
                cpu_percent=process.cpu_percent(interval=None),
                io_read_bytes=io_read_delta,
                io_write_bytes=io_write_delta,
                gc_collections=gc_collections,
                timestamp=end_time
            )

            # Store metrics
            with self._lock:
                if self.current_session:
                    self.current_session.metrics.append(metrics)
                self.metrics_history.append(metrics)

                # Update statistical baselines
                self._update_baselines(function_name, execution_time)

    def profile_method(self, method_name: Optional[str] = None):
        """
        Decorator for profiling class methods.

        Args:
            method_name: Optional custom name for the method
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = method_name or f"{func.__qualname__}"
                with self.profile_function(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def _update_baselines(self, function_name: str, execution_time: float):
        """Update statistical baselines for a function."""
        if function_name not in self.baseline_metrics:
            self.baseline_metrics[function_name] = {
                'times': [],
                'mean': 0.0,
                'std': 0.0,
                'count': 0
            }

        baseline = self.baseline_metrics[function_name]
        baseline['times'].append(execution_time)
        baseline['count'] += 1

        # Keep only recent measurements
        if len(baseline['times']) > 100:
            baseline['times'] = baseline['times'][-100:]

        # Update statistics
        if len(baseline['times']) > 1:
            baseline['mean'] = statistics.mean(baseline['times'])
            baseline['std'] = statistics.stdev(baseline['times'])

    def detect_anomalies(self, function_name: str, current_time: float) -> bool:
        """
        Detect performance anomalies for a function.

        Args:
            function_name: Name of the function to check
            current_time: Current execution time

        Returns:
            True if anomaly detected
        """
        if function_name not in self.baseline_metrics:
            return False

        baseline = self.baseline_metrics[function_name]
        if baseline['count'] < 10:  # Need minimum samples
            return False

        threshold = self.anomaly_thresholds.get(function_name, 3.0)  # 3-sigma rule
        z_score = abs(current_time - baseline['mean']) / baseline['std']

        return z_score > threshold

    def generate_report(self, session_id: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive performance report.

        Args:
            session_id: Optional session ID to report on

        Returns:
            Performance report dictionary
        """
        with self._lock:
            if session_id:
                session = self.sessions.get(session_id)
                if not session:
                    return {"error": f"Session {session_id} not found"}
                metrics = session.metrics
            else:
                metrics = list(self.metrics_history)

            if not metrics:
                return {"error": "No metrics available"}

            # Group metrics by function
            function_metrics = defaultdict(list)
            for metric in metrics:
                function_metrics[metric.function_name].append(metric)

            # Generate report
            report = {
                "summary": {
                    "total_functions": len(function_metrics),
                    "total_measurements": len(metrics),
                    "time_range": {
                        "start": min(m.timestamp for m in metrics),
                        "end": max(m.timestamp for m in metrics)
                    }
                },
                "functions": {}
            }

            for func_name, func_metrics in function_metrics.items():
                execution_times = [m.execution_time for m in func_metrics]
                memory_usages = [m.memory_usage for m in func_metrics]

                report["functions"][func_name] = {
                    "call_count": len(func_metrics),
                    "execution_time": {
                        "total": sum(execution_times),
                        "mean": statistics.mean(execution_times),
                        "median": statistics.median(execution_times),
                        "min": min(execution_times),
                        "max": max(execution_times),
                        "std": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                    },
                    "memory_usage": {
                        "total": sum(memory_usages),
                        "mean": statistics.mean(memory_usages),
                        "peak": max(m.memory_peak for m in func_metrics)
                    },
                    "performance_trends": self._analyze_trends(func_metrics)
                }

            return report

    def _analyze_trends(self, metrics: List[PerformanceMetrics]) -> Dict:
        """Analyze performance trends for a set of metrics."""
        if len(metrics) < 2:
            return {"trend": "insufficient_data"}

        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        times = [m.execution_time for m in sorted_metrics]

        # Simple linear trend
        n = len(times)
        x = list(range(n))
        slope = np.polyfit(x, times, 1)[0]

        return {
            "trend": "improving" if slope < 0 else "degrading",
            "slope": slope,
            "volatility": statistics.stdev(times) if n > 1 else 0
        }

    def get_hotspots(self, top_n: int = 10) -> List[Dict]:
        """
        Identify performance hotspots.

        Args:
            top_n: Number of top hotspots to return

        Returns:
            List of hotspot dictionaries
        """
        with self._lock:
            if not self.metrics_history:
                return []

            # Aggregate by function
            function_stats = defaultdict(lambda: {
                'total_time': 0.0,
                'call_count': 0,
                'avg_time': 0.0,
                'memory_total': 0
            })

            for metric in self.metrics_history:
                stats = function_stats[metric.function_name]
                stats['total_time'] += metric.execution_time
                stats['call_count'] += 1
                stats['memory_total'] += metric.memory_usage

            # Calculate averages
            hotspots = []
            for func_name, stats in function_stats.items():
                stats['avg_time'] = stats['total_time'] / stats['call_count']
                hotspots.append({
                    'function': func_name,
                    'total_time': stats['total_time'],
                    'avg_time': stats['avg_time'],
                    'call_count': stats['call_count'],
                    'memory_total': stats['memory_total']
                })

            # Sort by total time (descending)
            hotspots.sort(key=lambda x: x['total_time'], reverse=True)

            return hotspots[:top_n]

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            # Fallback to tracemalloc if psutil is not available
            try:
                current, _ = tracemalloc.get_traced_memory()
                return current
            except Exception:
                return 0

    def enable_memory_tracking(self):
        """Enable detailed memory tracking."""
        self.memory_tracking = True
        logger.info("Memory tracking enabled")

    def disable_memory_tracking(self):
        """Disable detailed memory tracking."""
        self.memory_tracking = False
        logger.info("Memory tracking disabled")

    def get_memory_report(self) -> Dict:
        """Generate memory usage report."""
        if not self.memory_snapshots:
            return {"error": "No memory snapshots available"}

        # Analyze memory snapshots
        current, peak = tracemalloc.get_traced_memory()
        top_stats = tracemalloc.take_snapshot().statistics('lineno')

        return {
            "current_memory": current,
            "peak_memory": peak,
            "top_allocations": [
                {
                    "file": stat.traceback[0].filename,
                    "line": stat.traceback[0].lineno,
                    "size": stat.size,
                    "count": stat.count
                }
                for stat in top_stats[:10]
            ]
        }

    async def async_profile_function(self, coro, function_name: str):
        """
        Profile an async function.

        Args:
            coro: Coroutine to profile
            function_name: Name of the function
        """
        if not self.is_profiling:
            return await coro

        start_time = time.time()
        with self.profile_function(function_name):
            result = await coro
        end_time = time.time()

        logger.debug(f"Async function {function_name} took {end_time - start_time:.4f}s")
        return result

    def cleanup(self):
        """Clean up profiler resources."""
        self.stop_profiling()
        self.executor.shutdown(wait=True)
        logger.info("Performance Profiler cleaned up")


# Global profiler instance
_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _profiler


def profile_function(function_name: Optional[str] = None):
    """
    Decorator to profile a function.

    Args:
        function_name: Optional custom name for the function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = function_name or f"{func.__qualname__}"
            with _profiler.profile_function(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def profiling_session(session_id: str):
    """
    Context manager for a profiling session.

    Args:
        session_id: Unique identifier for the session
    """
    _profiler.start_profiling(session_id)
    try:
        yield _profiler
    finally:
        _profiler.stop_profiling()
