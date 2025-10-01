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

import functools
import gc
import json
import logging
import os
import statistics
import subprocess
import tempfile
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil

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

        # Optimized historical data storage using numpy arrays
        self.metrics_history: deque = deque(maxlen=max_history)
        self.baseline_metrics: Dict[str, np.ndarray] = {}

        # Pre-allocated numpy arrays for performance metrics
        self._execution_times_buffer = np.zeros(1000, dtype=np.float64)
        self._memory_usage_buffer = np.zeros(1000, dtype=np.int64)
        self._cpu_usage_buffer = np.zeros(1000, dtype=np.float64)
        self._buffer_index = 0

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

        # Profiler overhead compensation
        self._profiler_overhead: Optional[float] = None
        self._measure_profiler_overhead()

        logger.info("Performance Profiler initialized")

    def _measure_profiler_overhead(self):
        """Measure the overhead introduced by the profiler itself."""
        # Measure overhead by profiling an empty function multiple times
        overhead_times = []

        for _ in range(100):  # Take multiple measurements for accuracy
            # Measure time without profiler
            start = time.perf_counter()
            # Empty function - just pass
            end = time.perf_counter()
            baseline_time = end - start

            # Measure time with profiler
            start = time.perf_counter()
            with self.profile_function("__overhead_test__"):
                pass  # Empty profiled block
            end = time.perf_counter()
            profiled_time = end - start

            # Overhead is the difference
            overhead = profiled_time - baseline_time
            if overhead > 0:  # Only count positive overhead
                overhead_times.append(overhead)

        # Calculate average overhead
        if overhead_times:
            self._profiler_overhead = statistics.mean(overhead_times)
            logger.debug(f"Measured profiler overhead: {self._profiler_overhead:.6f}s")
        else:
            self._profiler_overhead = 0.0
            logger.debug("Could not measure profiler overhead accurately")

    def get_profiler_overhead(self) -> float:
        """Get the measured profiler overhead in seconds."""
        return self._profiler_overhead or 0.0

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
                session_id=session_id, start_time=time.time()
            )
            self.is_profiling = True

            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True
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
                io_read_delta = (
                    current_io.read_bytes - last_io.read_bytes if last_io else 0
                )
                io_write_delta = (
                    current_io.write_bytes - last_io.write_bytes if last_io else 0
                )
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
            raw_execution_time = end_time - start_time
            # Compensate for profiler overhead (if measured)
            overhead = self._profiler_overhead if self._profiler_overhead is not None else 0.0
            execution_time = max(0.0, raw_execution_time - overhead)
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
                timestamp=end_time,
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
                "times": [],
                "mean": 0.0,
                "std": 0.0,
                "count": 0,
            }

        baseline = self.baseline_metrics[function_name]
        baseline["times"].append(execution_time)
        baseline["count"] += 1

        # Keep only recent measurements
        if len(baseline["times"]) > 100:
            baseline["times"] = baseline["times"][-100:]

        # Update statistics
        if len(baseline["times"]) > 1:
            baseline["mean"] = statistics.mean(baseline["times"])
            baseline["std"] = statistics.stdev(baseline["times"])

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
        if baseline["count"] < 10:  # Need minimum samples
            return False

        threshold = self.anomaly_thresholds.get(function_name, 3.0)  # 3-sigma rule
        z_score = abs(current_time - baseline["mean"]) / baseline["std"]

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
                        "end": max(m.timestamp for m in metrics),
                    },
                },
                "functions": {},
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
                        "std": statistics.stdev(execution_times)
                        if len(execution_times) > 1
                        else 0,
                    },
                    "memory_usage": {
                        "total": sum(memory_usages),
                        "mean": statistics.mean(memory_usages),
                        "peak": max(m.memory_peak for m in func_metrics),
                    },
                    "performance_trends": self._analyze_trends(func_metrics),
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
            "volatility": statistics.stdev(times) if n > 1 else 0,
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
            function_stats = defaultdict(
                lambda: {
                    "total_time": 0.0,
                    "call_count": 0,
                    "avg_time": 0.0,
                    "memory_total": 0,
                }
            )

            for metric in self.metrics_history:
                stats = function_stats[metric.function_name]
                stats["total_time"] += metric.execution_time
                stats["call_count"] += 1
                stats["memory_total"] += metric.memory_usage

            # Calculate averages
            hotspots = []
            for func_name, stats in function_stats.items():
                stats["avg_time"] = stats["total_time"] / stats["call_count"]
                hotspots.append(
                    {
                        "function": func_name,
                        "total_time": stats["total_time"],
                        "avg_time": stats["avg_time"],
                        "call_count": stats["call_count"],
                        "memory_total": stats["memory_total"],
                    }
                )

            # Sort by total time (descending)
            hotspots.sort(key=lambda x: x["total_time"], reverse=True)

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
        top_stats = tracemalloc.take_snapshot().statistics("lineno")

        return {
            "current_memory": current,
            "peak_memory": peak,
            "top_allocations": [
                {
                    "file": stat.traceback[0].filename,
                    "line": stat.traceback[0].lineno,
                    "size": stat.size,
                    "count": stat.count,
                }
                for stat in top_stats[:10]
            ],
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

        logger.debug(
            f"Async function {function_name} took {end_time - start_time:.4f}s"
        )
        return result

    def cleanup(self):
        """Clean up profiler resources."""
        self.stop_profiling()
        self.executor.shutdown(wait=True)
        logger.info("Performance Profiler cleaned up")


class AdvancedProfiler:
    """
    Advanced profiling tools integration for N1V1 framework.

    Supports multiple profiling backends:
    - py-spy: Sampling profiler with flamegraph support
    - scalene: CPU and memory profiler with web UI
    - memory_profiler: Line-by-line memory usage analysis
    - cProfile: Standard Python profiler with enhanced features
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.profiling_enabled = self.config.get("enabled", True)
        self.output_dir = Path(self.config.get("output_dir", "performance_reports"))
        self.output_dir.mkdir(exist_ok=True)

        # Tool availability
        self._check_tool_availability()

        # Active profiling processes
        self._active_processes: Dict[str, subprocess.Popen] = {}

        logger.info("AdvancedProfiler initialized")

    def _check_tool_availability(self):
        """Check which profiling tools are available."""
        self.tools_available = {
            "py_spy": self._is_tool_available("py-spy"),
            "scalene": self._is_tool_available("scalene"),
            "memory_profiler": self._is_tool_available("memory_profiler"),
            "flamegraph": self._is_tool_available("flamegraph"),
        }

        available_tools = [
            tool for tool, available in self.tools_available.items() if available
        ]
        logger.info(f"Available profiling tools: {available_tools}")

    def _is_tool_available(self, tool_name: str) -> bool:
        """Check if a profiling tool is available."""
        try:
            if tool_name == "py-spy":
                subprocess.run(["py-spy", "--version"], capture_output=True, check=True)
            elif tool_name == "scalene":
                subprocess.run(
                    ["scalene", "--version"], capture_output=True, check=True
                )
            elif tool_name == "memory_profiler":
                subprocess.run(
                    ["python", "-c", "import memory_profiler"],
                    capture_output=True,
                    check=True,
                )
            elif tool_name == "flamegraph":
                subprocess.run(["flamegraph"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def start_pyspy_profiling(
        self, pid: int = None, output_file: str = None
    ) -> Optional[str]:
        """
        Start py-spy profiling.

        Args:
            pid: Process ID to profile (default: current process)
            output_file: Output file for flamegraph

        Returns:
            Process ID of the profiling process
        """
        if not self.tools_available["py_spy"]:
            logger.warning("py-spy not available")
            return None

        if pid is None:
            pid = os.getpid()

        timestamp = int(time.time())
        output_file = output_file or f"pyspy_{pid}_{timestamp}.svg"

        output_path = self.output_dir / output_file

        cmd = [
            "py-spy",
            "record",
            "--pid",
            str(pid),
            "--format",
            "speedscope",
            "--output",
            str(output_path.with_suffix(".json")),
            "--duration",
            "60",  # 60 seconds
        ]

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self._active_processes[f"pyspy_{pid}"] = process
            logger.info(f"Started py-spy profiling for PID {pid}")

            # Generate flamegraph from speedscope
            self._generate_flamegraph_from_speedscope(
                output_path.with_suffix(".json"), output_path
            )

            return f"pyspy_{pid}"
        except Exception as e:
            logger.error(f"Failed to start py-spy profiling: {e}")
            return None

    def start_scalene_profiling(
        self, script_path: str = None, output_dir: str = None
    ) -> Optional[str]:
        """
        Start scalene profiling.

        Args:
            script_path: Python script to profile
            output_dir: Output directory for scalene results

        Returns:
            Process ID of the profiling process
        """
        if not self.tools_available["scalene"]:
            logger.warning("scalene not available")
            return None

        output_dir = output_dir or self.output_dir / f"scalene_{int(time.time())}"
        output_dir.mkdir(exist_ok=True)

        cmd = [
            "scalene",
            "--cpu",
            "--memory",
            "--html",
            "--outfile",
            str(output_dir / "profile.html"),
        ]

        if script_path:
            cmd.append(script_path)

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self._active_processes[f'scalene_{script_path or "main"}'] = process
            logger.info(f"Started scalene profiling for {script_path or 'main script'}")
            return f'scalene_{script_path or "main"}'
        except Exception as e:
            logger.error(f"Failed to start scalene profiling: {e}")
            return None

    def profile_with_memory_profiler(self, func: Callable, *args, **kwargs) -> Dict:
        """
        Profile function with memory_profiler.

        Args:
            func: Function to profile
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Memory profiling results
        """
        if not self.tools_available["memory_profiler"]:
            logger.warning("memory_profiler not available")
            return {}

        try:
            from memory_profiler import memory_usage
            from memory_profiler import profile as memory_profile

            # Profile memory usage over time
            mem_usage = memory_usage((func, args, kwargs), interval=0.1, timeout=60)

            # Get line-by-line memory usage
            timestamp = int(time.time())
            output_file = self.output_dir / f"memory_profile_{timestamp}.txt"

            # Use memory_profiler's profile decorator
            @memory_profile
            def profiled_func():
                return func(*args, **kwargs)

            # Redirect output to file
            with open(output_file, "w") as f:
                import sys

                old_stdout = sys.stdout
                sys.stdout = f
                try:
                    result = profiled_func()
                finally:
                    sys.stdout = old_stdout

            return {
                "memory_usage_over_time": mem_usage,
                "peak_memory": max(mem_usage) if mem_usage else 0,
                "average_memory": sum(mem_usage) / len(mem_usage) if mem_usage else 0,
                "profile_file": str(output_file),
                "result": result,
            }

        except Exception as e:
            logger.error(f"Failed to profile with memory_profiler: {e}")
            return {}

    def generate_flamegraph(
        self, profile_data: Dict, output_file: str = None
    ) -> Optional[str]:
        """
        Generate flamegraph from profiling data.

        Args:
            profile_data: Profiling data dictionary
            output_file: Output file for flamegraph

        Returns:
            Path to generated flamegraph
        """
        if not self.tools_available["flamegraph"]:
            logger.warning("flamegraph tool not available")
            return None

        timestamp = int(time.time())
        output_file = output_file or f"flamegraph_{timestamp}.svg"
        output_path = self.output_dir / output_file

        try:
            # Convert profile data to flamegraph format
            stacks = self._convert_to_flamegraph_format(profile_data)

            # Write stacks to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                for stack in stacks:
                    f.write(stack + "\n")
                temp_file = f.name

            # Generate flamegraph
            cmd = ["flamegraph.pl", temp_file]
            with open(output_path, "w") as f:
                subprocess.run(cmd, stdout=f, check=True)

            # Clean up temp file
            os.unlink(temp_file)

            logger.info(f"Generated flamegraph: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to generate flamegraph: {e}")
            return None

    def _convert_to_flamegraph_format(self, profile_data: Dict) -> List[str]:
        """Convert profiling data to flamegraph stack format."""
        stacks = []

        # This is a simplified conversion - in practice, you'd need
        # to properly format the stack traces based on the profiler output
        if "functions" in profile_data:
            for func_name, metrics in profile_data["functions"].items():
                # Create stack trace format: function1;function2;function3 count
                stack = f"{func_name} {int(metrics.get('call_count', 1))}"
                stacks.append(stack)

        return stacks

    def _generate_flamegraph_from_speedscope(
        self, speedscope_file: Path, output_file: Path
    ):
        """Generate flamegraph from py-spy speedscope output."""
        try:
            # This would require additional tools to convert speedscope to flamegraph
            # For now, we'll just copy the speedscope file
            import shutil

            shutil.copy2(speedscope_file, output_file.with_suffix(".json"))
            logger.info(
                f"Generated speedscope file: {output_file.with_suffix('.json')}"
            )
        except Exception as e:
            logger.error(f"Failed to generate flamegraph from speedscope: {e}")

    def stop_profiling(self, process_id: str):
        """Stop a specific profiling process."""
        if process_id in self._active_processes:
            process = self._active_processes[process_id]
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"Stopped profiling process: {process_id}")
            except Exception as e:
                logger.error(f"Failed to stop profiling process {process_id}: {e}")
                process.kill()
            finally:
                del self._active_processes[process_id]

    def stop_all_profiling(self):
        """Stop all active profiling processes."""
        for process_id in list(self._active_processes.keys()):
            self.stop_profiling(process_id)

    def get_profiling_status(self) -> Dict[str, bool]:
        """Get status of all profiling processes."""
        return {
            process_id: process.poll() is None
            for process_id, process in self._active_processes.items()
        }

    def cleanup(self):
        """Clean up profiling resources."""
        self.stop_all_profiling()
        logger.info("AdvancedProfiler cleaned up")


class RegressionDetector:
    """
    Performance regression detection system.

    Tracks performance baselines and detects significant deviations
    that may indicate performance regressions.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.baselines_file = Path(
            self.config.get("baselines_file", "performance_baselines.json")
        )
        self.thresholds = {
            "latency_increase": self.config.get(
                "latency_threshold", 0.20
            ),  # 20% increase
            "memory_increase": self.config.get(
                "memory_threshold", 0.30
            ),  # 30% increase
            "cpu_increase": self.config.get("cpu_threshold", 0.25),  # 25% increase
        }

        self.baselines = self._load_baselines()
        logger.info("RegressionDetector initialized")

    def _load_baselines(self) -> Dict:
        """Load performance baselines from file."""
        if self.baselines_file.exists():
            try:
                with open(self.baselines_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load baselines: {e}")

        return {"functions": {}, "last_updated": time.time()}

    def _save_baselines(self):
        """Save performance baselines to file."""
        try:
            with open(self.baselines_file, "w") as f:
                json.dump(self.baselines, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")

    def update_baseline(self, function_name: str, metrics: Dict):
        """Update baseline for a function."""
        if function_name not in self.baselines["functions"]:
            self.baselines["functions"][function_name] = {
                "latency_mean": 0.0,
                "latency_std": 0.0,
                "memory_mean": 0,
                "memory_std": 0,
                "cpu_mean": 0.0,
                "cpu_std": 0.0,
                "samples": [],
            }

        baseline = self.baselines["functions"][function_name]
        baseline["samples"].append(
            {
                "timestamp": time.time(),
                "latency": metrics.get("execution_time", 0),
                "memory": metrics.get("memory_usage", 0),
                "cpu": metrics.get("cpu_percent", 0),
            }
        )

        # Keep only recent samples (last 100)
        if len(baseline["samples"]) > 100:
            baseline["samples"] = baseline["samples"][-100:]

        # Update statistics
        self._update_statistics(baseline)
        self.baselines["last_updated"] = time.time()
        self._save_baselines()

    def _update_statistics(self, baseline: Dict):
        """Update statistical measures for a baseline."""
        samples = baseline["samples"]
        if not samples:
            return

        latencies = [s["latency"] for s in samples]
        memories = [s["memory"] for s in samples]
        cpus = [s["cpu"] for s in samples]

        baseline["latency_mean"] = statistics.mean(latencies)
        baseline["latency_std"] = (
            statistics.stdev(latencies) if len(latencies) > 1 else 0
        )
        baseline["memory_mean"] = statistics.mean(memories)
        baseline["memory_std"] = statistics.stdev(memories) if len(memories) > 1 else 0
        baseline["cpu_mean"] = statistics.mean(cpus)
        baseline["cpu_std"] = statistics.stdev(cpus) if len(cpus) > 1 else 0

    def detect_regression(self, function_name: str, current_metrics: Dict) -> Dict:
        """
        Detect performance regression for a function.

        Args:
            function_name: Name of the function
            current_metrics: Current performance metrics

        Returns:
            Regression detection results
        """
        if function_name not in self.baselines["functions"]:
            return {"regression_detected": False, "reason": "no_baseline"}

        baseline = self.baselines["functions"][function_name]
        if len(baseline["samples"]) < 10:  # Need minimum samples
            return {"regression_detected": False, "reason": "insufficient_samples"}

        results = {
            "regression_detected": False,
            "issues": [],
            "current_values": current_metrics,
            "baseline_values": {
                "latency_mean": baseline["latency_mean"],
                "memory_mean": baseline["memory_mean"],
                "cpu_mean": baseline["cpu_mean"],
            },
        }

        # Check latency regression
        current_latency = current_metrics.get("execution_time", 0)
        latency_threshold = baseline["latency_mean"] * (
            1 + self.thresholds["latency_increase"]
        )
        if current_latency > latency_threshold:
            results["regression_detected"] = True
            results["issues"].append(
                {
                    "type": "latency",
                    "current": current_latency,
                    "baseline": baseline["latency_mean"],
                    "threshold": latency_threshold,
                    "increase_percent": (current_latency - baseline["latency_mean"])
                    / baseline["latency_mean"],
                }
            )

        # Check memory regression
        current_memory = current_metrics.get("memory_usage", 0)
        memory_threshold = baseline["memory_mean"] * (
            1 + self.thresholds["memory_increase"]
        )
        if current_memory > memory_threshold:
            results["regression_detected"] = True
            results["issues"].append(
                {
                    "type": "memory",
                    "current": current_memory,
                    "baseline": baseline["memory_mean"],
                    "threshold": memory_threshold,
                    "increase_percent": (current_memory - baseline["memory_mean"])
                    / baseline["memory_mean"],
                }
            )

        # Check CPU regression
        current_cpu = current_metrics.get("cpu_percent", 0)
        cpu_threshold = baseline["cpu_mean"] * (1 + self.thresholds["cpu_increase"])
        if current_cpu > cpu_threshold:
            results["regression_detected"] = True
            results["issues"].append(
                {
                    "type": "cpu",
                    "current": current_cpu,
                    "baseline": baseline["cpu_mean"],
                    "threshold": cpu_threshold,
                    "increase_percent": (current_cpu - baseline["cpu_mean"])
                    / baseline["cpu_mean"],
                }
            )

        return results

    def get_regression_report(self) -> Dict:
        """Generate a comprehensive regression report."""
        return {
            "baselines": self.baselines,
            "thresholds": self.thresholds,
            "last_updated": self.baselines.get("last_updated", 0),
        }


# Global instances
_advanced_profiler = None
_regression_detector = None


def get_advanced_profiler(config: Dict[str, Any] = None) -> AdvancedProfiler:
    """Get the global advanced profiler instance."""
    global _advanced_profiler
    if _advanced_profiler is None:
        _advanced_profiler = AdvancedProfiler(config)
    return _advanced_profiler


def get_regression_detector(config: Dict[str, Any] = None) -> RegressionDetector:
    """Get the global regression detector instance."""
    global _regression_detector
    if _regression_detector is None:
        _regression_detector = RegressionDetector(config)
    return _regression_detector


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
