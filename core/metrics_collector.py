"""
Metrics Collection System for N1V1 Trading Framework

This module implements a comprehensive metrics collection system following
Prometheus exposition format. It provides real-time monitoring of trading
performance, system health, risk metrics, and business KPIs.

Key Features:
- Efficient metrics collection with minimal performance impact
- Comprehensive metric categories (trading, system, risk, strategy)
- Async collection to avoid blocking main trading loop
- Metrics aggregation and caching for performance
- Integration with existing framework components

Metrics Categories:
- Trading Performance: PnL, win rate, Sharpe ratio, drawdown
- Order Execution: Latency, success rate, slippage, exchange performance
- System Health: CPU, memory, disk I/O, network usage
- Risk Metrics: VaR, exposure, concentration, circuit breaker status
- Strategy Metrics: Individual strategy performance, signal quality
"""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricSample:
    """A single metric measurement."""

    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None
    help_text: str = ""

    def to_prometheus(self) -> str:
        """Convert to Prometheus exposition format."""
        # Add HELP comment if provided
        help_line = ""
        if self.help_text:
            help_line = f"# HELP {self.name} {self.help_text}\n"

        # Add TYPE comment (inferred from metric name)
        metric_type = self._infer_metric_type()
        type_line = f"# TYPE {self.name} {metric_type}\n"

        # Format labels
        labels_str = ""
        if self.labels:
            label_parts = [f'{k}="{v}"' for k, v in self.labels.items()]
            labels_str = f"{{{','.join(label_parts)}}}"

        # Format metric line
        timestamp_str = ""
        if self.timestamp:
            timestamp_str = f" {int(self.timestamp * 1000)}"

        metric_line = f"{self.name}{labels_str} {self.value}{timestamp_str}\n"

        return f"{help_line}{type_line}{metric_line}"

    def _infer_metric_type(self) -> str:
        """Infer Prometheus metric type from metric name."""
        if self.name.endswith("_total") or self.name.endswith("_count"):
            return "counter"
        elif self.name.endswith("_seconds") or self.name.endswith("_duration"):
            return "histogram"
        elif "rate" in self.name or "ratio" in self.name:
            return "gauge"
        else:
            return "gauge"  # Default to gauge


@dataclass
class MetricSeries:
    """A time series of metric measurements with efficient storage."""

    name: str
    help_text: str = ""
    samples: List[MetricSample] = field(default_factory=list)
    max_samples: int = 1000

    # Efficient storage using numpy arrays for large datasets
    _values: Optional[np.ndarray] = field(default=None, init=False)
    _timestamps: Optional[np.ndarray] = field(default=None, init=False)
    _labels_list: List[Dict[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Initialize efficient storage structures."""
        if self.max_samples > 100:  # Use numpy arrays for larger datasets
            self._values = np.full(self.max_samples, np.nan, dtype=np.float64)
            self._timestamps = np.full(self.max_samples, np.nan, dtype=np.float64)
            self._current_index = 0
            self._is_full = False

    def add_sample(
        self, value: float, labels: Dict[str, str] = None, timestamp: float = None
    ) -> None:
        """Add a new sample to the series with efficient storage."""
        sample = MetricSample(
            name=self.name,
            value=value,
            labels=labels or {},
            timestamp=timestamp or time.time(),
            help_text=self.help_text,
        )

        # Always add to samples list for compatibility with existing code
        self.samples.append(sample)
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples :]

        # Use efficient numpy storage for large datasets
        if self._values is not None:
            self._values[self._current_index] = value
            self._timestamps[self._current_index] = sample.timestamp
            self._labels_list.append(labels or {})

            self._current_index += 1
            if self._current_index >= self.max_samples:
                self._current_index = 0
                self._is_full = True

            # Keep labels list in sync with numpy arrays
            if len(self._labels_list) > self.max_samples:
                self._labels_list = self._labels_list[-self.max_samples :]

    def get_latest_sample(self) -> Optional[MetricSample]:
        """Get the most recent sample with efficient lookup."""
        if self._values is not None:
            if self._is_full:
                # Get the most recent value (circular buffer)
                latest_idx = (self._current_index - 1) % self.max_samples
                if not np.isnan(self._values[latest_idx]):
                    return MetricSample(
                        name=self.name,
                        value=self._values[latest_idx],
                        labels=self._labels_list[latest_idx]
                        if latest_idx < len(self._labels_list)
                        else {},
                        timestamp=self._timestamps[latest_idx],
                        help_text=self.help_text,
                    )
            elif self._current_index > 0:
                latest_idx = self._current_index - 1
                return MetricSample(
                    name=self.name,
                    value=self._values[latest_idx],
                    labels=self._labels_list[latest_idx]
                    if latest_idx < len(self._labels_list)
                    else {},
                    timestamp=self._timestamps[latest_idx],
                    help_text=self.help_text,
                )
            return None
        else:
            # Fallback to list lookup
            return self.samples[-1] if self.samples else None

    def get_samples_in_range(
        self, start_time: float, end_time: float
    ) -> List[MetricSample]:
        """Get samples within a time range with efficient filtering."""
        if self._values is not None:
            # Use numpy boolean indexing for efficient filtering
            if self._is_full:
                # Handle circular buffer
                mask = (
                    (self._timestamps >= start_time) & (self._timestamps <= end_time)
                ) & (~np.isnan(self._timestamps))
                valid_indices = np.where(mask)[0]

                samples = []
                for idx in valid_indices:
                    samples.append(
                        MetricSample(
                            name=self.name,
                            value=self._values[idx],
                            labels=self._labels_list[idx]
                            if idx < len(self._labels_list)
                            else {},
                            timestamp=self._timestamps[idx],
                            help_text=self.help_text,
                        )
                    )
                return samples
            else:
                # Handle linear buffer
                valid_mask = (
                    (self._timestamps[: self._current_index] >= start_time)
                    & (self._timestamps[: self._current_index] <= end_time)
                    & (~np.isnan(self._timestamps[: self._current_index]))
                )
                valid_indices = np.where(valid_mask)[0]

                samples = []
                for idx in valid_indices:
                    samples.append(
                        MetricSample(
                            name=self.name,
                            value=self._values[idx],
                            labels=self._labels_list[idx]
                            if idx < len(self._labels_list)
                            else {},
                            timestamp=self._timestamps[idx],
                            help_text=self.help_text,
                        )
                    )
                return samples
        else:
            # Fallback to list filtering
            return [
                sample
                for sample in self.samples
                if start_time <= sample.timestamp <= end_time
            ]


class MetricsCollector:
    """
    Main metrics collector for the N1V1 trading framework.

    Provides comprehensive metrics collection with minimal performance impact,
    following Prometheus best practices for metric naming and exposition.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Metrics storage
        self.metrics: Dict[str, MetricSeries] = {}
        self.custom_collectors: List[Callable] = []

        # Collection settings
        self.collection_interval = self.config.get(
            "collection_interval", 15.0
        )  # seconds
        self.max_samples_per_metric = self.config.get("max_samples_per_metric", 1000)

        # Async collection
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Async queue for batched recording to reduce CPU overhead
        self._record_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._record_task: Optional[asyncio.Task] = None

        # System metrics
        self.process = psutil.Process()

        logger.info("MetricsCollector initialized")

    async def start(self) -> None:
        """Start the metrics collection system."""
        if self._running:
            return

        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        self._record_task = asyncio.create_task(self._record_loop())

        # Register system metrics collectors
        await self._register_system_collectors()

        logger.info("✅ MetricsCollector started")

    async def stop(self) -> None:
        """Stop the metrics collection system."""
        if not self._running:
            return

        self._running = False

        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        if self._record_task:
            self._record_task.cancel()
            try:
                await self._record_task
            except asyncio.CancelledError:
                pass

        logger.info("✅ MetricsCollector stopped")

    def register_metric(
        self, name: str, help_text: str = "", max_samples: int = None
    ) -> MetricSeries:
        """Register a new metric series."""
        if name in self.metrics:
            # Return existing metric series
            return self.metrics[name]

        series = MetricSeries(
            name=name,
            help_text=help_text,
            max_samples=max_samples or self.max_samples_per_metric,
        )

        self.metrics[name] = series
        logger.debug(f"Registered metric: {name}")

        return series

    def add_custom_collector(self, collector_func: Callable) -> None:
        """Add a custom metrics collector function."""
        self.custom_collectors.append(collector_func)
        logger.debug("Added custom metrics collector")

    async def record_metric(
        self, name: str, value: float, labels: Dict[str, str] = None
    ) -> None:
        """Record a metric value asynchronously without lock for performance."""
        # Register metric if it doesn't exist
        if name not in self.metrics:
            self.register_metric(name)
        series = self.metrics[name]

        series.add_sample(value, labels)

    async def increment_counter(self, name: str, labels: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        # Get current value (this needs to be synchronous for accuracy)
        current_value = self.get_metric_value(name, labels) or 0
        new_value = current_value + 1

        # Use async recording for the increment
        await self.record_metric(name, new_value, labels)

    async def observe_histogram(
        self, name: str, value: float, labels: Dict[str, str] = None
    ) -> None:
        """Observe a value in a histogram metric."""
        # For simplicity, we'll store individual observations
        # In a full implementation, this would aggregate into buckets
        await self.record_metric(name, value, labels)

    async def record_latency(
        self, name: str, latency_ms: float, labels: Dict[str, str] = None
    ) -> None:
        """Record a latency measurement."""
        await self.record_metric(name, latency_ms, labels)

    def get_metric_value(
        self, name: str, labels: Dict[str, str] = None
    ) -> Optional[float]:
        """Get the latest value of a metric."""
        if name not in self.metrics:
            return None

        series = self.metrics[name]
        latest = series.get_latest_sample()

        if not latest:
            return None

        # If labels are specified, check if they match
        if labels and latest.labels != labels:
            return None

        return latest.value

    def get_prometheus_output(self) -> str:
        """Generate Prometheus exposition format output."""
        output_lines = []

        for series in self.metrics.values():
            latest_sample = series.get_latest_sample()
            if latest_sample:
                output_lines.append(latest_sample.to_prometheus())

        return "\n".join(output_lines)

    async def _record_loop(self) -> None:
        """Background loop for processing queued metric recordings."""
        while self._running:
            try:
                # Get batch of recordings from queue
                batch = []
                try:
                    # Try to get first item with timeout
                    item = await asyncio.wait_for(self._record_queue.get(), timeout=0.1)
                    batch.append(item)

                    # Get more items if available (batching)
                    while len(batch) < 100:  # Process up to 100 items at once
                        try:
                            item = self._record_queue.get_nowait()
                            batch.append(item)
                        except asyncio.QueueEmpty:
                            break
                except asyncio.TimeoutError:
                    continue  # No items, continue loop

                # Process the batch
                async with self._lock:
                    for name, value, labels in batch:
                        # Register metric if it doesn't exist
                        if name not in self.metrics:
                            self.register_metric(name)
                        series = self.metrics[name]
                        series.add_sample(value, labels)

                        # Mark task as done
                        self._record_queue.task_done()

            except Exception as e:
                logger.exception(f"Error in record loop: {e}")
                await asyncio.sleep(0.1)  # Brief pause before retry

    async def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self._running:
            try:
                start_time = time.time()

                # Collect system metrics
                await self._collect_system_metrics()

                # Collect custom metrics
                await self._collect_custom_metrics()

                # Calculate collection time
                collection_time = time.time() - start_time

                # Record collection performance
                await self.record_metric(
                    "metrics_collection_duration_seconds",
                    collection_time,
                    {"collector": "main"},
                )

                # Wait for next collection interval
                await asyncio.sleep(max(0, self.collection_interval - collection_time))

            except Exception as e:
                logger.exception(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _register_system_collectors(self) -> None:
        """Register system-level metrics collectors."""
        # CPU usage
        self.register_metric("system_cpu_usage_percent", "System CPU usage percentage")

        # Memory usage
        self.register_metric(
            "system_memory_usage_bytes", "System memory usage in bytes"
        )

        # Disk I/O
        self.register_metric("system_disk_read_bytes_total", "Total disk bytes read")
        self.register_metric(
            "system_disk_write_bytes_total", "Total disk bytes written"
        )

        # Network I/O
        self.register_metric(
            "system_network_receive_bytes_total", "Total network bytes received"
        )
        self.register_metric(
            "system_network_transmit_bytes_total", "Total network bytes transmitted"
        )

        # Process metrics
        self.register_metric(
            "process_cpu_usage_percent", "Process CPU usage percentage"
        )
        self.register_metric(
            "process_memory_usage_bytes", "Process memory usage in bytes"
        )
        self.register_metric("process_threads_count", "Number of process threads")

    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1.0)
            await self.record_metric("system_cpu_usage_percent", cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            await self.record_metric("system_memory_usage_bytes", memory.used)

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                await self.record_metric(
                    "system_disk_read_bytes_total", disk_io.read_bytes
                )
                await self.record_metric(
                    "system_disk_write_bytes_total", disk_io.write_bytes
                )

            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                await self.record_metric(
                    "system_network_receive_bytes_total", net_io.bytes_recv
                )
                await self.record_metric(
                    "system_network_transmit_bytes_total", net_io.bytes_sent
                )

            # Process metrics
            process_cpu = self.process.cpu_percent()
            process_memory = self.process.memory_info().rss
            process_threads = self.process.num_threads()

            await self.record_metric("process_cpu_usage_percent", process_cpu)
            await self.record_metric("process_memory_usage_bytes", process_memory)
            await self.record_metric("process_threads_count", process_threads)

        except Exception as e:
            logger.exception(f"Error collecting system metrics: {e}")

    async def _collect_custom_metrics(self) -> None:
        """Collect metrics from custom collectors."""
        for collector in self.custom_collectors:
            try:
                await collector(self)
            except Exception as e:
                logger.exception(f"Error in custom metrics collector: {e}")


# Trading-specific metrics collectors
async def collect_trading_metrics(collector: MetricsCollector) -> None:
    """Collect trading performance metrics."""
    # These would be populated from actual trading data
    # For demonstration, we'll use mock data

    # Trading performance metrics
    await collector.record_metric("trading_total_pnl_usd", 1250.75, {"account": "main"})

    await collector.record_metric("trading_win_rate_ratio", 0.68, {"account": "main"})

    await collector.record_metric("trading_sharpe_ratio", 1.45, {"account": "main"})

    await collector.record_metric(
        "trading_max_drawdown_percent", 8.5, {"account": "main"}
    )

    # Order execution metrics
    await collector.record_metric(
        "trading_orders_total", 1250, {"account": "main", "status": "filled"}
    )

    await collector.record_metric(
        "trading_orders_skipped_safe_mode_total",
        0,  # Would be populated from actual data
        {"account": "main"},
    )

    await collector.record_metric(
        "trading_order_latency_seconds",
        0.045,
        {"account": "main", "exchange": "binance"},
    )

    await collector.record_metric(
        "trading_slippage_bps", 2.5, {"account": "main", "symbol": "BTC/USDT"}
    )

    # Calculate order failure rate excluding safe mode skips
    total_orders = (
        collector.get_metric_value(
            "trading_orders_total", {"account": "main", "status": "filled"}
        )
        or 0
    )
    skipped_orders = (
        collector.get_metric_value(
            "trading_orders_skipped_safe_mode_total", {"account": "main"}
        )
        or 0
    )
    failed_orders = (
        collector.get_metric_value("trading_orders_failed_total", {"account": "main"})
        or 0
    )

    # Failure rate calculation: failed orders / (total orders - skipped orders)
    if total_orders > skipped_orders:
        failure_rate = failed_orders / (total_orders - skipped_orders)
        await collector.record_metric(
            "trading_order_failure_rate_ratio", failure_rate, {"account": "main"}
        )


async def collect_binary_model_metrics(collector: MetricsCollector) -> None:
    """Collect binary model specific metrics."""
    try:
        from core.binary_model_metrics import get_binary_model_metrics_collector

        binary_metrics = get_binary_model_metrics_collector()
        await binary_metrics.collect_binary_model_metrics(collector)
    except ImportError:
        # Binary model metrics not available
        pass
    except Exception as e:
        logger.error(f"Error collecting binary model metrics: {e}")


async def collect_risk_metrics(collector: MetricsCollector) -> None:
    """Collect risk management metrics."""
    # Risk metrics
    await collector.record_metric(
        "risk_value_at_risk_usd",
        500.0,
        {"account": "main", "confidence": "95", "horizon": "1d"},
    )

    await collector.record_metric(
        "risk_portfolio_exposure_usd", 25000.0, {"account": "main"}
    )

    await collector.record_metric(
        "risk_concentration_ratio", 0.15, {"account": "main", "asset": "BTC"}
    )

    await collector.record_metric(
        "risk_circuit_breaker_status",
        0,  # 0 = normal, 1 = triggered
        {"account": "main"},
    )


async def collect_strategy_metrics(collector: MetricsCollector) -> None:
    """Collect strategy performance metrics."""
    strategies = ["rsi_strategy", "macd_strategy", "bollinger_strategy"]

    for strategy in strategies:
        # Strategy performance
        await collector.record_metric(
            "strategy_pnl_usd", 350.0, {"strategy": strategy, "account": "main"}
        )

        await collector.record_metric(
            "strategy_win_rate_ratio", 0.72, {"strategy": strategy, "account": "main"}
        )

        await collector.record_metric(
            "strategy_signals_total", 450, {"strategy": strategy, "account": "main"}
        )

        await collector.record_metric(
            "strategy_signal_quality_ratio",
            0.85,
            {"strategy": strategy, "account": "main"},
        )


async def collect_distributed_metrics(collector: MetricsCollector) -> None:
    """Collect distributed task processing metrics."""
    try:
        # Get task manager instance (would need to be injected or accessed globally)
        # For now, we'll use mock data

        # Queue depth
        await collector.record_metric(
            "distributed_queue_depth", 5, {"queue_type": "in_memory"}
        )

        # Active workers
        await collector.record_metric(
            "distributed_active_workers", 4, {"worker_type": "signal_processor"}
        )

        # Task success rate
        await collector.record_metric(
            "distributed_task_success_rate", 0.95, {"task_type": "signal"}
        )

        # Task processing latency
        await collector.record_metric(
            "distributed_task_processing_seconds", 0.45, {"task_type": "signal"}
        )

        # Worker uptime
        await collector.record_metric(
            "distributed_worker_uptime_seconds", 3600, {"worker_id": "worker_0"}
        )

        # Failed tasks
        await collector.record_metric(
            "distributed_failed_tasks_total", 2, {"failure_reason": "timeout"}
        )

        # Tasks processed per minute
        await collector.record_metric(
            "distributed_tasks_processed_per_minute", 12.5, {"task_type": "signal"}
        )

        # Queue throughput
        await collector.record_metric(
            "distributed_queue_throughput_per_second", 0.2, {"queue_type": "in_memory"}
        )

    except Exception as e:
        logger.error(f"Error collecting distributed metrics: {e}")


async def collect_exchange_metrics(collector: MetricsCollector) -> None:
    """Collect exchange connectivity metrics."""
    exchanges = ["binance", "coinbase", "kraken"]

    for exchange in exchanges:
        # Exchange connectivity
        await collector.record_metric(
            "exchange_connectivity_status",
            1,  # 1 = connected, 0 = disconnected
            {"exchange": exchange},
        )

        await collector.record_metric(
            "exchange_latency_seconds", 0.032, {"exchange": exchange}
        )

        await collector.record_metric(
            "exchange_rate_limit_usage_ratio", 0.45, {"exchange": exchange}
        )

        await collector.record_metric(
            "exchange_api_requests_total",
            1250,
            {"exchange": exchange, "endpoint": "orders"},
        )


async def collect_performance_metrics(collector: MetricsCollector) -> None:
    """Collect performance profiling metrics."""
    try:
        from core.performance_profiler import get_profiler, get_regression_detector

        profiler = get_profiler()
        regression_detector = get_regression_detector()

        # Get current performance report
        report = profiler.generate_report()

        # Record function execution metrics
        for func_name, metrics in report.get("functions", {}).items():
            await collector.record_metric(
                "performance_function_execution_time_seconds",
                metrics["execution_time"]["mean"],
                {"function": func_name},
            )

            await collector.record_metric(
                "performance_function_call_count",
                metrics["call_count"],
                {"function": func_name},
            )

            await collector.record_metric(
                "performance_function_memory_usage_bytes",
                metrics["memory_usage"]["mean"],
                {"function": func_name},
            )

        # Record system performance metrics
        await collector.record_metric(
            "performance_total_functions_profiled",
            report.get("summary", {}).get("total_functions", 0),
        )

        await collector.record_metric(
            "performance_total_measurements",
            report.get("summary", {}).get("total_measurements", 0),
        )

        # Record regression detection status
        regression_report = regression_detector.get_regression_report()
        await collector.record_metric(
            "performance_regression_baselines_count",
            len(regression_report.get("baselines", {}).get("functions", {})),
        )

        # Record performance hotspots
        hotspots = profiler.get_hotspots(top_n=5)
        for i, hotspot in enumerate(hotspots):
            await collector.record_metric(
                "performance_hotspot_execution_time_seconds",
                hotspot["avg_time"],
                {"function": hotspot["function"], "rank": str(i + 1)},
            )

        # Record memory usage trends
        if hasattr(profiler, "get_memory_report"):
            memory_report = profiler.get_memory_report()
            if "current_memory" in memory_report:
                await collector.record_metric(
                    "performance_memory_current_bytes", memory_report["current_memory"]
                )
            if "peak_memory" in memory_report:
                await collector.record_metric(
                    "performance_memory_peak_bytes", memory_report["peak_memory"]
                )

    except Exception as e:
        logger.error(f"Error collecting performance metrics: {e}")


async def collect_profiling_tool_metrics(collector: MetricsCollector) -> None:
    """Collect metrics from advanced profiling tools."""
    try:
        from core.performance_profiler import get_advanced_profiler

        advanced_profiler = get_advanced_profiler()

        # Record tool availability
        tools_available = advanced_profiler.tools_available
        for tool_name, available in tools_available.items():
            await collector.record_metric(
                "profiling_tool_available", 1 if available else 0, {"tool": tool_name}
            )

        # Record active profiling processes
        profiling_status = advanced_profiler.get_profiling_status()
        await collector.record_metric(
            "profiling_active_processes_count",
            len([pid for pid, active in profiling_status.items() if active]),
        )

        # Record profiling artifacts generated
        output_dir = Path(advanced_profiler.output_dir)
        if output_dir.exists():
            artifact_count = len(list(output_dir.glob("*")))
            await collector.record_metric(
                "profiling_artifacts_generated_total", artifact_count
            )

    except Exception as e:
        logger.error(f"Error collecting profiling tool metrics: {e}")


async def collect_regression_detection_metrics(collector: MetricsCollector) -> None:
    """Collect regression detection metrics."""
    try:
        from core.performance_profiler import get_regression_detector

        regression_detector = get_regression_detector()

        # Record regression thresholds
        thresholds = regression_detector.thresholds
        await collector.record_metric(
            "regression_threshold_latency_increase_ratio",
            thresholds["latency_increase"],
        )

        await collector.record_metric(
            "regression_threshold_memory_increase_ratio", thresholds["memory_increase"]
        )

        await collector.record_metric(
            "regression_threshold_cpu_increase_ratio", thresholds["cpu_increase"]
        )

        # Record baseline statistics
        baselines = regression_detector.baselines
        functions_with_baselines = len(baselines.get("functions", {}))
        await collector.record_metric(
            "regression_functions_with_baselines_count", functions_with_baselines
        )

        # Calculate total baseline samples
        total_samples = sum(
            len(func_data.get("samples", []))
            for func_data in baselines.get("functions", {}).values()
        )
        await collector.record_metric(
            "regression_baseline_samples_total", total_samples
        )

    except Exception as e:
        logger.error(f"Error collecting regression detection metrics: {e}")


async def collect_performance_anomaly_metrics(collector: MetricsCollector) -> None:
    """Collect performance anomaly detection metrics."""
    try:
        from core.performance_profiler import get_profiler

        profiler = get_profiler()

        # Count functions with anomaly detection baselines
        anomaly_functions = len(profiler.baseline_metrics)
        await collector.record_metric(
            "performance_anomaly_functions_monitored_count", anomaly_functions
        )

        # Record anomaly detection statistics
        total_measurements = len(profiler.metrics_history)
        await collector.record_metric(
            "performance_measurements_total", total_measurements
        )

        # Calculate anomaly rate (simplified)
        if hasattr(profiler, "_detected_anomalies"):
            anomaly_count = len(profiler._detected_anomalies)
            await collector.record_metric(
                "performance_anomalies_detected_total", anomaly_count
            )

            if total_measurements > 0:
                anomaly_rate = anomaly_count / total_measurements
                await collector.record_metric(
                    "performance_anomaly_rate_ratio", anomaly_rate
                )

    except Exception as e:
        logger.error(f"Error collecting performance anomaly metrics: {e}")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector({})
    return _metrics_collector


def create_metrics_collector(
    config: Optional[Dict[str, Any]] = None
) -> MetricsCollector:
    """Create a new metrics collector instance."""
    return MetricsCollector(config or {})
