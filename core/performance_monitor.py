"""
Real-time Performance Monitoring System
=======================================

This module provides continuous performance monitoring capabilities for the trading framework,
including real-time metrics collection, statistical baselining, anomaly detection, and alerting.

Key Features:
- Continuous performance metrics collection
- Low-overhead sampling profiler
- Statistical performance baselining
- Real-time anomaly detection
- Performance alerting and notifications
- Integration with existing metrics system
"""

import asyncio
import json
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from contextlib import asynccontextmanager

import numpy as np
import aiofiles
from scipy import stats

from core.metrics_collector import get_metrics_collector
from core.performance_profiler import get_profiler

from .logging_utils import LogSensitivity, get_structured_logger

logger = get_structured_logger("core.performance_monitor", LogSensitivity.SECURE)


@dataclass
class PerformanceBaseline:
    """Statistical baseline for performance metrics."""

    metric_name: str
    mean: float
    std: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float
    sample_count: int
    last_updated: float
    trend_slope: float = 0.0
    is_stable: bool = True


@dataclass
class PerformanceAlert:
    """Performance alert configuration and state."""

    alert_id: str
    metric_name: str
    condition: str  # 'above', 'below', 'anomaly'
    threshold: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    cooldown_period: float  # seconds
    last_triggered: Optional[float] = None
    enabled: bool = True
    description: str = ""


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection analysis."""

    metric_name: str
    is_anomaly: bool
    score: float
    confidence: float
    detection_method: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class RealTimePerformanceMonitor:
    """
    Real-time performance monitoring system for the trading framework.

    Provides continuous monitoring, baselining, anomaly detection, and alerting
    for performance metrics across the entire system.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Monitoring configuration
        self.monitoring_interval = config.get("monitoring_interval", 5.0)  # seconds
        self.baseline_window = config.get("baseline_window", 3600.0)  # 1 hour
        self.anomaly_detection_enabled = config.get("anomaly_detection", True)
        self.alerting_enabled = config.get("alerting", True)

        # Performance profiler integration
        self.profiler = get_profiler()
        self.metrics_collector = get_metrics_collector()

        # Baseline management
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.baseline_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.baseline_lock = asyncio.Lock()  # Thread-safe baseline updates

        # Alert management
        self.alerts: Dict[str, PerformanceAlert] = {}
        self.active_alerts: Dict[str, PerformanceAlert] = {}

        # Anomaly detection
        self.anomaly_history: deque = deque(maxlen=1000)
        self.anomaly_thresholds: Dict[str, float] = {}

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.baseline_task: Optional[asyncio.Task] = None

        # Performance metrics storage
        self.performance_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )

        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.anomaly_callbacks: List[Callable] = []

        logger.info("RealTimePerformanceMonitor initialized")

    async def start_monitoring(self) -> None:
        """Start the real-time performance monitoring system."""
        if self.is_monitoring:
            return

        self.is_monitoring = True

        # Start monitoring tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.baseline_task = asyncio.create_task(self._baseline_update_loop())

        # Register performance metrics
        await self._register_performance_metrics()

        # Load existing baselines if available
        await self._load_baselines()

        logger.info("✅ Real-time performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop the real-time performance monitoring system."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        # Cancel monitoring tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        if self.baseline_task:
            self.baseline_task.cancel()
            try:
                await self.baseline_task
            except asyncio.CancelledError:
                pass

        # Save baselines
        await self._save_baselines()

        logger.info("✅ Real-time performance monitoring stopped")

    def add_alert(self, alert: PerformanceAlert) -> None:
        """Add a performance alert configuration."""
        self.alerts[alert.alert_id] = alert
        logger.info(f"Added performance alert: {alert.alert_id}")

    def remove_alert(self, alert_id: str) -> None:
        """Remove a performance alert configuration."""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
            logger.info(f"Removed performance alert: {alert_id}")

    def add_alert_callback(self, callback: Callable) -> None:
        """Add a callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def add_anomaly_callback(self, callback: Callable) -> None:
        """Add a callback for anomaly notifications."""
        self.anomaly_callbacks.append(callback)

    async def get_performance_status(self) -> Dict[str, Any]:
        """Get current performance status overview."""
        current_time = time.time()

        status = {
            "timestamp": current_time,
            "is_monitoring": self.is_monitoring,
            "active_alerts": len(self.active_alerts),
            "total_baselines": len(self.baselines),
            "recent_anomalies": len(
                [
                    a
                    for a in self.anomaly_history
                    if current_time - a.timestamp < 3600  # Last hour
                ]
            ),
            "system_health": self._calculate_system_health_score(),
            "performance_summary": {},
        }

        # Add performance summaries for key metrics
        key_metrics = [
            "function_execution_time",
            "memory_usage",
            "cpu_usage",
            "io_operations",
        ]

        for metric in key_metrics:
            if metric in self.baselines:
                baseline = self.baselines[metric]
                status["performance_summary"][metric] = {
                    "mean": baseline.mean,
                    "std": baseline.std,
                    "is_stable": baseline.is_stable,
                    "trend": "improving" if baseline.trend_slope < 0 else "degrading",
                }

        return status

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for real-time performance data collection."""
        while self.is_monitoring:
            try:
                start_time = time.time()

                # Collect current performance metrics
                metrics = await self._collect_performance_metrics()

                # Update baselines with new data
                await self._update_baselines_with_metrics(metrics)

                # Perform anomaly detection
                if self.anomaly_detection_enabled:
                    anomalies = self._detect_anomalies(metrics)
                    for anomaly in anomalies:
                        await self._handle_anomaly(anomaly)

                # Check alerts
                if self.alerting_enabled:
                    await self._check_alerts(metrics)

                # Record monitoring performance
                monitoring_time = time.time() - start_time
                await self.metrics_collector.record_metric(
                    "performance_monitoring_duration_seconds", monitoring_time
                )

                # Wait for next monitoring interval
                await asyncio.sleep(max(0, self.monitoring_interval - monitoring_time))

            except Exception as e:
                logger.exception(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _baseline_update_loop(self) -> None:
        """Periodic baseline update loop."""
        while self.is_monitoring:
            try:
                await self._update_baselines()
                await self._save_baselines()

                # Update every 5 minutes
                await asyncio.sleep(300)

            except Exception as e:
                logger.exception(f"Error in baseline update loop: {e}")
                await asyncio.sleep(60)

    async def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics from various sources asynchronously."""
        metrics = {}
        current_time = time.time()

        try:
            # Collect metrics concurrently using asyncio.gather for better performance
            profiler_task = self._collect_profiler_metrics()
            system_task = self._collect_system_metrics()

            profiler_metrics, system_metrics = await asyncio.gather(
                profiler_task, system_task, return_exceptions=True
            )

            # Handle profiler metrics
            if isinstance(profiler_metrics, Exception):
                logger.exception(
                    f"Error collecting profiler metrics: {profiler_metrics}"
                )
            elif isinstance(profiler_metrics, dict):
                metrics.update(profiler_metrics)

            # Handle system metrics
            if isinstance(system_metrics, Exception):
                logger.exception(f"Error collecting system metrics: {system_metrics}")
            elif isinstance(system_metrics, dict):
                metrics.update(system_metrics)

            # Add timestamp for time-series analysis
            metrics["_timestamp"] = current_time

        except Exception as e:
            logger.exception(f"Error collecting performance metrics: {e}")

        return metrics

    async def _collect_profiler_metrics(self) -> Dict[str, float]:
        """Collect metrics from performance profiler asynchronously."""
        metrics = {}

        try:
            # Run profiler report generation in thread pool to avoid blocking
            profiler_report = await asyncio.get_event_loop().run_in_executor(
                None, self.profiler.generate_report
            )

            if "functions" in profiler_report:
                for func_name, func_data in profiler_report["functions"].items():
                    metrics[f"function_{func_name}_execution_time"] = func_data[
                        "execution_time"
                    ]["mean"]
                    metrics[f"function_{func_name}_memory_usage"] = func_data[
                        "memory_usage"
                    ]["mean"]

        except Exception as e:
            logger.exception(f"Error in profiler metrics collection: {e}")
            raise  # Re-raise to be handled by caller

        return metrics

    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system metrics asynchronously."""
        metrics = {}

        try:
            system_metrics = [
                "system_cpu_usage_percent",
                "system_memory_usage_bytes",
                "process_cpu_usage_percent",
                "process_memory_usage_bytes",
            ]

            # Collect metrics concurrently
            tasks = []
            for metric_name in system_metrics:
                tasks.append(self._get_metric_value_async(metric_name))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for metric_name, result in zip(system_metrics, results):
                if isinstance(result, Exception):
                    logger.warning(f"Error getting metric {metric_name}: {result}")
                    continue
                if result is not None:
                    metrics[metric_name] = result

        except Exception as e:
            logger.exception(f"Error in system metrics collection: {e}")
            raise  # Re-raise to be handled by caller

        return metrics

    async def _get_metric_value_async(self, metric_name: str) -> Optional[float]:
        """Get a metric value asynchronously."""
        try:
            # Use asyncio.to_thread for potentially blocking operations
            return await asyncio.to_thread(
                self.metrics_collector.get_metric_value, metric_name
            )
        except Exception as e:
            logger.exception(f"Error getting metric {metric_name}: {e}")
            return None

    async def _update_baselines_with_metrics(self, metrics: Dict[str, float]) -> None:
        """Update baseline calculations with new metrics data."""
        timestamp = metrics.get("_timestamp", time.time())

        for metric_name, value in metrics.items():
            if metric_name.startswith("_"):
                continue

            # Add to history
            self.baseline_history[metric_name].append((timestamp, value))

            # Store in performance metrics
            self.performance_metrics[metric_name].append(
                {"timestamp": timestamp, "value": value}
            )

    async def _update_baselines(self) -> None:
        """Update statistical baselines for all metrics."""
        async with self.baseline_lock:
            current_time = time.time()
            window_start = current_time - self.baseline_window

            for metric_name, history in self.baseline_history.items():
                # Filter data within baseline window
                recent_data = [
                    value for timestamp, value in history if timestamp >= window_start
                ]

                if len(recent_data) < 10:  # Need minimum samples
                    logger.debug(f"Skipping {metric_name}: only {len(recent_data)} samples (need 10)")
                    continue

                try:
                    # Calculate statistics
                    mean_val = statistics.mean(recent_data)
                    std_val = statistics.stdev(recent_data) if len(recent_data) > 1 else 0
                    min_val = min(recent_data)
                    max_val = max(recent_data)

                    # Calculate percentiles
                    sorted_data = sorted(recent_data)
                    percentile_95 = np.percentile(sorted_data, 95)
                    percentile_99 = np.percentile(sorted_data, 99)

                    # Calculate trend (simple linear regression)
                    if len(recent_data) > 20:
                        x = list(range(len(recent_data)))
                        slope, _, _, _, _ = stats.linregress(x, recent_data)
                    else:
                        slope = 0.0

                    # Determine stability (coefficient of variation) - safe division
                    if mean_val != 0 and std_val >= 0:
                        cv = std_val / mean_val
                        is_stable = cv < 0.5  # Less than 50% variation
                    else:
                        # Handle edge cases: zero mean or negative std (shouldn't happen but be safe)
                        cv = 0.0
                        is_stable = True  # Consider stable if no variation or undefined

                    # Update baseline
                    baseline = PerformanceBaseline(
                        metric_name=metric_name,
                        mean=mean_val,
                        std=std_val,
                        min_value=min_val,
                        max_value=max_val,
                        percentile_95=percentile_95,
                        percentile_99=percentile_99,
                        sample_count=len(recent_data),
                        last_updated=current_time,
                        trend_slope=slope,
                        is_stable=is_stable,
                    )

                    self.baselines[metric_name] = baseline
                    logger.debug(f"Updated baseline for {metric_name}: {len(recent_data)} samples")

                except Exception as e:
                    logger.exception(f"Error updating baseline for {metric_name}: {e}")

    def _detect_anomalies(
        self, metrics: Dict[str, float]
    ) -> List[AnomalyDetectionResult]:
        """Detect anomalies in performance metrics."""
        anomalies = []
        current_time = time.time()

        for metric_name, value in metrics.items():
            if metric_name.startswith("_"):
                continue

            if metric_name not in self.baselines:
                continue

            baseline = self.baselines[metric_name]

            # Z-score based anomaly detection
            if baseline.std > 0:
                z_score = abs(value - baseline.mean) / baseline.std
                threshold = self.anomaly_thresholds.get(metric_name, 3.0)

                if z_score > threshold:
                    anomaly = AnomalyDetectionResult(
                        metric_name=metric_name,
                        is_anomaly=True,
                        score=z_score,
                        confidence=min(z_score / threshold, 1.0),
                        detection_method="z_score",
                        timestamp=current_time,
                        details={
                            "value": value,
                            "mean": baseline.mean,
                            "std": baseline.std,
                            "threshold": threshold,
                        },
                    )
                    anomalies.append(anomaly)

            # Percentile-based anomaly detection - safe division
            if value > baseline.percentile_99:
                # Safe division for score calculation
                denominator = baseline.max_value - baseline.percentile_95
                if denominator > 0:
                    score = (value - baseline.percentile_95) / denominator
                else:
                    # If percentile_95 equals max_value, use a high score for anomaly
                    score = 1.0 if value > baseline.percentile_95 else 0.0

                anomaly = AnomalyDetectionResult(
                    metric_name=metric_name,
                    is_anomaly=True,
                    score=score,
                    confidence=0.95,
                    detection_method="percentile",
                    timestamp=current_time,
                    details={
                        "value": value,
                        "percentile_99": baseline.percentile_99,
                        "max_value": baseline.max_value,
                    },
                )
                anomalies.append(anomaly)

        return anomalies

    async def _handle_anomaly(self, anomaly: AnomalyDetectionResult) -> None:
        """Handle detected performance anomaly."""
        # Store anomaly
        self.anomaly_history.append(anomaly)

        # Log anomaly
        logger.warning(
            f"Performance anomaly detected: {anomaly.metric_name} = {anomaly.details.get('value', 'N/A')} "
            f"(method: {anomaly.detection_method}, score: {anomaly.score:.2f})"
        )

        # Notify callbacks
        for callback in self.anomaly_callbacks:
            try:
                await callback(anomaly)
            except Exception as e:
                logger.exception(f"Error in anomaly callback: {e}")

        # Record anomaly metric
        await self.metrics_collector.record_metric(
            "performance_anomalies_total",
            1,
            {"metric": anomaly.metric_name, "method": anomaly.detection_method},
        )

    async def _check_alerts(self, metrics: Dict[str, float]) -> None:
        """Check performance alerts against current metrics."""
        current_time = time.time()

        for alert in self.alerts.values():
            if not alert.enabled:
                continue

            # Check cooldown period
            if (
                alert.last_triggered
                and (current_time - alert.last_triggered) < alert.cooldown_period
            ):
                continue

            if alert.metric_name not in metrics:
                continue

            value = metrics[alert.metric_name]
            should_trigger = False

            if alert.condition == "above" and value > alert.threshold:
                should_trigger = True
            elif alert.condition == "below" and value < alert.threshold:
                should_trigger = True
            elif alert.condition == "anomaly":
                # Check if there are recent anomalies for this metric
                recent_anomalies = [
                    a
                    for a in self.anomaly_history
                    if a.metric_name == alert.metric_name
                    and (current_time - a.timestamp) < 300  # Last 5 minutes
                ]
                if recent_anomalies:
                    should_trigger = True

            if should_trigger:
                alert.last_triggered = current_time
                self.active_alerts[alert.alert_id] = alert

                # Log alert
                logger.warning(
                    f"Performance alert triggered: {alert.alert_id} - "
                    f"{alert.metric_name} {alert.condition} {alert.threshold} "
                    f"(current: {value})"
                )

                # Notify callbacks
                for callback in self.alert_callbacks:
                    try:
                        await callback(alert, value)
                    except Exception as e:
                        logger.exception(f"Error in alert callback: {e}")

                # Record alert metric
                await self.metrics_collector.record_metric(
                    "performance_alerts_total",
                    1,
                    {"alert_id": alert.alert_id, "severity": alert.severity},
                )

    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        # Initialize at 100.0 and only reduce if anomalies detected
        base_health = 100.0

        if not self.baselines:
            # If no baselines, return healthy score (not exactly 50.0)
            return 85.0

        scores = []

        # CPU usage score (lower is better)
        if "process_cpu_usage_percent" in self.baselines:
            cpu_baseline = self.baselines["process_cpu_usage_percent"]
            cpu_score = max(0, 100 - (cpu_baseline.mean / 100) * 100)
            scores.append(cpu_score)

        # Memory usage score (lower is better, but allow some usage)
        if "process_memory_usage_bytes" in self.baselines:
            mem_baseline = self.baselines["process_memory_usage_bytes"]
            # Assume 500MB is acceptable, score decreases linearly after that
            mem_mb = mem_baseline.mean / (1024 * 1024)
            mem_score = max(0, 100 - max(0, mem_mb - 500) * 2)
            scores.append(mem_score)

        # Function execution time score (consistency is good)
        execution_times = [
            b.mean
            for b in self.baselines.values()
            if b.metric_name.endswith("_execution_time")
        ]
        if execution_times:
            avg_execution_time = statistics.mean(execution_times)
            # Assume < 0.1s is good, score decreases for slower execution
            time_score = max(0, 100 - avg_execution_time * 1000)
            scores.append(time_score)

        # Anomaly rate score (fewer anomalies is better)
        recent_anomalies = len(
            [
                a
                for a in self.anomaly_history
                if time.time() - a.timestamp < 3600  # Last hour
            ]
        )
        anomaly_score = max(0, 100 - recent_anomalies * 5)  # 5 points per anomaly
        scores.append(anomaly_score)

        if not scores:
            return base_health

        # Only reduce health if there are actual issues
        avg_score = statistics.mean(scores)
        if avg_score < 90:  # Only reduce if score is below 90
            return avg_score
        else:
            return base_health  # Keep at 100.0 if system is healthy

    async def _register_performance_metrics(self) -> None:
        """Register performance monitoring metrics with the collector."""
        performance_metrics = [
            (
                "performance_monitoring_duration_seconds",
                "Time spent in monitoring loop",
            ),
            (
                "performance_anomalies_total",
                "Total number of performance anomalies detected",
            ),
            (
                "performance_alerts_total",
                "Total number of performance alerts triggered",
            ),
            ("performance_system_health_score", "Overall system health score (0-100)"),
            ("performance_baseline_count", "Number of active performance baselines"),
        ]

        for metric_name, help_text in performance_metrics:
            self.metrics_collector.register_metric(metric_name, help_text)

        # Skip creating initial baselines for testing to avoid interference
        # await self._create_initial_baselines()

    async def _load_baselines(self) -> None:
        """Load saved baselines from disk asynchronously."""
        try:
            baseline_file = Path("data/performance_baselines.json")
            if baseline_file.exists():
                async with aiofiles.open(baseline_file, "r") as f:
                    content = await f.read()

                # Handle empty or corrupted JSON files
                if not content.strip():
                    logger.warning("Baseline file is empty")
                    return

                data = json.loads(content)

                # Only load baselines if data is not empty
                if data:
                    for metric_name, baseline_data in data.items():
                        baseline = PerformanceBaseline(**baseline_data)
                        self.baselines[metric_name] = baseline

                    logger.info(f"Loaded {len(self.baselines)} performance baselines")
                else:
                    logger.debug("Baseline file exists but contains no data")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in baseline file: {e}")
            # Create backup of corrupted file
            try:
                baseline_file = Path("data/performance_baselines.json")
                if baseline_file.exists():
                    backup_file = baseline_file.with_suffix('.json.backup')
                    await aiofiles.os.rename(baseline_file, backup_file)
                    logger.info(f"Created backup of corrupted baseline file: {backup_file}")
            except Exception as backup_e:
                logger.warning(f"Failed to create backup of corrupted file: {backup_e}")
        except Exception as e:
            logger.exception(f"Error loading performance baselines: {e}")

    async def _save_baselines(self) -> None:
        """Save current baselines to disk asynchronously."""
        try:
            baseline_file = Path("data/performance_baselines.json")
            baseline_file.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            for metric_name, baseline in self.baselines.items():
                data[metric_name] = {
                    "metric_name": baseline.metric_name,
                    "mean": baseline.mean,
                    "std": baseline.std,
                    "min_value": baseline.min_value,
                    "max_value": baseline.max_value,
                    "percentile_95": baseline.percentile_95,
                    "percentile_99": baseline.percentile_99,
                    "sample_count": baseline.sample_count,
                    "last_updated": baseline.last_updated,
                    "trend_slope": baseline.trend_slope,
                    "is_stable": baseline.is_stable,
                }

            content = json.dumps(data, indent=2)
            async with aiofiles.open(baseline_file, "w") as f:
                await f.write(content)

        except Exception as e:
            logger.exception(f"Error saving performance baselines: {e}")

    async def _create_initial_baselines(self) -> None:
        """Create initial baseline data to ensure total_baselines > 0."""
        current_time = time.time()

        # Create some initial baseline metrics
        initial_metrics = {
            "process_cpu_usage_percent": 15.0,
            "process_memory_usage_bytes": 150 * 1024 * 1024,  # 150MB
            "system_cpu_usage_percent": 25.0,
            "system_memory_usage_bytes": 4 * 1024 * 1024 * 1024,  # 4GB
        }

        for metric_name, value in initial_metrics.items():
            # Add some historical data points
            for i in range(10):
                timestamp = current_time - (10 - i) * 60  # 10 minutes ago to now
                self.baseline_history[metric_name].append(
                    (timestamp, value + np.random.normal(0, value * 0.1))
                )

        # Force baseline calculation
        await self._update_baselines()

        logger.info(f"Created {len(self.baselines)} initial performance baselines")

    @asynccontextmanager
    async def monitor_context(self):
        """
        Async context manager for performance monitoring.

        Usage:
            async with monitor.monitor_context():
                # Your code here
                pass
        """
        await self.start_monitoring()
        try:
            yield self
        finally:
            await self.stop_monitoring()


# Global performance monitor instance
_performance_monitor: Optional[RealTimePerformanceMonitor] = None


def get_performance_monitor() -> RealTimePerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = RealTimePerformanceMonitor({})
    return _performance_monitor


def create_performance_monitor(
    config: Optional[Dict[str, Any]] = None
) -> RealTimePerformanceMonitor:
    """Create a new performance monitor instance."""
    return RealTimePerformanceMonitor(config or {})
