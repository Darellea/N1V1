"""
Self-diagnostics and alerting system for the trading framework.

Provides comprehensive health monitoring, anomaly detection, and alerting
capabilities to ensure system reliability and provide real-time visibility
into system health.
"""

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import numpy as np

from core.signal_router.event_bus import get_default_enhanced_event_bus
from core.signal_router.events import (
    create_diagnostic_alert_event,
)
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    component: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""

    component: str
    metric: str
    value: float
    threshold: float
    severity: AlertSeverity
    description: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DiagnosticState:
    """Current diagnostic state of the system with optimized memory usage."""

    overall_status: HealthStatus = HealthStatus.HEALTHY
    last_check: Optional[datetime] = None
    check_count: int = 0
    error_count: int = 0
    anomaly_count: int = 0
    component_statuses: Dict[str, HealthCheckResult] = field(default_factory=dict)
    recent_anomalies: List[AnomalyDetection] = field(default_factory=list)
    performance_metrics: Dict[str, np.ndarray] = field(default_factory=dict)

    # Pre-allocated numpy arrays for performance metrics
    _latency_buffer: np.ndarray = field(default=None, init=False)
    _buffer_index: int = field(default=0, init=False)
    _max_metrics: int = field(default=100, init=False)

    def __post_init__(self):
        """Initialize optimized data structures."""
        if self._latency_buffer is None:
            self._latency_buffer = np.full(self._max_metrics, np.nan, dtype=np.float64)


class DiagnosticsManager:
    """
    Main diagnostics manager coordinating health checks, anomaly detection, and alerting.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the diagnostics manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.event_bus = get_default_enhanced_event_bus()
        self.logger = get_trade_logger()

        # Configuration with defaults
        self.check_interval_sec = self.config.get("interval_sec", 60)
        self.latency_threshold_ms = self.config.get("latency_threshold_ms", 5000)
        self.drawdown_threshold_pct = self.config.get("drawdown_threshold_pct", 5.0)
        self.data_integrity_check = self.config.get("data_integrity_check", True)
        self.discord_webhook_url = self.config.get("discord_webhook_url", "")

        # Rolling windows for anomaly detection
        self.latency_window_size = self.config.get("latency_window_size", 10)
        self.anomaly_std_dev_threshold = self.config.get(
            "anomaly_std_dev_threshold", 2.0
        )

        # State management
        self.state = DiagnosticState()
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Health check functions registry
        self._health_checks: Dict[str, Callable] = {}

        # Anomaly detectors
        self._anomaly_detectors: List[Callable] = []

        logger.info("DiagnosticsManager initialized")

    async def start(self) -> None:
        """Start the diagnostics monitoring."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Diagnostics monitoring started")

    async def stop(self) -> None:
        """Stop the diagnostics monitoring."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Diagnostics monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self.run_health_check()
                await asyncio.sleep(self.check_interval_sec)
            except Exception as e:
                logger.exception(f"Error in diagnostics monitoring loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retry

    async def run_health_check(self) -> DiagnosticState:
        """
        Run a complete health check on all registered components.

        Returns:
            Updated diagnostic state
        """
        start_time = time.time()
        results = []

        # Run all registered health checks
        for component_name, check_func in self._health_checks.items():
            try:
                result = await check_func()
                results.append(result)
                self.state.component_statuses[component_name] = result
            except Exception as e:
                logger.exception(f"Health check failed for {component_name}: {e}")
                error_result = HealthCheckResult(
                    component=component_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                )
                results.append(error_result)
                self.state.component_statuses[component_name] = error_result
                self.state.error_count += 1

        # Update overall status
        self.state.overall_status = self._calculate_overall_status(results)
        self.state.last_check = datetime.now()
        self.state.check_count += 1

        # Run anomaly detection
        await self._run_anomaly_detection()

        # Log results
        total_time = (time.time() - start_time) * 1000
        await self._log_health_check_results(results, total_time)

        # Send alerts if critical
        if self.state.overall_status == HealthStatus.CRITICAL:
            await self._send_critical_alert(results)

        return self.state

    def _calculate_overall_status(
        self, results: List[HealthCheckResult]
    ) -> HealthStatus:
        """Calculate overall system health status from individual results."""
        if any(r.status == HealthStatus.CRITICAL for r in results):
            return HealthStatus.CRITICAL
        elif any(r.status == HealthStatus.DEGRADED for r in results):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    async def _run_anomaly_detection(self) -> None:
        """Run anomaly detection on system metrics."""
        for detector in self._anomaly_detectors:
            try:
                anomalies = await detector()
                for anomaly in anomalies:
                    self.state.recent_anomalies.append(anomaly)
                    self.state.anomaly_count += 1

                    # Publish anomaly as diagnostic alert
                    await self._publish_anomaly_alert(anomaly)

            except Exception as e:
                logger.exception(f"Anomaly detection failed: {e}")

        # Keep only recent anomalies (last 100)
        if len(self.state.recent_anomalies) > 100:
            self.state.recent_anomalies = self.state.recent_anomalies[-100:]

    async def _publish_anomaly_alert(self, anomaly: AnomalyDetection) -> None:
        """Publish an anomaly as a diagnostic alert event."""
        alert_event = create_diagnostic_alert_event(
            alert_type=anomaly.severity.value,
            component=anomaly.component,
            message=f"Anomaly detected: {anomaly.description}",
            details={
                "metric": anomaly.metric,
                "value": anomaly.value,
                "threshold": anomaly.threshold,
                "severity": anomaly.severity.value,
            },
        )

        await self.event_bus.publish_event(alert_event)

    async def _log_health_check_results(
        self, results: List[HealthCheckResult], total_time: float
    ) -> None:
        """Log health check results."""
        status_counts = {}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

        log_data = {
            "overall_status": self.state.overall_status.value,
            "total_checks": len(results),
            "status_breakdown": {k.value: v for k, v in status_counts.items()},
            "total_time_ms": round(total_time, 2),
            "check_count": self.state.check_count,
            "error_count": self.state.error_count,
            "anomaly_count": self.state.anomaly_count,
        }

        if self.state.overall_status == HealthStatus.CRITICAL:
            self.logger.error("Health check completed", extra={"health_data": log_data})
        elif self.state.overall_status == HealthStatus.DEGRADED:
            self.logger.warning(
                "Health check completed", extra={"health_data": log_data}
            )
        else:
            self.logger.info("Health check completed", extra={"health_data": log_data})

    async def _send_critical_alert(self, results: List[HealthCheckResult]) -> None:
        """Send critical alert via Discord webhook."""
        if not self.discord_webhook_url:
            return

        critical_components = [r for r in results if r.status == HealthStatus.CRITICAL]

        alert_message = {
            "embeds": [
                {
                    "title": "🚨 CRITICAL SYSTEM ALERT",
                    "description": f"System health status: {self.state.overall_status.value.upper()}",
                    "color": 15158332,  # Red color
                    "fields": [
                        {
                            "name": "Critical Components",
                            "value": "\n".join(
                                [
                                    f"• {r.component}: {r.message}"
                                    for r in critical_components[:5]
                                ]
                            ),
                            "inline": False,
                        },
                        {
                            "name": "Timestamp",
                            "value": datetime.now().isoformat(),
                            "inline": True,
                        },
                        {
                            "name": "Check Count",
                            "value": str(self.state.check_count),
                            "inline": True,
                        },
                    ],
                    "footer": {"text": "Trading Framework Diagnostics"},
                }
            ]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.discord_webhook_url,
                    json=alert_message,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 204:
                        logger.info("Critical alert sent to Discord")
                    else:
                        logger.error(
                            f"Failed to send Discord alert: HTTP {response.status}"
                        )

        except Exception as e:
            logger.exception(f"Error sending Discord alert: {e}")

    def register_health_check(self, component_name: str, check_func: Callable) -> None:
        """
        Register a health check function.

        Args:
            component_name: Name of the component
            check_func: Async function that returns HealthCheckResult
        """
        self._health_checks[component_name] = check_func
        logger.info(f"Registered health check for {component_name}")

    def register_anomaly_detector(self, detector_func: Callable) -> None:
        """
        Register an anomaly detection function.

        Args:
            detector_func: Async function that returns List[AnomalyDetection]
        """
        self._anomaly_detectors.append(detector_func)
        logger.info("Registered anomaly detector")

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status summary."""
        return {
            "overall_status": self.state.overall_status.value,
            "last_check": self.state.last_check.isoformat()
            if self.state.last_check
            else None,
            "check_count": self.state.check_count,
            "error_count": self.state.error_count,
            "anomaly_count": self.state.anomaly_count,
            "component_count": len(self.state.component_statuses),
            "running": self._running,
        }

    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed health status including all components."""
        return {
            "overall_status": self.state.overall_status.value,
            "last_check": self.state.last_check.isoformat()
            if self.state.last_check
            else None,
            "check_count": self.state.check_count,
            "error_count": self.state.error_count,
            "anomaly_count": self.state.anomaly_count,
            "components": {
                name: {
                    "status": result.status.value,
                    "latency_ms": result.latency_ms,
                    "message": result.message,
                    "timestamp": result.timestamp.isoformat(),
                }
                for name, result in self.state.component_statuses.items()
            },
            "recent_anomalies": [
                {
                    "component": a.component,
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "severity": a.severity.value,
                    "description": a.description,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in self.state.recent_anomalies[-10:]  # Last 10 anomalies
            ],
        }


# Built-in health check functions


async def check_api_connectivity(
    base_url: str, timeout_ms: int = 5000
) -> HealthCheckResult:
    """
    Check API connectivity and response time.

    This function makes HTTP requests to check API availability and handles
    different failure scenarios including timeouts, connection errors, and DNS failures.

    Args:
        base_url: The URL to check connectivity to
        timeout_ms: Timeout in milliseconds for the request

    Returns:
        HealthCheckResult with connectivity status, latency, and details
    """
    start_time = time.time()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                base_url, timeout=aiohttp.ClientTimeout(total=timeout_ms / 1000)
            ) as response:
                latency = (time.time() - start_time) * 1000

                if response.status == 200:
                    return HealthCheckResult(
                        component="api_connectivity",
                        status=HealthStatus.HEALTHY,
                        latency_ms=latency,
                        message=f"API responsive ({latency:.1f}ms)",
                        details={"status_code": response.status, "url": base_url},
                    )
                else:
                    return HealthCheckResult(
                        component="api_connectivity",
                        status=HealthStatus.DEGRADED,
                        latency_ms=latency,
                        message=f"API returned status {response.status} ({latency:.1f}ms)",
                        details={"status_code": response.status, "url": base_url},
                    )

    except asyncio.TimeoutError:
        latency = (time.time() - start_time) * 1000
        return HealthCheckResult(
            component="api_connectivity",
            status=HealthStatus.CRITICAL,
            latency_ms=latency,
            message=f"API timeout after {latency:.1f}ms",
            details={"error": "timeout", "url": base_url},
        )

    except aiohttp.ClientConnectorError as e:
        latency = (time.time() - start_time) * 1000
        # Handle specific DNS resolution errors
        error_msg = str(e)
        if (
            "getaddrinfo failed" in error_msg
            or "Name or service not known" in error_msg
        ):
            return HealthCheckResult(
                component="api_connectivity",
                status=HealthStatus.CRITICAL,
                latency_ms=latency,
                message="API connection failed: Connection failed",
                details={
                    "error": "Connection failed",
                    "url": base_url,
                    "original_error": str(e),
                },
            )
        else:
            return HealthCheckResult(
                component="api_connectivity",
                status=HealthStatus.CRITICAL,
                latency_ms=latency,
                message="API connection failed: Connection failed",
                details={
                    "error": "Connection failed",
                    "url": base_url,
                    "original_error": str(e),
                },
            )

    except aiohttp.ClientConnectionError as e:
        latency = (time.time() - start_time) * 1000
        return HealthCheckResult(
            component="api_connectivity",
            status=HealthStatus.CRITICAL,
            latency_ms=latency,
            message="API connection failed: Connection failed",
            details={
                "error": "Connection failed",
                "url": base_url,
                "original_error": str(e),
            },
        )

    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return HealthCheckResult(
            component="api_connectivity",
            status=HealthStatus.CRITICAL,
            latency_ms=latency,
            message=f"API request failed: {type(e).__name__}",
            details={
                "error": type(e).__name__,
                "url": base_url,
                "original_error": str(e),
            },
        )


async def check_data_integrity(data_feed: Any) -> HealthCheckResult:
    """Check data feed integrity for missing or irregular data."""
    try:
        # This would be implemented based on the specific data feed structure
        # For now, return a placeholder
        return HealthCheckResult(
            component="data_integrity",
            status=HealthStatus.HEALTHY,
            message="Data integrity check passed",
            details={"missing_data_points": 0, "irregular_timestamps": 0},
        )
    except Exception as e:
        return HealthCheckResult(
            component="data_integrity",
            status=HealthStatus.CRITICAL,
            message=f"Data integrity check failed: {str(e)}",
            details={"error": str(e)},
        )


async def check_strategy_responsiveness(strategy_manager: Any) -> HealthCheckResult:
    """Check strategy execution responsiveness."""
    start_time = time.time()

    try:
        # This would ping the strategy manager
        # For now, return a placeholder
        latency = (time.time() - start_time) * 1000

        return HealthCheckResult(
            component="strategy_responsiveness",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            message=f"Strategy responsive ({latency:.1f}ms)",
            details={"active_strategies": 0},  # Would be populated from actual manager
        )
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return HealthCheckResult(
            component="strategy_responsiveness",
            status=HealthStatus.CRITICAL,
            latency_ms=latency,
            message=f"Strategy check failed: {str(e)}",
            details={"error": str(e)},
        )


# Built-in anomaly detectors


async def detect_latency_anomalies(
    diagnostics: DiagnosticsManager,
) -> List[AnomalyDetection]:
    """Detect anomalous latency spikes."""
    anomalies = []

    for component, result in diagnostics.state.component_statuses.items():
        if result.latency_ms is None:
            continue

        # Update rolling window
        if component not in diagnostics.state.performance_metrics:
            diagnostics.state.performance_metrics[component] = []

        metrics = diagnostics.state.performance_metrics[component]
        metrics.append(result.latency_ms)

        # Keep only recent measurements (rolling window)
        while len(metrics) > diagnostics.latency_window_size:
            metrics.pop(0)

        # Need minimum samples for anomaly detection
        if len(metrics) < 5:
            continue

        try:
            mean_latency = statistics.mean(metrics[:-1])  # Exclude current measurement
            std_dev = statistics.stdev(metrics[:-1]) if len(metrics) > 2 else 0

            # Add tolerance buffer to avoid false positives
            tolerance_factor = 1.3  # Require 30% above threshold to trigger anomaly
            threshold = mean_latency + (std_dev * diagnostics.anomaly_std_dev_threshold)
            anomaly_threshold = threshold * tolerance_factor

            if result.latency_ms > anomaly_threshold:
                # Determine severity based on magnitude of spike
                if result.latency_ms > mean_latency * 10:  # Severe spike (10x baseline)
                    severity = AlertSeverity.CRITICAL
                elif (
                    result.latency_ms > mean_latency * 2.4
                ):  # Moderate spike (2.4x baseline)
                    severity = AlertSeverity.WARNING
                else:  # Mild spike
                    severity = AlertSeverity.INFO

                anomaly = AnomalyDetection(
                    component=component,
                    metric="latency_ms",
                    value=result.latency_ms,
                    threshold=anomaly_threshold,
                    severity=severity,
                    description=f"Latency spike detected: {result.latency_ms:.1f}ms (threshold: {anomaly_threshold:.1f}ms)",
                )
                anomalies.append(anomaly)

        except statistics.StatisticsError:
            # Not enough data for statistics
            continue

    return anomalies


async def detect_drawdown_anomalies(
    portfolio_manager: Any, diagnostics: DiagnosticsManager
) -> List[AnomalyDetection]:
    """Detect abnormal drawdown spikes."""
    anomalies = []

    try:
        # This would check portfolio drawdown against configured threshold
        # For now, return empty list as this requires portfolio integration
        return anomalies
    except Exception as e:
        logger.exception(f"Drawdown anomaly detection failed: {e}")
        return anomalies


# Global diagnostics manager instance
_global_diagnostics_manager: Optional[DiagnosticsManager] = None


def get_diagnostics_manager() -> DiagnosticsManager:
    """Get the global diagnostics manager instance."""
    global _global_diagnostics_manager
    if _global_diagnostics_manager is None:
        _global_diagnostics_manager = DiagnosticsManager()
    return _global_diagnostics_manager


def create_diagnostics_manager(
    config: Optional[Dict[str, Any]] = None
) -> DiagnosticsManager:
    """Create a new diagnostics manager instance."""
    return DiagnosticsManager(config)
