"""
Test suite for the diagnostics system.

Tests cover health checks, anomaly detection, alerting, and system monitoring.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from core.diagnostics import (
    DiagnosticsManager,
    HealthStatus,
    AlertSeverity,
    HealthCheckResult,
    AnomalyDetection,
    check_api_connectivity,
    detect_latency_anomalies,
    create_diagnostics_manager
)


class TestHealthStatus:
    """Test health status enum and logic."""

    def test_health_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.CRITICAL.value == "critical"

    def test_alert_severity_values(self):
        """Test alert severity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestHealthCheckResult:
    """Test health check result data structure."""

    def test_health_check_result_creation(self):
        """Test creating a health check result."""
        result = HealthCheckResult(
            component="test_component",
            status=HealthStatus.HEALTHY,
            latency_ms=150.5,
            message="Test passed",
            details={"test": "data"}
        )

        assert result.component == "test_component"
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms == 150.5
        assert result.message == "Test passed"
        assert result.details == {"test": "data"}
        assert isinstance(result.timestamp, datetime)


class TestAnomalyDetection:
    """Test anomaly detection data structure."""

    def test_anomaly_detection_creation(self):
        """Test creating an anomaly detection result."""
        anomaly = AnomalyDetection(
            component="api_connectivity",
            metric="latency_ms",
            value=2500.0,
            threshold=2000.0,
            severity=AlertSeverity.WARNING,
            description="Latency spike detected"
        )

        assert anomaly.component == "api_connectivity"
        assert anomaly.metric == "latency_ms"
        assert anomaly.value == 2500.0
        assert anomaly.threshold == 2000.0
        assert anomaly.severity == AlertSeverity.WARNING
        assert anomaly.description == "Latency spike detected"
        assert isinstance(anomaly.timestamp, datetime)


class TestDiagnosticsManager:
    """Test the main diagnostics manager."""

    @pytest.fixture
    def diagnostics_manager(self):
        """Create a diagnostics manager for testing."""
        return DiagnosticsManager()

    def test_initialization(self, diagnostics_manager):
        """Test diagnostics manager initialization."""
        assert diagnostics_manager.check_interval_sec == 60
        assert diagnostics_manager.latency_threshold_ms == 5000
        assert diagnostics_manager.drawdown_threshold_pct == 5.0
        assert diagnostics_manager.state.overall_status == HealthStatus.HEALTHY
        assert not diagnostics_manager._running
        assert len(diagnostics_manager._health_checks) == 0
        assert len(diagnostics_manager._anomaly_detectors) == 0

    def test_custom_config(self):
        """Test diagnostics manager with custom config."""
        config = {
            'interval_sec': 30,
            'latency_threshold_ms': 3000,
            'drawdown_threshold_pct': 10.0
        }
        diagnostics = DiagnosticsManager(config)

        assert diagnostics.check_interval_sec == 30
        assert diagnostics.latency_threshold_ms == 3000
        assert diagnostics.drawdown_threshold_pct == 10.0

    @pytest.mark.asyncio
    async def test_register_health_check(self, diagnostics_manager):
        """Test registering health check functions."""
        async def mock_check():
            return HealthCheckResult(
                component="mock",
                status=HealthStatus.HEALTHY,
                message="Mock check passed"
            )

        diagnostics_manager.register_health_check("mock_check", mock_check)

        assert "mock_check" in diagnostics_manager._health_checks
        assert len(diagnostics_manager._health_checks) == 1

    @pytest.mark.asyncio
    async def test_register_anomaly_detector(self, diagnostics_manager):
        """Test registering anomaly detector functions."""
        async def mock_detector():
            return []

        diagnostics_manager.register_anomaly_detector(mock_detector)

        assert len(diagnostics_manager._anomaly_detectors) == 1

    @pytest.mark.asyncio
    async def test_run_health_check_no_checks(self, diagnostics_manager):
        """Test running health check with no registered checks."""
        state = await diagnostics_manager.run_health_check()

        assert state.overall_status == HealthStatus.HEALTHY
        assert state.check_count == 1
        assert len(state.component_statuses) == 0

    @pytest.mark.asyncio
    async def test_run_health_check_with_healthy_component(self, diagnostics_manager):
        """Test running health check with healthy component."""
        async def healthy_check():
            return HealthCheckResult(
                component="test_component",
                status=HealthStatus.HEALTHY,
                latency_ms=100.0,
                message="All good"
            )

        diagnostics_manager.register_health_check("test_component", healthy_check)
        state = await diagnostics_manager.run_health_check()

        assert state.overall_status == HealthStatus.HEALTHY
        assert state.check_count == 1
        assert "test_component" in state.component_statuses
        assert state.component_statuses["test_component"].status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_run_health_check_with_critical_component(self, diagnostics_manager):
        """Test running health check with critical component."""
        async def critical_check():
            return HealthCheckResult(
                component="failing_component",
                status=HealthStatus.CRITICAL,
                message="Component failed"
            )

        diagnostics_manager.register_health_check("failing_component", critical_check)
        state = await diagnostics_manager.run_health_check()

        assert state.overall_status == HealthStatus.CRITICAL
        assert state.check_count == 1
        assert state.component_statuses["failing_component"].status == HealthStatus.CRITICAL

    @pytest.mark.asyncio
    async def test_run_health_check_with_mixed_statuses(self, diagnostics_manager):
        """Test running health check with mixed component statuses."""
        async def healthy_check():
            return HealthCheckResult(
                component="healthy_comp",
                status=HealthStatus.HEALTHY,
                message="OK"
            )

        async def degraded_check():
            return HealthCheckResult(
                component="degraded_comp",
                status=HealthStatus.DEGRADED,
                message="Slow"
            )

        diagnostics_manager.register_health_check("healthy_comp", healthy_check)
        diagnostics_manager.register_health_check("degraded_comp", degraded_check)

        state = await diagnostics_manager.run_health_check()

        assert state.overall_status == HealthStatus.DEGRADED  # Degraded takes precedence over healthy
        assert state.check_count == 1

    @pytest.mark.asyncio
    async def test_run_health_check_with_exception(self, diagnostics_manager):
        """Test running health check when component throws exception."""
        async def failing_check():
            raise Exception("Test exception")

        diagnostics_manager.register_health_check("failing_check", failing_check)
        state = await diagnostics_manager.run_health_check()

        assert state.overall_status == HealthStatus.CRITICAL
        assert state.error_count == 1
        assert "failing_check" in state.component_statuses
        assert state.component_statuses["failing_check"].status == HealthStatus.CRITICAL

    def test_get_health_status(self, diagnostics_manager):
        """Test getting health status summary."""
        status = diagnostics_manager.get_health_status()

        expected_keys = [
            "overall_status", "last_check", "check_count",
            "error_count", "anomaly_count", "component_count", "running"
        ]

        for key in expected_keys:
            assert key in status

        assert status["overall_status"] == "healthy"
        assert status["running"] is False

    def test_get_detailed_status(self, diagnostics_manager):
        """Test getting detailed health status."""
        # Add some mock component data
        diagnostics_manager.state.component_statuses["test_comp"] = HealthCheckResult(
            component="test_comp",
            status=HealthStatus.HEALTHY,
            latency_ms=150.0,
            message="Test message"
        )

        status = diagnostics_manager.get_detailed_status()

        assert "components" in status
        assert "test_comp" in status["components"]
        assert status["components"]["test_comp"]["status"] == "healthy"
        assert status["components"]["test_comp"]["latency_ms"] == 150.0

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, diagnostics_manager):
        """Test starting and stopping the monitoring."""
        assert not diagnostics_manager._running

        await diagnostics_manager.start()
        assert diagnostics_manager._running

        await diagnostics_manager.stop()
        assert not diagnostics_manager._running


class TestBuiltInHealthChecks:
    """Test built-in health check functions."""

    @pytest.mark.asyncio
    async def test_check_api_connectivity_success(self):
        """Test successful API connectivity check."""
        # Mock the entire aiohttp session and response
        with patch('core.diagnostics.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            # Mock the context manager properly
            mock_response = AsyncMock()
            mock_response.status = 200

            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await check_api_connectivity("https://api.example.com", 5000)

            assert result.component == "api_connectivity"
            assert result.status == HealthStatus.HEALTHY
            assert "API responsive" in result.message
            assert result.latency_ms is not None
            assert result.details["status_code"] == 200

    @pytest.mark.asyncio
    async def test_check_api_connectivity_timeout(self):
        """Test API connectivity check with timeout."""
        with patch('core.diagnostics.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            # Mock timeout
            mock_session.get.side_effect = asyncio.TimeoutError()

            result = await check_api_connectivity("https://api.example.com", 1000)

            assert result.component == "api_connectivity"
            assert result.status == HealthStatus.CRITICAL
            assert "API timeout" in result.message
            assert result.details["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_check_api_connectivity_failure(self):
        """Test API connectivity check with connection failure."""
        with patch('core.diagnostics.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            # Mock connection error
            mock_session.get.side_effect = Exception("Connection failed")

            result = await check_api_connectivity("https://api.example.com", 5000)

            assert result.component == "api_connectivity"
            assert result.status == HealthStatus.CRITICAL
            assert "API connection failed" in result.message
            assert result.details["error"] == "Connection failed"


class TestAnomalyDetectors:
    """Test built-in anomaly detection functions."""

    @pytest.mark.asyncio
    async def test_detect_latency_anomalies_no_data(self):
        """Test latency anomaly detection with no data."""
        diagnostics = DiagnosticsManager()

        anomalies = await detect_latency_anomalies(diagnostics)

        assert len(anomalies) == 0

    @pytest.mark.asyncio
    async def test_detect_latency_anomalies_normal(self):
        """Test latency anomaly detection with normal latency."""
        diagnostics = DiagnosticsManager()

        # Add normal latency data
        diagnostics.state.component_statuses["api"] = HealthCheckResult(
            component="api",
            status=HealthStatus.HEALTHY,
            latency_ms=100.0
        )

        # Add historical data
        diagnostics.state.performance_metrics["api"] = [95.0, 105.0, 98.0, 102.0, 97.0]

        anomalies = await detect_latency_anomalies(diagnostics)

        assert len(anomalies) == 0

    @pytest.mark.asyncio
    async def test_detect_latency_anomalies_spike(self):
        """Test latency anomaly detection with latency spike."""
        diagnostics = DiagnosticsManager()

        # Add high latency data
        diagnostics.state.component_statuses["api"] = HealthCheckResult(
            component="api",
            status=HealthStatus.DEGRADED,
            latency_ms=500.0  # Much higher than normal
        )

        # Add historical data with low variance
        diagnostics.state.performance_metrics["api"] = [95.0, 105.0, 98.0, 102.0, 97.0]

        anomalies = await detect_latency_anomalies(diagnostics)

        assert len(anomalies) == 1
        assert anomalies[0].component == "api"
        assert anomalies[0].metric == "latency_ms"
        assert anomalies[0].value == 500.0
        assert anomalies[0].severity == AlertSeverity.WARNING
        assert "Latency spike detected" in anomalies[0].description

    @pytest.mark.asyncio
    async def test_detect_latency_anomalies_insufficient_data(self):
        """Test latency anomaly detection with insufficient historical data."""
        diagnostics = DiagnosticsManager()

        # Add latency data
        diagnostics.state.component_statuses["api"] = HealthCheckResult(
            component="api",
            status=HealthStatus.HEALTHY,
            latency_ms=200.0
        )

        # Add insufficient historical data
        diagnostics.state.performance_metrics["api"] = [100.0, 110.0]  # Less than 5 samples

        anomalies = await detect_latency_anomalies(diagnostics)

        assert len(anomalies) == 0  # Should not detect anomalies with insufficient data


class TestGlobalFunctions:
    """Test global diagnostics functions."""

    def test_create_diagnostics_manager(self):
        """Test creating a diagnostics manager."""
        config = {'interval_sec': 30}
        diagnostics = create_diagnostics_manager(config)

        assert isinstance(diagnostics, DiagnosticsManager)
        assert diagnostics.check_interval_sec == 30

    def test_get_diagnostics_manager(self):
        """Test getting the global diagnostics manager."""
        from core.diagnostics import get_diagnostics_manager

        diagnostics = get_diagnostics_manager()
        assert isinstance(diagnostics, DiagnosticsManager)

        # Should return the same instance
        diagnostics2 = get_diagnostics_manager()
        assert diagnostics is diagnostics2


class TestIntegration:
    """Test integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_health_check_workflow(self):
        """Test complete health check workflow."""
        diagnostics = DiagnosticsManager()

        # Register multiple health checks
        async def api_check():
            return HealthCheckResult(
                component="api",
                status=HealthStatus.HEALTHY,
                latency_ms=120.0,
                message="API OK"
            )

        async def db_check():
            return HealthCheckResult(
                component="database",
                status=HealthStatus.HEALTHY,
                latency_ms=50.0,
                message="DB OK"
            )

        diagnostics.register_health_check("api", api_check)
        diagnostics.register_health_check("database", db_check)

        # Register anomaly detector
        diagnostics.register_anomaly_detector(detect_latency_anomalies)

        # Run health check
        state = await diagnostics.run_health_check()

        assert state.overall_status == HealthStatus.HEALTHY
        assert state.check_count == 1
        assert len(state.component_statuses) == 2
        assert "api" in state.component_statuses
        assert "database" in state.component_statuses

    @pytest.mark.asyncio
    async def test_monitoring_loop(self):
        """Test the monitoring loop functionality."""
        diagnostics = DiagnosticsManager()

        # Override check interval for faster testing
        diagnostics.check_interval_sec = 0.1

        # Register a simple health check
        call_count = 0
        async def counting_check():
            nonlocal call_count
            call_count += 1
            return HealthCheckResult(
                component="counter",
                status=HealthStatus.HEALTHY,
                message=f"Call {call_count}"
            )

        diagnostics.register_health_check("counter", counting_check)

        # Start monitoring
        await diagnostics.start()

        # Wait for a few check cycles
        await asyncio.sleep(0.35)  # Should allow ~3 checks

        # Stop monitoring
        await diagnostics.stop()

        # Verify checks were run
        assert call_count >= 2  # Should have run at least 2-3 times
        assert diagnostics.state.check_count >= 2


if __name__ == "__main__":
    pytest.main([__file__])
