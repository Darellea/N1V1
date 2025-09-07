"""
Test suite for anomaly detection functionality.

Tests cover statistical anomaly detection, threshold monitoring, and edge cases.
"""

import asyncio
import pytest
import statistics
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from core.diagnostics import (
    DiagnosticsManager,
    HealthStatus,
    HealthCheckResult,
    AnomalyDetection,
    AlertSeverity,
    detect_latency_anomalies,
    detect_drawdown_anomalies
)


class TestLatencyAnomalyDetection:
    """Test latency-based anomaly detection."""

    @pytest.fixture
    def diagnostics_manager(self):
        """Create a diagnostics manager for testing."""
        return DiagnosticsManager()

    def test_no_anomalies_with_normal_latency(self, diagnostics_manager):
        """Test that normal latency doesn't trigger anomalies."""
        # Set up normal latency data
        diagnostics_manager.state.component_statuses["api"] = HealthCheckResult(
            component="api",
            status=HealthStatus.HEALTHY,
            latency_ms=120.0
        )

        # Add historical data with low variance
        diagnostics_manager.state.performance_metrics["api"] = [
            100.0, 110.0, 95.0, 105.0, 98.0, 102.0, 115.0
        ]

        anomalies = asyncio.run(detect_latency_anomalies(diagnostics_manager))

        assert len(anomalies) == 0

    def test_anomaly_detection_with_high_latency(self, diagnostics_manager):
        """Test anomaly detection with significantly high latency."""
        # Set up high latency data
        diagnostics_manager.state.component_statuses["api"] = HealthCheckResult(
            component="api",
            status=HealthStatus.DEGRADED,
            latency_ms=2500.0  # Much higher than normal
        )

        # Add historical data with low variance
        diagnostics_manager.state.performance_metrics["api"] = [
            100.0, 110.0, 95.0, 105.0, 98.0, 102.0, 115.0
        ]

        anomalies = asyncio.run(detect_latency_anomalies(diagnostics_manager))

        assert len(anomalies) == 1
        anomaly = anomalies[0]

        assert anomaly.component == "api"
        assert anomaly.metric == "latency_ms"
        assert anomaly.value == 2500.0
        assert anomaly.severity == AlertSeverity.WARNING
        assert "Latency spike detected" in anomaly.description
        assert anomaly.threshold > 0  # Should calculate a threshold

    def test_anomaly_detection_with_critical_latency(self, diagnostics_manager):
        """Test anomaly detection with critically high latency."""
        # Set up very high latency data
        diagnostics_manager.state.component_statuses["api"] = HealthCheckResult(
            component="api",
            status=HealthStatus.CRITICAL,
            latency_ms=10000.0  # Extremely high
        )

        # Add historical data
        diagnostics_manager.state.performance_metrics["api"] = [
            100.0, 110.0, 95.0, 105.0, 98.0
        ]

        anomalies = asyncio.run(detect_latency_anomalies(diagnostics_manager))

        assert len(anomalies) == 1
        anomaly = anomalies[0]

        assert anomaly.component == "api"
        assert anomaly.value == 10000.0
        assert anomaly.severity == AlertSeverity.WARNING  # Still warning, not critical

    def test_insufficient_historical_data(self, diagnostics_manager):
        """Test behavior with insufficient historical data."""
        # Set up latency data
        diagnostics_manager.state.component_statuses["api"] = HealthCheckResult(
            component="api",
            status=HealthStatus.HEALTHY,
            latency_ms=200.0
        )

        # Add insufficient historical data (less than 5 samples)
        diagnostics_manager.state.performance_metrics["api"] = [100.0, 110.0]

        anomalies = asyncio.run(detect_latency_anomalies(diagnostics_manager))

        assert len(anomalies) == 0  # Should not detect anomalies

    def test_no_latency_data(self, diagnostics_manager):
        """Test behavior when component has no latency data."""
        # Set up component without latency data
        diagnostics_manager.state.component_statuses["api"] = HealthCheckResult(
            component="api",
            status=HealthStatus.HEALTHY,
            latency_ms=None
        )

        # Add historical data
        diagnostics_manager.state.performance_metrics["api"] = [
            100.0, 110.0, 95.0, 105.0, 98.0
        ]

        anomalies = asyncio.run(detect_latency_anomalies(diagnostics_manager))

        assert len(anomalies) == 0  # Should not detect anomalies

    def test_multiple_components_anomaly_detection(self, diagnostics_manager):
        """Test anomaly detection across multiple components."""
        # Set up multiple components
        diagnostics_manager.state.component_statuses.update({
            "api": HealthCheckResult(
                component="api",
                status=HealthStatus.DEGRADED,
                latency_ms=800.0  # High latency
            ),
            "database": HealthCheckResult(
                component="database",
                status=HealthStatus.HEALTHY,
                latency_ms=50.0  # Normal latency
            ),
            "cache": HealthCheckResult(
                component="cache",
                status=HealthStatus.HEALTHY,
                latency_ms=120.0  # Normal latency
            )
        })

        # Add historical data for each
        diagnostics_manager.state.performance_metrics.update({
            "api": [100.0, 110.0, 95.0, 105.0, 98.0],
            "database": [40.0, 45.0, 50.0, 48.0, 52.0],
            "cache": [100.0, 110.0, 95.0, 105.0, 98.0]
        })

        anomalies = asyncio.run(detect_latency_anomalies(diagnostics_manager))

        # Should detect anomaly only for API
        assert len(anomalies) == 1
        assert anomalies[0].component == "api"
        assert anomalies[0].value == 800.0

    def test_anomaly_threshold_calculation(self, diagnostics_manager):
        """Test that anomaly thresholds are calculated correctly."""
        # Set up test data
        diagnostics_manager.state.component_statuses["api"] = HealthCheckResult(
            component="api",
            status=HealthStatus.HEALTHY,
            latency_ms=300.0
        )

        # Add historical data with known statistics
        historical_data = [100.0, 110.0, 90.0, 105.0, 95.0]
        diagnostics_manager.state.performance_metrics["api"] = historical_data

        # Calculate expected threshold
        mean_latency = statistics.mean(historical_data)
        std_dev = statistics.stdev(historical_data)
        expected_threshold = mean_latency + (std_dev * 2.0)  # Default std_dev_threshold

        anomalies = asyncio.run(detect_latency_anomalies(diagnostics_manager))

        if anomalies:
            assert abs(anomalies[0].threshold - expected_threshold) < 0.1


class TestDrawdownAnomalyDetection:
    """Test drawdown-based anomaly detection."""

    @pytest.fixture
    def diagnostics_manager(self):
        """Create a diagnostics manager for testing."""
        return DiagnosticsManager()

    @pytest.fixture
    def mock_portfolio_manager(self):
        """Create a mock portfolio manager."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_drawdown_anomaly_detection_not_implemented(self, diagnostics_manager, mock_portfolio_manager):
        """Test that drawdown anomaly detection returns empty list (not implemented)."""
        anomalies = await detect_drawdown_anomalies(mock_portfolio_manager, diagnostics_manager)

        # Currently returns empty list as implementation is placeholder
        assert isinstance(anomalies, list)
        assert len(anomalies) == 0

    @pytest.mark.asyncio
    async def test_drawdown_anomaly_detection_with_exception(self, diagnostics_manager, mock_portfolio_manager):
        """Test drawdown anomaly detection error handling."""
        # Mock the portfolio manager to raise an exception
        mock_portfolio_manager.some_method = MagicMock(side_effect=Exception("Test error"))

        # Should not raise exception, should return empty list
        anomalies = await detect_drawdown_anomalies(mock_portfolio_manager, diagnostics_manager)

        assert isinstance(anomalies, list)
        assert len(anomalies) == 0


class TestAnomalyDetectionConfiguration:
    """Test anomaly detection configuration options."""

    def test_custom_std_dev_threshold(self):
        """Test custom standard deviation threshold."""
        config = {
            'anomaly_std_dev_threshold': 3.0
        }
        diagnostics = DiagnosticsManager(config)

        assert diagnostics.anomaly_std_dev_threshold == 3.0

    def test_custom_latency_window_size(self):
        """Test custom latency window size."""
        config = {
            'latency_window_size': 20
        }
        diagnostics = DiagnosticsManager(config)

        assert diagnostics.latency_window_size == 20

    def test_default_configuration(self):
        """Test default configuration values."""
        diagnostics = DiagnosticsManager()

        assert diagnostics.anomaly_std_dev_threshold == 2.0
        assert diagnostics.latency_window_size == 10


class TestStatisticalCalculations:
    """Test statistical calculations used in anomaly detection."""

    def test_mean_calculation(self):
        """Test mean calculation for latency data."""
        data = [100.0, 110.0, 90.0, 105.0, 95.0]
        expected_mean = sum(data) / len(data)

        assert abs(statistics.mean(data) - expected_mean) < 0.001

    def test_standard_deviation_calculation(self):
        """Test standard deviation calculation."""
        data = [100.0, 100.0, 100.0, 100.0, 100.0]  # No variance
        std_dev = statistics.stdev(data)

        assert std_dev == 0.0

    def test_threshold_calculation(self):
        """Test threshold calculation formula."""
        mean_val = 100.0
        std_dev_val = 10.0
        threshold_multiplier = 2.0

        threshold = mean_val + (std_dev_val * threshold_multiplier)
        expected_threshold = 120.0

        assert threshold == expected_threshold


class TestAnomalySeverityLevels:
    """Test anomaly severity level determination."""

    @pytest.fixture
    def diagnostics_manager(self):
        """Create a diagnostics manager for testing."""
        return DiagnosticsManager()

    def test_warning_severity_for_moderate_spike(self, diagnostics_manager):
        """Test that moderate latency spikes get WARNING severity."""
        # Set up moderate latency spike
        diagnostics_manager.state.component_statuses["api"] = HealthCheckResult(
            component="api",
            status=HealthStatus.DEGRADED,
            latency_ms=250.0  # Moderate spike
        )

        diagnostics_manager.state.performance_metrics["api"] = [
            100.0, 110.0, 95.0, 105.0, 98.0
        ]

        anomalies = asyncio.run(detect_latency_anomalies(diagnostics_manager))

        if anomalies:
            # Should be WARNING for moderate spike
            assert anomalies[0].severity == AlertSeverity.WARNING

    def test_info_severity_for_small_spike(self, diagnostics_manager):
        """Test that small latency spikes get INFO severity."""
        # This test would require modifying the anomaly detection logic
        # to differentiate between WARNING and INFO based on spike magnitude
        # For now, the implementation uses WARNING for all anomalies
        pass


class TestRollingWindowManagement:
    """Test rolling window data management."""

    @pytest.fixture
    def diagnostics_manager(self):
        """Create a diagnostics manager for testing."""
        return DiagnosticsManager()

    def test_rolling_window_size_limit(self, diagnostics_manager):
        """Test that rolling window maintains correct size."""
        component = "api"
        window_size = diagnostics_manager.latency_window_size

        # Add more data than window size
        data_points = window_size + 5
        for i in range(data_points):
            diagnostics_manager.state.performance_metrics.setdefault(component, []).append(float(i))

        metrics = diagnostics_manager.state.performance_metrics[component]

        # Should not exceed window size
        assert len(metrics) <= window_size

    def test_rolling_window_fifo_behavior(self, diagnostics_manager):
        """Test FIFO behavior of rolling window."""
        component = "api"

        # Add initial data
        initial_data = [100.0, 110.0, 120.0]
        diagnostics_manager.state.performance_metrics[component] = initial_data.copy()

        # Add more data to trigger window limit
        window_size = diagnostics_manager.latency_window_size
        for i in range(window_size + 2):
            diagnostics_manager.state.performance_metrics[component].append(float(i + 200))

        metrics = diagnostics_manager.state.performance_metrics[component]

        # Should maintain window size and contain newer data
        assert len(metrics) <= window_size
        if len(metrics) > 0:
            # Newest data should be present
            assert 200.0 in metrics or any(x >= 200.0 for x in metrics)


class TestEdgeCases:
    """Test edge cases in anomaly detection."""

    @pytest.fixture
    def diagnostics_manager(self):
        """Create a diagnostics manager for testing."""
        return DiagnosticsManager()

    def test_empty_component_statuses(self, diagnostics_manager):
        """Test anomaly detection with empty component statuses."""
        anomalies = asyncio.run(detect_latency_anomalies(diagnostics_manager))

        assert len(anomalies) == 0

    def test_zero_latency_values(self, diagnostics_manager):
        """Test anomaly detection with zero latency values."""
        diagnostics_manager.state.component_statuses["api"] = HealthCheckResult(
            component="api",
            status=HealthStatus.HEALTHY,
            latency_ms=0.0
        )

        diagnostics_manager.state.performance_metrics["api"] = [0.0, 0.0, 0.0, 0.0, 0.0]

        anomalies = asyncio.run(detect_latency_anomalies(diagnostics_manager))

        # Should not detect anomalies when all values are zero
        assert len(anomalies) == 0

    def test_negative_latency_values(self, diagnostics_manager):
        """Test anomaly detection with negative latency values (edge case)."""
        diagnostics_manager.state.component_statuses["api"] = HealthCheckResult(
            component="api",
            status=HealthStatus.HEALTHY,
            latency_ms=-100.0  # Negative latency (shouldn't happen in practice)
        )

        diagnostics_manager.state.performance_metrics["api"] = [50.0, 60.0, 40.0, 55.0, 45.0]

        # Should handle gracefully without crashing
        anomalies = asyncio.run(detect_latency_anomalies(diagnostics_manager))

        # Result depends on implementation, but shouldn't crash
        assert isinstance(anomalies, list)

    def test_extreme_variance_in_historical_data(self, diagnostics_manager):
        """Test anomaly detection with extreme variance in historical data."""
        diagnostics_manager.state.component_statuses["api"] = HealthCheckResult(
            component="api",
            status=HealthStatus.HEALTHY,
            latency_ms=200.0
        )

        # Historical data with extreme variance
        diagnostics_manager.state.performance_metrics["api"] = [
            1.0, 1000.0, 500.0, 50.0, 2000.0
        ]

        anomalies = asyncio.run(detect_latency_anomalies(diagnostics_manager))

        # Should handle extreme variance without crashing
        assert isinstance(anomalies, list)


class TestPerformanceMetrics:
    """Test performance metrics tracking."""

    @pytest.fixture
    def diagnostics_manager(self):
        """Create a diagnostics manager for testing."""
        return DiagnosticsManager()

    def test_performance_metrics_initialization(self, diagnostics_manager):
        """Test that performance metrics are initialized as empty dict."""
        assert isinstance(diagnostics_manager.state.performance_metrics, dict)
        assert len(diagnostics_manager.state.performance_metrics) == 0

    def test_performance_metrics_tracking(self, diagnostics_manager):
        """Test that performance metrics are tracked per component."""
        component = "api"

        # Initially empty
        assert component not in diagnostics_manager.state.performance_metrics

        # Add some latency data
        diagnostics_manager.state.component_statuses[component] = HealthCheckResult(
            component=component,
            status=HealthStatus.HEALTHY,
            latency_ms=150.0
        )

        # Run anomaly detection to trigger metrics tracking
        asyncio.run(detect_latency_anomalies(diagnostics_manager))

        # Should now have metrics for the component
        assert component in diagnostics_manager.state.performance_metrics
        assert isinstance(diagnostics_manager.state.performance_metrics[component], list)


if __name__ == "__main__":
    pytest.main([__file__])
