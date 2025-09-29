"""
Comprehensive Monitoring & Observability Testing Suite
======================================================

This module provides comprehensive testing for the Monitoring & Observability feature,
covering metrics collection, data accuracy, Grafana integration, alerting system,
and performance impact as specified in the testing strategy.
"""

import asyncio
import math
import time
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from core.alert_rules_manager import AlertRulesManager
from core.dashboard_manager import DashboardManager
from core.metrics_collector import (
    MetricSample,
    MetricsCollector,
    collect_exchange_metrics,
    collect_risk_metrics,
    collect_strategy_metrics,
    collect_trading_metrics,
)
from core.metrics_endpoint import MetricsEndpoint
from utils.logger import get_logger

logger = get_logger(__name__)


class TestMetricsCollection:
    """Test metrics collection functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = {"collection_interval": 15.0, "max_samples_per_metric": 1000}
        self.collector = MetricsCollector(self.config)

    def test_initialization(self):
        """Test metrics collector initialization."""
        assert self.collector.collection_interval == 15.0
        assert self.collector.max_samples_per_metric == 1000
        assert len(self.collector.metrics) == 0
        assert len(self.collector.custom_collectors) == 0

    def test_metric_registration(self):
        """Test metric series registration."""
        # Register a metric
        series = self.collector.register_metric(
            "test_metric", "Test metric description"
        )

        assert "test_metric" in self.collector.metrics
        assert self.collector.metrics["test_metric"] == series
        assert series.name == "test_metric"
        assert series.help_text == "Test metric description"

        # Test duplicate registration returns existing
        series2 = self.collector.register_metric("test_metric", "Different description")
        assert series2 == series  # Should return existing

    @pytest.mark.asyncio
    async def test_metric_recording(self):
        """Test metric value recording."""
        # Register metric
        await self.collector.record_metric("test_gauge", 42.5, {"label": "test"})

        # Check metric was recorded
        assert "test_gauge" in self.collector.metrics
        series = self.collector.metrics["test_gauge"]

        assert len(series.samples) == 1
        sample = series.samples[0]
        assert sample.value == 42.5
        assert sample.labels == {"label": "test"}
        assert sample.timestamp is not None

    @pytest.mark.asyncio
    async def test_counter_increment(self):
        """Test counter metric incrementing."""
        # Increment counter
        await self.collector.increment_counter("test_counter", {"type": "test"})

        # Check counter value
        value = self.collector.get_metric_value("test_counter", {"type": "test"})
        assert value == 1

        # Increment again
        await self.collector.increment_counter("test_counter", {"type": "test"})
        value = self.collector.get_metric_value("test_counter", {"type": "test"})
        assert value == 2

    @pytest.mark.asyncio
    async def test_histogram_observation(self):
        """Test histogram metric observations."""
        # Add histogram observations
        await self.collector.observe_histogram(
            "test_histogram", 0.1, {"bucket": "fast"}
        )
        await self.collector.observe_histogram(
            "test_histogram", 0.5, {"bucket": "medium"}
        )
        await self.collector.observe_histogram(
            "test_histogram", 2.0, {"bucket": "slow"}
        )

        # Check observations were recorded
        series = self.collector.metrics["test_histogram"]
        assert len(series.samples) == 3

        values = [s.value for s in series.samples]
        assert 0.1 in values
        assert 0.5 in values
        assert 2.0 in values

    def test_prometheus_format(self):
        """Test Prometheus exposition format generation."""
        # Register and populate metrics
        series = self.collector.register_metric("test_metric", "Test metric")
        series.add_sample(42.5, {"label": "value"}, 1234567890.0)

        # Generate Prometheus output
        output = self.collector.get_prometheus_output()

        # Verify format
        assert "# HELP test_metric Test metric" in output
        assert "# TYPE test_metric gauge" in output
        assert 'test_metric{label="value"} 42.5 1234567890000' in output

    def test_metric_sample_prometheus_conversion(self):
        """Test individual metric sample Prometheus conversion."""
        sample = MetricSample(
            name="test_sample",
            value=99.9,
            labels={"env": "test", "version": "1.0"},
            timestamp=1234567890.0,
            help_text="Sample metric",
        )

        prometheus_output = sample.to_prometheus()

        assert "# HELP test_sample Sample metric" in prometheus_output
        assert "# TYPE test_sample gauge" in prometheus_output
        assert (
            'test_sample{env="test",version="1.0"} 99.9 1234567890000'
            in prometheus_output
        )

    def test_metric_types_inference(self):
        """Test automatic metric type inference."""
        # Counter pattern
        sample = MetricSample(name="requests_total", value=100)
        assert sample._infer_metric_type() == "counter"

        # Gauge pattern (default)
        sample = MetricSample(name="temperature", value=25.5)
        assert sample._infer_metric_type() == "gauge"

        # Histogram pattern
        sample = MetricSample(name="request_duration_seconds", value=0.5)
        assert sample._infer_metric_type() == "histogram"


class TestDataAccuracy:
    """Test metrics data accuracy and validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.collector = MetricsCollector({})

    @pytest.mark.asyncio
    async def test_timestamp_accuracy(self):
        """Test metric timestamp accuracy."""
        # Record metric
        start_time = time.time()
        await self.collector.record_metric("timestamp_test", 1.0)
        end_time = time.time()

        # Check timestamp is within reasonable range
        series = self.collector.metrics["timestamp_test"]
        sample = series.samples[0]

        assert start_time <= sample.timestamp <= end_time
        assert abs(sample.timestamp - time.time()) < 1.0  # Within 1 second

    @pytest.mark.asyncio
    async def test_metric_value_precision(self):
        """Test metric value precision preservation."""
        # Test various numeric types and precisions
        test_values = [
            0,
            1,
            -1,
            3.14159,
            1e-6,
            1e6,
            float("inf"),
            float("-inf"),
            float("nan"),
        ]

        for i, value in enumerate(test_values):
            metric_name = f"precision_test_{i}"
            await self.collector.record_metric(metric_name, value)

            retrieved_value = self.collector.get_metric_value(metric_name)
            if not (math.isnan(value) if math.isnan(retrieved_value) else False):
                assert retrieved_value == value

    @pytest.mark.asyncio
    async def test_metric_persistence(self):
        """Test metrics persistence across collector operations."""
        # Record metrics
        await self.collector.record_metric(
            "persistence_test", 42.0, {"persistent": "true"}
        )

        # Perform other operations
        await self.collector.record_metric("other_metric", 24.0)
        await self.collector.increment_counter("counter_test")

        # Verify original metric still exists and is correct
        value = self.collector.get_metric_value(
            "persistence_test", {"persistent": "true"}
        )
        assert value == 42.0

    @pytest.mark.asyncio
    async def test_concurrent_metric_recording(self):
        """Test metric recording under concurrent access."""

        async def record_metrics(worker_id: int):
            """Worker function to record metrics concurrently."""
            for i in range(100):
                await self.collector.record_metric(
                    f"concurrent_test_{worker_id}", float(i), {"worker": str(worker_id)}
                )

        # Run multiple workers concurrently
        tasks = [record_metrics(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify all metrics were recorded
        total_samples = 0
        for series in self.collector.metrics.values():
            if series.name.startswith("concurrent_test_"):
                total_samples += len(series.samples)

        assert total_samples == 500  # 5 workers * 100 samples each

    def test_metric_series_limits(self):
        """Test metric series sample limits."""
        # Create collector with small limit
        collector = MetricsCollector({"max_samples_per_metric": 5})

        series = collector.register_metric("limit_test")

        # Add more samples than limit
        for i in range(10):
            series.add_sample(float(i))

        # Should only keep the most recent samples
        assert len(series.samples) == 5
        # Should keep the most recent ones
        values = [s.value for s in series.samples]
        assert values == [5.0, 6.0, 7.0, 8.0, 9.0]


class TestTradingMetricsCollection:
    """Test trading-specific metrics collection."""

    def setup_method(self):
        """Setup test fixtures."""
        self.collector = MetricsCollector({})

    @pytest.mark.asyncio
    async def test_trading_metrics_collection(self):
        """Test collection of trading performance metrics."""
        # Collect trading metrics
        await collect_trading_metrics(self.collector)

        # Verify trading metrics were recorded
        expected_metrics = [
            "trading_total_pnl_usd",
            "trading_win_rate_ratio",
            "trading_sharpe_ratio",
            "trading_max_drawdown_percent",
            "trading_orders_total",
            "trading_order_latency_seconds",
            "trading_slippage_bps",
        ]

        for metric_name in expected_metrics:
            assert metric_name in self.collector.metrics
            value = self.collector.get_metric_value(metric_name)
            assert value is not None

    @pytest.mark.asyncio
    async def test_risk_metrics_collection(self):
        """Test collection of risk management metrics."""
        await collect_risk_metrics(self.collector)

        expected_metrics = [
            "risk_value_at_risk_usd",
            "risk_portfolio_exposure_usd",
            "risk_concentration_ratio",
            "risk_circuit_breaker_status",
        ]

        for metric_name in expected_metrics:
            assert metric_name in self.collector.metrics

    @pytest.mark.asyncio
    async def test_strategy_metrics_collection(self):
        """Test collection of strategy performance metrics."""
        await collect_strategy_metrics(self.collector)

        # Should have metrics for multiple strategies
        strategy_metrics = [
            name
            for name in self.collector.metrics.keys()
            if name.startswith("strategy_")
        ]

        assert len(strategy_metrics) > 0

        # Check specific strategy metrics exist
        assert any("strategy_pnl_usd" in name for name in strategy_metrics)
        assert any("strategy_win_rate_ratio" in name for name in strategy_metrics)

    @pytest.mark.asyncio
    async def test_exchange_metrics_collection(self):
        """Test collection of exchange connectivity metrics."""
        await collect_exchange_metrics(self.collector)

        # Should have metrics for multiple exchanges
        exchange_metrics = [
            name
            for name in self.collector.metrics.keys()
            if name.startswith("exchange_")
        ]

        assert len(exchange_metrics) > 0

        # Check specific exchange metrics exist
        assert any("exchange_connectivity_status" in name for name in exchange_metrics)
        assert any("exchange_latency_seconds" in name for name in exchange_metrics)


class TestMetricsEndpoint:
    """Test metrics HTTP endpoint functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.collector = MetricsCollector({})
        self.endpoint = MetricsEndpoint(self.collector)

    @pytest.mark.asyncio
    async def test_endpoint_creation(self):
        """Test metrics endpoint creation."""
        app = self.endpoint.create_app()
        assert isinstance(app, web.Application)

    @pytest.mark.asyncio
    async def test_metrics_endpoint_response(self):
        """Test /metrics endpoint response."""
        # Add some test metrics
        await self.collector.record_metric("test_metric", 42.0, {"test": "true"})

        # Create test client
        app = self.endpoint.create_app()
        async with TestServer(app) as server:
            async with TestClient(server) as client:
                # Request metrics
                resp = await client.get("/metrics")

                assert resp.status == 200
                text = await resp.text()

                # Verify Prometheus format
                assert "# TYPE test_metric gauge" in text
                assert 'test_metric{test="true"} 42.0' in text

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test /health endpoint."""
        app = self.endpoint.create_app()
        async with TestServer(app) as server:
            async with TestClient(server) as client:
                resp = await client.get("/health")

                assert resp.status == 200
                data = await resp.json()

                assert "status" in data
                assert data["status"] == "healthy"
                assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_invalid_endpoint(self):
        """Test invalid endpoint handling."""
        app = self.endpoint.create_app()
        async with TestServer(app) as server:
            async with TestClient(server) as client:
                resp = await client.get("/invalid")

                assert resp.status == 404


class TestAlertingSystem:
    """Test alerting system functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.alert_manager = AlertRulesManager({})

    @pytest.mark.asyncio
    async def test_alert_rule_creation(self):
        """Test alert rule creation and validation."""
        # Create alert rule
        rule_config = {
            "name": "high_cpu_alert",
            "query": "cpu_usage > 80",
            "duration": "5m",
            "severity": "warning",
            "description": "High CPU usage detected",
        }

        rule = await self.alert_manager.create_rule(rule_config)

        assert rule.name == "high_cpu_alert"
        assert rule.query == "cpu_usage > 80"
        assert rule.severity == "warning"

    @pytest.mark.asyncio
    async def test_alert_evaluation(self):
        """Test alert condition evaluation."""
        # Create rule
        rule_config = {
            "name": "test_alert",
            "query": "test_metric > 90",
            "duration": "1m",
            "severity": "critical",
        }

        rule = await self.alert_manager.create_rule(rule_config)

        # Test alert triggering
        metrics_data = {"test_metric": 95}
        alert_triggered = await rule.evaluate(metrics_data)

        assert alert_triggered

        # Test alert not triggering
        metrics_data = {"test_metric": 85}
        alert_triggered = await rule.evaluate(metrics_data)

        assert not alert_triggered

    @pytest.mark.asyncio
    async def test_alert_deduplication(self):
        """Test alert deduplication logic."""
        rule_config = {
            "name": "duplicate_test",
            "query": "error_rate > 5",
            "duration": "1m",
            "severity": "warning",
        }

        rule = await self.alert_manager.create_rule(rule_config)

        # Trigger alert multiple times
        metrics_data = {"error_rate": 10}

        # First trigger
        alerts1 = await self.alert_manager.evaluate_rules(metrics_data)
        assert len(alerts1) == 1

        # Second trigger (should be deduplicated)
        alerts2 = await self.alert_manager.evaluate_rules(metrics_data)
        assert len(alerts2) == 0  # Should be suppressed due to deduplication

    @pytest.mark.asyncio
    async def test_notification_delivery(self):
        """Test alert notification delivery."""
        # Mock notification channels
        with patch.object(
            self.alert_manager, "_send_discord_notification"
        ) as mock_discord, patch.object(
            self.alert_manager, "_send_email_notification"
        ) as mock_email:
            rule_config = {
                "name": "notification_test",
                "query": "memory_usage > 90",
                "duration": "1m",
                "severity": "critical",
                "channels": ["discord", "email"],
            }

            rule = await self.alert_manager.create_rule(rule_config)

            # Trigger alert
            metrics_data = {"memory_usage": 95}
            alerts = await self.alert_manager.evaluate_rules(metrics_data)
            assert len(alerts) == 1

            # Verify notifications were sent
            mock_discord.assert_called_once()
            mock_email.assert_called_once()


class TestPerformanceImpact:
    """Test monitoring system performance impact."""

    def setup_method(self):
        """Setup test fixtures."""
        self.collector = MetricsCollector({})

    @pytest.mark.asyncio
    async def test_collection_performance(self):
        """Test metrics collection performance overhead."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Measure baseline CPU
        baseline_cpu = process.cpu_percent(interval=0.1)

        # Perform metrics collection
        start_time = time.time()

        # Collect metrics in batch
        for i in range(1000):
            await self.collector.record_metric(f"perf_test_{i % 10}", float(i))

        collection_time = time.time() - start_time
        collection_cpu = process.cpu_percent(interval=0.1)

        # Performance requirements
        assert (
            collection_time < 1.0
        ), f"Collection took {collection_time:.2f}s (should be < 1.0s)"
        assert (
            collection_cpu - baseline_cpu < 10
        ), f"CPU overhead {collection_cpu - baseline_cpu:.1f}% (should be < 10%)"

    @pytest.mark.asyncio
    async def test_memory_overhead(self):
        """Test memory overhead of metrics collection."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss

        # Perform intensive metrics collection
        for i in range(10000):
            await self.collector.record_metric(
                "memory_test", float(i), {"batch": str(i % 10)}
            )

        final_memory = process.memory_info().rss
        memory_increase = final_memory - baseline_memory

        # Memory requirements (reasonable for metrics collection)
        assert (
            memory_increase < 100 * 1024 * 1024
        ), f"Memory increase {memory_increase/1024/1024:.1f}MB exceeds 100MB limit"

    @pytest.mark.asyncio
    async def test_scalability_under_load(self):
        """Test monitoring system scalability under high load."""
        # Simulate high-frequency metrics generation
        start_time = time.time()

        # Generate metrics at high rate
        tasks = []
        for i in range(100):
            task = asyncio.create_task(self._generate_metrics_batch(i))
            tasks.append(task)

        await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Scalability requirements
        assert (
            total_time < 5.0
        ), f"High-load test took {total_time:.2f}s (should be < 5.0s)"

        # Verify all metrics were collected
        total_metrics = len(self.collector.metrics)
        assert (
            total_metrics >= 100
        ), f"Only {total_metrics} metrics collected (expected >= 100)"

    async def _generate_metrics_batch(self, batch_id: int):
        """Generate a batch of metrics for scalability testing."""
        for i in range(100):
            await self.collector.record_metric(
                f"scalability_test_{batch_id}_{i % 10}",
                float(i),
                {"batch": str(batch_id), "metric": str(i % 10)},
            )


class TestGrafanaIntegration:
    """Test Grafana dashboard integration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.dashboard_manager = DashboardManager({})

    @pytest.mark.asyncio
    async def test_dashboard_creation(self):
        """Test Grafana dashboard creation."""
        dashboard_config = {
            "title": "Test Dashboard",
            "description": "Test monitoring dashboard",
            "panels": [
                {
                    "title": "CPU Usage",
                    "type": "graph",
                    "targets": [{"expr": "cpu_usage"}],
                }
            ],
        }

        dashboard = await self.dashboard_manager.create_dashboard(dashboard_config)

        assert dashboard["title"] == "Test Dashboard"
        assert len(dashboard["panels"]) == 1

    @pytest.mark.asyncio
    async def test_dashboard_rendering(self):
        """Test dashboard rendering with real data."""
        # Create dashboard
        dashboard_config = {
            "title": "Data Test",
            "panels": [
                {
                    "title": "Test Panel",
                    "type": "graph",
                    "targets": [{"expr": "test_metric"}],
                }
            ],
        }

        dashboard = await self.dashboard_manager.create_dashboard(dashboard_config)

        # Test rendering
        result = await self.dashboard_manager.render_dashboard(dashboard["id"])

        assert result["status"] == "rendered"
        assert result["title"] == "Data Test"
        assert len(result["panels"]) == 1

    @pytest.mark.asyncio
    async def test_query_performance(self):
        """Test Grafana query performance."""
        start_time = time.time()
        result = await self.dashboard_manager.query_metrics("test_metric", "1h")
        query_time = time.time() - start_time

        # Performance requirements
        assert query_time < 1.0, f"Query took {query_time:.2f}s (should be < 1.0s)"
        assert "results" in result
        assert result["status"] == "success"


# Integration test fixtures
@pytest.fixture
def metrics_collector():
    """Metrics collector fixture."""
    return MetricsCollector({})


@pytest.fixture
def metrics_endpoint():
    """Metrics endpoint fixture."""
    collector = MetricsCollector({})
    return MetricsEndpoint(collector)


@pytest.fixture
def alert_manager():
    """Alert manager fixture."""
    return AlertRulesManager({})


@pytest.fixture
async def sample_metrics(metrics_collector):
    """Sample metrics fixture."""
    # Add some test metrics
    await metrics_collector.record_metric("cpu_usage", 65.5, {"host": "test"})
    await metrics_collector.record_metric("memory_usage", 78.2, {"host": "test"})
    await metrics_collector.record_metric("disk_usage", 45.1, {"host": "test"})

    return metrics_collector


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
