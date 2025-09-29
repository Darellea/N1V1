#!/usr/bin/env python3
"""
Chaos engineering tests for N1V1 trading system.

This module implements comprehensive chaos experiments to validate system resilience
against real-world operational hazards. It simulates network partitions, exchange
downtime, rate limiting, and database outages to ensure the system recovers
automatically within SLA.

Key Features:
- Network partition simulation (60s timeout)
- Rate-limit flood simulation (429 responses)
- Exchange downtime simulation (5min outage)
- Database outage simulation (connection errors)
- Automatic recovery validation
- SLA compliance verification
- Structured chaos reporting
"""

import asyncio
import json
import time
import unittest.mock as mock
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import pytest

from core.order_executor import OrderExecutor
from core.watchdog import WatchdogService
from utils.logger import get_logger

logger = get_logger(__name__)


class MockOrderManagerWithChaos:
    """Mock order manager that can be affected by chaos injection."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.executed_orders: Dict[str, Dict[str, Any]] = {}
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        self.execution_latencies: List[float] = []
        self.failure_rate = 0.002  # 0.2% baseline failure rate
        self.latency_distribution = {"mean": 50.0, "std": 25.0}  # ms  # ms
        # Chaos state
        self.chaos_active = False
        self.chaos_type = None

    def set_chaos_state(self, active: bool, chaos_type: str = None):
        """Set the chaos state for this order manager."""
        self.chaos_active = active
        self.chaos_type = chaos_type

    async def execute_order(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a mock order with realistic latency and failure simulation."""
        start_time = time.time()

        order_id = signal["id"]

        # Simulate processing delay
        processing_delay = np.random.exponential(0.01)  # Mean 10ms
        await asyncio.sleep(processing_delay)

        # Check if chaos is active and should cause failure
        should_fail = False
        if self.chaos_active:
            if self.chaos_type == "network_partition":
                should_fail = np.random.random() < 0.95  # 95% failure rate
            elif self.chaos_type == "rate_limit_flood":
                should_fail = np.random.random() < 0.50  # 50% failure rate
            elif self.chaos_type == "exchange_downtime":
                should_fail = np.random.random() < 0.98  # 98% failure rate
            elif self.chaos_type == "database_outage":
                should_fail = np.random.random() < 0.80  # 80% failure rate

        if should_fail:
            # Failed order due to chaos
            result = {
                "id": order_id,
                "status": "failed",
                "error": f"Simulated {self.chaos_type} failure",
                "timestamp": datetime.now(),
                "latency_ms": (time.time() - start_time) * 1000,
            }
        else:
            # Successful order
            # Simulate realistic latency with some outliers
            base_latency = np.random.normal(
                self.latency_distribution["mean"], self.latency_distribution["std"]
            )
            # Add occasional spikes
            if np.random.random() < 0.05:  # 5% chance of spike
                base_latency *= np.random.uniform(3, 10)

            total_latency = max(1.0, base_latency + processing_delay * 1000)

            result = {
                "id": order_id,
                "status": "filled",
                "symbol": signal["symbol"],
                "side": signal["side"],
                "amount": signal["amount"],
                "price": np.random.uniform(100, 50000),  # Simulated fill price
                "pnl": np.random.normal(0, 10),  # Simulated P&L
                "fee": signal["amount"] * 0.001,  # 0.1% fee
                "timestamp": datetime.now(),
                "latency_ms": total_latency,
            }

        self.execution_latencies.append(result["latency_ms"])
        self.executed_orders[order_id] = result
        self.order_history.append(result)

        return result

    async def cancel_all_orders(self):
        """Cancel all pending orders."""
        cancelled_count = len(self.pending_orders)
        self.pending_orders.clear()
        logger.info(f"Cancelled {cancelled_count} pending orders")

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific order."""
        return self.executed_orders.get(order_id)

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get all active orders."""
        return list(self.pending_orders.values())

    def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent order history."""
        return self.order_history[-limit:]


@dataclass
class ChaosScenario:
    """Definition of a chaos scenario."""

    name: str
    description: str
    duration_seconds: int
    recovery_timeout_seconds: int = 300  # 5 minutes default
    expected_failure_rate: float = 0.0  # Expected failure rate during chaos
    sla_recovery_time_seconds: int = 60  # SLA for recovery

    def __post_init__(self):
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.recovery_time: Optional[float] = None
        self.failure_injected = False
        self.recovery_detected = False


@dataclass
class ChaosTestMetrics:
    """Metrics collected during chaos testing."""

    scenario_name: str
    orders_before_chaos: int = 0
    orders_during_chaos: int = 0
    orders_after_chaos: int = 0
    failures_before_chaos: int = 0
    failures_during_chaos: int = 0
    failures_after_chaos: int = 0
    latency_before_chaos: List[float] = field(default_factory=list)
    latency_during_chaos: List[float] = field(default_factory=list)
    latency_after_chaos: List[float] = field(default_factory=list)
    watchdog_alerts: List[Dict[str, Any]] = field(default_factory=list)
    recovery_events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def chaos_failure_rate(self) -> float:
        """Calculate failure rate during chaos period."""
        total = self.orders_during_chaos
        return (self.failures_during_chaos / total) * 100 if total > 0 else 0.0

    @property
    def recovery_successful(self) -> bool:
        """Check if system recovered successfully."""
        # Recovery is successful if post-chaos failure rate is similar to pre-chaos
        pre_rate = (
            (self.failures_before_chaos / self.orders_before_chaos) * 100
            if self.orders_before_chaos > 0
            else 0.0
        )
        post_rate = (
            (self.failures_after_chaos / self.orders_after_chaos) * 100
            if self.orders_after_chaos > 0
            else 0.0
        )
        return abs(pre_rate - post_rate) < 1.0  # Within 1% tolerance


class ChaosInjector:
    """Base class for chaos injection mechanisms."""

    def __init__(self, scenario: ChaosScenario):
        self.scenario = scenario
        self.active = False

    @asynccontextmanager
    async def inject_chaos(self):
        """Context manager for injecting chaos."""
        self.active = True
        self.scenario.failure_injected = True
        self.scenario.start_time = datetime.now()

        logger.warning(
            f"Injecting chaos: {self.scenario.name} - {self.scenario.description}"
        )

        # Set chaos state on order manager if available
        if hasattr(self, "_order_manager") and self._order_manager:
            self._order_manager.set_chaos_state(True, self.scenario.name)

        try:
            yield self
        finally:
            self.active = False
            self.scenario.end_time = datetime.now()

            # Clear chaos state on order manager
            if hasattr(self, "_order_manager") and self._order_manager:
                self._order_manager.set_chaos_state(False)

            logger.info(f"Chaos injection ended: {self.scenario.name}")


class NetworkPartitionInjector(ChaosInjector):
    """Simulates network partition by delaying all HTTP requests."""

    def __init__(self, scenario: ChaosScenario, delay_seconds: float = 60.0):
        super().__init__(scenario)
        self.delay_seconds = delay_seconds
        self.original_session_init = None

    async def _delayed_request(self, *args, **kwargs):
        """Make a delayed HTTP request."""
        if self.active:
            await asyncio.sleep(self.delay_seconds)
        # This would normally make the actual request
        # For simulation, we'll just delay and return a timeout

    @asynccontextmanager
    async def inject_chaos(self):
        """Inject network partition by patching aiohttp."""
        with mock.patch("aiohttp.ClientSession._request") as mock_request:

            async def delayed_request(*args, **kwargs):
                if self.active:
                    # Simulate network timeout
                    await asyncio.sleep(self.delay_seconds)
                    raise aiohttp.ServerTimeoutError("Simulated network partition")
                else:
                    # Call original method - this is tricky in mock
                    return await self._make_real_request(*args, **kwargs)

            mock_request.side_effect = delayed_request

            async with super().inject_chaos():
                yield self

    async def _make_real_request(self, *args, **kwargs):
        """Make a real HTTP request when chaos is not active."""
        # This would need to be implemented to call the actual aiohttp method
        pass


class RateLimitInjector(ChaosInjector):
    """Simulates rate limiting by returning 429 responses."""

    def __init__(self, scenario: ChaosScenario, rate_limit_responses: int = 10):
        super().__init__(scenario)
        self.rate_limit_responses = rate_limit_responses
        self.response_count = 0

    @asynccontextmanager
    async def inject_chaos(self):
        """Inject rate limiting by patching HTTP responses."""
        with mock.patch("aiohttp.ClientSession._request") as mock_request:

            async def rate_limited_request(*args, **kwargs):
                if self.active and self.response_count < self.rate_limit_responses:
                    self.response_count += 1
                    # Create a mock response with 429 status
                    mock_response = mock.MagicMock()
                    mock_response.status = 429
                    mock_response.headers = {"Retry-After": "60"}
                    mock_response.text = mock.AsyncMock(
                        return_value='{"error": "rate limit exceeded"}'
                    )
                    mock_response.json = mock.AsyncMock(
                        return_value={"error": "rate limit exceeded"}
                    )
                    return mock_response
                else:
                    # Call original method
                    return await self._make_real_request(*args, **kwargs)

            mock_request.side_effect = rate_limited_request

            async with super().inject_chaos():
                yield self

    async def _make_real_request(self, *args, **kwargs):
        """Make a real HTTP request when chaos is not active."""
        pass


class ExchangeDowntimeInjector(ChaosInjector):
    """Simulates complete exchange downtime."""

    def __init__(self, scenario: ChaosScenario):
        super().__init__(scenario)

    @asynccontextmanager
    async def inject_chaos(self):
        """Inject exchange downtime by blocking all requests."""
        with mock.patch("aiohttp.ClientSession._request") as mock_request:

            async def downtime_request(*args, **kwargs):
                if self.active:
                    # Simulate complete downtime
                    raise aiohttp.ClientConnectionError("Simulated exchange downtime")
                else:
                    return await self._make_real_request(*args, **kwargs)

            mock_request.side_effect = downtime_request

            async with super().inject_chaos():
                yield self

    async def _make_real_request(self, *args, **kwargs):
        """Make a real HTTP request when chaos is not active."""
        pass


class DatabaseOutageInjector(ChaosInjector):
    """Simulates database connection failures."""

    def __init__(self, scenario: ChaosScenario):
        super().__init__(scenario)

    @asynccontextmanager
    async def inject_chaos(self):
        """Inject database outage by patching database connections."""
        # For database outage, we rely on the order manager's chaos state
        # The actual failure injection happens in MockOrderManagerWithChaos
        async with super().inject_chaos():
            yield self


class ChaosTestRunner:
    """Main chaos test runner coordinating scenarios and validation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = ChaosTestMetrics("unknown")
        self.watchdog = None
        self.order_executor = None
        self.order_manager = None
        self.performance_tracker = None
        self.notifier = None

        # Chaos scenarios
        self.scenarios = self._define_scenarios()

        # Test control
        self.background_tasks: List[asyncio.Task] = []
        self.chaos_reports: List[Dict[str, Any]] = []

    def _define_scenarios(self) -> Dict[str, ChaosScenario]:
        """Define all chaos scenarios."""
        return {
            "network_partition": ChaosScenario(
                name="network_partition",
                description="Network partition causing 60s API timeouts",
                duration_seconds=5,  # Reduced for testing
                recovery_timeout_seconds=15,  # Reduced for testing
                expected_failure_rate=100.0,  # All requests should fail
                sla_recovery_time_seconds=10,  # Reduced SLA for testing
            ),
            "rate_limit_flood": ChaosScenario(
                name="rate_limit_flood",
                description="Exchange rate limiting with 429 responses",
                duration_seconds=3,  # Reduced for testing
                recovery_timeout_seconds=10,  # Reduced for testing
                expected_failure_rate=50.0,  # Partial failure expected
                sla_recovery_time_seconds=5,  # Reduced SLA for testing
            ),
            "exchange_downtime": ChaosScenario(
                name="exchange_downtime",
                description="Complete exchange API downtime",
                duration_seconds=5,  # Reduced for testing
                recovery_timeout_seconds=15,  # Reduced for testing
                expected_failure_rate=100.0,
                sla_recovery_time_seconds=8,  # Reduced SLA for testing
            ),
            "database_outage": ChaosScenario(
                name="database_outage",
                description="Database connection failures",
                duration_seconds=4,  # Reduced for testing
                recovery_timeout_seconds=12,  # Reduced for testing
                expected_failure_rate=80.0,  # Most operations should fail
                sla_recovery_time_seconds=6,  # Reduced SLA for testing
            ),
        }

    def _create_injector(self, scenario: ChaosScenario) -> ChaosInjector:
        """Create appropriate injector for scenario."""
        injector = None
        if scenario.name == "network_partition":
            injector = NetworkPartitionInjector(scenario)
        elif scenario.name == "rate_limit_flood":
            injector = RateLimitInjector(scenario)
        elif scenario.name == "exchange_downtime":
            injector = ExchangeDowntimeInjector(scenario)
        elif scenario.name == "database_outage":
            injector = DatabaseOutageInjector(scenario)
        else:
            raise ValueError(f"Unknown scenario: {scenario.name}")

        # Set order manager reference for chaos state management
        if injector:
            injector._order_manager = self.order_manager

        return injector

    async def setup_test_environment(self):
        """Set up the test environment with mock components."""
        # Create mock order manager that can be affected by chaos
        self.order_manager = MockOrderManagerWithChaos({})

        # Create order executor
        self.order_executor = OrderExecutor(
            {"max_retries": 3, "retry_base_delay": 1.0},
            self.order_manager,
            self.performance_tracker,
            self.notifier,
        )

        # Create watchdog
        self.watchdog = WatchdogService(
            {"failure_detection": {"anomaly_threshold": 2.0, "history_window": 50}}
        )

        await self.watchdog.start()

        # Store reference to order manager in injectors
        self._order_manager = self.order_manager

    async def teardown_test_environment(self):
        """Clean up test environment."""
        if self.watchdog:
            await self.watchdog.stop()

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)

    async def run_chaos_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Run a specific chaos scenario."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.scenarios[scenario_name]
        self.metrics = ChaosTestMetrics(scenario_name)

        logger.info(f"Running chaos scenario: {scenario.name}")

        # Create injector
        injector = self._create_injector(scenario)

        # Start background order generation
        order_task = asyncio.create_task(self._generate_background_orders(scenario))
        self.background_tasks.append(order_task)

        # Start watchdog monitoring
        monitor_task = asyncio.create_task(self._monitor_watchdog_alerts(scenario))
        self.background_tasks.append(monitor_task)

        try:
            # Run baseline period (30 seconds)
            await self._run_baseline_period()

            # Inject chaos
            async with injector.inject_chaos():
                await asyncio.sleep(scenario.duration_seconds)

            # Run recovery period
            recovery_success = await self._run_recovery_period(scenario)

            # Validate recovery
            validation_results = await self._validate_recovery(scenario)

            # Generate report
            report = self._generate_scenario_report(scenario, validation_results)

            logger.info(
                f"Chaos scenario {scenario_name} completed: {'PASS' if validation_results['sla_met'] else 'FAIL'}"
            )

            return report

        finally:
            # Stop background tasks
            order_task.cancel()
            monitor_task.cancel()

    async def _run_baseline_period(self, duration: int = 30):
        """Run baseline period before chaos injection."""
        logger.info("Running baseline period")
        await asyncio.sleep(duration)

    async def _run_recovery_period(self, scenario: ChaosScenario) -> bool:
        """Run recovery period and check if system recovers."""
        logger.info(f"Running recovery period for {scenario.recovery_timeout_seconds}s")

        start_time = time.time()
        recovery_detected = False

        while time.time() - start_time < scenario.recovery_timeout_seconds:
            # Check if system has recovered (simplified check)
            if await self._check_system_recovered():
                recovery_time = time.time() - start_time
                scenario.recovery_time = recovery_time
                scenario.recovery_detected = True
                recovery_detected = True
                logger.info(f"System recovery detected after {recovery_time:.1f}s")
                break

            await asyncio.sleep(5)  # Check every 5 seconds

        return recovery_detected

    async def _check_system_recovered(self) -> bool:
        """Check if system has recovered from chaos."""
        # Simplified recovery check - in real implementation this would
        # check actual system health metrics
        try:
            # Try to execute a test order
            test_signal = {
                "id": "recovery_test_order",
                "strategy_id": "recovery_test",
                "symbol": "BTC/USDT",
                "side": "buy",
                "type": "market",
                "amount": 0.001,
            }

            result = await self.order_executor._execute_with_retry(test_signal)
            return result is not None and result.get("status") == "filled"

        except Exception:
            return False

    async def _validate_recovery(self, scenario: ChaosScenario) -> Dict[str, Any]:
        """Validate that recovery meets SLA requirements."""
        validation = {
            "recovery_detected": scenario.recovery_detected,
            "recovery_time_seconds": scenario.recovery_time,
            "sla_met": False,
            "failure_rate_acceptable": self.metrics.chaos_failure_rate
            <= scenario.expected_failure_rate,
            "watchdog_alerts_fired": len(self.metrics.watchdog_alerts) > 0,
            "recovery_events_logged": len(self.metrics.recovery_events) > 0,
        }

        if (
            scenario.recovery_time
            and scenario.recovery_time <= scenario.sla_recovery_time_seconds
        ):
            validation["sla_met"] = True

        return validation

    async def _generate_background_orders(self, scenario: ChaosScenario):
        """Generate background orders throughout the test."""
        order_count = 0

        while True:
            try:
                # Generate test order
                order_signal = {
                    "id": f"chaos_order_{order_count}",
                    "strategy_id": "chaos_test",
                    "symbol": "BTC/USDT",
                    "side": "buy" if order_count % 2 == 0 else "sell",
                    "type": "market",
                    "amount": 0.001,
                    "timestamp": datetime.now(),
                }

                start_time = time.time()
                result = await self.order_executor._execute_with_retry(order_signal)
                latency = (time.time() - start_time) * 1000

                # Categorize by time period
                now = datetime.now()
                if scenario.start_time and now < scenario.start_time:
                    # Before chaos
                    self.metrics.orders_before_chaos += 1
                    self.metrics.latency_before_chaos.append(latency)
                    if result and result.get("status") != "filled":
                        self.metrics.failures_before_chaos += 1
                elif scenario.end_time and now > scenario.end_time:
                    # After chaos
                    self.metrics.orders_after_chaos += 1
                    self.metrics.latency_after_chaos.append(latency)
                    if result and result.get("status") != "filled":
                        self.metrics.failures_after_chaos += 1
                else:
                    # During chaos
                    self.metrics.orders_during_chaos += 1
                    self.metrics.latency_during_chaos.append(latency)
                    if result and result.get("status") != "filled":
                        self.metrics.failures_during_chaos += 1

                order_count += 1
                await asyncio.sleep(0.1)  # Generate order every 100ms

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Error generating background order: {e}")
                await asyncio.sleep(1)

    async def _monitor_watchdog_alerts(self, scenario: ChaosScenario):
        """Monitor watchdog alerts during chaos testing."""
        # This would integrate with actual watchdog event system
        # For simulation, we'll periodically check system state

        while True:
            try:
                # Simulate watchdog checks
                if scenario.failure_injected and not scenario.recovery_detected:
                    # During chaos period, simulate alerts
                    alert = {
                        "timestamp": datetime.now(),
                        "type": "chaos_failure",
                        "component": "order_executor",
                        "message": f"Chaos scenario {scenario.name} active",
                    }
                    self.metrics.watchdog_alerts.append(alert)

                if scenario.recovery_detected:
                    # Recovery event
                    event = {
                        "timestamp": datetime.now(),
                        "type": "recovery_successful",
                        "component": "order_executor",
                        "message": f"Recovered from {scenario.name}",
                    }
                    self.metrics.recovery_events.append(event)

                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Error monitoring watchdog: {e}")

    def _generate_scenario_report(
        self, scenario: ChaosScenario, validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive report for chaos scenario."""
        report = {
            "scenario": {
                "name": scenario.name,
                "description": scenario.description,
                "duration_seconds": scenario.duration_seconds,
                "expected_failure_rate": scenario.expected_failure_rate,
                "sla_recovery_time_seconds": scenario.sla_recovery_time_seconds,
            },
            "execution": {
                "start_time": scenario.start_time.isoformat()
                if scenario.start_time
                else None,
                "end_time": scenario.end_time.isoformat()
                if scenario.end_time
                else None,
                "recovery_time_seconds": scenario.recovery_time,
                "failure_injected": scenario.failure_injected,
                "recovery_detected": scenario.recovery_detected,
            },
            "metrics": {
                "orders_before_chaos": self.metrics.orders_before_chaos,
                "orders_during_chaos": self.metrics.orders_during_chaos,
                "orders_after_chaos": self.metrics.orders_after_chaos,
                "failures_before_chaos": self.metrics.failures_before_chaos,
                "failures_during_chaos": self.metrics.failures_during_chaos,
                "failures_after_chaos": self.metrics.failures_after_chaos,
                "chaos_failure_rate": self.metrics.chaos_failure_rate,
                "recovery_successful": self.metrics.recovery_successful,
            },
            "validation": validation,
            "watchdog": {
                "alerts_count": len(self.metrics.watchdog_alerts),
                "recovery_events_count": len(self.metrics.recovery_events),
            },
            "timestamp": datetime.now().isoformat(),
        }

        self.chaos_reports.append(report)
        return report

    def export_chaos_reports(self, output_file: str = "chaos_reports.json"):
        """Export all chaos test reports."""
        with open(output_file, "w") as f:
            json.dump(self.chaos_reports, f, indent=2, default=str)

        logger.info(f"Chaos reports exported to {output_file}")


# Pytest fixtures and test cases


@pytest.fixture
async def chaos_runner():
    """Fixture for chaos test runner."""
    runner = ChaosTestRunner({})
    await runner.setup_test_environment()
    yield runner
    await runner.teardown_test_environment()


@pytest.mark.asyncio
async def test_network_partition_chaos(chaos_runner):
    """Test system resilience against network partitions."""
    report = await chaos_runner.run_chaos_scenario("network_partition")

    assert report["execution"]["failure_injected"] == True
    assert report["validation"]["recovery_detected"] == True
    assert report["validation"]["sla_met"] == True

    # Verify failure rate during chaos was high
    assert report["metrics"]["chaos_failure_rate"] > 90.0


@pytest.mark.asyncio
async def test_rate_limit_flood_chaos(chaos_runner):
    """Test system resilience against rate limiting."""
    report = await chaos_runner.run_chaos_scenario("rate_limit_flood")

    assert report["execution"]["failure_injected"] == True
    assert report["validation"]["recovery_detected"] == True
    assert report["validation"]["sla_met"] == True

    # Verify some failures occurred but system recovered
    assert report["metrics"]["failures_during_chaos"] > 0
    assert report["metrics"]["recovery_successful"] == True


@pytest.mark.asyncio
async def test_exchange_downtime_chaos(chaos_runner):
    """Test system resilience against complete exchange downtime."""
    report = await chaos_runner.run_chaos_scenario("exchange_downtime")

    assert report["execution"]["failure_injected"] == True
    assert report["validation"]["recovery_detected"] == True
    assert report["validation"]["sla_met"] == True

    # Verify high failure rate during downtime
    assert report["metrics"]["chaos_failure_rate"] > 95.0


@pytest.mark.asyncio
async def test_database_outage_chaos(chaos_runner):
    """Test system resilience against database outages."""
    report = await chaos_runner.run_chaos_scenario("database_outage")

    assert report["execution"]["failure_injected"] == True
    assert report["validation"]["recovery_detected"] == True
    assert report["validation"]["sla_met"] == True

    # Verify significant failures during outage
    assert report["metrics"]["chaos_failure_rate"] > 70.0


@pytest.mark.asyncio
async def test_chaos_recovery_assertions(chaos_runner):
    """Test that chaos tests assert recovery, not just failure injection."""
    # Run all scenarios
    scenarios = [
        "network_partition",
        "rate_limit_flood",
        "exchange_downtime",
        "database_outage",
    ]

    for scenario_name in scenarios:
        report = await chaos_runner.run_chaos_scenario(scenario_name)

        # Assert that recovery was tested and validated
        assert (
            report["validation"]["recovery_detected"] == True
        ), f"Recovery not detected for {scenario_name}"
        assert (
            report["validation"]["sla_met"] == True
        ), f"SLA not met for {scenario_name}"
        assert (
            report["metrics"]["recovery_successful"] == True
        ), f"Recovery not successful for {scenario_name}"

    # Export final reports
    chaos_runner.export_chaos_reports()


if __name__ == "__main__":
    # Allow running chaos tests directly
    pytest.main([__file__, "-v"])
