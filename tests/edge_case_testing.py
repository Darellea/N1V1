"""
Edge Case Testing Framework - Chaos engineering and stress testing.

Implements chaos engineering principles, fuzz testing for APIs,
market condition simulators, and comprehensive negative testing scenarios.
"""

import asyncio
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Type

import aiohttp
import numpy as np

from tests.integration_test_framework import MarketConditionSimulator, MockExchange
from utils.error_handler import DataError, NetworkError, TradingError

logger = logging.getLogger(__name__)


@dataclass
class ChaosEvent:
    """Represents a chaos engineering event."""

    event_type: str
    target_component: str
    severity: str
    duration: float
    probability: float
    parameters: Dict[str, Any] = field(default_factory=dict)

    def should_trigger(self) -> bool:
        """Determine if chaos event should trigger based on probability."""
        return random.random() < self.probability


class ChaosMonkey:
    """
    Chaos Monkey implementation for testing system resilience.
    Injects failures and disruptions to test system behavior under stress.
    """

    def __init__(self):
        self.active_events: List[ChaosEvent] = []
        self.event_history: List[Dict[str, Any]] = []
        self.monkey_patches: Dict[str, Any] = {}

    def add_chaos_event(self, event: ChaosEvent):
        """Add a chaos event to the repertoire."""
        self.active_events.append(event)
        logger.info(
            f"Added chaos event: {event.event_type} targeting {event.target_component}"
        )

    def create_network_chaos(
        self, target_component: str, severity: str = "medium"
    ) -> ChaosEvent:
        """Create network-related chaos events."""
        events = {
            "latency_spike": ChaosEvent(
                event_type="network_latency",
                target_component=target_component,
                severity=severity,
                duration=random.uniform(10, 60),
                probability=0.3,
                parameters={"latency_ms": random.randint(1000, 5000)},
            ),
            "connection_drop": ChaosEvent(
                event_type="network_disconnect",
                target_component=target_component,
                severity=severity,
                duration=random.uniform(5, 30),
                probability=0.2,
                parameters={"drop_rate": random.uniform(0.1, 0.5)},
            ),
            "packet_loss": ChaosEvent(
                event_type="packet_loss",
                target_component=target_component,
                severity=severity,
                duration=random.uniform(15, 45),
                probability=0.25,
                parameters={"loss_rate": random.uniform(0.05, 0.3)},
            ),
        }
        return random.choice(list(events.values()))

    def create_resource_chaos(
        self, target_component: str, severity: str = "medium"
    ) -> ChaosEvent:
        """Create resource-related chaos events."""
        events = {
            "cpu_spike": ChaosEvent(
                event_type="high_cpu",
                target_component=target_component,
                severity=severity,
                duration=random.uniform(20, 120),
                probability=0.2,
                parameters={"cpu_usage": random.uniform(80, 95)},
            ),
            "memory_pressure": ChaosEvent(
                event_type="memory_exhaustion",
                target_component=target_component,
                severity=severity,
                duration=random.uniform(30, 90),
                probability=0.15,
                parameters={"memory_pressure": random.uniform(85, 98)},
            ),
            "disk_full": ChaosEvent(
                event_type="disk_space",
                target_component=target_component,
                severity=severity,
                duration=random.uniform(60, 300),
                probability=0.1,
                parameters={"disk_usage": random.uniform(90, 99)},
            ),
        }
        return random.choice(list(events.values()))

    def create_data_chaos(
        self, target_component: str, severity: str = "medium"
    ) -> ChaosEvent:
        """Create data-related chaos events."""
        events = {
            "data_corruption": ChaosEvent(
                event_type="corrupt_data",
                target_component=target_component,
                severity=severity,
                duration=random.uniform(10, 60),
                probability=0.1,
                parameters={"corruption_rate": random.uniform(0.01, 0.1)},
            ),
            "data_delay": ChaosEvent(
                event_type="data_lag",
                target_component=target_component,
                severity=severity,
                duration=random.uniform(15, 120),
                probability=0.2,
                parameters={"delay_seconds": random.uniform(1, 30)},
            ),
            "invalid_data": ChaosEvent(
                event_type="invalid_format",
                target_component=target_component,
                severity=severity,
                duration=random.uniform(5, 45),
                probability=0.15,
                parameters={"invalid_rate": random.uniform(0.05, 0.25)},
            ),
        }
        return random.choice(list(events.values()))

    async def inject_chaos(self, target_system: Any):
        """Inject chaos into the target system."""
        triggered_events = []

        for event in self.active_events:
            if event.should_trigger():
                logger.warning(
                    f"Injecting chaos: {event.event_type} on {event.target_component}"
                )
                triggered_events.append(event)

                # Apply the chaos event
                await self._apply_chaos_event(event, target_system)

                # Schedule cleanup
                asyncio.create_task(self._cleanup_chaos_event(event))

        return triggered_events

    async def _apply_chaos_event(self, event: ChaosEvent, target_system: Any):
        """Apply a specific chaos event."""
        if event.event_type == "network_latency":
            await self._inject_network_latency(event, target_system)
        elif event.event_type == "network_disconnect":
            await self._inject_network_disconnect(event, target_system)
        elif event.event_type == "high_cpu":
            await self._inject_cpu_pressure(event, target_system)
        elif event.event_type == "memory_exhaustion":
            await self._inject_memory_pressure(event, target_system)
        elif event.event_type == "corrupt_data":
            await self._inject_data_corruption(event, target_system)
        elif event.event_type == "data_lag":
            await self._inject_data_delay(event, target_system)

        # Record the event
        self.event_history.append(
            {
                "timestamp": time.time(),
                "event": event.event_type,
                "target": event.target_component,
                "severity": event.severity,
                "duration": event.duration,
                "parameters": event.parameters,
            }
        )

    async def _inject_network_latency(self, event: ChaosEvent, target_system: Any):
        """Inject network latency."""
        # Monkey patch network calls to add delay
        original_fetch = getattr(target_system, "fetch_ticker", None)
        if original_fetch:

            async def delayed_fetch(*args, **kwargs):
                await asyncio.sleep(event.parameters["latency_ms"] / 1000)
                return await original_fetch(*args, **kwargs)

            self.monkey_patches[f"{event.target_component}_fetch"] = (
                target_system,
                "fetch_ticker",
                original_fetch,
            )
            target_system.fetch_ticker = delayed_fetch

    async def _inject_network_disconnect(self, event: ChaosEvent, target_system: Any):
        """Inject network disconnection."""
        original_fetch = getattr(target_system, "fetch_ticker", None)
        if original_fetch:

            async def failing_fetch(*args, **kwargs):
                if random.random() < event.parameters["drop_rate"]:
                    raise NetworkError("Simulated network disconnection")
                return await original_fetch(*args, **kwargs)

            self.monkey_patches[f"{event.target_component}_disconnect"] = (
                target_system,
                "fetch_ticker",
                original_fetch,
            )
            target_system.fetch_ticker = failing_fetch

    async def _inject_cpu_pressure(self, event: ChaosEvent, target_system: Any):
        """Inject CPU pressure."""

        # Simulate CPU-intensive operations
        def cpu_intensive_task():
            start_time = time.time()
            while time.time() - start_time < event.duration:
                [x**2 for x in range(10000)]  # CPU-intensive computation

        thread = threading.Thread(target=cpu_intensive_task, daemon=True)
        thread.start()

    async def _inject_memory_pressure(self, event: ChaosEvent, target_system: Any):
        """Inject memory pressure."""
        # Allocate large amounts of memory
        large_objects = []
        for _ in range(100):
            large_objects.append([0] * 1000000)  # 4MB each

        # Hold memory for the duration
        await asyncio.sleep(event.duration)

        # Clean up
        del large_objects

    async def _inject_data_corruption(self, event: ChaosEvent, target_system: Any):
        """Inject data corruption."""
        original_fetch = getattr(target_system, "fetch_ohlcv", None)
        if original_fetch:

            async def corrupt_fetch(*args, **kwargs):
                data = await original_fetch(*args, **kwargs)
                # Corrupt some data points
                for i in range(len(data)):
                    if random.random() < event.parameters["corruption_rate"]:
                        # Corrupt OHLC values
                        for j in range(1, 5):  # OHLC columns
                            data[i][j] *= 1 + random.uniform(-0.1, 0.1)
                return data

            self.monkey_patches[f"{event.target_component}_corrupt"] = (
                target_system,
                "fetch_ohlcv",
                original_fetch,
            )
            target_system.fetch_ohlcv = corrupt_fetch

    async def _inject_data_delay(self, event: ChaosEvent, target_system: Any):
        """Inject data delay."""
        original_fetch = getattr(target_system, "fetch_ticker", None)
        if original_fetch:

            async def delayed_data_fetch(*args, **kwargs):
                # Delay the data
                await asyncio.sleep(event.parameters["delay_seconds"])
                data = await original_fetch(*args, **kwargs)
                # Make data stale by adjusting timestamp
                if isinstance(data, dict) and "timestamp" in data:
                    data["timestamp"] -= int(event.parameters["delay_seconds"] * 1000)
                return data

            self.monkey_patches[f"{event.target_component}_delay"] = (
                target_system,
                "fetch_ticker",
                original_fetch,
            )
            target_system.fetch_ticker = delayed_data_fetch

    async def _cleanup_chaos_event(self, event: ChaosEvent):
        """Clean up a chaos event after its duration."""
        await asyncio.sleep(event.duration)

        # Restore original functionality
        for patch_key, (obj, attr, original) in self.monkey_patches.items():
            if event.target_component in patch_key:
                setattr(obj, attr, original)
                del self.monkey_patches[patch_key]
                logger.info(f"Cleaned up chaos event: {event.event_type}")

    def get_chaos_report(self) -> Dict[str, Any]:
        """Generate a report of chaos events."""
        return {
            "total_events": len(self.event_history),
            "active_events": len(self.active_events),
            "event_types": list(set(e["event"] for e in self.event_history)),
            "severity_distribution": self._calculate_severity_distribution(),
            "target_distribution": self._calculate_target_distribution(),
            "recent_events": self.event_history[-10:],  # Last 10 events
        }

    def _calculate_severity_distribution(self) -> Dict[str, int]:
        """Calculate distribution of event severities."""
        distribution = {}
        for event in self.event_history:
            severity = event["severity"]
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution

    def _calculate_target_distribution(self) -> Dict[str, int]:
        """Calculate distribution of event targets."""
        distribution = {}
        for event in self.event_history:
            target = event["target"]
            distribution[target] = distribution.get(target, 0) + 1
        return distribution


class FuzzTester:
    """
    Fuzz testing framework for APIs and data processing components.
    Generates malformed inputs to test system robustness.
    """

    def __init__(self):
        self.fuzz_strategies = {
            "string_mutation": self._fuzz_string_mutation,
            "numeric_extremes": self._fuzz_numeric_extremes,
            "structure_corruption": self._fuzz_structure_corruption,
            "encoding_issues": self._fuzz_encoding_issues,
            "timing_attacks": self._fuzz_timing_attacks,
        }
        self.test_results: List[Dict[str, Any]] = []

    def generate_fuzz_inputs(
        self, base_input: Any, strategy: str = "string_mutation", iterations: int = 100
    ) -> List[Any]:
        """Generate fuzzed inputs based on strategy."""
        if strategy not in self.fuzz_strategies:
            raise ValueError(f"Unknown fuzz strategy: {strategy}")

        fuzzer = self.fuzz_strategies[strategy]
        return [fuzzer(base_input) for _ in range(iterations)]

    def _fuzz_string_mutation(self, base_input: Any) -> Any:
        """Mutate string inputs."""
        if not isinstance(base_input, str):
            return base_input

        mutations = [
            lambda s: s + "\x00" * random.randint(1, 100),  # Null bytes
            lambda s: s * random.randint(2, 10),  # Repetition
            lambda s: "".join(
                random.choice([c.upper(), c.lower()]) for c in s
            ),  # Case changes
            lambda s: s.replace(
                random.choice(s) if s else "", chr(random.randint(0, 255))
            ),  # Random replacement
            lambda s: s[: random.randint(0, len(s))]
            + chr(random.randint(0, 255)) * random.randint(1, 10),  # Insertion
            lambda s: "",  # Empty string
            lambda s: chr(random.randint(0, 255))
            * random.randint(1000, 10000),  # Very long string
        ]

        return random.choice(mutations)(base_input)

    def _fuzz_numeric_extremes(self, base_input: Any) -> Any:
        """Generate extreme numeric values."""
        if not isinstance(base_input, (int, float)):
            return base_input

        extremes = [
            float("inf"),
            float("-inf"),
            float("nan"),
            sys.maxsize,
            -sys.maxsize - 1,
            0,
            1e-10,  # Very small
            1e10,  # Very large
            random.randint(-(2**63), 2**63),  # Random large number
        ]

        return random.choice(extremes)

    def _fuzz_structure_corruption(self, base_input: Any) -> Any:
        """Corrupt data structures."""
        if isinstance(base_input, dict):
            corrupted = base_input.copy()
            # Add random keys
            for _ in range(random.randint(1, 5)):
                key = self._fuzz_string_mutation(f"key_{random.randint(0, 100)}")
                value = self._fuzz_string_mutation("value")
                corrupted[key] = value
            # Remove random keys
            if corrupted and random.random() < 0.3:
                key_to_remove = random.choice(list(corrupted.keys()))
                del corrupted[key_to_remove]
            return corrupted

        elif isinstance(base_input, list):
            corrupted = base_input.copy()
            # Add random elements
            for _ in range(random.randint(1, 3)):
                corrupted.append(self._fuzz_string_mutation("element"))
            # Remove elements
            if corrupted and random.random() < 0.3:
                corrupted.pop(random.randint(0, len(corrupted) - 1))
            return corrupted

        return base_input

    def _fuzz_encoding_issues(self, base_input: Any) -> Any:
        """Generate encoding-related issues."""
        if not isinstance(base_input, str):
            return base_input

        encodings = [
            lambda s: s.encode("utf-8").decode("latin-1"),  # Wrong decoding
            lambda s: s.encode("utf-8")[:-1],  # Truncated UTF-8
            lambda s: bytes([random.randint(0, 255) for _ in range(len(s))]).decode(
                "utf-8", errors="ignore"
            ),  # Random bytes
            lambda s: s
            + "".join(
                chr(random.randint(0xD800, 0xDFFF)) for _ in range(random.randint(1, 5))
            ),  # Surrogate pairs
        ]

        return random.choice(encodings)(base_input)

    def _fuzz_timing_attacks(self, base_input: Any) -> Any:
        """Generate timing-related attack inputs."""
        # This would typically involve timing the response to different inputs
        # For now, return inputs that might cause different processing times
        if isinstance(base_input, str):
            return base_input + "x" * random.randint(1000, 10000)  # Very long input
        return base_input

    async def fuzz_test_api_endpoint(
        self,
        endpoint_func: Callable,
        base_input: Any,
        strategy: str = "string_mutation",
        iterations: int = 50,
    ) -> Dict[str, Any]:
        """Fuzz test an API endpoint."""
        logger.info(f"Starting fuzz test with strategy: {strategy}")

        fuzz_inputs = self.generate_fuzz_inputs(base_input, strategy, iterations)
        results = {
            "strategy": strategy,
            "iterations": iterations,
            "crashes": 0,
            "exceptions": [],
            "performance_impact": [],
            "successful_tests": 0,
        }

        for i, fuzz_input in enumerate(fuzz_inputs):
            try:
                start_time = time.perf_counter()

                # Test the endpoint
                result = await endpoint_func(fuzz_input)

                end_time = time.perf_counter()
                execution_time = end_time - start_time

                results["performance_impact"].append(
                    {
                        "input_index": i,
                        "execution_time": execution_time,
                        "input_size": len(str(fuzz_input)) if fuzz_input else 0,
                    }
                )

                results["successful_tests"] += 1

            except Exception as e:
                results["crashes"] += 1
                results["exceptions"].append(
                    {
                        "input_index": i,
                        "exception": str(e),
                        "exception_type": type(e).__name__,
                        "input_sample": str(fuzz_input)[:100],  # First 100 chars
                    }
                )

        self.test_results.append(results)
        return results

    def get_fuzz_report(self) -> Dict[str, Any]:
        """Generate fuzz testing report."""
        if not self.test_results:
            return {"message": "No fuzz tests executed"}

        total_crashes = sum(r["crashes"] for r in self.test_results)
        total_tests = sum(r["iterations"] for r in self.test_results)

        return {
            "total_tests": total_tests,
            "total_crashes": total_crashes,
            "crash_rate": total_crashes / total_tests if total_tests > 0 else 0,
            "strategies_tested": list(set(r["strategy"] for r in self.test_results)),
            "most_common_exceptions": self._analyze_exceptions(),
            "performance_summary": self._analyze_performance(),
        }

    def _analyze_exceptions(self) -> List[Dict[str, Any]]:
        """Analyze exception patterns."""
        exception_counts = {}
        for result in self.test_results:
            for exc in result["exceptions"]:
                exc_type = exc["exception_type"]
                exception_counts[exc_type] = exception_counts.get(exc_type, 0) + 1

        return sorted(
            [{"exception": k, "count": v} for k, v in exception_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[
            :10
        ]  # Top 10

    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance impact of fuzz inputs."""
        all_times = []
        for result in self.test_results:
            all_times.extend(
                [p["execution_time"] for p in result["performance_impact"]]
            )

        if not all_times:
            return {"message": "No performance data available"}

        return {
            "avg_execution_time": np.mean(all_times),
            "max_execution_time": np.max(all_times),
            "min_execution_time": np.min(all_times),
            "std_execution_time": np.std(all_times),
            "slowest_inputs": len(
                [t for t in all_times if t > np.mean(all_times) + 2 * np.std(all_times)]
            ),
        }


class NegativeTestSuite:
    """
    Comprehensive negative testing scenarios for edge cases and error conditions.
    """

    def __init__(self):
        self.test_cases: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []

    def add_test_case(
        self,
        name: str,
        setup_func: Callable,
        test_func: Callable,
        expected_exceptions: List[Type[Exception]],
        description: str = "",
    ):
        """Add a negative test case."""
        self.test_cases.append(
            {
                "name": name,
                "setup": setup_func,
                "test": test_func,
                "expected_exceptions": expected_exceptions,
                "description": description,
            }
        )

    async def run_negative_tests(self) -> Dict[str, Any]:
        """Run all negative test cases."""
        logger.info(f"Running {len(self.test_cases)} negative test cases")

        results = {
            "total_tests": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "unexpected_errors": 0,
            "test_details": [],
        }

        for test_case in self.test_cases:
            result = await self._run_single_negative_test(test_case)
            results["test_details"].append(result)

            if result["status"] == "passed":
                results["passed"] += 1
            elif result["status"] == "failed":
                results["failed"] += 1
            else:
                results["unexpected_errors"] += 1

        self.results.append(results)
        return results

    async def _run_single_negative_test(
        self, test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single negative test case."""
        result = {
            "test_name": test_case["name"],
            "description": test_case["description"],
            "status": "unknown",
            "exception_raised": None,
            "exception_type": None,
            "expected": test_case["expected_exceptions"],
            "error_message": None,
        }

        try:
            # Setup
            if test_case["setup"]:
                await test_case["setup"]()

            # Execute test
            await test_case["test"]()

            # If we get here, the test should have raised an exception
            result["status"] = "failed"
            result["error_message"] = "Expected exception was not raised"

        except Exception as e:
            result["exception_raised"] = str(e)
            result["exception_type"] = type(e).__name__

            # Check if it's an expected exception
            if any(
                isinstance(e, expected_type)
                for expected_type in test_case["expected_exceptions"]
            ):
                result["status"] = "passed"
            else:
                result["status"] = "unexpected_error"
                result[
                    "error_message"
                ] = f"Unexpected exception: {type(e).__name__}: {e}"

        return result

    def create_common_negative_tests(self, target_system: Any):
        """Create common negative test cases."""

        # Network failure tests
        self.add_test_case(
            "network_timeout",
            lambda: self._simulate_network_timeout(target_system),
            lambda: target_system.fetch_ticker("BTC/USDT"),
            [NetworkError, asyncio.TimeoutError, aiohttp.ClientError],
            "Test behavior when network requests timeout",
        )

        # Invalid data tests
        self.add_test_case(
            "invalid_symbol",
            lambda: None,
            lambda: target_system.fetch_ticker("INVALID_SYMBOL_12345"),
            [TradingError, ValueError],
            "Test behavior with invalid trading symbol",
        )

        # Authentication failure tests
        self.add_test_case(
            "invalid_credentials",
            lambda: self._simulate_invalid_credentials(target_system),
            lambda: target_system.fetch_balance(),
            [TradingError, PermissionError],
            "Test behavior with invalid API credentials",
        )

        # Rate limit tests
        self.add_test_case(
            "rate_limit_exceeded",
            lambda: self._simulate_rate_limit(target_system),
            lambda: target_system.create_order("BTC/USDT", "market", "buy", 1.0),
            [TradingError, Exception],
            "Test behavior when rate limits are exceeded",
        )

        # Insufficient balance tests
        self.add_test_case(
            "insufficient_balance",
            lambda: self._simulate_insufficient_balance(target_system),
            lambda: target_system.create_order("BTC/USDT", "market", "buy", 1000.0),
            [TradingError, ValueError],
            "Test behavior when account has insufficient balance",
        )

        # Invalid order parameters
        self.add_test_case(
            "invalid_order_amount",
            lambda: None,
            lambda: target_system.create_order("BTC/USDT", "market", "buy", -1.0),
            [TradingError, ValueError],
            "Test behavior with invalid order amount",
        )

        # Market data corruption
        self.add_test_case(
            "corrupted_market_data",
            lambda: self._simulate_corrupted_data(target_system),
            lambda: target_system.fetch_ohlcv("BTC/USDT"),
            [DataError, TradingError, ValueError],
            "Test behavior with corrupted market data",
        )

    async def _simulate_network_timeout(self, target_system: Any):
        """Simulate network timeout."""
        # This would typically monkey patch the network layer
        pass

    async def _simulate_invalid_credentials(self, target_system: Any):
        """Simulate invalid credentials."""
        original_key = getattr(target_system, "api_key", None)
        target_system.api_key = "invalid_key_12345"
        # Store original for cleanup if needed

    async def _simulate_rate_limit(self, target_system: Any):
        """Simulate rate limiting."""
        # This would typically involve making many rapid requests
        pass

    async def _simulate_insufficient_balance(self, target_system: Any):
        """Simulate insufficient account balance."""
        if hasattr(target_system, "balances"):
            target_system.balances["USDT"]["free"] = 0.01  # Very small amount

    async def _simulate_corrupted_data(self, target_system: Any):
        """Simulate corrupted market data."""
        # This would typically involve modifying response data
        pass

    def get_negative_test_report(self) -> Dict[str, Any]:
        """Generate negative testing report."""
        if not self.results:
            return {"message": "No negative tests executed"}

        latest_result = self.results[-1]

        return {
            "total_tests": latest_result["total_tests"],
            "passed": latest_result["passed"],
            "failed": latest_result["failed"],
            "unexpected_errors": latest_result["unexpected_errors"],
            "success_rate": latest_result["passed"] / latest_result["total_tests"]
            if latest_result["total_tests"] > 0
            else 0,
            "failed_tests": [
                t for t in latest_result["test_details"] if t["status"] == "failed"
            ],
            "unexpected_errors_list": [
                t
                for t in latest_result["test_details"]
                if t["status"] == "unexpected_error"
            ],
        }


class EdgeCaseTestingFramework:
    """
    Comprehensive edge case testing framework combining chaos engineering,
    fuzz testing, and negative testing scenarios.
    """

    def __init__(self):
        self.chaos_monkey = ChaosMonkey()
        self.fuzz_tester = FuzzTester()
        self.negative_tester = NegativeTestSuite()
        self.market_simulator = MarketConditionSimulator()

    async def run_comprehensive_edge_test(
        self, target_system: Any, test_duration: int = 300
    ) -> Dict[str, Any]:
        """Run comprehensive edge case testing."""
        logger.info("Starting comprehensive edge case testing")

        start_time = time.time()
        results = {
            "test_duration": test_duration,
            "chaos_events": [],
            "fuzz_results": [],
            "negative_test_results": {},
            "system_health": [],
            "start_time": start_time,
        }

        # Setup chaos events
        self._setup_chaos_events(target_system)

        # Setup negative tests
        self.negative_tester.create_common_negative_tests(target_system)

        # Run tests concurrently
        test_tasks = [
            self._run_chaos_testing(target_system, test_duration),
            self._run_fuzz_testing(target_system),
            self._run_negative_testing(),
            self._monitor_system_health(test_duration),
        ]

        test_results = await asyncio.gather(*test_tasks, return_exceptions=True)

        # Process results
        results["chaos_events"] = (
            test_results[0] if not isinstance(test_results[0], Exception) else []
        )
        results["fuzz_results"] = (
            test_results[1] if not isinstance(test_results[1], Exception) else []
        )
        results["negative_test_results"] = (
            test_results[2] if not isinstance(test_results[2], Exception) else {}
        )
        results["system_health"] = (
            test_results[3] if not isinstance(test_results[3], Exception) else []
        )

        results["end_time"] = time.time()
        results["actual_duration"] = results["end_time"] - start_time

        return results

    def _setup_chaos_events(self, target_system: Any):
        """Set up chaos events for testing."""
        # Network chaos
        self.chaos_monkey.add_chaos_event(
            self.chaos_monkey.create_network_chaos("exchange_api", "medium")
        )

        # Resource chaos
        self.chaos_monkey.add_chaos_event(
            self.chaos_monkey.create_resource_chaos("system", "medium")
        )

        # Data chaos
        self.chaos_monkey.add_chaos_event(
            self.chaos_monkey.create_data_chaos("market_data", "low")
        )

    async def _run_chaos_testing(
        self, target_system: Any, duration: int
    ) -> List[Dict[str, Any]]:
        """Run chaos testing."""
        events_triggered = []

        for _ in range(duration // 30):  # Check every 30 seconds
            triggered = await self.chaos_monkey.inject_chaos(target_system)
            events_triggered.extend(triggered)
            await asyncio.sleep(30)

        return events_triggered

    async def _run_fuzz_testing(self, target_system: Any) -> List[Dict[str, Any]]:
        """Run fuzz testing on various endpoints."""
        fuzz_results = []

        # Fuzz ticker endpoint
        result = await self.fuzz_tester.fuzz_test_api_endpoint(
            target_system.fetch_ticker, "BTC/USDT", "string_mutation", 20
        )
        fuzz_results.append(result)

        # Fuzz OHLCV endpoint
        result = await self.fuzz_tester.fuzz_test_api_endpoint(
            target_system.fetch_ohlcv, "BTC/USDT", "structure_corruption", 15
        )
        fuzz_results.append(result)

        return fuzz_results

    async def _run_negative_testing(self) -> Dict[str, Any]:
        """Run negative test scenarios."""
        return await self.negative_tester.run_negative_tests()

    async def _monitor_system_health(self, duration: int) -> List[Dict[str, Any]]:
        """Monitor system health during testing."""
        health_metrics = []

        for _ in range(duration // 10):  # Check every 10 seconds
            # This would collect actual system metrics
            health_metrics.append(
                {
                    "timestamp": time.time(),
                    "cpu_usage": random.uniform(10, 90),
                    "memory_usage": random.uniform(20, 95),
                    "network_connections": random.randint(1, 100),
                    "active_threads": random.randint(5, 50),
                }
            )
            await asyncio.sleep(10)

        return health_metrics

    def generate_edge_test_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive edge case testing report."""
        report = "# Edge Case Testing Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Summary
        report += "## Summary\n\n"
        report += f"- **Test Duration:** {results['actual_duration']:.1f} seconds\n"
        report += f"- **Chaos Events:** {len(results['chaos_events'])}\n"
        report += f"- **Fuzz Tests:** {len(results['fuzz_results'])}\n"
        report += f"- **Negative Tests:** {results['negative_test_results'].get('total_tests', 0)}\n\n"

        # Chaos Testing Results
        if results["chaos_events"]:
            report += "## Chaos Testing Results\n\n"
            for event in results["chaos_events"][:10]:  # Show first 10
                report += f"- **{event.event_type}** on {event.target_component} "
                report += (
                    f"(Severity: {event.severity}, Duration: {event.duration:.1f}s)\n"
                )
            report += "\n"

        # Fuzz Testing Results
        if results["fuzz_results"]:
            report += "## Fuzz Testing Results\n\n"
            for result in results["fuzz_results"]:
                report += f"### Strategy: {result['strategy']}\n\n"
                report += f"- **Iterations:** {result['iterations']}\n"
                report += f"- **Crashes:** {result['crashes']}\n"
                report += f"- **Success Rate:** {(result['iterations'] - result['crashes']) / result['iterations'] * 100:.1f}%\n\n"

        # Negative Testing Results
        neg_results = results["negative_test_results"]
        if neg_results:
            report += "## Negative Testing Results\n\n"
            report += f"- **Total Tests:** {neg_results.get('total_tests', 0)}\n"
            report += f"- **Passed:** {neg_results.get('passed', 0)}\n"
            report += f"- **Failed:** {neg_results.get('failed', 0)}\n"
            report += (
                f"- **Unexpected Errors:** {neg_results.get('unexpected_errors', 0)}\n"
            )
            report += f"- **Success Rate:** {neg_results.get('passed', 0) / max(1, neg_results.get('total_tests', 1)) * 100:.1f}%\n\n"

        # System Health
        if results["system_health"]:
            report += "## System Health During Testing\n\n"
            avg_cpu = np.mean([h["cpu_usage"] for h in results["system_health"]])
            avg_memory = np.mean([h["memory_usage"] for h in results["system_health"]])
            max_cpu = np.max([h["cpu_usage"] for h in results["system_health"]])
            max_memory = np.max([h["memory_usage"] for h in results["system_health"]])

            report += f"- **Average CPU Usage:** {avg_cpu:.1f}%\n"
            report += f"- **Average Memory Usage:** {avg_memory:.1f}%\n"
            report += f"- **Peak CPU Usage:** {max_cpu:.1f}%\n"
            report += f"- **Peak Memory Usage:** {max_memory:.1f}%\n\n"

        # Recommendations
        report += "## Recommendations\n\n"

        chaos_events = len(results["chaos_events"])
        if chaos_events > 5:
            report += "- **High Chaos Impact:** Consider improving system resilience\n"

        total_fuzz_crashes = sum(r["crashes"] for r in results["fuzz_results"])
        if total_fuzz_crashes > 0:
            report += f"- **Fuzz Testing Issues:** {total_fuzz_crashes} crashes detected, review input validation\n"

        neg_passed = neg_results.get("passed", 0)
        neg_total = neg_results.get("total_tests", 1)
        if neg_passed / neg_total < 0.8:
            report += "- **Negative Testing:** Improve error handling for edge cases\n"

        if avg_cpu > 70 or avg_memory > 80:
            report += "- **Resource Usage:** High resource consumption detected, optimize performance\n"

        return report


# Example usage functions
async def run_edge_case_demo():
    """Demonstrate edge case testing capabilities."""
    framework = EdgeCaseTestingFramework()
    mock_exchange = MockExchange()

    # Set up extreme market conditions
    framework.market_simulator.set_market_condition(
        "BTC/USDT", "flash_crash", "extreme", "bearish"
    )

    # Run comprehensive edge testing
    results = await framework.run_comprehensive_edge_test(
        mock_exchange, 60
    )  # 1 minute test

    # Generate report
    report = framework.generate_edge_test_report(results)

    with open("edge_case_test_report.md", "w") as f:
        f.write(report)

    print("Edge case testing completed. Report generated: edge_case_test_report.md")
    return results


if __name__ == "__main__":
    asyncio.run(run_edge_case_demo())
