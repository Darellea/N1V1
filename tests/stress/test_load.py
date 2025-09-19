#!/usr/bin/env python3
"""
Load stress testing for N1V1 trading system.

This module implements comprehensive load testing to validate system performance
under high concurrency and order volume. It simulates realistic trading scenarios
with multiple concurrent strategies generating thousands of orders.

Key Features:
- Concurrent strategy simulation (50+ strategies)
- Burst order generation (10k+ orders)
- Latency validation (<150ms p95)
- Failure rate monitoring (<0.5%)
- Performance metrics collection
- SLA compliance verification
"""

import asyncio
import time
import logging
import statistics
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime, timedelta
import json
import argparse
import os

from core.order_executor import OrderExecutor
from core.order_manager import OrderManager
from core.performance_tracker import PerformanceTracker
from notifier.discord_bot import DiscordNotifier
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for load stress testing."""
    num_strategies: int = 50
    orders_per_burst: int = 1000
    num_bursts: int = 10
    burst_interval: float = 1.0  # seconds between bursts
    max_concurrent_orders: int = 100
    target_latency_p95: float = 150.0  # ms
    target_failure_rate: float = 0.005  # 0.5%
    test_duration: int = 300  # seconds
    ramp_up_time: int = 30  # seconds
    cooldown_time: int = 30  # seconds

    def __post_init__(self):
        self.total_orders = self.num_strategies * self.orders_per_burst * self.num_bursts


@dataclass
class LoadTestMetrics:
    """Comprehensive metrics collected during load testing."""
    total_orders_submitted: int = 0
    total_orders_processed: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    order_latencies: List[float] = field(default_factory=list)
    strategy_latencies: Dict[str, List[float]] = field(default_factory=dict)
    throughput_per_second: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        total = self.total_orders_processed
        return (self.failed_orders / total) * 100 if total > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        """Calculate average latency."""
        return statistics.mean(self.order_latencies) if self.order_latencies else 0.0

    @property
    def p95_latency(self) -> float:
        """Calculate 95th percentile latency."""
        return np.percentile(self.order_latencies, 95) if self.order_latencies else 0.0

    @property
    def p99_latency(self) -> float:
        """Calculate 99th percentile latency."""
        return np.percentile(self.order_latencies, 99) if self.order_latencies else 0.0

    @property
    def avg_throughput(self) -> float:
        """Calculate average throughput."""
        return statistics.mean(self.throughput_per_second) if self.throughput_per_second else 0.0


class MockStrategy:
    """Mock trading strategy for load testing."""

    def __init__(self, strategy_id: str, config: LoadTestConfig):
        self.strategy_id = strategy_id
        self.config = config
        self.order_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.latencies: List[float] = []

    def generate_order_signal(self) -> Dict[str, Any]:
        """Generate a mock order signal."""
        self.order_count += 1

        # Simulate realistic trading signal
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]
        sides = ["buy", "sell"]

        return {
            "id": f"{self.strategy_id}_order_{self.order_count}",
            "strategy_id": self.strategy_id,
            "symbol": np.random.choice(symbols),
            "side": np.random.choice(sides),
            "type": "market",
            "amount": np.random.uniform(0.001, 1.0),
            "price": None,  # Market order
            "timestamp": datetime.now(),
            "metadata": {
                "strategy_version": "1.0.0",
                "risk_level": np.random.choice(["low", "medium", "high"]),
                "confidence": np.random.uniform(0.5, 1.0)
            }
        }

    def record_result(self, latency: float, success: bool):
        """Record order execution result."""
        self.latencies.append(latency)
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        total_orders = self.success_count + self.failure_count
        success_rate = (self.success_count / total_orders) * 100 if total_orders > 0 else 0.0
        avg_latency = statistics.mean(self.latencies) if self.latencies else 0.0

        return {
            "strategy_id": self.strategy_id,
            "total_orders": total_orders,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "p95_latency": np.percentile(self.latencies, 95) if self.latencies else 0.0
        }


class MockOrderManager:
    """Mock order manager for load testing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.executed_orders: Dict[str, Dict[str, Any]] = {}
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        self.execution_latencies: List[float] = []
        self.failure_rate = 0.002  # 0.2% baseline failure rate
        self.latency_distribution = {
            'mean': 50.0,  # ms
            'std': 25.0   # ms
        }

    async def execute_order(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a mock order with realistic latency and failure simulation."""
        start_time = time.time()

        order_id = signal["id"]

        # Simulate processing delay
        processing_delay = np.random.exponential(0.01)  # Mean 10ms
        await asyncio.sleep(processing_delay)

        # Simulate random failures
        if np.random.random() < self.failure_rate:
            # Failed order
            result = {
                "id": order_id,
                "status": "failed",
                "error": "Simulated exchange error",
                "timestamp": datetime.now(),
                "latency_ms": (time.time() - start_time) * 1000
            }
        else:
            # Successful order
            # Simulate realistic latency with some outliers
            base_latency = np.random.normal(
                self.latency_distribution['mean'],
                self.latency_distribution['std']
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
                "latency_ms": total_latency
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


class LoadTestRunner:
    """Main load test runner coordinating multiple strategies."""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.metrics = LoadTestMetrics()
        self.strategies: Dict[str, MockStrategy] = {}
        self.order_manager = MockOrderManager({})
        self.executor = None
        self.notifier = None
        self.performance_tracker = None

        # Initialize strategies
        for i in range(config.num_strategies):
            strategy_id = f"strategy_{i:03d}"
            self.strategies[strategy_id] = MockStrategy(strategy_id, config)

        # Initialize order executor
        self.order_executor = OrderExecutor(
            {"max_retries": 2, "retry_base_delay": 0.1},
            self.order_manager,
            self.performance_tracker,
            self.notifier
        )

        # Test control
        self.running = False
        self.start_time = None
        self.end_time = None

    async def run_load_test(self) -> LoadTestMetrics:
        """Run the complete load test."""
        logger.info(f"Starting load test: {self.config.num_strategies} strategies, "
                   f"{self.config.total_orders} total orders")

        self.start_time = time.time()
        self.running = True

        try:
            # Ramp up phase
            await self._ramp_up()

            # Main test phase
            await self._run_main_test()

            # Cooldown phase
            await self._cooldown()

        finally:
            self.running = False
            self.end_time = time.time()

        # Calculate final metrics
        self._calculate_final_metrics()

        logger.info(f"Load test completed: {self.metrics.total_orders_processed} orders processed, "
                   f"{self.metrics.failure_rate:.2f}% failure rate, "
                   f"{self.metrics.p95_latency:.1f}ms p95 latency")

        return self.metrics

    async def _ramp_up(self):
        """Ramp up test load gradually."""
        logger.info(f"Ramping up over {self.config.ramp_up_time} seconds")

        ramp_start = time.time()
        ramp_end = ramp_start + self.config.ramp_up_time

        while time.time() < ramp_end and self.running:
            # Gradually increase load
            progress = (time.time() - ramp_start) / self.config.ramp_up_time
            current_strategies = int(self.config.num_strategies * progress)

            if current_strategies > 0:
                # Run a small burst with subset of strategies
                await self._run_order_burst(current_strategies, orders_per_strategy=10)

            await asyncio.sleep(1.0)

        logger.info("Ramp up completed")

    async def _run_main_test(self):
        """Run the main test phase with full load."""
        logger.info("Starting main test phase")

        for burst in range(self.config.num_bursts):
            if not self.running:
                break

            logger.info(f"Running burst {burst + 1}/{self.config.num_bursts}")

            # Run order burst
            await self._run_order_burst(
                self.config.num_strategies,
                self.config.orders_per_burst
            )

            # Wait between bursts
            await asyncio.sleep(self.config.burst_interval)

        logger.info("Main test phase completed")

    async def _run_order_burst(self, num_strategies: int, orders_per_strategy: int):
        """Run a burst of orders from multiple strategies."""
        burst_start = time.time()

        # Select strategies for this burst
        active_strategy_ids = list(self.strategies.keys())[:num_strategies]
        active_strategies = [self.strategies[sid] for sid in active_strategy_ids]

        # Generate orders for all strategies concurrently
        tasks = []
        for strategy in active_strategies:
            task = asyncio.create_task(
                self._run_strategy_burst(strategy, orders_per_strategy)
            )
            tasks.append(task)

        # Wait for all strategies to complete their burst
        await asyncio.gather(*tasks, return_exceptions=True)

        # Record throughput
        burst_duration = time.time() - burst_start
        orders_in_burst = num_strategies * orders_per_strategy
        throughput = orders_in_burst / burst_duration if burst_duration > 0 else 0
        self.metrics.throughput_per_second.append(throughput)

    async def _run_strategy_burst(self, strategy: MockStrategy, num_orders: int):
        """Run a burst of orders for a single strategy."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_orders)

        async def execute_single_order(order_signal: Dict[str, Any]):
            async with semaphore:
                order_start = time.time()

                # Execute order
                result = await self.order_executor._execute_with_retry(order_signal)

                order_latency = (time.time() - order_start) * 1000  # Convert to ms

                # Record metrics
                success = result is not None and result.get("status") == "filled"
                strategy.record_result(order_latency, success)

                # Update global metrics
                self.metrics.total_orders_submitted += 1
                if result:
                    self.metrics.total_orders_processed += 1
                    self.metrics.order_latencies.append(order_latency)

                    if success:
                        self.metrics.successful_orders += 1
                    else:
                        self.metrics.failed_orders += 1
                        error_type = result.get("error", "unknown")
                        self.metrics.error_counts[error_type] = self.metrics.error_counts.get(error_type, 0) + 1

                # Store strategy latency
                if strategy.strategy_id not in self.metrics.strategy_latencies:
                    self.metrics.strategy_latencies[strategy.strategy_id] = []
                self.metrics.strategy_latencies[strategy.strategy_id].append(order_latency)

        # Generate and execute orders for this strategy
        tasks = []
        for _ in range(num_orders):
            order_signal = strategy.generate_order_signal()
            task = asyncio.create_task(execute_single_order(order_signal))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _cooldown(self):
        """Cooldown phase to ensure all orders are processed."""
        logger.info(f"Starting cooldown for {self.config.cooldown_time} seconds")

        await asyncio.sleep(self.config.cooldown_time)

        # Cancel any remaining orders
        await self.order_executor.cancel_all_orders()

        logger.info("Cooldown completed")

    def _calculate_final_metrics(self):
        """Calculate final test metrics."""
        # Ensure we have processed all submitted orders
        self.metrics.total_orders_processed = min(
            self.metrics.total_orders_submitted,
            self.metrics.successful_orders + self.metrics.failed_orders
        )

    def validate_sla_compliance(self) -> Dict[str, Any]:
        """Validate SLA compliance based on test results."""
        sla_results = {
            "latency_sla_met": self.metrics.p95_latency <= self.config.target_latency_p95,
            "failure_rate_sla_met": self.metrics.failure_rate <= self.config.target_failure_rate,
            "throughput_sufficient": self.metrics.avg_throughput >= 100,  # 100 orders/sec minimum
            "overall_pass": False
        }

        sla_results["overall_pass"] = (
            sla_results["latency_sla_met"] and
            sla_results["failure_rate_sla_met"] and
            sla_results["throughput_sufficient"]
        )

        return sla_results

    def get_strategy_performance_report(self) -> List[Dict[str, Any]]:
        """Get detailed performance report for each strategy."""
        return [strategy.get_stats() for strategy in self.strategies.values()]

    def export_results(self, output_file: str):
        """Export test results to JSON file."""
        results = {
            "test_config": {
                "num_strategies": self.config.num_strategies,
                "orders_per_burst": self.config.orders_per_burst,
                "num_bursts": self.config.num_bursts,
                "total_orders": self.config.total_orders,
                "target_latency_p95": self.config.target_latency_p95,
                "target_failure_rate": self.config.target_failure_rate
            },
            "performance_metrics": {
                "total_orders_submitted": self.metrics.total_orders_submitted,
                "total_orders_processed": self.metrics.total_orders_processed,
                "successful_orders": self.metrics.successful_orders,
                "failed_orders": self.metrics.failed_orders,
                "failure_rate_percent": self.metrics.failure_rate,
                "avg_latency_ms": self.metrics.avg_latency,
                "p95_latency_ms": self.metrics.p95_latency,
                "p99_latency_ms": self.metrics.p99_latency,
                "avg_throughput_orders_per_sec": self.metrics.avg_throughput,
                "error_breakdown": self.metrics.error_counts
            },
            "sla_compliance": self.validate_sla_compliance(),
            "strategy_performance": self.get_strategy_performance_report(),
            "test_duration_seconds": self.end_time - self.start_time if self.end_time else 0,
            "timestamp": datetime.now().isoformat()
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Test results exported to {output_file}")


async def run_load_test(config: LoadTestConfig, output_file: str = "load_test_results.json"):
    """Run load stress test with given configuration."""
    runner = LoadTestRunner(config)

    try:
        metrics = await runner.run_load_test()

        # Export results
        runner.export_results(output_file)

        # Validate SLA compliance
        sla_results = runner.validate_sla_compliance()

        # Print summary
        print("\n" + "="*80)
        print("LOAD STRESS TEST RESULTS")
        print("="*80)
        print(f"Strategies: {config.num_strategies}")
        print(f"Total Orders: {metrics.total_orders_processed}")
        print(f"Failure Rate: {metrics.failure_rate:.2f}%")
        print(f"Avg Latency: {metrics.avg_latency:.1f}ms")
        print(f"P95 Latency: {metrics.p95_latency:.1f}ms")
        print(f"P99 Latency: {metrics.p99_latency:.1f}ms")
        print(f"Avg Throughput: {metrics.avg_throughput:.1f} orders/sec")
        print()
        print("SLA COMPLIANCE:")
        print(f"Latency (<{config.target_latency_p95}ms p95): {'✓' if sla_results['latency_sla_met'] else '✗'}")
        print(f"Failure Rate (<{config.target_failure_rate*100:.1f}%): {'✓' if sla_results['failure_rate_sla_met'] else '✗'}")
        print(f"Throughput (>100/sec): {'✓' if sla_results['throughput_sufficient'] else '✗'}")
        print(f"Overall SLA: {'PASS' if sla_results['overall_pass'] else 'FAIL'}")
        print("="*80)

        return sla_results['overall_pass']

    except Exception as e:
        logger.exception(f"Load test failed: {e}")
        return False


def main():
    """Main entry point for load testing."""
    parser = argparse.ArgumentParser(description="N1V1 Load Stress Test")
    parser.add_argument("--strategies", type=int, default=50,
                       help="Number of concurrent strategies")
    parser.add_argument("--orders-per-burst", type=int, default=1000,
                       help="Orders per burst per strategy")
    parser.add_argument("--bursts", type=int, default=10,
                       help="Number of bursts")
    parser.add_argument("--burst-interval", type=float, default=1.0,
                       help="Seconds between bursts")
    parser.add_argument("--target-latency", type=float, default=150.0,
                       help="Target P95 latency in ms")
    parser.add_argument("--target-failure-rate", type=float, default=0.005,
                       help="Target failure rate (0.005 = 0.5%)")
    parser.add_argument("--output", default="load_test_results.json",
                       help="Output file for results")
    parser.add_argument("--duration", type=int, default=300,
                       help="Test duration in seconds")

    args = parser.parse_args()

    config = LoadTestConfig(
        num_strategies=args.strategies,
        orders_per_burst=args.orders_per_burst,
        num_bursts=args.bursts,
        burst_interval=args.burst_interval,
        target_latency_p95=args.target_latency,
        target_failure_rate=args.target_failure_rate,
        test_duration=args.duration
    )

    logger.info("Starting N1V1 load stress test")
    logger.info(f"Configuration: {config}")

    # Run the test
    success = asyncio.run(run_load_test(config, args.output))

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
