"""
Integration Test Framework - Comprehensive testing infrastructure.

Provides realistic trading scenarios, mock exchanges, end-to-end validation,
and performance benchmarking for the N1V1 Trading Framework.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from utils.constants import (
    DEFAULT_RISK_CONFIG,
    DEFAULT_TRADING_CONFIG,
)
from utils.error_handler import ErrorHandler, TradingError

logger = logging.getLogger(__name__)


@dataclass
class MockExchange:
    """Mock exchange for realistic trading simulation."""

    name: str = "mock_exchange"
    base_url: str = "https://mock.exchange.com"
    api_key: str = "mock_api_key"
    api_secret: str = "mock_api_secret"
    sandbox: bool = True

    # Market data
    symbols: List[str] = field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
    )
    prices: Dict[str, float] = field(
        default_factory=lambda: {
            "BTC/USDT": 50000.0,
            "ETH/USDT": 3000.0,
            "ADA/USDT": 1.50,
        }
    )

    # Order book data
    order_books: Dict[str, Dict[str, List[List[float]]]] = field(default_factory=dict)

    # Account data
    balances: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "USDT": {"free": 10000.0, "used": 0.0, "total": 10000.0},
            "BTC": {"free": 0.2, "used": 0.0, "total": 0.2},
            "ETH": {"free": 3.0, "used": 0.0, "total": 3.0},
        }
    )

    # Orders
    orders: List[Dict[str, Any]] = field(default_factory=list)
    order_counter: int = 0

    # Trades
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize order books and other data structures."""
        for symbol in self.symbols:
            self.order_books[symbol] = {
                "bids": [
                    [self.prices[symbol] * 0.999, 10.0],
                    [self.prices[symbol] * 0.998, 15.0],
                ],
                "asks": [
                    [self.prices[symbol] * 1.001, 10.0],
                    [self.prices[symbol] * 1.002, 15.0],
                ],
            }

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker data."""
        if symbol not in self.prices:
            raise TradingError(f"Symbol {symbol} not found")

        price = self.prices[symbol]
        # Add some price movement
        price *= 1 + np.random.normal(0, 0.001)

        return {
            "symbol": symbol,
            "last": price,
            "bid": price * 0.999,
            "ask": price * 1.001,
            "volume": np.random.uniform(100, 1000),
            "timestamp": int(time.time() * 1000),
        }

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> List[List[float]]:
        """Fetch OHLCV data."""
        if symbol not in self.prices:
            raise TradingError(f"Symbol {symbol} not found")

        base_price = self.prices[symbol]
        data = []

        for i in range(limit):
            timestamp = int((time.time() - (limit - i) * 3600) * 1000)
            open_price = base_price * (1 + np.random.normal(0, 0.01))
            high_price = open_price * (1 + abs(np.random.normal(0, 0.005)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.005)))
            close_price = open_price * (1 + np.random.normal(0, 0.01))
            volume = np.random.uniform(10, 100)

            data.append(
                [timestamp, open_price, high_price, low_price, close_price, volume]
            )

        return data

    async def fetch_order_book(
        self, symbol: str, limit: int = 20
    ) -> Dict[str, List[List[float]]]:
        """Fetch order book."""
        if symbol not in self.order_books:
            raise TradingError(f"Symbol {symbol} not found")

        return self.order_books[symbol]

    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create an order."""
        self.order_counter += 1
        order_id = f"order_{self.order_counter}"

        order = {
            "id": order_id,
            "clientOrderId": f"client_{order_id}",
            "timestamp": int(time.time() * 1000),
            "status": "open",
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
            "filled": 0.0,
            "remaining": amount,
            "cost": 0.0,
        }

        self.orders.append(order)

        # Simulate partial fill after a delay
        asyncio.create_task(self._simulate_fill(order_id))

        return order

    async def _simulate_fill(self, order_id: str):
        """Simulate order fill."""
        await asyncio.sleep(np.random.uniform(0.1, 2.0))

        for order in self.orders:
            if order["id"] == order_id and order["status"] == "open":
                fill_amount = order["amount"] * np.random.uniform(0.5, 1.0)
                order["filled"] = fill_amount
                order["remaining"] = order["amount"] - fill_amount
                order["cost"] = fill_amount * (
                    order["price"] or self.prices[order["symbol"]]
                )

                if order["remaining"] <= 0.0001:
                    order["status"] = "closed"
                else:
                    order["status"] = "open"

                # Record trade
                trade = {
                    "id": f"trade_{len(self.trades) + 1}",
                    "order": order_id,
                    "timestamp": int(time.time() * 1000),
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "amount": fill_amount,
                    "price": order["price"] or self.prices[order["symbol"]],
                    "cost": order["cost"],
                }
                self.trades.append(trade)
                break

    async def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """Fetch account balance."""
        return self.balances.copy()

    async def fetch_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch orders."""
        if symbol:
            return [order for order in self.orders if order["symbol"] == symbol]
        return self.orders.copy()

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        for order in self.orders:
            if order["id"] == order_id:
                order["status"] = "canceled"
                return order

        raise TradingError(f"Order {order_id} not found")


@dataclass
class MarketConditionSimulator:
    """Simulates various market conditions for stress testing."""

    volatility_levels: Dict[str, float] = field(
        default_factory=lambda: {
            "low": 0.005,  # 0.5% daily volatility
            "normal": 0.02,  # 2% daily volatility
            "high": 0.08,  # 8% daily volatility
            "extreme": 0.15,  # 15% daily volatility
        }
    )

    trend_types: List[str] = field(
        default_factory=lambda: [
            "bullish",
            "bearish",
            "sideways",
            "volatile",
            "flash_crash",
        ]
    )

    def __init__(self):
        self.current_conditions: Dict[str, Any] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}

    def set_market_condition(
        self,
        symbol: str,
        condition: str,
        volatility: str = "normal",
        trend: str = "sideways",
    ):
        """Set market conditions for a symbol."""
        self.current_conditions[symbol] = {
            "condition": condition,
            "volatility": self.volatility_levels.get(
                volatility, self.volatility_levels["normal"]
            ),
            "trend": trend,
            "start_time": time.time(),
        }

        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []

    def generate_price_movement(self, symbol: str, current_price: float) -> float:
        """Generate realistic price movement based on current conditions."""
        if symbol not in self.current_conditions:
            # Default normal market
            return current_price * (1 + np.random.normal(0, 0.001))

        condition = self.current_conditions[symbol]

        # Base movement
        base_movement = np.random.normal(
            0, condition["volatility"] / 16
        )  # Hourly movement

        # Add trend bias
        if condition["trend"] == "bullish":
            base_movement += abs(np.random.normal(0, condition["volatility"] / 32))
        elif condition["trend"] == "bearish":
            base_movement -= abs(np.random.normal(0, condition["volatility"] / 32))

        # Add condition-specific behavior
        if condition["condition"] == "flash_crash":
            # Sudden large drop
            if np.random.random() < 0.01:  # 1% chance
                base_movement -= condition["volatility"] * 5
        elif condition["condition"] == "high_volatility":
            base_movement *= 2

        new_price = current_price * (1 + base_movement)

        # Store history
        self.price_history[symbol].append(new_price)
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]

        return max(new_price, 0.01)  # Prevent negative prices

    def generate_volume(self, symbol: str, base_volume: float) -> float:
        """Generate realistic volume based on conditions."""
        if symbol not in self.current_conditions:
            return base_volume * np.random.uniform(0.5, 1.5)

        condition = self.current_conditions[symbol]
        volatility_multiplier = (
            condition["volatility"] / self.volatility_levels["normal"]
        )

        volume = base_volume * np.random.uniform(0.5, 2.0) * volatility_multiplier

        self.volume_history[symbol].append(volume)
        if len(self.volume_history[symbol]) > 1000:
            self.volume_history[symbol] = self.volume_history[symbol][-1000:]

        return volume


class IntegrationTestFramework:
    """
    Comprehensive integration test framework for the N1V1 Trading Framework.
    """

    def __init__(self):
        self.mock_exchange = MockExchange()
        self.market_simulator = MarketConditionSimulator()
        self.test_results: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.error_handler = ErrorHandler()

    async def setup_test_environment(
        self, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Set up test environment with mock components."""
        test_config = config or {
            "environment": {"mode": "paper", "debug": True},
            "exchange": {"name": "mock_exchange", "sandbox": True},
            "trading": DEFAULT_TRADING_CONFIG,
            "risk_management": DEFAULT_RISK_CONFIG,
        }

        # Initialize mock exchange
        await self._initialize_mock_exchange()

        # Set up market conditions
        self._setup_market_conditions()

        return test_config

    async def _initialize_mock_exchange(self):
        """Initialize mock exchange with realistic data."""
        # Add more symbols and realistic price data
        additional_symbols = ["SOL/USDT", "DOT/USDT", "LINK/USDT"]
        self.mock_exchange.symbols.extend(additional_symbols)

        for symbol in additional_symbols:
            base_price = np.random.uniform(10, 500)
            self.mock_exchange.prices[symbol] = base_price

    def _setup_market_conditions(self):
        """Set up various market conditions for testing."""
        # Set different conditions for different symbols
        self.market_simulator.set_market_condition(
            "BTC/USDT", "normal", "normal", "bullish"
        )
        self.market_simulator.set_market_condition(
            "ETH/USDT", "volatile", "high", "sideways"
        )
        self.market_simulator.set_market_condition(
            "ADA/USDT", "trending", "normal", "bearish"
        )

    async def run_trading_scenario(
        self, scenario_name: str, strategy_func: Callable, duration_minutes: int = 60
    ) -> Dict[str, Any]:
        """Run a complete trading scenario."""
        logger.info(f"Starting trading scenario: {scenario_name}")

        start_time = time.time()
        scenario_results = {
            "scenario": scenario_name,
            "start_time": start_time,
            "duration": duration_minutes,
            "trades": [],
            "performance": {},
            "errors": [],
        }

        try:
            # Initialize trading components
            await self._initialize_trading_components()

            # Run scenario
            await self._execute_scenario(
                strategy_func, duration_minutes, scenario_results
            )

            # Calculate performance metrics
            scenario_results["performance"] = self._calculate_scenario_performance(
                scenario_results
            )

        except Exception as e:
            logger.exception(f"Error in scenario {scenario_name}: {e}")
            scenario_results["errors"].append(str(e))

        scenario_results["end_time"] = time.time()
        scenario_results["execution_time"] = scenario_results["end_time"] - start_time

        self.test_results.append(scenario_results)
        return scenario_results

    async def _initialize_trading_components(self):
        """Initialize trading components for testing."""
        # This would initialize actual trading components with mock dependencies
        pass

    async def _execute_scenario(
        self,
        strategy_func: Callable,
        duration_minutes: int,
        scenario_results: Dict[str, Any],
    ):
        """Execute the trading scenario."""
        end_time = time.time() + (duration_minutes * 60)

        while time.time() < end_time:
            try:
                # Update market data
                await self._update_market_data()

                # Execute strategy
                trades = await strategy_func()

                if trades:
                    scenario_results["trades"].extend(trades)

                # Small delay to simulate real-time execution
                await asyncio.sleep(1)

            except Exception as e:
                logger.exception(f"Error during scenario execution: {e}")
                scenario_results["errors"].append(str(e))

    async def _update_market_data(self):
        """Update market data with simulated movements."""
        for symbol in self.mock_exchange.symbols:
            current_price = self.mock_exchange.prices[symbol]
            new_price = self.market_simulator.generate_price_movement(
                symbol, current_price
            )
            self.mock_exchange.prices[symbol] = new_price

    def _calculate_scenario_performance(
        self, scenario_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate performance metrics for the scenario."""
        trades = scenario_results.get("trades", [])

        if not trades:
            return {"total_trades": 0, "profit_loss": 0.0}

        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.get("profit", 0) > 0)
        losing_trades = total_trades - winning_trades

        total_profit = sum(trade.get("profit", 0) for trade in trades)
        total_volume = sum(trade.get("volume", 0) for trade in trades)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_volume": total_volume,
            "avg_profit_per_trade": total_profit / total_trades
            if total_trades > 0
            else 0,
            "sharpe_ratio": self._calculate_sharpe_ratio(trades),
        }

    def _calculate_sharpe_ratio(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate Sharpe ratio for the trades."""
        if not trades:
            return 0.0

        profits = [trade.get("profit", 0) for trade in trades]
        returns = np.array(profits)

        if len(returns) < 2:
            return 0.0

        # Annualize assuming daily returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Sharpe ratio = (mean return - risk free rate) / std deviation
        risk_free_rate = 0.02 / 365  # Daily risk-free rate
        sharpe = (mean_return - risk_free_rate) / std_return

        return sharpe * np.sqrt(365)  # Annualize

    async def run_stress_test(
        self, test_name: str, stress_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run stress test with extreme market conditions."""
        logger.info(f"Starting stress test: {test_name}")

        # Set extreme market conditions
        for symbol in self.mock_exchange.symbols:
            self.market_simulator.set_market_condition(
                symbol,
                stress_config.get("condition", "extreme"),
                stress_config.get("volatility", "extreme"),
                stress_config.get("trend", "volatile"),
            )

        # Run test scenario
        results = await self.run_trading_scenario(
            f"stress_{test_name}",
            self._stress_test_strategy,
            stress_config.get("duration", 30),
        )

        # Additional stress metrics
        results["stress_metrics"] = {
            "max_drawdown": self._calculate_max_drawdown(results["trades"]),
            "volatility_exposure": self._calculate_volatility_exposure(
                results["trades"]
            ),
            "error_rate": len(results["errors"]) / max(1, len(results["trades"])),
        }

        return results

    async def _stress_test_strategy(self) -> List[Dict[str, Any]]:
        """Simple strategy for stress testing."""
        trades = []

        # Random trading decisions
        if np.random.random() < 0.1:  # 10% chance to trade
            symbol = np.random.choice(self.mock_exchange.symbols)
            side = np.random.choice(["buy", "sell"])
            amount = np.random.uniform(0.001, 0.01)

            trade = {
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": self.mock_exchange.prices[symbol],
                "profit": np.random.normal(0, 10),  # Random profit/loss
            }
            trades.append(trade)

        return trades

    def _calculate_max_drawdown(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate maximum drawdown from trades."""
        if not trades:
            return 0.0

        cumulative = 0
        peak = 0
        max_drawdown = 0

        for trade in trades:
            cumulative += trade.get("profit", 0)
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_volatility_exposure(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate volatility exposure from trades."""
        if not trades:
            return 0.0

        profits = [trade.get("profit", 0) for trade in trades]
        return np.std(profits) if profits else 0.0

    async def run_performance_benchmark(
        self, benchmark_name: str, operations: List[Callable], strict_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Run performance benchmark on specified operations.

        Args:
            benchmark_name: Name of the benchmark
            operations: List of operations to benchmark
            strict_mode: If True, exceptions cause immediate failure.
                        If False, exceptions are recorded but benchmark continues.
        """
        logger.info(f"Starting performance benchmark: {benchmark_name}")

        benchmark_results = {
            "benchmark": benchmark_name,
            "start_time": time.time(),
            "operation_results": [],
            "strict_mode": strict_mode,
        }

        for operation in operations:
            op_result = await self._benchmark_operation(operation, strict_mode=strict_mode)
            benchmark_results["operation_results"].append(op_result)

        benchmark_results["end_time"] = time.time()
        benchmark_results["total_time"] = (
            benchmark_results["end_time"] - benchmark_results["start_time"]
        )

        # Calculate aggregate metrics
        benchmark_results["aggregate_metrics"] = self._calculate_aggregate_metrics(
            benchmark_results["operation_results"]
        )

        self.performance_metrics[benchmark_name] = benchmark_results
        return benchmark_results

    async def _benchmark_operation(
        self, operation: Callable, strict_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Benchmark a single operation with proper exception handling.

        Args:
            operation: The operation to benchmark
            strict_mode: If True, exceptions are raised immediately.
                        If False, exceptions are recorded but not raised.

        Returns:
            Dictionary containing benchmark results and success/failure status
        """
        iterations = 100
        successful_times = []
        failed_times = []
        errors = []

        for i in range(iterations):
            start_time = time.perf_counter()

            try:
                await operation()
                execution_time = time.perf_counter() - start_time
                successful_times.append(execution_time)

            except Exception as e:
                execution_time = time.perf_counter() - start_time
                failed_times.append(execution_time)
                errors.append((i, str(e), type(e).__name__))
                logger.error(f"Benchmark operation failed: {e}")
                if strict_mode:
                    raise

        # Calculate metrics
        all_times = successful_times + failed_times
        success_count = len(successful_times)
        failure_count = len(failed_times)

        result = {
            "operation": operation.__name__,
            "iterations": iterations,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / iterations if iterations > 0 else 0,
            "errors": errors,
        }

        # Add timing metrics if we have successful executions
        if successful_times:
            result.update({
                "avg_time": np.mean(successful_times),
                "min_time": np.min(successful_times),
                "max_time": np.max(successful_times),
                "std_time": np.std(successful_times),
                "total_successful_time": np.sum(successful_times),
            })

        # Add failure timing metrics if we have failures
        if failed_times:
            result.update({
                "avg_failure_time": np.mean(failed_times),
                "min_failure_time": np.min(failed_times),
                "max_failure_time": np.max(failed_times),
                "std_failure_time": np.std(failed_times),
                "total_failed_time": np.sum(failed_times),
            })

        # Overall timing metrics
        if all_times:
            result.update({
                "overall_avg_time": np.mean(all_times),
                "overall_total_time": np.sum(all_times),
            })

        return result

    def _calculate_aggregate_metrics(
        self, operation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate aggregate metrics across all operations."""
        total_operations = sum(op["iterations"] for op in operation_results)
        total_successful = sum(op["success_count"] for op in operation_results)
        total_failed = sum(op["failure_count"] for op in operation_results)

        # Use overall total time if available, otherwise sum successful time
        total_time = sum(
            op.get("overall_total_time", op.get("total_successful_time", 0))
            for op in operation_results
        )

        return {
            "total_operations": total_operations,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "overall_success_rate": total_successful / total_operations if total_operations > 0 else 0,
            "total_time": total_time,
            "avg_operation_time": total_time / total_operations if total_operations > 0 else 0,
            "operations_per_second": total_operations / total_time if total_time > 0 else 0,
        }

    def generate_test_report(self, output_file: str = "integration_test_report.md"):
        """Generate comprehensive test report."""
        report = "# Integration Test Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Summary
        total_scenarios = len(self.test_results)
        successful_scenarios = sum(1 for r in self.test_results if not r.get("errors"))
        error_rate = (
            (total_scenarios - successful_scenarios) / total_scenarios
            if total_scenarios > 0
            else 0
        )

        report += "## Summary\n\n"
        report += f"- **Total Scenarios:** {total_scenarios}\n"
        report += f"- **Successful:** {successful_scenarios}\n"
        report += f"- **Failed:** {total_scenarios - successful_scenarios}\n"
        report += f"- **Success Rate:** {(1 - error_rate) * 100:.1f}%\n\n"

        # Scenario Details
        report += "## Scenario Results\n\n"
        for result in self.test_results:
            report += f"### {result['scenario']}\n\n"
            report += f"- **Duration:** {result['duration']} minutes\n"
            report += f"- **Execution Time:** {result['execution_time']:.2f} seconds\n"
            report += f"- **Trades:** {len(result.get('trades', []))}\n"

            perf = result.get("performance", {})
            if perf:
                report += f"- **Win Rate:** {perf.get('win_rate', 0) * 100:.1f}%\n"
                report += f"- **Total Profit:** ${perf.get('total_profit', 0):.2f}\n"
                report += f"- **Sharpe Ratio:** {perf.get('sharpe_ratio', 0):.2f}\n"

            if result.get("errors"):
                report += f"- **Errors:** {len(result['errors'])}\n"
                for error in result["errors"][:3]:  # Show first 3 errors
                    report += f"  - {error}\n"

            report += "\n"

        # Performance Benchmarks
        if self.performance_metrics:
            report += "## Performance Benchmarks\n\n"
            for benchmark_name, benchmark in self.performance_metrics.items():
                report += f"### {benchmark_name}\n\n"
                agg = benchmark.get("aggregate_metrics", {})
                report += f"- **Total Operations:** {agg.get('total_operations', 0)}\n"
                report += f"- **Total Time:** {agg.get('total_time', 0):.2f}s\n"
                report += f"- **Operations/Second:** {agg.get('operations_per_second', 0):.0f}\n\n"

                report += "#### Operation Details\n\n"
                for op in benchmark.get("operation_results", []):
                    report += (
                        f"- **{op['operation']}:** {op['avg_time']*1000:.2f}ms avg\n"
                    )

                report += "\n"

        with open(output_file, "w") as f:
            f.write(report)

        logger.info(f"Integration test report generated: {output_file}")


# Test Scenarios
async def scenario_basic_trading():
    """Basic trading scenario for integration testing."""
    framework = IntegrationTestFramework()
    await framework.setup_test_environment()

    async def simple_strategy():
        trades = []
        # Simple buy/sell logic
        return trades

    return await framework.run_trading_scenario("basic_trading", simple_strategy, 30)


async def scenario_high_volatility():
    """High volatility trading scenario."""
    framework = IntegrationTestFramework()
    config = await framework.setup_test_environment()

    # Set high volatility conditions
    for symbol in framework.mock_exchange.symbols:
        framework.market_simulator.set_market_condition(
            symbol, "volatile", "high", "sideways"
        )

    async def volatility_strategy():
        trades = []
        # Strategy that handles high volatility
        return trades

    return await framework.run_trading_scenario(
        "high_volatility", volatility_strategy, 45
    )


async def scenario_stress_test():
    """Stress test scenario."""
    framework = IntegrationTestFramework()
    await framework.setup_test_environment()

    stress_config = {
        "condition": "flash_crash",
        "volatility": "extreme",
        "trend": "bearish",
        "duration": 20,
    }

    return await framework.run_stress_test("market_crash", stress_config)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Run basic integration test
        results = await scenario_basic_trading()
        print(
            f"Basic trading scenario completed: {len(results.get('trades', []))} trades"
        )

        # Run stress test
        stress_results = await scenario_stress_test()
        print(f"Stress test completed: {stress_results.get('performance', {})}")

    asyncio.run(main())
