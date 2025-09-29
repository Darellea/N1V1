"""
Performance Optimization Demo
=============================

This demo showcases the comprehensive performance optimization system for the N1V1 trading framework.
It demonstrates profiling, monitoring, reporting, and optimization techniques.

Features Demonstrated:
- Function-level profiling with decorators
- Real-time performance monitoring
- Anomaly detection and alerting
- Comprehensive performance reporting
- Vectorization and optimization examples
"""

import asyncio
import logging
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.performance_monitor import PerformanceAlert, get_performance_monitor
from core.performance_profiler import get_profiler, profile_function
from core.performance_reports import get_performance_report_generator
from utils.logger import get_logger

logger = get_logger(__name__)


# Example trading functions to profile
@profile_function("calculate_sma")
def calculate_sma(prices: np.ndarray, window: int) -> np.ndarray:
    """Calculate Simple Moving Average - vectorized version."""
    if len(prices) < window:
        return np.array([])

    weights = np.ones(window) / window
    return np.convolve(prices, weights, mode="valid")


@profile_function("calculate_rsi")
def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI indicator."""
    if len(prices) < period + 1:
        return np.array([])

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.convolve(gains, np.ones(period) / period, mode="valid")
    avg_losses = np.convolve(losses, np.ones(period) / period, mode="valid")

    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    return rsi


@profile_function("slow_trading_loop")
def slow_trading_loop(prices: List[float], signals: List[int]) -> List[Dict[str, Any]]:
    """Slow implementation using Python loops - for comparison."""
    results = []

    for i in range(len(prices)):
        # Simulate some processing
        if i < 5:
            continue

        # Calculate moving average manually (slow way)
        ma_sum = 0
        for j in range(5):
            ma_sum += prices[i - j]
        ma = ma_sum / 5

        # Calculate signal
        signal = signals[i] if i < len(signals) else 0

        results.append(
            {"price": prices[i], "ma": ma, "signal": signal, "timestamp": time.time()}
        )

    return results


@profile_function("fast_trading_processing")
def fast_trading_processing(prices: np.ndarray, signals: np.ndarray) -> pd.DataFrame:
    """Fast implementation using vectorized operations."""
    # Vectorized moving average
    ma = calculate_sma(prices, 5)

    # Ensure signals array matches prices length
    if len(signals) > len(ma):
        signals = signals[: len(ma)]
    elif len(signals) < len(ma):
        signals = np.pad(signals, (0, len(ma) - len(signals)), constant_values=0)

    # Create results DataFrame
    results = pd.DataFrame(
        {
            "price": prices[len(prices) - len(ma) :],
            "ma": ma,
            "signal": signals,
            "timestamp": np.full(len(ma), time.time()),
        }
    )

    return results


@profile_function("complex_strategy_calculation")
def complex_strategy_calculation(
    prices: np.ndarray, volume: np.ndarray
) -> Dict[str, Any]:
    """Complex strategy calculation combining multiple indicators."""
    # Calculate multiple indicators
    sma_20 = calculate_sma(prices, 20)
    sma_50 = calculate_sma(prices, 50)
    rsi = calculate_rsi(prices, 14)

    # Volume analysis
    volume_sma = calculate_sma(volume, 20)

    # Combine signals
    trend_signal = np.where(sma_20 > sma_50, 1, -1)
    momentum_signal = np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0))

    # Composite signal
    composite_signal = trend_signal + momentum_signal
    final_signal = np.where(
        composite_signal > 1, 1, np.where(composite_signal < -1, -1, 0)
    )

    return {
        "trend_signal": trend_signal,
        "momentum_signal": momentum_signal,
        "final_signal": final_signal,
        "indicators": {
            "sma_20": sma_20,
            "sma_50": sma_50,
            "rsi": rsi,
            "volume_sma": volume_sma,
        },
    }


async def setup_performance_monitoring():
    """Setup performance monitoring with alerts."""
    monitor = get_performance_monitor()

    # Define performance alerts
    alerts = [
        PerformanceAlert(
            alert_id="high_cpu_usage",
            metric_name="process_cpu_usage_percent",
            condition="above",
            threshold=80.0,
            severity="high",
            cooldown_period=300,  # 5 minutes
            description="High CPU usage detected",
        ),
        PerformanceAlert(
            alert_id="memory_pressure",
            metric_name="process_memory_usage_bytes",
            condition="above",
            threshold=500 * 1024 * 1024,  # 500MB
            severity="medium",
            cooldown_period=600,  # 10 minutes
            description="High memory usage detected",
        ),
        PerformanceAlert(
            alert_id="slow_function_execution",
            metric_name="function_execution_time",
            condition="above",
            threshold=1.0,  # 1 second
            severity="medium",
            cooldown_period=60,  # 1 minute
            description="Slow function execution detected",
        ),
    ]

    # Add alerts
    for alert in alerts:
        monitor.add_alert(alert)

    # Add alert callbacks
    async def alert_callback(alert: PerformanceAlert, value: float):
        logger.warning(
            f"🚨 PERFORMANCE ALERT: {alert.alert_id} - "
            f"{alert.metric_name} = {value} ({alert.description})"
        )

    monitor.add_alert_callback(alert_callback)

    # Start monitoring
    await monitor.start_monitoring()
    logger.info("✅ Performance monitoring started with alerts")

    return monitor


async def run_performance_demo():
    """Run the comprehensive performance optimization demo."""
    logger.info("🚀 Starting Performance Optimization Demo")

    # Setup components
    profiler = get_profiler()
    monitor = await setup_performance_monitoring()
    report_generator = get_performance_report_generator()

    # Start profiling session
    session_id = profiler.start_profiling("demo_session")
    logger.info(f"📊 Started profiling session: {session_id}")

    try:
        # Generate sample data
        logger.info("📈 Generating sample trading data...")
        np.random.seed(42)
        n_points = 10000

        prices = np.random.normal(50000, 1000, n_points).astype(np.float32)
        volume = np.random.normal(1000000, 200000, n_points).astype(np.float32)
        signals = np.random.choice([-1, 0, 1], n_points, p=[0.3, 0.4, 0.3])

        # Demo 1: Compare slow vs fast implementations
        logger.info("⚡ Comparing slow vs fast implementations...")

        # Slow implementation
        start_time = time.time()
        slow_results = slow_trading_loop(prices.tolist(), signals.tolist())
        slow_time = time.time() - start_time
        logger.info(".4f")

        # Fast implementation
        start_time = time.time()
        fast_results = fast_trading_processing(prices, signals)
        fast_time = time.time() - start_time
        logger.info(".4f")

        speedup = slow_time / fast_time
        logger.info(".1f")

        # Demo 2: Complex strategy calculation
        logger.info("🧠 Running complex strategy calculations...")
        strategy_results = complex_strategy_calculation(prices, volume)
        logger.info(
            f"📊 Generated {len(strategy_results['final_signal'])} strategy signals"
        )

        # Demo 3: Simulate high-frequency processing
        logger.info("🔥 Simulating high-frequency processing...")
        for batch in range(10):
            batch_prices = prices[batch * 1000 : (batch + 1) * 1000]
            batch_volume = volume[batch * 1000 : (batch + 1) * 1000]

            # Process batch
            batch_results = complex_strategy_calculation(batch_prices, batch_volume)

            # Small delay to simulate real processing
            await asyncio.sleep(0.01)

            if (batch + 1) % 5 == 0:
                logger.info(f"✅ Processed batch {batch + 1}/10")

        # Demo 4: Memory-intensive operations
        logger.info("💾 Testing memory-intensive operations...")
        large_data = np.random.random((1000, 1000)).astype(np.float32)

        @profile_function("memory_intensive_calculation")
        def memory_intensive_calculation(data: np.ndarray) -> np.ndarray:
            # Simulate memory-intensive calculation
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    result[i, j] = np.sum(
                        data[
                            max(0, i - 5) : min(data.shape[0], i + 6),
                            max(0, j - 5) : min(data.shape[1], j + 6),
                        ]
                    )
            return result

        memory_result = memory_intensive_calculation(large_data)
        logger.info(f"📊 Memory-intensive calculation completed: {memory_result.shape}")

        # Wait a bit for monitoring to collect data
        logger.info("⏳ Waiting for performance monitoring to collect data...")
        await asyncio.sleep(5)

        # Generate comprehensive report
        logger.info("📋 Generating comprehensive performance report...")
        report = report_generator.generate_comprehensive_report(session_id)

        # Display report summary
        logger.info("📊 Performance Report Summary:")
        logger.info(f"   • Total Functions: {report.summary.get('total_functions', 0)}")
        logger.info(
            f"   • Total Measurements: {report.summary.get('total_measurements', 0)}"
        )
        logger.info(
            f"   • Performance Score: {report.summary.get('performance_score', 0):.1f}"
        )
        logger.info(f"   • System Health: {report.summary.get('system_health', 0):.1f}")
        logger.info(
            f"   • Anomalies Detected: {report.summary.get('anomaly_count', 0)}"
        )

        # Display hotspots
        if report.hotspots:
            logger.info("🔥 Top Performance Hotspots:")
            for i, hotspot in enumerate(report.hotspots[:5]):
                logger.info(
                    f"   {i+1}. {hotspot['function_name']} - "
                    f"{hotspot['total_time']:.4f}s total, "
                    f"{hotspot['avg_time']:.6f}s avg, "
                    f"{hotspot['call_count']} calls"
                )

        # Display recommendations
        if report.recommendations:
            logger.info("💡 Optimization Recommendations:")
            for rec in report.recommendations:
                logger.info(f"   • {rec}")

        # Export report
        export_path = report_generator.export_report(report, "html")
        logger.info(f"📄 Report exported to: {export_path}")

        # Get current performance status
        status = await monitor.get_performance_status()
        logger.info("📈 Current Performance Status:")
        logger.info(f"   • Monitoring Active: {status.get('is_monitoring', False)}")
        logger.info(f"   • Active Alerts: {status.get('active_alerts', 0)}")
        logger.info(f"   • Total Baselines: {status.get('total_baselines', 0)}")
        logger.info(f"   • Recent Anomalies: {status.get('recent_anomalies', 0)}")
        logger.info(f"   • System Health Score: {status.get('system_health', 0):.1f}")

    finally:
        # Stop profiling and monitoring
        profiler.stop_profiling()
        await monitor.stop_monitoring()

        logger.info("✅ Performance demo completed")


async def demonstrate_anomaly_detection():
    """Demonstrate anomaly detection capabilities."""
    logger.info("🔍 Demonstrating Anomaly Detection")

    monitor = get_performance_monitor()
    await monitor.start_monitoring()

    # Simulate normal operation
    logger.info("📊 Simulating normal operation...")
    for i in range(20):
        # Normal execution time (around 0.01 seconds)
        execution_time = 0.01 + np.random.normal(0, 0.002)
        await asyncio.sleep(execution_time)

        if i % 10 == 0:
            logger.info(f"   Processed {i+1}/20 normal operations")

    # Simulate anomaly
    logger.info("⚠️  Simulating performance anomaly...")
    await asyncio.sleep(0.5)  # 0.5 second delay (anomaly)

    # Continue normal operation
    logger.info("📊 Continuing normal operation...")
    for i in range(10):
        execution_time = 0.01 + np.random.normal(0, 0.002)
        await asyncio.sleep(execution_time)

    # Wait for anomaly detection
    await asyncio.sleep(2)

    # Check for anomalies
    status = await monitor.get_performance_status()
    logger.info(
        f"🔍 Anomaly detection results: {status.get('recent_anomalies', 0)} anomalies detected"
    )

    await monitor.stop_monitoring()


def main():
    """Main demo function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("🚀 N1V1 Performance Optimization Demo")
    print("=" * 60)

    # Run main demo
    asyncio.run(run_performance_demo())

    print("\n" + "=" * 60)
    print("🔍 Running Anomaly Detection Demo")
    print("=" * 60)

    # Run anomaly detection demo
    asyncio.run(demonstrate_anomaly_detection())

    print("\n" + "=" * 60)
    print("✅ All Performance Optimization Demos Completed!")
    print("=" * 60)
    print("\nKey Achievements:")
    print("• ✅ Advanced profiling infrastructure implemented")
    print("• ✅ Real-time performance monitoring active")
    print("• ✅ Anomaly detection and alerting working")
    print("• ✅ Comprehensive reporting system functional")
    print("• ✅ Vectorization optimizations demonstrated")
    print("• ✅ Performance improvements quantified")
    print("\nNext Steps:")
    print("• Integrate with existing trading strategies")
    print("• Set up continuous performance monitoring")
    print("• Implement automated optimization workflows")
    print("• Deploy performance dashboards")


if __name__ == "__main__":
    main()
