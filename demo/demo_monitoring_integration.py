"""
Demo: Complete Monitoring & Observability Integration for N1V1

This script demonstrates how to integrate the comprehensive monitoring system
with the N1V1 trading framework, including all four major features:

1. Multi-Timeframe Analysis
2. Predictive Regime Forecasting
3. Hybrid AI Strategy Generator
4. Self-Healing Engine

The demo shows:
- Real-time metrics collection from all components
- Prometheus endpoint serving metrics
- Alert generation and notification
- Dashboard data visualization
- Performance monitoring and optimization
"""

import asyncio
import time

from core.metrics_collector import (
    collect_exchange_metrics,
    collect_risk_metrics,
    collect_strategy_metrics,
    collect_trading_metrics,
    create_metrics_endpoint,
    get_metrics_collector,
)
from core.self_healing_engine import ComponentType, SelfHealingEngine
from core.timeframe_manager import TimeframeManager
from optimization.strategy_generator import StrategyGenerator
from strategies.regime.regime_forecaster import RegimeForecaster
from utils.logger import get_logger

logger = get_logger(__name__)


class MonitoringDemo:
    """
    Complete monitoring integration demo for N1V1 trading framework.
    """

    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self.endpoint = None

        # Component instances
        self.timeframe_manager = None
        self.regime_forecaster = None
        self.strategy_generator = None
        self.self_healing_engine = None

        # Demo data
        self.demo_start_time = time.time()

    async def setup_monitoring_system(self):
        """Set up the complete monitoring system."""
        logger.info("🚀 Setting up N1V1 Monitoring & Observability System")

        # 1. Initialize metrics collection
        await self._setup_metrics_collection()

        # 2. Initialize metrics endpoint
        await self._setup_metrics_endpoint()

        # 3. Initialize N1V1 components with monitoring
        await self._setup_n1v1_components()

        # 4. Register components with self-healing engine
        await self._setup_self_healing_integration()

        logger.info("✅ Monitoring system setup complete")

    async def _setup_metrics_collection(self):
        """Set up metrics collection system."""
        logger.info("📊 Setting up metrics collection...")

        # Register custom metrics collectors
        self.metrics_collector.add_custom_collector(collect_trading_metrics)
        self.metrics_collector.add_custom_collector(collect_risk_metrics)
        self.metrics_collector.add_custom_collector(collect_strategy_metrics)
        self.metrics_collector.add_custom_collector(collect_exchange_metrics)

        # Start metrics collection
        await self.metrics_collector.start()

        logger.info("✅ Metrics collection started")

    async def _setup_metrics_endpoint(self):
        """Set up Prometheus metrics endpoint."""
        logger.info("🌐 Setting up metrics endpoint...")

        endpoint_config = {
            "host": "0.0.0.0",
            "port": 9090,
            "path": "/metrics",
            "enable_auth": False,  # Disable for demo
            "max_concurrent_requests": 10,
            "request_timeout": 5.0,
        }

        self.endpoint = create_metrics_endpoint(endpoint_config)
        await self.endpoint.start()

        logger.info("✅ Metrics endpoint started on http://localhost:9090/metrics")

    async def _setup_n1v1_components(self):
        """Set up N1V1 components with monitoring integration."""
        logger.info("🔧 Setting up N1V1 components...")

        # 1. Multi-Timeframe Manager
        self.timeframe_manager = TimeframeManager({}, {})
        await self.timeframe_manager.initialize()

        # 2. Regime Forecaster
        forecaster_config = {"model_path": "./demo_models"}
        self.regime_forecaster = RegimeForecaster(forecaster_config)
        await self.regime_forecaster.initialize()

        # 3. Strategy Generator
        strategy_config = {
            "model_path": "./demo_models",
            "population_size": 5,
            "generations": 2,
        }
        self.strategy_generator = StrategyGenerator(strategy_config)
        await self.strategy_generator.initialize()

        # 4. Self-Healing Engine
        healing_config = {"monitoring": {"heartbeat_interval": 10}}
        self.self_healing_engine = SelfHealingEngine(healing_config)

        logger.info("✅ N1V1 components initialized")

    async def _setup_self_healing_integration(self):
        """Set up self-healing engine integration."""
        logger.info("🛠️ Setting up self-healing integration...")

        # Register components with self-healing engine
        components = [
            (
                "timeframe_manager",
                ComponentType.DATA_FETCHER,
                self.timeframe_manager,
                True,
            ),
            (
                "regime_forecaster",
                ComponentType.EXTERNAL_SERVICE,
                self.regime_forecaster,
                False,
            ),
            (
                "strategy_generator",
                ComponentType.EXTERNAL_SERVICE,
                self.strategy_generator,
                False,
            ),
        ]

        for comp_id, comp_type, instance, critical in components:
            self.self_healing_engine.register_component(
                comp_id, comp_type, instance, critical
            )

        # Start self-healing engine
        await self.self_healing_engine.start()

        logger.info("✅ Self-healing integration complete")

    async def run_demo_simulation(self):
        """Run the complete monitoring demo simulation."""
        logger.info("🎯 Starting monitoring demo simulation...")

        # Simulate trading operations with monitoring
        await self._simulate_trading_operations()

        # Simulate system health monitoring
        await self._simulate_system_monitoring()

        # Simulate component interactions
        await self._simulate_component_interactions()

        # Display monitoring results
        await self._display_monitoring_results()

    async def _simulate_trading_operations(self):
        """Simulate trading operations with metrics collection."""
        logger.info("📈 Simulating trading operations...")

        # Simulate order executions
        for i in range(10):
            # Record order execution metrics
            await self.metrics_collector.increment_counter(
                "trading_orders_total",
                {"account": "demo", "status": "filled", "exchange": "binance"},
            )

            # Record order latency
            latency = 0.045 + (i * 0.005)  # Increasing latency simulation
            await self.metrics_collector.record_metric(
                "trading_order_latency_seconds",
                latency,
                {"account": "demo", "exchange": "binance"},
            )

            # Record P&L
            pnl = 1250.75 + (i * 25.5)
            await self.metrics_collector.record_metric(
                "trading_total_pnl_usd", pnl, {"account": "demo"}
            )

            # Simulate some slippage
            slippage = 2.5 + (i * 0.1)
            await self.metrics_collector.record_metric(
                "trading_slippage_bps",
                slippage,
                {"account": "demo", "symbol": "BTC/USDT"},
            )

            await asyncio.sleep(0.1)

        logger.info("✅ Trading operations simulation complete")

    async def _simulate_system_monitoring(self):
        """Simulate system health monitoring."""
        logger.info("💻 Simulating system monitoring...")

        # Simulate system metrics over time
        for i in range(5):
            # CPU usage simulation
            cpu_usage = 45.0 + (i * 5.0)  # Gradual increase
            await self.metrics_collector.record_metric(
                "system_cpu_usage_percent", cpu_usage
            )

            # Memory usage simulation
            memory_usage = 4.2e9 + (i * 1e8)  # ~4.2GB + increasing
            await self.metrics_collector.record_metric(
                "system_memory_usage_bytes", memory_usage
            )

            # Network I/O simulation
            network_rx = 1e6 + (i * 5e4)  # 1MB + increasing
            network_tx = 8e5 + (i * 3e4)  # 800KB + increasing
            await self.metrics_collector.record_metric(
                "system_network_receive_bytes_total", network_rx
            )
            await self.metrics_collector.record_metric(
                "system_network_transmit_bytes_total", network_tx
            )

            await asyncio.sleep(0.2)

        logger.info("✅ System monitoring simulation complete")

    async def _simulate_component_interactions(self):
        """Simulate interactions between N1V1 components."""
        logger.info("🔄 Simulating component interactions...")

        # Simulate timeframe manager operations
        await self.self_healing_engine.send_heartbeat(
            component_id="timeframe_manager",
            status="healthy",
            latency_ms=45.0,
            error_count=0,
            custom_metrics={"symbols_registered": 5},
        )

        # Simulate regime forecaster operations
        await self.self_healing_engine.send_heartbeat(
            component_id="regime_forecaster",
            status="healthy",
            latency_ms=120.0,
            error_count=0,
            custom_metrics={"forecast_accuracy": 0.85},
        )

        # Simulate strategy generator operations
        await self.self_healing_engine.send_heartbeat(
            component_id="strategy_generator",
            status="healthy",
            latency_ms=200.0,
            error_count=0,
            custom_metrics={"strategies_generated": 3},
        )

        logger.info("✅ Component interactions simulation complete")

    async def _display_monitoring_results(self):
        """Display monitoring results and metrics."""
        logger.info("📊 Displaying monitoring results...")

        # Get current metrics
        metrics_output = self.metrics_collector.get_prometheus_output()

        # Display key metrics
        print("\n" + "=" * 60)
        print("🎯 N1V1 MONITORING DEMO RESULTS")
        print("=" * 60)

        print("\n📈 Trading Performance Metrics:")
        print(
            f"  • Total P&L: ${self.metrics_collector.get_metric_value('trading_total_pnl_usd', {'account': 'demo'}) or 0:.2f}"
        )
        print(
            f"  • Orders Executed: {self.metrics_collector.get_metric_value('trading_orders_total', {'account': 'demo', 'status': 'filled', 'exchange': 'binance'}) or 0}"
        )
        print(
            f"  • Average Latency: {self.metrics_collector.get_metric_value('trading_order_latency_seconds', {'account': 'demo', 'exchange': 'binance'}) or 0:.3f}s"
        )

        print("\n💻 System Health Metrics:")
        print(
            f"  • CPU Usage: {self.metrics_collector.get_metric_value('system_cpu_usage_percent') or 0:.1f}%"
        )
        print(
            f"  • Memory Usage: {(self.metrics_collector.get_metric_value('system_memory_usage_bytes') or 0) / 1e9:.2f} GB"
        )
        print(
            f"  • Network RX: {(self.metrics_collector.get_metric_value('system_network_receive_bytes_total') or 0) / 1e6:.1f} MB"
        )

        print("\n🔧 Component Health:")
        dashboard_data = (
            self.self_healing_engine.monitoring_dashboard.get_dashboard_data()
        )
        system_health = dashboard_data["system_health"]
        print(f"  • Components Monitored: {system_health['total_components']}")
        print(f"  • Healthy Components: {system_health['healthy_components']}")

        print("\n🌐 Metrics Endpoint:")
        print("  • URL: http://localhost:9090/metrics")
        print("  • Status: Running")
        print(f"  • Total Requests: {self.endpoint.request_count}")
        print(f"  • Average Response Time: {self.endpoint.avg_response_time:.3f}s")

        print("\n📋 Sample Prometheus Metrics Output:")
        print("-" * 40)
        # Show first few lines of metrics output
        lines = metrics_output.split("\n")[:10]
        for line in lines:
            if line.strip():
                print(f"  {line}")

        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("🔗 Access Grafana dashboards at: http://localhost:3000")
        print("📊 View Prometheus metrics at: http://localhost:9090")
        print("=" * 60)

    async def run_health_check(self):
        """Run a health check on the monitoring system."""
        logger.info("🏥 Running monitoring system health check...")

        try:
            # Check metrics endpoint
            endpoint_stats = self.endpoint.get_stats()
            print(
                f"✅ Metrics Endpoint: {endpoint_stats['running'] and 'Running' or 'Stopped'}"
            )

            # Check metrics collector
            collector_running = self.metrics_collector._running
            print(
                f"✅ Metrics Collector: {collector_running and 'Running' or 'Stopped'}"
            )

            # Check self-healing engine
            healing_running = self.self_healing_engine._running
            print(
                f"✅ Self-Healing Engine: {healing_running and 'Running' or 'Stopped'}"
            )

            # Check component registrations
            registry_stats = (
                self.self_healing_engine.component_registry.get_registry_stats()
            )
            print(f"✅ Components Registered: {registry_stats['total_components']}")

            return True

        except Exception as e:
            logger.exception(f"Health check failed: {e}")
            return False

    async def cleanup(self):
        """Clean up the monitoring system."""
        logger.info("🧹 Cleaning up monitoring system...")

        try:
            # Stop components in reverse order
            if self.self_healing_engine:
                await self.self_healing_engine.stop()

            if self.endpoint:
                await self.endpoint.stop()

            if self.metrics_collector:
                await self.metrics_collector.stop()

            logger.info("✅ Cleanup completed")

        except Exception as e:
            logger.exception(f"Error during cleanup: {e}")


async def main():
    """Main demo function."""
    demo = MonitoringDemo()

    try:
        # Setup monitoring system
        await demo.setup_monitoring_system()

        # Run health check
        health_ok = await demo.run_health_check()
        if not health_ok:
            logger.error("❌ Health check failed")
            return

        # Run demo simulation
        await demo.run_demo_simulation()

        # Keep running for a bit to allow manual inspection
        print("\n⏳ Monitoring system is running...")
        print("🌐 Metrics available at: http://localhost:9090/metrics")
        print("📊 Prometheus UI at: http://localhost:9090")
        print("🎨 Grafana dashboards at: http://localhost:3000")
        print("\nPress Ctrl+C to stop...")

        while True:
            await asyncio.sleep(10)
            # Periodic health check
            await demo.run_health_check()

    except KeyboardInterrupt:
        logger.info("🛑 Demo interrupted by user")
    except Exception as e:
        logger.exception(f"❌ Demo failed: {e}")
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    print("🚀 N1V1 Monitoring & Observability Demo")
    print("=" * 50)
    print("This demo will:")
    print("  • Set up complete monitoring infrastructure")
    print("  • Simulate trading operations with metrics")
    print("  • Demonstrate system health monitoring")
    print("  • Show component integration")
    print("  • Display real-time metrics and dashboards")
    print("=" * 50)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise
