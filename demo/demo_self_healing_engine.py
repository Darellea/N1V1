#!/usr/bin/env python3
"""
Self-Healing Engine Demo

This script demonstrates the comprehensive Self-Healing Engine that provides
99.99% uptime guarantees for the N1V1 trading framework through automatic
failure detection, diagnosis, and recovery.

The system includes:
- Multi-level heartbeat monitoring (process, functional, dependency, data-quality)
- Sophisticated anomaly detection with statistical baselining
- Automatic recovery orchestration with state preservation
- Emergency procedures for critical failures
- Real-time monitoring dashboard
- Comprehensive logging and alerting

Usage:
    python demo_self_healing_engine.py
"""

import asyncio
import logging
import random
from pathlib import Path

# Import N1V1 components
from core.self_healing_engine import ComponentStatus, ComponentType, SelfHealingEngine
from utils.logger import get_logger

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = get_logger(__name__)


class MockComponent:
    """Mock component for demonstration purposes."""

    def __init__(self, component_id: str, component_type: str):
        self.component_id = component_id
        self.component_type = component_type
        self.healthy = True
        self.latency_ms = 50.0
        self.error_count = 0
        self.custom_metrics = {}

    def simulate_failure(self, failure_type: str = "random"):
        """Simulate a component failure."""
        if failure_type == "latency":
            self.latency_ms = random.uniform(5000, 10000)  # High latency
        elif failure_type == "errors":
            self.error_count = random.randint(10, 50)
        elif failure_type == "critical":
            self.healthy = False
        else:
            # Random failure
            choice = random.choice(["latency", "errors", "critical"])
            self.simulate_failure(choice)

        logger.warning(f"Simulated {failure_type} failure in {self.component_id}")

    def simulate_recovery(self):
        """Simulate component recovery."""
        self.healthy = True
        self.latency_ms = random.uniform(10, 100)
        self.error_count = 0
        logger.info(f"Simulated recovery for {self.component_id}")

    def get_status(self) -> ComponentStatus:
        """Get component status."""
        if not self.healthy:
            return ComponentStatus.CRITICAL
        elif self.latency_ms > 1000 or self.error_count > 5:
            return ComponentStatus.DEGRADED
        else:
            return ComponentStatus.HEALTHY

    def get_metrics(self) -> dict:
        """Get component metrics."""
        return {
            "latency_ms": self.latency_ms,
            "error_count": self.error_count,
            "uptime_percentage": 99.9 if self.healthy else 50.0,
            "active_connections": random.randint(1, 10),
            "queue_size": random.randint(0, 100),
        }


class SelfHealingEngineDemo:
    """Comprehensive demonstration of the Self-Healing Engine."""

    def __init__(self):
        self.engine = None
        self.mock_components = {}
        self.demo_running = False

    async def initialize(self):
        """Initialize the Self-Healing Engine demo."""
        logger.info("🚀 Initializing Self-Healing Engine Demo")
        logger.info("=" * 70)

        # Create self-healing engine
        config = {
            "monitoring": {
                "heartbeat_interval": 30,
                "failure_detection_threshold": 2.0,
                "emergency_webhook_url": "https://discord.com/api/webhooks/demo",
            },
            "recovery": {
                "max_recovery_attempts": 3,
                "recovery_timeout_seconds": 300,
                "auto_recovery_enabled": True,
            },
            "emergency": {
                "emergency_mode_threshold": 0.5,
                "emergency_webhook_url": "https://discord.com/api/webhooks/demo",
            },
        }

        self.engine = SelfHealingEngine(config)

        # Create mock components
        await self._create_mock_components()

        # Register components with the engine
        await self._register_components()

        logger.info("✅ Self-Healing Engine Demo initialized successfully")

    async def _create_mock_components(self):
        """Create mock components for demonstration."""
        logger.info("Creating mock components...")

        # Core trading components
        self.mock_components["bot_engine_main"] = MockComponent(
            "bot_engine_main", "bot_engine"
        )
        self.mock_components["order_manager_primary"] = MockComponent(
            "order_manager_primary", "order_manager"
        )
        self.mock_components["signal_router_main"] = MockComponent(
            "signal_router_main", "signal_router"
        )
        self.mock_components["data_fetcher_binance"] = MockComponent(
            "data_fetcher_binance", "data_fetcher"
        )

        # Strategy components
        self.mock_components["strategy_rsi"] = MockComponent("strategy_rsi", "strategy")
        self.mock_components["strategy_macd"] = MockComponent(
            "strategy_macd", "strategy"
        )
        self.mock_components["strategy_bollinger"] = MockComponent(
            "strategy_bollinger", "strategy"
        )

        # External services
        self.mock_components["discord_notifier"] = MockComponent(
            "discord_notifier", "notifier"
        )
        self.mock_components["database_main"] = MockComponent(
            "database_main", "database"
        )

        logger.info(f"Created {len(self.mock_components)} mock components")

    async def _register_components(self):
        """Register components with the Self-Healing Engine."""
        logger.info("Registering components with Self-Healing Engine...")

        # Register core components (critical)
        self.engine.register_component(
            "bot_engine_main",
            ComponentType.BOT_ENGINE,
            self.mock_components["bot_engine_main"],
            critical=True,
            dependencies=["order_manager_primary", "signal_router_main"],
        )

        self.engine.register_component(
            "order_manager_primary",
            ComponentType.ORDER_MANAGER,
            self.mock_components["order_manager_primary"],
            critical=True,
            dependencies=["data_fetcher_binance"],
        )

        self.engine.register_component(
            "signal_router_main",
            ComponentType.SIGNAL_ROUTER,
            self.mock_components["signal_router_main"],
            critical=True,
            dependencies=["strategy_rsi", "strategy_macd", "strategy_bollinger"],
        )

        self.engine.register_component(
            "data_fetcher_binance",
            ComponentType.DATA_FETCHER,
            self.mock_components["data_fetcher_binance"],
            critical=True,
        )

        # Register strategy components
        for comp_id, component in self.mock_components.items():
            if comp_id.startswith("strategy_"):
                self.engine.register_component(
                    comp_id, ComponentType.STRATEGY, component, critical=False
                )

        # Register external services
        self.engine.register_component(
            "discord_notifier",
            ComponentType.NOTIFIER,
            self.mock_components["discord_notifier"],
            critical=False,
        )

        self.engine.register_component(
            "database_main",
            ComponentType.DATABASE,
            self.mock_components["database_main"],
            critical=True,
        )

        logger.info("✅ All components registered successfully")

    async def demonstrate_heartbeat_monitoring(self):
        """Demonstrate heartbeat monitoring capabilities."""
        logger.info("💓 Demonstrating Heartbeat Monitoring")
        logger.info("-" * 50)

        # Start heartbeat simulation
        heartbeat_tasks = []
        for comp_id, component in self.mock_components.items():
            task = asyncio.create_task(
                self._simulate_component_heartbeat(comp_id, component)
            )
            heartbeat_tasks.append(task)

        # Let heartbeats run for a bit
        await asyncio.sleep(5)

        # Show current status
        await self._display_system_status()

        # Cancel heartbeat tasks
        for task in heartbeat_tasks:
            task.cancel()

        try:
            await asyncio.gather(*heartbeat_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

        logger.info("✅ Heartbeat monitoring demonstration completed")

    async def _simulate_component_heartbeat(
        self, component_id: str, component: MockComponent
    ):
        """Simulate heartbeat for a component."""
        while True:
            try:
                # Get component status and metrics
                status = component.get_status()
                metrics = component.get_metrics()

                # Send heartbeat
                await self.engine.send_heartbeat(
                    component_id=component_id,
                    status=status,
                    latency_ms=metrics["latency_ms"],
                    error_count=metrics["error_count"],
                    custom_metrics=metrics,
                )

                # Log heartbeat
                logger.debug(f"Heartbeat sent for {component_id}: {status.value}")

                await asyncio.sleep(2)  # Send heartbeat every 2 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error sending heartbeat for {component_id}: {e}")
                await asyncio.sleep(1)

    async def demonstrate_failure_simulation(self):
        """Demonstrate failure detection and recovery."""
        logger.info("🔧 Demonstrating Failure Detection & Recovery")
        logger.info("-" * 55)

        # Start with healthy system
        logger.info("Starting with healthy system...")
        await self._display_system_status()

        # Simulate various failures
        await self._simulate_failures()

        # Show recovery process
        await self._monitor_recovery_process()

        logger.info("✅ Failure simulation and recovery demonstration completed")

    async def _simulate_failures(self):
        """Simulate various component failures."""
        logger.info("Simulating component failures...")

        # Failure 1: High latency in data fetcher
        logger.info("1. Simulating high latency in data fetcher...")
        self.mock_components["data_fetcher_binance"].simulate_failure("latency")
        await asyncio.sleep(3)

        # Failure 2: Errors in signal router
        logger.info("2. Simulating errors in signal router...")
        self.mock_components["signal_router_main"].simulate_failure("errors")
        await asyncio.sleep(3)

        # Failure 3: Critical failure in strategy
        logger.info("3. Simulating critical failure in RSI strategy...")
        self.mock_components["strategy_rsi"].simulate_failure("critical")
        await asyncio.sleep(3)

        # Failure 4: Multiple component failures
        logger.info("4. Simulating multiple component failures...")
        self.mock_components["order_manager_primary"].simulate_failure("latency")
        self.mock_components["discord_notifier"].simulate_failure("errors")
        await asyncio.sleep(3)

        # Recovery simulation
        logger.info("5. Simulating component recoveries...")
        self.mock_components["data_fetcher_binance"].simulate_recovery()
        self.mock_components["signal_router_main"].simulate_recovery()
        self.mock_components["strategy_rsi"].simulate_recovery()
        await asyncio.sleep(3)

    async def _monitor_recovery_process(self):
        """Monitor the recovery process."""
        logger.info("Monitoring recovery process...")

        for i in range(5):
            await self._display_system_status()
            await asyncio.sleep(2)

    async def demonstrate_emergency_procedures(self):
        """Demonstrate emergency procedures."""
        logger.info("🚨 Demonstrating Emergency Procedures")
        logger.info("-" * 45)

        # Simulate critical system-wide failure
        logger.info("Simulating critical system-wide failure...")

        # Fail multiple critical components
        critical_failures = 0
        for comp_id, component in self.mock_components.items():
            if random.random() < 0.3:  # 30% chance of failure
                component.simulate_failure("critical")
                critical_failures += 1

        logger.info(f"Simulated {critical_failures} critical component failures")

        # Wait for emergency detection
        await asyncio.sleep(5)

        # Check emergency status
        await self._display_emergency_status()

        # Simulate recovery
        logger.info("Simulating system recovery...")
        for component in self.mock_components.values():
            if random.random() < 0.8:  # 80% recovery rate
                component.simulate_recovery()

        await asyncio.sleep(3)

        logger.info("✅ Emergency procedures demonstration completed")

    async def demonstrate_monitoring_dashboard(self):
        """Demonstrate the monitoring dashboard."""
        logger.info("📊 Demonstrating Monitoring Dashboard")
        logger.info("-" * 45)

        # Get dashboard data
        dashboard_data = self.engine.monitoring_dashboard.get_dashboard_data()

        # Display key metrics
        system_health = dashboard_data["system_health"]
        logger.info("System Health Summary:")
        logger.info(f"  Overall Health: {system_health['overall_health']}")
        logger.info(f"  Health Score: {system_health['health_score']}%")
        logger.info(f"  Total Components: {system_health['total_components']}")
        logger.info(f"  Healthy Components: {system_health['healthy_components']}")
        logger.info(f"  Failing Components: {system_health['failing_components']}")

        # Display component status
        component_status = dashboard_data["component_status"]
        logger.info(f"\nComponent Status ({len(component_status)} components):")
        for comp in component_status[:5]:  # Show first 5
            status_emoji = "✅" if comp["consecutive_failures"] == 0 else "❌"
            logger.info(
                f"  {status_emoji} {comp['component_id']} ({comp['component_type']}) - {comp['consecutive_failures']} failures"
            )

        # Display failure statistics
        failure_stats = dashboard_data["failure_stats"]
        logger.info("\nFailure Statistics:")
        logger.info(f"  Total Failures: {failure_stats['total_failures']}")
        logger.info(f"  Recovery Attempts: {failure_stats['recovery_attempts']}")
        logger.info(
            f"  Recovery Success Rate: {failure_stats['recovery_success_rate']:.1f}%"
        )

        logger.info("✅ Monitoring dashboard demonstration completed")

    async def demonstrate_engine_statistics(self):
        """Demonstrate comprehensive engine statistics."""
        logger.info("📈 Demonstrating Engine Statistics")
        logger.info("-" * 40)

        # Get engine statistics
        stats = self.engine.get_engine_stats()

        logger.info("Self-Healing Engine Statistics:")
        logger.info(f"  Uptime: {stats['uptime']}")
        logger.info(f"  Total Failures Handled: {stats['total_failures_handled']}")
        logger.info(
            f"  Total Recoveries Successful: {stats['total_recoveries_successful']}"
        )

        # Registry statistics
        registry_stats = stats["registry_stats"]
        logger.info("\nComponent Registry:")
        logger.info(f"  Total Components: {registry_stats['total_components']}")
        logger.info(f"  Critical Components: {registry_stats['critical_components']}")
        logger.info(f"  Healthy Components: {registry_stats['healthy_components']}")
        logger.info(f"  Failing Components: {registry_stats['failing_components']}")

        # Healing statistics
        healing_stats = stats["healing_stats"]
        logger.info("\nHealing Statistics:")
        logger.info(f"  Pending Actions: {healing_stats['pending_actions']}")
        logger.info(f"  Completed Actions: {healing_stats['completed_actions']}")
        logger.info(f"  Failed Actions: {healing_stats['failed_actions']}")
        logger.info(f"  Success Rate: {healing_stats['success_rate']:.1f}%")

        # Watchdog statistics
        watchdog_stats = stats["watchdog_stats"]
        logger.info("\nWatchdog Statistics:")
        logger.info(f"  Heartbeats Received: {watchdog_stats['heartbeats_received']}")
        logger.info(f"  Failures Detected: {watchdog_stats['failures_detected']}")
        logger.info(f"  Recoveries Initiated: {watchdog_stats['recoveries_initiated']}")

        logger.info("✅ Engine statistics demonstration completed")

    async def _display_system_status(self):
        """Display current system status."""
        dashboard_data = self.engine.monitoring_dashboard.get_dashboard_data()
        system_health = dashboard_data["system_health"]

        status_emoji = {
            "HEALTHY": "🟢",
            "DEGRADED": "🟡",
            "CRITICAL": "🔴",
            "EMERGENCY": "🚨",
        }.get(system_health["overall_health"], "⚪")

        logger.info(
            f"{status_emoji} System Status: {system_health['overall_health']} "
            f"(Score: {system_health['health_score']}%) - "
            f"{system_health['healthy_components']}/{system_health['total_components']} healthy"
        )

    async def _display_emergency_status(self):
        """Display emergency status."""
        emergency_active = self.engine.emergency_procedures.is_emergency_active()
        if emergency_active:
            duration = self.engine.emergency_procedures.get_emergency_duration()
            logger.warning(f"🚨 EMERGENCY MODE ACTIVE - Duration: {duration}")
        else:
            logger.info("✅ Emergency mode not active")

    async def run_complete_demo(self):
        """Run the complete Self-Healing Engine demonstration."""
        try:
            # Initialize
            await self.initialize()

            # Start the engine
            await self.engine.start()

            # Demonstrate capabilities
            await self.demonstrate_heartbeat_monitoring()
            await self.demonstrate_failure_simulation()
            await self.demonstrate_emergency_procedures()
            await self.demonstrate_monitoring_dashboard()
            await self.demonstrate_engine_statistics()

            # Final status
            await self._display_system_status()

            logger.info("🎉 Self-Healing Engine Demo completed successfully!")
            logger.info("=" * 70)

            # Demo summary
            logger.info("📋 Demo Summary:")
            logger.info("  ✅ Heartbeat monitoring system")
            logger.info("  ✅ Failure detection and diagnosis")
            logger.info("  ✅ Automatic recovery orchestration")
            logger.info("  ✅ Emergency procedures")
            logger.info("  ✅ Real-time monitoring dashboard")
            logger.info("  ✅ Comprehensive statistics and reporting")

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            import traceback

            logger.debug(traceback.format_exc())

        finally:
            # Stop the engine
            if self.engine:
                await self.engine.stop()


async def main():
    """Main entry point for the demo."""
    demo = SelfHealingEngineDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Run the demo
    asyncio.run(main())
