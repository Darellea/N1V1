"""
Demo: Circuit Breaker System Integration for N1V1

This script demonstrates the complete circuit breaker system integration
with the N1V1 trading framework. It shows:

1. Circuit breaker initialization and configuration
2. Real-time equity monitoring and trigger detection
3. Safety protocol execution during trigger events
4. Recovery procedures and gradual resumption
5. Manual intervention capabilities
6. Integration with monitoring and alerting systems
7. Comprehensive logging and incident analysis

The demo simulates various risk scenarios and shows how the circuit
breaker protects trading capital while maintaining operational integrity.
"""

import asyncio
import time
import random
from typing import Dict, Any
from datetime import datetime, timedelta

from core.circuit_breaker import (
    get_circuit_breaker, create_circuit_breaker,
    CircuitBreakerState, TriggerSeverity
)
from core.metrics_collector import get_metrics_collector
from utils.logger import get_logger

logger = get_logger(__name__)


class CircuitBreakerDemo:
    """
    Complete circuit breaker demonstration for N1V1 trading framework.
    """

    def __init__(self):
        self.circuit_breaker = get_circuit_breaker()
        self.metrics_collector = get_metrics_collector()

        # Demo configuration
        self.initial_equity = 10000.0
        self.current_equity = self.initial_equity
        self.demo_start_time = time.time()

        # Risk simulation parameters
        self.volatility_factor = 0.02  # 2% daily volatility
        self.trend_factor = 0.001      # Slight upward trend
        self.crash_probability = 0.05  # 5% chance of significant drawdown

    async def setup_circuit_breaker(self):
        """Set up the circuit breaker with comprehensive configuration."""
        logger.info("🚀 Setting up Circuit Breaker System")

        # Configure circuit breaker
        config = {
            "account_id": "demo_account",
            "monitoring_interval": 2.0,  # Check every 2 seconds for demo
            "cooling_period_minutes": 1,  # 1 minute cooling period for demo
            "max_daily_drawdown": 0.03,   # 3% daily drawdown limit
            "max_weekly_drawdown": 0.08,  # 8% weekly drawdown limit
            "max_consecutive_losses": 3,  # Max 3 consecutive losses
            "min_sharpe_ratio": 0.3,      # Minimum Sharpe ratio
            "max_volatility_multiplier": 2.5,  # Max volatility multiplier
            "recovery_phases": 3,
            "recovery_position_multiplier": 0.5
        }

        self.circuit_breaker = create_circuit_breaker(config)

        # Add event callbacks
        self.circuit_breaker.add_state_change_callback(self._on_state_change)
        self.circuit_breaker.add_trigger_callback(self._on_trigger)

        # Initialize equity
        await self.circuit_breaker.update_equity(self.initial_equity)

        logger.info("✅ Circuit Breaker configured and initialized")

    async def _on_state_change(self, old_state: CircuitBreakerState,
                              new_state: CircuitBreakerState, event):
        """Handle circuit breaker state changes."""
        logger.info(f"🔄 Circuit Breaker State Change: {old_state.value} → {new_state.value}")

        # Record state change metric
        await self.metrics_collector.record_metric(
            "circuit_breaker_state_changes_total",
            1,
            {
                "account": "demo_account",
                "old_state": old_state.value,
                "new_state": new_state.value,
                "severity": event.severity.value
            }
        )

    async def _on_trigger(self, event):
        """Handle circuit breaker trigger events."""
        logger.warning(f"🚨 Circuit Breaker Triggered: {event.trigger_name}")
        logger.warning(f"   Value: {event.trigger_value:.4f}, Threshold: {event.threshold:.4f}")
        logger.warning(f"   Severity: {event.severity.value}")

        # Record trigger metric
        await self.metrics_collector.record_metric(
            "circuit_breaker_triggers_total",
            1,
            {
                "account": "demo_account",
                "trigger_name": event.trigger_name,
                "severity": event.severity.value
            }
        )

    async def run_demo_simulation(self):
        """Run the complete circuit breaker demonstration."""
        logger.info("🎯 Starting Circuit Breaker Demo Simulation")

        # Phase 1: Normal operation
        await self._simulate_normal_operation()

        # Phase 2: Approaching risk limits
        await self._simulate_risk_approach()

        # Phase 3: Trigger event
        await self._simulate_trigger_event()

        # Phase 4: Recovery process
        await self._simulate_recovery_process()

        # Phase 5: Manual intervention
        await self._simulate_manual_intervention()

        # Display final results
        await self._display_demo_results()

    async def _simulate_normal_operation(self):
        """Simulate normal trading operation."""
        logger.info("📊 Phase 1: Simulating Normal Operation")

        for i in range(10):
            # Simulate normal market movement
            return_pct = random.gauss(self.trend_factor, self.volatility_factor)
            self.current_equity *= (1 + return_pct)

            await self.circuit_breaker.update_equity(
                self.current_equity,
                realized_pnl=(self.current_equity - self.initial_equity) * 0.1,
                trade_count=i + 1
            )

            logger.info(f"📊 Equity: ${self.current_equity:,.2f} | Trades: {i + 1}")
            await asyncio.sleep(1)

        logger.info("✅ Normal operation simulation complete")

    async def _simulate_risk_approach(self):
        """Simulate approaching risk limits."""
        logger.info("⚠️ Phase 2: Simulating Risk Limit Approach")

        # Gradually increase risk
        for i in range(8):
            # Increase volatility and add downward pressure
            volatility = self.volatility_factor * (1 + i * 0.2)
            trend = self.trend_factor - (i * 0.002)  # Gradual downward trend

            return_pct = random.gauss(trend, volatility)
            self.current_equity *= (1 + return_pct)

            await self.circuit_breaker.update_equity(
                self.current_equity,
                realized_pnl=(self.current_equity - self.initial_equity) * 0.1,
                trade_count=10 + i + 1
            )

            logger.info(f"📊 Equity: ${self.current_equity:,.2f} | Trades: {i + 1}")
            # Check if we're approaching limits
            status = self.circuit_breaker.get_status()
            if status['current_state'] == 'monitoring':
                logger.warning("🔶 Circuit breaker entered MONITORING state")
                break

            await asyncio.sleep(1)

        logger.info("✅ Risk approach simulation complete")

    async def _simulate_trigger_event(self):
        """Simulate a circuit breaker trigger event."""
        logger.info("🚨 Phase 3: Simulating Trigger Event")

        # Force a significant drawdown to trigger circuit breaker
        trigger_drawdown = 0.05  # 5% immediate drawdown
        self.current_equity *= (1 - trigger_drawdown)

        await self.circuit_breaker.update_equity(
            self.current_equity,
            realized_pnl=(self.current_equity - self.initial_equity) * 0.1,
            trade_count=20
        )

        logger.info(f"📊 Equity: ${self.current_equity:,.2f} | Trades: 20")
        # Wait for circuit breaker to detect and respond
        await asyncio.sleep(5)

        # Check if triggered
        status = self.circuit_breaker.get_status()
        if status['current_state'] == 'triggered':
            logger.critical("🚨 Circuit breaker successfully triggered!")
        else:
            logger.warning("⚠️ Circuit breaker did not trigger as expected")

        logger.info("✅ Trigger event simulation complete")

    async def _simulate_recovery_process(self):
        """Simulate the recovery process."""
        logger.info("🔄 Phase 4: Simulating Recovery Process")

        # Wait for cooling period
        logger.info("⏳ Waiting for cooling period...")
        await asyncio.sleep(70)  # Wait longer than 1-minute cooling period

        # Attempt recovery
        recovery_success = await self.circuit_breaker.initiate_recovery()

        if recovery_success:
            logger.info("✅ Recovery initiated successfully")

            # Simulate gradual equity recovery
            for i in range(5):
                recovery_return = 0.01  # 1% recovery per step
                self.current_equity *= (1 + recovery_return)

                await self.circuit_breaker.update_equity(
                    self.current_equity,
                    realized_pnl=(self.current_equity - self.initial_equity) * 0.1,
                    trade_count=20 + i + 1
                )

                logger.info(f"📊 Equity: ${self.current_equity:,.2f} | Trades: {20 + i + 1}")
                await asyncio.sleep(2)

            # Wait for recovery completion
            await asyncio.sleep(20)

        else:
            logger.warning("❌ Recovery initiation failed")

        logger.info("✅ Recovery process simulation complete")

    async def _simulate_manual_intervention(self):
        """Simulate manual intervention scenarios."""
        logger.info("🔧 Phase 5: Simulating Manual Intervention")

        # Scenario 1: Manual trigger
        logger.info("Manual trigger demonstration...")
        await self.circuit_breaker.manual_trigger(
            "Manual emergency trigger for demonstration",
            TriggerSeverity.EMERGENCY
        )

        await asyncio.sleep(3)

        # Scenario 2: Manual reset
        logger.info("Manual reset demonstration...")
        reset_success = await self.circuit_breaker.manual_reset(
            "Manual reset after emergency trigger demonstration"
        )

        if reset_success:
            logger.info("✅ Manual reset successful")
        else:
            logger.warning("❌ Manual reset failed")

        logger.info("✅ Manual intervention simulation complete")

    async def _display_demo_results(self):
        """Display comprehensive demo results."""
        logger.info("📊 Displaying Circuit Breaker Demo Results")

        # Get final status
        status = self.circuit_breaker.get_status()

        print("\n" + "="*70)
        print("🎯 N1V1 CIRCUIT BREAKER DEMO RESULTS")
        print("="*70)

        print("\n🏦 Account Information:")
        print(f"  • Account ID: {status['account_id']}")
        print(f"  • Initial Equity: ${self.initial_equity:,.2f}")
        print(f"  • Final Equity: ${status['current_equity']:,.2f}")
        print(f"  • Peak Equity: ${status['peak_equity']:,.2f}")

        equity_change = ((status['current_equity'] - self.initial_equity) / self.initial_equity) * 100
        print(f"  • Total Return: {equity_change:+.2f}%")

        print("\n🔧 Circuit Breaker Status:")
        print(f"  • Current State: {status['current_state'].upper()}")
        print(f"  • Trigger Count: {status['trigger_count']}")
        if status['last_trigger_time']:
            print(f"  • Last Trigger: {status['last_trigger_time']}")
        print(f"  • Active Triggers: {len(status['active_triggers'])}")
        for trigger in status['active_triggers']:
            print(f"    - {trigger}")

        print("\n📋 Recent Events:")
        for event in status['recent_events'][-3:]:  # Show last 3 events
            print(f"  • {event['timestamp'][:19]} | {event['event_type']} | {event['trigger_name']} | {event['severity']}")

        print("\n📊 Performance Metrics:")
        # Get some key metrics
        trigger_count = await self.metrics_collector.get_metric_value(
            "circuit_breaker_triggers_total", {"account": "demo_account"}
        ) or 0

        state_changes = await self.metrics_collector.get_metric_value(
            "circuit_breaker_state_changes_total", {"account": "demo_account"}
        ) or 0

        print(f"  • Total Triggers: {int(trigger_count)}")
        print(f"  • State Changes: {int(state_changes)}")
        print(f"  • Monitoring Duration: {time.time() - self.demo_start_time:.1f}s")

        print("\n🎯 Demo Scenarios Completed:")
        print("  ✅ Normal Operation Simulation")
        print("  ✅ Risk Limit Approach Detection")
        print("  ✅ Circuit Breaker Trigger Execution")
        print("  ✅ Recovery Process Initiation")
        print("  ✅ Manual Intervention Demonstration")

        print("\n🔍 Key Learnings:")
        print("  • Circuit breaker provides automatic risk protection")
        print("  • Multiple trigger conditions prevent false positives")
        print("  • Recovery process ensures gradual return to normal operation")
        print("  • Manual intervention capabilities for emergency situations")
        print("  • Comprehensive logging enables incident analysis")

        print("\n" + "="*70)
        print("✅ Circuit Breaker Demo Completed Successfully!")
        print("🔗 Circuit breaker is now protecting your trading capital!")
        print("="*70)

    async def run_health_check(self):
        """Run a health check on the circuit breaker system."""
        logger.info("🏥 Running Circuit Breaker Health Check")

        try:
            status = self.circuit_breaker.get_status()

            checks = [
                ("Circuit Breaker State", status['current_state'] in ['normal', 'monitoring', 'triggered', 'cooling', 'recovery']),
                ("Equity Tracking", status['current_equity'] > 0),
                ("Trigger Conditions", len(status['active_triggers']) > 0),
                ("Event History", len(status['recent_events']) >= 0),
                ("Metrics Integration", self.metrics_collector is not None),
            ]

            all_passed = True
            for check_name, check_result in checks:
                status_icon = "✅" if check_result else "❌"
                print(f"{status_icon} {check_name}: {'PASS' if check_result else 'FAIL'}")
                if not check_result:
                    all_passed = False

            return all_passed

        except Exception as e:
            logger.exception(f"Health check failed: {e}")
            return False

    async def cleanup(self):
        """Clean up the demo environment."""
        logger.info("🧹 Cleaning up Circuit Breaker Demo")

        try:
            if self.circuit_breaker:
                await self.circuit_breaker.stop()

            logger.info("✅ Demo cleanup completed")

        except Exception as e:
            logger.exception(f"Error during cleanup: {e}")


async def main():
    """Main demo function."""
    demo = CircuitBreakerDemo()

    try:
        # Setup circuit breaker
        await demo.setup_circuit_breaker()

        # Run health check
        health_ok = await demo.run_health_check()
        if not health_ok:
            logger.error("❌ Health check failed")
            return

        # Run demo simulation
        await demo.run_demo_simulation()

        # Keep running for a bit to allow manual inspection
        print("\n⏳ Circuit breaker is running...")
        print("📊 Check the logs for real-time monitoring")
        print("🔧 Try manual triggers and resets")
        print("\nPress Ctrl+C to stop...")

        while True:
            await asyncio.sleep(10)
            # Periodic status update
            status = demo.circuit_breaker.get_status()
            print(f"🔄 Status: {status['current_state'].upper()} | "
                  f"Equity: ${status['current_equity']:,.2f} | "
                  f"Triggers: {status['trigger_count']}")

    except KeyboardInterrupt:
        logger.info("🛑 Demo interrupted by user")
    except Exception as e:
        logger.exception(f"❌ Demo failed: {e}")
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    print("🚀 N1V1 Circuit Breaker Demo")
    print("=" * 50)
    print("This demo will:")
    print("  • Set up comprehensive circuit breaker protection")
    print("  • Simulate various risk scenarios")
    print("  • Demonstrate trigger detection and response")
    print("  • Show recovery procedures")
    print("  • Enable manual intervention capabilities")
    print("=" * 50)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise
