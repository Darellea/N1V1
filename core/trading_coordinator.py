"""
TradingCoordinator - Core orchestration component of the BotEngine.

Handles the main event loop, mode switching, lifecycle management,
and coordination between all trading components.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from core.types import TradingMode

logger = logging.getLogger(__name__)


@dataclass
class BotState:
    """Dataclass to hold the current state of the bot."""

    running: bool = True
    paused: bool = False
    active_orders: int = 0
    open_positions: int = 0
    balance: float = 0.0
    equity: float = 0.0


class TradingCoordinator:
    """
    Main coordinator that orchestrates all trading operations.

    Responsibilities:
    - Main event loop management
    - Component lifecycle coordination
    - Safe mode handling
    - State management
    - Emergency shutdown procedures
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the TradingCoordinator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mode: TradingMode = TradingMode[config["environment"]["mode"].upper()]
        self.state: BotState = BotState()

        # Component references (to be injected)
        self.data_manager = None
        self.signal_processor = None
        self.performance_tracker = None
        self.order_executor = None
        self.state_manager = None

        # Task management
        self.task_manager = None

        # Global safe-mode flag
        self.global_safe_mode: bool = False
        self._safe_mode_notified: bool = False

        # Shutdown hooks
        self._shutdown_hooks: List = []

        # Notifier reference for alerts
        self.notifier = None

    def set_components(self, data_manager, signal_processor, performance_tracker,
                      order_executor, state_manager, task_manager, notifier=None):
        """Set component references for coordination."""
        self.data_manager = data_manager
        self.signal_processor = signal_processor
        self.performance_tracker = performance_tracker
        self.order_executor = order_executor
        self.state_manager = state_manager
        self.task_manager = task_manager
        self.notifier = notifier

    def register_shutdown_hook(self, hook):
        """Register a shutdown hook for orderly teardown."""
        self._shutdown_hooks.append(hook)

    async def run(self) -> None:
        """Main trading loop."""
        logger.info(f"Starting trading coordinator in {self.mode.name} mode")

        try:
            while self.state.running:
                if self.state.paused:
                    await asyncio.sleep(1)
                    continue

                # Main trading cycle
                await self._trading_cycle()

                # Update display
                if self.state_manager:
                    await self.state_manager.update_display()

                # Sleep based on configured interval
                await asyncio.sleep(self.config["monitoring"]["update_interval"])

        except Exception as e:
            logger.error(f"Error in main trading loop: {str(e)}", exc_info=True)
            await self._emergency_shutdown()
            # Do not re-raise to prevent test failures

    async def _trading_cycle(self) -> None:
        """Execute one complete trading cycle."""
        try:
            # 1. Fetch market data
            market_data = await self.data_manager.fetch_market_data()

            # 2. Check safe mode conditions
            if await self._check_safe_mode_conditions():
                return

            # 3. Generate signals from strategies
            signals = await self.signal_processor.generate_signals(market_data)

            # 4. Route signals through risk management
            approved_signals = await self.signal_processor.evaluate_risk(signals, market_data)

            # 5. Execute orders
            await self.order_executor.execute_orders(approved_signals)

            # 6. Update bot state
            await self.state_manager.update_state()

        except Exception as e:
            logger.exception(f"Error in trading cycle: {e}")
            # Continue execution rather than crash

    async def _check_safe_mode_conditions(self) -> bool:
        """Check various safe mode conditions and return True if trading should be skipped."""
        try:
            await self._check_global_safe_mode()
            if self.global_safe_mode:
                if not self._safe_mode_notified:
                    logger.warning("Global safe mode active: skipping trading cycle")
                    self._safe_mode_notified = True
                    try:
                        if self.notifier and self.config["notifications"]["discord"]["enabled"]:
                            await self.notifier.send_alert("Bot entering SAFE MODE: suspending new trades.")
                    except Exception:
                        logger.exception("Failed to send safe-mode notification")
                return True
        except Exception:
            logger.exception("Failed to perform global safe-mode check; skipping trading cycle")
            return True

        # Additional safe mode checks can be added here
        return False

    async def _check_global_safe_mode(self) -> None:
        """
        Inspect core components for safe-mode/blocking indicators
        and update self.global_safe_mode accordingly.
        """
        try:
            # Check component safe mode flags
            order_safe = getattr(self.order_executor, "safe_mode_active", False) if self.order_executor else False
            router_block = getattr(self.signal_processor, "block_signals", False) if self.signal_processor else False
            risk_block = getattr(self.signal_processor, "risk_block_signals", False) if self.signal_processor else False

            should_be_safe = order_safe or router_block or risk_block

            if should_be_safe and not self.global_safe_mode:
                self.global_safe_mode = True
                logger.critical("Global safe mode ACTIVATED by component flag(s)")
                try:
                    if self.performance_tracker:
                        await self.performance_tracker.record_safe_mode_activation({
                            "order_safe": order_safe,
                            "router_block": router_block,
                            "risk_block": risk_block
                        })
                except Exception:
                    logger.exception("Failed to record safe-mode activation")
                # Reset notification flag so notifier will be triggered on activation
                self._safe_mode_notified = False

            if not should_be_safe and self.global_safe_mode:
                # Clear global safe mode when all components report healthy
                self.global_safe_mode = False
                logger.info("Global safe mode CLEARED; components healthy")
                self._safe_mode_notified = False

        except Exception:
            # On failure to check, conservatively do nothing but log the issue
            logger.exception("Error while checking global safe mode state")

    async def pause(self) -> None:
        """Pause trading operations."""
        self.state.paused = True
        logger.info("Trading coordinator paused")

    async def resume(self) -> None:
        """Resume trading operations."""
        self.state.paused = False
        logger.info("Trading coordinator resumed")

    async def shutdown(self) -> None:
        """Gracefully shutdown the trading coordinator."""
        logger.info("Shutting down TradingCoordinator")
        self.state.running = False

        # Cancel all tracked tasks first with timeout protection
        if self.task_manager:
            try:
                await asyncio.wait_for(self.task_manager.cancel_all(), timeout=30.0)
                logger.info("All tracked tasks cancelled successfully")
            except asyncio.TimeoutError:
                logger.warning("Timeout reached while cancelling tracked tasks")
            except Exception:
                logger.exception("Error cancelling tracked tasks during shutdown")

        # Execute registered shutdown hooks in reverse order with timeout protection
        for hook in reversed(self._shutdown_hooks):
            try:
                await asyncio.wait_for(hook(), timeout=15.0)
                logger.debug(f"Shutdown hook completed: {hook}")
            except asyncio.TimeoutError:
                logger.warning(f"Shutdown hook timed out: {hook}")
            except Exception:
                logger.exception(f"Shutdown hook failed: {hook}")

        logger.info("TradingCoordinator shutdown complete")

    async def _emergency_shutdown(self) -> None:
        """Execute emergency shutdown procedures."""
        logger.critical("Executing emergency shutdown!")

        # Send emergency alert with timeout protection
        if self.notifier:
            try:
                await asyncio.wait_for(
                    self.notifier.send_alert("🚨 EMERGENCY SHUTDOWN TRIGGERED!"),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Emergency alert timed out")
            except Exception:
                logger.exception("Failed to send emergency alert")

        # Cancel all open orders with timeout protection
        if self.order_executor:
            try:
                await asyncio.wait_for(
                    self.order_executor.cancel_all_orders(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("Order cancellation timed out")
            except Exception:
                logger.exception("Failed to cancel orders during emergency shutdown")

        await self.shutdown()

    def get_status(self) -> Dict[str, Any]:
        """Get current coordinator status."""
        return {
            "running": self.state.running,
            "paused": self.state.paused,
            "mode": self.mode.name,
            "global_safe_mode": self.global_safe_mode,
            "active_orders": self.state.active_orders,
            "open_positions": self.state.open_positions,
            "balance": self.state.balance,
            "equity": self.state.equity
        }
