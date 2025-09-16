"""
StateManager - Bot state management and monitoring component.

Handles bot state updates, display management, status logging,
and monitoring of all bot components.
"""

import logging
import threading
from typing import Dict, Any, Optional, List
from utils.time import now_ms

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages bot state updates and monitoring.

    Responsibilities:
    - Bot state synchronization
    - Display updates
    - Status logging and reporting
    - Component health monitoring
    - State persistence
    """

    def __init__(self, config: Dict[str, Any], order_manager=None, performance_tracker=None):
        """Initialize the StateManager.

        Args:
            config: Configuration dictionary
            order_manager: OrderManager for state queries
            performance_tracker: PerformanceTracker for metrics
        """
        self.config = config
        self.order_manager = order_manager
        self.performance_tracker = performance_tracker

        # Thread synchronization locks
        self._health_lock = threading.RLock()  # For component_health operations
        self._state_lock = threading.RLock()   # For state tracking operations

        # Display components
        self.live_display = None
        self.display_table = None

        # State tracking
        self.last_update_time = 0
        self.update_interval = self.config.get("monitoring", {}).get("update_interval", 60)

        # Component health status
        self.component_health: Dict[str, Any] = {
            "data_manager": {"status": "unknown", "last_check": 0},
            "signal_processor": {"status": "unknown", "last_check": 0},
            "order_executor": {"status": "unknown", "last_check": 0},
            "performance_tracker": {"status": "unknown", "last_check": 0},
            "notifier": {"status": "unknown", "last_check": 0}
        }

    async def update_state(self) -> None:
        """Update the bot's internal state from various components."""
        try:
            current_time = now_ms()

            # Thread-safe access to state tracking
            with self._state_lock:
                # Throttle updates to prevent excessive logging
                if current_time - self.last_update_time < (self.update_interval * 1000):
                    return
                self.last_update_time = current_time

            # Update state from order manager
            if self.order_manager:
                try:
                    balance = await self.order_manager.get_balance()
                    equity = await self.order_manager.get_equity()
                    active_orders = await self.order_manager.get_active_order_count()
                    open_positions = await self.order_manager.get_open_position_count()

                    # Thread-safe update to component health
                    with self._health_lock:
                        self.component_health["order_executor"] = {
                            "status": "healthy",
                            "last_check": current_time,
                            "balance": balance,
                            "equity": equity,
                            "active_orders": active_orders,
                            "open_positions": open_positions
                        }

                except Exception as e:
                    logger.exception(f"Failed to update state from order manager: {e}")
                    with self._health_lock:
                        self.component_health["order_executor"] = {
                            "status": "error",
                            "last_check": current_time,
                            "error": str(e)
                        }

            # Update performance metrics
            if self.performance_tracker:
                try:
                    perf_stats = self.performance_tracker.get_performance_stats()
                    with self._health_lock:
                        self.component_health["performance_tracker"] = {
                            "status": "healthy",
                            "last_check": current_time,
                            "total_pnl": perf_stats.get("total_pnl", 0),
                            "win_rate": perf_stats.get("win_rate", 0),
                            "total_trades": perf_stats.get("wins", 0) + perf_stats.get("losses", 0)
                        }
                except Exception as e:
                    logger.exception(f"Failed to update performance state: {e}")
                    with self._health_lock:
                        self.component_health["performance_tracker"] = {
                            "status": "error",
                            "last_check": current_time,
                            "error": str(e)
                        }

        except Exception as e:
            logger.exception(f"Failed to update bot state: {e}")

    async def update_display(self) -> None:
        """Update the terminal display with latest state."""
        if not self.live_display:
            return

        try:
            # Gather latest state data
            state_data = await self._gather_display_data()

            # Update the display
            self.live_display.update(state_data)

        except Exception as e:
            logger.exception(f"Failed to update display: {e}")

    async def _gather_display_data(self) -> Dict[str, Any]:
        """Gather data for display update."""
        # Thread-safe access to component health
        with self._health_lock:
            component_health_copy = self.component_health.copy()

        display_data = {
            "timestamp": now_ms(),
            "component_health": component_health_copy
        }

        # Add order manager data
        if self.order_manager:
            try:
                display_data.update({
                    "balance": await self.order_manager.get_balance(),
                    "equity": await self.order_manager.get_equity(),
                    "active_orders": await self.order_manager.get_active_order_count(),
                    "open_positions": await self.order_manager.get_open_position_count()
                })
            except Exception as e:
                logger.exception(f"Failed to gather order data for display: {e}")

        # Add performance data
        if self.performance_tracker:
            try:
                perf_stats = self.performance_tracker.get_performance_stats()
                display_data.update({
                    "performance_stats": perf_stats,
                    "total_pnl": perf_stats.get("total_pnl", 0),
                    "win_rate": perf_stats.get("win_rate", 0),
                    "sharpe_ratio": perf_stats.get("sharpe_ratio", 0)
                })
            except Exception as e:
                logger.exception(f"Failed to gather performance data for display: {e}")

        return display_data

    def log_status(self) -> None:
        """Log the current bot status."""
        try:
            # Gather current state
            status_data = self._gather_status_data()

            # Format and log status
            self._format_and_log_status(status_data)

        except Exception as e:
            logger.exception(f"Failed to log status: {e}")

    def _gather_status_data(self) -> Dict[str, Any]:
        """Gather data for status logging."""
        # Thread-safe access to component health
        with self._health_lock:
            component_health_copy = self.component_health.copy()

        status_data = {
            "timestamp": now_ms(),
            "mode": self.config.get("environment", {}).get("mode", "unknown"),
            "component_health": component_health_copy
        }

        # Add order data (synchronous snapshot)
        if self.order_manager:
            try:
                # Use cached values if available
                order_health = component_health_copy.get("order_executor", {})
                status_data.update({
                    "balance": order_health.get("balance", 0),
                    "equity": order_health.get("equity", 0),
                    "active_orders": order_health.get("active_orders", 0),
                    "open_positions": order_health.get("open_positions", 0)
                })
            except Exception as e:
                logger.exception(f"Failed to gather order status data: {e}")

        # Add performance data
        if self.performance_tracker:
            try:
                perf_health = component_health_copy.get("performance_tracker", {})
                status_data.update({
                    "total_pnl": perf_health.get("total_pnl", 0),
                    "win_rate": perf_health.get("win_rate", 0),
                    "total_trades": perf_health.get("total_trades", 0)
                })
            except Exception as e:
                logger.exception(f"Failed to gather performance status data: {e}")

        return status_data

    def _format_and_log_status(self, status_data: Dict[str, Any]) -> None:
        """Format and log the status information."""
        try:
            mode = status_data.get("mode", "unknown")
            balance = status_data.get("balance", 0)
            equity = status_data.get("equity", 0)
            active_orders = status_data.get("active_orders", 0)
            open_positions = status_data.get("open_positions", 0)
            total_pnl = status_data.get("total_pnl", 0)
            win_rate = status_data.get("win_rate", 0)

            # Format currency display
            base_currency = self.config.get("exchange", {}).get("base_currency", "USD")
            balance_str = f"{float(balance):.2f} {base_currency}" if balance else "N/A"
            equity_str = f"{float(equity):.2f} {base_currency}" if equity else "N/A"

            logger.info(
                f"Bot Status - Mode: {mode}, Balance: {balance_str}, Equity: {equity_str}, "
                f"Active Orders: {active_orders}, Open Positions: {open_positions}, "
                f"Total PnL: {total_pnl:.2f}, Win Rate: {win_rate:.2%}"
            )

        except Exception as e:
            logger.exception(f"Failed to format status log: {e}")

    def print_status_table(self) -> None:
        """Print a formatted status table."""
        try:
            status_data = self._gather_status_data()
            self._print_formatted_table(status_data)
        except Exception as e:
            logger.exception(f"Failed to print status table: {e}")

    def _print_formatted_table(self, status_data: Dict[str, Any]) -> None:
        """Print a formatted status table."""
        mode = status_data.get("mode", "unknown")
        balance = status_data.get("balance", 0)
        equity = status_data.get("equity", 0)
        active_orders = status_data.get("active_orders", 0)
        open_positions = status_data.get("open_positions", 0)
        total_pnl = status_data.get("total_pnl", 0)
        win_rate = status_data.get("win_rate", 0)

        # Format values
        base_currency = self.config.get("exchange", {}).get("base_currency", "USD")
        balance_str = f"{float(balance):.2f} {base_currency}" if balance else "N/A"
        equity_str = f"{float(equity):.2f} {base_currency}" if equity else "N/A"
        pnl_str = f"{total_pnl:.2f}" if total_pnl else "0.00"
        win_rate_str = f"{win_rate:.2%}" if win_rate else "0.00%"

        # Print table
        print("\n+-----------------+---------------------+")
        print("| Trading Bot Status                  |")
        print("+-----------------+---------------------+")
        print(f"| Mode            | {mode:<19} |")
        print(f"| Balance         | {balance_str:<19} |")
        print(f"| Equity          | {equity_str:<19} |")
        print(f"| Active Orders   | {active_orders:<19} |")
        print(f"| Open Positions  | {open_positions:<19} |")
        print(f"| Total PnL       | {pnl_str:<19} |")
        print(f"| Win Rate        | {win_rate_str:<19} |")
        print("+-----------------+---------------------+")

    def update_component_health(self, component_name: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Update the health status of a component."""
        try:
            health_info = {
                "status": status,
                "last_check": now_ms()
            }

            if details:
                health_info.update(details)

            # Thread-safe update to component health
            with self._health_lock:
                self.component_health[component_name] = health_info

            if status == "error":
                logger.warning(f"Component {component_name} health status: {status}")
            else:
                logger.debug(f"Component {component_name} health status: {status}")

        except Exception as e:
            logger.exception(f"Failed to update component health for {component_name}: {e}")

    def get_component_health(self) -> Dict[str, Any]:
        """Get the health status of all components."""
        # Thread-safe access to component health
        with self._health_lock:
            return self.component_health.copy()

    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health assessment."""
        try:
            health_status = self.get_component_health()
            total_components = len(health_status)
            healthy_components = sum(1 for comp in health_status.values() if comp.get("status") == "healthy")
            error_components = sum(1 for comp in health_status.values() if comp.get("status") == "error")

            overall_status = "healthy"
            if error_components > 0:
                overall_status = "degraded"
            if error_components == total_components:
                overall_status = "critical"

            return {
                "overall_status": overall_status,
                "total_components": total_components,
                "healthy_components": healthy_components,
                "error_components": error_components,
                "health_percentage": (healthy_components / total_components * 100) if total_components > 0 else 0,
                "component_details": health_status
            }

        except Exception as e:
            logger.exception(f"Failed to assess overall health: {e}")
            return {
                "overall_status": "unknown",
                "error": str(e)
            }

    def initialize_display(self):
        """Initialize the display components."""
        try:
            # For now, just disable rich display
            self.live_display = None
            logger.info("Display components initialized (rich dependency removed)")
        except Exception as e:
            logger.exception(f"Failed to initialize display: {e}")

    async def shutdown(self):
        """Shutdown the state manager."""
        logger.info("Shutting down StateManager")

        if self.live_display:
            try:
                self.live_display.stop()
            except Exception as e:
                logger.debug("Error stopping live display", exc_info=True)

        logger.info("StateManager shutdown complete")
