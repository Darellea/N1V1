"""
OrderExecutor - Order execution and management component.

Handles order execution, result processing, and coordination
with the OrderManager for trade execution.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Executes trading orders and manages order lifecycle.

    Responsibilities:
    - Order execution coordination
    - Order result processing
    - Performance tracking integration
    - Notification handling
    - Safe mode management
    """

    def __init__(self, config: Dict[str, Any], order_manager=None, performance_tracker=None, notifier=None):
        """Initialize the OrderExecutor.

        Args:
            config: Configuration dictionary
            order_manager: OrderManager instance for order execution
            performance_tracker: PerformanceTracker for metrics
            notifier: Notifier for order notifications
        """
        self.config = config
        self.order_manager = order_manager
        self.performance_tracker = performance_tracker
        self.notifier = notifier

        # Safe mode flag
        self.safe_mode_active: bool = False

        # Execution statistics
        self.execution_stats: Dict[str, Any] = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "pending_orders": 0,
            "cancelled_orders": 0
        }

    async def execute_orders(self, approved_signals: List[Any]) -> None:
        """Execute approved trading signals and handle results."""
        if self.safe_mode_active:
            logger.warning("OrderExecutor in safe mode - skipping order execution")
            return

        if not self.order_manager:
            logger.error("No order manager available for order execution")
            return

        for signal in approved_signals:
            try:
                await self._execute_single_order(signal)
            except Exception as e:
                logger.exception(f"Error executing order for signal {signal}: {e}")
                # Continue with other signals

    async def _execute_single_order(self, signal) -> None:
        """Execute a single order and process the result."""
        try:
            self.execution_stats["total_orders"] += 1

            # Execute the order
            order_result = await self.order_manager.execute_order(signal)

            if order_result:
                self.execution_stats["successful_orders"] += 1

                # Process order result
                await self._process_order_result(order_result)

                # Send notifications
                if self.notifier:
                    await self.notifier.send_order_notification(order_result)

                logger.info(f"Order executed successfully: {order_result.get('id', 'unknown')}")
            else:
                self.execution_stats["failed_orders"] += 1
                logger.warning(f"Order execution failed for signal: {signal}")

        except Exception as e:
            self.execution_stats["failed_orders"] += 1
            logger.exception(f"Failed to execute order: {e}")

    async def _process_order_result(self, order_result: Dict[str, Any]) -> None:
        """Process the result of an executed order."""
        try:
            # Update performance metrics if PNL is available
            if "pnl" in order_result and self.performance_tracker:
                pnl = order_result["pnl"]
                self.performance_tracker.update_performance_metrics(pnl)

                # Record equity progression
                await self.performance_tracker.record_trade_equity(order_result)

            # Update execution statistics
            if order_result.get("status") == "filled":
                # Additional processing for filled orders
                pass
            elif order_result.get("status") == "cancelled":
                self.execution_stats["cancelled_orders"] += 1
            elif order_result.get("status") == "pending":
                self.execution_stats["pending_orders"] += 1

        except Exception as e:
            logger.exception(f"Error processing order result: {e}")

    async def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        if not self.order_manager:
            logger.error("No order manager available for order cancellation")
            return

        try:
            await self.order_manager.cancel_all_orders()
            logger.info("All orders cancelled successfully")
        except Exception as e:
            logger.exception(f"Failed to cancel all orders: {e}")

    def enable_safe_mode(self):
        """Enable safe mode - prevent order execution."""
        self.safe_mode_active = True
        logger.warning("OrderExecutor safe mode enabled - orders will not be executed")

    def disable_safe_mode(self):
        """Disable safe mode - allow order execution."""
        self.safe_mode_active = False
        logger.info("OrderExecutor safe mode disabled - orders can be executed")

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get order execution statistics."""
        return self.execution_stats.copy()

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific order."""
        if not self.order_manager:
            return None

        try:
            return self.order_manager.get_order_status(order_id)
        except Exception as e:
            logger.exception(f"Failed to get order status for {order_id}: {e}")
            return None

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get all active orders."""
        if not self.order_manager:
            return []

        try:
            return self.order_manager.get_active_orders()
        except Exception as e:
            logger.exception(f"Failed to get active orders: {e}")
            return []

    def get_order_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get order execution history."""
        if not self.order_manager:
            return []

        try:
            return self.order_manager.get_order_history(limit)
        except Exception as e:
            logger.exception(f"Failed to get order history: {e}")
            return []

    async def update_order_statuses(self) -> None:
        """Update the status of all pending orders."""
        if not self.order_manager:
            return

        try:
            await self.order_manager.update_order_statuses()
        except Exception as e:
            logger.exception(f"Failed to update order statuses: {e}")

    def set_risk_parameters(self, risk_params: Dict[str, Any]):
        """Update risk parameters for order execution."""
        if not self.order_manager:
            logger.warning("No order manager available to set risk parameters")
            return

        try:
            # This would depend on the OrderManager's interface
            if hasattr(self.order_manager, 'set_risk_parameters'):
                self.order_manager.set_risk_parameters(risk_params)
                logger.info("Risk parameters updated for order execution")
            else:
                logger.warning("OrderManager does not support dynamic risk parameter updates")
        except Exception as e:
            logger.exception(f"Failed to set risk parameters: {e}")

    async def initialize(self):
        """Initialize the order executor."""
        logger.info("Initializing OrderExecutor")

        if self.order_manager and hasattr(self.order_manager, 'initialize'):
            await self.order_manager.initialize()

        logger.info("OrderExecutor initialization complete")

    async def shutdown(self):
        """Shutdown the order executor."""
        logger.info("Shutting down OrderExecutor")

        # Cancel any pending orders before shutdown
        await self.cancel_all_orders()

        if self.order_manager and hasattr(self.order_manager, 'shutdown'):
            await self.order_manager.shutdown()

        logger.info("OrderExecutor shutdown complete")
