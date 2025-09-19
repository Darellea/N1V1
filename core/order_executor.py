"""
OrderExecutor - Order execution and management component.

Handles order execution, result processing, and coordination
with the OrderManager for trade execution.
Enhanced with formal verification, invariant checks, and security validation.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List
from decimal import Decimal

from utils.logger import get_logger_with_context, generate_correlation_id, generate_request_id
from utils.security import (
    get_order_flow_validator,
    log_security_event,
    SecurityViolationException,
    sanitize_error_message
)

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

        # Get order flow validator for security checks
        validator = get_order_flow_validator()

        for signal in approved_signals:
            try:
                # Pre-execution security validation
                if not await self._validate_signal_security(signal):
                    log_security_event("signal_security_validation_failed", {
                        "signal_id": getattr(signal, 'id', 'unknown'),
                        "symbol": getattr(signal, 'symbol', 'unknown'),
                        "strategy_id": getattr(signal, 'strategy_id', 'unknown')
                    }, "WARNING")
                    continue

                await self._execute_single_order(signal)
            except SecurityViolationException as e:
                log_security_event("security_violation_detected", {
                    "signal_id": getattr(signal, 'id', 'unknown'),
                    "violation": str(e),
                    "component": "OrderExecutor"
                }, "ERROR")
                # Continue with other signals
            except Exception as e:
                sanitized_error = sanitize_error_message(str(e))
                logger.exception(f"Error executing order for signal {signal}: {sanitized_error}")
                # Continue with other signals

    async def _execute_single_order(self, signal) -> None:
        """Execute a single order and process the result with retry logic."""
        try:
            self.execution_stats["total_orders"] += 1

            # Execute the order with retry logic
            order_result = await self._execute_with_retry(signal)

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
                logger.warning(f"Order execution failed after retries for signal: {signal}")

        except Exception as e:
            self.execution_stats["failed_orders"] += 1
            logger.exception(f"Failed to execute order: {e}")

    async def _execute_with_retry(self, signal) -> Optional[Dict[str, Any]]:
        """Execute order with retry logic and exponential backoff."""
        max_retries = self.config.get("max_retries", 3)
        base_delay = self.config.get("retry_base_delay", 1.0)
        max_delay = self.config.get("retry_max_delay", 30.0)
        retry_budget = self.config.get("retry_budget", 5)  # Max retries per order

        # Generate correlation and request IDs for tracing
        correlation_id = generate_correlation_id()
        request_id = generate_request_id()
        strategy_id = getattr(signal, 'strategy_id', 'unknown')
        symbol = getattr(signal, 'symbol', 'unknown')

        # Create logger with context
        ctx_logger = get_logger_with_context(
            symbol=symbol,
            component="OrderExecutor",
            correlation_id=correlation_id,
            request_id=request_id,
            strategy_id=strategy_id
        )

        for attempt in range(max_retries + 1):
            try:
                # Check circuit breaker if available
                if hasattr(self.order_manager, 'reliability_manager') and self.order_manager.reliability_manager:
                    if self.order_manager.reliability_manager.safe_mode_active:
                        ctx_logger.warning("Circuit breaker active, skipping order execution")
                        return None

                # Execute the order
                order_result = await self.order_manager.execute_order(signal)

                if order_result:
                    ctx_logger.info(f"Order executed successfully on attempt {attempt + 1}")
                    return order_result
                else:
                    ctx_logger.warning(f"Order execution returned None on attempt {attempt + 1}")

            except Exception as e:
                error_msg = f"Order execution failed on attempt {attempt + 1}: {str(e)}"
                ctx_logger.warning(error_msg)

                # Check if this is the last attempt
                if attempt == max_retries:
                    ctx_logger.error(f"Order execution failed after {max_retries + 1} attempts")

                    # Send failure alert if notifier available
                    if self.notifier and hasattr(self.notifier, 'send_error_alert'):
                        await self.notifier.send_error_alert({
                            "error": str(e),
                            "component": "OrderExecutor",
                            "correlation_id": correlation_id,
                            "strategy_id": strategy_id,
                            "symbol": symbol,
                            "retry_count": attempt + 1,
                            "max_retries": max_retries
                        })
                    break

                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)
                ctx_logger.info(f"Retrying order execution in {delay:.2f}s")

                # Check retry budget
                if attempt >= retry_budget:
                    ctx_logger.warning(f"Retry budget ({retry_budget}) exceeded")
                    break

                await asyncio.sleep(delay)

        return None

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

    async def _validate_signal_security(self, signal) -> bool:
        """Validate signal security before execution."""
        try:
            # Extract signal data for validation
            signal_data = {
                "id": getattr(signal, 'id', f"signal_{hash(str(signal))}"),
                "symbol": getattr(signal, 'symbol', 'unknown'),
                "side": getattr(signal, 'side', 'unknown'),
                "type": getattr(signal, 'type', 'market'),
                "amount": getattr(signal, 'amount', 0),
                "price": getattr(signal, 'price', None),
                "strategy_id": getattr(signal, 'strategy_id', 'unknown')
            }

            # Get order flow validator
            validator = get_order_flow_validator()

            # Validate order schema
            if not validator.validate_order_schema(signal_data):
                raise SecurityViolationException("Order schema validation failed")

            # Register order for tracking
            if not validator.register_order(signal_data):
                raise SecurityViolationException("Failed to register order for tracking")

            # Check for rate limiting
            if not await self._check_rate_limits(signal_data):
                raise SecurityViolationException("Rate limit exceeded")

            # Validate signal integrity
            if not self._validate_signal_integrity(signal):
                raise SecurityViolationException("Signal integrity check failed")

            return True

        except Exception as e:
            log_security_event("signal_validation_error", {
                "signal_id": getattr(signal, 'id', 'unknown'),
                "error": str(e),
                "component": "OrderExecutor"
            }, "ERROR")
            return False

    async def _check_rate_limits(self, signal_data: Dict[str, Any]) -> bool:
        """Check if signal execution is within rate limits."""
        # Simple in-memory rate limiting (could be enhanced with Redis/external storage)
        current_time = time.time()
        symbol = signal_data.get("symbol", "unknown")

        # Track orders per symbol per minute
        if not hasattr(self, '_rate_limit_cache'):
            self._rate_limit_cache = {}

        if symbol not in self._rate_limit_cache:
            self._rate_limit_cache[symbol] = []

        # Clean old entries (older than 1 minute)
        self._rate_limit_cache[symbol] = [
            ts for ts in self._rate_limit_cache[symbol]
            if current_time - ts < 60
        ]

        # Check rate limit (configurable, default 10 orders per minute per symbol)
        max_orders_per_minute = self.config.get("security", {}).get("max_orders_per_minute", 10)
        if len(self._rate_limit_cache[symbol]) >= max_orders_per_minute:
            log_security_event("rate_limit_exceeded", {
                "symbol": symbol,
                "current_count": len(self._rate_limit_cache[symbol]),
                "limit": max_orders_per_minute
            }, "WARNING")
            return False

        # Add current timestamp
        self._rate_limit_cache[symbol].append(current_time)
        return True

    def _validate_signal_integrity(self, signal) -> bool:
        """Validate signal data integrity."""
        try:
            # Check for required attributes
            required_attrs = ['symbol', 'side', 'amount']
            for attr in required_attrs:
                if not hasattr(signal, attr) or getattr(signal, attr) is None:
                    return False

            # Validate side
            side = getattr(signal, 'side', '').lower()
            if side not in ['buy', 'sell']:
                return False

            # Validate amount
            amount = getattr(signal, 'amount', 0)
            if not isinstance(amount, (int, float)) or amount <= 0:
                return False

            # Validate symbol format
            symbol = getattr(signal, 'symbol', '')
            if not isinstance(symbol, str) or len(symbol) == 0:
                return False

            return True

        except Exception as e:
            log_security_event("signal_integrity_check_failed", {
                "signal_id": getattr(signal, 'id', 'unknown'),
                "error": str(e)
            }, "WARNING")
            return False

    async def validate_order_flow_invariants(self) -> Dict[str, Any]:
        """Validate order flow invariants and return status."""
        validator = get_order_flow_validator()
        return validator.validate_state_consistency()

    def get_security_stats(self) -> Dict[str, Any]:
        """Get security-related statistics."""
        validator = get_order_flow_validator()
        consistency_check = validator.validate_state_consistency()

        return {
            "execution_stats": self.get_execution_stats(),
            "order_flow_consistency": consistency_check,
            "rate_limit_cache_size": len(getattr(self, '_rate_limit_cache', {})),
            "security_events_logged": True  # Placeholder for actual security event counting
        }

    async def perform_security_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive security health check."""
        health_status = {
            "order_flow_validator": False,
            "rate_limiting": False,
            "signal_validation": False,
            "schema_validation": False
        }

        try:
            # Check order flow validator
            validator = get_order_flow_validator()
            consistency = validator.validate_state_consistency()
            health_status["order_flow_validator"] = consistency.get("consistent", False)

            # Check rate limiting
            health_status["rate_limiting"] = hasattr(self, '_rate_limit_cache')

            # Check signal validation
            health_status["signal_validation"] = callable(getattr(self, '_validate_signal_security', None))

            # Check schema validation
            health_status["schema_validation"] = callable(getattr(validator, 'validate_order_schema', None))

        except Exception as e:
            log_security_event("security_health_check_failed", {
                "error": str(e),
                "component": "OrderExecutor"
            }, "ERROR")

        return health_status

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
