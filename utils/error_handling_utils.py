"""
Centralized Error Handling Utilities for N1V1 Framework.

Provides standardized error handling patterns, context preservation,
and structured logging to eliminate duplication across the codebase.
"""

import asyncio
import functools
import logging
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


@dataclass
class ErrorContext:
    """Structured error context for consistent error reporting."""

    component: str
    operation: str
    user_id: Optional[str] = None
    symbol: Optional[str] = None
    trade_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {
            "component": self.component,
            "operation": self.operation,
            "user_id": self.user_id,
            "symbol": self.symbol,
            "trade_id": self.trade_id,
            "additional_data": self.additional_data or {},
        }


class ErrorHandler:
    """
    Centralized error handler with consistent patterns and structured logging.
    """

    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.component_errors: Dict[str, List[Dict[str, Any]]] = {}

    async def handle_error(
        self,
        exception: Exception,
        context: ErrorContext,
        log_level: str = "ERROR",
        notify: bool = False,
    ) -> None:
        """
        Handle an exception with structured logging and optional notification.

        Args:
            exception: The exception that occurred
            context: Structured context about the error
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            notify: Whether to send notification for critical errors
        """
        error_type = type(exception).__name__
        error_message = str(exception)

        # Update error statistics
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Store error details for the component
        component = context.component
        if component not in self.component_errors:
            self.component_errors[component] = []

        error_record = {
            "timestamp": asyncio.get_event_loop().time(),
            "error_type": error_type,
            "message": error_message,
            "operation": context.operation,
            "traceback": traceback.format_exc(),
        }

        # Keep only last 100 errors per component
        errors = self.component_errors[component]
        errors.append(error_record)
        if len(errors) > 100:
            errors.pop(0)

        # Log with structured context
        log_data = {
            "error_type": error_type,
            "component": component,
            "operation": context.operation,
            "message": error_message,
            "traceback": traceback.format_exc(),
        }

        if context.user_id:
            log_data["user_id"] = context.user_id
        if context.symbol:
            log_data["symbol"] = context.symbol
        if context.trade_id:
            log_data["trade_id"] = context.trade_id
        if context.additional_data:
            log_data.update(context.additional_data)

        # Log at specified level
        log_method = getattr(logger, log_level.lower())
        log_method(
            f"Error in {component}.{context.operation}: {error_message}", extra=log_data
        )

        # Send notification for critical errors
        if notify and log_level in ["ERROR", "CRITICAL"]:
            await self._send_error_notification(exception, context)

    async def _send_error_notification(
        self, exception: Exception, context: ErrorContext
    ) -> None:
        """Send notification for critical errors."""
        try:
            # Use trade logger for notifications
            trade_logger.error(
                f"CRITICAL ERROR in {context.component}: {context.operation}",
                extra={
                    "error_type": type(exception).__name__,
                    "component": context.component,
                    "operation": context.operation,
                    "symbol": context.symbol,
                    "trade_id": context.trade_id,
                },
            )
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": dict(self.error_counts),
            "component_errors": {
                component: len(errors)
                for component, errors in self.component_errors.items()
            },
        }

    def get_recent_errors(
        self, component: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent errors for a specific component."""
        if component not in self.component_errors:
            return []

        errors = self.component_errors[component]
        return errors[-limit:] if limit > 0 else errors


# Global error handler instance
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _error_handler


# Decorator for consistent error handling
def handle_errors(
    component: str,
    operation: str,
    log_level: str = "ERROR",
    notify: bool = False,
    reraise: bool = True,
):
    """
    Decorator for consistent error handling across functions.

    Args:
        component: Component name (e.g., 'risk_manager', 'order_executor')
        operation: Operation name (e.g., 'calculate_position_size', 'execute_order')
        log_level: Logging level for errors
        notify: Whether to send notifications for errors
        reraise: Whether to re-raise the exception after handling

    Returns:
        Decorated function with error handling
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = ErrorContext(component=component, operation=operation)

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                await _error_handler.handle_error(e, context, log_level, notify)
                if reraise:
                    raise
                return None

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            context = ErrorContext(component=component, operation=operation)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, we need to run the async handler
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        _error_handler.handle_error(e, context, log_level, notify)
                    )
                finally:
                    loop.close()

                if reraise:
                    raise
                return None

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Context manager for error handling
@contextmanager
def error_context(component: str, operation: str, **context_kwargs):
    """
    Context manager for consistent error handling.

    Args:
        component: Component name
        operation: Operation name
        **context_kwargs: Additional context data
    """
    context = ErrorContext(
        component=component, operation=operation, additional_data=context_kwargs
    )

    try:
        yield context
    except Exception as e:
        # Run error handling synchronously in context manager
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                _error_handler.handle_error(e, context, "ERROR", False)
            )
        finally:
            loop.close()
        raise


# Utility functions for common error scenarios
async def handle_network_error(
    operation: str,
    exception: Exception,
    component: str = "network",
    retry_count: int = 0,
) -> None:
    """
    Handle network-related errors with structured logging.

    Args:
        operation: The operation that failed
        exception: The exception that occurred
        component: Component name (default: 'network')
        retry_count: Number of retries attempted
    """
    context = ErrorContext(
        component=component,
        operation=operation,
        additional_data={"retry_count": retry_count, "error_category": "network"},
    )

    await _error_handler.handle_error(exception, context, "WARNING", False)


async def handle_data_error(
    operation: str,
    exception: Exception,
    component: str = "data_processor",
    symbol: str = None,
) -> None:
    """
    Handle data processing errors with structured logging.

    Args:
        operation: The operation that failed
        exception: The exception that occurred
        component: Component name (default: 'data_processor')
        symbol: Trading symbol if applicable
    """
    context = ErrorContext(
        component=component,
        operation=operation,
        symbol=symbol,
        additional_data={"error_category": "data"},
    )

    await _error_handler.handle_error(exception, context, "ERROR", False)


async def handle_security_error(
    operation: str,
    exception: Exception,
    component: str = "security",
    user_id: str = None,
) -> None:
    """
    Handle security-related errors with structured logging and notification.

    Args:
        operation: The operation that failed
        exception: The exception that occurred
        component: Component name (default: 'security')
        user_id: User ID if applicable
    """
    context = ErrorContext(
        component=component,
        operation=operation,
        user_id=user_id,
        additional_data={"error_category": "security"},
    )

    # Security errors always trigger notifications
    await _error_handler.handle_error(exception, context, "CRITICAL", True)


async def handle_trading_error(
    operation: str,
    exception: Exception,
    component: str = "trading",
    symbol: str = None,
    trade_id: str = None,
) -> None:
    """
    Handle trading-related errors with structured logging.

    Args:
        operation: The operation that failed
        exception: The exception that occurred
        component: Component name (default: 'trading')
        symbol: Trading symbol if applicable
        trade_id: Trade ID if applicable
    """
    context = ErrorContext(
        component=component,
        operation=operation,
        symbol=symbol,
        trade_id=trade_id,
        additional_data={"error_category": "trading"},
    )

    await _error_handler.handle_error(exception, context, "ERROR", False)


# Safe execution utilities
async def safe_execute_async(
    func: Callable,
    *args,
    component: str = "unknown",
    operation: str = "execution",
    **kwargs,
) -> Any:
    """
    Safely execute an async function with error handling.

    Args:
        func: The async function to execute
        *args: Positional arguments for the function
        component: Component name for error context
        operation: Operation name for error context
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or None if error occurred
    """
    context = ErrorContext(component=component, operation=operation)

    try:
        return await func(*args, **kwargs)
    except Exception as e:
        await _error_handler.handle_error(e, context, "ERROR", False)
        return None


def safe_execute_sync(
    func: Callable,
    *args,
    component: str = "unknown",
    operation: str = "execution",
    **kwargs,
) -> Any:
    """
    Safely execute a sync function with error handling.

    Args:
        func: The sync function to execute
        *args: Positional arguments for the function
        component: Component name for error context
        operation: Operation name for error context
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or None if error occurred
    """
    context = ErrorContext(component=component, operation=operation)

    try:
        return func(*args, **kwargs)
    except Exception as e:
        # For sync functions, we need to run the async handler
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                _error_handler.handle_error(e, context, "ERROR", False)
            )
        finally:
            loop.close()
        return None


# Error recovery utilities
async def with_retry_async(
    func: Callable, max_retries: int = 3, backoff_factor: float = 1.0, *args, **kwargs
) -> Any:
    """
    Execute an async function with retry logic and error handling.

    Args:
        func: The async function to execute
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff factor for retry delays
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Function result

    Raises:
        Exception: The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if attempt < max_retries:
                delay = backoff_factor * (2**attempt)
                await asyncio.sleep(delay)

                logger.warning(
                    f"Attempt {attempt + 1} failed for {func.__name__}, "
                    f"retrying in {delay:.1f}s: {e}"
                )
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                )
                raise last_exception

    # This should never be reached, but just in case
    raise last_exception


def with_retry_sync(
    func: Callable, max_retries: int = 3, backoff_factor: float = 1.0, *args, **kwargs
) -> Any:
    """
    Execute a sync function with retry logic and error handling.

    Args:
        func: The sync function to execute
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff factor for retry delays
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Function result

    Raises:
        Exception: The last exception if all retries fail
    """
    import time

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if attempt < max_retries:
                delay = backoff_factor * (2**attempt)
                time.sleep(delay)

                logger.warning(
                    f"Attempt {attempt + 1} failed for {func.__name__}, "
                    f"retrying in {delay:.1f}s: {e}"
                )
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                )
                raise last_exception

    # This should never be reached, but just in case
    raise last_exception
