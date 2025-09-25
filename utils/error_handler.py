"""
Error Handler - Centralized error handling and recovery system.

Provides standardized error handling patterns, structured logging,
error recovery strategies, and comprehensive error context preservation.
"""

import logging
import traceback
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import sys

def _get_sanitize_error_message():
    from utils.security import sanitize_error_message
    return sanitize_error_message

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for structured handling."""
    NETWORK = "network"
    DATA = "data"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    EXTERNAL_API = "external_api"


@dataclass
class ErrorContext:
    """Structured error context information."""
    timestamp: float = field(default_factory=time.time)
    component: str = ""
    operation: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.SYSTEM
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    symbol: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    error_code: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    recovery_strategy: Optional[str] = None


class TradingError(Exception):
    """Base exception class for trading system errors."""

    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext()
        self.context.error_code = self.__class__.__name__


class NetworkError(TradingError):
    """Network-related errors."""
    pass


class DataError(TradingError):
    """Data processing and validation errors."""
    pass


class ConfigurationError(TradingError):
    """Configuration-related errors."""
    pass


class SecurityError(TradingError):
    """Security-related errors."""
    pass


class PerformanceError(TradingError):
    """Performance-related errors."""
    pass


class BusinessLogicError(TradingError):
    """Business logic errors."""
    pass


class ExternalAPIError(TradingError):
    """External API errors."""
    pass


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    async def execute_with_recovery(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with recovery strategy."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    await self._handle_retry(e, attempt)
                else:
                    raise e

        raise last_exception

    async def _handle_retry(self, exception: Exception, attempt: int):
        """Handle retry logic."""
        delay = self.backoff_factor ** attempt
        logger.warning(f"Retry attempt {attempt + 1} after {delay:.2f}s due to: {exception}")
        await asyncio.sleep(delay)


class ExponentialBackoffStrategy(ErrorRecoveryStrategy):
    """Exponential backoff recovery strategy."""
    pass


class CircuitBreakerStrategy(ErrorRecoveryStrategy):
    """Circuit breaker pattern for external service failures."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    async def execute_with_recovery(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                logger.info("Circuit breaker transitioning to half-open state")
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed - service recovered")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")

            raise e


class ErrorHandler:
    """
    Centralized error handling system with structured logging,
    recovery strategies, and comprehensive error context.
    """

    def __init__(self):
        self.recovery_strategies: Dict[str, ErrorRecoveryStrategy] = {}
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        self.error_history: List[ErrorContext] = []
        self.max_history_size = 1000

        # Register default recovery strategies
        self.register_recovery_strategy("network", CircuitBreakerStrategy())
        self.register_recovery_strategy("api", ExponentialBackoffStrategy(max_retries=5))
        self.register_recovery_strategy("data", ExponentialBackoffStrategy(max_retries=2))

        # Register default error handlers
        self.register_error_handler(TradingError, self._handle_trading_error)
        self.register_error_handler(Exception, self._handle_generic_error)

    def register_recovery_strategy(self, name: str, strategy: ErrorRecoveryStrategy):
        """Register a recovery strategy."""
        self.recovery_strategies[name] = strategy
        logger.info(f"Registered recovery strategy: {name}")

    def register_error_handler(self, exception_type: Type[Exception], handler: Callable):
        """Register an error handler for specific exception types."""
        self.error_handlers[exception_type] = handler
        logger.info(f"Registered error handler for: {exception_type.__name__}")

    def create_error_context(self, component: str = "", operation: str = "",
                           severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                           category: ErrorCategory = ErrorCategory.SYSTEM,
                           **kwargs) -> ErrorContext:
        """Create a structured error context."""
        context = ErrorContext(
            component=component,
            operation=operation,
            severity=severity,
            category=category
        )

        # Add any additional context
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
            else:
                context.parameters[key] = value

        return context

    async def handle_error(self, exception: Exception, context: Optional[ErrorContext] = None) -> None:
        """Handle an exception with structured logging and recovery."""
        # Create context if not provided
        if context is None:
            context = self.create_error_context()

        # Add exception details to context
        context.stack_trace = traceback.format_exc()
        context.parameters["exception_type"] = type(exception).__name__
        context.parameters["exception_message"] = str(exception)

        # Sanitize error message for security
        sanitize_func = _get_sanitize_error_message()
        sanitized_message = sanitize_func(str(exception))

        # Log error with structured context
        await self._log_error(exception, context, sanitized_message)

        # Store in error history
        self._store_error_history(context)

        # Apply recovery strategy if available
        await self._apply_recovery_strategy(exception, context)

        # Call specific error handler
        await self._call_error_handler(exception, context)

    async def _log_error(self, exception: Exception, context: ErrorContext, sanitized_message: str):
        """Log error with structured information."""
        log_data = {
            "component": context.component,
            "operation": context.operation,
            "severity": context.severity.value,
            "category": context.category.value,
            "error_code": context.error_code,
            "retry_count": context.retry_count,
            "user_id": context.user_id,
            "session_id": context.session_id,
            "request_id": context.request_id,
            "symbol": context.symbol,
            "parameters": context.parameters
        }

        # Choose log level based on severity
        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error in {context.component}.{context.operation}: {sanitized_message}",
                          extra={"error_context": log_data})
        elif context.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error in {context.component}.{context.operation}: {sanitized_message}",
                        extra={"error_context": log_data})
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error in {context.component}.{context.operation}: {sanitized_message}",
                          extra={"error_context": log_data})
        else:
            logger.info(f"Low severity error in {context.component}.{context.operation}: {sanitized_message}",
                       extra={"error_context": log_data})

    def _store_error_history(self, context: ErrorContext):
        """Store error in history for analysis."""
        self.error_history.append(context)

        # Keep only recent errors
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]

    async def _apply_recovery_strategy(self, exception: Exception, context: ErrorContext):
        """Apply appropriate recovery strategy."""
        strategy_name = self._determine_recovery_strategy(context.category)
        strategy = self.recovery_strategies.get(strategy_name)

        if strategy and context.retry_count < context.max_retries:
            context.recovery_strategy = strategy_name
            logger.info(f"Applying recovery strategy: {strategy_name} for {context.category.value}")

            # Note: Actual retry logic would be handled by the calling code
            # This just sets up the context for retry

    def _determine_recovery_strategy(self, category: ErrorCategory) -> Optional[str]:
        """Determine which recovery strategy to use based on error category."""
        strategy_map = {
            ErrorCategory.NETWORK: "network",
            ErrorCategory.EXTERNAL_API: "api",
            ErrorCategory.DATA: "data",
            ErrorCategory.CONFIGURATION: "data",
            ErrorCategory.SECURITY: "data",
            ErrorCategory.PERFORMANCE: "data",
            ErrorCategory.BUSINESS_LOGIC: "data",
            ErrorCategory.SYSTEM: "data"
        }

        return strategy_map.get(category)

    async def _call_error_handler(self, exception: Exception, context: ErrorContext):
        """Call the appropriate error handler."""
        # Find the most specific handler
        for exception_type, handler in self.error_handlers.items():
            if isinstance(exception, exception_type):
                try:
                    await handler(exception, context)
                except Exception as handler_error:
                    logger.error(f"Error in error handler for {exception_type.__name__}: {handler_error}")
                break

    async def _handle_trading_error(self, exception: TradingError, context: ErrorContext):
        """Handle trading-specific errors."""
        # Trading-specific error handling logic
        if context.category == ErrorCategory.SECURITY:
            logger.critical("Security error detected - potential breach")
            # Could trigger security protocols here

        elif context.category == ErrorCategory.EXTERNAL_API:
            logger.warning("External API error - may need rate limit adjustment")

        # Add trading-specific recovery actions
        if hasattr(exception, 'context') and exception.context:
            # Merge contexts
            context.parameters.update(exception.context.parameters)

    async def _handle_generic_error(self, exception: Exception, context: ErrorContext):
        """Handle generic exceptions."""
        # Generic error handling - log and potentially escalate
        if context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.error("Unhandled error with high severity - requires attention")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_history:
            return {"total_errors": 0}

        stats = {
            "total_errors": len(self.error_history),
            "severity_breakdown": {},
            "category_breakdown": {},
            "component_breakdown": {},
            "recent_errors": []
        }

        for error in self.error_history:
            # Severity breakdown
            severity = error.severity.value
            stats["severity_breakdown"][severity] = stats["severity_breakdown"].get(severity, 0) + 1

            # Category breakdown
            category = error.category.value
            stats["category_breakdown"][category] = stats["category_breakdown"].get(category, 0) + 1

            # Component breakdown
            component = error.component
            stats["component_breakdown"][component] = stats["component_breakdown"].get(component, 0) + 1

        # Recent errors (last 10)
        stats["recent_errors"] = [
            {
                "timestamp": error.timestamp,
                "component": error.component,
                "operation": error.operation,
                "severity": error.severity.value,
                "category": error.category.value,
                "error_code": error.error_code
            }
            for error in self.error_history[-10:]
        ]

        return stats

    def clear_error_history(self):
        """Clear the error history."""
        self.error_history.clear()
        logger.info("Error history cleared")


# Global error handler instance
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


# Decorator for error handling
def handle_errors(component: str = "", operation: str = "",
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.SYSTEM):
    """Decorator to handle errors in functions."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            context = error_handler.create_error_context(
                component=component,
                operation=operation or func.__name__,
                severity=severity,
                category=category
            )

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                await error_handler.handle_error(e, context)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            context = error_handler.create_error_context(
                component=component,
                operation=operation or func.__name__,
                severity=severity,
                category=category
            )

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, we need to run the async handler
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(error_handler.handle_error(e, context))
                finally:
                    loop.close()
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Utility functions for common error scenarios
async def handle_network_error(operation: str, exception: Exception, retry_count: int = 0):
    """Handle network-related errors with retry logic."""
    error_handler = get_error_handler()
    context = error_handler.create_error_context(
        component="network",
        operation=operation,
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.NETWORK,
        retry_count=retry_count
    )

    await error_handler.handle_error(exception, context)


async def handle_data_error(operation: str, exception: Exception, symbol: str = None):
    """Handle data processing errors."""
    error_handler = get_error_handler()
    context = error_handler.create_error_context(
        component="data_processor",
        operation=operation,
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.DATA,
        symbol=symbol
    )

    await error_handler.handle_error(exception, context)


async def handle_security_error(operation: str, exception: Exception, user_id: str = None):
    """Handle security-related errors."""
    error_handler = get_error_handler()
    context = error_handler.create_error_context(
        component="security",
        operation=operation,
        severity=ErrorSeverity.CRITICAL,
        category=ErrorCategory.SECURITY,
        user_id=user_id
    )

    await error_handler.handle_error(exception, context)
