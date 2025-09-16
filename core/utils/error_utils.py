"""
Common error handling utilities for the N1V1 trading framework.

This module provides standardized error handling patterns, retry mechanisms,
and graceful degradation strategies used across core components.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass
from enum import Enum
import traceback

from ..logging_utils import get_structured_logger

logger = get_structured_logger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for classification."""
    NETWORK = "network"
    CONFIGURATION = "configuration"
    DATA = "data"
    LOGIC = "logic"
    RESOURCE = "resource"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    component: str
    operation: str
    severity: ErrorSeverity
    category: ErrorCategory
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RetryConfig:
    """Configuration for retry mechanisms."""

    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise CircuitBreakerOpenException("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.last_failure_time is None:
            return True

        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            logger.info("Circuit breaker reset to closed state")

    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class ErrorHandler:
    """Centralized error handling with standardized patterns."""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = get_structured_logger(f"core.error_handler.{component_name}")

    async def handle_error(self,
                          error: Exception,
                          context: ErrorContext,
                          fallback_strategy: Optional[Callable] = None) -> Any:
        """
        Handle an error with appropriate logging and recovery.

        Args:
            error: The exception that occurred
            context: Context information about the error
            fallback_strategy: Optional fallback function to execute

        Returns:
            Result from fallback strategy if provided, None otherwise
        """
        # Log the error with structured information
        self._log_error(error, context)

        # Execute fallback strategy if provided
        if fallback_strategy:
            try:
                self.logger.info("Executing fallback strategy", {
                    "component": context.component,
                    "operation": context.operation
                })
                return await fallback_strategy()
            except Exception as fallback_error:
                self.logger.error("Fallback strategy failed", {
                    "component": context.component,
                    "operation": context.operation,
                    "fallback_error": str(fallback_error)
                })

        # Return appropriate default based on operation type
        return self._get_default_value(context.operation)

    def _log_error(self, error: Exception, context: ErrorContext):
        """Log error with appropriate severity."""
        error_details = {
            "component": context.component,
            "operation": context.operation,
            "severity": context.severity.value,
            "category": context.category.value,
            "retry_count": context.retry_count,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }

        if context.metadata:
            error_details.update(context.metadata)

        if context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical error occurred", error_details)
        elif context.severity == ErrorSeverity.HIGH:
            self.logger.error("High severity error occurred", error_details)
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity error occurred", error_details)
        else:
            self.logger.info("Low severity error occurred", error_details)

    def _get_default_value(self, operation: str) -> Any:
        """Get appropriate default value based on operation type."""
        defaults = {
            "fetch_data": {},
            "calculate_metrics": 0.0,
            "validate_config": False,
            "send_notification": False,
            "execute_order": None,
            "get_balance": 0.0,
            "get_equity": 0.0
        }
        return defaults.get(operation, None)


async def retry_async(func: Callable[..., T],
                      config: RetryConfig,
                      *args,
                      **kwargs) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: The async function to retry
        config: Retry configuration
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result from the function

    Raises:
        Exception: The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if attempt < config.max_attempts - 1:
                delay = min(config.base_delay * (config.backoff_factor ** attempt),
                           config.max_delay)

                if config.jitter:
                    delay = delay * (0.5 + 0.5 * time.time() % 1)  # Add jitter

                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s", {
                    "error": str(e),
                    "attempt": attempt + 1,
                    "max_attempts": config.max_attempts,
                    "delay": delay
                })

                await asyncio.sleep(delay)
            else:
                logger.error(f"All {config.max_attempts} attempts failed", {
                    "final_error": str(e),
                    "total_attempts": config.max_attempts
                })

    raise last_exception


def safe_execute(func: Callable[..., T], *args, **kwargs) -> Optional[T]:
    """
    Execute a function safely, returning None on any exception.

    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result or None if an exception occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.debug(f"Safe execution failed: {e}")
        return None


async def safe_execute_async(func: Callable[..., T], *args, **kwargs) -> Optional[T]:
    """
    Execute an async function safely, returning None on any exception.

    Args:
        func: Async function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result or None if an exception occurs
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.debug(f"Safe async execution failed: {e}")
        return None


class GracefulDegradationManager:
    """Manages graceful degradation of system capabilities."""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.degraded_features: Dict[str, bool] = {}
        self.logger = get_structured_logger(f"core.degradation.{component_name}")

    def mark_feature_degraded(self, feature_name: str, reason: str):
        """Mark a feature as degraded."""
        self.degraded_features[feature_name] = True
        self.logger.warning(f"Feature marked as degraded: {feature_name}", {
            "reason": reason,
            "component": self.component_name
        })

    def is_feature_degraded(self, feature_name: str) -> bool:
        """Check if a feature is degraded."""
        return self.degraded_features.get(feature_name, False)

    def restore_feature(self, feature_name: str):
        """Restore a degraded feature."""
        if feature_name in self.degraded_features:
            del self.degraded_features[feature_name]
            self.logger.info(f"Feature restored: {feature_name}", {
                "component": self.component_name
            })

    def get_degraded_features(self) -> List[str]:
        """Get list of currently degraded features."""
        return list(self.degraded_features.keys())


# Custom exceptions
class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class MaxRetriesExceededException(Exception):
    """Exception raised when maximum retry attempts are exceeded."""
    pass


class GracefulDegradationException(Exception):
    """Exception raised when a feature is unavailable due to graceful degradation."""
    pass


# Global instances
_error_handlers: Dict[str, ErrorHandler] = {}
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_degradation_managers: Dict[str, GracefulDegradationManager] = {}


def get_error_handler(component_name: str) -> ErrorHandler:
    """Get or create an error handler for a component."""
    if component_name not in _error_handlers:
        _error_handlers[component_name] = ErrorHandler(component_name)
    return _error_handlers[component_name]


def get_circuit_breaker(name: str,
                       failure_threshold: int = 5,
                       recovery_timeout: float = 60.0) -> CircuitBreaker:
    """Get or create a circuit breaker."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(failure_threshold, recovery_timeout)
    return _circuit_breakers[name]


def get_degradation_manager(component_name: str) -> GracefulDegradationManager:
    """Get or create a graceful degradation manager for a component."""
    if component_name not in _degradation_managers:
        _degradation_managers[component_name] = GracefulDegradationManager(component_name)
    return _degradation_managers[component_name]
