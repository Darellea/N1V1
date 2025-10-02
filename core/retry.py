"""
core/retry.py

Centralized retry logic with idempotency-first design.
Provides unified retry behavior across all N1V1 modules.
"""

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable, Optional

from core.api_protection import APICircuitBreaker, CircuitOpenError
from core.idempotency import RetryNotAllowedError

logger = logging.getLogger(__name__)


class RetryConfig:
    """Global retry configuration."""

    def __init__(self):
        self.max_attempts = 3
        self.base_delay = 0.5
        self.jitter = 0.1
        self.max_delay = 30.0

    def update_from_config(self, config: dict) -> None:
        """Update configuration from global config."""
        self.max_attempts = config.get("max_attempts", self.max_attempts)
        self.base_delay = config.get("base_delay", self.base_delay)
        self.jitter = config.get("jitter", self.jitter)
        self.max_delay = config.get("max_delay", self.max_delay)


# Global retry configuration instance
_global_retry_config = RetryConfig()


def get_global_retry_config() -> RetryConfig:
    """Get the global retry configuration."""
    return _global_retry_config


def update_global_retry_config(config: dict) -> None:
    """Update the global retry configuration."""
    _global_retry_config.update_from_config(config)
    logger.info(f"Global retry config updated: {config}")


async def retry_call(
    fn: Callable[..., Any],
    *args,
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    jitter: Optional[float] = None,
    idempotency_key: Optional[str] = None,
    circuit_breaker: Optional[APICircuitBreaker] = None,
    is_side_effect: bool = False,
    **kwargs,
) -> Any:
    """
    Centralized retry helper with idempotency-first design.

    This is the ONLY retry function that should be used across N1V1.
    All modules must call this function instead of implementing custom retry loops.

    Args:
        fn: Function to retry
        *args: Positional arguments for fn
        max_attempts: Maximum number of attempts (uses global config if None)
        base_delay: Base delay for exponential backoff (uses global config if None)
        jitter: Jitter factor for backoff randomization (uses global config if None)
        idempotency_key: Required for side-effect functions
        circuit_breaker: Optional circuit breaker integration
        is_side_effect: Whether fn has side effects
        **kwargs: Keyword arguments for fn

    Returns:
        Result of successful function call

    Raises:
        RetryNotAllowedError: If retrying side-effect function without idempotency_key
        CircuitOpenError: If circuit breaker is open
        Exception: Last exception from failed attempts
    """
    # Filter out retry_call parameters from kwargs to avoid passing them to fn
    retry_params = {
        "max_attempts",
        "base_delay",
        "jitter",
        "idempotency_key",
        "circuit_breaker",
        "is_side_effect",
    }
    fn_kwargs = {k: v for k, v in kwargs.items() if k not in retry_params}

    # Use global config if parameters not specified
    config = get_global_retry_config()
    max_attempts = max_attempts if max_attempts is not None else config.max_attempts
    base_delay = base_delay if base_delay is not None else config.base_delay
    jitter = jitter if jitter is not None else config.jitter

    # Safety check: side-effect functions require idempotency key
    if is_side_effect and idempotency_key is None:
        raise RetryNotAllowedError(
            f"Retry not allowed on side-effect function {fn.__name__} without idempotency_key"
        )

    for attempt in range(max_attempts):
        # Check circuit breaker before each attempt
        if circuit_breaker and circuit_breaker.is_open():
            logger.warning(f"Circuit breaker is open for {fn.__name__}, aborting retry")
            raise CircuitOpenError(f"Circuit is open for {fn.__name__}, aborting retry")

        try:
            logger.debug(
                f"Calling {fn.__name__} (attempt {attempt + 1}/{max_attempts})"
            )
            result = await _call_fn(fn, *args, **fn_kwargs)

            # Record success with circuit breaker
            if circuit_breaker:
                circuit_breaker.record_success()

            return result

        except Exception as e:
            # Record failure with circuit breaker
            if circuit_breaker:
                circuit_breaker.record_failure()

            is_last_attempt = attempt + 1 >= max_attempts

            if is_last_attempt:
                logger.error(
                    f"All {max_attempts} attempts failed for {fn.__name__}: {e}"
                )
                raise

            # Calculate exponential backoff with jitter
            sleep_time = _calculate_delay(attempt, base_delay, jitter, config.max_delay)

            logger.warning(
                f"Attempt {attempt + 1} failed for {fn.__name__}: {e}. "
                f"Retrying in {sleep_time:.2f}s..."
            )

            await asyncio.sleep(sleep_time)


async def retry_call_sync(
    fn: Callable[..., Any],
    *args,
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    jitter: Optional[float] = None,
    idempotency_key: Optional[str] = None,
    circuit_breaker: Optional[APICircuitBreaker] = None,
    is_side_effect: bool = False,
    **kwargs,
) -> Any:
    """
    Synchronous version of retry_call for non-async functions.

    This wraps sync functions to work with the async retry logic.
    """
    # Use global config if parameters not specified
    config = get_global_retry_config()
    max_attempts = max_attempts if max_attempts is not None else config.max_attempts
    base_delay = base_delay if base_delay is not None else config.base_delay
    jitter = jitter if jitter is not None else config.jitter

    # Safety check: side-effect functions require idempotency key
    if is_side_effect and idempotency_key is None:
        raise RetryNotAllowedError(
            f"Retry not allowed on side-effect function {fn.__name__} without idempotency_key"
        )

    for attempt in range(max_attempts):
        # Check circuit breaker before each attempt
        if circuit_breaker and circuit_breaker.is_open():
            logger.warning(f"Circuit breaker is open for {fn.__name__}, aborting retry")
            raise CircuitOpenError(f"Circuit is open for {fn.__name__}, aborting retry")

        try:
            logger.debug(
                f"Calling {fn.__name__} (attempt {attempt + 1}/{max_attempts})"
            )
            # Run sync function in thread pool
            return await asyncio.to_thread(fn, *args, **kwargs)

        except Exception as e:
            is_last_attempt = attempt + 1 >= max_attempts

            if is_last_attempt:
                logger.error(
                    f"All {max_attempts} attempts failed for {fn.__name__}: {e}"
                )
                raise

            # Calculate exponential backoff with jitter
            sleep_time = _calculate_delay(attempt, base_delay, jitter, config.max_delay)

            logger.warning(
                f"Attempt {attempt + 1} failed for {fn.__name__}: {e}. "
                f"Retrying in {sleep_time:.2f}s..."
            )

            await asyncio.sleep(sleep_time)


async def _call_fn(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Call function, handling both sync and async functions.

    Args:
        fn: Function to call
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result
    """
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    else:
        # Run sync function in thread pool to avoid blocking
        return await asyncio.to_thread(fn, *args, **kwargs)


def _calculate_delay(
    attempt: int, base_delay: float, jitter: float, max_delay: float
) -> float:
    """
    Calculate delay with exponential backoff and jitter.

    Formula: base_delay * (2 ^ attempt) + random(0, jitter)

    Args:
        attempt: Attempt number (0-based)
        base_delay: Base delay in seconds
        jitter: Maximum jitter in seconds
        max_delay: Maximum allowed delay

    Returns:
        Delay in seconds
    """
    # Exponential backoff
    delay = base_delay * (2**attempt)

    # Add jitter to prevent thundering herd
    delay += random.uniform(0, jitter)

    # Cap at maximum delay
    delay = min(delay, max_delay)

    return delay


# Backwards compatibility - these will be deprecated
async def retry_with_backoff(*args, **kwargs):
    """Deprecated: Use retry_call instead."""
    logger.warning("retry_with_backoff is deprecated, use retry_call instead")
    return await retry_call(*args, **kwargs)


def exponential_backoff(*args, **kwargs):
    """Deprecated: Use retry_call instead."""
    logger.warning("exponential_backoff is deprecated, use retry_call instead")
    return retry_call(*args, **kwargs)
