"""
core/api_protection.py

Centralized circuit-breaker and rate-limit middleware for all exchange API calls.
Ensures consistent protection across all execution paths.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """API circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking all calls
    HALF_OPEN = "half-open"  # Testing recovery


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""

    pass


@dataclass
class CircuitBreakerConfig:
    """Configuration for API circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    max_half_open_calls: int = 5  # Max calls allowed in half-open state


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiter."""

    requests_per_second: float = 10.0
    burst_limit: int = 20


@dataclass
class APICircuitBreaker:
    """Circuit breaker for API calls."""

    config: CircuitBreakerConfig
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    half_open_call_count: int = 0

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_call_count = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return False
            return True
        return False

    def _should_attempt_recovery(self) -> bool:
        """Check if we should attempt recovery from open state."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    def record_success(self) -> None:
        """Record a successful call."""
        self.failure_count = 0

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker closed after successful recovery")
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset any lingering state
            self.success_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Any failure in half-open state immediately transitions back to open
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker re-opened after half-open failure")
        elif self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )

    def get_state(self) -> str:
        """Get current state as string."""
        return self.state.value


@dataclass
class TokenBucketRateLimiter:
    """Token bucket rate limiter."""

    config: RateLimiterConfig
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens = self.config.burst_limit
        self.last_refill = time.time()

    def acquire(self) -> bool:
        """Try to acquire a token. Returns True if successful."""
        now = time.time()
        time_passed = now - self.last_refill

        # Refill tokens based on time passed
        self.tokens = min(
            self.config.burst_limit,
            self.tokens + time_passed * self.config.requests_per_second,
        )
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


async def guarded_call(
    fn: Callable[..., Any],
    *args,
    circuit_breaker: Optional[APICircuitBreaker] = None,
    rate_limiter: Optional[TokenBucketRateLimiter] = None,
    **kwargs,
) -> Any:
    """
    Execute an API call with circuit breaker and rate limiting protection.

    Args:
        fn: The function to call
        *args: Positional arguments for the function
        circuit_breaker: Optional circuit breaker instance
        rate_limiter: Optional rate limiter instance
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call

    Raises:
        CircuitOpenError: If circuit breaker is open
        RateLimitExceededError: If rate limit is exceeded
        Exception: Any exception from the original function
    """
    # Check circuit breaker
    if circuit_breaker and circuit_breaker.is_open():
        logger.warning(f"Circuit breaker is OPEN for {fn.__name__}")
        raise CircuitOpenError(f"Circuit is open for {fn.__name__}")

    # Check rate limiter
    if rate_limiter and not rate_limiter.acquire():
        logger.warning(f"Rate limit exceeded for {fn.__name__}")
        raise RateLimitExceededError(f"Rate limit exceeded for {fn.__name__}")

    try:
        # Execute the call
        result = fn(*args, **kwargs)

        # Handle async functions
        if asyncio.iscoroutine(result):
            result = await result

        # Record success
        if circuit_breaker:
            circuit_breaker.record_success()

        return result

    except Exception as e:
        # Record failure
        if circuit_breaker:
            circuit_breaker.record_failure()
        raise


# Global instances for shared use
_default_circuit_breaker = APICircuitBreaker(CircuitBreakerConfig())
_default_rate_limiter = TokenBucketRateLimiter(RateLimiterConfig())


def get_default_circuit_breaker() -> APICircuitBreaker:
    """Get the default circuit breaker instance."""
    return _default_circuit_breaker


def get_default_rate_limiter() -> TokenBucketRateLimiter:
    """Get the default rate limiter instance."""
    return _default_rate_limiter
