"""
tests/test_api_protection.py

Tests for centralized circuit-breaker and rate-limit middleware.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.api_protection import (
    APICircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitOpenError,
    RateLimitExceededError,
    RateLimiterConfig,
    TokenBucketRateLimiter,
    get_default_circuit_breaker,
    get_default_rate_limiter,
    guarded_call,
)


class TestAPICircuitBreaker:
    """Test cases for API circuit breaker functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a fresh circuit breaker instance."""
        return APICircuitBreaker(CircuitBreakerConfig())

    def test_initial_state_closed(self, circuit_breaker):
        """Test circuit breaker starts in closed state."""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0
        assert not circuit_breaker.is_open()

    def test_record_success_closed_state(self, circuit_breaker):
        """Test recording success in closed state."""
        circuit_breaker.record_success()
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_record_failure_opens_circuit(self, circuit_breaker):
        """Test recording failures opens the circuit after threshold."""
        # Record failures up to threshold
        for i in range(4):
            circuit_breaker.record_failure()
            assert circuit_breaker.state == CircuitBreakerState.CLOSED
            assert not circuit_breaker.is_open()

        # Fifth failure should open the circuit
        circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.is_open()

    def test_is_open_returns_true_when_open(self, circuit_breaker):
        """Test is_open returns True when circuit is open."""
        # Open the circuit
        for _ in range(5):
            circuit_breaker.record_failure()

        assert circuit_breaker.is_open()

    def test_recovery_timeout_opens_half_open(self, circuit_breaker):
        """Test circuit transitions to half-open after recovery timeout."""
        # Open the circuit
        for _ in range(5):
            circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Simulate time passing (set last_failure_time to past)
        circuit_breaker.last_failure_time = time.time() - 70  # 70 seconds ago

        # Next is_open() call should transition to half-open
        assert not circuit_breaker.is_open()
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

    def test_half_open_success_transitions_to_closed(self, circuit_breaker):
        """Test successful calls in half-open state eventually close the circuit."""
        # Get to half-open state
        for _ in range(5):
            circuit_breaker.record_failure()
        circuit_breaker.last_failure_time = time.time() - 70
        circuit_breaker.is_open()  # This transitions to half-open

        # Record successes
        for i in range(2):
            circuit_breaker.record_success()
            assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Third success should close the circuit
        circuit_breaker.record_success()
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    def test_half_open_failure_transitions_back_to_open(self, circuit_breaker):
        """Test failures in half-open state transition back to open."""
        # Get to half-open state
        for _ in range(5):
            circuit_breaker.record_failure()
        circuit_breaker.last_failure_time = time.time() - 70
        circuit_breaker.is_open()  # This transitions to half-open

        # Record a failure
        circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    def test_get_state_returns_string(self, circuit_breaker):
        """Test get_state returns current state as string."""
        assert circuit_breaker.get_state() == "closed"

        # Open circuit
        for _ in range(5):
            circuit_breaker.record_failure()
        assert circuit_breaker.get_state() == "open"


class TestTokenBucketRateLimiter:
    """Test cases for token bucket rate limiter."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a rate limiter with fast refill for testing."""
        return TokenBucketRateLimiter(
            RateLimiterConfig(requests_per_second=10.0, burst_limit=5)
        )

    def test_initial_tokens_equal_burst_limit(self, rate_limiter):
        """Test rate limiter starts with tokens equal to burst limit."""
        assert rate_limiter.tokens == 5

    def test_acquire_returns_true_when_tokens_available(self, rate_limiter):
        """Test acquire returns True when tokens are available."""
        assert rate_limiter.acquire()
        assert rate_limiter.tokens == 4

    def test_acquire_returns_false_when_no_tokens(self, rate_limiter):
        """Test acquire returns False when no tokens available."""
        # Use up all tokens
        for _ in range(5):
            assert rate_limiter.acquire()

        # Next acquire should fail
        assert not rate_limiter.acquire()
        assert rate_limiter.tokens == 0

    def test_tokens_refill_over_time(self, rate_limiter):
        """Test tokens refill over time."""
        # Use up all tokens
        for _ in range(5):
            rate_limiter.acquire()

        # Simulate time passing (0.5 seconds = 5 tokens at 10/sec)
        rate_limiter.last_refill = time.time() - 0.5

        # Should be able to acquire again
        assert rate_limiter.acquire()
        assert rate_limiter.tokens < 5  # Should have refilled some but used one


class TestGuardedCall:
    """Test cases for guarded_call function."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker for testing."""
        return APICircuitBreaker(CircuitBreakerConfig())

    @pytest.fixture
    def rate_limiter(self):
        """Create a rate limiter for testing."""
        return TokenBucketRateLimiter(
            RateLimiterConfig(
                requests_per_second=100.0, burst_limit=10  # Fast for testing
            )
        )

    @pytest.mark.asyncio
    async def test_guarded_call_success(self):
        """Test successful guarded call."""

        async def success_fn():
            return "success"

        result = await guarded_call(success_fn)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_guarded_call_with_sync_function(self):
        """Test guarded call with synchronous function."""

        def sync_fn():
            return "sync_success"

        result = await guarded_call(sync_fn)
        assert result == "sync_success"

    @pytest.mark.asyncio
    async def test_guarded_call_circuit_open_raises_error(self, circuit_breaker):
        """Test guarded call raises CircuitOpenError when circuit is open."""
        # Open the circuit
        for _ in range(5):
            circuit_breaker.record_failure()

        async def dummy_fn():
            return "should_not_execute"

        with pytest.raises(CircuitOpenError, match="Circuit is open"):
            await guarded_call(dummy_fn, circuit_breaker=circuit_breaker)

    @pytest.mark.asyncio
    async def test_guarded_call_rate_limit_exceeded_raises_error(self, rate_limiter):
        """Test guarded call raises RateLimitExceededError when rate limited."""
        # Use up all tokens
        for _ in range(10):
            rate_limiter.acquire()

        async def dummy_fn():
            return "should_not_execute"

        with pytest.raises(RateLimitExceededError, match="Rate limit exceeded"):
            await guarded_call(dummy_fn, rate_limiter=rate_limiter)

    @pytest.mark.asyncio
    async def test_guarded_call_records_success(self, circuit_breaker):
        """Test guarded call records success with circuit breaker."""

        async def success_fn():
            return "success"

        result = await guarded_call(success_fn, circuit_breaker=circuit_breaker)
        assert result == "success"
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_guarded_call_records_failure(self, circuit_breaker):
        """Test guarded call records failure with circuit breaker."""

        async def failing_fn():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await guarded_call(failing_fn, circuit_breaker=circuit_breaker)

        assert circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_guarded_call_with_args_and_kwargs(self):
        """Test guarded call passes through args and kwargs."""

        async def fn_with_args(*args, **kwargs):
            return {"args": args, "kwargs": kwargs}

        result = await guarded_call(
            fn_with_args, "arg1", "arg2", kwarg1="value1", kwarg2="value2"
        )

        assert result == {
            "args": ("arg1", "arg2"),
            "kwargs": {"kwarg1": "value1", "kwarg2": "value2"},
        }


class TestGlobalInstances:
    """Test global circuit breaker and rate limiter instances."""

    def test_get_default_circuit_breaker_returns_instance(self):
        """Test get_default_circuit_breaker returns an instance."""
        cb = get_default_circuit_breaker()
        assert isinstance(cb, APICircuitBreaker)

    def test_get_default_rate_limiter_returns_instance(self):
        """Test get_default_rate_limiter returns an instance."""
        rl = get_default_rate_limiter()
        assert isinstance(rl, TokenBucketRateLimiter)

    def test_global_instances_are_singletons(self):
        """Test that global instances are singletons."""
        cb1 = get_default_circuit_breaker()
        cb2 = get_default_circuit_breaker()
        assert cb1 is cb2

        rl1 = get_default_rate_limiter()
        rl2 = get_default_rate_limiter()
        assert rl1 is rl2
