"""
tests/test_retry.py

Tests for centralized retry logic with idempotency-first design.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.retry import retry_call, retry_call_sync, RetryConfig, get_global_retry_config, update_global_retry_config
from core.api_protection import APICircuitBreaker, CircuitOpenError
from core.idempotency import RetryNotAllowedError


class TestRetryCall:
    """Test cases for the centralized retry_call function."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a test circuit breaker."""
        cb = MagicMock(spec=APICircuitBreaker)
        cb.is_open.return_value = False
        return cb

    @pytest.mark.asyncio
    async def test_safe_function_retries_with_max_attempts(self, circuit_breaker):
        """Test that safe function retries exactly max_attempts times."""
        call_count = 0

        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 attempts
                raise ValueError("Temporary failure")
            return "success"

        start_time = time.time()
        result = await retry_call(
            failing_function,
            max_attempts=3,
            base_delay=0.01,  # Very short delay for test
            circuit_breaker=circuit_breaker
        )
        elapsed = time.time() - start_time

        assert result == "success"
        assert call_count == 3  # Should be called exactly 3 times
        assert elapsed >= 0.02  # Should have delays between retries

    @pytest.mark.asyncio
    async def test_side_effect_function_without_idempotency_raises_error(self, circuit_breaker):
        """Test that side-effect function without idempotency_key raises RetryNotAllowedError."""
        async def side_effect_function():
            return "success"

        with pytest.raises(RetryNotAllowedError, match="Retry not allowed.*without idempotency_key"):
            await retry_call(
                side_effect_function,
                is_side_effect=True,
                circuit_breaker=circuit_breaker
            )

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_aborts_retry(self, circuit_breaker):
        """Test that circuit breaker open prevents retry."""
        circuit_breaker.is_open.return_value = True

        async def failing_function():
            raise ValueError("Should not be called")

        with pytest.raises(CircuitOpenError, match="Circuit is open"):
            await retry_call(
                failing_function,
                circuit_breaker=circuit_breaker
            )

    @pytest.mark.asyncio
    async def test_exponential_backoff_with_jitter(self):
        """Test that backoff timing increases exponentially with jitter."""
        # Test that the delay calculation works correctly
        from core.retry import _calculate_delay

        # Test various attempts
        delay1 = _calculate_delay(0, 0.1, 0.05, 30.0)  # First retry
        delay2 = _calculate_delay(1, 0.1, 0.05, 30.0)  # Second retry
        delay3 = _calculate_delay(2, 0.1, 0.05, 30.0)  # Third retry

        # Check exponential growth (allowing for jitter)
        assert 0.05 <= delay1 <= 0.25  # base_delay * 2^0 + jitter
        assert 0.15 <= delay2 <= 0.55  # base_delay * 2^1 + jitter
        assert 0.35 <= delay3 <= 1.05  # base_delay * 2^2 + jitter

        # Each delay should be roughly double the previous (accounting for jitter)
        assert delay2 >= delay1 * 1.5  # At least 1.5x growth
        assert delay3 >= delay2 * 1.5  # At least 1.5x growth

    @pytest.mark.asyncio
    async def test_sync_function_wrapped_in_thread_pool(self):
        """Test that sync functions are properly wrapped in thread pool."""
        def sync_function(x, y):
            time.sleep(0.01)  # Blocking sleep
            return x + y

        result = await retry_call_sync(
            sync_function,
            5, 3,
            max_attempts=1  # No retries for this test
        )

        assert result == 8

    @pytest.mark.asyncio
    async def test_global_config_defaults(self):
        """Test that global config provides sensible defaults."""
        config = get_global_retry_config()

        assert config.max_attempts == 3
        assert config.base_delay == 0.5
        assert config.jitter == 0.1
        assert config.max_delay == 30.0

    @pytest.mark.asyncio
    async def test_global_config_update(self):
        """Test that global config can be updated."""
        original_config = get_global_retry_config()

        # Update config
        update_global_retry_config({
            "max_attempts": 5,
            "base_delay": 1.0,
            "jitter": 0.2
        })

        updated_config = get_global_retry_config()
        assert updated_config.max_attempts == 5
        assert updated_config.base_delay == 1.0
        assert updated_config.jitter == 0.2
        assert updated_config.max_delay == 30.0  # Unchanged

        # Reset to original
        update_global_retry_config({
            "max_attempts": original_config.max_attempts,
            "base_delay": original_config.base_delay,
            "jitter": original_config.jitter
        })

    @pytest.mark.asyncio
    async def test_uses_global_config_when_params_none(self):
        """Test that global config is used when parameters are None."""
        # Update global config
        update_global_retry_config({
            "max_attempts": 2,
            "base_delay": 0.01,
        })

        try:
            call_count = 0

            async def failing_function():
                nonlocal call_count
                call_count += 1
                raise ValueError("Always fails")

            with pytest.raises(ValueError):
                await retry_call(failing_function)  # No explicit params

            assert call_count == 2  # Should use global max_attempts=2

        finally:
            # Reset global config
            update_global_retry_config({
                "max_attempts": 3,
                "base_delay": 0.5,
            })

    @pytest.mark.asyncio
    async def test_explicit_params_override_global_config(self):
        """Test that explicit parameters override global config."""
        # Set global config
        update_global_retry_config({
            "max_attempts": 5,
            "base_delay": 1.0,
        })

        try:
            call_count = 0

            async def failing_function():
                nonlocal call_count
                call_count += 1
                raise ValueError("Always fails")

            with pytest.raises(ValueError):
                await retry_call(
                    failing_function,
                    max_attempts=2,  # Override global
                    base_delay=0.01
                )

            assert call_count == 2  # Should use explicit max_attempts=2

        finally:
            # Reset global config
            update_global_retry_config({
                "max_attempts": 3,
                "base_delay": 0.5,
            })

    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        from core.retry import _calculate_delay

        # Test that delay calculation respects max_delay
        delay = _calculate_delay(10, 10.0, 1.0, 5.0)  # High base delay, low cap
        assert delay <= 5.0  # Should be capped at max_delay

    @pytest.mark.asyncio
    async def test_async_and_sync_functions_both_supported(self):
        """Test that both async and sync functions work."""
        async def async_func():
            return "async_result"

        def sync_func():
            return "sync_result"

        async_result = await retry_call(async_func, max_attempts=1)
        sync_result = await retry_call(sync_func, max_attempts=1)

        assert async_result == "async_result"
        assert sync_result == "sync_result"


class TestRetryConfig:
    """Test cases for RetryConfig class."""

    def test_config_initialization(self):
        """Test RetryConfig initialization."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 0.5
        assert config.jitter == 0.1
        assert config.max_delay == 30.0

    def test_config_update(self):
        """Test RetryConfig update."""
        config = RetryConfig()

        config.update_from_config({
            "max_attempts": 5,
            "base_delay": 1.0,
            "jitter": 0.2,
            "max_delay": 60.0
        })

        assert config.max_attempts == 5
        assert config.base_delay == 1.0
        assert config.jitter == 0.2
        assert config.max_delay == 60.0

    def test_config_partial_update(self):
        """Test partial config update."""
        config = RetryConfig()

        config.update_from_config({
            "max_attempts": 10,
            # Other fields should keep defaults
        })

        assert config.max_attempts == 10
        assert config.base_delay == 0.5  # Unchanged
        assert config.jitter == 0.1      # Unchanged


class TestBackwardsCompatibility:
    """Test backwards compatibility functions."""

    @pytest.mark.asyncio
    async def test_deprecated_functions_still_work(self):
        """Test that deprecated functions still work but log warnings."""
        from core.retry import retry_with_backoff, exponential_backoff

        async def test_func():
            return "success"

        # These should work but log warnings
        result1 = await retry_with_backoff(test_func)
        result2 = await exponential_backoff(test_func)

        assert result1 == "success"
        assert result2 == "success"
