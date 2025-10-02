"""
tests/test_retry_integration.py

Integration tests for centralized retry system with circuit breaker and idempotency.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pytest import MonkeyPatch

from core.retry import retry_call, update_global_retry_config
from core.api_protection import (
    APICircuitBreaker,
    CircuitBreakerConfig,
    get_default_circuit_breaker,
)
from core.idempotency import RetryNotAllowedError
from core.execution.retry_manager import RetryManager
from core.execution.execution_types import ExecutionStatus


class TestRetryIntegration:
    """Integration tests combining retry with circuit breaker and idempotency."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a test circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)
        cb = APICircuitBreaker(config)
        return cb

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker_integration(self, circuit_breaker):
        """Test retry integrates properly with circuit breaker."""
        call_count = 0

        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Network error")  # Should trigger circuit breaker

        # First few calls should fail and eventually open circuit
        for i in range(3):
            with pytest.raises(Exception):
                await retry_call(
                    failing_function,
                    max_attempts=1,  # No retries, just test circuit breaker
                    circuit_breaker=circuit_breaker,
                )

        # Circuit should now be open
        assert circuit_breaker.is_open()

        # Next call should fail fast due to open circuit
        with pytest.raises(Exception, match="Circuit is open"):
            await retry_call(failing_function, circuit_breaker=circuit_breaker)

        assert call_count == 2  # Only called twice before circuit opened

    @pytest.mark.asyncio
    async def test_side_effect_retry_requires_idempotency(self, circuit_breaker):
        """Test that side-effect operations require idempotency key."""

        async def order_function():
            return {"order_id": "123", "status": "filled"}

        # Should fail without idempotency key
        with pytest.raises(RetryNotAllowedError):
            await retry_call(
                order_function, is_side_effect=True, circuit_breaker=circuit_breaker
            )

        # Should work with idempotency key
        result = await retry_call(
            order_function,
            idempotency_key="order_123",
            is_side_effect=True,
            circuit_breaker=circuit_breaker,
        )

        assert result["order_id"] == "123"

    @pytest.mark.asyncio
    async def test_global_config_changes_retry_behavior(self):
        """Test that global config changes affect retry behavior."""
        # Set restrictive global config
        update_global_retry_config(
            {
                "max_attempts": 2,
                "base_delay": 0.01,
            }
        )

        try:
            call_count = 0

            async def failing_function():
                nonlocal call_count
                call_count += 1
                raise ValueError("Always fails")

            start_time = time.time()
            with pytest.raises(ValueError):
                await retry_call(failing_function)  # Uses global config

            elapsed = time.time() - start_time

            assert call_count == 2  # Should use global max_attempts=2
            assert (
                elapsed < 0.5
            )  # Should use global base_delay=0.01 with some tolerance

        finally:
            # Reset global config
            update_global_retry_config(
                {
                    "max_attempts": 3,
                    "base_delay": 0.5,
                }
            )

    @pytest.mark.asyncio
    async def test_backoff_timing_observation(self):
        """Test that backoff timing can be observed and increases per attempt."""
        delays = []

        async def mock_sleep(delay):
            delays.append(delay)
            # Don't call asyncio.sleep to avoid recursion with MonkeyPatch

        call_count = 0

        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Attempt {call_count} failed")

        with pytest.raises(ValueError, match="Attempt 4 failed"):
            m = MonkeyPatch()
            m.setattr(asyncio, "sleep", mock_sleep)
            await retry_call(
                failing_function, max_attempts=4, base_delay=0.1, jitter=0.01
            )

        # Should have 3 delays (between attempts 1-2, 2-3, 3-4)
        assert len(delays) == 3
        assert call_count == 4

        # Verify ascending delays (exponential backoff)
        assert delays[0] <= delays[1] <= delays[2]

        # First delay should be around base_delay + jitter
        assert 0.09 <= delays[0] <= 0.12

        # Second delay should be around 2 * base_delay + jitter
        assert 0.19 <= delays[1] <= 0.22

        # Third delay should be around 4 * base_delay + jitter
        assert 0.39 <= delays[2] <= 0.42

    @pytest.mark.asyncio
    async def test_retry_manager_delegates_to_centralized_retry(self):
        """Test that RetryManager.retry_call_centralized delegates to core.retry.retry_call."""
        manager = RetryManager(
            {
                "max_retries": 2,
                "backoff_base": 0.1,
            }
        )

        call_count = 0

        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        # Should succeed on 3rd attempt
        result = await manager.retry_call_centralized(
            failing_function, idempotency_key="test_key", is_side_effect=True
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_recording(self, circuit_breaker):
        """Test that circuit breaker records failures properly."""

        async def failing_function():
            raise Exception("Test failure")

        # Make multiple failing calls
        for i in range(3):
            with pytest.raises(Exception):
                await retry_call(
                    failing_function,
                    max_attempts=1,  # No retries
                    circuit_breaker=circuit_breaker,
                )

        # Circuit should be open after failures
        assert circuit_breaker.is_open()

    @pytest.mark.asyncio
    async def test_mixed_sync_async_functions_in_retry(self):
        """Test that retry works with both sync and async functions in same test suite."""

        async def async_func():
            await asyncio.sleep(0.001)
            return "async"

        def sync_func():
            time.sleep(0.001)
            return "sync"

        # Both should work
        async_result = await retry_call(async_func, max_attempts=1)
        sync_result = await retry_call(sync_func, max_attempts=1)

        assert async_result == "async"
        assert sync_result == "sync"

    @pytest.mark.asyncio
    async def test_retry_preserves_function_arguments(self):
        """Test that retry preserves all function arguments correctly."""

        async def complex_function(a, b=None, *args, **kwargs):
            return {"a": a, "b": b, "args": args, "kwargs": kwargs}

        # Test with simple arguments to avoid parameter conflicts
        result = await retry_call(
            complex_function, "test_a", b="test_b", key1="value1", max_attempts=1
        )

        assert result["a"] == "test_a"
        assert result["b"] == "test_b"
        assert result["args"] == ()
        assert result["kwargs"] == {"key1": "value1"}

    @pytest.mark.asyncio
    async def test_retry_with_real_world_delays(self):
        """Test retry with realistic delays that would be used in production."""
        delays = []

        async def mock_sleep(delay):
            delays.append(delay)
            # Don't call asyncio.sleep to avoid recursion with MonkeyPatch

        call_count = 0

        async def network_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 attempts
                raise ConnectionError("Network timeout")
            return "connected"

        m = MonkeyPatch()
        m.setattr(asyncio, "sleep", mock_sleep)

        result = await retry_call(
            network_function,
            max_attempts=3,
            base_delay=1.0,  # 1 second base delay
            jitter=0.5,  # 0.5 second jitter
        )

        assert result == "connected"
        assert call_count == 3
        assert len(delays) == 2  # 2 delays between 3 attempts

        # Check delays are in reasonable range
        for delay in delays:
            assert 0.5 <= delay <= 3.0  # base_delay + jitter range


class TestRetryManagerIntegration:
    """Test integration with the existing RetryManager."""

    @pytest.fixture
    def retry_manager(self):
        """Create a test retry manager."""
        return RetryManager(
            {
                "max_retries": 2,
                "backoff_base": 0.1,
            }
        )

    @pytest.mark.asyncio
    async def test_retry_manager_config_updates_global_config(self, retry_manager):
        """Test that RetryManager config updates affect global retry config."""
        from core.retry import get_global_retry_config

        # Update manager config
        retry_manager.update_retry_config(
            {
                "max_retries": 5,
                "backoff_base": 2.0,
            }
        )

        # Global config should be updated
        global_config = get_global_retry_config()
        assert global_config.max_attempts == 6  # max_retries + 1
        assert global_config.base_delay == 2.0

    @pytest.mark.asyncio
    async def test_retry_manager_backward_compatibility(self, retry_manager):
        """Test that old RetryManager methods still work."""
        # Old method should still work (though deprecated)
        call_count = 0

        async def test_func(signal, policy, context):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return {"status": "failed", "error_message": "Network timeout"}
            return {"status": "completed", "orders": [{"id": "123"}]}

        # This should work but use the old complex logic
        from core.execution.execution_types import ExecutionPolicy

        result = await retry_manager.execute_with_retry(
            test_func,
            None,  # signal
            ExecutionPolicy.MARKET,  # policy
            {},  # context
            allow_side_effect_retry=True,
            idempotency_key="test",
        )

        assert result["status"] == ExecutionStatus.COMPLETED
        assert call_count >= 1


class TestErrorScenarios:
    """Test various error scenarios in retry integration."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_calls_when_open(self):
        """Test that circuit breaker blocks calls when open."""
        config = CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=1.0  # Long timeout for test
        )
        cb = APICircuitBreaker(config)

        call_count = 0

        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fails")

        # First 2 calls fail, opening circuit
        for i in range(2):
            with pytest.raises(Exception):
                await retry_call(failing_function, max_attempts=1, circuit_breaker=cb)

        assert cb.is_open()

        # Subsequent calls should be blocked
        with pytest.raises(Exception, match="Circuit is open"):
            await retry_call(failing_function, max_attempts=1, circuit_breaker=cb)

        # Only the first 2 calls should have been made
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_nested_retry_calls_allowed(self):
        """Test that nesting retry calls works properly."""

        async def inner_function():
            return "inner_result"

        async def outer_function():
            # This should work fine - nested retry calls are allowed
            return await retry_call(inner_function, max_attempts=1)

        result = await retry_call(outer_function, max_attempts=1)

        assert result == "inner_result"
