"""
tests/test_circuit_retry_integration.py

Tests for circuit breaker and retry manager integration.
Ensures retries respect circuit breaker state and don't bypass protection.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.api_protection import (
    APICircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
)
from core.execution.execution_types import ExecutionPolicy, ExecutionStatus
from core.execution.retry_manager import RetryManager


class TestCircuitBreakerRetryIntegration:
    """Test cases for circuit breaker and retry manager integration."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker for testing."""
        return APICircuitBreaker(CircuitBreakerConfig())

    @pytest.fixture
    def retry_manager(self):
        """Create a retry manager for testing."""
        return RetryManager(
            {"max_retries": 2, "backoff_base": 0.01}
        )  # Fast retries for testing

    @pytest.fixture
    def mock_execution_func(self):
        """Create a mock execution function."""
        return AsyncMock()

    @pytest.fixture
    def mock_signal(self):
        """Create a mock signal."""
        return MagicMock()

    @pytest.fixture
    def mock_context(self):
        """Create a mock context."""
        return {}

    def test_retry_aborts_when_circuit_open_before_first_attempt(
        self,
        retry_manager,
        circuit_breaker,
        mock_execution_func,
        mock_signal,
        mock_context,
    ):
        """Test that retry immediately aborts when circuit is open before first attempt."""
        # Open the circuit
        for _ in range(5):
            circuit_breaker.record_failure()

        # Attempt retry
        with pytest.raises(CircuitOpenError, match="Circuit is open.*aborting retry"):
            asyncio.run(
                retry_manager.execute_with_retry(
                    mock_execution_func,
                    mock_signal,
                    ExecutionPolicy.MARKET,
                    mock_context,
                    circuit_breaker=circuit_breaker,
                )
            )

        # Verify execution function was never called
        mock_execution_func.assert_not_called()

    def test_retry_aborts_when_circuit_opens_during_retry_loop(
        self,
        retry_manager,
        circuit_breaker,
        mock_execution_func,
        mock_signal,
        mock_context,
    ):
        """Test that retry loop aborts immediately when circuit opens during retries."""
        call_count = 0

        async def failing_execution(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # Open circuit on second call
            if call_count == 2:
                for _ in range(5):
                    circuit_breaker.record_failure()

            return {
                "status": ExecutionStatus.FAILED,
                "error_message": "Network error",
                "orders": [],
            }

        # Attempt retry
        with pytest.raises(CircuitOpenError, match="Circuit is open.*aborting retry"):
            asyncio.run(
                retry_manager.execute_with_retry(
                    failing_execution,
                    mock_signal,
                    ExecutionPolicy.MARKET,
                    mock_context,
                    circuit_breaker=circuit_breaker,
                )
            )

        # Should have made exactly 2 calls (first attempt + one retry)
        assert call_count == 2

    def test_retry_continues_normally_when_circuit_closed(
        self,
        retry_manager,
        circuit_breaker,
        mock_execution_func,
        mock_signal,
        mock_context,
    ):
        """Test that retries work normally when circuit remains closed."""
        # Set up successful execution on third attempt
        call_count = 0

        async def eventually_successful_execution(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                return {
                    "status": ExecutionStatus.FAILED,
                    "error_message": "Network error",
                    "orders": [],
                }
            else:
                return {
                    "status": ExecutionStatus.COMPLETED,
                    "orders": [{"id": "test"}],
                    "executed_amount": 1.0,
                }

        result = asyncio.run(
            retry_manager.execute_with_retry(
                eventually_successful_execution,
                mock_signal,
                ExecutionPolicy.MARKET,
                mock_context,
                circuit_breaker=circuit_breaker,
            )
        )

        # Should have succeeded on third attempt
        assert result["status"] == ExecutionStatus.COMPLETED
        assert call_count == 3
        assert circuit_breaker.failure_count == 0  # Reset on success

    def test_circuit_open_error_not_retried(
        self,
        retry_manager,
        circuit_breaker,
        mock_execution_func,
        mock_signal,
        mock_context,
    ):
        """Test that CircuitOpenError exceptions are not retried."""
        call_count = 0

        async def circuit_open_execution(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise CircuitOpenError("Circuit is open")

        # Attempt retry
        with pytest.raises(CircuitOpenError, match="Circuit is open"):
            asyncio.run(
                retry_manager.execute_with_retry(
                    circuit_open_execution,
                    mock_signal,
                    ExecutionPolicy.MARKET,
                    mock_context,
                    circuit_breaker=circuit_breaker,
                )
            )

        # Should have made exactly 1 call (CircuitOpenError prevents retries)
        assert call_count == 1

    def test_retry_records_success_with_circuit_breaker(
        self,
        retry_manager,
        circuit_breaker,
        mock_execution_func,
        mock_signal,
        mock_context,
    ):
        """Test that successful retries record success with circuit breaker."""

        async def successful_execution(*args, **kwargs):
            return {
                "status": ExecutionStatus.COMPLETED,
                "orders": [{"id": "test"}],
                "executed_amount": 1.0,
            }

        result = asyncio.run(
            retry_manager.execute_with_retry(
                successful_execution,
                mock_signal,
                ExecutionPolicy.MARKET,
                mock_context,
                circuit_breaker=circuit_breaker,
            )
        )

        assert result["status"] == ExecutionStatus.COMPLETED
        # Circuit breaker should have recorded success (failure_count reset to 0)
        assert circuit_breaker.failure_count == 0

    def test_retry_records_failure_with_circuit_breaker(
        self,
        retry_manager,
        circuit_breaker,
        mock_execution_func,
        mock_signal,
        mock_context,
    ):
        """Test that failed retries record failure with circuit breaker."""

        async def failing_execution(*args, **kwargs):
            return {
                "status": ExecutionStatus.FAILED,
                "error_message": "Network error",
                "orders": [],
            }

        result = asyncio.run(
            retry_manager.execute_with_retry(
                failing_execution,
                mock_signal,
                ExecutionPolicy.MARKET,
                mock_context,
                circuit_breaker=circuit_breaker,
            )
        )

        assert result["status"] == ExecutionStatus.FAILED
        # Circuit breaker should have recorded failure
        assert circuit_breaker.failure_count == 1

    def test_retry_respects_circuit_breaker_recovery(
        self,
        retry_manager,
        circuit_breaker,
        mock_execution_func,
        mock_signal,
        mock_context,
    ):
        """Test that retries work again after circuit breaker recovery."""
        # First, open the circuit
        for _ in range(5):
            circuit_breaker.record_failure()
        assert circuit_breaker.is_open()

        # Simulate time passing for recovery
        circuit_breaker.last_failure_time = time.time() - 70  # 70 seconds ago

        # Now circuit should allow attempts (half-open)
        assert not circuit_breaker.is_open()

        async def successful_execution(*args, **kwargs):
            return {
                "status": ExecutionStatus.COMPLETED,
                "orders": [{"id": "test"}],
                "executed_amount": 1.0,
            }

        result = asyncio.run(
            retry_manager.execute_with_retry(
                successful_execution,
                mock_signal,
                ExecutionPolicy.MARKET,
                mock_context,
                circuit_breaker=circuit_breaker,
            )
        )

        assert result["status"] == ExecutionStatus.COMPLETED
        # Circuit should still be half-open after 1 success (needs 3 successes to close)
        assert circuit_breaker.get_state() == "half-open"
        assert circuit_breaker.success_count == 1

    def test_retry_without_circuit_breaker_works_normally(
        self, retry_manager, mock_execution_func, mock_signal, mock_context
    ):
        """Test that retry works normally when no circuit breaker is provided."""
        call_count = 0

        async def eventually_successful_execution(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count < 2:
                return {
                    "status": ExecutionStatus.FAILED,
                    "error_message": "Network error",
                    "orders": [],
                }
            else:
                return {
                    "status": ExecutionStatus.COMPLETED,
                    "orders": [{"id": "test"}],
                    "executed_amount": 1.0,
                }

        result = asyncio.run(
            retry_manager.execute_with_retry(
                eventually_successful_execution,
                mock_signal,
                ExecutionPolicy.MARKET,
                mock_context
                # No circuit_breaker parameter
            )
        )

        assert result["status"] == ExecutionStatus.COMPLETED
        assert call_count == 2
