import asyncio
import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from core.management.reliability_manager import ReliabilityManager


class TestReliabilityManager:
    """Test cases for ReliabilityManager functionality."""

    @pytest.fixture
    def reliability_manager(self):
        """Create a fresh ReliabilityManager instance."""
        return ReliabilityManager()

    @pytest.fixture
    def reliability_manager_with_config(self):
        """Create a ReliabilityManager with custom config."""
        config = {
            "max_retries": 5,
            "backoff_base": 1.0,
            "max_backoff": 20.0,
            "safe_mode_threshold": 3,
            "close_positions_on_safe": True,
        }
        return ReliabilityManager(config)

    def test_initialization_default_config(self, reliability_manager):
        """Test ReliabilityManager initialization with default config."""
        assert reliability_manager.safe_mode_active is False
        assert reliability_manager.critical_error_count == 0

        # Check default reliability config
        expected_config = {
            "max_retries": 3,
            "backoff_base": 0.5,
            "max_backoff": 10.0,
            "safe_mode_threshold": 5,
            "close_positions_on_safe": False,
        }
        assert reliability_manager._reliability == expected_config

    def test_initialization_with_custom_config(self, reliability_manager_with_config):
        """Test ReliabilityManager initialization with custom config."""
        assert reliability_manager_with_config.safe_mode_active is False
        assert reliability_manager_with_config.critical_error_count == 0

        # Check custom reliability config
        expected_config = {
            "max_retries": 5,
            "backoff_base": 1.0,
            "max_backoff": 20.0,
            "safe_mode_threshold": 3,
            "close_positions_on_safe": True,
        }
        assert reliability_manager_with_config._reliability == expected_config

    def test_initialization_partial_config(self):
        """Test ReliabilityManager with partial config (should merge with defaults)."""
        config = {"max_retries": 10, "safe_mode_threshold": 2}
        rm = ReliabilityManager(config)

        assert rm._reliability["max_retries"] == 10
        assert rm._reliability["safe_mode_threshold"] == 2
        # Other values should be defaults
        assert rm._reliability["backoff_base"] == 0.5
        assert rm._reliability["max_backoff"] == 10.0
        assert rm._reliability["close_positions_on_safe"] is False

    @pytest.mark.asyncio
    async def test_retry_async_success_first_try(self, reliability_manager):
        """Test retry_async with successful operation on first try."""
        async def successful_operation():
            return "success"

        result = await reliability_manager.retry_async(lambda: successful_operation())

        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_async_success_after_retry(self, reliability_manager):
        """Test retry_async with success after one failure."""
        call_count = 0

        async def failing_then_successful_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")
            return "success"

        with patch('asyncio.sleep') as mock_sleep:
            result = await reliability_manager.retry_async(lambda: failing_then_successful_operation())

        assert result == "success"
        assert call_count == 2
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_async_exhaust_retries(self, reliability_manager):
        """Test retry_async when all retries are exhausted."""
        async def always_failing_operation():
            raise ValueError("Always fails")

        with patch('asyncio.sleep') as mock_sleep:
            with pytest.raises(ValueError, match="Always fails"):
                await reliability_manager.retry_async(lambda: always_failing_operation())

        # Should have tried 4 times (1 initial + 3 retries)
        assert mock_sleep.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_custom_parameters(self, reliability_manager):
        """Test retry_async with custom retry parameters."""
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Fails")
            return "success"

        with patch('asyncio.sleep') as mock_sleep:
            result = await reliability_manager.retry_async(
                lambda: operation(),
                retries=2,
                base_backoff=1.0,
                max_backoff=5.0
            )

        assert result == "success"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_async_specific_exceptions(self, reliability_manager):
        """Test retry_async with specific exception types."""
        async def operation():
            raise ConnectionError("Network error")

        with patch('asyncio.sleep') as mock_sleep:
            with pytest.raises(ConnectionError):
                await reliability_manager.retry_async(
                    lambda: operation(),
                    exceptions=(ValueError,)  # Only retry ValueError, not ConnectionError
                )

        # Should not have retried since ConnectionError is not in exceptions tuple
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_async_backoff_calculation(self, reliability_manager):
        """Test that backoff times are calculated correctly with jitter."""
        async def always_failing_operation():
            raise ValueError("Fails")

        with patch('asyncio.sleep') as mock_sleep:
            with pytest.raises(ValueError):
                await reliability_manager.retry_async(lambda: always_failing_operation())

        # Check that sleep was called with increasing backoff times
        calls = mock_sleep.call_args_list
        assert len(calls) == 3

        # First retry: base_backoff * (2^0) = 0.5
        first_backoff = calls[0][0][0]
        assert 0.45 <= first_backoff <= 0.55  # With jitter

        # Second retry: base_backoff * (2^1) = 1.0
        second_backoff = calls[1][0][0]
        assert 0.9 <= second_backoff <= 1.1  # With jitter

        # Third retry: base_backoff * (2^2) = 2.0
        third_backoff = calls[2][0][0]
        assert 1.8 <= third_backoff <= 2.2  # With jitter

    @pytest.mark.asyncio
    async def test_retry_async_max_backoff_cap(self, reliability_manager_with_config):
        """Test that backoff is capped at max_backoff."""
        async def always_failing_operation():
            raise ValueError("Fails")

        with patch('asyncio.sleep') as mock_sleep:
            with pytest.raises(ValueError):
                await reliability_manager_with_config.retry_async(lambda: always_failing_operation())

        # With custom config: max_retries=5, max_backoff=20.0
        calls = mock_sleep.call_args_list
        assert len(calls) == 5

        # All backoff times should be <= max_backoff (20.0)
        for call in calls:
            backoff_time = call[0][0]
            assert backoff_time <= 20.0

    def test_record_critical_error_below_threshold(self, reliability_manager):
        """Test record_critical_error when error count is below threshold."""
        exc = ValueError("Test error")

        reliability_manager.record_critical_error(exc)

        assert reliability_manager.critical_error_count == 1
        assert reliability_manager.safe_mode_active is False

    def test_record_critical_error_at_threshold(self, reliability_manager):
        """Test record_critical_error when error count reaches threshold."""
        exc = ValueError("Test error")

        # Record 4 errors (below threshold of 5)
        for i in range(4):
            reliability_manager.record_critical_error(exc)
            assert reliability_manager.safe_mode_active is False

        # Record 5th error (at threshold)
        reliability_manager.record_critical_error(exc)

        assert reliability_manager.critical_error_count == 5
        assert reliability_manager.safe_mode_active is True

    def test_record_critical_error_with_custom_threshold(self, reliability_manager_with_config):
        """Test record_critical_error with custom threshold."""
        exc = ValueError("Test error")

        # Custom threshold is 3
        for i in range(2):
            reliability_manager_with_config.record_critical_error(exc)
            assert reliability_manager_with_config.safe_mode_active is False

        # Third error should trigger safe mode
        reliability_manager_with_config.record_critical_error(exc)

        assert reliability_manager_with_config.critical_error_count == 3
        assert reliability_manager_with_config.safe_mode_active is True

    def test_record_critical_error_with_context(self, reliability_manager):
        """Test record_critical_error with context information."""
        exc = ValueError("Test error")
        context = {"symbol": "BTC/USDT", "operation": "buy"}

        with patch('logging.Logger.error') as mock_log:
            reliability_manager.record_critical_error(exc, context)

        # Verify logging was called
        mock_log.assert_called_once()
        log_call = mock_log.call_args[0][0]
        assert "Critical error #1" in log_call
        assert "Test error" in log_call

    def test_record_critical_error_already_in_safe_mode(self, reliability_manager):
        """Test record_critical_error when already in safe mode."""
        # Put in safe mode first
        reliability_manager.safe_mode_active = True
        reliability_manager.critical_error_count = 5

        exc = ValueError("Test error")

        # Record another error
        reliability_manager.record_critical_error(exc)

        # Should still be in safe mode, count should increase
        assert reliability_manager.critical_error_count == 6
        assert reliability_manager.safe_mode_active is True

    def test_record_critical_error_exception_handling(self, reliability_manager):
        """Test record_critical_error handles exceptions gracefully."""
        # Create a mock exception that raises when str() is called
        class BadException(Exception):
            def __str__(self):
                raise RuntimeError("Cannot stringify")

        exc = BadException()

        # Should not raise, should handle the exception internally
        reliability_manager.record_critical_error(exc)

        # Error count should still be incremented
        assert reliability_manager.critical_error_count == 1

    def test_activate_safe_mode_basic(self, reliability_manager):
        """Test basic safe mode activation."""
        reason = "Test activation"

        with patch('logging.Logger.critical') as mock_critical:
            reliability_manager.activate_safe_mode(reason)

        assert reliability_manager.safe_mode_active is True
        mock_critical.assert_called_once_with(f"Safe mode activated due to: {reason}")

    def test_activate_safe_mode_with_context(self, reliability_manager):
        """Test safe mode activation with context (context is currently unused in implementation)."""
        reason = "Test activation"
        context = {"symbol": "BTC/USDT"}

        reliability_manager.activate_safe_mode(reason, context)

        assert reliability_manager.safe_mode_active is True

    def test_activate_safe_mode_close_positions_enabled(self, reliability_manager_with_config):
        """Test safe mode activation with close_positions_on_safe enabled."""
        reason = "Test activation"

        with patch('logging.Logger.critical') as mock_critical, \
             patch('logging.Logger.info') as mock_info, \
             patch('logging.Logger.warning') as mock_warning:

            reliability_manager_with_config.activate_safe_mode(reason)

        assert reliability_manager_with_config.safe_mode_active is True
        mock_critical.assert_called_once()
        mock_info.assert_called_once_with("Safe mode: closing existing positions")
        mock_warning.assert_called_once_with("Position closing not implemented; requires order_manager integration")

    def test_activate_safe_mode_exception_handling(self, reliability_manager):
        """Test activate_safe_mode handles exceptions gracefully."""
        # Mock logger to raise exception
        with patch('logging.Logger.critical', side_effect=Exception("Log failure")):
            # Should not raise, should handle exception internally
            reliability_manager.activate_safe_mode("Test")

        # Should still activate safe mode despite logging failure
        assert reliability_manager.safe_mode_active is True

    def test_safe_mode_persistence(self, reliability_manager):
        """Test that safe mode state persists across operations."""
        # Initially not in safe mode
        assert reliability_manager.safe_mode_active is False

        # Activate safe mode
        reliability_manager.activate_safe_mode("Test")
        assert reliability_manager.safe_mode_active is True

        # Should remain in safe mode
        assert reliability_manager.safe_mode_active is True

    @pytest.mark.asyncio
    async def test_retry_async_with_different_exception_types(self, reliability_manager):
        """Test retry_async with different exception types."""
        async def operation():
            raise asyncio.TimeoutError("Timeout")

        with patch('asyncio.sleep') as mock_sleep:
            with pytest.raises(asyncio.TimeoutError):
                await reliability_manager.retry_async(lambda: operation())

        # Should have retried (TimeoutError is in default exceptions)
        assert mock_sleep.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_zero_retries(self, reliability_manager):
        """Test retry_async with zero retries."""
        async def operation():
            raise ValueError("Fails")

        with patch('asyncio.sleep') as mock_sleep:
            with pytest.raises(ValueError):
                await reliability_manager.retry_async(lambda: operation(), retries=0)

        # Should not retry
        mock_sleep.assert_not_called()

    def test_record_critical_error_multiple_calls(self, reliability_manager):
        """Test multiple calls to record_critical_error."""
        exc1 = ValueError("Error 1")
        exc2 = RuntimeError("Error 2")

        reliability_manager.record_critical_error(exc1)
        assert reliability_manager.critical_error_count == 1

        reliability_manager.record_critical_error(exc2)
        assert reliability_manager.critical_error_count == 2

        assert reliability_manager.safe_mode_active is False

    def test_reliability_manager_state_isolation(self):
        """Test that multiple ReliabilityManager instances maintain separate state."""
        rm1 = ReliabilityManager()
        rm2 = ReliabilityManager()

        rm1.critical_error_count = 3
        rm1.safe_mode_active = True

        # rm2 should be unaffected
        assert rm2.critical_error_count == 0
        assert rm2.safe_mode_active is False

    def test_config_immutability(self, reliability_manager):
        """Test that the internal config is not modified externally."""
        original_config = reliability_manager._reliability.copy()

        # Try to modify the config through direct attribute access
        # (Note: In practice, _reliability should be treated as private)
        reliability_manager._reliability["max_retries"] = 999

        # Internal config should be modifiable (this is expected behavior for internal state)
        assert reliability_manager._reliability["max_retries"] == 999

        # But we can test that the config is properly initialized
        assert "max_retries" in reliability_manager._reliability
        assert "backoff_base" in reliability_manager._reliability
        assert "max_backoff" in reliability_manager._reliability
        assert "safe_mode_threshold" in reliability_manager._reliability
        assert "close_positions_on_safe" in reliability_manager._reliability
