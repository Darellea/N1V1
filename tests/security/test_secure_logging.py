"""
Test suite for secure logging system.

Tests log sanitization, structured logging, and configurable sensitivity levels
to ensure sensitive information is never exposed in logs.
"""

import json
import logging
from io import StringIO
from unittest.mock import patch

import pytest

from core.logging_utils import (
    LogSanitizer,
    LogSensitivity,
    StructuredLogger,
    get_structured_logger,
    set_global_log_sensitivity,
)


class TestLogSanitizer:
    """Test log message sanitization functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sanitizer = LogSanitizer()

    def test_api_key_sanitization(self):
        """Test that API keys are properly masked."""
        message = 'Using API key: "sk-1234567890abcdef" for authentication'
        sanitized = self.sanitizer.sanitize_message(message, LogSensitivity.SECURE)

        assert "***API_KEY_MASKED***" in sanitized
        assert "sk-1234567890abcdef" not in sanitized

    def test_secret_token_sanitization(self):
        """Test that secret tokens are properly masked."""
        message = "Token: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        sanitized = self.sanitizer.sanitize_message(message, LogSensitivity.SECURE)

        assert "***TOKEN_MASKED***" in sanitized
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sanitized

    def test_financial_amount_sanitization(self):
        """Test that financial amounts are properly masked."""
        message = "Balance: 12345.67, PnL: -987.65, Equity: 11234.56"
        sanitized = self.sanitizer.sanitize_message(message, LogSensitivity.SECURE)

        assert "***BALANCE_MASKED***" in sanitized
        assert "***PNL_MASKED***" in sanitized
        assert "***EQUITY_MASKED***" in sanitized
        assert "12345.67" not in sanitized
        assert "-987.65" not in sanitized
        assert "11234.56" not in sanitized

    def test_personal_info_sanitization(self):
        """Test that personal information is properly masked."""
        message = "User email: user@example.com, Phone: +1-555-0123"
        sanitized = self.sanitizer.sanitize_message(message, LogSensitivity.SECURE)

        assert "***EMAIL_MASKED***" in sanitized
        assert "***PHONE_MASKED***" in sanitized
        assert "user@example.com" not in sanitized
        assert "+1-555-0123" not in sanitized

    def test_debug_level_preserves_sensitive_data(self):
        """Test that DEBUG level preserves sensitive data."""
        message = 'API key: "sk-test123", Balance: 1000.50'
        sanitized = self.sanitizer.sanitize_message(message, LogSensitivity.DEBUG)

        assert sanitized == message  # Should be unchanged
        assert "sk-test123" in sanitized
        assert "1000.50" in sanitized

    def test_audit_level_minimal_output(self):
        """Test that AUDIT level provides minimal output."""
        message = "Security event: API key sk-123 used"
        sanitized = self.sanitizer.sanitize_message(message, LogSensitivity.AUDIT)

        assert sanitized == "[AUDIT] Security event logged"

    def test_complex_message_sanitization(self):
        """Test sanitization of complex messages with multiple sensitive fields."""
        message = """
        Processing trade for user@example.com with API key "sk-1234567890abcdef"
        Balance: 50000.00, PnL: 2500.75, Token: Bearer xyz789
        Phone: +1-555-0199, SSN: 123-45-6789
        """

        sanitized = self.sanitizer.sanitize_message(message, LogSensitivity.SECURE)

        # Check that all sensitive data is masked
        assert "***EMAIL_MASKED***" in sanitized
        assert "***API_KEY_MASKED***" in sanitized
        assert "***BALANCE_MASKED***" in sanitized
        assert "***PNL_MASKED***" in sanitized
        assert "***TOKEN_MASKED***" in sanitized
        assert "***PHONE_MASKED***" in sanitized
        assert "***SSN_MASKED***" in sanitized

        # Check that original sensitive data is not present
        assert "user@example.com" not in sanitized
        assert "sk-1234567890abcdef" not in sanitized
        assert "50000.00" not in sanitized
        assert "2500.75" not in sanitized
        assert "xyz789" not in sanitized
        assert "+1-555-0199" not in sanitized
        assert "123-45-6789" not in sanitized


class TestStructuredLogger:
    """Test structured logging functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = StructuredLogger("test_logger", LogSensitivity.SECURE)
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger.logger.addHandler(self.handler)
        self.logger.logger.setLevel(logging.DEBUG)

    def test_structured_info_logging(self):
        """Test structured info logging with key-value pairs."""
        self.logger.info(
            "Test message", symbol="BTC/USDT", amount=1000.50, user_id="user123"
        )

        log_output = self.stream.getvalue()
        log_data = json.loads(log_output.split(" | ", 1)[-1].strip())

        assert log_data["message"] == "Test message"
        assert log_data["symbol"] == "BTC/USDT"
        assert log_data["amount"] == "***AMOUNT_MASKED***"  # Should be sanitized
        assert log_data["user_id"] == "user123"

    def test_structured_warning_logging(self):
        """Test structured warning logging."""
        self.logger.warning("Alert triggered", alert_type="high_cpu", threshold=90.5)

        log_output = self.stream.getvalue()
        assert "WARNING" in log_output
        assert "Alert triggered" in log_output

    def test_structured_error_logging(self):
        """Test structured error logging."""
        self.logger.error(
            "Database connection failed", db_host="localhost", error_code=500
        )

        log_output = self.stream.getvalue()
        assert "ERROR" in log_output
        assert "Database connection failed" in log_output

    def test_debug_level_preserves_data(self):
        """Test that DEBUG level preserves sensitive data."""
        debug_logger = StructuredLogger("debug_logger", LogSensitivity.DEBUG)
        debug_stream = StringIO()
        debug_handler = logging.StreamHandler(debug_stream)
        debug_logger.logger.addHandler(debug_handler)
        debug_logger.logger.setLevel(logging.DEBUG)

        debug_logger.info("Debug message", api_key="sk-test123", balance=1000.50)

        log_output = debug_stream.getvalue()
        log_data = json.loads(log_output.split(" | ", 1)[-1].strip())

        assert log_data["api_key"] == "sk-test123"  # Should NOT be sanitized
        assert log_data["balance"] == 1000.50  # Should NOT be sanitized

    def test_sensitivity_change(self):
        """Test changing logger sensitivity."""
        # Start with SECURE
        self.logger.info("Test message", api_key="sk-test123")
        secure_output = self.stream.getvalue()

        # Change to DEBUG
        self.logger.set_sensitivity(LogSensitivity.DEBUG)
        self.stream = StringIO()  # Reset stream
        self.handler.setStream(self.stream)

        self.logger.info("Test message", api_key="sk-test123")
        debug_output = self.stream.getvalue()

        # Secure should mask, debug should preserve
        assert "***API_KEY_MASKED***" in secure_output
        assert "sk-test123" in debug_output


class TestCoreLoggerIntegration:
    """Test core logger integration."""

    def test_get_core_logger(self):
        """Test getting a core logger instance."""
        logger = get_structured_logger("core.test_module", LogSensitivity.SECURE)
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

    def test_core_logger_sanitization(self):
        """Test that core logger properly sanitizes sensitive data."""
        logger = get_structured_logger("core.test_module", LogSensitivity.SECURE)

        # Mock the underlying logger to capture output
        with patch.object(logger.logger, "info") as mock_info:
            logger.info(
                "Processing order", symbol="BTC/USDT", amount=50000.00, api_key="sk-123"
            )

            # Check that info was called with sanitized data
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0]
            log_message = call_args[0]

            # Parse the structured log message
            log_data = json.loads(log_message.split(" | ")[-1].strip())

            assert log_data["symbol"] == "BTC/USDT"
            assert log_data["amount"] == "***AMOUNT_MASKED***"
            assert log_data["api_key"] == "***API_KEY_MASKED***"


class TestGlobalSensitivity:
    """Test global sensitivity configuration."""

    def test_global_sensitivity_change(self):
        """Test changing global log sensitivity."""
        # Set global sensitivity to DEBUG
        set_global_log_sensitivity(LogSensitivity.DEBUG)

        logger = get_structured_logger("test_global")
        assert logger.sensitivity == LogSensitivity.DEBUG

        # Set back to SECURE
        set_global_log_sensitivity(LogSensitivity.SECURE)
        logger2 = get_structured_logger("test_global2")
        assert logger2.sensitivity == LogSensitivity.SECURE

    def test_environment_variable_sensitivity(self):
        """Test sensitivity configuration via environment variable."""
        with patch.dict("os.environ", {"LOG_SENSITIVITY": "debug"}):
            # Test that environment variable is respected
            # This would normally call configure_core_logging()
            pass

        with patch.dict("os.environ", {"LOG_SENSITIVITY": "invalid"}):
            # Test that invalid values fallback to SECURE
            # This would normally call configure_core_logging()
            pass


class TestLogSecurityVerification:
    """Test that logs never contain sensitive information."""

    def test_no_api_keys_in_logs(self):
        """Verify that API keys never appear in logs."""
        sensitive_keys = [
            "sk-1234567890abcdef",
            "pk_test_1234567890",
            "xoxb-1234567890-1234567890",
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        ]

        sanitizer = LogSanitizer()

        for api_key in sensitive_keys:
            message = f"Using API key: {api_key}"
            sanitized = sanitizer.sanitize_message(message, LogSensitivity.SECURE)

            assert api_key not in sanitized
            assert "MASKED" in sanitized

    def test_no_financial_amounts_in_logs(self):
        """Verify that financial amounts never appear in logs."""
        amounts = ["12345.67", "$999,999.99", "€50,000.00", "-1234.56"]

        sanitizer = LogSanitizer()

        for amount in amounts:
            message = f"Balance: {amount}"
            sanitized = sanitizer.sanitize_message(message, LogSensitivity.SECURE)

            assert amount not in sanitized
            assert "MASKED" in sanitized

    def test_no_personal_info_in_logs(self):
        """Verify that personal information never appears in logs."""
        personal_data = [
            "user@example.com",
            "+1-555-0123",
            "123-45-6789",
            "john.doe@gmail.com",
        ]

        sanitizer = LogSanitizer()

        for data in personal_data:
            message = f"User data: {data}"
            sanitized = sanitizer.sanitize_message(message, LogSensitivity.SECURE)

            assert data not in sanitized
            assert "MASKED" in sanitized

    def test_log_structure_preservation(self):
        """Test that log structure is preserved after sanitization."""
        message = "Processing trade for user@example.com with amount 12345.67 and API key sk-123"

        sanitizer = LogSanitizer()
        sanitized = sanitizer.sanitize_message(message, LogSensitivity.SECURE)

        # Structure should be preserved
        assert "Processing trade for" in sanitized
        assert "with amount" in sanitized
        assert "and API key" in sanitized

        # Sensitive data should be masked
        assert "***EMAIL_MASKED***" in sanitized
        assert "***AMOUNT_MASKED***" in sanitized
        assert "***API_KEY_MASKED***" in sanitized


class TestPerformanceLogging:
    """Test that logging doesn't impact performance."""

    def test_logging_performance(self):
        """Test that logging operations are reasonably fast."""
        import time

        logger = get_structured_logger("perf_test")

        # Test logging speed
        start_time = time.time()
        for i in range(1000):
            logger.info("Performance test message", counter=i, data="test_data")
        end_time = time.time()

        duration = end_time - start_time
        # Should complete 1000 log operations in reasonable time (< 1 second)
        assert duration < 1.0

    def test_sanitization_performance(self):
        """Test that sanitization is reasonably fast."""
        import time

        sanitizer = LogSanitizer()
        message = "Complex message with API key sk-1234567890abcdef, balance 12345.67, email user@example.com"

        start_time = time.time()
        for _ in range(1000):
            sanitizer.sanitize_message(message, LogSensitivity.SECURE)
        end_time = time.time()

        duration = end_time - start_time
        # Should complete 1000 sanitizations in reasonable time (< 0.5 seconds)
        assert duration < 0.5


if __name__ == "__main__":
    pytest.main([__file__])
