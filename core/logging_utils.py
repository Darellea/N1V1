"""
Logging utilities for secure and structured logging in the N1V1 trading framework.

This module provides sanitization and structured logging capabilities to prevent
information disclosure while maintaining useful debugging information.
"""

import json
import logging
import re
from enum import Enum
from typing import Any, Dict, Optional


class LogSensitivity(Enum):
    """Log sensitivity levels for controlling information exposure."""

    DEBUG = "debug"  # Full details, including sensitive data
    INFO = "info"  # General information, sanitized sensitive data
    SECURE = "secure"  # Minimal information, heavily sanitized
    AUDIT = "audit"  # Security-relevant events only


class LogSanitizer:
    """
    Sanitizes log messages to prevent information disclosure.

    Masks sensitive information like API keys, balances, PnL, and personal data
    while preserving useful debugging information.
    """

    # Patterns for sensitive data that should be masked (ordered by priority)
    SENSITIVE_PATTERNS = [
        # Bearer tokens (highest priority)
        (r"Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*", "***TOKEN_MASKED***"),
        # Phone numbers (require separators to avoid matching parts of other strings)
        (r"\+?\d{1,3}[-]\d{3}[-]\d{4}", "***PHONE_MASKED***"),
        # Emails
        (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "***EMAIL_MASKED***"),
        # SSNs
        (r"\b\d{3}-\d{2}-\d{4}\b", "***SSN_MASKED***"),
        # API keys - prefixed formats (high priority, before amounts)
        (r"\b(sk|pk|rk|xoxb)[_-]?[A-Za-z0-9]{1,}\b", "***API_KEY_MASKED***"),
        # API keys - generic long strings (avoid masked)
        (
            r"(?<!\*\*\*)(?<!MASKED)\b[A-Za-z0-9_-]{16,}\b(?!\*\*\*)",
            "***API_KEY_MASKED***",
        ),
        # Financial amounts with labels (semantic masking)
        (
            r'\b(balance|equity|pnl|profit|loss)[\'"]?\s*[:=]\s*[\'"]?(-?[\d,]+\.?\d*)[\'"]?',
            lambda m: f"***{m.group(1).upper()}_MASKED***",
        ),
        # Generic financial amounts (avoid matching already masked or API keys)
        (
            r"(?<!\*\*\*)(?<!MASKED)(?<!sk)(?<!pk)(?<!rk)(?<!xoxb)\b\d+(\.\d+)?\b(?!\*\*\*)",
            "***AMOUNT_MASKED***",
        ),
    ]

    @classmethod
    def sanitize_message(
        cls, message: str, sensitivity: LogSensitivity = LogSensitivity.INFO
    ) -> str:
        """
        Sanitize a log message based on sensitivity level.

        Args:
            message: The log message to sanitize
            sensitivity: The sensitivity level for sanitization

        Returns:
            Sanitized message
        """
        if sensitivity == LogSensitivity.DEBUG:
            # Debug level shows full information
            return message
        elif sensitivity == LogSensitivity.AUDIT:
            # Audit level shows minimal information
            return cls._sanitize_for_audit(message)
        else:
            # Info and Secure levels apply sanitization
            return cls._apply_sanitization(message)

    @classmethod
    def _apply_sanitization(cls, message: str) -> str:
        """Apply sanitization patterns to the message."""
        sanitized = message

        for pattern, replacement in cls.SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized

    @classmethod
    def _sanitize_for_audit(cls, message: str) -> str:
        """Sanitize message for audit logging (minimal information)."""
        # For audit logs, always return minimal fixed string
        return "[AUDIT] Security event logged"


class StructuredLogger:
    """
    Structured logger that supports different sensitivity levels and sanitization.

    Provides methods for logging with structured data while ensuring sensitive
    information is properly masked based on the configured sensitivity level.
    """

    def __init__(self, name: str, sensitivity: LogSensitivity = LogSensitivity.INFO):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            sensitivity: Default sensitivity level
        """
        self.logger = logging.getLogger(name)
        self.sensitivity = sensitivity
        self.sanitizer = LogSanitizer()

    def set_sensitivity(self, sensitivity: LogSensitivity):
        """Set the logging sensitivity level."""
        self.sensitivity = sensitivity

    def _format_structured_message(
        self, message: str, log_level: str = "INFO", **kwargs
    ) -> str:
        """Format a structured log message with both key=value and JSON formats."""
        # Fields that should not be sanitized (metadata fields)
        non_sensitive_fields = {"component", "operation", "user_id", "level", "message"}

        # Sanitize kwargs based on sensitivity
        sanitized_kwargs = {}
        for key, value in kwargs.items():
            if self.sensitivity in [LogSensitivity.DEBUG, LogSensitivity.AUDIT]:
                # Keep original types for DEBUG/AUDIT
                sanitized_kwargs[key] = value
            elif key in non_sensitive_fields:
                # Don't sanitize metadata fields
                sanitized_kwargs[key] = value
            else:
                # For SECURE/INFO, apply semantic masking for known fields
                if key in [
                    "amount",
                    "balance",
                    "pnl",
                    "api_key",
                ] and self.sensitivity in [LogSensitivity.SECURE, LogSensitivity.INFO]:
                    sanitized_kwargs[key] = f"***{key.upper()}_MASKED***"
                else:
                    # Otherwise sanitize string representations
                    if isinstance(value, str):
                        sanitized_kwargs[key] = self.sanitizer.sanitize_message(
                            value, self.sensitivity
                        )
                    else:
                        str_value = str(value)
                        sanitized_kwargs[key] = self.sanitizer.sanitize_message(
                            str_value, self.sensitivity
                        )

        # Create key=value pairs for backward compatibility
        key_value_parts = []
        for key, value in sanitized_kwargs.items():
            # Use unquoted values for backward compatibility (key=value format)
            key_value_parts.append(f"{key}={value}")

        # Combine key=value format with JSON separated by " | "
        key_value_str = " ".join(key_value_parts)
        json_str = json.dumps({"message": message, **sanitized_kwargs})

        # Return combined format: key=value pairs | JSON
        return f"{key_value_str} | {json_str}"

    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        formatted_message = self._format_structured_message(
            message, log_level="DEBUG", **kwargs
        )
        self.logger.debug(f"DEBUG - {formatted_message}")

    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        # Remove 'level' from kwargs to avoid conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "level"}
        formatted_message = self._format_structured_message(
            message, log_level="INFO", **filtered_kwargs
        )
        # For structured logging, values are already sanitized in JSON, don't sanitize again
        self.logger.info(f"INFO - {formatted_message}")

    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        formatted_message = self._format_structured_message(
            message, log_level="WARNING", **kwargs
        )
        # For structured logging, values are already sanitized in JSON, don't sanitize again
        self.logger.warning(f"WARNING - {formatted_message}")

    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        formatted_message = self._format_structured_message(
            message, log_level="ERROR", **kwargs
        )
        # For structured logging, values are already sanitized in JSON, don't sanitize again
        self.logger.error(f"ERROR - {formatted_message}")

    def critical(self, message: str, **kwargs):
        """Log critical message with structured data."""
        formatted_message = self._format_structured_message(
            message, log_level="CRITICAL", **kwargs
        )
        # For structured logging, values are already sanitized in JSON, don't sanitize again
        self.logger.critical(f"CRITICAL - {formatted_message}")

    def exception(self, message: str, **kwargs):
        """Log exception with structured data."""
        formatted_message = self._format_structured_message(
            message, log_level="ERROR", **kwargs
        )
        # For structured logging, values are already sanitized in JSON, don't sanitize again
        self.logger.exception(f"ERROR - {formatted_message}")


# Global logger factory
_loggers: Dict[str, StructuredLogger] = {}
_global_sensitivity: LogSensitivity = LogSensitivity.INFO


def get_structured_logger(
    name: str, sensitivity: Optional[LogSensitivity] = None
) -> StructuredLogger:
    """
    Get or create a structured logger instance.

    Args:
        name: Logger name
        sensitivity: Optional sensitivity level (uses global if not provided)

    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        # Use provided sensitivity or global default
        effective_sensitivity = sensitivity or _global_sensitivity
        _loggers[name] = StructuredLogger(name, effective_sensitivity)
    return _loggers[name]


def set_global_log_sensitivity(sensitivity: LogSensitivity):
    """Set the sensitivity level for all structured loggers."""
    global _global_sensitivity
    _global_sensitivity = sensitivity

    # Update all existing loggers
    for logger in _loggers.values():
        logger.set_sensitivity(sensitivity)


# Configuration helpers
def create_secure_logger_config(
    log_level: str = "INFO",
    sensitivity: LogSensitivity = LogSensitivity.INFO,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a secure logger configuration.

    Args:
        log_level: Standard logging level
        sensitivity: Information sensitivity level
        log_file: Optional log file path

    Returns:
        Logger configuration dictionary
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "secure": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "structured": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "secure",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "core": {"level": log_level, "handlers": ["console"], "propagate": False},
            "trading": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {"level": log_level, "handlers": ["console"]},
    }

    # Add file handler if specified
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": log_level,
            "formatter": "secure",
            "filename": log_file,
        }
        # Add file handler to all loggers
        for logger_config in config["loggers"].values():
            logger_config["handlers"].append("file")
        config["root"]["handlers"].append("file")

    return config
