"""
Logging utilities for secure and structured logging in the N1V1 trading framework.

This module provides sanitization and structured logging capabilities to prevent
information disclosure while maintaining useful debugging information.
"""

import logging
import re
from typing import Dict, Any, Optional, Union
from enum import Enum


class LogSensitivity(Enum):
    """Log sensitivity levels for controlling information exposure."""
    DEBUG = "debug"      # Full details, including sensitive data
    INFO = "info"        # General information, sanitized sensitive data
    SECURE = "secure"    # Minimal information, heavily sanitized
    AUDIT = "audit"      # Security-relevant events only


class LogSanitizer:
    """
    Sanitizes log messages to prevent information disclosure.

    Masks sensitive information like API keys, balances, PnL, and personal data
    while preserving useful debugging information.
    """

    # Patterns for sensitive data that should be masked
    SENSITIVE_PATTERNS = [
        # API keys and secrets (more flexible patterns)
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?', '***API_KEY_MASKED***'),
        (r'secret["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?', '***SECRET_MASKED***'),
        (r'token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?', '***TOKEN_MASKED***'),
        (r'password["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{8,})["\']?', '***PASSWORD_MASKED***'),

        # Direct API key patterns (like sk-1234567890abcdef)
        (r'\b(sk|pk|rk|xoxb)-[a-zA-Z0-9_-]{20,}\b', '***API_KEY_MASKED***'),

        # Bearer tokens
        (r'\bBearer\s+[a-zA-Z0-9_.-]{20,}\b', '***TOKEN_MASKED***'),

        # Financial amounts (balance, equity, PnL) - more flexible
        (r'\b(?:balance|equity|pnl|profit|loss)[\'"]?\s*[:=]\s*[\'"]?(-?[\d,]+\.?\d*)[\'"]?', lambda m: f"***{m.group(1).upper()}_MASKED***"),
        (r'\b\d{1,3}(?:,\d{3})*\.\d{2}\b', '***AMOUNT_MASKED***'),  # Currency amounts like 12345.67

        # Personal information
        (r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '***EMAIL_MASKED***'),
        (r'\b\+?[\d\s\-\(\)]{10,}\b', '***PHONE_MASKED***'),
        (r'\b\d{3}-?\d{2}-?\d{4}\b', '***SSN_MASKED***'),

        # Generic sensitive patterns
        (r'"[a-zA-Z0-9_-]{32,}"', '***SENSITIVE_DATA_MASKED***'),  # Long alphanumeric strings
        (r"'[a-zA-Z0-9_-]{32,}'", '***SENSITIVE_DATA_MASKED***'),
    ]

    @classmethod
    def sanitize_message(cls, message: str, sensitivity: LogSensitivity = LogSensitivity.INFO) -> str:
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
        # For audit logs, only keep essential security events
        if any(keyword in message.lower() for keyword in ['auth', 'login', 'access', 'security', 'breach']):
            return cls._apply_sanitization(message)
        else:
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

    def _format_structured_message(self, message: str, **kwargs) -> str:
        """Format a structured log message with key-value pairs."""
        if not kwargs:
            return message

        # Sanitize kwargs based on sensitivity
        sanitized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                # Numeric values are generally safe
                sanitized_kwargs[key] = value
            elif isinstance(value, str):
                # String values may need sanitization
                if self.sensitivity in [LogSensitivity.DEBUG, LogSensitivity.AUDIT]:
                    sanitized_kwargs[key] = value
                else:
                    sanitized_kwargs[key] = self.sanitizer.sanitize_message(value, self.sensitivity)
            else:
                # Other types - convert to string and sanitize
                str_value = str(value)
                if self.sensitivity in [LogSensitivity.DEBUG, LogSensitivity.AUDIT]:
                    sanitized_kwargs[key] = str_value
                else:
                    sanitized_kwargs[key] = self.sanitizer.sanitize_message(str_value, self.sensitivity)

        # Format as structured message
        structured_parts = [message]
        for key, value in sanitized_kwargs.items():
            structured_parts.append(f"{key}={value}")

        return " | ".join(structured_parts)

    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        if self.logger.isEnabledFor(logging.DEBUG):
            formatted_message = self._format_structured_message(message, **kwargs)
            self.logger.debug(formatted_message)

    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        if self.logger.isEnabledFor(logging.INFO):
            formatted_message = self._format_structured_message(message, **kwargs)
            sanitized_message = self.sanitizer.sanitize_message(formatted_message, self.sensitivity)
            self.logger.info(sanitized_message)

    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        if self.logger.isEnabledFor(logging.WARNING):
            formatted_message = self._format_structured_message(message, **kwargs)
            sanitized_message = self.sanitizer.sanitize_message(formatted_message, self.sensitivity)
            self.logger.warning(sanitized_message)

    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        if self.logger.isEnabledFor(logging.ERROR):
            formatted_message = self._format_structured_message(message, **kwargs)
            sanitized_message = self.sanitizer.sanitize_message(formatted_message, self.sensitivity)
            self.logger.error(sanitized_message)

    def critical(self, message: str, **kwargs):
        """Log critical message with structured data."""
        if self.logger.isEnabledFor(logging.CRITICAL):
            formatted_message = self._format_structured_message(message, **kwargs)
            sanitized_message = self.sanitizer.sanitize_message(formatted_message, self.sensitivity)
            self.logger.critical(sanitized_message)

    def exception(self, message: str, **kwargs):
        """Log exception with structured data."""
        if self.logger.isEnabledFor(logging.ERROR):
            formatted_message = self._format_structured_message(message, **kwargs)
            sanitized_message = self.sanitizer.sanitize_message(formatted_message, self.sensitivity)
            self.logger.exception(sanitized_message)


# Global logger factory
_loggers: Dict[str, StructuredLogger] = {}
_global_sensitivity: Optional[LogSensitivity] = None

def get_structured_logger(name: str, sensitivity: LogSensitivity = LogSensitivity.INFO) -> StructuredLogger:
    """
    Get or create a structured logger instance.

    Args:
        name: Logger name
        sensitivity: Default sensitivity level

    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        # Use global sensitivity if set, otherwise use provided sensitivity
        effective_sensitivity = _global_sensitivity if _global_sensitivity is not None else sensitivity
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
def create_secure_logger_config(log_level: str = "INFO",
                               sensitivity: LogSensitivity = LogSensitivity.INFO,
                               log_file: Optional[str] = None) -> Dict[str, Any]:
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
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'secure': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'structured': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'secure',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            'core': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            },
            'trading': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': log_level,
            'handlers': ['console']
        }
    }

    # Add file handler if specified
    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'level': log_level,
            'formatter': 'secure',
            'filename': log_file
        }
        # Add file handler to all loggers
        for logger_config in config['loggers'].values():
            logger_config['handlers'].append('file')
        config['root']['handlers'].append('file')

    return config
