"""
Security utilities for the trading framework.

Provides secure logging, credential management, and security-related functions.
"""

import re
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
import os
from datetime import datetime


# Sensitive data patterns to mask in logs
SENSITIVE_PATTERNS = [
    (r'api_key["\']?\s*:\s*["\']([^"\']+)["\']', 'api_key: "***"'),
    (r'api_secret["\']?\s*:\s*["\']([^"\']+)["\']', 'api_secret: "***"'),
    (r'api_passphrase["\']?\s*:\s*["\']([^"\']+)["\']', 'api_passphrase: "***"'),
    (r'password["\']?\s*:\s*["\']([^"\']+)["\']', 'password: "***"'),
    (r'token["\']?\s*:\s*["\']([^"\']+)["\']', 'token: "***"'),
    (r'secret["\']?\s*:\s*["\']([^"\']+)["\']', 'secret: "***"'),
    (r'key["\']?\s*:\s*["\']([^"\']+)["\']', 'key: "***"'),
    # Environment variable patterns
    (r'CRYPTOBOT_EXCHANGE_API_KEY["\']?\s*:\s*["\']([^"\']+)["\']', 'CRYPTOBOT_EXCHANGE_API_KEY: "***"'),
    (r'CRYPTOBOT_EXCHANGE_API_SECRET["\']?\s*:\s*["\']([^"\']+)["\']', 'CRYPTOBOT_EXCHANGE_API_SECRET: "***"'),
    (r'CRYPTOBOT_EXCHANGE_API_PASSPHRASE["\']?\s*:\s*["\']([^"\']+)["\']', 'CRYPTOBOT_EXCHANGE_API_PASSPHRASE: "***"'),
    (r'CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN["\']?\s*:\s*["\']([^"\']+)["\']', 'CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN: "***"'),
    (r'CRYPTOBOT_NOTIFICATIONS_DISCORD_WEBHOOK_URL["\']?\s*:\s*["\']([^"\']+)["\']', 'CRYPTOBOT_NOTIFICATIONS_DISCORD_WEBHOOK_URL: "***"'),
    (r'CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID["\']?\s*:\s*["\']([^"\']+)["\']', 'CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID: "***"'),
    (r'API_KEY["\']?\s*:\s*["\']([^"\']+)["\']', 'API_KEY: "***"'),
    # Raw API key patterns (common formats)
    (r'\b[A-Za-z0-9]{32,}\b', '***'),  # Generic long alphanumeric strings
    (r'\b[A-Za-z0-9]{64,}\b', '***'),  # Longer keys
]


class SecurityFormatter(logging.Formatter):
    """Logging formatter that masks sensitive data in log messages."""

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)
        self.sensitive_patterns = SENSITIVE_PATTERNS

    def format(self, record):
        """Format the log record, masking sensitive data."""
        # Get the original formatted message
        message = super().format(record)

        # Mask sensitive data in the message
        masked_message = self._mask_sensitive_data(message)

        # Also mask in extra fields if they contain sensitive data
        if hasattr(record, 'extra') and record.extra:
            for key, value in record.extra.items():
                if isinstance(value, str):
                    record.extra[key] = self._mask_sensitive_data(value)
                elif isinstance(value, dict):
                    record.extra[key] = self._mask_dict_sensitive_data(value)

        return masked_message

    def _mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data patterns in text."""
        if not isinstance(text, str):
            return text

        masked_text = text
        for pattern, replacement in self.sensitive_patterns:
            masked_text = re.sub(pattern, replacement, masked_text, flags=re.IGNORECASE)

        return masked_text

    def _mask_dict_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask sensitive data in dictionary."""
        if not isinstance(data, dict):
            return data

        masked_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                masked_data[key] = self._mask_sensitive_data(value)
            elif isinstance(value, dict):
                masked_data[key] = self._mask_dict_sensitive_data(value)
            elif isinstance(value, list):
                masked_data[key] = [
                    self._mask_dict_sensitive_data(item) if isinstance(item, dict)
                    else self._mask_sensitive_data(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                masked_data[key] = value

        return masked_data


class CredentialManager:
    """Centralized credential management system."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.credentials = {}
        self.audit_log = []
        self._load_credentials()

    def _load_credentials(self):
        """Load credentials from environment variables and config file."""
        # Load from environment variables
        env_creds = {
            'exchange_api_key': os.getenv('CRYPTOBOT_EXCHANGE_API_KEY'),
            'exchange_api_secret': os.getenv('CRYPTOBOT_EXCHANGE_API_SECRET'),
            'exchange_api_passphrase': os.getenv('CRYPTOBOT_EXCHANGE_API_PASSPHRASE'),
            'discord_bot_token': os.getenv('CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN'),
            'discord_webhook_url': os.getenv('CRYPTOBOT_NOTIFICATIONS_DISCORD_WEBHOOK_URL'),
            'discord_channel_id': os.getenv('CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID'),
            'api_key': os.getenv('API_KEY'),
        }

        # Load from config file if exists
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                # Extract credentials from config
                exchange = config.get('exchange', {})
                discord = config.get('notifications', {}).get('discord', {})

                file_creds = {
                    'exchange_api_key': exchange.get('api_key'),
                    'exchange_api_secret': exchange.get('api_secret'),
                    'exchange_api_passphrase': exchange.get('api_passphrase'),
                    'discord_bot_token': discord.get('bot_token'),
                    'discord_webhook_url': discord.get('webhook_url'),
                    'discord_channel_id': discord.get('channel_id'),
                }

                # Environment variables take precedence
                for key, value in file_creds.items():
                    if not env_creds.get(key) and value:
                        env_creds[key] = value

            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to load credentials from config: {e}")

        self.credentials = env_creds
        self._audit_access('credentials_loaded', 'system')

    def get_credential(self, key: str) -> Optional[str]:
        """Get a credential value, logging access for audit."""
        value = self.credentials.get(key)
        if value:
            self._audit_access(key, 'accessed')
        return value

    def _audit_access(self, credential_key: str, action: str):
        """Log credential access for audit purposes."""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'credential': credential_key,
            'action': action,
            'masked_value': '***' if self.credentials.get(credential_key) else None
        }
        self.audit_log.append(audit_entry)

        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]

    def get_audit_log(self) -> list:
        """Get the audit log of credential accesses."""
        return self.audit_log.copy()

    def validate_credentials(self) -> Dict[str, bool]:
        """Validate that required credentials are present and properly formatted."""
        validation = {}

        # Exchange credentials
        exchange_creds = ['exchange_api_key', 'exchange_api_secret']
        validation['exchange'] = all(self.credentials.get(key) for key in exchange_creds)

        # Discord credentials (optional)
        discord_creds = ['discord_bot_token', 'discord_channel_id']
        validation['discord_bot'] = all(self.credentials.get(key) for key in discord_creds)
        validation['discord_webhook'] = bool(self.credentials.get('discord_webhook_url'))

        # API key for web interface
        validation['api_key'] = bool(self.credentials.get('api_key'))

        return validation


# Global credential manager instance
_credential_manager = None

def get_credential_manager() -> CredentialManager:
    """Get the global credential manager instance."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager()
    return _credential_manager


class SecurityException(Exception):
    """Base class for security-related exceptions."""
    pass


class CredentialExposureException(SecurityException):
    """Exception raised when sensitive credentials are exposed."""
    pass


class InvalidCredentialException(SecurityException):
    """Exception raised when credentials are invalid or missing."""
    pass


class SecurityViolationException(SecurityException):
    """Exception raised when a security violation is detected."""
    pass


def sanitize_error_message(message: str) -> str:
    """Sanitize error messages to prevent information leakage."""
    # Remove sensitive data from error messages
    formatter = SecurityFormatter()
    return formatter._mask_sensitive_data(message)


def log_security_event(event_type: str, details: Dict[str, Any], level: str = 'INFO'):
    """Log security-related events with appropriate masking."""
    logger = logging.getLogger('security')
    formatter = SecurityFormatter()

    # Mask sensitive data in details
    masked_details = formatter._mask_dict_sensitive_data(details)

    message = f"Security event: {event_type}"

    if level.upper() == 'ERROR':
        logger.error(message, extra={'security_details': masked_details})
    elif level.upper() == 'WARNING':
        logger.warning(message, extra={'security_details': masked_details})
    elif level.upper() == 'DEBUG':
        logger.debug(message, extra={'security_details': masked_details})
    else:
        logger.info(message, extra={'security_details': masked_details})
