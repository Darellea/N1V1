import json
import logging
import os
import re
import secrets
import types
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Safe import for boto3 to allow patching even if not installed
try:
    import boto3
except ImportError:
    import sys
    import types

    boto3 = types.ModuleType("boto3")
    boto3.Session = types.SimpleNamespace()
    sys.modules["boto3"] = boto3

# Safe import for aiohttp to allow patching
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

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
    (
        r'CRYPTOBOT_EXCHANGE_API_KEY["\']?\s*:\s*["\']([^"\']+)["\']',
        'CRYPTOBOT_EXCHANGE_API_KEY: "***"',
    ),
    (
        r'CRYPTOBOT_EXCHANGE_API_SECRET["\']?\s*:\s*["\']([^"\']+)["\']',
        'CRYPTOBOT_EXCHANGE_API_SECRET: "***"',
    ),
    (
        r'CRYPTOBOT_EXCHANGE_API_PASSPHRASE["\']?\s*:\s*["\']([^"\']+)["\']',
        'CRYPTOBOT_EXCHANGE_API_PASSPHRASE: "***"',
    ),
    (
        r'CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN["\']?\s*:\s*["\']([^"\']+)["\']',
        'CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN: "***"',
    ),
    (
        r'CRYPTOBOT_NOTIFICATIONS_DISCORD_WEBHOOK_URL["\']?\s*:\s*["\']([^"\']+)["\']',
        'CRYPTOBOT_NOTIFICATIONS_DISCORD_WEBHOOK_URL: "***"',
    ),
    (
        r'CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID["\']?\s*:\s*["\']([^"\']+)["\']',
        'CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID: "***"',
    ),
    (r'API_KEY["\']?\s*:\s*["\']([^"\']+)["\']', 'API_KEY: "***"'),
    # Raw API key patterns (common formats)
    (r"\b[A-Za-z0-9]{32,}\b", "***"),  # Generic long alphanumeric strings
    (r"\b[A-Za-z0-9]{64,}\b", "***"),  # Longer keys
]


class SecurityFormatter(logging.Formatter):
    """Logging formatter that masks sensitive data in log messages."""

    def __init__(self, fmt=None, datefmt=None, style="%", validate=True):
        super().__init__(fmt, datefmt, style, validate)
        self.sensitive_patterns = SENSITIVE_PATTERNS

    def format(self, record):
        """Format the log record, masking sensitive data."""
        # Get the original formatted message
        message = super().format(record)

        # Mask sensitive data in the message
        masked_message = self._mask_sensitive_data(message)

        # Also mask in extra fields if they contain sensitive data
        if hasattr(record, "extra") and record.extra:
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
                    self._mask_dict_sensitive_data(item)
                    if isinstance(item, dict)
                    else self._mask_sensitive_data(item)
                    if isinstance(item, str)
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
            "exchange_api_key": os.getenv("CRYPTOBOT_EXCHANGE_API_KEY"),
            "exchange_api_secret": os.getenv("CRYPTOBOT_EXCHANGE_API_SECRET"),
            "exchange_api_passphrase": os.getenv("CRYPTOBOT_EXCHANGE_API_PASSPHRASE"),
            "discord_bot_token": os.getenv("CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN"),
            "discord_webhook_url": os.getenv(
                "CRYPTOBOT_NOTIFICATIONS_DISCORD_WEBHOOK_URL"
            ),
            "discord_channel_id": os.getenv(
                "CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID"
            ),
            "api_key": os.getenv("API_KEY"),
        }

        # Load from config file if exists
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)

                # Extract credentials from config
                exchange = config.get("exchange", {})
                discord = config.get("notifications", {}).get("discord", {})

                file_creds = {
                    "exchange_api_key": exchange.get("api_key"),
                    "exchange_api_secret": exchange.get("api_secret"),
                    "exchange_api_passphrase": exchange.get("api_passphrase"),
                    "discord_bot_token": discord.get("bot_token"),
                    "discord_webhook_url": discord.get("webhook_url"),
                    "discord_channel_id": discord.get("channel_id"),
                }

                # Environment variables take precedence
                for key, value in file_creds.items():
                    if not env_creds.get(key) and value:
                        env_creds[key] = value

            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to load credentials from config: {e}"
                )

        self.credentials = env_creds
        self._audit_access("credentials_loaded", "system")

    def get_credential(self, key: str) -> Optional[str]:
        """Get a credential value, logging access for audit."""
        value = self.credentials.get(key)
        if value:
            self._audit_access(key, "accessed")
        return value

    def _audit_access(self, credential_key: str, action: str):
        """Log credential access for audit purposes."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "credential": credential_key,
            "action": action,
            "masked_value": "***" if self.credentials.get(credential_key) else None,
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
        exchange_creds = ["exchange_api_key", "exchange_api_secret"]
        validation["exchange"] = all(
            self.credentials.get(key) for key in exchange_creds
        )

        # Discord credentials (optional)
        discord_creds = ["discord_bot_token", "discord_channel_id"]
        validation["discord_bot"] = all(
            self.credentials.get(key) for key in discord_creds
        )
        validation["discord_webhook"] = bool(
            self.credentials.get("discord_webhook_url")
        )

        # API key for web interface
        validation["api_key"] = bool(self.credentials.get("api_key"))

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


SECRET_PATTERNS = [
    re.compile(r"(API_KEY\s*=\s*)([^\s]+)", re.IGNORECASE),
    re.compile(r"(TOKEN\s*=\s*)([^\s]+)", re.IGNORECASE),
    re.compile(r"(SECRET\s*=\s*)([^\s]+)", re.IGNORECASE),
    re.compile(r"(PASSWORD\s*=\s*)([^\s]+)", re.IGNORECASE),
]


def sanitize_error_message(message: str) -> str:
    """Sanitize error messages to prevent information leakage."""
    if not isinstance(message, str):
        return message
    sanitized = message
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub(r"\1***SECRET_MASKED***", sanitized)
    return sanitized


def log_security_event(event_type: str, details: Dict[str, Any], level: str = "INFO"):
    """Log security-related events with appropriate masking."""
    logger = logging.getLogger("security")
    formatter = SecurityFormatter()

    # Mask sensitive data in details
    masked_details = formatter._mask_dict_sensitive_data(details)

    message = f"Security event: {event_type}"

    if level.upper() == "ERROR":
        logger.error(message, extra={"security_details": masked_details})
    elif level.upper() == "WARNING":
        logger.warning(message, extra={"security_details": masked_details})
    elif level.upper() == "DEBUG":
        logger.debug(message, extra={"security_details": masked_details})
    else:
        logger.info(message, extra={"security_details": masked_details})


class KeyManagementService(ABC):
    """Abstract base class for key management services (Vault, KMS, etc.)."""

    @abstractmethod
    async def get_secret(self, service: str, key: str) -> Optional[str]:
        """Retrieve a secret from the key management service."""
        pass

    @abstractmethod
    async def store_secret(self, service: str, key: str, value: str) -> bool:
        """Store a secret in the key management service."""
        pass

    @abstractmethod
    async def rotate_key(self, service: str, key: str) -> bool:
        """Rotate a key in the key management service."""
        pass

    @abstractmethod
    async def list_secrets(self, service: str) -> List[str]:
        """List secrets in a path."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the key management service is healthy."""
        pass


class VaultKeyManager(KeyManagementService):
    """HashiCorp Vault integration for key management."""

    def __init__(
        self,
        token: str = None,
        mount_point: str = "secret",
        url: Optional[str] = None,
        **kwargs,
    ):
        # Handle both old and new calling conventions
        if token is None and "token" in kwargs:
            token = kwargs["token"]
        if url is None and "url" in kwargs:
            url = kwargs["url"]
        if mount_point == "secret" and "mount_point" in kwargs:
            mount_point = kwargs["mount_point"]

        self.token = token
        self.mount_point = mount_point
        self.url = url or os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")
        self.session = None

    async def _ensure_session(self):
        """Ensure aiohttp session is available."""
        if self.session is None or self.session.closed:
            import aiohttp

            self.session = aiohttp.ClientSession()

    async def get_secret(self, service: str, key: str) -> Optional[str]:
        """Retrieve a secret from Vault."""
        try:
            await self._ensure_session()
            headers = {"X-Vault-Token": self.token}
            url = f"{self.url}/v1/{self.mount_point}/{service}"
            response = await self.session.get(url, headers=headers)
            try:
                if response.status == 200:
                    data = await response.json()
                    value = data.get("data", {}).get("data", {}).get(key)
                    if value is not None:
                        return value
                log_security_event(
                    "vault_error", {"service": service, "key": key}, "ERROR"
                )
                return None
            finally:
                response.close()
        except Exception as e:
            log_security_event(
                "vault_error",
                {"service": service, "key": key, "error": str(e)},
                "ERROR",
            )
            return None

    async def store_secret(self, service: str, key: str, value: str) -> bool:
        """Store a secret in Vault."""
        try:
            await self._ensure_session()
            headers = {"X-Vault-Token": self.token}
            url = f"{self.url}/v1/{self.mount_point}/{service}"
            payload = {"data": {key: value}}
            response = await self.session.post(url, headers=headers, json=payload)
            try:
                if response.status == 200:
                    return True
                log_security_event(
                    "vault_error", {"service": service, "key": key}, "ERROR"
                )
                return False
            finally:
                response.close()
        except Exception as e:
            log_security_event(
                "vault_error",
                {"service": service, "key": key, "error": str(e)},
                "ERROR",
            )
            return False

    async def rotate_key(self, service: str, key: str) -> bool:
        """Rotate a key by generating a new version."""
        new_value = secrets.token_urlsafe(32)
        return await self.store_secret(service, key, new_value)

    async def list_secrets(self, path: str) -> List[str]:
        """List secrets in a path."""
        await self._ensure_session()
        try:
            url = f"{self.url}/v1/{self.mount_point}/metadata/{path}"
            headers = {"X-Vault-Token": self.token}

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return list(data.get("data", {}).get("keys", []))
                return []
        except Exception as e:
            log_security_event(
                "vault_error",
                {"operation": "list_secrets", "path": path, "error": str(e)},
                "ERROR",
            )
            return []

    async def health_check(self) -> bool:
        """Check Vault health."""
        await self._ensure_session()
        try:
            url = f"{self.url}/v1/sys/health"
            response = await self.session.get(url)
            try:
                return response.status == 200
            finally:
                response.close()
        except Exception:
            return False

    async def close(self):
        """Close the session."""
        if self.session:
            await self.session.close()


class AWSKMSKeyManager(KeyManagementService):
    """AWS KMS integration for key management."""

    def __init__(self, region: str = None, profile: str = None):
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.profile = profile
        self.client = None

    async def _ensure_client(self):
        """Ensure boto3 KMS client is available."""
        if self.client is None:
            try:
                import boto3

                session = (
                    boto3.Session(profile_name=self.profile)
                    if self.profile
                    else boto3.Session()
                )
                self.client = session.client("kms", region_name=self.region)
            except ImportError:
                raise SecurityException("boto3 required for AWS KMS integration")

    async def get_secret(self, service: str, key: str) -> Optional[str]:
        """Retrieve a secret from AWS Systems Manager Parameter Store."""
        await self._ensure_client()
        try:
            import boto3

            ssm = boto3.Session(profile_name=self.profile).client(
                "ssm", region_name=self.region
            )
            parameter_name = f"/{service}/{key}"

            response = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
            return response["Parameter"]["Value"]
        except Exception as e:
            log_security_event(
                "kms_get_failed",
                {"service": service, "key": key, "error": str(e)},
                "WARNING",
            )
            return None

    async def store_secret(self, service: str, key: str, value: str) -> bool:
        """Store a secret in AWS Systems Manager Parameter Store."""
        await self._ensure_client()
        try:
            import boto3

            ssm = boto3.Session(profile_name=self.profile).client(
                "ssm", region_name=self.region
            )
            parameter_name = f"/{service}/{key}"

            ssm.put_parameter(
                Name=parameter_name, Value=value, Type="SecureString", Overwrite=True
            )
            return True
        except Exception as e:
            log_security_event(
                "kms_store_failed",
                {"service": service, "key": key, "error": str(e)},
                "ERROR",
            )
            return False

    async def rotate_key(self, service: str, key: str) -> bool:
        """Rotate a key by generating a new version."""
        new_value = secrets.token_urlsafe(32)
        return await self.store_secret(service, key, new_value)

    async def list_secrets(self, service: str) -> List[str]:
        """List secrets in a path."""
        await self._ensure_client()
        try:
            import boto3

            ssm = boto3.Session(profile_name=self.profile).client(
                "ssm", region_name=self.region
            )

            response = ssm.describe_parameters(
                ParameterFilters=[
                    {"Key": "Name", "Values": [f"/{service}/"]},
                    {"Key": "Type", "Values": ["SecureString"]},
                ]
            )

            return [
                param["Name"].replace(f"/{service}/", "")
                for param in response.get("Parameters", [])
            ]
        except Exception as e:
            log_security_event(
                "kms_error",
                {"operation": "list_secrets", "service": service, "error": str(e)},
                "ERROR",
            )
            return []

    async def health_check(self) -> bool:
        """Check AWS KMS health."""
        await self._ensure_client()
        try:
            # Try to list keys to check connectivity
            self.client.list_keys(Limit=1)
            return True
        except Exception:
            return False


class SecureCredentialManager:
    """Enhanced credential manager with Vault/KMS integration."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.key_manager = None
        self.local_credentials = {}
        self.encryption_keys = {}
        self.key_rotation_schedule = {}
        self._initialize_key_manager()

    def _initialize_key_manager(self):
        """Initialize the appropriate key management service."""
        security_config = self.config.get("security", {})

        if security_config.get("vault", {}).get("enabled"):
            vault_config = security_config["vault"]
            self.key_manager = VaultKeyManager(
                token=vault_config["token"],
                mount_point=vault_config.get("mount_point", "secret"),
                url=vault_config.get("url"),
            )
        elif security_config.get("kms", {}).get("enabled"):
            kms_config = security_config["kms"]
            self.key_manager = AWSKMSKeyManager(
                region=kms_config.get("region"), profile=kms_config.get("profile")
            )
        else:
            # Fallback to local credential manager
            self.key_manager = None

    async def get_credential(self, service: str, key: str) -> Optional[str]:
        """Get a credential with fallback to local storage."""
        if self.key_manager:
            try:
                value = await self.key_manager.get_secret(service, key)
                if value:
                    log_security_event(
                        "credential_retrieved",
                        {"service": service, "key": key, "source": "key_manager"},
                    )
                    return value
            except Exception as e:
                log_security_event(
                    "key_manager_error",
                    {"service": service, "key": key, "error": str(e)},
                    "WARNING",
                )

        # Fallback to local credentials
        return self.local_credentials.get(service, {}).get(key)

    async def store_credential(self, service: str, key: str, value: str) -> bool:
        """Store a credential securely."""
        if self.key_manager:
            try:
                success = await self.key_manager.store_secret(service, key, value)
                if success:
                    log_security_event(
                        "credential_stored",
                        {"service": service, "key": key, "source": "key_manager"},
                    )
                    return True
            except Exception as e:
                log_security_event(
                    "key_manager_store_error",
                    {"service": service, "key": key, "error": str(e)},
                    "ERROR",
                )

        # Fallback to local storage (not recommended for production)
        if service not in self.local_credentials:
            self.local_credentials[service] = {}
        self.local_credentials[service][key] = value
        log_security_event(
            "credential_stored_local",
            {"service": service, "key": key, "warning": "stored_locally"},
            "WARNING",
        )
        return True

    async def rotate_key(self, service: str, key: str) -> bool:
        """Rotate a credential key."""
        if self.key_manager:
            try:
                success = await self.key_manager.rotate_key(service, key)
                if success:
                    log_security_event("key_rotated", {"service": service, "key": key})
                    # Update rotation timestamp
                    rotation_key = f"{service}/{key}"
                    self.key_rotation_schedule[rotation_key] = datetime.utcnow()
                return success
            except Exception as e:
                log_security_event(
                    "key_rotation_failed",
                    {"service": service, "key": key, "error": str(e)},
                    "ERROR",
                )
                return False

        # Handle local credential rotation
        if service in self.local_credentials and key in self.local_credentials[service]:
            import uuid

            new_value = str(uuid.uuid4())
            self.local_credentials[service][key] = new_value
            log_security_event("key_rotated_local", {"service": service, "key": key})
            # Update rotation timestamp
            rotation_key = f"{service}/{key}"
            self.key_rotation_schedule[rotation_key] = datetime.utcnow()
            return True

        log_security_event(
            "key_rotation_failed",
            {"service": service, "key": key, "reason": "credential_not_found"},
            "ERROR",
        )
        return False

    async def validate_credentials(self) -> Dict[str, bool]:
        """Validate all configured credentials."""
        validation_results = {}

        # Check exchange credentials
        exchange_key = await self.get_credential("exchange", "api_key")
        exchange_secret = await self.get_credential("exchange", "api_secret")
        validation_results["exchange"] = bool(exchange_key and exchange_secret)

        # Check notification credentials
        discord_token = await self.get_credential("discord", "bot_token")
        validation_results["discord"] = bool(discord_token)

        # Check API credentials
        api_key = await self.get_credential("api", "key")
        validation_results["api"] = bool(api_key)

        return validation_results

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "key_manager": False,
            "credentials": False,
            "encryption": True,  # Local encryption always available
        }

        if self.key_manager:
            health_status["key_manager"] = await self.key_manager.health_check()

        # Check if we have valid credentials
        validation = await self.validate_credentials()
        health_status["credentials"] = any(validation.values())

        return health_status

    def get_key_rotation_status(self) -> Dict[str, Any]:
        """Get status of key rotations."""
        return {
            "rotation_schedule": self.key_rotation_schedule,
            "next_rotations": self._calculate_next_rotations(),
        }

    def _calculate_next_rotations(self) -> Dict[str, datetime]:
        """Calculate next rotation times based on policy."""
        next_rotations = {}
        rotation_days = self.config.get("security", {}).get("key_rotation_days", 90)

        for key_path, last_rotation in self.key_rotation_schedule.items():
            next_rotations[key_path] = last_rotation + timedelta(days=rotation_days)

        return next_rotations


# Global secure credential manager instance
_secure_credential_manager = None


async def get_secure_credential_manager(
    config: Dict[str, Any]
) -> SecureCredentialManager:
    """Get the global secure credential manager instance."""
    global _secure_credential_manager
    if _secure_credential_manager is None:
        _secure_credential_manager = SecureCredentialManager(config)
    return _secure_credential_manager


# Helper functions for secure secret retrieval
async def get_secret(secret_name: str) -> Optional[str]:
    """
    Retrieve a secret securely from Vault/KMS with dev fallback.

    Args:
        secret_name: Name of the secret to retrieve. Supported names:
            - "exchange_api_key"
            - "exchange_api_secret"
            - "exchange_api_passphrase"
            - "discord_token"
            - "discord_channel_id"
            - "discord_webhook_url"
            - "api_key"

    Returns:
        The secret value or None if not found

    Raises:
        SecurityException: If secret is required but not found in production
    """
    # Map secret names to service/key pairs
    secret_mapping = {
        "exchange_api_key": ("exchange", "api_key"),
        "exchange_api_secret": ("exchange", "api_secret"),
        "exchange_api_passphrase": ("exchange", "api_passphrase"),
        "discord_token": ("discord", "bot_token"),
        "discord_channel_id": ("discord", "channel_id"),
        "discord_webhook_url": ("discord", "webhook_url"),
        "api_key": ("api", "key"),
    }

    if secret_name not in secret_mapping:
        log_security_event(
            "invalid_secret_name", {"secret_name": secret_name}, "WARNING"
        )
        return None

    service, key = secret_mapping[secret_name]

    # Check if we're in dev mode (fallback to .env)
    env_mode = os.getenv("ENV", "live").lower()
    if env_mode == "dev":
        # Fallback to environment variables for development
        env_var_mapping = {
            "exchange_api_key": "CRYPTOBOT_EXCHANGE_API_KEY",
            "exchange_api_secret": "CRYPTOBOT_EXCHANGE_API_SECRET",
            "exchange_api_passphrase": "CRYPTOBOT_EXCHANGE_API_PASSPHRASE",
            "discord_token": "CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN",
            "discord_channel_id": "CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID",
            "discord_webhook_url": "CRYPTOBOT_NOTIFICATIONS_DISCORD_WEBHOOK_URL",
            "api_key": "API_KEY",
        }

        env_var = env_var_mapping.get(secret_name)
        if env_var:
            value = os.getenv(env_var)
            if value:
                log_security_event(
                    "secret_retrieved_from_env",
                    {"secret_name": secret_name, "env_var": env_var},
                    "INFO",
                )
                return value

    # Try secure credential manager
    try:
        # Load config to initialize secure manager
        config = _load_security_config()
        manager = await get_secure_credential_manager(config)
        if manager.key_manager and hasattr(manager.key_manager, "get_secret"):
            value = await manager.key_manager.get_secret(service, key)
        else:
            value = await manager.get_credential(service, key)

        if value:
            log_security_event(
                "secret_retrieved_securely",
                {"secret_name": secret_name, "service": service, "key": key},
            )
            return value

    except Exception as e:
        log_security_event(
            "secure_secret_retrieval_failed",
            {"secret_name": secret_name, "error": str(e)},
            "ERROR",
        )

    # For production/live mode, raise error if secret not found
    if env_mode in ["live", "production"]:
        raise SecurityException(
            f"Required secret '{secret_name}' not found in secure storage"
        )

    # For test/dev modes, return a test secret
    if env_mode in ["test", "dev", "ci"]:
        test_secret = "TEST_SECRET_12345"
        log_security_event(
            "test_secret_returned",
            {"secret_name": secret_name, "env_mode": env_mode},
            "INFO",
        )
        return test_secret

    # For other modes, return None
    log_security_event(
        "secret_not_found",
        {"secret_name": secret_name, "env_mode": env_mode},
        "WARNING",
    )
    return None


async def rotate_key(secret_name: str) -> bool:
    """
    Rotate a secret key securely.

    Args:
        secret_name: Name of the secret to rotate

    Returns:
        True if rotation successful, False otherwise
    """
    secret_mapping = {
        "exchange_api_key": ("exchange", "api_key"),
        "exchange_api_secret": ("exchange", "api_secret"),
        "exchange_api_passphrase": ("exchange", "api_passphrase"),
        "discord_token": ("discord", "bot_token"),
        "discord_channel_id": ("discord", "channel_id"),
        "discord_webhook_url": ("discord", "webhook_url"),
        "api_key": ("api", "key"),
    }

    if secret_name not in secret_mapping:
        log_security_event(
            "invalid_secret_for_rotation", {"secret_name": secret_name}, "WARNING"
        )
        return False

    service, key = secret_mapping[secret_name]

    try:
        config = _load_security_config()
        manager = await get_secure_credential_manager(config)
        if manager.key_manager and hasattr(manager.key_manager, "rotate_key"):
            return await manager.key_manager.rotate_key(service, key)
    except Exception as e:
        log_security_event("key_rotation_failed", {"error": str(e)}, "ERROR")
    return False


def _load_security_config() -> Dict[str, Any]:
    """Load security configuration from config.json or environment."""
    config = {"security": {}}

    # Try to load from config.json
    try:
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                file_config = json.load(f)
                config.update(file_config)
    except Exception:
        pass

    # Override with environment variables
    if os.getenv("VAULT_URL"):
        config["security"]["vault"] = {
            "enabled": True,
            "url": os.getenv("VAULT_URL"),
            "token": os.getenv("VAULT_TOKEN"),
            "mount_point": os.getenv("VAULT_MOUNT_POINT", "secret"),
        }

    if os.getenv("AWS_REGION"):
        config["security"]["kms"] = {
            "enabled": True,
            "region": os.getenv("AWS_REGION"),
            "profile": os.getenv("AWS_PROFILE"),
        }

    # Set defaults if no secure storage configured
    if not config["security"].get("vault", {}).get("enabled") and not config[
        "security"
    ].get("kms", {}).get("enabled"):
        config["security"]["vault"] = {"enabled": False}
        config["security"]["kms"] = {"enabled": False}

    return config


class OrderFlowValidator:
    """Validator for order flow invariants and security checks."""

    def __init__(self):
        self.order_ids = set()
        self.order_states = {}
        self.schema_cache = {}

    def validate_order_schema(self, order: Dict[str, Any]) -> bool:
        """Validate order against schema requirements."""
        required_fields = ["id", "symbol", "side", "type", "amount"]

        # Check required fields
        for field in required_fields:
            if field not in order:
                log_security_event(
                    "order_schema_validation_failed",
                    {"order_id": order.get("id", "unknown"), "missing_field": field},
                    "WARNING",
                )
                return False

        # Validate order ID format
        order_id = order["id"]
        if not self._is_valid_order_id(order_id):
            log_security_event("invalid_order_id", {"order_id": order_id}, "WARNING")
            return False

        # Check for duplicate order IDs
        if order_id in self.order_ids:
            log_security_event("duplicate_order_id", {"order_id": order_id}, "ERROR")
            return False

        # Validate amount and price
        if not self._validate_numeric_fields(order):
            return False

        return True

    def _is_valid_order_id(self, order_id: str) -> bool:
        """Validate order ID format."""
        # Order IDs should be alphanumeric with hyphens/underscores
        import re

        return bool(re.match(r"^[a-zA-Z0-9_-]{8,64}$", str(order_id)))

    def _validate_numeric_fields(self, order: Dict[str, Any]) -> bool:
        """Validate numeric fields in order."""
        try:
            amount = float(order["amount"])
            if amount <= 0:
                log_security_event(
                    "invalid_order_amount",
                    {"order_id": order.get("id"), "amount": amount},
                    "WARNING",
                )
                return False

            # Validate price if present
            if "price" in order:
                price = float(order["price"])
                if price <= 0:
                    log_security_event(
                        "invalid_order_price",
                        {"order_id": order.get("id"), "price": price},
                        "WARNING",
                    )
                    return False

            return True
        except (ValueError, TypeError):
            log_security_event(
                "non_numeric_order_fields",
                {
                    "order_id": order.get("id"),
                    "amount": order.get("amount"),
                    "price": order.get("price"),
                },
                "WARNING",
            )
            return False

    def register_order(self, order: Dict[str, Any]) -> bool:
        """Register a new order for tracking."""
        if not self.validate_order_schema(order):
            return False

        order_id = order["id"]
        self.order_ids.add(order_id)
        self.order_states[order_id] = {
            "status": "pending",
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
        }

        log_security_event(
            "order_registered",
            {
                "order_id": order_id,
                "symbol": order.get("symbol"),
                "side": order.get("side"),
            },
        )
        return True

    def update_order_state(self, order_id: str, new_status: str) -> bool:
        """Update order state with consistency checks."""
        if order_id not in self.order_states:
            log_security_event(
                "unknown_order_update",
                {"order_id": order_id, "new_status": new_status},
                "WARNING",
            )
            return False

        current_state = self.order_states[order_id]["status"]
        valid_transitions = {
            "pending": ["filled", "cancelled", "rejected"],
            "filled": [],  # Terminal state
            "cancelled": [],  # Terminal state
            "rejected": [],  # Terminal state
        }

        if new_status not in valid_transitions.get(current_state, []):
            log_security_event(
                "invalid_state_transition",
                {
                    "order_id": order_id,
                    "current_state": current_state,
                    "new_status": new_status,
                },
                "ERROR",
            )
            return False

        self.order_states[order_id]["status"] = new_status
        self.order_states[order_id]["last_updated"] = datetime.utcnow()

        log_security_event(
            "order_state_updated",
            {
                "order_id": order_id,
                "old_status": current_state,
                "new_status": new_status,
            },
        )
        return True

    def validate_state_consistency(self) -> Dict[str, Any]:
        """Validate that all order states are consistent."""
        inconsistencies = []
        terminal_states = {"filled", "cancelled", "rejected"}

        for order_id, state_info in self.order_states.items():
            status = state_info["status"]
            created_at = state_info["created_at"]
            last_updated = state_info["last_updated"]

            # Check for stale orders
            if status not in terminal_states:
                age_hours = (datetime.utcnow() - created_at).total_seconds() / 3600
                if age_hours > 24:  # Configurable threshold
                    inconsistencies.append(
                        {
                            "type": "stale_order",
                            "order_id": order_id,
                            "age_hours": age_hours,
                        }
                    )

            # Check for orders updated before creation
            if last_updated < created_at:
                inconsistencies.append(
                    {
                        "type": "time_anomaly",
                        "order_id": order_id,
                        "created_at": created_at.isoformat(),
                        "last_updated": last_updated.isoformat(),
                    }
                )

        return {
            "consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies,
            "total_orders": len(self.order_states),
        }

    def cleanup_completed_orders(self, max_age_days: int = 30):
        """Clean up old completed orders."""
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        terminal_states = {"filled", "cancelled", "rejected"}

        to_remove = []
        for order_id, state_info in self.order_states.items():
            if (
                state_info["status"] in terminal_states
                and state_info["last_updated"] < cutoff_date
            ):
                to_remove.append(order_id)

        for order_id in to_remove:
            del self.order_states[order_id]
            self.order_ids.discard(order_id)

        if to_remove:
            log_security_event(
                "orders_cleaned_up",
                {"removed_count": len(to_remove), "max_age_days": max_age_days},
            )


# Global order flow validator instance
_order_flow_validator = None


def get_order_flow_validator() -> OrderFlowValidator:
    """Get the global order flow validator instance."""
    global _order_flow_validator
    if _order_flow_validator is None:
        _order_flow_validator = OrderFlowValidator()
    return _order_flow_validator
