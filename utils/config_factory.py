"""
Unified Configuration Factory for N1V1 Framework.

Provides centralized configuration loading with caching, validation,
environment-specific support, and schema enforcement to eliminate
duplication across the codebase.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List, Set, Union
from pathlib import Path
from dataclasses import dataclass
import hashlib
import time
from threading import Lock

from utils.constants import PROJECT_ROOT
from utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


@dataclass
class ConfigCacheEntry:
    """Cache entry for configuration data."""

    data: Dict[str, Any]
    timestamp: float
    checksum: str
    file_path: Optional[Path] = None


class ConfigValidator:
    """
    Configuration validator with schema enforcement.
    """

    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self._load_default_schemas()

    def _load_default_schemas(self):
        """Load default configuration schemas."""
        # Trading configuration schema
        self.schemas["trading"] = {
            "required": ["symbols", "timeframes"],
            "types": {
                "symbols": list,
                "timeframes": list,
                "risk_management": dict,
                "execution": dict
            }
        }

        # Risk management schema
        self.schemas["risk"] = {
            "required": ["max_position_size", "max_drawdown"],
            "types": {
                "max_position_size": (int, float),
                "max_drawdown": (int, float),
                "stop_loss_percentage": (int, float),
                "take_profit_percentage": (int, float)
            }
        }

        # Logging schema
        self.schemas["logging"] = {
            "types": {
                "level": str,
                "file_logging": bool,
                "console": bool,
                "max_size": int,
                "backup_count": int
            }
        }

    def validate_config(self, config: Dict[str, Any], schema_name: str) -> List[str]:
        """
        Validate configuration against a schema.

        Args:
            config: Configuration to validate
            schema_name: Name of the schema to use

        Returns:
            List of validation errors (empty if valid)
        """
        if schema_name not in self.schemas:
            return [f"Unknown schema: {schema_name}"]

        schema = self.schemas[schema_name]
        errors = []

        # Check required fields
        for field in schema.get("required", []):
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Check field types
        for field, expected_type in schema.get("types", {}).items():
            if field in config:
                value = config[field]
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Field '{field}' has wrong type. Expected {expected_type}, got {type(value)}"
                    )

        return errors

    def add_schema(self, name: str, schema: Dict[str, Any]):
        """Add a custom schema for validation."""
        self.schemas[name] = schema


class ConfigFactory:
    """
    Unified configuration factory with caching and validation.

    Eliminates duplication by providing a single point for configuration
    loading, caching, and validation across the entire framework.
    """

    def __init__(self):
        self.cache: Dict[str, ConfigCacheEntry] = {}
        self.cache_lock = Lock()
        self.validator = ConfigValidator()
        self.environment = os.getenv("N1V1_ENV", "development")
        self.config_paths: Dict[str, Path] = {}
        self._initialize_config_paths()

    def _initialize_config_paths(self):
        """Initialize configuration file paths."""
        self.config_paths = {
            "main": PROJECT_ROOT / "config.json",
            "trading": PROJECT_ROOT / "config_trading.json",
            "risk": PROJECT_ROOT / "config_risk.json",
            "logging": PROJECT_ROOT / "config_logging.json",
            "api": PROJECT_ROOT / "config_api.json",
            "database": PROJECT_ROOT / "config_database.json",
            "notifications": PROJECT_ROOT / "config_notifications.json"
        }

        # Add environment-specific paths
        for config_type in list(self.config_paths.keys()):
            env_path = PROJECT_ROOT / f"config_{config_type}_{self.environment}.json"
            if env_path.exists():
                self.config_paths[f"{config_type}_{self.environment}"] = env_path

    def get_config(self, config_type: str = "main", use_cache: bool = True,
                  validate: bool = True) -> Dict[str, Any]:
        """
        Get configuration with caching and validation.

        Args:
            config_type: Type of configuration to load
            use_cache: Whether to use cached configuration
            validate: Whether to validate the configuration

        Returns:
            Configuration dictionary
        """
        cache_key = f"{config_type}_{self.environment}"

        # Check cache first
        if use_cache and cache_key in self.cache:
            entry = self.cache[cache_key]

            # Check if file has changed
            if entry.file_path and entry.file_path.exists():
                current_checksum = self._calculate_file_checksum(entry.file_path)
                if current_checksum == entry.checksum:
                    return entry.data.copy()

            # Cache is stale, remove it
            with self.cache_lock:
                del self.cache[cache_key]

        # Load configuration
        config = self._load_config(config_type)

        # Validate if requested
        if validate and config_type in self.validator.schemas:
            errors = self.validator.validate_config(config, config_type)
            if errors:
                logger.warning(f"Configuration validation errors for {config_type}: {errors}")

        # Cache the result
        if use_cache:
            file_path = self.config_paths.get(config_type)
            checksum = self._calculate_file_checksum(file_path) if file_path and file_path.exists() else ""

            entry = ConfigCacheEntry(
                data=config.copy(),
                timestamp=time.time(),
                checksum=checksum,
                file_path=file_path
            )

            with self.cache_lock:
                self.cache[cache_key] = entry

        return config

    def _load_config(self, config_type: str) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        # Try environment-specific config first
        env_config_key = f"{config_type}_{self.environment}"
        if env_config_key in self.config_paths:
            config_path = self.config_paths[env_config_key]
            if config_path.exists():
                return self._load_config_file(config_path)

        # Try base config
        if config_type in self.config_paths:
            config_path = self.config_paths[config_type]
            if config_path.exists():
                return self._load_config_file(config_path)

        # Return default configuration
        return self._get_default_config(config_type)

    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            logger.debug(f"Loaded configuration from {config_path}")
            return config

        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}

    def _get_default_config(self, config_type: str) -> Dict[str, Any]:
        """Get default configuration for a config type."""
        defaults = {
            "main": {
                "environment": self.environment,
                "debug": self.environment == "development",
                "version": "1.0.0"
            },
            "trading": {
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "risk_management": {
                    "max_position_size": 0.1,
                    "max_drawdown": 0.05
                },
                "execution": {
                    "default_order_type": "limit",
                    "slippage_tolerance": 0.001
                }
            },
            "risk": {
                "max_position_size": 0.1,
                "max_drawdown": 0.05,
                "stop_loss_percentage": 0.02,
                "take_profit_percentage": 0.05
            },
            "logging": {
                "level": "INFO",
                "file_logging": True,
                "console": True,
                "max_size": 10485760,  # 10MB
                "backup_count": 5
            },
            "api": {
                "host": "localhost",
                "port": 8000,
                "debug": self.environment == "development"
            },
            "database": {
                "type": "sqlite",
                "path": str(PROJECT_ROOT / "data" / "trading.db")
            },
            "notifications": {
                "discord": {
                    "enabled": False,
                    "webhook_url": ""
                },
                "email": {
                    "enabled": False,
                    "smtp_server": "",
                    "smtp_port": 587
                }
            }
        }

        return defaults.get(config_type, {})

    def set_config_value(self, config_type: str, key: str, value: Any,
                        persist: bool = False) -> None:
        """
        Set a configuration value.

        Args:
            config_type: Type of configuration
            key: Configuration key
            value: Value to set
            persist: Whether to persist to file
        """
        config = self.get_config(config_type, use_cache=False)

        # Set nested values using dot notation
        keys = key.split('.')
        target = config

        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value

        # Update cache
        cache_key = f"{config_type}_{self.environment}"
        if cache_key in self.cache:
            with self.cache_lock:
                self.cache[cache_key].data = config.copy()
                self.cache[cache_key].timestamp = time.time()

        # Persist to file if requested
        if persist:
            self._persist_config(config_type, config)

    def _persist_config(self, config_type: str, config: Dict[str, Any]) -> None:
        """Persist configuration to file."""
        if config_type not in self.config_paths:
            logger.warning(f"No config path defined for {config_type}")
            return

        config_path = self.config_paths[config_type]

        try:
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, default=str)

            logger.info(f"Persisted configuration to {config_path}")

        except IOError as e:
            logger.error(f"Failed to persist config to {config_path}: {e}")

    def reload_config(self, config_type: str) -> Dict[str, Any]:
        """
        Force reload configuration from file.

        Args:
            config_type: Type of configuration to reload

        Returns:
            Reloaded configuration
        """
        cache_key = f"{config_type}_{self.environment}"

        # Remove from cache
        with self.cache_lock:
            if cache_key in self.cache:
                del self.cache[cache_key]

        # Reload
        return self.get_config(config_type, use_cache=False)

    def get_config_value(self, config_type: str, key: str,
                        default: Any = None) -> Any:
        """
        Get a specific configuration value.

        Args:
            config_type: Type of configuration
            key: Configuration key (dot notation supported)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        config = self.get_config(config_type)

        # Support dot notation
        keys = key.split('.')
        value = config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except IOError:
            return ""

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        with self.cache_lock:
            return {
                "cached_configs": len(self.cache),
                "cache_entries": list(self.cache.keys()),
                "total_cache_size": sum(
                    len(json.dumps(entry.data)) for entry in self.cache.values()
                )
            }

    def clear_cache(self, config_type: Optional[str] = None) -> None:
        """
        Clear configuration cache.

        Args:
            config_type: Specific config type to clear (None for all)
        """
        with self.cache_lock:
            if config_type:
                cache_key = f"{config_type}_{self.environment}"
                if cache_key in self.cache:
                    del self.cache[cache_key]
            else:
                self.cache.clear()

        logger.info(f"Cleared configuration cache for {config_type or 'all'}")

    def list_available_configs(self) -> List[str]:
        """List all available configuration types."""
        return list(self.config_paths.keys())

    def validate_all_configs(self) -> Dict[str, List[str]]:
        """
        Validate all available configurations.

        Returns:
            Dictionary mapping config types to validation errors
        """
        results = {}

        for config_type in self.list_available_configs():
            if config_type in self.validator.schemas:
                config = self.get_config(config_type, validate=False)
                errors = self.validator.validate_config(config, config_type)
                if errors:
                    results[config_type] = errors

        return results


# Global configuration factory instance
_config_factory = ConfigFactory()

def get_config_factory() -> ConfigFactory:
    """Get the global configuration factory instance."""
    return _config_factory


# Convenience functions for backward compatibility
def get_config(config_type: str = "main", use_cache: bool = True) -> Dict[str, Any]:
    """Get configuration (backward compatibility)."""
    return _config_factory.get_config(config_type, use_cache)


def set_config_value(config_type: str, key: str, value: Any, persist: bool = False) -> None:
    """Set configuration value (backward compatibility)."""
    _config_factory.set_config_value(config_type, key, value, persist)


def get_config_value(config_type: str, key: str, default: Any = None) -> Any:
    """Get configuration value (backward compatibility)."""
    return _config_factory.get_config_value(config_type, key, default)


def reload_config(config_type: str) -> Dict[str, Any]:
    """Reload configuration (backward compatibility)."""
    return _config_factory.reload_config(config_type)


# Environment-specific configuration helpers
def get_trading_config() -> Dict[str, Any]:
    """Get trading-specific configuration."""
    return get_config("trading")


def get_risk_config() -> Dict[str, Any]:
    """Get risk management configuration."""
    return get_config("risk")


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration."""
    return get_config("logging")


def get_api_config() -> Dict[str, Any]:
    """Get API configuration."""
    return get_config("api")


def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    return get_config("database")


def get_notifications_config() -> Dict[str, Any]:
    """Get notifications configuration."""
    return get_config("notifications")
