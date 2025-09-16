"""
Common configuration utilities for the N1V1 trading framework.

This module provides standardized configuration loading, validation,
and management patterns used across core components.
"""

import os
import json
import yaml
import logging
from typing import Any, Callable, Optional, Type, TypeVar
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
from enum import Enum

from ..logging_utils import get_structured_logger

logger = get_structured_logger(__name__)

T = TypeVar('T')


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"


@dataclass
class ConfigSource:
    """Configuration source information."""
    path: str
    format: ConfigFormat
    priority: int = 0
    required: bool = False
    last_modified: Optional[float] = None
    checksum: Optional[str] = None


@dataclass
class ConfigValidationRule:
    """Configuration validation rule."""
    field_path: str
    rule_type: str
    value: Any
    error_message: str


class ConfigLoader:
    """Centralized configuration loader with multiple sources and validation."""

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.sources: list[ConfigSource] = []
        self.cache: dict[str, dict[str, Any]] = {}
        self.validators: dict[str, list[ConfigValidationRule]] = {}
        self.logger = get_structured_logger("core.config_loader")

    def add_source(self,
                   path: str,
                   format: ConfigFormat,
                   priority: int = 0,
                   required: bool = False) -> None:
        """Add a configuration source."""
        source = ConfigSource(
            path=path,
            format=format,
            priority=priority,
            required=required
        )
        self.sources.append(source)
        self.sources.sort(key=lambda s: s.priority, reverse=True)  # Higher priority first

        self.logger.info("Added configuration source", {
            "path": path,
            "format": format.value,
            "priority": priority,
            "required": required
        })

    def add_validator(self, config_key: str, rules: list[ConfigValidationRule]) -> None:
        """Add validation rules for a configuration section."""
        self.validators[config_key] = rules

    def load_config(self, key: Optional[str] = None) -> dict[str, Any]:
        """
        Load configuration from all sources.

        Args:
            key: Optional specific configuration key to load

        Returns:
            Merged configuration dictionary
        """
        merged_config = {}

        for source in self.sources:
            try:
                config_data = self._load_source(source)
                if config_data:
                    if key and key in config_data:
                        merged_config.update(config_data[key])
                    elif not key:
                        merged_config.update(config_data)

            except Exception as e:
                if source.required:
                    self.logger.error(f"Failed to load required config source: {source.path}", {
                        "error": str(e),
                        "source": source.path
                    })
                    raise
                else:
                    self.logger.warning(f"Failed to load optional config source: {source.path}", {
                        "error": str(e),
                        "source": source.path
                    })

        # Validate configuration
        self._validate_config(merged_config, key)

        return merged_config

    def _load_source(self, source: ConfigSource) -> dict[str, Any]:
        """Load configuration from a specific source."""
        file_path = self.base_path / source.path

        # Check if file has changed
        if file_path.exists():
            stat = file_path.stat()
            current_mtime = stat.st_mtime

            # Calculate checksum for change detection
            with open(file_path, 'rb') as f:
                current_checksum = hashlib.md5(f.read()).hexdigest()

            if (source.last_modified == current_mtime and
                source.checksum == current_checksum):
                # Return cached version
                cache_key = str(file_path)
                if cache_key in self.cache:
                    return self.cache[cache_key]

            source.last_modified = current_mtime
            source.checksum = current_checksum

        if source.format == ConfigFormat.JSON:
            return self._load_json(file_path)
        elif source.format == ConfigFormat.YAML:
            return self._load_yaml(file_path)
        elif source.format == ConfigFormat.ENV:
            return self._load_env()
        else:
            raise ValueError(f"Unsupported config format: {source.format}")

    def _load_json(self, file_path: Path) -> dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Cache the loaded data
            self.cache[str(file_path)] = data
            return data

        except FileNotFoundError:
            if self._is_required_source(file_path):
                raise
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {file_path}", {
                "error": str(e)
            })
            raise

    def _load_yaml(self, file_path: Path) -> dict[str, Any]:
        """Load YAML configuration file."""
        try:
            import yaml
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Cache the loaded data
            self.cache[str(file_path)] = data
            return data or {}

        except ImportError:
            self.logger.warning("PyYAML not installed, skipping YAML config loading")
            return {}
        except FileNotFoundError:
            if self._is_required_source(file_path):
                raise
            return {}
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in config file: {file_path}", {
                "error": str(e)
            })
            raise

    def _load_env(self) -> dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}

        # Load common trading configuration from environment
        env_mappings = {
            'EXCHANGE_API_KEY': ('exchange', 'api_key'),
            'EXCHANGE_SECRET': ('exchange', 'secret'),
            'EXCHANGE_BASE_CURRENCY': ('exchange', 'base_currency'),
            'TRADING_MODE': ('environment', 'mode'),
            'LOG_LEVEL': ('logging', 'level'),
            'LOG_SENSITIVITY': ('logging', 'sensitivity'),
            'REDIS_HOST': ('cache', 'host'),
            'REDIS_PORT': ('cache', 'port'),
            'DATABASE_URL': ('database', 'url')
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_config(config, config_path, value)

        return config

    def _is_required_source(self, file_path: Path) -> bool:
        """Check if a source file is required."""
        for source in self.sources:
            if str(self.base_path / source.path) == str(file_path):
                return source.required
        return False

    def _set_nested_config(self, config: dict[str, Any], path: tuple, value: Any) -> None:
        """Set a value in a nested configuration dictionary."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _validate_config(self, config: dict[str, Any], key: Optional[str] = None) -> None:
        """Validate configuration against defined rules."""
        validation_key = key or 'global'

        if validation_key not in self.validators:
            return

        rules = self.validators[validation_key]
        errors = []

        for rule in rules:
            try:
                field_value = self._get_nested_value(config, rule.field_path.split('.'))

                if not self._validate_rule(field_value, rule):
                    errors.append({
                        'field': rule.field_path,
                        'error': rule.error_message,
                        'value': field_value
                    })

            except KeyError:
                if rule.rule_type != 'optional':
                    errors.append({
                        'field': rule.field_path,
                        'error': f"Required field missing: {rule.field_path}"
                    })

        if errors:
            error_msg = f"Configuration validation failed: {errors}"
            self.logger.error("Configuration validation failed", {
                "errors": errors,
                "config_key": validation_key
            })
            raise ValueError(error_msg)

    def _get_nested_value(self, config: dict[str, Any], path: list[str]) -> Any:
        """Get a nested value from configuration."""
        current = config
        for key in path:
            current = current[key]
        return current

    def _validate_rule(self, value: Any, rule: ConfigValidationRule) -> bool:
        """Validate a single configuration rule."""
        if rule.rule_type == 'required':
            return value is not None
        elif rule.rule_type == 'type':
            return isinstance(value, rule.value)
        elif rule.rule_type == 'range':
            min_val, max_val = rule.value
            return min_val <= value <= max_val
        elif rule.rule_type == 'enum':
            return value in rule.value
        elif rule.rule_type == 'pattern':
            import re
            return bool(re.match(rule.value, str(value)))
        elif rule.rule_type == 'optional':
            return True  # Optional fields always pass
        else:
            return True  # Unknown rule types pass by default


class ConfigManager:
    """Configuration manager with hot-reloading and validation."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.loader = ConfigLoader(self.config_dir)
        self.current_config: dict[str, Any] = {}
        self.config_listeners: list[Callable] = []
        self.logger = get_structured_logger("core.config_manager")

        # Set up default configuration sources
        self._setup_default_sources()

    def _setup_default_sources(self):
        """Set up default configuration sources."""
        # Environment variables (highest priority)
        self.loader.add_source("", ConfigFormat.ENV, priority=100)

        # Local overrides
        self.loader.add_source("config.local.json", ConfigFormat.JSON, priority=90, required=False)
        self.loader.add_source("config.local.yaml", ConfigFormat.YAML, priority=90, required=False)

        # Main configuration files
        self.loader.add_source("config.json", ConfigFormat.JSON, priority=50, required=False)
        self.loader.add_source("config.yaml", ConfigFormat.YAML, priority=50, required=False)

        # Default configuration
        self.loader.add_source("config.default.json", ConfigFormat.JSON, priority=10, required=False)

    def load_config(self) -> dict[str, Any]:
        """Load and validate configuration."""
        try:
            self.current_config = self.loader.load_config()
            self._validate_required_sections()
            self._notify_listeners()

            self.logger.info("Configuration loaded successfully", {
                "sources_loaded": len(self.loader.sources),
                "config_keys": list(self.current_config.keys())
            })

            return self.current_config

        except Exception as e:
            self.logger.error("Failed to load configuration", {
                "error": str(e)
            })
            raise

    def get_config(self, key: Optional[str] = None) -> Any:
        """Get configuration value."""
        if key is None:
            return self.current_config

        keys = key.split('.')
        value = self.current_config

        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return None

    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        keys = key.split('.')
        config = self.current_config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        self._notify_listeners()

    def add_config_listener(self, listener: Callable) -> None:
        """Add a configuration change listener."""
        self.config_listeners.append(listener)

    def _validate_required_sections(self):
        """Validate that required configuration sections are present."""
        required_sections = ['exchange', 'trading', 'risk_management']

        for section in required_sections:
            if section not in self.current_config:
                self.logger.warning(f"Required configuration section missing: {section}")

    def _notify_listeners(self):
        """Notify all configuration listeners of changes."""
        for listener in self.config_listeners:
            try:
                listener(self.current_config)
            except Exception as e:
                self.logger.error("Configuration listener failed", {
                    "error": str(e)
                })

    def reload_config(self) -> dict[str, Any]:
        """Reload configuration from sources."""
        self.logger.info("Reloading configuration")
        return self.load_config()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(key: Optional[str] = None) -> Any:
    """Load configuration using the global config manager."""
    manager = get_config_manager()
    if not manager.current_config:
        manager.load_config()
    return manager.get_config(key)


def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value with optional default."""
    value = load_config(key)
    return value if value is not None else default


# Utility functions for common configuration patterns
def validate_trading_config(config: dict[str, Any]) -> list[str]:
    """Validate trading configuration."""
    errors = []

    # Required fields
    required_fields = [
        'exchange.api_key',
        'exchange.secret',
        'trading.symbol',
        'trading.initial_balance'
    ]

    for field in required_fields:
        if get_config(field) is None:
            errors.append(f"Required field missing: {field}")

    # Validate balance
    balance = get_config('trading.initial_balance')
    if balance is not None and balance <= 0:
        errors.append("Initial balance must be positive")

    return errors


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple configuration dictionaries."""
    result = {}

    for config in configs:
        _deep_merge(result, config)

    return result


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Deep merge source dictionary into target."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
