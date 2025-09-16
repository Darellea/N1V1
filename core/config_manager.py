"""
Centralized configuration management system.

This module provides a unified configuration system that replaces
hardcoded values throughout the codebase with configurable parameters.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict

from .interfaces import (
    CacheConfig,
    MemoryConfig,
    DataManagerConfig,
    PerformanceTrackerConfig,
    TradingCoordinatorConfig
)

logger = logging.getLogger(__name__)


@dataclass
class CoreConfig:
    """Centralized configuration for all core components."""

    # Cache configuration
    cache: CacheConfig = None

    # Memory management configuration
    memory: MemoryConfig = None

    # Data manager configuration
    data_manager: DataManagerConfig = None

    # Performance tracker configuration
    performance_tracker: PerformanceTrackerConfig = None

    # Trading coordinator configuration
    trading_coordinator: TradingCoordinatorConfig = None

    # Bot engine configuration
    bot_engine: Dict[str, Any] = None

    def __post_init__(self):
        if self.cache is None:
            self.cache = CacheConfig()
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.data_manager is None:
            self.data_manager = DataManagerConfig()
        if self.performance_tracker is None:
            self.performance_tracker = PerformanceTrackerConfig()
        if self.trading_coordinator is None:
            self.trading_coordinator = TradingCoordinatorConfig()
        if self.bot_engine is None:
            self.bot_engine = {}


class ConfigManager:
    """
    Centralized configuration manager for the trading bot.

    Handles loading, validation, and access to all configuration parameters,
    replacing hardcoded values throughout the codebase.
    """

    def __init__(self, config_file: Optional[str] = None):
        """Initialize the configuration manager.

        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file or self._find_config_file()
        self._config = CoreConfig()
        self._overrides: Dict[str, Any] = {}

        # Load configuration
        self._load_config()

    def _find_config_file(self) -> str:
        """Find configuration file in standard locations."""
        search_paths = [
            "config.json",
            "config/config.json",
            "../config.json",
            "core/config.json"
        ]

        for path in search_paths:
            if os.path.exists(path):
                return path

        # Return default path if no config file found
        return "config.json"

    def _load_config(self) -> None:
        """Load configuration from file and environment variables."""
        # Load from file if it exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                self._merge_config(file_config)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")
                self._load_defaults()

        # Override with environment variables
        self._load_from_environment()

        # Apply any runtime overrides
        self._apply_overrides()

    def _merge_config(self, file_config: Dict[str, Any]) -> None:
        """Merge file configuration with defaults."""
        # Handle cache configuration
        if "cache" in file_config:
            cache_config = CacheConfig(**file_config["cache"])
            self._config.cache = cache_config

        # Handle memory configuration
        if "memory" in file_config:
            memory_config = MemoryConfig(**file_config["memory"])
            self._config.memory = memory_config

        # Handle data manager configuration
        if "data_manager" in file_config:
            dm_config = DataManagerConfig(**file_config["data_manager"])
            self._config.data_manager = dm_config

        # Handle performance tracker configuration
        if "performance_tracker" in file_config:
            pt_config = PerformanceTrackerConfig(**file_config["performance_tracker"])
            self._config.performance_tracker = pt_config

        # Handle trading coordinator configuration
        if "trading_coordinator" in file_config:
            tc_config = TradingCoordinatorConfig(**file_config["trading_coordinator"])
            self._config.trading_coordinator = tc_config

        # Handle bot engine configuration
        if "bot_engine" in file_config:
            self._config.bot_engine.update(file_config["bot_engine"])

    def _load_defaults(self) -> None:
        """Load default configuration values."""
        # Configuration is already initialized with defaults in CoreConfig.__post_init__
        logger.info("Using default configuration")

    def _load_from_environment(self) -> None:
        """Load configuration overrides from environment variables."""
        # Cache configuration from environment
        if os.getenv("CACHE_HOST"):
            self._config.cache.host = os.getenv("CACHE_HOST")
        if os.getenv("CACHE_PORT"):
            self._config.cache.port = int(os.getenv("CACHE_PORT"))
        if os.getenv("CACHE_PASSWORD"):
            self._config.cache.password = os.getenv("CACHE_PASSWORD")

        # Memory configuration from environment
        if os.getenv("MAX_MEMORY_MB"):
            self._config.memory.max_memory_mb = float(os.getenv("MAX_MEMORY_MB"))
        if os.getenv("MEMORY_WARNING_MB"):
            self._config.memory.warning_memory_mb = float(os.getenv("MEMORY_WARNING_MB"))
        if os.getenv("MEMORY_CLEANUP_MB"):
            self._config.memory.cleanup_memory_mb = float(os.getenv("MEMORY_CLEANUP_MB"))

        # Data manager configuration from environment
        if os.getenv("CACHE_ENABLED"):
            self._config.data_manager.cache_enabled = os.getenv("CACHE_ENABLED").lower() == "true"
        if os.getenv("CACHE_TTL"):
            self._config.data_manager.cache_ttl = int(os.getenv("CACHE_TTL"))

        # Performance tracker configuration from environment
        if os.getenv("STARTING_BALANCE"):
            self._config.performance_tracker.starting_balance = float(os.getenv("STARTING_BALANCE"))

    def _apply_overrides(self) -> None:
        """Apply runtime configuration overrides."""
        for key, value in self._overrides.items():
            self._set_nested_value(self._config, key.split("."), value)

    def _set_nested_value(self, obj: Any, keys: list, value: Any) -> None:
        """Set a nested configuration value."""
        if len(keys) == 1:
            setattr(obj, keys[0], value)
        else:
            attr = getattr(obj, keys[0])
            if attr is None:
                # Create nested object if it doesn't exist
                setattr(obj, keys[0], type(attr)())
                attr = getattr(obj, keys[0])
            self._set_nested_value(attr, keys[1:], value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key.

        Args:
            key: Dot-separated configuration key (e.g., "cache.ttl_config.market_ticker")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            keys = key.split(".")
            value = self._config

            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            return value
        except Exception:
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-separated key.

        Args:
            key: Dot-separated configuration key
            value: Value to set
        """
        self._overrides[key] = value
        self._apply_overrides()

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section as dictionary.

        Args:
            section: Section name

        Returns:
            Dictionary of section configuration
        """
        try:
            section_obj = getattr(self._config, section)
            if hasattr(section_obj, '__dict__'):
                return asdict(section_obj)
            elif isinstance(section_obj, dict):
                return section_obj.copy()
            else:
                return {}
        except Exception:
            return {}

    def save_config(self, file_path: Optional[str] = None) -> bool:
        """Save current configuration to file.

        Args:
            file_path: Path to save configuration (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            save_path = file_path or self.config_file

            # Convert config to dictionary
            config_dict = {
                "cache": asdict(self._config.cache),
                "memory": asdict(self._config.memory),
                "data_manager": asdict(self._config.data_manager),
                "performance_tracker": asdict(self._config.performance_tracker),
                "trading_coordinator": asdict(self._config.trading_coordinator),
                "bot_engine": self._config.bot_engine
            }

            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)

            logger.info(f"Configuration saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def reload_config(self) -> bool:
        """Reload configuration from file.

        Returns:
            True if successful, False otherwise
        """
        try:
            self._load_config()
            logger.info("Configuration reloaded")
            return True
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False

    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration."""
        return self._config.cache

    def get_memory_config(self) -> MemoryConfig:
        """Get memory configuration."""
        return self._config.memory

    def get_data_manager_config(self) -> DataManagerConfig:
        """Get data manager configuration."""
        return self._config.data_manager

    def get_performance_tracker_config(self) -> PerformanceTrackerConfig:
        """Get performance tracker configuration."""
        return self._config.performance_tracker

    def get_trading_coordinator_config(self) -> TradingCoordinatorConfig:
        """Get trading coordinator configuration."""
        return self._config.trading_coordinator

    def get_bot_engine_config(self) -> Dict[str, Any]:
        """Get bot engine configuration."""
        return self._config.bot_engine.copy()

    def validate_config(self) -> List[str]:
        """Validate current configuration.

        Returns:
            List of validation error messages
        """
        errors = []

        # Validate cache configuration
        if self._config.cache.port <= 0 or self._config.cache.port > 65535:
            errors.append("Cache port must be between 1 and 65535")

        if self._config.cache.max_cache_size <= 0:
            errors.append("Cache max size must be positive")

        # Validate memory configuration
        if self._config.memory.max_memory_mb <= 0:
            errors.append("Max memory must be positive")

        if self._config.memory.warning_memory_mb >= self._config.memory.max_memory_mb:
            errors.append("Warning memory must be less than max memory")

        if self._config.memory.cleanup_memory_mb >= self._config.memory.warning_memory_mb:
            errors.append("Cleanup memory must be less than warning memory")

        # Validate performance tracker configuration
        if self._config.performance_tracker.starting_balance <= 0:
            errors.append("Starting balance must be positive")

        return errors

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging."""
        return {
            "config_file": self.config_file,
            "cache_enabled": self._config.cache is not None,
            "memory_monitoring_enabled": self._config.memory.enable_monitoring,
            "data_cache_enabled": self._config.data_manager.cache_enabled,
            "performance_tracking_enabled": self._config.performance_tracker.enable_detailed_tracking,
            "validation_errors": self.validate_config()
        }


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config_value(key: str, default: Any = None) -> Any:
    """Get configuration value by key (convenience function)."""
    return get_config_manager().get(key, default)

def set_config_value(key: str, value: Any) -> None:
    """Set configuration value by key (convenience function)."""
    get_config_manager().set(key, value)
