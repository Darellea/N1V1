"""
Centralized configuration management system.

This module provides a unified configuration system that replaces
hardcoded values throughout the codebase with configurable parameters.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict
from pydantic import BaseModel, ValidationError, Field
from pydantic_core import PydanticCustomError
import jsonschema

from .interfaces import (
    CacheConfig,
    MemoryConfig,
    DataManagerConfig,
    PerformanceTrackerConfig,
    TradingCoordinatorConfig
)
from utils.security import get_secret, SecurityException

logger = logging.getLogger(__name__)


# Pydantic models for strict validation
class EnvironmentConfig(BaseModel):
    """Environment configuration with strict validation."""
    mode: str = Field(..., pattern="^(live|paper|backtest)$", description="Trading mode")
    debug: bool = False
    log_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")


class ExchangeConfig(BaseModel):
    """Exchange configuration with strict validation."""
    name: str = Field(..., min_length=1, description="Exchange name")
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_passphrase: Optional[str] = None
    sandbox: bool = False
    timeout: int = Field(30000, ge=1000, le=120000, description="API timeout in ms")
    rate_limit: int = Field(10, ge=1, le=1000, description="API rate limit")


class RiskManagementConfig(BaseModel):
    """Risk management configuration with strict validation."""
    stop_loss: float = Field(0.02, ge=0.001, le=0.5, description="Stop loss percentage")
    take_profit: float = Field(0.04, ge=0.001, le=0.5, description="Take profit percentage")
    trailing_stop: bool = True
    position_size: float = Field(0.1, ge=0.01, le=1.0, description="Position size percentage")
    max_position_size: float = Field(0.3, ge=0.01, le=1.0, description="Max position size percentage")
    risk_reward_ratio: float = Field(2.0, ge=1.0, le=10.0, description="Risk-reward ratio")
    max_daily_drawdown: float = Field(0.1, ge=0.01, le=1.0, description="Max daily drawdown")
    circuit_breaker_enabled: bool = Field(True, description="Circuit breaker enabled")


class MonitoringConfig(BaseModel):
    """Monitoring configuration with strict validation."""
    enabled: bool = True
    update_interval: int = Field(5, ge=1, le=300, description="Update interval in seconds")
    alert_on_errors: bool = True
    log_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")


class StrategiesConfig(BaseModel):
    """Strategies configuration with strict validation."""
    default: str = Field("RSIStrategy", min_length=1, description="Default strategy")
    active_strategies: List[str] = Field(default_factory=list, description="Active strategies list")
    max_concurrent_strategies: int = Field(3, ge=1, le=10, description="Max concurrent strategies")


class MainConfig(BaseModel):
    """Main configuration model with strict validation."""
    environment: EnvironmentConfig
    exchange: ExchangeConfig
    risk_management: RiskManagementConfig
    monitoring: MonitoringConfig
    strategies: StrategiesConfig

    class Config:
        extra = "allow"  # Allow extra known sections


# Safe defaults for missing configuration
SAFE_DEFAULTS = {
    "environment": {
        "mode": "paper",
        "debug": False,
        "log_level": "INFO"
    },
    "exchange": {
        "name": "kucoin",
        "sandbox": True,
        "timeout": 30000,
        "rate_limit": 10
    },
    "risk_management": {
        "stop_loss": 0.02,
        "take_profit": 0.04,
        "trailing_stop": True,
        "position_size": 0.1,
        "max_position_size": 0.3,
        "risk_reward_ratio": 2.0,
        "max_daily_drawdown": 0.1,
        "circuit_breaker_enabled": True
    },
    "monitoring": {
        "enabled": True,
        "update_interval": 5,
        "alert_on_errors": True,
        "log_level": "INFO"
    },
    "strategies": {
        "default": "RSIStrategy",
        "active_strategies": ["RSIStrategy"],
        "max_concurrent_strategies": 3
    }
}


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
        """Load configuration from file and environment variables with validation."""
        config_dict = {}

        # Load from file if it exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                config_dict.update(file_config)
                logger.info(f"Loaded configuration from {self.config_file}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in configuration file {self.config_file}: {e}")
                raise ValueError(f"Invalid JSON in configuration file: {e}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")
                raise ValueError(f"Failed to load configuration file: {e}")

        # Apply safe defaults for missing sections
        config_dict = self.apply_safe_defaults(config_dict)

        # Validate main configuration schema
        validation_errors = self.validate_main_config(config_dict)
        if validation_errors:
            error_msg = f"Configuration validation failed: {'; '.join(validation_errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate environment-specific requirements
        env_errors = self.validate_environment_config(config_dict)
        if env_errors:
            error_msg = f"Environment validation failed: {'; '.join(env_errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Merge validated config with existing config
        self._merge_config(config_dict)

        # Override with environment variables
        self._load_from_environment()

        # Apply any runtime overrides
        self._apply_overrides()

        logger.info("Configuration loaded and validated successfully")

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
        """Load configuration overrides from environment variables and secure secrets."""
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

        # Load secrets securely (async operation)
        try:
            # Try to run async secret loading in current event loop if available
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is running, we can't use run_until_complete
                # Defer secret loading to a separate async call
                logger.info("Event loop running, deferring secure secret loading")
                # We'll load secrets later via async method
            else:
                # No event loop running, we can load secrets synchronously
                loop.run_until_complete(self._load_secrets_securely())
        except RuntimeError:
            # No event loop, defer loading
            logger.info("No event loop available, deferring secure secret loading")

    async def _load_secrets_securely(self) -> None:
        """Load secrets securely from Vault/KMS with fallback to environment."""
        try:
            # Load exchange secrets
            api_key = await get_secret("exchange_api_key")
            if api_key:
                self._config.exchange.api_key = api_key
                logger.info("Loaded exchange API key from secure storage")

            api_secret = await get_secret("exchange_api_secret")
            if api_secret:
                self._config.exchange.api_secret = api_secret
                logger.info("Loaded exchange API secret from secure storage")

            api_passphrase = await get_secret("exchange_api_passphrase")
            if api_passphrase:
                self._config.exchange.api_passphrase = api_passphrase
                logger.info("Loaded exchange API passphrase from secure storage")

            # Load Discord secrets
            discord_token = await get_secret("discord_token")
            if discord_token:
                # Store in a way that can be accessed later
                self._overrides["discord_token"] = discord_token
                logger.info("Loaded Discord token from secure storage")

            discord_channel_id = await get_secret("discord_channel_id")
            if discord_channel_id:
                self._overrides["discord_channel_id"] = discord_channel_id
                logger.info("Loaded Discord channel ID from secure storage")

            discord_webhook_url = await get_secret("discord_webhook_url")
            if discord_webhook_url:
                self._overrides["discord_webhook_url"] = discord_webhook_url
                logger.info("Loaded Discord webhook URL from secure storage")

            # Load API key
            api_key = await get_secret("api_key")
            if api_key:
                self._overrides["api_key"] = api_key
                logger.info("Loaded API key from secure storage")

        except SecurityException as e:
            env_mode = os.getenv("ENV", "live").lower()
            if env_mode in ["live", "production"]:
                logger.error(f"Failed to load required secrets in production mode: {e}")
                raise
            else:
                logger.warning(f"Failed to load secrets securely, will use environment fallback: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error loading secrets securely: {e}")

    async def ensure_secrets_loaded(self) -> None:
        """Ensure secrets are loaded securely (call this from async contexts)."""
        await self._load_secrets_securely()
        self._apply_overrides()

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

    def validate_main_config(self, config_dict: Dict[str, Any]) -> List[str]:
        """Validate main configuration sections with strict schema validation.

        Args:
            config_dict: Configuration dictionary to validate

        Returns:
            List of validation error messages
        """
        errors = []

        try:
            # Validate against Pydantic model (strict validation)
            MainConfig(**config_dict)
        except ValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                error_msg = f"Field '{field_path}': {error['msg']}"
                if "ctx" in error and error["ctx"]:
                    error_msg += f" (expected: {error['ctx']})"
                errors.append(error_msg)
                logger.error(f"Configuration validation error: {error_msg}")
        except Exception as e:
            errors.append(f"Unexpected validation error: {str(e)}")
            logger.error(f"Unexpected validation error: {str(e)}")

        return errors

    def validate_environment_config(self, config_dict: Dict[str, Any]) -> List[str]:
        """Validate environment-specific configuration requirements.

        Args:
            config_dict: Configuration dictionary to validate

        Returns:
            List of validation error messages
        """
        errors = []
        env_config = config_dict.get("environment", {})
        mode = env_config.get("mode", "paper").lower()

        if mode == "live":
            # Live mode requirements
            exchange_config = config_dict.get("exchange", {})

            # Check for API keys
            if not exchange_config.get("api_key"):
                errors.append("Live mode requires exchange.api_key to be set")
                logger.error("Live mode validation failed: missing exchange.api_key")

            if not exchange_config.get("api_secret"):
                errors.append("Live mode requires exchange.api_secret to be set")
                logger.error("Live mode validation failed: missing exchange.api_secret")

            # Check circuit breaker
            risk_config = config_dict.get("risk_management", {})
            if not risk_config.get("circuit_breaker_enabled", False):
                errors.append("Live mode requires risk_management.circuit_breaker_enabled = true")
                logger.error("Live mode validation failed: circuit_breaker_enabled must be true")

            # Check sandbox mode
            if exchange_config.get("sandbox", False):
                errors.append("Live mode cannot use sandbox exchange")
                logger.error("Live mode validation failed: sandbox mode not allowed")

        elif mode in ["paper", "backtest"]:
            # Paper/backtest mode - more relaxed but still validate basics
            exchange_config = config_dict.get("exchange", {})
            if not exchange_config.get("name"):
                errors.append("Exchange name is required")
                logger.warning("Exchange name missing in paper/backtest mode")

        else:
            errors.append(f"Invalid environment mode: {mode}")
            logger.error(f"Invalid environment mode: {mode}")

        return errors

    def apply_safe_defaults(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply safe defaults to missing configuration sections.

        Args:
            config_dict: Configuration dictionary to update

        Returns:
            Updated configuration dictionary with safe defaults
        """
        updated_config = config_dict.copy()

        for section, defaults in SAFE_DEFAULTS.items():
            if section not in updated_config:
                updated_config[section] = defaults.copy()
                logger.info(f"Applied safe defaults for missing section: {section}")
            else:
                # Apply defaults for missing fields within sections
                section_config = updated_config[section]
                for key, default_value in defaults.items():
                    if key not in section_config:
                        section_config[key] = default_value
                        logger.info(f"Applied safe default for {section}.{key}: {default_value}")

        return updated_config

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
