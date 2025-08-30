"""
utils/config_loader.py

Handles configuration loading, validation, and management with support for:
- JSON configuration files
- Environment variable overrides
- Runtime modifications
- Schema validation
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import copy
from dotenv import load_dotenv
import jsonschema
from pydantic import BaseModel, ValidationError
from deepmerge import always_merger

logger = logging.getLogger(__name__)

# Load environment variables from .env file if present (explicit)
dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Fallback to default behavior (searches for .env in parents)
    load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration model."""

    host: str = "localhost"
    port: int = 5432
    username: str
    password: str
    database: str = "crypto_bot"
    pool_size: int = 10


class ExchangeConfig(BaseModel):
    """Exchange configuration model.

    NOTE: API credentials are now optional in the JSON config and should
    be provided via environment variables for production use. See .env.example.
    """
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_passphrase: Optional[str] = None
    sandbox: bool = False
    timeout: int = 30000
    rate_limit: int = 10


class RiskConfig(BaseModel):
    """Risk management configuration model."""

    stop_loss: float = 0.02
    take_profit: float = 0.04
    trailing_stop: bool = True
    position_size: float = 0.1
    max_position_size: float = 0.3
    risk_reward_ratio: float = 2.0
    max_daily_drawdown: float = 0.1


class ConfigModel(BaseModel):
    """Main configuration model."""

    environment: Dict[str, Any]
    exchange: ExchangeConfig
    trading: Dict[str, Any]
    risk_management: RiskConfig
    backtesting: Dict[str, Any]
    strategies: Dict[str, Any]
    monitoring: Dict[str, Any]
    notifications: Dict[str, Any]
    logging: Dict[str, Any]
    advanced: Dict[str, Any]


class ConfigLoader:
    """
    Loads and manages configuration with environment variable support.
    Handles validation, defaults, and runtime modifications.
    """

    DEFAULT_CONFIG = {
        "environment": {"mode": "paper", "debug": False, "log_level": "INFO"},
        "trading": {
            "initial_balance": 1000.0,
            "max_concurrent_trades": 3,
            "slippage": 0.001,
            "order_timeout": 60,
            "trade_fee": 0.001,
            "order": {
                # Dynamic take-profit toggle and defaults
                "dynamic_take_profit": False,
                "dynamic_tp": {
                    "enabled": False,
                    "trend_window": "1h",
                    "trend_lookback": 20
                },
                # Profit-based re-entry configuration
                "reentry": {
                    "enabled": False,
                    "profit_threshold": 50.0,     # minimum profit (in quote currency) to consider re-entry
                    "reentry_fraction": 0.5,      # fraction of profit to deploy for re-entry
                    "max_reentries": 1
                }
            },
        },
        "backtesting": {
            "data_dir": "historical_data",
            "commission": 0.001,
            "slippage_model": "fixed",
        },
        # Risk management defaults (medium-term feature support)
        "risk_management": {
            "require_stop_loss": True,
            "position_sizing_method": "fixed_percent",  # fixed_percent | volatility | kelly
            "fixed_percent": 0.1,  # When using fixed_percent, uses this fraction of balance
            "max_position_size": 0.3,
            "max_daily_drawdown": 0.1,
            "risk_reward_ratio": 2.0,
        },
        "portfolio": {
            "pair_allocations": {}
        },
        "logging": {
            "file_logging": True,
            "log_file": "logs/crypto_bot.log",
            "max_size": 10485760,
            "backup_count": 5,
        },
        "ml": {
            "enabled": False,
            "model_path": "models/lightgbm_model.pkl",
            "confidence_threshold": 0.6
        },
    }

    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "environment": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["live", "paper", "backtest"]},
                    "debug": {"type": "boolean"},
                    "log_level": {
                        "type": "string",
                        "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                    },
                },
                "required": ["mode"],
            },
            "exchange": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "api_key": {"type": "string"},
                    "api_secret": {"type": "string"},
                    "api_passphrase": {"type": "string"},
                    "sandbox": {"type": "boolean"},
                    "timeout": {"type": "number", "minimum": 1000},
                    "rate_limit": {"type": "number", "minimum": 1},
                },
                "required": ["name"],
            },
            "risk_management": {
                "type": "object",
                "properties": {
                    "stop_loss": {"type": "number", "minimum": 0.001, "maximum": 0.5},
                    "take_profit": {"type": "number", "minimum": 0.001, "maximum": 0.5},
                    "position_size": {
                        "type": "number",
                        "minimum": 0.01,
                        "maximum": 1.0,
                    },
                    "max_position_size": {
                        "type": "number",
                        "minimum": 0.01,
                        "maximum": 1.0,
                    },
                },
            },
            "ml": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "model_path": {"type": "string"},
                    "confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                }
            },
        },
        "required": ["environment", "exchange"],
    }

    def __init__(self):
        """Initialize the config loader."""
        # Use a deep copy to avoid accidental shared references between DEFAULT_CONFIG and runtime config
        self._config = copy.deepcopy(self.DEFAULT_CONFIG)
        self._original_config = None
        self._env_prefix = "CRYPTOBOT_"

    def load_config(self, config_path: str = "config.json") -> Dict[str, Any]:
        """
        Load configuration from file with environment variable overrides.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            Merged configuration dictionary
        """
        try:
            # Load from file
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, "r") as f:
                    file_config = json.load(f)
                self._config = always_merger.merge(self._config, file_config)

            # Apply environment variable overrides
            self._apply_env_overrides()

            # Validate the final config
            self._validate_config()

            # Keep original for reference (deep copy to preserve full structure)
            self._original_config = copy.deepcopy(self._config)

            logger.info("Configuration loaded successfully")
            return self._config

        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to the configuration.

        Environment variables are expected to be prefixed with CRYPTOBOT_ and use
        underscores instead of dots for nested keys. For example:
          - config key: exchange.api_key
          - env var name: CRYPTOBOT_EXCHANGE_API_KEY
        """
        for key, value in self._flatten_config(self._config).items():
            # Convert dot notation to underscore to form a valid env var name
            env_key = f"{self._env_prefix}{key.upper().replace('.', '_')}"
            if env_key in os.environ:
                # Try to parse JSON values first (to support lists/dicts)
                try:
                    env_value = json.loads(os.environ[env_key])
                except json.JSONDecodeError:
                    env_value = os.environ[env_key]

                self._set_config_value(key, env_value)

    def _flatten_config(
        self, config: Dict[str, Any], parent_key: str = ""
    ) -> Dict[str, Any]:
        """Flatten nested configuration dictionary.

        Enhancements:
        - Preserves whole-list entries at their parent key (so env overrides can target entire lists)
        - Also exposes indexed list elements (key.0, key.1, ...) for fine-grained overrides when useful

        Args:
            config: Nested configuration dictionary.
            parent_key: Internal use for recursion to build dot-separated keys.

        Returns:
            A flat dictionary mapping dot-notated keys to values.
        """
        items: Dict[str, Any] = {}
        for key, value in config.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                items.update(self._flatten_config(value, new_key))
            elif isinstance(value, list):
                # Expose the whole list under the parent key (useful for JSON env overrides)
                items[new_key] = value
                # Also expose indexed entries for convenience
                for i, v in enumerate(value):
                    idx_key = f"{new_key}.{i}"
                    if isinstance(v, dict):
                        items.update(self._flatten_config(v, idx_key))
                    else:
                        items[idx_key] = v
            else:
                items[new_key] = value
        return items

    def _set_config_value(self, key_path: str, value: Any) -> None:
        """Set a value in the nested config using dot notation.

        This implementation supports:
        - dict traversal/creation for missing keys
        - setting entire lists (when env value is JSON array)
        - indexed list assignment when key path contains numeric segments (e.g., 'pairs.0.symbol')
        """
        keys = key_path.split(".")
        current = self._config

        for key in keys[:-1]:
            # Handle numeric index into a list
            if key.isdigit():
                idx = int(key)
                if isinstance(current, list):
                    # Ensure list is large enough
                    while len(current) <= idx:
                        current.append({})
                    current = current[idx]
                else:
                    # If current is a dict and the digit key is unexpected, create a list placeholder
                    # under this key so later assignment can proceed. This is a pragmatic fallback.
                    if key not in current or not isinstance(current.get(key), list):
                        current[key] = []
                    current = current[key]
            else:
                if key not in current or not isinstance(current[key], (dict, list)):
                    current[key] = {}
                current = current[key]

        last = keys[-1]
        if last.isdigit():
            idx = int(last)
            if isinstance(current, list):
                while len(current) <= idx:
                    current.append(None)
                current[idx] = value
            else:
                # If current is a dict, fallback to using the string key
                current[last] = value
        else:
            current[last] = value

    def _validate_config(self) -> None:
        """Validate configuration against schema and models."""
        # JSON Schema validation
        jsonschema.validate(instance=self._config, schema=self.CONFIG_SCHEMA)

        # Pydantic model validation
        try:
            ConfigModel(**self._config)
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., 'exchange.api_key')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        current = self._config

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any, validate: bool = True) -> None:
        """
        Set a configuration value at runtime.

        Args:
            key: Dot-separated key path
            value: Value to set
            validate: Whether to validate after setting
        """
        self._set_config_value(key, value)
        if validate:
            self._validate_config()

    def reset(self) -> None:
        """Reset configuration to originally loaded values."""
        if self._original_config:
            # Use deep copy to avoid sharing mutable references with the stored original
            self._config = copy.deepcopy(self._original_config)
            logger.info("Configuration reset to original values")

    def mask_sensitive(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Return a copy of config with sensitive values masked.

        Args:
            config: Optional config dict to mask (uses current config if None)

        Returns:
            Masked configuration dictionary
        """
        # Use deepcopy to avoid mutating original structures
        cfg: Dict[str, Any] = copy.deepcopy(config or self._config)
        sensitive_keys = ["api_key", "api_secret", "api_passphrase", "password"]

        def mask_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in d.items():
                if isinstance(v, dict):
                    mask_dict(v)
                elif k in sensitive_keys and v:
                    d[k] = "*****"
            return d

        return mask_dict(cfg)

    @classmethod
    def generate_template(cls, path: str = "config_template.json") -> None:
        """
        Generate a configuration template file.

        Args:
            path: Path to save the template file
        """
        template = {
            "environment": {"mode": "paper", "debug": False, "log_level": "INFO"},
            "exchange": {
                "name": "kucoin",
                "api_key": "your_api_key",
                "api_secret": "your_api_secret",
                "api_passphrase": "your_passphrase_if_required",
                "sandbox": False,
                "timeout": 30000,
                "rate_limit": 10,
            },
            "trading": {
                "initial_balance": 1000.0,
                "max_concurrent_trades": 3,
                "slippage": 0.001,
                "order_timeout": 60,
                "trade_fee": 0.001,
                "order": {
                    "dynamic_take_profit": False,
                    "dynamic_tp": {
                        "enabled": False,
                        "trend_window": "1h",
                        "trend_lookback": 20
                    },
                    "reentry": {
                        "enabled": False,
                        "profit_threshold": 50.0,
                        "reentry_fraction": 0.5,
                        "max_reentries": 1
                    }
                },
            },
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "trailing_stop": True,
                "position_size": 0.1,
                "max_position_size": 0.3,
                "risk_reward_ratio": 2.0,
            },
            "backtesting": {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "timeframe": "1h",
                "commission": 0.001,
            },
            "strategies": {
                "default": "RSIStrategy",
                "active_strategies": ["RSIStrategy"],
                "strategy_config": {
                    "RSIStrategy": {
                        "timeframe": "1h",
                        "rsi_period": 14,
                        "rsi_overbought": 70,
                        "rsi_oversold": 30,
                    }
                },
            },
            "notifications": {
                "discord": {"enabled": False, "webhook_url": "your_discord_webhook_url"}
            },
        }

        with open(path, "w") as f:
            json.dump(template, f, indent=2)

        logger.info(f"Configuration template generated at {path}")


# Global configuration loader instance
_config_loader = ConfigLoader()


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration using the global loader."""
    return _config_loader.load_config(config_path)


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value using the global loader."""
    return _config_loader.get(key, default)


def set_config(key: str, value: Any, validate: bool = True) -> None:
    """Set configuration value using the global loader."""
    _config_loader.set(key, value, validate)


def get_masked_config() -> Dict[str, Any]:
    """Get masked configuration using the global loader."""
    return _config_loader.mask_sensitive()
