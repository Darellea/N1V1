"""
Comprehensive tests for core/config_manager.py

Tests configuration validation, safe defaults, environment-specific validation,
and error handling for the hardened configuration system.
"""

import copy
import json
import os
import tempfile

import pytest
from pydantic import ValidationError

from core.config_manager import (
    SAFE_DEFAULTS,
    ConfigManager,
    EnvironmentConfig,
    ExchangeConfig,
    MainConfig,
    MonitoringConfig,
    RiskManagementConfig,
    StrategiesConfig,
)


class TestConfigManagerValidation:
    """Test cases for ConfigManager validation features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")

    def teardown_method(self):
        """Clean up after each test."""
        # Remove temp files
        if os.path.exists(self.config_file):
            os.unlink(self.config_file)
        os.rmdir(self.temp_dir)

    def test_validate_main_config_valid(self):
        """Test validation of valid main configuration."""
        manager = ConfigManager.__new__(ConfigManager)  # Create without __init__

        valid_config = {
            "environment": {"mode": "paper", "debug": False, "log_level": "INFO"},
            "exchange": {
                "name": "kucoin",
                "sandbox": True,
                "timeout": 30000,
                "rate_limit": 10,
            },
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "trailing_stop": True,
                "position_size": 0.1,
                "max_position_size": 0.3,
                "risk_reward_ratio": 2.0,
                "max_daily_drawdown": 0.1,
                "circuit_breaker_enabled": True,
            },
            "monitoring": {
                "enabled": True,
                "update_interval": 5,
                "alert_on_errors": True,
                "log_level": "INFO",
            },
            "strategies": {
                "default": "RSIStrategy",
                "active_strategies": ["RSIStrategy"],
                "max_concurrent_strategies": 3,
            },
        }

        errors = manager.validate_main_config(valid_config)
        assert len(errors) == 0

    def test_validate_main_config_invalid_mode(self):
        """Test validation with invalid environment mode."""
        manager = ConfigManager.__new__(ConfigManager)

        invalid_config = {
            "environment": {"mode": "invalid_mode"},
            "exchange": {"name": "kucoin"},
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "circuit_breaker_enabled": True,
            },
            "monitoring": {"enabled": True, "update_interval": 5},
            "strategies": {"default": "RSIStrategy"},
        }

        errors = manager.validate_main_config(invalid_config)
        assert len(errors) > 0
        assert any("environment.mode" in error for error in errors)

    def test_validate_main_config_missing_required_field(self):
        """Test validation with missing required field."""
        manager = ConfigManager.__new__(ConfigManager)

        invalid_config = {
            "environment": {"mode": "paper"},
            # Missing exchange section
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "circuit_breaker_enabled": True,
            },
            "monitoring": {"enabled": True, "update_interval": 5},
            "strategies": {"default": "RSIStrategy"},
        }

        errors = manager.validate_main_config(invalid_config)
        assert len(errors) > 0
        assert any("exchange" in error for error in errors)

    def test_validate_main_config_invalid_type(self):
        """Test validation with invalid field type."""
        manager = ConfigManager.__new__(ConfigManager)

        invalid_config = {
            "environment": {
                "mode": "paper",
                "debug": "not_a_boolean",
            },  # Should be boolean
            "exchange": {"name": "kucoin"},
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "circuit_breaker_enabled": True,
            },
            "monitoring": {"enabled": True, "update_interval": 5},
            "strategies": {"default": "RSIStrategy"},
        }

        errors = manager.validate_main_config(invalid_config)
        assert len(errors) > 0
        assert any("debug" in error for error in errors)

    def test_validate_main_config_unknown_field(self):
        """Test validation rejects unknown fields."""
        manager = ConfigManager.__new__(ConfigManager)

        invalid_config = {
            "environment": {"mode": "paper"},
            "exchange": {"name": "kucoin"},
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "circuit_breaker_enabled": True,
            },
            "monitoring": {"enabled": True, "update_interval": 5},
            "strategies": {"default": "RSIStrategy"},
            "unknown_section": {"some_field": "value"},  # Unknown section
        }

        errors = manager.validate_main_config(invalid_config)
        assert len(errors) > 0
        assert any(
            "unknown_section" in error or "extra" in error.lower() for error in errors
        )

    def test_validate_environment_config_live_mode_valid(self):
        """Test environment validation for valid live mode config."""
        manager = ConfigManager.__new__(ConfigManager)

        valid_live_config = {
            "environment": {"mode": "live"},
            "exchange": {
                "name": "kucoin",
                "api_key": "test_key",
                "api_secret": "test_secret",
                "sandbox": False,
            },
            "risk_management": {"circuit_breaker_enabled": True},
        }

        errors = manager.validate_environment_config(valid_live_config)
        assert len(errors) == 0

    def test_validate_environment_config_live_mode_missing_api_key(self):
        """Test environment validation for live mode without API key."""
        manager = ConfigManager.__new__(ConfigManager)

        invalid_config = {
            "environment": {"mode": "live"},
            "exchange": {
                "name": "kucoin",
                "api_secret": "test_secret",
                "sandbox": False,
            },
            "risk_management": {"circuit_breaker_enabled": True},
        }

        errors = manager.validate_environment_config(invalid_config)
        assert len(errors) > 0
        assert any("api_key" in error for error in errors)

    def test_validate_environment_config_live_mode_missing_api_secret(self):
        """Test environment validation for live mode without API secret."""
        manager = ConfigManager.__new__(ConfigManager)

        invalid_config = {
            "environment": {"mode": "live"},
            "exchange": {"name": "kucoin", "api_key": "test_key", "sandbox": False},
            "risk_management": {"circuit_breaker_enabled": True},
        }

        errors = manager.validate_environment_config(invalid_config)
        assert len(errors) > 0
        assert any("api_secret" in error for error in errors)

    def test_validate_environment_config_live_mode_circuit_breaker_disabled(self):
        """Test environment validation for live mode with circuit breaker disabled."""
        manager = ConfigManager.__new__(ConfigManager)

        invalid_config = {
            "environment": {"mode": "live"},
            "exchange": {
                "name": "kucoin",
                "api_key": "test_key",
                "api_secret": "test_secret",
                "sandbox": False,
            },
            "risk_management": {"circuit_breaker_enabled": False},
        }

        errors = manager.validate_environment_config(invalid_config)
        assert len(errors) > 0
        assert any("circuit_breaker_enabled" in error for error in errors)

    def test_validate_environment_config_live_mode_sandbox_enabled(self):
        """Test environment validation for live mode with sandbox enabled."""
        manager = ConfigManager.__new__(ConfigManager)

        invalid_config = {
            "environment": {"mode": "live"},
            "exchange": {
                "name": "kucoin",
                "api_key": "test_key",
                "api_secret": "test_secret",
                "sandbox": True,  # Should be False for live
            },
            "risk_management": {"circuit_breaker_enabled": True},
        }

        errors = manager.validate_environment_config(invalid_config)
        assert len(errors) > 0
        assert any("sandbox" in error for error in errors)

    def test_validate_environment_config_paper_mode_valid(self):
        """Test environment validation for valid paper mode config."""
        manager = ConfigManager.__new__(ConfigManager)

        valid_paper_config = {
            "environment": {"mode": "paper"},
            "exchange": {"name": "kucoin"},
        }

        errors = manager.validate_environment_config(valid_paper_config)
        assert len(errors) == 0

    def test_validate_environment_config_paper_mode_missing_exchange_name(self):
        """Test environment validation for paper mode without exchange name."""
        manager = ConfigManager.__new__(ConfigManager)

        invalid_config = {
            "environment": {"mode": "paper"},
            "exchange": {},  # Missing name
        }

        errors = manager.validate_environment_config(invalid_config)
        assert len(errors) > 0
        assert any("Exchange name is required" in error for error in errors)

    def test_validate_environment_config_invalid_mode(self):
        """Test environment validation with invalid mode."""
        manager = ConfigManager.__new__(ConfigManager)

        invalid_config = {
            "environment": {"mode": "invalid"},
            "exchange": {"name": "kucoin"},
        }

        errors = manager.validate_environment_config(invalid_config)
        assert len(errors) > 0
        assert any("Invalid environment mode" in error for error in errors)

    def test_apply_safe_defaults_missing_sections(self):
        """Test applying safe defaults for missing sections."""
        manager = ConfigManager.__new__(ConfigManager)

        incomplete_config = {
            "environment": {"mode": "paper"}
            # Missing exchange, risk_management, monitoring, strategies
        }

        updated_config = manager.apply_safe_defaults(incomplete_config)

        # Check that all required sections are now present
        assert "environment" in updated_config
        assert "exchange" in updated_config
        assert "risk_management" in updated_config
        assert "monitoring" in updated_config
        assert "strategies" in updated_config

        # Check that safe defaults were applied
        assert updated_config["exchange"]["name"] == "kucoin"
        assert updated_config["exchange"]["sandbox"] is True
        assert updated_config["risk_management"]["circuit_breaker_enabled"] is True

    def test_apply_safe_defaults_missing_fields(self):
        """Test applying safe defaults for missing fields within sections."""
        manager = ConfigManager.__new__(ConfigManager)

        incomplete_config = {
            "environment": {"mode": "paper"},
            "exchange": {"name": "kucoin"},  # Missing sandbox, timeout, rate_limit
            "risk_management": {"stop_loss": 0.02},  # Missing other fields
            "monitoring": {},
            "strategies": {},
        }

        updated_config = manager.apply_safe_defaults(incomplete_config)

        # Check that missing fields were filled with defaults
        assert updated_config["exchange"]["sandbox"] is True
        assert updated_config["exchange"]["timeout"] == 30000
        assert updated_config["risk_management"]["take_profit"] == 0.04
        assert updated_config["monitoring"]["enabled"] is True
        assert updated_config["strategies"]["default"] == "RSIStrategy"

    def test_apply_safe_defaults_no_changes_needed(self):
        """Test applying safe defaults when all fields are present."""
        manager = ConfigManager.__new__(ConfigManager)

        complete_config = {
            "environment": {"mode": "paper", "debug": False, "log_level": "INFO"},
            "exchange": {
                "name": "kucoin",
                "sandbox": True,
                "timeout": 30000,
                "rate_limit": 10,
            },
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "trailing_stop": True,
                "position_size": 0.1,
                "max_position_size": 0.3,
                "risk_reward_ratio": 2.0,
                "max_daily_drawdown": 0.1,
                "circuit_breaker_enabled": True,
            },
            "monitoring": {
                "enabled": True,
                "update_interval": 5,
                "alert_on_errors": True,
                "log_level": "INFO",
            },
            "strategies": {
                "default": "RSIStrategy",
                "active_strategies": ["RSIStrategy"],
                "max_concurrent_strategies": 3,
            },
        }

        original_config = copy.deepcopy(complete_config)
        updated_config = manager.apply_safe_defaults(complete_config)

        # Configuration should remain unchanged
        assert updated_config == original_config

    def test_config_manager_init_with_valid_config(self):
        """Test ConfigManager initialization with valid configuration."""
        valid_config = {
            "environment": {"mode": "paper", "debug": False, "log_level": "INFO"},
            "exchange": {
                "name": "kucoin",
                "sandbox": True,
                "timeout": 30000,
                "rate_limit": 10,
            },
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "trailing_stop": True,
                "position_size": 0.1,
                "max_position_size": 0.3,
                "risk_reward_ratio": 2.0,
                "max_daily_drawdown": 0.1,
                "circuit_breaker_enabled": True,
            },
            "monitoring": {
                "enabled": True,
                "update_interval": 5,
                "alert_on_errors": True,
                "log_level": "INFO",
            },
            "strategies": {
                "default": "RSIStrategy",
                "active_strategies": ["RSIStrategy"],
                "max_concurrent_strategies": 3,
            },
        }

        # Write config to file
        with open(self.config_file, "w") as f:
            json.dump(valid_config, f)

        # Should initialize without errors
        manager = ConfigManager(self.config_file)
        assert manager is not None

    def test_config_manager_init_with_invalid_config(self):
        """Test ConfigManager initialization with invalid configuration."""
        invalid_config = {
            "environment": {"mode": "invalid_mode"},  # Invalid mode
            "exchange": {"name": "kucoin"},
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "circuit_breaker_enabled": True,
            },
            "monitoring": {"enabled": True, "update_interval": 5},
            "strategies": {"default": "RSIStrategy"},
        }

        # Write config to file
        with open(self.config_file, "w") as f:
            json.dump(invalid_config, f)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Configuration validation failed"):
            ConfigManager(self.config_file)

    def test_config_manager_init_with_live_config_missing_keys(self):
        """Test ConfigManager initialization with live config missing API keys."""
        live_config = {
            "environment": {"mode": "live"},
            "exchange": {"name": "kucoin", "sandbox": False},  # Missing API keys
            "risk_management": {"circuit_breaker_enabled": True},
            "monitoring": {"enabled": True, "update_interval": 5},
            "strategies": {"default": "RSIStrategy"},
        }

        # Write config to file
        with open(self.config_file, "w") as f:
            json.dump(live_config, f)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Environment validation failed"):
            ConfigManager(self.config_file)

    def test_config_manager_init_with_missing_sections(self):
        """Test ConfigManager initialization with missing sections (should apply defaults)."""
        minimal_config = {
            "environment": {"mode": "paper"}
            # Missing other sections
        }

        # Write config to file
        with open(self.config_file, "w") as f:
            json.dump(minimal_config, f)

        # Should initialize successfully with defaults applied
        manager = ConfigManager(self.config_file)
        assert manager is not None

    def test_config_manager_init_with_invalid_json(self):
        """Test ConfigManager initialization with invalid JSON."""
        # Write invalid JSON to file
        with open(self.config_file, "w") as f:
            f.write("{invalid json content")

        # Should raise ValueError
        with pytest.raises(ValueError, match="Invalid JSON in configuration file"):
            ConfigManager(self.config_file)

    def test_config_manager_init_with_unknown_fields(self):
        """Test ConfigManager initialization with unknown fields."""
        config_with_unknown = {
            "environment": {"mode": "paper"},
            "exchange": {"name": "kucoin"},
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "circuit_breaker_enabled": True,
            },
            "monitoring": {"enabled": True, "update_interval": 5},
            "strategies": {"default": "RSIStrategy"},
            "unknown_section": {"field": "value"},  # Unknown section
        }

        # Write config to file
        with open(self.config_file, "w") as f:
            json.dump(config_with_unknown, f)

        # Should raise ValueError due to unknown fields
        with pytest.raises(ValueError, match="Configuration validation failed"):
            ConfigManager(self.config_file)


class TestPydanticModels:
    """Test cases for individual Pydantic configuration models."""

    def test_environment_config_valid(self):
        """Test EnvironmentConfig with valid data."""
        config = EnvironmentConfig(mode="paper", debug=False, log_level="INFO")
        assert config.mode == "paper"
        assert config.debug is False
        assert config.log_level == "INFO"

    def test_environment_config_invalid_mode(self):
        """Test EnvironmentConfig with invalid mode."""
        with pytest.raises(ValidationError):
            EnvironmentConfig(mode="invalid_mode")

    def test_environment_config_invalid_log_level(self):
        """Test EnvironmentConfig with invalid log level."""
        with pytest.raises(ValidationError):
            EnvironmentConfig(mode="paper", log_level="INVALID")

    def test_exchange_config_valid(self):
        """Test ExchangeConfig with valid data."""
        config = ExchangeConfig(
            name="kucoin",
            api_key="test_key",
            api_secret="test_secret",
            sandbox=True,
            timeout=30000,
            rate_limit=10,
        )
        assert config.name == "kucoin"
        assert config.api_key == "test_key"
        assert config.sandbox is True

    def test_exchange_config_invalid_timeout(self):
        """Test ExchangeConfig with invalid timeout."""
        with pytest.raises(ValidationError):
            ExchangeConfig(name="kucoin", timeout=500)  # Too low

    def test_exchange_config_invalid_rate_limit(self):
        """Test ExchangeConfig with invalid rate limit."""
        with pytest.raises(ValidationError):
            ExchangeConfig(name="kucoin", rate_limit=0)  # Too low

    def test_risk_management_config_valid(self):
        """Test RiskManagementConfig with valid data."""
        config = RiskManagementConfig(
            stop_loss=0.02,
            take_profit=0.04,
            position_size=0.1,
            circuit_breaker_enabled=True,
        )
        assert config.stop_loss == 0.02
        assert config.circuit_breaker_enabled is True

    def test_risk_management_config_invalid_stop_loss(self):
        """Test RiskManagementConfig with invalid stop loss."""
        with pytest.raises(ValidationError):
            RiskManagementConfig(stop_loss=0.0001)  # Too low

        with pytest.raises(ValidationError):
            RiskManagementConfig(stop_loss=0.6)  # Too high

    def test_monitoring_config_valid(self):
        """Test MonitoringConfig with valid data."""
        config = MonitoringConfig(
            enabled=True, update_interval=5, alert_on_errors=True, log_level="INFO"
        )
        assert config.enabled is True
        assert config.update_interval == 5

    def test_monitoring_config_invalid_update_interval(self):
        """Test MonitoringConfig with invalid update interval."""
        with pytest.raises(ValidationError):
            MonitoringConfig(update_interval=0)  # Too low

    def test_strategies_config_valid(self):
        """Test StrategiesConfig with valid data."""
        config = StrategiesConfig(
            default="RSIStrategy",
            active_strategies=["RSIStrategy", "EMACrossStrategy"],
            max_concurrent_strategies=3,
        )
        assert config.default == "RSIStrategy"
        assert len(config.active_strategies) == 2

    def test_strategies_config_invalid_max_concurrent(self):
        """Test StrategiesConfig with invalid max concurrent strategies."""
        with pytest.raises(ValidationError):
            StrategiesConfig(max_concurrent_strategies=0)  # Too low

    def test_main_config_valid(self):
        """Test MainConfig with valid data."""
        config_data = {
            "environment": {"mode": "paper"},
            "exchange": {"name": "kucoin"},
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "circuit_breaker_enabled": True,
            },
            "monitoring": {"enabled": True, "update_interval": 5},
            "strategies": {"default": "RSIStrategy"},
        }

        config = MainConfig(**config_data)
        assert config.environment.mode == "paper"
        assert config.exchange.name == "kucoin"

    def test_main_config_unknown_field(self):
        """Test MainConfig rejects unknown fields."""
        config_data = {
            "environment": {"mode": "paper"},
            "exchange": {"name": "kucoin"},
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "circuit_breaker_enabled": True,
            },
            "monitoring": {"enabled": True, "update_interval": 5},
            "strategies": {"default": "RSIStrategy"},
            "unknown_field": "value",
        }

        with pytest.raises(ValidationError):
            MainConfig(**config_data)


class TestSafeDefaults:
    """Test cases for SAFE_DEFAULTS."""

    def test_safe_defaults_structure(self):
        """Test that SAFE_DEFAULTS has all required sections."""
        required_sections = [
            "environment",
            "exchange",
            "risk_management",
            "monitoring",
            "strategies",
        ]

        for section in required_sections:
            assert section in SAFE_DEFAULTS
            assert isinstance(SAFE_DEFAULTS[section], dict)

    def test_safe_defaults_values(self):
        """Test that SAFE_DEFAULTS contains reasonable values."""
        # Environment defaults
        assert SAFE_DEFAULTS["environment"]["mode"] == "paper"
        assert SAFE_DEFAULTS["environment"]["debug"] is False

        # Exchange defaults
        assert SAFE_DEFAULTS["exchange"]["name"] == "kucoin"
        assert SAFE_DEFAULTS["exchange"]["sandbox"] is True

        # Risk management defaults
        assert SAFE_DEFAULTS["risk_management"]["circuit_breaker_enabled"] is True
        assert SAFE_DEFAULTS["risk_management"]["stop_loss"] == 0.02

        # Monitoring defaults
        assert SAFE_DEFAULTS["monitoring"]["enabled"] is True

        # Strategies defaults
        assert SAFE_DEFAULTS["strategies"]["default"] == "RSIStrategy"
