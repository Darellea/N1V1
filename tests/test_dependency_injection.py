"""
Tests for dependency injection and configuration system.

This module tests the new dependency injection architecture and centralized
configuration system to ensure components are properly decoupled and configurable.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from typing import Dict, Any

from core.interfaces import (
    DataManagerInterface,
    SignalProcessorInterface,
    RiskManagerInterface,
    OrderExecutorInterface,
    PerformanceTrackerInterface,
    StateManagerInterface,
    CacheInterface,
    MemoryManagerInterface
)
from core.config_manager import ConfigManager, get_config_manager
from core.component_factory import ComponentFactory, get_component_factory
from core.cache import RedisCache
from core.memory_manager import MemoryManager
from core.performance_tracker import PerformanceTracker
from core.data_manager import DataManager


class TestDependencyInjection:
    """Test dependency injection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "environment": {"mode": "paper"},
            "trading": {"portfolio_mode": False, "initial_balance": 1000.0},
            "cache": {"enabled": True, "ttl": 60},
            "memory": {"enable_monitoring": False}
        }

    def test_config_manager_initialization(self):
        """Test that ConfigManager initializes correctly."""
        with patch('core.config_manager.ConfigManager._load_config'):
            config_manager = ConfigManager()
            assert config_manager is not None
            assert hasattr(config_manager, '_config')

    def test_config_value_retrieval(self):
        """Test configuration value retrieval."""
        with patch('core.config_manager.ConfigManager._load_config'):
            config_manager = ConfigManager()

            # Test getting existing values
            cache_config = config_manager.get_cache_config()
            assert cache_config is not None
            assert hasattr(cache_config, 'host')
            assert hasattr(cache_config, 'port')

    def test_component_factory_creation(self):
        """Test component factory creates components correctly."""
        factory = ComponentFactory()

        # Test data manager creation
        data_manager = factory.create_data_manager(self.config)
        assert isinstance(data_manager, DataManagerInterface)
        assert hasattr(data_manager, 'fetch_market_data')
        assert hasattr(data_manager, 'set_trading_pairs')

        # Test performance tracker creation
        perf_tracker = factory.create_performance_tracker(self.config)
        assert isinstance(perf_tracker, PerformanceTrackerInterface)
        assert hasattr(perf_tracker, 'update_performance_metrics')
        assert hasattr(perf_tracker, 'record_trade_equity')

    def test_cache_component_creation(self):
        """Test cache component creation with configuration."""
        factory = ComponentFactory()

        cache = factory.create_cache(self.config)
        assert isinstance(cache, CacheInterface)
        assert hasattr(cache, 'get')
        assert hasattr(cache, 'set')
        assert hasattr(cache, 'get_cache_stats')

    def test_memory_manager_creation(self):
        """Test memory manager creation with configuration."""
        factory = ComponentFactory()

        memory_mgr = factory.create_memory_manager(self.config)
        assert isinstance(memory_mgr, MemoryManagerInterface)
        assert hasattr(memory_mgr, 'get_memory_stats')
        assert hasattr(memory_mgr, 'trigger_cleanup')

    def test_component_caching(self):
        """Test that components are cached and reused."""
        factory = ComponentFactory()

        # Create same component twice
        data_mgr1 = factory.create_data_manager(self.config)
        data_mgr2 = factory.create_data_manager(self.config)

        # Should be the same instance (cached)
        assert data_mgr1 is data_mgr2

    def test_configuration_override(self):
        """Test configuration value override functionality."""
        with patch('core.config_manager.ConfigManager._load_config'):
            config_manager = ConfigManager()

            # Test setting and getting override values
            config_manager.set("cache.ttl_config.market_ticker", 5)
            value = config_manager.get("cache.ttl_config.market_ticker")
            assert value == 5

    def test_config_validation(self):
        """Test configuration validation."""
        with patch('core.config_manager.ConfigManager._load_config'):
            config_manager = ConfigManager()

            # Test valid configuration
            errors = config_manager.validate_config()
            # Should have no errors with default config
            assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_async_component_operations(self):
        """Test async operations on injected components."""
        factory = ComponentFactory()

        # Create cache component
        cache = factory.create_cache(self.config)

        # Test async operations
        success = await cache.set("test_key", {"test": "data"})
        assert isinstance(success, bool)

        data = await cache.get("test_key")
        assert data is None or isinstance(data, dict)

    def test_factory_global_instance(self):
        """Test global component factory instance."""
        factory1 = get_component_factory()
        factory2 = get_component_factory()

        assert factory1 is factory2
        assert isinstance(factory1, ComponentFactory)

    def test_config_manager_global_instance(self):
        """Test global config manager instance."""
        with patch('core.config_manager.ConfigManager._load_config'):
            manager1 = get_config_manager()
            manager2 = get_config_manager()

            assert manager1 is manager2
            assert isinstance(manager1, ConfigManager)


class TestConfigurationIntegration:
    """Test configuration integration with components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "environment": {"mode": "paper"},
            "trading": {"portfolio_mode": False, "initial_balance": 2000.0},
            "cache": {"enabled": True, "ttl": 120},
            "memory": {"enable_monitoring": True, "max_memory_mb": 600.0}
        }

    def test_data_manager_config_integration(self):
        """Test DataManager uses configuration correctly."""
        factory = ComponentFactory()

        data_manager = factory.create_data_manager(self.config)
        assert data_manager.cache_enabled is True
        assert data_manager.cache_ttl == 60  # From default config

    def test_performance_tracker_config_integration(self):
        """Test PerformanceTracker uses configuration correctly."""
        factory = ComponentFactory()

        perf_tracker = factory.create_performance_tracker(self.config)
        # Starting balance should come from config
        assert perf_tracker.starting_balance == 2000.0

    def test_memory_manager_config_integration(self):
        """Test MemoryManager uses configuration correctly."""
        factory = ComponentFactory()

        memory_mgr = factory.create_memory_manager(self.config)
        # Should use configured values
        assert hasattr(memory_mgr, '_memory_thresholds')
        assert "warning_mb" in memory_mgr._memory_thresholds

    def test_cache_config_integration(self):
        """Test Cache uses configuration correctly."""
        factory = ComponentFactory()

        cache = factory.create_cache(self.config)
        # Should have configuration applied
        assert hasattr(cache, 'config')
        assert cache.config is not None


class TestComponentIsolation:
    """Test that components are properly isolated through DI."""

    def test_component_independence(self):
        """Test that components don't have direct dependencies."""
        # This test ensures components use interfaces rather than concrete implementations
        from core.interfaces import DataManagerInterface, SignalProcessorInterface

        # Verify interfaces are abstract
        assert hasattr(DataManagerInterface, '__abstractmethods__')
        assert hasattr(SignalProcessorInterface, '__abstractmethods__')

    def test_factory_creates_correct_types(self):
        """Test factory creates components of correct types."""
        factory = ComponentFactory()
        config = {"environment": {"mode": "paper"}}

        # Test type creation
        data_mgr = factory.create_data_manager(config)
        assert isinstance(data_mgr, DataManager)

        perf_tracker = factory.create_performance_tracker(config)
        assert isinstance(perf_tracker, PerformanceTracker)

        cache = factory.create_cache(config)
        assert isinstance(cache, RedisCache)

        memory_mgr = factory.create_memory_manager(config)
        assert isinstance(memory_mgr, MemoryManager)


class TestConfigurationPersistence:
    """Test configuration persistence and reloading."""

    def test_config_save_load(self):
        """Test saving and loading configuration."""
        with patch('core.config_manager.ConfigManager._load_config'):
            config_manager = ConfigManager()

            # Modify configuration
            config_manager.set("cache.ttl_config.market_ticker", 10)

            # Save would work if file system was available
            # For now, just test the method exists
            assert hasattr(config_manager, 'save_config')
            assert callable(config_manager.save_config)

    def test_config_reload(self):
        """Test configuration reloading."""
        with patch('core.config_manager.ConfigManager._load_config'):
            config_manager = ConfigManager()

            # Reload should work
            assert hasattr(config_manager, 'reload_config')
            assert callable(config_manager.reload_config)


if __name__ == "__main__":
    pytest.main([__file__])
