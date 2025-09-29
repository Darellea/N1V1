"""
Component factory for dependency injection.

This module provides a factory that creates and configures all core components
with proper dependency injection, replacing direct instantiation throughout the codebase.
"""

import logging
from typing import Any, Dict, Optional

from .cache import RedisCache
from .config_manager import get_config_manager
from .data_manager import DataManager
from .interfaces import (
    CacheInterface,
    ComponentFactoryInterface,
    DataManagerInterface,
    MemoryManagerInterface,
    OrderExecutorInterface,
    PerformanceTrackerInterface,
    RiskManagerInterface,
    SignalProcessorInterface,
    StateManagerInterface,
)
from .memory_manager import MemoryManager
from .performance_tracker import PerformanceTracker
from .signal_processor import SignalProcessor

logger = logging.getLogger(__name__)


class ComponentFactory(ComponentFactoryInterface):
    """
    Factory for creating core components with dependency injection.

    This factory centralizes component creation and ensures all dependencies
    are properly injected, reducing tight coupling throughout the system.
    """

    def __init__(self):
        """Initialize the component factory."""
        self._config_manager = get_config_manager()
        self._component_cache: Dict[str, Any] = {}

    def create_data_manager(self, config: Dict[str, Any]) -> DataManagerInterface:
        """Create data manager instance with injected dependencies."""
        cache_key = f"data_manager_{hash(str(config))}"

        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        try:
            # Get configuration
            dm_config = self._config_manager.get_data_manager_config()

            # Create data manager with configuration
            data_manager = DataManager(config)
            data_manager.cache_enabled = dm_config.cache_enabled
            data_manager.cache_ttl = dm_config.cache_ttl

            # Cache the instance
            self._component_cache[cache_key] = data_manager

            logger.info("Created DataManager with dependency injection")
            return data_manager

        except Exception as e:
            logger.error(f"Failed to create DataManager: {e}")
            raise

    def create_signal_processor(
        self,
        config: Dict[str, Any],
        risk_manager: Optional[RiskManagerInterface] = None,
    ) -> SignalProcessorInterface:
        """Create signal processor instance with injected dependencies."""
        cache_key = f"signal_processor_{hash(str(config))}_{id(risk_manager)}"

        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        try:
            # Create signal processor with optional risk manager
            signal_processor = SignalProcessor(config, risk_manager)

            # Cache the instance
            self._component_cache[cache_key] = signal_processor

            logger.info("Created SignalProcessor with dependency injection")
            return signal_processor

        except Exception as e:
            logger.error(f"Failed to create SignalProcessor: {e}")
            raise

    def create_risk_manager(self, config: Dict[str, Any]) -> RiskManagerInterface:
        """Create risk manager instance with injected dependencies."""
        cache_key = f"risk_manager_{hash(str(config))}"

        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        try:
            # Import here to avoid circular imports
            from risk.risk_manager import RiskManager

            # Create risk manager with configuration
            risk_config = config.get("risk_management", {})
            risk_manager = RiskManager(risk_config)

            # Cache the instance
            self._component_cache[cache_key] = risk_manager

            logger.info("Created RiskManager with dependency injection")
            return risk_manager

        except Exception as e:
            logger.error(f"Failed to create RiskManager: {e}")
            raise

    def create_order_executor(self, config: Dict[str, Any]) -> OrderExecutorInterface:
        """Create order executor instance with injected dependencies."""
        cache_key = f"order_executor_{hash(str(config))}"

        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        try:
            # Import here to avoid circular imports
            from .order_manager import OrderManager

            # Get trading mode from config
            from .types import TradingMode

            mode_str = config.get("environment", {}).get("mode", "paper")
            mode = TradingMode[mode_str.upper()]

            # Create order manager with configuration
            order_executor = OrderManager(config, mode)

            # Cache the instance
            self._component_cache[cache_key] = order_executor

            logger.info("Created OrderExecutor with dependency injection")
            return order_executor

        except Exception as e:
            logger.error(f"Failed to create OrderExecutor: {e}")
            raise

    def create_performance_tracker(
        self, config: Dict[str, Any]
    ) -> PerformanceTrackerInterface:
        """Create performance tracker instance with injected dependencies."""
        cache_key = f"performance_tracker_{hash(str(config))}"

        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        try:
            # Get configuration
            pt_config = self._config_manager.get_performance_tracker_config()

            # Create performance tracker with configuration
            performance_tracker = PerformanceTracker(config)
            performance_tracker.starting_balance = pt_config.starting_balance

            # Cache the instance
            self._component_cache[cache_key] = performance_tracker

            logger.info("Created PerformanceTracker with dependency injection")
            return performance_tracker

        except Exception as e:
            logger.error(f"Failed to create PerformanceTracker: {e}")
            raise

    def create_state_manager(self, config: Dict[str, Any]) -> StateManagerInterface:
        """Create state manager instance with injected dependencies."""
        cache_key = f"state_manager_{hash(str(config))}"

        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        try:
            # Import here to avoid circular imports
            from .state_manager import StateManager

            # Create state manager
            state_manager = StateManager()

            # Cache the instance
            self._component_cache[cache_key] = state_manager

            logger.info("Created StateManager with dependency injection")
            return state_manager

        except Exception as e:
            logger.error(f"Failed to create StateManager: {e}")
            raise

    def create_cache(self, config: Dict[str, Any]) -> CacheInterface:
        """Create cache instance with injected dependencies."""
        cache_key = f"cache_{hash(str(config))}"

        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        try:
            # Get cache configuration
            cache_config = self._config_manager.get_cache_config()

            # Create Redis cache with configuration
            cache = RedisCache(cache_config)

            # Cache the instance
            self._component_cache[cache_key] = cache

            logger.info("Created Cache with dependency injection")
            return cache

        except Exception as e:
            logger.error(f"Failed to create Cache: {e}")
            raise

    def create_memory_manager(self, config: Dict[str, Any]) -> MemoryManagerInterface:
        """Create memory manager instance with injected dependencies."""
        cache_key = f"memory_manager_{hash(str(config))}"

        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        try:
            # Get memory configuration
            memory_config = self._config_manager.get_memory_config()

            # Create memory manager with configuration
            memory_manager = MemoryManager(
                enable_monitoring=memory_config.enable_monitoring,
                cleanup_interval=memory_config.cleanup_interval,
            )

            # Configure memory thresholds
            memory_manager.set_memory_thresholds(
                warning_mb=memory_config.warning_memory_mb,
                critical_mb=memory_config.max_memory_mb,
                cleanup_mb=memory_config.cleanup_memory_mb,
            )

            # Cache the instance
            self._component_cache[cache_key] = memory_manager

            logger.info("Created MemoryManager with dependency injection")
            return memory_manager

        except Exception as e:
            logger.error(f"Failed to create MemoryManager: {e}")
            raise

    def create_trading_coordinator(
        self, config: Dict[str, Any]
    ) -> "TradingCoordinator":
        """Create trading coordinator with all dependencies injected."""
        from .trading_coordinator import TradingCoordinator

        try:
            # Get configuration
            tc_config = self._config_manager.get_trading_coordinator_config()

            # Create coordinator
            coordinator = TradingCoordinator(config)
            coordinator.update_interval = tc_config.update_interval

            # Create and inject dependencies
            data_manager = self.create_data_manager(config)
            signal_processor = self.create_signal_processor(config)
            risk_manager = self.create_risk_manager(config)
            order_executor = self.create_order_executor(config)
            performance_tracker = self.create_performance_tracker(config)
            state_manager = self.create_state_manager(config)

            # Set components
            coordinator.set_components(
                data_manager=data_manager,
                signal_processor=signal_processor,
                risk_manager=risk_manager,
                order_executor=order_executor,
                performance_tracker=performance_tracker,
                state_manager=state_manager,
            )

            logger.info("Created TradingCoordinator with full dependency injection")
            return coordinator

        except Exception as e:
            logger.error(f"Failed to create TradingCoordinator: {e}")
            raise

    def clear_cache(self) -> None:
        """Clear the component cache."""
        self._component_cache.clear()
        logger.info("Component cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get component cache statistics."""
        return {
            "cached_components": len(self._component_cache),
            "component_types": list(self._component_cache.keys()),
        }


# Global component factory instance
_component_factory: Optional[ComponentFactory] = None


def get_component_factory() -> ComponentFactory:
    """Get the global component factory instance."""
    global _component_factory
    if _component_factory is None:
        _component_factory = ComponentFactory()
    return _component_factory


def create_data_manager(config: Dict[str, Any]) -> DataManagerInterface:
    """Create data manager (convenience function)."""
    return get_component_factory().create_data_manager(config)


def create_signal_processor(
    config: Dict[str, Any], risk_manager: Optional[RiskManagerInterface] = None
) -> SignalProcessorInterface:
    """Create signal processor (convenience function)."""
    return get_component_factory().create_signal_processor(config, risk_manager)


def create_risk_manager(config: Dict[str, Any]) -> RiskManagerInterface:
    """Create risk manager (convenience function)."""
    return get_component_factory().create_risk_manager(config)


def create_order_executor(config: Dict[str, Any]) -> OrderExecutorInterface:
    """Create order executor (convenience function)."""
    return get_component_factory().create_order_executor(config)


def create_performance_tracker(config: Dict[str, Any]) -> PerformanceTrackerInterface:
    """Create performance tracker (convenience function)."""
    return get_component_factory().create_performance_tracker(config)


def create_state_manager(config: Dict[str, Any]) -> StateManagerInterface:
    """Create state manager (convenience function)."""
    return get_component_factory().create_state_manager(config)


def create_cache(config: Dict[str, Any]) -> CacheInterface:
    """Create cache (convenience function)."""
    return get_component_factory().create_cache(config)


def create_memory_manager(config: Dict[str, Any]) -> MemoryManagerInterface:
    """Create memory manager (convenience function)."""
    return get_component_factory().create_memory_manager(config)


def create_trading_coordinator(config: Dict[str, Any]) -> "TradingCoordinator":
    """Create trading coordinator (convenience function)."""
    return get_component_factory().create_trading_coordinator(config)
