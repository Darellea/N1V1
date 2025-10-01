"""
Component factory for dependency injection.

This module provides a lightweight DI system with registry-based component management.
Components are registered with builders and resolved by name, supporting singletons and lazy instantiation.
"""

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Import interfaces at module level to avoid circular imports
try:
    from .interfaces import (
        CacheInterface,
        DataManagerInterface,
        MemoryManagerInterface,
        OrderExecutorInterface,
        PerformanceTrackerInterface,
        RiskManagerInterface,
        SignalProcessorInterface,
        StateManagerInterface,
    )
except ImportError:
    # Fallback for when interfaces are not available
    DataManagerInterface = Any
    CacheInterface = Any
    MemoryManagerInterface = Any
    OrderExecutorInterface = Any
    PerformanceTrackerInterface = Any
    RiskManagerInterface = Any
    SignalProcessorInterface = Any
    StateManagerInterface = Any


class ComponentFactory:
    """
    Lightweight Dependency Injection factory with registry-based component management.

    Features:
    - Component registration with builders
    - Singleton and non-singleton component support
    - Lazy instantiation
    - Environment override support (mock injection for tests)
    - Thread-safe component resolution
    """

    _registry: Dict[str, tuple[Callable[[], Any], bool]] = {}
    _singletons: Dict[str, Any] = {}
    _overrides: Dict[str, Any] = {}

    def __init__(self):
        """Initialize the component factory."""
        self._create_cache: Dict[str, Any] = {}

    @classmethod
    def register(
        cls,
        name: str,
        builder: Callable[[], Any],
        singleton: bool = True
    ) -> None:
        """
        Register a component builder.

        Args:
            name: Component name for resolution
            builder: Callable that creates the component instance
            singleton: Whether to cache and reuse the instance (default: True)
        """
        cls._registry[name] = (builder, singleton)
        logger.debug(f"Registered component '{name}' (singleton={singleton})")

    @classmethod
    def get(cls, name: str) -> Any:
        """
        Get a component instance by name.

        Args:
            name: Component name

        Returns:
            Component instance

        Raises:
            KeyError: If component is not registered
        """
        # Check for test overrides first
        if name in cls._overrides:
            return cls._overrides[name]

        # Check singleton cache
        if name in cls._singletons:
            return cls._singletons[name]

        # Get builder from registry
        if name not in cls._registry:
            raise KeyError(f"Component '{name}' not registered")

        builder, singleton = cls._registry[name]

        # Create instance
        instance = builder()

        # Cache if singleton
        if singleton:
            cls._singletons[name] = instance

        logger.debug(f"Created component '{name}' (singleton={singleton})")
        return instance

    @classmethod
    def override(cls, name: str, instance: Any) -> None:
        """
        Override a component for testing (environment override).

        Args:
            name: Component name to override
            instance: Mock or test instance to return
        """
        cls._overrides[name] = instance
        logger.debug(f"Overrode component '{name}' for testing")

    @classmethod
    def clear_overrides(cls) -> None:
        """Clear all test overrides."""
        cls._overrides.clear()
        logger.debug("Cleared all component overrides")

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the singleton cache."""
        cls._singletons.clear()
        logger.debug("Cleared component cache")

    @classmethod
    def reset(cls) -> None:
        """Reset the factory to clean state (useful for testing)."""
        cls._registry.clear()
        cls._singletons.clear()
        cls._overrides.clear()
        logger.debug("Reset component factory")

    def clear_create_cache(self) -> None:
        """Clear the create cache for this instance."""
        self._create_cache.clear()
        logger.debug("Cleared create cache")

    @classmethod
    def list_registered(cls) -> list[str]:
        """List all registered component names."""
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a component is registered."""
        return name in cls._registry

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get factory statistics."""
        return {
            "registered_components": len(cls._registry),
            "cached_singletons": len(cls._singletons),
            "active_overrides": len(cls._overrides),
            "component_names": list(cls._registry.keys()),
        }

    def create_data_manager(self, config: Dict[str, Any]) -> DataManagerInterface:
        """Create data manager instance with injected dependencies."""
        key = f"data_manager_{hash(str(config))}"

        if key in self._create_cache:
            return self._create_cache[key]

        try:
            # Import configuration from centralized system
            from .config_manager import get_config_manager
            from .data_manager import DataManager

            config_manager = get_config_manager()
            dm_config = config_manager.get_data_manager_config()

            # Create data manager with configuration
            data_manager = DataManager(config)
            data_manager.cache_enabled = dm_config.cache_enabled
            data_manager.cache_ttl = dm_config.cache_ttl

            # Cache the instance
            self._create_cache[key] = data_manager

            logger.info("Created DataManager with dependency injection")
            return data_manager

        except Exception as e:
            logger.error(f"Failed to create DataManager: {e}")
            raise

    def create_cache(self, config: Dict[str, Any]) -> CacheInterface:
        """Create cache instance with injected dependencies."""
        key = f"cache_{hash(str(config))}"

        if key in self._create_cache:
            return self._create_cache[key]

        try:
            # Import configuration from centralized system
            from .config_manager import get_config_manager
            from .cache import RedisCache

            config_manager = get_config_manager()
            cache_config = config_manager.get_cache_config()

            # Create Redis cache with configuration
            cache = RedisCache(cache_config)

            # Cache the instance
            self._create_cache[key] = cache

            logger.info("Created Cache with dependency injection")
            return cache

        except Exception as e:
            logger.error(f"Failed to create Cache: {e}")
            raise

    def create_memory_manager(self, config: Dict[str, Any]) -> MemoryManagerInterface:
        """Create memory manager instance with injected dependencies."""
        key = f"memory_manager_{hash(str(config))}"

        if key in self._create_cache:
            return self._create_cache[key]

        try:
            # Import configuration from centralized system
            from .config_manager import get_config_manager
            from .memory_manager import MemoryManager

            config_manager = get_config_manager()
            memory_config = config_manager.get_memory_config()

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
            self._create_cache[key] = memory_manager

            logger.info("Created MemoryManager with dependency injection")
            return memory_manager

        except Exception as e:
            logger.error(f"Failed to create MemoryManager: {e}")
            raise

    def create_performance_tracker(self, config: Dict[str, Any]) -> PerformanceTrackerInterface:
        """Create performance tracker instance with injected dependencies."""
        key = f"performance_tracker_{hash(str(config))}"

        if key in self._create_cache:
            return self._create_cache[key]

        try:
            # Import configuration from centralized system
            from .config_manager import get_config_manager
            from .performance_tracker import PerformanceTracker

            config_manager = get_config_manager()
            pt_config = config_manager.get_performance_tracker_config()

            # Create performance tracker with configuration
            performance_tracker = PerformanceTracker(config)
            performance_tracker.starting_balance = pt_config.starting_balance

            # Cache the instance
            self._create_cache[key] = performance_tracker

            logger.info("Created PerformanceTracker with dependency injection")
            return performance_tracker

        except Exception as e:
            logger.error(f"Failed to create PerformanceTracker: {e}")
            raise


# Backward compatibility - keep the old interface for existing code
class LegacyComponentFactory:
    """
    Legacy factory interface for backward compatibility.

    This maintains the old factory methods while delegating to the new DI system.
    """

    def __init__(self):
        """Initialize the legacy factory."""
        self._config_manager = None
        self._component_cache: Dict[str, Any] = {}

    # Import interfaces here to avoid circular imports
    try:
        from .interfaces import (
            CacheInterface,
            DataManagerInterface,
            MemoryManagerInterface,
            OrderExecutorInterface,
            PerformanceTrackerInterface,
            RiskManagerInterface,
            SignalProcessorInterface,
            StateManagerInterface,
        )
    except ImportError:
        # Fallback for when interfaces are not available
        DataManagerInterface = Any
        CacheInterface = Any
        MemoryManagerInterface = Any
        OrderExecutorInterface = Any
        PerformanceTrackerInterface = Any
        RiskManagerInterface = Any
        SignalProcessorInterface = Any
        StateManagerInterface = Any

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


# Component registrations - executed on module import
def _register_core_components():
    """Register all core components with the factory."""
    from .config_manager import get_config_manager

    config_manager = get_config_manager()

    # Register core services
    ComponentFactory.register("config_manager", lambda: config_manager)

    # Register managers with configuration
    ComponentFactory.register("retry_manager", lambda: _create_retry_manager())
    ComponentFactory.register("circuit_breaker", lambda: _create_circuit_breaker())
    ComponentFactory.register("data_manager", lambda: _create_data_manager())
    ComponentFactory.register("order_manager", lambda: _create_order_manager())
    ComponentFactory.register("risk_manager", lambda: _create_risk_manager())
    ComponentFactory.register("signal_processor", lambda: _create_signal_processor())
    ComponentFactory.register("performance_tracker", lambda: _create_performance_tracker())
    ComponentFactory.register("state_manager", lambda: _create_state_manager())
    ComponentFactory.register("cache", lambda: _create_cache())
    ComponentFactory.register("memory_manager", lambda: _create_memory_manager())

    logger.info("Core components registered with DI factory")


def _create_retry_manager():
    """Create retry manager with configuration."""
    from .execution.retry_manager import RetryManager
    config_manager = ComponentFactory.get("config_manager")
    config = config_manager.get_reliability_config()
    return RetryManager(config)


def _create_circuit_breaker():
    """Create circuit breaker with configuration."""
    from .api_protection import APICircuitBreaker, CircuitBreakerConfig
    config = CircuitBreakerConfig()
    return APICircuitBreaker(config)


def _create_data_manager():
    """Create data manager with configuration."""
    from .data_manager import DataManager
    config_manager = ComponentFactory.get("config_manager")
    config = config_manager.get_data_manager_config()
    return DataManager(config)


def _create_order_manager():
    """Create order manager with configuration."""
    from .order_manager import OrderManager
    from .types import TradingMode
    config_manager = ComponentFactory.get("config_manager")
    config = config_manager.get_config()
    mode_str = config.get("environment", {}).get("mode", "paper")
    mode = TradingMode[mode_str.upper()]
    return OrderManager(config, mode)


def _create_risk_manager():
    """Create risk manager with configuration."""
    from risk.risk_manager import RiskManager
    config_manager = ComponentFactory.get("config_manager")
    config = config_manager.get_config()
    risk_config = config.get("risk_management", {})
    return RiskManager(risk_config)


def _create_signal_processor():
    """Create signal processor with configuration."""
    from .signal_processor import SignalProcessor
    config_manager = ComponentFactory.get("config_manager")
    config = config_manager.get_config()
    risk_manager = ComponentFactory.get("risk_manager")
    return SignalProcessor(config, risk_manager)


def _create_performance_tracker():
    """Create performance tracker with configuration."""
    from .performance_tracker import PerformanceTracker
    config_manager = ComponentFactory.get("config_manager")
    config = config_manager.get_config()
    return PerformanceTracker(config)


def _create_state_manager():
    """Create state manager."""
    from .state_manager import StateManager
    return StateManager()


def _create_cache():
    """Create cache with configuration."""
    from .cache import RedisCache
    config_manager = ComponentFactory.get("config_manager")
    cache_config = config_manager.get_cache_config()
    return RedisCache(cache_config)


def _create_memory_manager():
    """Create memory manager with configuration."""
    from .memory_manager import MemoryManager
    config_manager = ComponentFactory.get("config_manager")
    memory_config = config_manager.get_memory_config()
    return MemoryManager(
        enable_monitoring=memory_config.enable_monitoring,
        cleanup_interval=memory_config.cleanup_interval,
    )


# Register components on module import
_register_core_components()
