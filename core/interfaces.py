"""
Core interfaces and abstract base classes for dependency injection.

This module defines the contracts that components must implement,
enabling loose coupling and testability through dependency injection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class EvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live based


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    max_memory_mb: float = 500.0
    warning_memory_mb: float = 400.0
    cleanup_memory_mb: float = 350.0
    hard_limit_mb: float = 600.0  # Hard limit that triggers immediate action
    component_limits: Dict[str, float] = None  # Per-component memory limits
    graceful_degradation_threshold: float = 450.0  # When to start graceful degradation
    emergency_cleanup_threshold: float = 550.0  # When to trigger emergency cleanup
    forecasting_window_minutes: int = 10  # Memory usage forecasting window
    degradation_steps: List[str] = None  # Ordered degradation steps
    eviction_batch_size: int = 100
    memory_check_interval: float = 60.0
    enable_monitoring: bool = True
    enable_hard_limits: bool = True  # Enable hard memory limits
    enable_forecasting: bool = True  # Enable memory usage forecasting
    cleanup_interval: float = 300.0

    def __post_init__(self):
        if self.component_limits is None:
            self.component_limits = {
                "cache": 100.0,
                "data_manager": 150.0,
                "signal_processor": 100.0,
                "ml_models": 200.0,
                "default": 50.0,
            }
        if self.degradation_steps is None:
            self.degradation_steps = [
                "reduce_cache_size",
                "clear_unused_objects",
                "disable_non_critical_features",
                "force_garbage_collection",
                "emergency_cleanup",
            ]


# Configuration Protocol
class ConfigProvider(Protocol):
    """Protocol for configuration providers."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        ...

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        ...


# Data Management Interfaces
class DataManagerInterface(ABC):
    """Abstract interface for data management operations."""

    @abstractmethod
    async def fetch_market_data(self) -> Dict[str, Any]:
        """Fetch market data with caching support."""
        pass

    @abstractmethod
    def set_trading_pairs(self, pairs: List[str]) -> None:
        """Set trading pairs for data management."""
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear data cache."""
        pass

    @abstractmethod
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status information."""
        pass


class SignalProcessorInterface(ABC):
    """Abstract interface for signal processing operations."""

    @abstractmethod
    def set_trading_pairs(self, pairs: List[str]) -> None:
        """Set trading pairs for signal processing."""
        pass

    @abstractmethod
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Any]:
        """Generate trading signals from market data."""
        pass

    @abstractmethod
    async def evaluate_risk(
        self, signals: List[Any], market_data: Dict[str, Any]
    ) -> List[Any]:
        """Evaluate signals through risk management."""
        pass

    @abstractmethod
    def get_strategy_info(self) -> List[Dict[str, Any]]:
        """Get information about active strategies."""
        pass


class RiskManagerInterface(ABC):
    """Abstract interface for risk management operations."""

    @abstractmethod
    async def evaluate_signal(self, signal: Any, market_data: Dict[str, Any]) -> bool:
        """Evaluate a trading signal for risk compliance."""
        pass

    @abstractmethod
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        pass


class OrderExecutorInterface(ABC):
    """Abstract interface for order execution operations."""

    @abstractmethod
    async def execute_order(self, signal: Any) -> Optional[Dict[str, Any]]:
        """Execute a trading order."""
        pass

    @abstractmethod
    async def cancel_all_orders(self) -> bool:
        """Cancel all active orders."""
        pass

    @abstractmethod
    async def get_balance(self) -> float:
        """Get current account balance."""
        pass

    @abstractmethod
    async def get_equity(self) -> float:
        """Get current account equity."""
        pass

    @abstractmethod
    def get_active_order_count(self) -> int:
        """Get count of active orders."""
        pass

    @abstractmethod
    def get_open_position_count(self) -> int:
        """Get count of open positions."""
        pass


class PerformanceTrackerInterface(ABC):
    """Abstract interface for performance tracking operations."""

    @abstractmethod
    def update_performance_metrics(
        self, pnl: float, current_equity: Optional[float] = None
    ) -> None:
        """Update performance metrics after a trade."""
        pass

    @abstractmethod
    async def record_trade_equity(self, order_result: Dict[str, Any]) -> None:
        """Record equity progression after a trade."""
        pass

    @abstractmethod
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        pass

    @abstractmethod
    def reset_performance(self) -> None:
        """Reset performance tracking."""
        pass


class StateManagerInterface(ABC):
    """Abstract interface for state management operations."""

    @abstractmethod
    async def update_state(self) -> None:
        """Update bot state."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current bot state."""
        pass

    @abstractmethod
    def set_safe_mode(self, enabled: bool) -> None:
        """Set safe mode state."""
        pass


class CacheInterface(ABC):
    """Abstract interface for caching operations."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def clear_all(self) -> bool:
        """Clear all cached data."""
        pass

    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryManagerInterface(ABC):
    """Abstract interface for memory management operations."""

    @abstractmethod
    def get_object_from_pool(
        self,
        pool_name: str,
        factory_func: callable,
        max_pool_size: int = 50,
        *args,
        **kwargs,
    ) -> Any:
        """Get object from pool or create new one."""
        pass

    @abstractmethod
    def return_object_to_pool(self, obj: Any) -> None:
        """Return object to pool."""
        pass

    @abstractmethod
    def trigger_cleanup(self) -> None:
        """Trigger memory cleanup."""
        pass

    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        pass


# Component Factory Interfaces
class ComponentFactoryInterface(ABC):
    """Abstract interface for component factories."""

    @abstractmethod
    def create_data_manager(self, config: Dict[str, Any]) -> DataManagerInterface:
        """Create data manager instance."""
        pass

    @abstractmethod
    def create_signal_processor(
        self,
        config: Dict[str, Any],
        risk_manager: Optional[RiskManagerInterface] = None,
    ) -> SignalProcessorInterface:
        """Create signal processor instance."""
        pass

    @abstractmethod
    def create_risk_manager(self, config: Dict[str, Any]) -> RiskManagerInterface:
        """Create risk manager instance."""
        pass

    @abstractmethod
    def create_order_executor(self, config: Dict[str, Any]) -> OrderExecutorInterface:
        """Create order executor instance."""
        pass

    @abstractmethod
    def create_performance_tracker(
        self, config: Dict[str, Any]
    ) -> PerformanceTrackerInterface:
        """Create performance tracker instance."""
        pass

    @abstractmethod
    def create_state_manager(self, config: Dict[str, Any]) -> StateManagerInterface:
        """Create state manager instance."""
        pass

    @abstractmethod
    def create_cache(self, config: Dict[str, Any]) -> CacheInterface:
        """Create cache instance."""
        pass

    @abstractmethod
    def create_memory_manager(self, config: Dict[str, Any]) -> MemoryManagerInterface:
        """Create memory manager instance."""
        pass


# Configuration dataclasses
@dataclass
class CacheConfig:
    """Configuration for cache behavior."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    ttl_config: Dict[str, int] = None
    max_cache_size: int = 10000
    eviction_policy: EvictionPolicy = EvictionPolicy.TTL
    memory_config: Optional[MemoryConfig] = None

    def __post_init__(self):
        if self.ttl_config is None:
            self.ttl_config = {
                "market_ticker": 2,
                "ohlcv": 60,
                "account_balance": 5,
                "order_book": 5,
                "trades": 10,
                "klines": 30,
                "funding_rate": 60,
                "mark_price": 10,
                "default": 30,
            }
        if self.memory_config is None:
            self.memory_config = MemoryConfig()


@dataclass
class DataManagerConfig:
    """Configuration for data manager."""

    cache_enabled: bool = True
    cache_ttl: int = 60
    portfolio_mode: bool = False


@dataclass
class PerformanceTrackerConfig:
    """Configuration for performance tracker."""

    starting_balance: float = 1000.0
    enable_detailed_tracking: bool = True


@dataclass
class TradingCoordinatorConfig:
    """Configuration for trading coordinator."""

    update_interval: int = 60
    enable_safe_mode_checks: bool = True
    max_concurrent_operations: int = 10


# Dependency Injection Container
class DependencyContainer:
    """Simple dependency injection container."""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}

    def register(self, interface: type, implementation: type, singleton: bool = False):
        """Register a service implementation."""
        self._services[interface.__name__] = (implementation, singleton)

    def register_instance(self, interface: type, instance: Any):
        """Register a service instance."""
        self._services[interface.__name__] = (instance, True)

    def resolve(self, interface: type) -> Any:
        """Resolve a service implementation."""
        if interface.__name__ not in self._services:
            raise ValueError(f"No registration found for {interface.__name__}")

        impl, singleton = self._services[interface.__name__]

        if singleton:
            if interface.__name__ not in self._singletons:
                if isinstance(impl, type):
                    self._singletons[interface.__name__] = impl()
                else:
                    self._singletons[interface.__name__] = impl
            return self._singletons[interface.__name__]
        else:
            if isinstance(impl, type):
                return impl()
            else:
                return impl
