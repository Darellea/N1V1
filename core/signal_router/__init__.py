"""
Signal Router Package

Provides modular signal routing functionality for the trading bot.
"""

# Import ML functions first to avoid circular import
from ml.model_loader import predict as ml_predict

from .event_bus import EventBus, SignalEvent, get_default_event_bus
from .retry_hooks import ErrorHandler, RetryManager, safe_async_call, with_retry_async
from .route_policies import MarketRegimeRoutePolicy, MLRoutePolicy, RoutePolicy
from .router import JournalWriter, SignalRouter  # Import from router.py
from .signal_validators import SignalValidator

# Re-export key classes and functions for easy access
__all__ = [
    # Signal validation
    "SignalValidator",
    # Retry and error handling
    "RetryManager",
    "ErrorHandler",
    "with_retry_async",
    "safe_async_call",
    # Routing policies
    "RoutePolicy",
    "MLRoutePolicy",
    "MarketRegimeRoutePolicy",
    # Event bus
    "EventBus",
    "SignalRouter",
    "SignalEvent",
    "get_default_event_bus",
    # Journal writer
    "JournalWriter",
    # ML functions (for test compatibility)
    "ml_predict",
    # Legacy compatibility
    "LegacySignalRouter",
]

# Version info
__version__ = "1.0.0"
