"""
core.types package initializer.

Expose lightweight shared types used across the codebase to avoid import-time
conflicts between the top-level module `core/types.py` and the `core/types/`
package.

This file intentionally defines TradingMode so that `from core.types import
TradingMode` works even when the `core/types/` package exists. It also re-exports
order-related types defined in `core/types/order_types.py` so existing imports
that expect `from core.types import OrderType` continue to work.
"""

from enum import Enum

class TradingMode(Enum):
    """Canonical trading modes used across the project."""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"

# Re-export order-related types for backward compatibility
from .order_types import Order, OrderType, OrderStatus  # noqa: E402,F401

__all__ = ["TradingMode", "Order", "OrderType", "OrderStatus"]
