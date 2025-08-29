"""
core/types.py

Shared core enums and lightweight types to avoid circular imports.
"""

from enum import Enum

class TradingMode(Enum):
    """Canonical trading modes used across the project."""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


class OrderType(Enum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Order status states."""

    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
