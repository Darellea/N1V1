"""
core/contracts.py

Shared signal contracts used across modules to avoid circular imports.

This module defines the TradingSignal dataclass and related enums so they can be
imported by strategies, risk manager, order manager, and tests without importing
the full SignalRouter implementation (which reduces circular import risk).
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Any
from decimal import Decimal
import time
from utils.time import now_ms, to_ms


class SignalType(Enum):
    """Types of trading signals."""

    ENTRY_LONG = auto()
    ENTRY_SHORT = auto()
    EXIT_LONG = auto()
    EXIT_SHORT = auto()
    STOP_LOSS = auto()
    TAKE_PROFIT = auto()
    TRAILING_STOP = auto()


class SignalStrength(Enum):
    """Signal strength levels."""

    WEAK = 1
    MODERATE = 2
    STRONG = 3
    EXTREME = 4


@dataclass
class TradingSignal:
    """
    Dataclass representing a trading signal.

    Attributes:
        strategy_id: ID of the strategy that generated the signal
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        signal_type: Type of signal (entry/exit/etc.)
        signal_strength: Strength of the signal
        order_type: Type of order to execute
        amount: Size of the position (in base currency)
        price: Target price for limit orders
        current_price: Current market price when signal was generated
        timestamp: Time when signal was generated (ms)
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        trailing_stop: Trailing stop config (optional)
        metadata: Additional strategy-specific data
    """

    strategy_id: str
    symbol: str
    signal_type: SignalType
    signal_strength: SignalStrength
    order_type: Any
    amount: Decimal
    price: Optional[Decimal] = None
    current_price: Optional[Decimal] = None
    timestamp: int = 0
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop: Optional[Dict] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        """Set timestamp if not provided and normalize provided values to ms."""
        if not self.timestamp:
            self.timestamp = now_ms()
        else:
            # Normalize any provided timestamp (seconds, ms, ISO string, datetime) to ms
            normalized = to_ms(self.timestamp)
            if normalized is not None:
                self.timestamp = normalized
            else:
                # Fallback to now in case normalization fails
                self.timestamp = now_ms()

    def copy(self):
        """Return a shallow copy of this TradingSignal (tests expect .copy())."""
        from dataclasses import replace

        return replace(self)
