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
        quantity: Size of the position (in base currency)
        side: Trading side ("buy" or "sell")
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
    quantity: Decimal
    side: str
    price: Optional[Decimal] = None
    current_price: Optional[Decimal] = None
    timestamp: int = 0
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop: Optional[Dict] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        """Set timestamp if not provided and normalize provided values to ms.

        Also normalize numeric fields to Decimal for internal consistency.
        """
        # Normalize timestamp
        if not self.timestamp:
            self.timestamp = now_ms()
        else:
            normalized = to_ms(self.timestamp)
            self.timestamp = normalized if normalized is not None else now_ms()

        # Normalize numeric fields to Decimal where appropriate so the rest of the
        # codebase can rely on Decimal-typed numeric members.
        try:
            if self.quantity is not None and not isinstance(self.quantity, Decimal):
                self.quantity = Decimal(str(self.quantity))
        except Exception:
            # If conversion fails, set a safe default (zero) to avoid crashes
            self.quantity = Decimal(0)

        for field_name in ("price", "current_price", "stop_loss", "take_profit"):
            try:
                val = getattr(self, field_name)
                if val is not None and not isinstance(val, Decimal):
                    setattr(self, field_name, Decimal(str(val)))
            except Exception:
                setattr(self, field_name, None)

    def normalize_amount(self, total_balance: Optional[Decimal] = None) -> None:
        """
        Convert 'quantity' from fraction -> notional when signaled in metadata.

        Behavior:
        - If signal.metadata contains {"amount_is_fraction": True} then `quantity`
          is interpreted as a fractional value (0..1) and will be converted to a
          notional (base currency amount) by multiplying with provided
          `total_balance`. The method mutates `self.quantity` to the computed
          notional (Decimal) and clears metadata['amount_is_fraction'].

        - If metadata flag is absent or False, this method is a no-op.

        Args:
            total_balance: Decimal-like total balance used to convert fraction -> notional.

        Raises:
            ValueError: if fraction conversion is requested but total_balance is not provided.
        """
        if not self.metadata or not isinstance(self.metadata, dict):
            return

        if not self.metadata.get("amount_is_fraction"):
            return

        if total_balance is None:
            raise ValueError("total_balance is required to convert fractional amount to notional")

        try:
            frac = Decimal(str(self.quantity))
            # Clamp fraction to [0,1]
            if frac < 0:
                frac = Decimal(0)
            if frac > 1:
                frac = Decimal(1)
            notional = (Decimal(str(total_balance)) * frac)
            # Quantize to a sensible precision (8 decimal places) but keep Decimal type
            self.quantity = notional.quantize(Decimal("0.00000001"))
            # Clear the metadata flag to indicate amount is now absolute/notional
            self.metadata["amount_is_fraction"] = False
        except Exception as e:
            # Bubble up a clear error to calling code/tests
            raise ValueError(f"Failed to normalize amount: {e}")

    def copy(self):
        """Return a shallow copy of this TradingSignal (tests expect .copy())."""
        from dataclasses import replace

        return replace(self)
