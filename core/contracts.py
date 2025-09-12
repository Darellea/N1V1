"""
core/contracts.py

Shared signal contracts used across modules to avoid circular imports.

This module defines the TradingSignal dataclass and related enums so they can be
imported by strategies, risk manager, order manager, and tests without importing
the full SignalRouter implementation (which reduces circular import risk).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Any
from decimal import Decimal
import time
from datetime import datetime
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
        amount: Size of the position (in base currency) - optional, defaults to quantity if provided
        current_price: Current market price when signal was generated - optional, defaults to price if provided
        timestamp: Time when signal was generated
        side: Trading side ("buy" or "sell") - deprecated, use signal_type instead
        price: Target price for limit orders - deprecated, use order_type and current_price
        quantity: Size of the position (in base currency) - deprecated, use amount instead
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        trailing_stop: Trailing stop config (optional)
        metadata: Additional strategy-specific data
    """

    strategy_id: str
    symbol: str
    signal_type: SignalType
    signal_strength: SignalStrength
    order_type: str
    amount: Optional[float] = None
    current_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Deprecated fields - kept for backward compatibility
    side: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[float] = None

    # Optional fields
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[Dict] = None
    metadata: Optional[Dict] = field(default_factory=dict)

    def __post_init__(self):
        """Set timestamp if not provided and normalize provided values."""
        # Store original timestamp for validation purposes
        self._original_timestamp = self.timestamp

        # Ensure timestamp is a datetime object
        if isinstance(self.timestamp, int):
            self.timestamp = datetime.fromtimestamp(self.timestamp / 1000)
        elif not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.now()

        # Handle deprecated fields for backward compatibility
        if self.quantity is not None and self.amount is None:
            self.amount = self.quantity
        if self.price is not None and self.current_price is None:
            self.current_price = self.price

        # Set defaults if still None
        if self.amount is None:
            self.amount = 0.0
        if self.current_price is None:
            self.current_price = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the TradingSignal to a dictionary for JSON serialization."""
        return {
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'signal_type': self.signal_type.name,
            'signal_strength': self.signal_strength.name,
            'order_type': self.order_type,
            'amount': self.amount,
            'current_price': self.current_price,
            'timestamp': self.timestamp.isoformat(),
            'side': self.side,
            'price': self.price,
            'quantity': self.quantity,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'metadata': self.metadata
        }

    def __repr__(self) -> str:
        """Return a string representation of the TradingSignal for logging/debugging."""
        return (f"TradingSignal(strategy_id='{self.strategy_id}', symbol='{self.symbol}', "
                f"signal_type={self.signal_type.name}, signal_strength={self.signal_strength.name}, "
                f"order_type='{self.order_type}', amount={self.amount}, "
                f"current_price={self.current_price}, timestamp='{self.timestamp.isoformat()}')")

    def __eq__(self, other) -> bool:
        """Check equality between two TradingSignal objects."""
        if not isinstance(other, TradingSignal):
            return False
        
        # Compare all fields except timestamp which might have slight variations
        return (self.strategy_id == other.strategy_id and
                self.symbol == other.symbol and
                self.signal_type == other.signal_type and
                self.signal_strength == other.signal_strength and
                self.order_type == other.order_type and
                self.amount == other.amount and
                self.current_price == other.current_price and
                self.side == other.side and
                self.price == other.price and
                self.quantity == other.quantity and
                self.stop_loss == other.stop_loss and
                self.take_profit == other.take_profit and
                self.trailing_stop == other.trailing_stop and
                self.metadata == other.metadata)

    def normalize_amount(self, total_balance: Optional[float] = None) -> None:
        """
        Convert 'amount' from fraction -> notional when signaled in metadata.

        Behavior:
        - If signal.metadata contains {"amount_is_fraction": True} then `amount`
          is interpreted as a fractional value (0..1) and will be converted to a
          notional (base currency amount) by multiplying with provided
          `total_balance`. The method mutates `self.amount` to the computed
          notional and clears metadata['amount_is_fraction'].

        - If metadata flag is absent or False, this method is a no-op.

        Args:
            total_balance: Float-like total balance used to convert fraction -> notional.

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
            frac = float(self.amount)
            # Clamp fraction to [0,1]
            if frac < 0:
                frac = 0.0
            if frac > 1:
                frac = 1.0
            notional = (float(total_balance) * frac)
            # Round to 8 decimal places
            self.amount = round(notional, 8)
            # Clear the metadata flag to indicate amount is now absolute/notional
            self.metadata["amount_is_fraction"] = False
        except Exception as e:
            # Bubble up a clear error to calling code/tests
            raise ValueError(f"Failed to normalize amount: {e}")

    def copy(self):
        """Return a shallow copy of this TradingSignal (tests expect .copy())."""
        from dataclasses import replace

        return replace(self)
