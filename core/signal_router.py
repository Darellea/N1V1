"""
core/signal_router.py

Handles the routing and processing of trading signals between strategies,
risk management, and order execution. Implements signal validation,
prioritization, and conflict resolution.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum, auto
from decimal import Decimal
import time
from core.types import OrderType

from utils.logger import TradeLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)
trade_logger = TradeLogger()


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
    order_type: OrderType
    amount: Decimal
    price: Optional[Decimal] = None
    current_price: Optional[Decimal] = None
    timestamp: int = 0
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop: Optional[Dict] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = int(time.time() * 1000)

    def copy(self):
        """Return a shallow copy of this TradingSignal (tests expect .copy())."""
        from dataclasses import replace
        return replace(self)


class SignalRouter:
    """
    Routes trading signals between strategies, risk management, and execution.
    Handles signal validation, prioritization, and conflict resolution.
    """

    def __init__(self, risk_manager: "RiskManager"):
        """Initialize the SignalRouter."""
        self.risk_manager = risk_manager
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_history: List[TradingSignal] = []
        self.conflict_resolution_rules = {
            'strength_based': True,
            'newer_first': False,
            'exit_over_entry': True
        }

    async def process_signal(self, signal: TradingSignal) -> Optional[TradingSignal]:
        """
        Process and validate a trading signal.
        
        Args:
            signal: The trading signal to process
            
        Returns:
            Approved signal if it passes all checks, None otherwise
        """
        # 1. Validate basic signal properties
        if not self._validate_signal(signal):
            logger.warning(f"Invalid signal rejected: {signal}")
            return None

        # 2. Check for signal conflicts
        conflicting = self._check_signal_conflicts(signal)
        if conflicting:
            signal = self._resolve_conflicts(signal, conflicting)
            if not signal:
                return None

        # 3. Apply risk management checks
        risk_approved = await self.risk_manager.evaluate_signal(signal)
        if not risk_approved:
            logger.info(f"Signal rejected by risk manager: {signal}")
            trade_logger.log_rejected_signal(signal, "risk_check")
            return None

        # 4. Finalize and store the signal
        self._store_signal(signal)
        logger.info(f"Signal approved: {signal}")
        trade_logger.log_signal(signal)
        return signal

    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate a trading signal's basic properties."""
        if not signal.symbol or not signal.amount or signal.amount <= 0:
            return False

        if signal.order_type == OrderType.LIMIT and not signal.price:
            return False

        if signal.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
            if not signal.stop_loss and self.risk_manager.require_stop_loss:
                return False

        return True

    def _check_signal_conflicts(self, new_signal: TradingSignal) -> List[TradingSignal]:
        """
        Check for conflicting signals for the same symbol.
        
        Args:
            new_signal: The new signal to check
            
        Returns:
            List of conflicting signals
        """
        conflicts = []
        
        # Check against active signals for the same symbol
        for signal_id, active_signal in self.active_signals.items():
            if active_signal.symbol == new_signal.symbol:
                # Check if signals are in opposite directions
                if self._is_opposite_signal(new_signal, active_signal):
                    conflicts.append(active_signal)
        
        return conflicts

    def _is_opposite_signal(self, signal1: TradingSignal, signal2: TradingSignal) -> bool:
        """Check if two signals are in opposite directions."""
        entry_types = {SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT}
        exit_types = {SignalType.EXIT_LONG, SignalType.EXIT_SHORT}
        
        # Both are entry signals in opposite directions
        if (signal1.signal_type in entry_types and 
            signal2.signal_type in entry_types):
            return (signal1.signal_type == SignalType.ENTRY_LONG and 
                    signal2.signal_type == SignalType.ENTRY_SHORT) or (
                    signal1.signal_type == SignalType.ENTRY_SHORT and 
                    signal2.signal_type == SignalType.ENTRY_LONG)
        
        # One is entry long and other is exit long (or short equivalents)
        if (signal1.signal_type == SignalType.ENTRY_LONG and 
            signal2.signal_type == SignalType.EXIT_LONG):
            return True
        if (signal1.signal_type == SignalType.ENTRY_SHORT and 
            signal2.signal_type == SignalType.EXIT_SHORT):
            return True
        
        return False

    def _resolve_conflicts(self, new_signal: TradingSignal, 
                         conflicting_signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """
        Resolve conflicting signals based on configured rules.
        
        Args:
            new_signal: The new signal
            conflicting_signals: List of conflicting signals
            
        Returns:
            The winning signal after conflict resolution, or None if new signal should be rejected
        """
        if not conflicting_signals:
            return new_signal

        # Apply conflict resolution rules in priority order
        if self.conflict_resolution_rules['exit_over_entry']:
            exits = [s for s in conflicting_signals 
                    if s.signal_type in {SignalType.EXIT_LONG, SignalType.EXIT_SHORT}]
            if exits and new_signal.signal_type in {SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT}:
                logger.info(f"New entry signal rejected due to existing exit signal")
                return None

        if self.conflict_resolution_rules['strength_based']:
            strongest_conflict = max(conflicting_signals, key=lambda x: x.signal_strength.value)
            if new_signal.signal_strength.value <= strongest_conflict.signal_strength.value:
                logger.info(f"New signal rejected due to stronger conflicting signal")
                return None
            else:
                # New signal is stronger - cancel conflicting signals
                for signal in conflicting_signals:
                    self._cancel_signal(signal)
                return new_signal

        if self.conflict_resolution_rules['newer_first']:
            newest_conflict = max(conflicting_signals, key=lambda x: x.timestamp)
            if new_signal.timestamp <= newest_conflict.timestamp:
                logger.info(f"New signal rejected due to newer conflicting signal")
                return None
            else:
                # New signal is newer - cancel conflicting signals
                for signal in conflicting_signals:
                    self._cancel_signal(signal)
                return new_signal

        # Default: reject new signal if any conflicts exist
        return None

    def _store_signal(self, signal: TradingSignal) -> None:
        """Store the signal in active signals and history."""
        signal_id = self._generate_signal_id(signal)
        self.active_signals[signal_id] = signal
        self.signal_history.append(signal)

    def _cancel_signal(self, signal: TradingSignal) -> None:
        """Cancel an active signal."""
        signal_id = self._generate_signal_id(signal)
        if signal_id in self.active_signals:
            del self.active_signals[signal_id]
            logger.info(f"Cancelled signal: {signal}")

    def _generate_signal_id(self, signal: TradingSignal) -> str:
        """Generate a unique ID for a signal."""
        return f"{signal.strategy_id}_{signal.symbol}_{signal.timestamp}"

    async def update_signal_status(self, signal: TradingSignal, 
                                 status: str, reason: str = "") -> None:
        """
        Update the status of a signal (e.g., when order is executed).
        
        Args:
            signal: The signal to update
            status: New status ('executed', 'rejected', 'expired')
            reason: Reason for status change
        """
        signal_id = self._generate_signal_id(signal)
        if signal_id in self.active_signals:
            del self.active_signals[signal_id]
            logger.info(f"Signal {status}: {signal} ({reason})")

    def get_active_signals(self, symbol: str = None) -> List[TradingSignal]:
        """
        Get all active signals, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of active signals
        """
        if symbol:
            return [s for s in self.active_signals.values() if s.symbol == symbol]
        return list(self.active_signals.values())

    def get_signal_history(self, limit: int = 100) -> List[TradingSignal]:
        """
        Get recent signal history.
        
        Args:
            limit: Maximum number of historical signals to return
            
        Returns:
            List of historical signals
        """
        return self.signal_history[-limit:] if self.signal_history else []

    def clear_signals(self) -> None:
        """Clear all active signals (e.g., on bot shutdown)."""
        self.active_signals.clear()
        logger.info("Cleared all active signals")
