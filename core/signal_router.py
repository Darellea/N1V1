"""
core/signal_router.py

Handles the routing and processing of trading signals between strategies,
risk management, and order execution. Implements signal validation,
prioritization, and conflict resolution.

Enhanced with robust error handling, retry/backoff for risk checks, and
structured logging categories to improve resilience.
"""

import logging
import asyncio
import random
import time
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum, auto
from decimal import Decimal

from utils.logger import TradeLogger
from core.types import OrderType  # Backward-compatible export: tests expect OrderType here
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)
trade_logger = TradeLogger()

# Error / category codes for structured logs
ERROR_NETWORK = "NETWORK_ERROR"
ERROR_ORDER_REJECTED = "ORDER_REJECTED"
ERROR_STRATEGY = "STRATEGY_ERROR"
ERROR_VALIDATION = "VALIDATION_ERROR"
ERROR_RISK = "RISK_MANAGER_ERROR"


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

    This version adds:
      - defensive error handling around risk manager calls
      - retry/backoff for risk_manager.evaluate_signal (configurable per instance)
      - structured logging categories and counters for critical errors
      - optional blocking of signals if too many critical errors occur
    """

    def __init__(
        self,
        risk_manager: "RiskManager",
        max_retries: int = 2,
        backoff_base: float = 0.5,
        max_backoff: float = 5.0,
        safe_mode_threshold: int = 10,
    ):
        """Initialize the SignalRouter."""
        self.risk_manager = risk_manager
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_history: List[TradingSignal] = []
        self.conflict_resolution_rules = {
            "strength_based": True,
            "newer_first": False,
            "exit_over_entry": True,
        }

        # Retry/backoff configuration for critical async calls (e.g., risk_manager.evaluate_signal)
        self.retry_config = {
            "max_retries": int(max_retries),
            "backoff_base": float(backoff_base),
            "max_backoff": float(max_backoff),
        }

        # Safe-mode like local blocking: if router experiences too many critical errors,
        # it can temporarily block processing of new signals to prevent cascades.
        self.critical_errors = 0
        self.safe_mode_threshold = int(safe_mode_threshold)
        self.block_signals = False

    async def process_signal(self, signal: TradingSignal, market_data: Dict = None) -> Optional[TradingSignal]:
        """
        Process and validate a trading signal.

        Args:
            signal: The trading signal to process
            market_data: Optional market data context

        Returns:
            Approved signal if it passes all checks, None otherwise
        """
        # If router is in blocking state, reject quickly
        if self.block_signals:
            logger.warning("SignalRouter is currently blocking new signals due to repeated errors")
            trade_logger.log_rejected_signal(signal, "router_blocking")
            return None

        # 1. Validate basic signal properties
        try:
            if not self._validate_signal(signal):
                logger.debug("Signal validation failed", exc_info=False)
                trade_logger.log_rejected_signal(signal, ERROR_VALIDATION)
                return None
        except Exception as e:
            logger.exception("Unexpected error during signal validation")
            trade_logger.log_rejected_signal(signal, ERROR_VALIDATION)
            # don't escalate further
            return None

        # 2. Check for signal conflicts
        try:
            conflicting = self._check_signal_conflicts(signal)
            if conflicting:
                signal = self._resolve_conflicts(signal, conflicting)
                if not signal:
                    return None
        except Exception as e:
            logger.exception("Error while resolving signal conflicts")
            # Log and reject this signal to avoid unexpected behavior
            trade_logger.log_rejected_signal(signal, "conflict_resolution_error")
            return None

        # 3. Apply risk management checks with retry/backoff
        try:
            # risk_manager.evaluate_signal may be an async call that occasionally fails (network/external),
            # so use a retry loop with exponential backoff and jitter.
            approved = await self._retry_async_call(
                lambda: self.risk_manager.evaluate_signal(signal, market_data),
                retries=self.retry_config["max_retries"],
                base_backoff=self.retry_config["backoff_base"],
                max_backoff=self.retry_config["max_backoff"],
            )
            if not approved:
                logger.info("Signal rejected by RiskManager")
                trade_logger.log_rejected_signal(signal, "risk_check")
                return None
        except Exception as e:
            # Record critical error and possibly block further signals temporarily
            self._record_router_error(e, context={"symbol": getattr(signal, "symbol", None)})
            logger.exception("Risk manager evaluation failed after retries")
            trade_logger.log_rejected_signal(signal, ERROR_RISK)
            return None

        # 4. Finalize and store the signal
        try:
            self._store_signal(signal)
            logger.info(f"Signal approved: {signal}")
            trade_logger.log_signal(signal.__dict__ if hasattr(signal, "__dict__") else {"signal": str(signal)})
            return signal
        except Exception as e:
            logger.exception("Failed to store approved signal")
            trade_logger.log_rejected_signal(signal, "store_failed")
            return None

    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate a trading signal's basic properties."""
        if not signal.symbol or not signal.amount or signal.amount <= 0:
            return False

        # OrderType is flexible; require price for LIMIT orders if represented as string 'limit'
        if signal.order_type and getattr(signal.order_type, "value", None) == "limit" and not signal.price:
            return False

        # For entry signals, ensure stop loss if risk manager requires it
        if signal.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
            try:
                if getattr(self.risk_manager, "require_stop_loss", False) and not signal.stop_loss:
                    return False
            except Exception:
                # If risk_manager lacks attribute or errors, be conservative and require stop_loss
                if not signal.stop_loss:
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
        for signal_id, active_signal in self.active_signals.items():
            if active_signal.symbol == new_signal.symbol:
                if self._is_opposite_signal(new_signal, active_signal):
                    conflicts.append(active_signal)
        return conflicts

    def _is_opposite_signal(self, signal1: TradingSignal, signal2: TradingSignal) -> bool:
        """Check if two signals are in opposite directions."""
        entry_types = {SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT}
        exit_types = {SignalType.EXIT_LONG, SignalType.EXIT_SHORT}

        # Both are entry signals in opposite directions
        if signal1.signal_type in entry_types and signal2.signal_type in entry_types:
            return (
                signal1.signal_type == SignalType.ENTRY_LONG
                and signal2.signal_type == SignalType.ENTRY_SHORT
            ) or (
                signal1.signal_type == SignalType.ENTRY_SHORT
                and signal2.signal_type == SignalType.ENTRY_LONG
            )

        # One is entry long and other is exit long (or short equivalents)
        if (
            signal1.signal_type == SignalType.ENTRY_LONG
            and signal2.signal_type == SignalType.EXIT_LONG
        ):
            return True
        if (
            signal1.signal_type == SignalType.ENTRY_SHORT
            and signal2.signal_type == SignalType.EXIT_SHORT
        ):
            return True

        return False

    def _resolve_conflicts(self, new_signal: TradingSignal, conflicting_signals: List[TradingSignal]) -> Optional[TradingSignal]:
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

        try:
            # Prefer exit over entry to avoid flipping into a new position when an exit exists
            if self.conflict_resolution_rules["exit_over_entry"]:
                exits = [
                    s
                    for s in conflicting_signals
                    if s.signal_type in {SignalType.EXIT_LONG, SignalType.EXIT_SHORT}
                ]
                if exits and new_signal.signal_type in {
                    SignalType.ENTRY_LONG,
                    SignalType.ENTRY_SHORT,
                }:
                    logger.info("New entry signal rejected due to existing exit signal")
                    return None

            if self.conflict_resolution_rules["strength_based"]:
                strongest_conflict = max(
                    conflicting_signals, key=lambda x: x.signal_strength.value
                )
                if (
                    new_signal.signal_strength.value
                    <= strongest_conflict.signal_strength.value
                ):
                    logger.info("New signal rejected due to stronger conflicting signal")
                    return None
                else:
                    # New signal is stronger - cancel conflicting signals
                    for signal in conflicting_signals:
                        self._cancel_signal(signal)
                    return new_signal

            if self.conflict_resolution_rules["newer_first"]:
                newest_conflict = max(conflicting_signals, key=lambda x: x.timestamp)
                if new_signal.timestamp <= newest_conflict.timestamp:
                    logger.info("New signal rejected due to newer conflicting signal")
                    return None
                else:
                    for signal in conflicting_signals:
                        self._cancel_signal(signal)
                    return new_signal
        except Exception:
            logger.exception("Error during conflict resolution")
            return None

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

    async def _retry_async_call(
        self,
        call_fn: Callable[[], "Coroutine[Any, Any, Any]"],
        retries: int = 2,
        base_backoff: float = 0.5,
        max_backoff: float = 5.0,
    ) -> Any:
        """
        Retry an async callable with exponential backoff and jitter.

        Args:
            call_fn: zero-arg callable returning coroutine
            retries: number of retry attempts
            base_backoff: base delay in seconds
            max_backoff: maximum delay cap

        Returns:
            Result of the coroutine or raises last exception
        """
        attempt = 0
        while True:
            try:
                attempt += 1
                return await call_fn()
            except Exception as e:
                if attempt > retries:
                    logger.error(f"Retry exhausted for async call after {attempt-1} retries: {str(e)}")
                    raise
                # compute backoff + jitter
                backoff = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
                jitter = backoff * 0.1
                sleep_for = backoff + random.uniform(-jitter, jitter)
                logger.warning(f"Async call failed (attempt {attempt}/{retries}), retrying in {sleep_for:.2f}s: {str(e)}")
                trade_logger.trade("SignalRouter retry", {"attempt": attempt, "error": str(e), "backoff": sleep_for})
                await asyncio.sleep(max(0.0, sleep_for))
                continue

    def _record_router_error(self, exc: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a router-level critical error and enter blocking state if threshold exceeded.
        """
        try:
            self.critical_errors += 1
            trade_logger.trade("Router critical error", {"count": self.critical_errors, "error": str(exc), "context": context})
            logger.error(f"Router critical error #{self.critical_errors}: {str(exc)}")
            if self.critical_errors >= self.safe_mode_threshold:
                self.block_signals = True
                trade_logger.trade("SignalRouter entering blocking state", {"reason": "threshold_exceeded", "count": self.critical_errors})
                logger.critical("SignalRouter blocking new signals due to repeated errors")
        except Exception:
            logger.exception("Failed to record router error")

    async def update_signal_status(self, signal: TradingSignal, status: str, reason: str = "") -> None:
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
