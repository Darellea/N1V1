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
from decimal import Decimal
import pandas as pd
from core.contracts import TradingSignal, SignalType, SignalStrength

from utils.logger import get_trade_logger
from core.types import OrderType  # Backward-compatible export: tests expect OrderType here
from typing import TYPE_CHECKING
from utils.adapter import signal_to_dict

# ML integration
from utils.config_loader import get_config
from ml.model_loader import load_model, predict as ml_predict

if TYPE_CHECKING:
    from risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()

# Error / category codes for structured logs
ERROR_NETWORK = "NETWORK_ERROR"
ERROR_ORDER_REJECTED = "ORDER_REJECTED"
ERROR_STRATEGY = "STRATEGY_ERROR"
ERROR_VALIDATION = "VALIDATION_ERROR"
ERROR_RISK = "RISK_MANAGER_ERROR"








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

        # ML integration: attempt to load ML confirmation model from config (if configured)
        try:
            ml_cfg = get_config("ml", {})
            self.ml_enabled = bool(ml_cfg.get("enabled", False))
            self.ml_model = None
            self.ml_confidence_threshold = float(ml_cfg.get("confidence_threshold", 0.6))
            ml_path = ml_cfg.get("model_path")
            if self.ml_enabled and ml_path:
                try:
                    self.ml_model = load_model(ml_path)
                    logger.info(f"ML model loaded for signal confirmation: {ml_path}")
                except Exception as e:
                    self.ml_enabled = False
                    logger.warning(f"Failed to load ML model at {ml_path}: {e}")
        except Exception:
            # If config not loaded or any error occurs, disable ML integration gracefully
            self.ml_enabled = False

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
            trade_logger.log_rejected_signal(signal_to_dict(signal), "router_blocking")
            return None

        # 1. Validate basic signal properties
        try:
            if not self._validate_signal(signal):
                logger.debug("Signal validation failed", exc_info=False)
                trade_logger.log_rejected_signal(signal_to_dict(signal), ERROR_VALIDATION)
                return None
        except Exception as e:
            logger.exception("Unexpected error during signal validation")
            trade_logger.log_rejected_signal(signal_to_dict(signal), ERROR_VALIDATION)
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
            trade_logger.log_rejected_signal(signal_to_dict(signal), "conflict_resolution_error")
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
                trade_logger.log_rejected_signal(signal_to_dict(signal), "risk_check")
                return None
        except Exception as e:
            # Record critical error and possibly block further signals temporarily
            self._record_router_error(e, context={"symbol": getattr(signal, "symbol", None)})
            logger.exception("Risk manager evaluation failed after retries")
            trade_logger.log_rejected_signal(signal_to_dict(signal), ERROR_RISK)
            return None

        # 4. ML confirmation layer (optional)
        try:
            if self.ml_enabled and self.ml_model and signal.signal_type in {SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT}:
                features_df = None
                # Try to extract features from market_data
                try:
                    if market_data and isinstance(market_data, dict):
                        f = market_data.get("features") or market_data.get("feature_row") or market_data.get("features_df")
                        if f is not None:
                            if isinstance(f, pd.DataFrame):
                                features_df = f
                            elif isinstance(f, dict):
                                features_df = pd.DataFrame([f])
                            else:
                                try:
                                    features_df = pd.DataFrame([f])
                                except Exception:
                                    features_df = None
                        if features_df is None and "ohlcv" in market_data:
                            try:
                                ohlcv = market_data.get("ohlcv")
                                features_df = pd.DataFrame([ohlcv]) if ohlcv is not None else None
                            except Exception:
                                features_df = None
                    # Fallback to features in signal metadata
                    if features_df is None and getattr(signal, "metadata", None):
                        m = signal.metadata.get("features") if isinstance(signal.metadata, dict) else None
                        if m is not None:
                            if isinstance(m, pd.DataFrame):
                                features_df = m
                            elif isinstance(m, dict):
                                features_df = pd.DataFrame([m])
                            else:
                                try:
                                    features_df = pd.DataFrame([m])
                                except Exception:
                                    features_df = None
                except Exception:
                    features_df = None

                if features_df is None or features_df.empty:
                    logger.info("ML confirmation skipped: no feature row available")
                else:
                    # ensure single-row input (use most recent / last row)
                    if isinstance(features_df, pd.DataFrame) and len(features_df) > 1:
                        features_df = features_df.iloc[[-1]]
                    try:
                        ml_out = ml_predict(self.ml_model, features_df)
                        ml_pred = ml_out.iloc[0]["prediction"]
                        ml_conf = float(ml_out.iloc[0].get("confidence", 0.0))
                    except Exception as e:
                        logger.warning(f"ML prediction failed: {e}")
                        ml_pred = None
                        ml_conf = 0.0

                    # Determine desired direction from signal
                    desired = 1 if signal.signal_type == SignalType.ENTRY_LONG else -1
                    sig_text = "BUY" if desired == 1 else "SELL"
                    ml_text = "UNK"
                    if ml_pred == 1:
                        ml_text = "BUY"
                    elif ml_pred == -1:
                        ml_text = "SELL"
                    elif ml_pred == 0:
                        ml_text = "HOLD"

                    decision = "ACCEPT"
                    # Apply confirmation rules only when ML returned a prediction
                    if ml_pred is not None:
                        if ml_conf >= self.ml_confidence_threshold:
                            if ml_pred == desired:
                                decision = "ACCEPT"
                            else:
                                # ML disagrees: reduce strength or reject if already weak
                                if signal.signal_strength == SignalStrength.WEAK:
                                    decision = "REJECT"
                                else:
                                    # degrade strength by one level
                                    try:
                                        new_value = max(SignalStrength.WEAK.value, signal.signal_strength.value - 1)
                                        signal.signal_strength = SignalStrength(new_value)
                                        decision = "REDUCE"
                                    except Exception:
                                        decision = "REDUCE"
                        else:
                            decision = "NO_ML"  # low-confidence ML -> ignore ML
                    # Log combined decision
                    logger.info(f"Signal: {sig_text} | ML: {ml_text} ({ml_conf:.2f} confidence) → Decision: {decision}")
                    trade_logger.trade("ML confirmation", {"signal": sig_text, "ml": ml_text, "confidence": ml_conf, "decision": decision})
                    if decision == "REJECT":
                        trade_logger.log_rejected_signal(signal_to_dict(signal), "ml_reject")
                        return None
        except Exception:
            logger.exception("ML confirmation step failed; proceeding with indicator-only decision")

        # 5. Finalize and store the signal
        try:
            self._store_signal(signal)
            logger.info(f"Signal approved: {signal}")
            trade_logger.log_signal(signal_to_dict(signal))
            return signal
        except Exception as e:
            logger.exception("Failed to store approved signal")
            trade_logger.log_rejected_signal(signal_to_dict(signal), "store_failed")
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
