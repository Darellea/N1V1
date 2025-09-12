"""
Signal Router Implementation

Contains the main SignalRouter and JournalWriter classes.
"""

import logging
import asyncio
import json
import random
import time
from typing import List, Dict, Optional, Callable, Any, TYPE_CHECKING
from decimal import Decimal
from pathlib import Path
import pandas as pd

from utils.time import now_ms
from core.contracts import TradingSignal, SignalType, SignalStrength
from utils.logger import get_trade_logger
from core.types import OrderType  # Backward-compatible export: tests expect OrderType here
from utils.adapter import signal_to_dict
from utils.config_loader import get_config

# Import new modular components
from .signal_validators import SignalValidator
from .retry_hooks import RetryManager, ErrorHandler
from .route_policies import RoutePolicy, MLRoutePolicy
from .event_bus import EventBus, SignalRouter as ModularSignalRouter

# Legacy imports for backward compatibility
from core.ensemble_manager import EnsembleManager
from predictive_models import PredictiveModelManager
from ml.model_loader import load_model
from ml.ml_filter import create_ml_filter, load_ml_filter
import sys  # For dynamic access to patched ml_predict

if TYPE_CHECKING:
    from risk.risk_manager import RiskManager
    from core.task_manager import TaskManager

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()

# Error / category codes for structured logs (backward compatibility)
ERROR_NETWORK = "NETWORK_ERROR"
ERROR_ORDER_REJECTED = "ORDER_REJECTED"
ERROR_STRATEGY = "STRATEGY_ERROR"
ERROR_VALIDATION = "VALIDATION_ERROR"
ERROR_RISK = "RISK_MANAGER_ERROR"


# Lightweight async journal writer (best-effort, no extra deps)
class JournalWriter:
    """
    Simple append-only JSONL writer that offloads disk I/O to an asyncio
    background task. Designed to be best-effort: if no running event loop
    is available it falls back to synchronous writes.
    """

    def __init__(self, path: Path, task_manager: Optional["TaskManager"] = None):
        self.path: Path = Path(path)
        self._queue: "asyncio.Queue" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self.task_manager = task_manager

    def _ensure_task(self) -> None:
        """
        Ensure the background worker task is running. If called from a thread
        without a running loop, this will be deferred until append() sees a
        running loop and schedules the worker.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                if not self._task or self._task.done():
                    # Create the worker task using task_manager if available
                    if self.task_manager:
                        self._task = self.task_manager.create_task(self._worker(), name="JournalWriter")
                    else:
                        self._task = loop.create_task(self._worker())
        except RuntimeError:
            # No running loop; worker will be lazily started when possible
            pass

    async def _worker(self) -> None:
        try:
            while True:
                entry = await self._queue.get()
                if entry is None:
                    # sentinel to stop
                    break
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, default=str) + "\n")
                except Exception:
                    logger.exception("JournalWriter failed to write entry")
        except asyncio.CancelledError:
            pass

    def append(self, entry: Dict) -> None:
        """
        Enqueue an entry for background writing. This method is synchronous
        and safe to call from non-async code; it will fall back to a
        synchronous write if no running event loop is available.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in this thread: write synchronously as a best-effort fallback
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            except Exception:
                logger.exception("JournalWriter failed to write entry (sync fallback)")
            return

        # If loop is running, ensure worker is started and push to queue thread-safely
        if loop.is_running():
            # lazily create the worker on the running loop if needed
            if not self._task or self._task.done():
                try:
                    if self.task_manager:
                        loop.call_soon_threadsafe(lambda: self.task_manager.create_task(self._worker(), name="JournalWriter"))
                    else:
                        loop.call_soon_threadsafe(lambda: asyncio.create_task(self._worker()))
                except Exception:
                    # fallback to synchronous write
                    try:
                        self.path.parent.mkdir(parents=True, exist_ok=True)
                        with open(self.path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, default=str) + "\n")
                    except Exception:
                        logger.exception("JournalWriter failed to write entry (fallback)")
                    return
            # enqueue without awaiting
            try:
                loop.call_soon_threadsafe(self._queue.put_nowait, entry)
            except Exception:
                # final fallback to synchronous write
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, default=str) + "\n")
                except Exception:
                    logger.exception("JournalWriter failed to write entry (fallback)")
        else:
            # No running loop - synchronous write
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            except Exception:
                logger.exception("JournalWriter failed to write entry (no-loop fallback)")

    async def stop(self) -> None:
        """
        Stop the background worker, flush pending entries and wait for the task
        to finish. Best-effort: will not raise on failure.
        """
        try:
            # Put sentinel and wait for the worker to consume it
            try:
                await self._queue.put(None)
            except Exception:
                pass
            if self._task:
                try:
                    await self._task
                except Exception:
                    pass
        except Exception:
            logger.exception("Error while stopping JournalWriter")


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
        risk_manager,
        task_manager=None,
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
        self._lock = asyncio.Lock()
        # Per-symbol locks to allow concurrent processing across different symbols
        self._symbol_locks: Dict[str, asyncio.Lock] = {}

        # Task manager for tracking background tasks
        self.task_manager = task_manager or None

        # Ensemble manager for combining multiple strategies
        self.ensemble_manager = EnsembleManager()

        # Predictive models manager
        try:
            predictive_cfg = get_config("predictive_models", {})
            # For test environments, disable predictive models by default
            if not predictive_cfg.get("enabled", False):
                self.predictive_manager = PredictiveModelManager({"enabled": False})
            else:
                self.predictive_manager = PredictiveModelManager(predictive_cfg)
                if self.predictive_manager.enabled:
                    self.predictive_manager.load_models()
                    logger.info("Predictive models loaded and ready")
        except Exception as e:
            logger.warning(f"Failed to initialize predictive models: {e}")
            self.predictive_manager = PredictiveModelManager({"enabled": False})

        # ML integration: attempt to load ML filter from config (if configured)
        try:
            ml_cfg = get_config("ml", {})
            self.ml_enabled = bool(ml_cfg.get("enabled", False))
            self.ml_filter = None
            self.ml_model = None
            self.ml_fallback_to_raw = bool(ml_cfg.get("fallback_to_raw_signals", True))
            self.ml_confidence_threshold = float(ml_cfg.get("confidence_threshold", 0.6))

            if self.ml_enabled:
                ml_path = ml_cfg.get("model_path")
                model_type = ml_cfg.get("model_type", "logistic_regression")

                if ml_path and Path(ml_path).exists():
                    try:
                        self.ml_filter = load_ml_filter(ml_path)
                        self.ml_model = load_model(ml_path)  # Load the raw model for ml_predict
                        logger.info(f"ML filter and model loaded: {model_type} from {ml_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load ML filter/model from {ml_path}: {e}")
                        # Try to create a new filter if loading fails
                        try:
                            self.ml_filter = create_ml_filter(model_type, ml_cfg)
                            self.ml_model = self.ml_filter  # Use filter as model if no separate model
                            logger.info(f"Created new ML filter: {model_type}")
                        except Exception as e2:
                            logger.warning(f"Failed to create ML filter: {e2}")
                            self.ml_enabled = False
                else:
                    # Create new filter if no saved model exists
                    try:
                        self.ml_filter = create_ml_filter(model_type, ml_cfg)
                        self.ml_model = self.ml_filter  # Use filter as model
                        logger.info(f"Created new ML filter: {model_type}")
                    except Exception as e:
                        logger.warning(f"Failed to create ML filter: {e}")
                        self.ml_enabled = False

        except Exception as e:
            # If config not loaded or any error occurs, disable ML integration gracefully
            logger.warning(f"ML integration initialization failed: {e}")
            self.ml_enabled = False
            self.ml_filter = None
            self.ml_model = None

        # Optional append-only signal journal for recovery (best-effort).
        try:
            journal_cfg = get_config("journal", {})
            self.journal_path = Path(journal_cfg.get("path", "logs/signal_journal.jsonl"))
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            self._journal_enabled = bool(journal_cfg.get("enabled", False))
            # Journal writer (async background writer, best-effort). It will lazily
            # start background worker when the first entry is appended.
            self._journal_writer = JournalWriter(self.journal_path, task_manager=self.task_manager)
            # Replay journal only when explicitly enabled.
            if self._journal_enabled:
                try:
                    self.recover_from_journal()
                except Exception:
                    # If recover fails, disable journal to avoid further issues
                    self._journal_enabled = False
        except Exception:
            self._journal_enabled = False
            self._journal_writer = None

    def _extract_features_for_ml(self, market_data, signal) -> Optional[pd.DataFrame]:
        """
        Extract features for ML prediction from market_data or signal metadata.
        Returns a single-row DataFrame or None if extraction fails.
        """
        if market_data is None:
            candidate = None
        elif isinstance(market_data, dict):
            candidate = market_data.get("features")
            if candidate is None:
                candidate = market_data.get("feature_row")
            if candidate is None:
                candidate = market_data.get("features_df")
            if candidate is None:
                candidate = market_data.get("ohlcv")
        elif isinstance(market_data, pd.DataFrame):
            candidate = market_data
        elif isinstance(market_data, pd.Series):
            candidate = market_data
        else:
            candidate = None

        # Fallback to signal.metadata if candidate is None
        if candidate is None and hasattr(signal, 'metadata') and signal.metadata and isinstance(signal.metadata, dict):
            candidate = signal.metadata.get("features")
            if candidate is None:
                candidate = signal.metadata.get("feature_row")
            if candidate is None:
                candidate = signal.metadata.get("features_df")
            if candidate is None:
                candidate = signal.metadata.get("ohlcv")

        if candidate is None:
            return None

        try:
            if isinstance(candidate, pd.DataFrame):
                if candidate.empty:
                    return None
                if len(candidate) > 1:
                    return candidate.iloc[[-1]]
                return candidate
            elif isinstance(candidate, pd.Series):
                return pd.DataFrame([candidate.to_dict()])
            elif isinstance(candidate, dict):
                return pd.DataFrame([candidate])
            elif isinstance(candidate, list):
                if not candidate:
                    return None
                if isinstance(candidate[0], dict):
                    df = pd.DataFrame(candidate)
                    if df.empty:
                        return None
                    return df.iloc[[-1]]
                elif all(isinstance(x, (int, float, Decimal)) for x in candidate):
                    # Assume list of scalars, use 'value' column
                    return pd.DataFrame({'value': candidate}).iloc[[-1]]
                else:
                    return None
            else:
                # Try to convert to DataFrame
                try:
                    df = pd.DataFrame([candidate])
                    # Validate that the DataFrame has usable data
                    if df.empty or df.shape[1] == 0:
                        return None
                    # Check if all columns contain only object types (indicating invalid data)
                    if all(df.dtypes == 'object'):
                        # Check if the data is actually unusable (like object())
                        try:
                            # Try to access the first value to see if it's meaningful
                            first_val = df.iloc[0, 0]
                            if str(first_val).startswith('<') and str(first_val).endswith('>'):
                                return None
                        except (IndexError, KeyError, TypeError):
                            return None
                    return df
                except (ValueError, TypeError):
                    return None
        except (AttributeError, TypeError, ValueError):
            return None

    def _extract_market_data_for_prediction(self, market_data, signal) -> Optional[pd.DataFrame]:
        """
        Extract market data DataFrame for predictive models from market_data or signal metadata.
        Returns a DataFrame with OHLCV columns or None if extraction fails.
        """
        if market_data is None:
            candidate = None
        elif isinstance(market_data, dict):
            candidate = market_data.get("ohlcv")
            if candidate is None:
                candidate = market_data.get("data")
            if candidate is None:
                candidate = market_data.get("market_data")
        elif isinstance(market_data, pd.DataFrame):
            candidate = market_data
        else:
            candidate = None

        # Fallback to signal.metadata if candidate is None
        if candidate is None and hasattr(signal, 'metadata') and signal.metadata and isinstance(signal.metadata, dict):
            candidate = signal.metadata.get("ohlcv")
            if candidate is None:
                candidate = signal.metadata.get("data")
            if candidate is None:
                candidate = signal.metadata.get("market_data")

        if candidate is None:
            return None

        try:
            if isinstance(candidate, pd.DataFrame):
                if candidate.empty:
                    return None
                # Ensure required columns exist
                required_cols = ['open', 'high', 'low', 'close']
                if not all(col in candidate.columns for col in required_cols):
                    return None
                return candidate
            elif isinstance(candidate, dict):
                # Try to convert dict to DataFrame
                try:
                    df = pd.DataFrame([candidate])
                    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                        return None
                    return df
                except Exception:
                    return None
            elif isinstance(candidate, list):
                if not candidate:
                    return None
                try:
                    df = pd.DataFrame(candidate)
                    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                        return None
                    return df
                except Exception:
                    return None
            else:
                return None
        except Exception:
            return None

    def _reject_signal(self, signal: TradingSignal, reason: str) -> None:
        """
        Centralized helper to reject a signal and log the reason.

        Args:
            signal: The signal to reject
            reason: Reason for rejection
        """
        logger.warning(f"Signal rejected: {reason}")
        trade_logger.log_rejected_signal(signal_to_dict(signal), reason)
        return None

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
            return self._reject_signal(signal, "router_blocking")

        # 1. Validate timestamp
        try:
            if not self._validate_timestamp(signal):
                return self._reject_signal(signal, "corrupted_timestamp")
        except Exception as e:
            logger.exception("Error validating timestamp")
            return self._reject_signal(signal, "corrupted_timestamp")

        # 2. Check ensemble manager first (if enabled and signal is from individual strategy)
        if (self.ensemble_manager.enabled and
            signal.strategy_id != "ensemble" and
            hasattr(signal, 'metadata') and
            not signal.metadata.get('ensemble', False)):

            try:
                ensemble_signal = self.ensemble_manager.get_ensemble_signal(market_data)
                if ensemble_signal:
                    logger.info("Ensemble signal generated, using ensemble decision")
                    signal = ensemble_signal
                elif self.ensemble_manager.voting_mode.value != "majority_vote":
                    # For non-majority modes, if no consensus, don't process individual signal
                    return self._reject_signal(signal, "no_ensemble_consensus")
            except Exception as e:
                logger.warning(f"Ensemble processing failed: {e}, proceeding with individual signal")

        # 3. Validate basic signal properties
        try:
            if not self._validate_signal(signal):
                return self._reject_signal(signal, ERROR_VALIDATION)
        except Exception as e:
            logger.exception("Unexpected error during signal validation")
            return self._reject_signal(signal, ERROR_VALIDATION)

        # 4. Validate order with execution validator
        try:
            if hasattr(self, 'validator') and self.validator:
                if not self.validator.validate_order(signal):
                    return self._reject_signal(signal, "validator_rejection")
        except Exception as e:
            logger.exception("Error during order validation")
            return self._reject_signal(signal, "validator_error")

        # 2-5. Per-symbol serialization: acquire a symbol-specific lock to avoid races
        try:
            symbol = getattr(signal, "symbol", None)
            symbol_lock = await self._get_symbol_lock(symbol)
            async with symbol_lock:
                # Re-check validation in case state changed (defensive)
                if not self._validate_signal(signal):
                    return self._reject_signal(signal, ERROR_VALIDATION)

                # Check & resolve conflicts while holding the symbol lock to guarantee
                # at-most-one active signal per symbol during concurrent processing.
                conflicting = self._check_signal_conflicts(signal)
                if conflicting:
                    signal = await self._resolve_conflicts(signal, conflicting)
                    if not signal:
                        return self._reject_signal(signal, "conflict_resolution")

                # Apply risk manager checks (still under symbol lock to avoid races)
                try:
                    approved = await self._retry_async_call(
                        lambda: self.risk_manager.evaluate_signal(signal, market_data),
                        retries=self.retry_config["max_retries"],
                        base_backoff=self.retry_config["backoff_base"],
                        max_backoff=self.retry_config["max_backoff"],
                    )
                    if not approved:
                        return self._reject_signal(signal, "risk_check")
                except Exception as e:
                    await self._record_router_error(e, context={"symbol": getattr(signal, "symbol", None)})
                    logger.exception("Risk manager evaluation failed after retries")
                    return self._reject_signal(signal, ERROR_RISK)

                # ML confirmation (optional)
                try:
                    if self.ml_enabled and self.ml_filter:
                        ml_result = self.ml_filter.filter_signal(signal, market_data)
                        if not ml_result.get("approved", True):
                            logger.info(
                                f"ML filter rejected signal: {ml_result.get('reason')} "
                                f"(confidence={ml_result.get('confidence', 0):.2f})"
                            )
                            return self._reject_signal(signal, "ml_filter_rejection")

                    if (self.ml_enabled and self.ml_model and
                        signal.signal_type in {SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT}):

                        features_df = self._extract_features_for_ml(market_data, signal)
                        if features_df is None or features_df.empty:
                            logger.info("ML confirmation skipped: no feature row available")
                            if not self.ml_fallback_to_raw:
                                logger.warning("ML required but no features available, rejecting signal")
                                return self._reject_signal(signal, "ml_no_features")
                        else:
                            # Call ML prediction
                            logger.debug(f"Calling ML prediction with features shape: {features_df.shape}")
                            ml_result = sys.modules['core.signal_router'].ml_predict(self.ml_model, features_df)

                            if ml_result is None or ml_result.empty:
                                logger.warning("ML prediction returned None or empty result")
                                if not self.ml_fallback_to_raw:
                                    return self._reject_signal(signal, "ml_error")
                            else:
                                # Check if ML rejected the signal
                                if not ml_result.get("approved", True):
                                    logger.info(
                                        f"ML rejected signal {signal.symbol} "
                                        f"(confidence={ml_result.get('confidence')}, reason={ml_result.get('reason')})"
                                    )
                                    return self._reject_signal(signal, "ml_rejection")
                                # Extract prediction and confidence
                                prediction_row = ml_result.iloc[0]
                                ml_prediction = prediction_row.get('prediction', 0)
                                ml_confidence = prediction_row.get('confidence', 0.0)

                                # Convert prediction to signal direction
                                desired_direction = 1 if signal.signal_type == SignalType.ENTRY_LONG else -1
                                ml_direction = 1 if ml_prediction > 0 else -1

                                logger.info(f"ML prediction: direction={ml_direction}, confidence={ml_confidence:.3f}")

                                # Apply ML decision rules
                                confidence_threshold = getattr(self, 'ml_confidence_threshold', 0.6)

                                if ml_confidence >= confidence_threshold:
                                    # High confidence ML prediction - apply decision rules
                                    if signal.signal_strength == SignalStrength.WEAK:
                                        # Weak signals require ML confirmation
                                        if ml_direction == desired_direction:
                                            logger.info("Weak signal approved by ML (same direction, high confidence)")
                                            trade_logger.trade("ML confirmation", {
                                                "signal": "BUY" if desired_direction == 1 else "SELL",
                                                "confidence": ml_confidence,
                                                "approved": True,
                                                "reason": "weak_signal_confirmed"
                                            })
                                        else:
                                            logger.info("Weak signal rejected by ML (opposite direction)")
                                            return self._reject_signal(signal, "ml_opposite_direction")
                                    else:
                                        # Strong signals bypass ML rejection but still log the result
                                        if ml_direction != desired_direction:
                                            # Reduce signal strength due to ML disagreement
                                            if signal.signal_strength == SignalStrength.STRONG:
                                                signal.signal_strength = SignalStrength.MODERATE
                                                logger.info("Signal strength reduced due to ML disagreement")
                                            elif signal.signal_strength == SignalStrength.MODERATE:
                                                signal.signal_strength = SignalStrength.WEAK
                                                logger.info("Signal strength reduced due to ML disagreement")

                                        trade_logger.trade("ML confirmation", {
                                            "signal": "BUY" if desired_direction == 1 else "SELL",
                                            "confidence": ml_confidence,
                                            "approved": True,
                                            "reason": "strong_signal_bypass"
                                        })
                                else:
                                    # Low confidence - ignore ML result and proceed with original signal
                                    logger.info(f"ML confidence too low ({ml_confidence:.3f} < {confidence_threshold}), ignoring ML result")
                                    trade_logger.trade("ML confirmation", {
                                        "signal": "BUY" if desired_direction == 1 else "SELL",
                                        "confidence": ml_confidence,
                                        "approved": True,
                                        "reason": "low_confidence_ignored"
                                    })

                except Exception as e:
                    logger.exception("ML confirmation step failed")
                    if not self.ml_fallback_to_raw:
                        logger.warning("ML required but failed, rejecting signal")
                        return self._reject_signal(signal, "ml_error")
                    else:
                        logger.warning("ML failed, proceeding with raw signal")

                # Predictive models filtering (optional)
                try:
                    if (self.predictive_manager.enabled and
                        signal.signal_type in {SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT}):

                        # Extract market data for predictions
                        market_df = self._extract_market_data_for_prediction(market_data, signal)
                        if market_df is not None and not market_df.empty:
                            # Generate predictions
                            predictions = self.predictive_manager.predict(market_df)

                            # Check if signal should be allowed based on predictions
                            signal_type_str = "BUY" if signal.signal_type == SignalType.ENTRY_LONG else "SELL"
                            if not self.predictive_manager.should_allow_signal(signal_type_str, predictions):
                                logger.info(f"Signal rejected by predictive models: {predictions.price_direction}, {predictions.volatility}, surge={predictions.volume_surge}")
                                return self._reject_signal(signal, "predictive_filter")
                            else:
                                logger.debug(f"Signal approved by predictive models (confidence: {predictions.confidence:.3f})")
                                trade_logger.trade("Predictive models", {
                                    "signal": signal_type_str,
                                    "price_direction": predictions.price_direction,
                                    "volatility": predictions.volatility,
                                    "volume_surge": predictions.volume_surge,
                                    "confidence": predictions.confidence,
                                    "approved": True
                                })

                                # Store predictions in signal metadata for later use
                                if not hasattr(signal, 'metadata') or signal.metadata is None:
                                    signal.metadata = {}
                                signal.metadata['predictions'] = predictions.to_dict()
                        else:
                            logger.debug("Predictive models skipped: no market data available")
                except Exception as e:
                    logger.exception("Predictive models filtering failed")
                    # Don't reject signal on predictive model failure - continue with processing

                # Finalize and store the signal while still holding the symbol lock.
                try:
                    self._store_signal(signal)
                    logger.info(f"Signal approved: {signal}")
                    trade_logger.log_signal(signal_to_dict(signal))
                    return signal
                except Exception as e:
                    logger.exception("Failed to store approved signal")
                    return self._reject_signal(signal, "store_failed")
        except Exception as e:
            logger.exception("Error processing signal under symbol lock")
            return self._reject_signal(signal, "processing_error")

    def _validate_timestamp(self, signal: TradingSignal) -> bool:
        """
        Validate the signal's timestamp to ensure it's a valid datetime.

        Args:
            signal: The trading signal to validate

        Returns:
            True if timestamp is valid, False otherwise
        """
        try:
            import datetime
            import time

            # First check if the original timestamp was corrupted
            # The TradingSignal.__post_init__ stores the original timestamp
            original_timestamp = getattr(signal, '_original_timestamp', signal.timestamp)
            if original_timestamp is None:
                return False

            # If original timestamp was not a datetime, int, or valid string, it was corrupted
            if isinstance(original_timestamp, datetime.datetime):
                # Original was a valid datetime - check if it's reasonable
                # For timezone-naive datetimes, assume they represent UTC time
                try:
                    # Get timestamp, assuming naive datetime is in UTC
                    if original_timestamp.tzinfo is None:
                        # Naive datetime - assume UTC
                        import calendar
                        ts = calendar.timegm(original_timestamp.timetuple()) + original_timestamp.microsecond / 1e6
                    else:
                        # Aware datetime - use standard conversion
                        ts = original_timestamp.timestamp()

                    if ts <= 0:
                        return False
                    current_time = time.time()
                    if ts > current_time + 31536000:  # 1 year in seconds
                        return False
                    return True
                except (OSError, ValueError):
                    # Invalid datetime (e.g., out of range)
                    return False
            elif isinstance(original_timestamp, (int, float)):
                # Original was a valid numeric timestamp - check if reasonable
                if original_timestamp <= 0:
                    return False
                current_time = time.time()
                if original_timestamp > current_time + 31536000:  # 1 year in seconds
                    return False
                return True
            elif isinstance(original_timestamp, str):
                # Original was a string - try to parse it
                try:
                    parsed = datetime.datetime.fromisoformat(original_timestamp.replace('Z', '+00:00'))
                    # Check if parsed timestamp is reasonable
                    if parsed.timestamp() <= 0:
                        return False
                    current_time = time.time()
                    if parsed.timestamp() > current_time + 31536000:  # 1 year in seconds
                        return False
                    return True
                except (ValueError, TypeError):
                    # String couldn't be parsed - it was corrupted
                    return False
            else:
                # Original was some other type that got converted by __post_init__
                # This means it was corrupted
                return False

        except Exception:
            return False

    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate a trading signal's basic properties."""
        # Check for amount (primary field) or quantity (deprecated but still supported)
        amount = signal.amount if signal.amount is not None else signal.quantity
        if not signal.symbol or amount is None or amount <= 0:
            return False

        # OrderType is flexible; require price for LIMIT orders if represented as string 'limit'
        if signal.order_type and getattr(signal.order_type, "value", None) == "limit" and not signal.price:
            return False

        # For entry signals, ensure stop loss if risk manager requires it
        if signal.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
            try:
                require_sl = getattr(self.risk_manager, "require_stop_loss", False)
                if require_sl and not signal.stop_loss:
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
                # Check for opposite signals (always conflicting)
                if self._is_opposite_signal(new_signal, active_signal):
                    conflicts.append(active_signal)
                # Check for same-type signals when strength_based or newer_first is enabled
                elif (self.conflict_resolution_rules["strength_based"] or self.conflict_resolution_rules["newer_first"]) and active_signal.signal_type == new_signal.signal_type:
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

    async def _resolve_conflicts(self, new_signal: TradingSignal, conflicting_signals: List[TradingSignal]) -> Optional[TradingSignal]:
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
                    return self._reject_signal(new_signal, "exit_over_entry")

            if self.conflict_resolution_rules["strength_based"]:
                strongest_conflict = max(
                    conflicting_signals, key=lambda x: x.signal_strength.value
                )
                if (
                    new_signal.signal_strength.value
                    <= strongest_conflict.signal_strength.value
                ):
                    logger.info("New signal rejected due to stronger conflicting signal")
                    return self._reject_signal(new_signal, "stronger_conflict")
                else:
                    # New signal is stronger - cancel conflicting signals
                    for signal in conflicting_signals:
                        await self._cancel_signal(signal)
                    return new_signal

            if self.conflict_resolution_rules["newer_first"]:
                newest_conflict = max(conflicting_signals, key=lambda x: x.timestamp)
                if new_signal.timestamp <= newest_conflict.timestamp:
                    logger.info(f"Resolved conflict: kept newer signal from strategy {newest_conflict.strategy_id}, rejected old from strategy {new_signal.strategy_id}")
                    return self._reject_signal(new_signal, "newer_conflict")
                else:
                    logger.info(f"Resolved conflict: kept newer signal from strategy {new_signal.strategy_id}, rejected old from strategy {newest_conflict.strategy_id}")
                    for signal in conflicting_signals:
                        await self._cancel_signal(signal)
                    return new_signal
        except Exception:
            logger.exception("Error during conflict resolution")
            return self._reject_signal(new_signal, "conflict_resolution_error")

        # Default: reject new signal if any conflicts exist
        return self._reject_signal(new_signal, "default_conflict")

    def _store_signal(self, signal: TradingSignal) -> None:
        """Store the signal in active signals and history."""
        signal_id = self._generate_signal_id(signal)
        self.active_signals[signal_id] = signal
        self.signal_history.append(signal)
        # Append to journal (JSONL) for recovery (best-effort)
        try:
            if getattr(self, "_journal_enabled", False) and getattr(self, "_journal_writer", None):
                entry = {"action": "store", "id": signal_id, "timestamp": now_ms(), "signal": signal_to_dict(signal)}
                try:
                    self._journal_writer.append(entry)
                except Exception:
                    logger.exception("Failed to enqueue journal entry")
        except Exception:
            logger.exception("Failed to append signal to journal")

    async def _cancel_signal(self, signal: TradingSignal) -> None:
        """Cancel an active signal."""
        signal_id = self._generate_signal_id(signal)
        try:
            async with self._lock:
                if signal_id in self.active_signals:
                    del self.active_signals[signal_id]
                    logger.info(f"Cancelled signal: {signal}")
                    # Append cancel to journal (best-effort)
                    try:
                        if getattr(self, "_journal_enabled", False) and getattr(self, "_journal_writer", None):
                            entry = {"action": "cancel", "id": signal_id, "timestamp": now_ms()}
                            try:
                                self._journal_writer.append(entry)
                            except Exception:
                                logger.exception("Failed to enqueue journal cancel")
                    except Exception:
                        logger.exception("Failed to append cancel to journal")
        except Exception:
            logger.exception("Failed to cancel signal")

    def _generate_signal_id(self, signal: TradingSignal) -> str:
        """Generate a unique ID for a signal."""
        return f"{signal.strategy_id}_{signal.symbol}_{signal.timestamp}"

    def recover_from_journal(self) -> None:
        """Replay the JSONL journal to restore active_signals on startup (best-effort).

        This implementation attempts to reconstruct `TradingSignal` dataclass instances
        from journaled dicts (best-effort). If reconstruction fails for a line, the
        original dict is preserved to avoid losing history.
        """
        try:
            if not getattr(self, "_journal_enabled", False):
                return
            if not self.journal_path.exists():
                return
            with open(self.journal_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        action = entry.get("action")
                        sid = entry.get("id")
                        sig = entry.get("signal")
                        if action == "store" and sid and sig:
                            # Restore only if not already present
                            if sid not in self.active_signals:
                                restored = None
                                try:
                                    if isinstance(sig, dict):
                                        # Helper to convert possible enum/string/int to Enum
                                        def _to_signal_type(val):
                                            if isinstance(val, SignalType):
                                                return val
                                            if isinstance(val, str):
                                                try:
                                                    return SignalType[val]
                                                except Exception:
                                                    try:
                                                        return SignalType[int(val)]
                                                    except Exception:
                                                        return None
                                            try:
                                                return SignalType(int(val))
                                            except Exception:
                                                return None

                                        def _to_signal_strength(val):
                                            if isinstance(val, SignalStrength):
                                                return val
                                            if isinstance(val, str):
                                                try:
                                                    return SignalStrength[val]
                                                except Exception:
                                                    try:
                                                        return SignalStrength[int(val)]
                                                    except Exception:
                                                        return None
                                            try:
                                                return SignalStrength(int(val))
                                            except Exception:
                                                return None

                                        def _to_decimal(v):
                                            if v is None:
                                                return None
                                            try:
                                                return Decimal(str(v))
                                            except Exception:
                                                return None

                                        stype = _to_signal_type(sig.get("signal_type"))
                                        sstrength = _to_signal_strength(sig.get("signal_strength"))
                                        amount = _to_decimal(sig.get("amount"))
                                        price = _to_decimal(sig.get("price"))
                                        current_price = _to_decimal(sig.get("current_price"))
                                        stop_loss = _to_decimal(sig.get("stop_loss"))
                                        take_profit = _to_decimal(sig.get("take_profit"))
                                        timestamp = sig.get("timestamp") or 0
                                        metadata = sig.get("metadata")
                                        strategy_id = sig.get("strategy_id") or sig.get("strategy")
                                        symbol = sig.get("symbol")
                                        order_type = sig.get("order_type")

                                        # Ensure required fields exist; fall back to sensible defaults
                                        if strategy_id is None or symbol is None or amount is None:
                                            # Not enough info to build TradingSignal; keep raw dict
                                            restored = sig
                                        else:
                                            restored = TradingSignal(
                                                strategy_id=str(strategy_id),
                                                symbol=str(symbol),
                                                signal_type=stype or SignalType.ENTRY_LONG,
                                                signal_strength=sstrength or SignalStrength.WEAK,
                                                order_type=order_type,
                                                amount=amount,
                                                price=price,
                                                current_price=current_price,
                                                timestamp=int(timestamp) if timestamp else 0,
                                                stop_loss=stop_loss,
                                                take_profit=take_profit,
                                                trailing_stop=sig.get("trailing_stop"),
                                                metadata=metadata,
                                            )
                                    else:
                                        restored = sig
                                except Exception:
                                    # On any reconstruction error, preserve the original dict
                                    restored = sig

                                self.active_signals[sid] = restored
                                self.signal_history.append(restored)
                        elif action == "cancel" and sid:
                            if sid in self.active_signals:
                                del self.active_signals[sid]
                    except Exception:
                        # ignore malformed lines
                        continue
        except Exception:
            logger.exception("Failed to recover signals from journal")

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

    async def _record_router_error(self, exc: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a router-level critical error and enter blocking state if threshold exceeded.
        """
        try:
            async with self._lock:
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
        async with self._lock:
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

    async def close_journal(self) -> None:
        """Gracefully stop the journal writer and flush pending entries."""
        if getattr(self, "_journal_writer", None):
            try:
                await self._journal_writer.stop()
            except Exception:
                logger.exception("Failed to close journal writer")
        return

    async def _get_symbol_lock(self, symbol: Optional[str]) -> asyncio.Lock:
        """
        Return an asyncio.Lock for the given symbol, creating it if necessary.

        This helper serializes lock creation using the router-wide self._lock to
        avoid races when multiple coroutines attempt to create a per-symbol lock.
        If symbol is None, return the global router lock.
        """
        if not symbol:
            return self._lock
        async with self._lock:
            lock = self._symbol_locks.get(symbol)
            if not lock:
                lock = asyncio.Lock()
                self._symbol_locks[symbol] = lock
            return lock
