"""
utils/logger.py

Improved logging system for the trading bot.

- Replaces print statements with Python logging.
- Adds RotatingFileHandler for persistent logs.
- Adds colored console output via colorama.
- Provides TradeLogger (specialized logger) with helper methods for trade/performance logging.
- setup_logging(config) initializes global logging configuration and returns a configured TradeLogger.

Enhancements added:
- Support attaching structured context fields (symbol, component, correlation_id) via LoggerAdapter or passing `extra` to TradeLogger methods.
- Helper `get_logger_with_context()` returns a LoggerAdapter that will propagate structured fields automatically.
- Helper `generate_correlation_id()` to create a simple trace id for correlation between logs.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, Any, List
from datetime import datetime
from pathlib import Path
import json
import csv
import uuid
import errno
import os

from utils.time import now_ms, to_ms, to_iso

from colorama import Fore, Back, Style, init as colorama_init

from utils.adapter import signal_to_dict
from utils.security import SecurityFormatter

# Lazy import to avoid circular dependency
def _get_default_enhanced_event_bus():
    from core.signal_router.event_bus import get_default_enhanced_event_bus
    return get_default_enhanced_event_bus

# Module-level logger for internal library errors (avoid using TradeLogger for internal errors)
logger = logging.getLogger(__name__)

# Initialize colorama (for Windows support)
colorama_init(autoreset=True)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


class ColorFormatter(logging.Formatter):
    """Formatter that adds simple color codes based on levelname."""

    LEVEL_COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.WHITE + Back.RED,
        "TRADE": Fore.MAGENTA,
        "PERF": Fore.BLUE,
    }

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        color = self.LEVEL_COLORS.get(levelname, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging with required fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with required structured fields."""
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",  # UTC ISO8601
            "level": record.levelname,
            "module": getattr(record, 'name', 'unknown'),
            "message": record.getMessage(),
            "correlation_id": getattr(record, 'correlation_id', None),
            "request_id": getattr(record, 'request_id', None),
            "strategy_id": getattr(record, 'strategy_id', None),
        }

        # Add any extra fields from the record
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                              'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                              'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process', 'getMessage']:
                    if value is not None:
                        log_entry[key] = value

        # Handle exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class PrettyFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    def __init__(self, fmt=None, datefmt=None):
        default_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        super().__init__(fmt or default_fmt, datefmt or "%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        """Format with context information."""
        # Add context info to message
        context_parts = []
        if hasattr(record, 'correlation_id') and record.correlation_id:
            context_parts.append(f"corr_id={record.correlation_id}")
        if hasattr(record, 'request_id') and record.request_id:
            context_parts.append(f"req_id={record.request_id}")
        if hasattr(record, 'strategy_id') and record.strategy_id:
            context_parts.append(f"strategy={record.strategy_id}")
        if hasattr(record, 'component') and record.component:
            context_parts.append(f"component={record.component}")

        if context_parts:
            original_msg = record.getMessage()
            record.msg = f"{original_msg} ({' | '.join(context_parts)})"

        return super().format(record)


class TradeLogger(logging.Logger):
    """
    Specialized logger with convenience methods for trade and performance logging.

    This class extends logging.Logger; instantiate it by name (e.g., TradeLogger('crypto_bot')) or use
    setup_logging(...) which returns a configured TradeLogger instance.

    Methods accept an optional `extra` dict (e.g., {"symbol": "BTC/USDT", "component": "order_manager", "correlation_id": "..."})
    which will be passed through to the logging call and can be consumed by external systems or handlers.
    """

    def __init__(self, name: str = "crypto_bot"):
        super().__init__(name)
        self.trades: list[Dict[str, Any]] = []
        self.performance_stats = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
        }

        # CSV file for persisting trades (timestamp, pair, action, size, entry_price, exit_price, pnl)
        self.trade_csv: Path = LOGS_DIR / "trades.csv"
        # Initialize trade CSV header and file unconditionally (independent of logging handlers).
        # This ensures CSV persistence is consistent across different logger configurations.
        self._init_trade_csv()

        # Ensure handlers are only added once if logger already configured
        if not self.handlers:
            # Default console handler; real setup likely done in setup_logging
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                SecurityFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.addHandler(console_handler)

    def _init_trade_csv(self) -> None:
        """Create the trade CSV file with header if it does not exist."""
        try:
            # Ensure parent dir exists
            self.trade_csv.parent.mkdir(parents=True, exist_ok=True)
            if not self.trade_csv.exists():
                with open(self.trade_csv, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [
                            "timestamp",
                            "pair",
                            "action",
                            "size",
                            "entry_price",
                            "exit_price",
                            "pnl",
                        ]
                    )
        except (OSError, IOError) as e:
            # I/O issue while initializing trade CSV — log and continue (non-fatal)
            logger.exception(f"Failed to initialize trade CSV at {self.trade_csv}: {e}")
        except Exception:
            # Unexpected error — log full context and re-raise so calling code can detect programming errors
            logger.exception("Unexpected error during trade CSV initialization")
            raise

    def trade(self, msg: str, trade_data: Dict[str, Any], extra: Optional[Dict[str, Any]] = None, *args, **kwargs) -> None:
        """Log a trade with structured data. Accepts optional extra context."""
        if self.isEnabledFor(logging.INFO):
            extra = extra or {}
            try:
                self.log(21, msg, *args, extra=extra, **kwargs)  # 21 = TRADE custom level
            except TypeError:
                # Older Python logging may not accept extra in this position with custom levels; fallback:
                self.log(21, msg, *args, **kwargs)
            # Record trade; errors here should be surfaced if they are programming errors,
            # but I/O write failures will be handled inside _record_trade.
            self._record_trade({**(trade_data or {}), **extra})

    def performance(self, msg: str, metrics: Dict[str, Any], extra: Optional[Dict[str, Any]] = None, *args, **kwargs) -> None:
        """Log performance metrics with optional extra context."""
        if self.isEnabledFor(22):
            extra = extra or {}
            try:
                self.log(22, msg, *args, extra=extra, **kwargs)
            except TypeError:
                self.log(22, msg, *args, **kwargs)
            self._update_performance(metrics)

    def log_signal(self, signal: Any, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a trading signal (structured). Accepts dataclass/objects and dicts."""
        try:
            sig = signal if isinstance(signal, dict) else signal_to_dict(signal)
            self.trade("New trading signal", {"signal": sig}, extra=extra)
        except (TypeError, AttributeError, ValueError) as e:
            # Likely a conversion or unexpected signal object; surface for debugging.
            logger.exception("Failed to convert or log signal")
            raise
        except Exception:
            logger.exception("Unexpected error while logging signal")
            raise

    def log_order(self, order: Dict[str, Any], mode: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log an order execution in structured form with optional context."""
        try:
            od = order.copy() if isinstance(order, dict) else {"order": str(order)}
            od["mode"] = mode
            self.trade(f"Order executed: {od.get('id', 'n/a')}", od, extra=extra)
        except (TypeError, AttributeError, ValueError) as e:
            logger.exception("Failed to convert or log order")
            raise
        except Exception:
            logger.exception("Unexpected error while logging order")
            raise

    def log_rejected_signal(self, signal: Any, reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Convenience for rejected signal logging. Accepts objects and dicts."""
        try:
            sig = signal if isinstance(signal, dict) else signal_to_dict(signal)
            self.trade(f"Signal rejected: {reason}", {"signal": sig, "reason": reason}, extra=extra)
        except (TypeError, AttributeError, ValueError) as e:
            logger.exception("Failed to convert or log rejected signal")
            raise
        except Exception:
            logger.exception("Unexpected error while logging rejected signal")
            raise

    def log_failed_order(self, signal: Any, error: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Convenience for failed order logging. Accepts objects and dicts."""
        try:
            sig = signal if isinstance(signal, dict) else signal_to_dict(signal)
            self.trade(f"Order failed: {error}", {"signal": sig, "error": error}, extra=extra)
        except (TypeError, AttributeError, ValueError) as e:
            logger.exception("Failed to convert or log failed order")
            raise
        except Exception:
            logger.exception("Unexpected error while logging failed order")
            raise

    def log_binary_prediction(self, symbol: str, probability: float, threshold: float,
                             regime: str, features: Dict[str, float], extra: Optional[Dict[str, Any]] = None) -> None:
        """Log binary model prediction with structured data."""
        try:
            prediction_data = {
                "symbol": symbol,
                "probability": probability,
                "threshold": threshold,
                "regime": regime,
                "features": features,
                "decision": "trade" if probability >= threshold else "skip",
                "confidence": abs(probability - 0.5) * 2  # Scale to 0-1 confidence
            }

            combined_extra = extra or {}
            combined_extra.update({"binary_prediction": prediction_data})

            self.log(PERF_LEVEL, f"Binary prediction: {symbol} p={probability:.3f} ({regime})",
                    extra=combined_extra)

        except Exception as e:
            logger.exception(f"Failed to log binary prediction for {symbol}: {e}")

    def log_binary_decision(self, symbol: str, decision: str, outcome: str, pnl: float,
                           regime: str, strategy: str, probability: float,
                           extra: Optional[Dict[str, Any]] = None) -> None:
        """Log binary model decision outcome with comprehensive details."""
        try:
            decision_data = {
                "symbol": symbol,
                "decision": decision,
                "outcome": outcome,
                "pnl": pnl,
                "regime": regime,
                "strategy": strategy,
                "probability": probability,
                "was_correct": (decision == "trade" and outcome == "profit") or
                              (decision == "skip" and outcome != "profit"),
                "timestamp": now_ms()
            }

            combined_extra = extra or {}
            combined_extra.update({"binary_decision": decision_data})

            outcome_emoji = "✅" if decision_data["was_correct"] else "❌"
            self.log(TRADE_LEVEL, f"{outcome_emoji} Binary decision: {symbol} {decision} -> {outcome} (PnL: {pnl:.2f})",
                    extra=combined_extra)

        except Exception as e:
            logger.exception(f"Failed to log binary decision for {symbol}: {e}")

    def log_binary_model_health(self, metrics: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
        """Log binary model health metrics."""
        try:
            health_data = {
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "calibration_error": metrics.get("calibration_error", 0),
                "prediction_stability": metrics.get("prediction_stability", 0),
                "trade_decision_ratio": metrics.get("trade_decision_ratio", 0),
                "timestamp": now_ms()
            }

            self.log(PERF_LEVEL, f"Binary model health: acc={health_data['accuracy']:.3f}, calib_err={health_data['calibration_error']:.3f}",
                    extra=extra or {}, **{"binary_health": health_data})

        except Exception as e:
            logger.exception(f"Failed to log binary model health: {e}")

    def log_binary_drift_alert(self, alert_type: str, value: float, threshold: float,
                              description: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log binary model drift detection alerts."""
        try:
            alert_data = {
                "alert_type": alert_type,
                "value": value,
                "threshold": threshold,
                "description": description,
                "severity": "critical" if alert_type == "accuracy_drop" else "warning",
                "timestamp": now_ms()
            }

            severity_emoji = "🚨" if alert_data["severity"] == "critical" else "⚠️"
            self.log(logging.WARNING if alert_data["severity"] == "warning" else logging.CRITICAL,
                    f"{severity_emoji} Binary model alert: {alert_type} - {description}",
                    extra=extra or {}, **{"binary_alert": alert_data})

        except Exception as e:
            logger.exception(f"Failed to log binary drift alert: {e}")

    def _record_trade(self, trade_data: Dict[str, Any]) -> None:
        """Append trade to history, update lightweight stats, and persist trade to CSV.

        Expected trade_data keys (recommended):
            - timestamp: ISO string or datetime
            - pair: trading pair (e.g., 'BTC/USDT')
            - action: 'BUY' or 'SELL'
            - size: position size
            - entry_price: entry price
            - exit_price: exit price
            - pnl: profit or loss (float)
            - symbol/component/correlation_id may also be present in trade_data
        """
        try:
            # In-memory record
            self.trades.append(trade_data)

            # Numeric PnL for stats and CSV
            pnl = float(trade_data.get("pnl", 0.0) or 0.0)

            # Update performance statistics
            self.performance_stats["total_trades"] += 1
            self.performance_stats["total_pnl"] += pnl
            if pnl >= 0:
                self.performance_stats["wins"] += 1
                self.performance_stats["max_win"] = max(
                    self.performance_stats["max_win"], pnl
                )
            else:
                self.performance_stats["losses"] += 1
                self.performance_stats["max_loss"] = min(
                    self.performance_stats["max_loss"], pnl
                )

            total = self.performance_stats["wins"] + self.performance_stats["losses"]
            if total > 0:
                self.performance_stats["win_rate"] = (
                    self.performance_stats["wins"] / total
                )

            # Persist to CSV for external analysis (best-effort; only I/O errors are swallowed)
            try:
                # Prefer provided timestamp; default to now (ms)
                timestamp = trade_data.get("timestamp", now_ms())

                # Normalize for CSV output: convert to ISO string using ms normalization
                ts_ms = to_ms(timestamp)
                csv_ts = to_iso(ts_ms if ts_ms is not None else now_ms())

                row = [
                    csv_ts,
                    trade_data.get("pair", ""),
                    trade_data.get("action", trade_data.get("side", "")),
                    trade_data.get("size", ""),
                    trade_data.get("entry_price", ""),
                    trade_data.get("exit_price", ""),
                    pnl,
                ]
                with open(self.trade_csv, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row)
            except (OSError, IOError) as e:
                # I/O error while writing CSV should not crash the application; log and continue.
                logger.exception(f"Failed to write trade to CSV at {self.trade_csv}: {e}")
            except Exception:
                # Unexpected error during write — log and re-raise.
                logger.exception("Unexpected error while writing trade to CSV")
                raise

        except (TypeError, ValueError) as e:
            # Likely a data formatting bug; surface it for debugging.
            logger.exception("Failed to record trade due to data error")
            raise
        except Exception:
            logger.exception("Unexpected error in _record_trade")
            raise

    def _update_performance(self, metrics: Dict[str, Any]) -> None:
        """Merge simple performance metrics into performance_stats."""
        try:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.performance_stats[k] = float(
                        self.performance_stats.get(k, 0.0)
                    ) + float(v)
        except (TypeError, ValueError) as e:
            logger.exception("Failed to update performance stats due to bad metric types")
            raise
        except Exception:
            logger.exception("Unexpected error updating performance stats")
            raise

    def display_performance(self) -> Dict[str, Any]:
        """Return a snapshot of performance statistics."""
        return self.performance_stats.copy()

    def get_trade_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Return the most recent trades, up to `limit`.
        Newer trades are returned first.
        """
        try:
            if limit is None or limit <= 0:
                return list(self.trades[:])
            # return most recent trades first
            return list(reversed(self.trades))[:limit]
        except (TypeError, IndexError) as e:
            logger.exception("Failed to get trade history due to bad arguments")
            raise
        except Exception:
            logger.exception("Unexpected error getting trade history")
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """Compatibility wrapper expected by notifier and other modules."""
        try:
            return self.display_performance()
        except Exception:
            logger.exception("Failed to get performance stats")
            raise

    # ===== EVENT-DRIVEN ARCHITECTURE METHODS =====

    async def handle_event(self, event) -> None:
        """
        Handle incoming events from the event bus.

        Args:
            event: The event to handle
        """
        # Lazy import to avoid circular dependency
        from core.signal_router.events import BaseEvent, EventType

        try:
            if event.event_type == EventType.TRADE_EXECUTED:
                await self._handle_trade_executed_event(event)
            elif event.event_type == EventType.STRATEGY_SWITCH:
                await self._handle_strategy_switch_event(event)
            elif event.event_type == EventType.RISK_LIMIT_TRIGGERED:
                await self._handle_risk_limit_triggered_event(event)
            elif event.event_type == EventType.DIAGNOSTIC_ALERT:
                await self._handle_diagnostic_alert_event(event)
            elif event.event_type == EventType.KNOWLEDGE_ENTRY_CREATED:
                await self._handle_knowledge_entry_created_event(event)
            elif event.event_type == EventType.REGIME_CHANGE:
                await self._handle_regime_change_event(event)
            elif event.event_type == EventType.SYSTEM_STATUS_UPDATE:
                await self._handle_system_status_update_event(event)
            else:
                # Log other events at debug level
                self.debug(f"Event received: {event.event_type.value}", extra={"event_data": event.to_dict()})

        except Exception as e:
            logger.exception(f"Error handling event {event.event_type.value}: {e}")

    async def _handle_trade_executed_event(self, event) -> None:
        """Handle trade executed events."""
        payload = event.payload
        trade_data = {
            "timestamp": event.timestamp.isoformat(),
            "pair": payload.get("symbol", ""),
            "action": payload.get("side", ""),
            "size": payload.get("quantity", ""),
            "entry_price": payload.get("price", ""),
            "pnl": 0.0,  # Will be updated when trade closes
            "strategy": payload.get("strategy", ""),
            "trade_id": payload.get("trade_id", ""),
            "slippage": payload.get("slippage", ""),
            "commission": payload.get("commission", "")
        }

        self.trade("Trade executed", trade_data, extra={
            "symbol": payload.get("symbol"),
            "component": event.source,
            "correlation_id": generate_correlation_id()
        })

    async def _handle_strategy_switch_event(self, event) -> None:
        """Handle strategy switch events."""
        payload = event.payload
        strategy_data = {
            "previous_strategy": payload.get("previous_strategy"),
            "new_strategy": payload.get("new_strategy"),
            "rationale": payload.get("rationale"),
            "confidence": payload.get("confidence"),
            "market_conditions": payload.get("market_conditions")
        }

        self.performance("Strategy switched", strategy_data, extra={
            "component": event.source,
            "correlation_id": generate_correlation_id()
        })

    async def _handle_risk_limit_triggered_event(self, event) -> None:
        """Handle risk limit triggered events."""
        payload = event.payload
        risk_data = {
            "risk_factor": payload.get("risk_factor"),
            "trigger_condition": payload.get("trigger_condition"),
            "current_value": payload.get("current_value"),
            "threshold_value": payload.get("threshold_value"),
            "defensive_action": payload.get("defensive_action"),
            "symbol": payload.get("symbol")
        }

        self.warning("Risk limit triggered", extra={
            "symbol": payload.get("symbol"),
            "component": event.source,
            "correlation_id": generate_correlation_id(),
            "risk_data": risk_data
        })

    async def _handle_diagnostic_alert_event(self, event) -> None:
        """Handle diagnostic alert events."""
        payload = event.payload
        alert_type = payload.get("alert_type", "info")
        component = payload.get("component", "unknown")
        message = payload.get("message", "")

        if alert_type == "error":
            self.error(f"Diagnostic alert from {component}: {message}", extra={
                "component": event.source,
                "correlation_id": generate_correlation_id(),
                "alert_details": payload.get("details")
            })
        elif alert_type == "warning":
            self.warning(f"Diagnostic alert from {component}: {message}", extra={
                "component": event.source,
                "correlation_id": generate_correlation_id(),
                "alert_details": payload.get("details")
            })
        else:
            self.info(f"Diagnostic alert from {component}: {message}", extra={
                "component": event.source,
                "correlation_id": generate_correlation_id(),
                "alert_details": payload.get("details")
            })

    async def _handle_knowledge_entry_created_event(self, event) -> None:
        """Handle knowledge entry created events."""
        payload = event.payload
        knowledge_data = {
            "entry_id": payload.get("entry_id"),
            "regime": payload.get("regime"),
            "strategy": payload.get("strategy"),
            "outcome": payload.get("outcome"),
            "performance_metrics": payload.get("performance_metrics")
        }

        # Use custom KNOWLEDGE level
        self.log(KNOWLEDGE_LEVEL, "Knowledge entry created", extra={
            "component": event.source,
            "correlation_id": generate_correlation_id(),
            "knowledge_data": knowledge_data
        })

    async def _handle_regime_change_event(self, event) -> None:
        """Handle regime change events."""
        payload = event.payload
        regime_data = {
            "old_regime": payload.get("old_regime"),
            "new_regime": payload.get("new_regime"),
            "confidence": payload.get("confidence")
        }

        self.info("Market regime changed", extra={
            "component": event.source,
            "correlation_id": generate_correlation_id(),
            "regime_data": regime_data
        })

    async def _handle_system_status_update_event(self, event) -> None:
        """Handle system status update events."""
        payload = event.payload
        status_data = {
            "component": payload.get("component"),
            "status": payload.get("status"),
            "details": payload.get("details")
        }

        self.info(f"System status update: {payload.get('component')} is {payload.get('status')}", extra={
            "component": event.source,
            "correlation_id": generate_correlation_id(),
            "status_data": status_data
        })


# Register custom log levels for TRADE, PERF, and KNOWLEDGE
TRADE_LEVEL = 21
PERF_LEVEL = 22
KNOWLEDGE_LEVEL = 23
logging.addLevelName(TRADE_LEVEL, "TRADE")
logging.addLevelName(PERF_LEVEL, "PERF")
logging.addLevelName(KNOWLEDGE_LEVEL, "KNOWLEDGE")


# Expose a module-level singleton logger (created by setup_logging)
_GLOBAL_TRADE_LOGGER: Optional[TradeLogger] = None


def setup_logging(config: Optional[Dict[str, Any]] = None) -> TradeLogger:
    """
    Configure root logging and return a configured TradeLogger instance.

    Supports environment variables:
    - LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
    - LOG_FILE: Path to log file (default: logs/crypto_bot.log)
    - LOG_FORMAT: json, pretty, or color (default: color)

    Args:
        config: Optional dict with logging configuration. Expected keys:
            - level (str or int)
            - file_logging (bool)
            - log_file (str)
            - max_size (int)
            - backup_count (int)
            - console (bool)
            - format (str): json, pretty, or color

    Returns:
        Configured TradeLogger
    """
    global _GLOBAL_TRADE_LOGGER

    # Get environment variables
    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    env_log_file = os.getenv("LOG_FILE")
    env_format = os.getenv("LOG_FORMAT", "color").lower()

    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if env_level not in valid_levels:
        print(f"Warning: Invalid LOG_LEVEL '{env_level}', using INFO")
        env_level = "INFO"

    # Defaults
    cfg = {
        "level": env_level,
        "file_logging": True,
        "log_file": env_log_file or str(LOGS_DIR / "crypto_bot.log"),
        "max_size": 10 * 1024 * 1024,
        "backup_count": 5,
        "console": True,
        "format": env_format,
    }
    if config:
        cfg.update(config)

    # Parse level
    level = cfg.get("level", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Configure basic logging on root
    logging.basicConfig(level=level)  # handlers will be added below

    # Create (or reuse) TradeLogger instance
    logger_name = "crypto_bot"
    if _GLOBAL_TRADE_LOGGER is None:
        _GLOBAL_TRADE_LOGGER = TradeLogger(logger_name)

    trade_logger = _GLOBAL_TRADE_LOGGER
    trade_logger.setLevel(level)

    # Remove existing handlers and reconfigure to avoid duplication
    for h in list(trade_logger.handlers):
        trade_logger.removeHandler(h)

    # Determine format
    log_format = cfg.get("format", "color")

    # Console handler
    if cfg.get("console", True):
        console_handler = logging.StreamHandler(sys.stdout)

        if log_format == "json":
            console_handler.setFormatter(JSONFormatter())
        elif log_format == "pretty":
            console_handler.setFormatter(PrettyFormatter())
        else:  # color or default
            console_handler.setFormatter(ColorFormatter())

        console_handler.setLevel(level)
        trade_logger.addHandler(console_handler)

    # File handler (always JSON for structured logging)
    if cfg.get("file_logging", True):
        log_file = Path(cfg.get("log_file"))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        max_bytes = int(cfg.get("max_size", 10 * 1024 * 1024))
        backup_count = int(cfg.get("backup_count", 5))

        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        # Always use JSON formatter for file output
        file_handler.setFormatter(JSONFormatter())
        file_handler.setLevel(level)
        trade_logger.addHandler(file_handler)

    # Prevent propagation to root handlers to avoid duplicate log messages
    trade_logger.propagate = False

    # Subscribe to event bus for event-driven logging
    try:
        event_bus_func = _get_default_enhanced_event_bus()
        event_bus = event_bus_func()

        # Lazy import to avoid circular dependency
        from core.signal_router.events import EventType

        # Subscribe to all event types for logging
        for event_type in EventType:
            event_bus.subscribe(event_type, trade_logger.handle_event)

        logger.info("TradeLogger subscribed to event bus for event-driven logging")

    except Exception as e:
        logger.warning(f"Failed to subscribe TradeLogger to event bus: {e}")
        # Continue without event subscription - logging will still work

    return trade_logger


def get_trade_logger() -> TradeLogger:
    """Return the configured global TradeLogger instance (calls setup_logging with defaults if needed)."""
    global _GLOBAL_TRADE_LOGGER
    if _GLOBAL_TRADE_LOGGER is None:
        setup_logging()
    return _GLOBAL_TRADE_LOGGER


def get_logger(name: str) -> logging.Logger:
    """Return a standard logger with the given name."""
    return logging.getLogger(name)


def generate_correlation_id() -> str:
    """Return a short correlation id for tracing log messages across components."""
    return uuid.uuid4().hex


def generate_request_id() -> str:
    """Return a unique request id for tracking individual requests."""
    return f"req_{uuid.uuid4().hex[:16]}"


def get_logger_with_context(symbol: Optional[str] = None, component: Optional[str] = None,
                           correlation_id: Optional[str] = None, request_id: Optional[str] = None,
                           strategy_id: Optional[str] = None) -> logging.LoggerAdapter:
    """
    Return a LoggerAdapter that will attach structured context fields to all log records.

    Usage:
        adapter = get_logger_with_context(
            symbol='BTC/USDT',
            component='order_manager',
            correlation_id=generate_correlation_id(),
            request_id=generate_request_id(),
            strategy_id='momentum_v1'
        )
        adapter.info('Starting execution')  # record will have those extra fields
    """
    base = get_trade_logger()

    extra = {}
    if symbol:
        extra["symbol"] = symbol
    if component:
        extra["component"] = component
    if correlation_id:
        extra["correlation_id"] = correlation_id
    if request_id:
        extra["request_id"] = request_id
    if strategy_id:
        extra["strategy_id"] = strategy_id

    return logging.LoggerAdapter(base, extra)


def log_to_file(data: Dict[str, Any], filename: str) -> None:
    """
    Append structured data to a JSON file inside the logs directory.

    Args:
        data: Data to append (dict)
        filename: Name of the file (without extension)
    """
    filepath = LOGS_DIR / f"{filename}.json"
    try:
        mode = "a" if filepath.exists() else "w"
        with open(filepath, mode, encoding="utf-8") as f:
            if mode == "a":
                f.write("\n")
            json.dump(data, f, indent=2, default=str)
    except (OSError, IOError) as e:
        # I/O errors are logged but do not raise to avoid breaking main flow
        logger.exception(f"Failed to log to file (I/O error): {filepath}: {e}")
    except TypeError as e:
        # Data not serializable - programming/data error: surface it
        logger.exception(f"Failed to serialize data for logging to file {filepath}: {e}")
        raise
    except Exception:
        logger.exception(f"Unexpected error while logging to file: {filepath}")
        raise
