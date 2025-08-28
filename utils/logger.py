"""
utils/logger.py

Improved logging system for the trading bot.

- Replaces print statements with Python logging.
- Adds RotatingFileHandler for persistent logs.
- Adds colored console output via colorama.
- Provides TradeLogger (specialized logger) with helper methods for trade/performance logging.
- setup_logging(config) initializes global logging configuration and returns a configured TradeLogger.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import json

from colorama import Fore, Back, Style, init as colorama_init

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


class TradeLogger(logging.Logger):
    """
    Specialized logger with convenience methods for trade and performance logging.

    This class extends logging.Logger; instantiate it by name (e.g., TradeLogger('crypto_bot')) or use
    setup_logging(...) which returns a configured TradeLogger instance.
    """

    def __init__(self, name: str = "crypto_bot"):
        super().__init__(name)
        self.trades = []
        self.performance_stats = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
        }
        # Ensure handlers are only added once if logger already configured
        if not self.handlers:
            # Default console handler; real setup likely done in setup_logging
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ColorFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.addHandler(console_handler)

    def trade(self, msg: str, trade_data: Dict[str, Any], *args, **kwargs) -> None:
        """Log a trade with structured data."""
        if self.isEnabledFor(logging.INFO):
            self.log(21, msg, *args, **kwargs)  # 21 = TRADE custom level
            self._record_trade(trade_data)

    def performance(self, msg: str, metrics: Dict[str, Any], *args, **kwargs) -> None:
        """Log performance metrics."""
        if self.isEnabledFor(22):
            self.log(22, msg, *args, **kwargs)
            self._update_performance(metrics)

    def log_signal(self, signal: Dict[str, Any]) -> None:
        """Log a trading signal (structured)."""
        try:
            self.trade("New trading signal", {"signal": signal})
        except Exception:
            self.exception("Failed to log signal")

    def log_order(self, order: Dict[str, Any], mode: str) -> None:
        """Log an order execution in structured form."""
        try:
            od = order.copy() if isinstance(order, dict) else {"order": str(order)}
            od["mode"] = mode
            self.trade(f"Order executed: {od.get('id', 'n/a')}", od)
        except Exception:
            self.exception("Failed to log order")

    def log_rejected_signal(self, signal: Dict[str, Any], reason: str) -> None:
        """Convenience for rejected signal logging."""
        try:
            self.trade(f"Signal rejected: {reason}", {"signal": signal, "reason": reason})
        except Exception:
            self.exception("Failed to log rejected signal")

    def log_failed_order(self, signal: Dict[str, Any], error: str) -> None:
        """Convenience for failed order logging."""
        try:
            self.trade(f"Order failed: {error}", {"signal": signal, "error": error})
        except Exception:
            self.exception("Failed to log failed order")

    def _record_trade(self, trade_data: Dict[str, Any]) -> None:
        """Append trade to history and update lightweight stats."""
        try:
            self.trades.append(trade_data)
            pnl = float(trade_data.get("pnl", 0.0) or 0.0)
            self.performance_stats["total_trades"] += 1
            self.performance_stats["total_pnl"] += pnl
            if pnl >= 0:
                self.performance_stats["wins"] += 1
                self.performance_stats["max_win"] = max(self.performance_stats["max_win"], pnl)
            else:
                self.performance_stats["losses"] += 1
                self.performance_stats["max_loss"] = min(self.performance_stats["max_loss"], pnl)

            total = self.performance_stats["wins"] + self.performance_stats["losses"]
            if total > 0:
                self.performance_stats["win_rate"] = self.performance_stats["wins"] / total
        except Exception:
            self.exception("Failed to record trade")

    def _update_performance(self, metrics: Dict[str, Any]) -> None:
        """Merge simple performance metrics into performance_stats."""
        try:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.performance_stats[k] = float(self.performance_stats.get(k, 0.0)) + float(v)
        except Exception:
            self.exception("Failed to update performance stats")

    def display_performance(self) -> Dict[str, Any]:
        """Return a snapshot of performance statistics."""
        return self.performance_stats.copy()


# Register custom log levels for TRADE and PERF
TRADE_LEVEL = 21
PERF_LEVEL = 22
logging.addLevelName(TRADE_LEVEL, "TRADE")
logging.addLevelName(PERF_LEVEL, "PERF")


# Expose a module-level singleton logger (created by setup_logging)
_GLOBAL_TRADE_LOGGER: Optional[TradeLogger] = None


def setup_logging(config: Optional[Dict[str, Any]] = None) -> TradeLogger:
    """
    Configure root logging and return a configured TradeLogger instance.

    Args:
        config: Optional dict with logging configuration. Expected keys:
            - level (str or int)
            - file_logging (bool)
            - log_file (str)
            - max_size (int)
            - backup_count (int)
            - console (bool)

    Returns:
        Configured TradeLogger
    """
    global _GLOBAL_TRADE_LOGGER

    # Defaults
    cfg = {
        "level": "INFO",
        "file_logging": True,
        "log_file": str(LOGS_DIR / "crypto_bot.log"),
        "max_size": 10 * 1024 * 1024,
        "backup_count": 5,
        "console": True,
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

    # Console handler
    if cfg.get("console", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        console_handler.setFormatter(ColorFormatter(console_fmt))
        console_handler.setLevel(level)
        trade_logger.addHandler(console_handler)

    # Rotating file handler
    if cfg.get("file_logging", True):
        log_file = Path(cfg.get("log_file"))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        max_bytes = int(cfg.get("max_size", 10 * 1024 * 1024))
        backup_count = int(cfg.get("backup_count", 5))

        file_handler = RotatingFileHandler(str(log_file), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        file_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file_handler.setFormatter(logging.Formatter(file_fmt))
        file_handler.setLevel(level)
        trade_logger.addHandler(file_handler)

    # Prevent propagation to root handlers to avoid duplicate log messages
    trade_logger.propagate = False

    return trade_logger


def get_trade_logger() -> TradeLogger:
    """Return the configured global TradeLogger instance (calls setup_logging with defaults if needed)."""
    global _GLOBAL_TRADE_LOGGER
    if _GLOBAL_TRADE_LOGGER is None:
        setup_logging()
    return _GLOBAL_TRADE_LOGGER


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
    except Exception:
        print(f"Failed to log to file: {filepath}", file=sys.stderr)
