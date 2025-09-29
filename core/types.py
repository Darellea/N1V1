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
