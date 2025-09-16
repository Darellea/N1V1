"""
Backtest module for trading strategy evaluation.

This module provides functionality for backtesting trading strategies,
including equity progression export, metrics computation, and regime-aware analysis.
"""

from .backtester import (
    Backtester,
    BacktestSecurityError,
    BacktestValidationError,
    export_equity_progression,
    compute_backtest_metrics,
    export_metrics,
    export_equity_from_botengine,
    compute_regime_aware_metrics,
    export_regime_aware_report,
    export_regime_aware_equity_progression,
    export_regime_aware_equity_from_botengine,
)

__all__ = [
    'Backtester',
    'BacktestSecurityError',
    'BacktestValidationError',
    'export_equity_progression',
    'compute_backtest_metrics',
    'export_metrics',
    'export_equity_from_botengine',
    'compute_regime_aware_metrics',
    'export_regime_aware_report',
    'export_regime_aware_equity_progression',
    'export_regime_aware_equity_from_botengine',
]
