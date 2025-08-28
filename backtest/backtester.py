"""
backtest/backtester.py

Utility to export equity progression produced during a backtest to CSV.

Provides:
- export_equity_progression(equity_progression, out_path)
- export_equity_from_botengine(bot_engine, out_path)

The CSV will be written to `results/equity_curve.csv` by default and will
contain columns: trade_id, timestamp, equity, pnl, cumulative_return.
"""
from __future__ import annotations

import os
import csv
from typing import List, Dict, Any, Optional

def _ensure_results_dir(path: str) -> None:
    """Ensure the parent directory for path exists."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def export_equity_progression(equity_progression: List[Dict[str, Any]],
                              out_path: str = "results/equity_curve.csv") -> str:
    """
    Export a list of equity progression records to CSV.

    Args:
        equity_progression: list of dicts with keys:
            'trade_id', 'timestamp', 'equity', 'pnl', 'cumulative_return'
        out_path: destination CSV path (default: results/equity_curve.csv)

    Returns:
        The path to the written CSV file.
    """
    # Normalize input
    rows = equity_progression or []
    _ensure_results_dir(out_path)

    fieldnames = ['trade_id', 'timestamp', 'equity', 'pnl', 'cumulative_return']

    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in rows:
            # Ensure keys exist; convert None to empty string
            out = {
                'trade_id': rec.get('trade_id', ''),
                'timestamp': rec.get('timestamp', ''),
                'equity': rec.get('equity', '') if rec.get('equity', None) is not None else '',
                'pnl': rec.get('pnl', '') if rec.get('pnl', None) is not None else '',
                'cumulative_return': rec.get('cumulative_return', '') if rec.get('cumulative_return', None) is not None else '',
            }
            writer.writerow(out)

    return out_path

def export_equity_from_botengine(bot_engine: Any,
                                 out_path: str = "results/equity_curve.csv") -> str:
    """
    Helper to extract equity_progression from a BotEngine-like object and export it.

    Args:
        bot_engine: object exposing performance_stats['equity_progression']
        out_path: destination CSV path

    Returns:
        The path to the written CSV file.
    """
    try:
        equity_progression = bot_engine.performance_stats.get('equity_progression', [])
    except Exception:
        # Defensive: try attribute access
        equity_progression = getattr(bot_engine, 'equity_progression', []) or []

    return export_equity_progression(equity_progression, out_path=out_path)
