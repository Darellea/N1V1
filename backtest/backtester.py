"""
backtest/backtester.py

Utility to export equity progression produced during a backtest to CSV and compute summary metrics.

Provides:
- export_equity_progression(equity_progression, out_path)
- compute_backtest_metrics(equity_progression)
- export_metrics(metrics, out_path)
- export_equity_from_botengine(bot_engine, out_path)

The CSV will be written to `results/equity_curve.csv` by default and will
contain columns: trade_id, timestamp, equity, pnl, cumulative_return.

Metrics produced:
- equity_curve (list)
- max_drawdown
- sharpe_ratio (annualized, sqrt(252))
- profit_factor
- total_return
- total_trades
- win_rate
"""
from __future__ import annotations

import os
import csv
import json
from typing import List, Dict, Any, Optional
from statistics import mean, stdev
from math import sqrt, isfinite


def _ensure_results_dir(path: str) -> None:
    """Ensure the parent directory for path exists."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def export_equity_progression(
    equity_progression: List[Dict[str, Any]], out_path: str = "results/equity_curve.csv"
) -> str:
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

    fieldnames = ["trade_id", "timestamp", "equity", "pnl", "cumulative_return"]

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in rows:
            # Ensure keys exist; convert None to empty string
            out = {
                "trade_id": rec.get("trade_id", ""),
                "timestamp": rec.get("timestamp", ""),
                "equity": rec.get("equity", "")
                if rec.get("equity", None) is not None
                else "",
                "pnl": rec.get("pnl", "") if rec.get("pnl", None) is not None else "",
                "cumulative_return": rec.get("cumulative_return", "")
                if rec.get("cumulative_return", None) is not None
                else "",
            }
            writer.writerow(out)

    return out_path


def compute_backtest_metrics(
    equity_progression: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute backtest summary metrics from equity progression.

    This function supports both single-series equity_progression (global portfolio)
    as well as mixed records containing a 'symbol' key (per-pair records). When per-symbol
    records are detected, it will compute per-symbol metrics and also return aggregated
    portfolio metrics computed from the full progression.

    Args:
        equity_progression: list of records containing 'equity' and 'pnl' per trade.
                           Records may optionally include 'symbol' to indicate the pair.

    Returns:
        Dict with keys:
          - equity_curve (list of equity values)
          - max_drawdown
          - sharpe_ratio
          - profit_factor
          - total_return
          - total_trades
          - wins
          - losses
          - win_rate
        If per-symbol records are present, adds 'per_symbol' mapping symbol->metrics.
    """
    def _compute_for_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute metrics for a flat list of records."""
        equity_vals: List[float] = []
        pnls: List[float] = []
        for rec in records:
            try:
                eq = rec.get("equity", 0.0)
                eqv = float(eq) if eq is not None else 0.0
            except Exception:
                eqv = 0.0
            equity_vals.append(eqv)

            try:
                pnl = rec.get("pnl", None)
                pnls.append(float(pnl) if pnl is not None else 0.0)
            except Exception:
                pnls.append(0.0)

        # Total return (relative to first equity if available)
        try:
            start_eq = equity_vals[0] if equity_vals else 0.0
            end_eq = equity_vals[-1] if equity_vals else 0.0
            total_return = (
                (end_eq - start_eq) / start_eq if start_eq and start_eq != 0 else 0.0
            )
        except Exception:
            total_return = 0.0

        # Equity returns per trade (percentage)
        returns: List[float] = []
        for i in range(1, len(equity_vals)):
            prev = equity_vals[i - 1]
            curr = equity_vals[i]
            if prev and prev != 0:
                returns.append((curr - prev) / prev)
            else:
                returns.append(0.0)

        # Max drawdown
        max_dd = 0.0
        peak = equity_vals[0] if equity_vals else 0.0
        for val in equity_vals:
            if val > peak:
                peak = val
            if peak and peak != 0:
                dd = (peak - val) / peak
                if dd > max_dd:
                    max_dd = dd

        # Sharpe ratio (annualized). Use 252 as annualization factor (trading days)
        sharpe = 0.0
        try:
            if len(returns) > 1:
                if stdev(returns) > 0:
                    sharpe = (mean(returns) / stdev(returns)) * sqrt(252)
        except Exception:
            sharpe = 0.0

        # Profit factor: gross profit / gross loss
        gross_profit = sum([p for p in pnls if p > 0])
        gross_loss = sum([-p for p in pnls if p < 0])  # positive number
        profit_factor = 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            if gross_profit > 0:
                profit_factor = float("inf")
            else:
                profit_factor = 0.0

        # Win rate & counts
        wins = len([p for p in pnls if p > 0])
        losses = len([p for p in pnls if p < 0])
        total_trades = wins + losses
        win_rate = (wins / total_trades) if total_trades > 0 else 0.0

        # Expectancy (per-trade): average winning trade * win_prob - average losing trade * loss_prob
        try:
            avg_win = (gross_profit / wins) if wins > 0 else 0.0
        except Exception:
            avg_win = 0.0
        try:
            avg_loss = (gross_loss / losses) if losses > 0 else 0.0
        except Exception:
            avg_loss = 0.0
        expectancy = (avg_win * (wins / total_trades) - avg_loss * (losses / total_trades)) if total_trades > 0 else 0.0

        return {
            "equity_curve": equity_vals,
            "max_drawdown": float(max_dd),
            "sharpe_ratio": float(sharpe) if isfinite(sharpe) else 0.0,
            "profit_factor": float(profit_factor)
            if isfinite(profit_factor)
            else float("inf"),
            "total_return": float(total_return),
            "total_trades": int(total_trades),
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "expectancy": float(expectancy),
        }

    metrics: Dict[str, Any] = {}
    if not equity_progression:
        # return defaults
        metrics.update(
            {
                "equity_curve": [],
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "total_return": 0.0,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
            }
        )
        return metrics

    # Detect whether records contain per-symbol information
    has_symbol = any(("symbol" in rec and rec.get("symbol") is not None) for rec in equity_progression)

    # Compute overall metrics from full progression (preserves existing behavior)
    overall = _compute_for_records(equity_progression)
    metrics.update(overall)

    # If per-symbol records exist, compute per-symbol metrics as well
    if has_symbol:
        per_symbol: Dict[str, Dict[str, Any]] = {}
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for rec in equity_progression:
            sym = rec.get("symbol") or "__unspecified__"
            groups.setdefault(sym, []).append(rec)

        for sym, recs in groups.items():
            per_symbol[sym] = _compute_for_records(recs)

        metrics["per_symbol"] = per_symbol

    return metrics


def export_metrics(
    metrics: Dict[str, Any], out_path: str = "results/metrics.json"
) -> str:
    """
    Export computed metrics to a JSON file and a companion CSV summary.

    Args:
        metrics: Metrics dict returned by compute_backtest_metrics
        out_path: Destination path for metrics JSON (default: results/metrics.json)

    Returns:
        Path to JSON metrics file.
    """
    _ensure_results_dir(out_path)
    # Write JSON
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # Also write a simple CSV summary for quick viewing
    csv_path = os.path.splitext(out_path)[0] + "_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        # Write scalar metrics only (skip equity_curve list)
        for k, v in metrics.items():
            if k == "equity_curve":
                continue
            writer.writerow([k, v])

    return out_path


def export_equity_from_botengine(
    bot_engine: Any, out_path: str = "results/equity_curve.csv"
) -> str:
    """
    Helper to extract equity_progression from a BotEngine-like object, export it,
    compute summary metrics and export those as well.

    Args:
        bot_engine: object exposing performance_stats['equity_progression']
        out_path: destination CSV path for equity curve

    Returns:
        The path to the written equity CSV file (out_path).
    """
    try:
        equity_progression = bot_engine.performance_stats.get("equity_progression", [])
    except Exception:
        # Defensive: try attribute access
        equity_progression = getattr(bot_engine, "equity_progression", []) or []

    # Export equity progression CSV
    equity_csv = export_equity_progression(equity_progression, out_path=out_path)

    # Compute metrics and export
    metrics = compute_backtest_metrics(equity_progression)
    metrics_json_path = os.path.join(
        os.path.dirname(out_path) or "results", "metrics.json"
    )
    export_metrics(metrics, out_path=metrics_json_path)

    return equity_csv
