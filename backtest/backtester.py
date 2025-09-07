"""
backtest/backtester.py

Utility to export equity progression produced during a backtest to CSV and compute summary metrics.
Enhanced with regime-aware backtesting capabilities.

Provides:
- export_equity_progression(equity_progression, out_path)
- compute_backtest_metrics(equity_progression)
- export_metrics(metrics, out_path)
- export_equity_from_botengine(bot_engine, out_path)
- compute_regime_aware_metrics(equity_progression, regime_data)
- export_regime_aware_report(metrics, out_path)

The CSV will be written to `results/equity_curve.csv` by default and will
contain columns: trade_id, timestamp, equity, pnl, cumulative_return, regime_name, confidence_score.

Metrics produced:
- equity_curve (list)
- max_drawdown
- sharpe_ratio (annualized, sqrt(252))
- profit_factor
- total_return
- total_trades
- wins
- losses
- win_rate
- per_regime_metrics (regime-aware breakdown)
"""
from __future__ import annotations

import os
import csv
import json
from typing import List, Dict, Any, Optional
from statistics import mean, stdev
from math import sqrt, isfinite
import pandas as pd
from datetime import datetime


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


def compare_ensemble_vs_individual(
    ensemble_equity: List[Dict[str, Any]],
    individual_equities: Dict[str, List[Dict[str, Any]]],
    out_path: str = "results/ensemble_comparison.json"
) -> Dict[str, Any]:
    """
    Compare ensemble performance against individual strategy performances.

    Args:
        ensemble_equity: Equity progression for ensemble strategy
        individual_equities: Dict of strategy_id -> equity progression
        out_path: Path to save comparison results

    Returns:
        Comparison metrics dictionary
    """
    _ensure_results_dir(out_path)

    # Compute metrics for ensemble
    ensemble_metrics = compute_backtest_metrics(ensemble_equity)

    # Compute metrics for each individual strategy
    individual_metrics = {}
    for strategy_id, equity in individual_equities.items():
        individual_metrics[strategy_id] = compute_backtest_metrics(equity)

    # Compute comparison metrics
    comparison = {
        "ensemble": ensemble_metrics,
        "individual_strategies": individual_metrics,
        "comparison": {}
    }

    # Calculate improvement metrics
    ensemble_return = ensemble_metrics.get("total_return", 0)
    ensemble_sharpe = ensemble_metrics.get("sharpe_ratio", 0)
    ensemble_win_rate = ensemble_metrics.get("win_rate", 0)

    best_individual_return = max(
        (metrics.get("total_return", 0) for metrics in individual_metrics.values()),
        default=0
    )
    best_individual_sharpe = max(
        (metrics.get("sharpe_ratio", 0) for metrics in individual_metrics.values()),
        default=0
    )
    best_individual_win_rate = max(
        (metrics.get("win_rate", 0) for metrics in individual_metrics.values()),
        default=0
    )

    comparison["comparison"] = {
        "return_improvement": ensemble_return - best_individual_return,
        "sharpe_improvement": ensemble_sharpe - best_individual_sharpe,
        "win_rate_improvement": ensemble_win_rate - best_individual_win_rate,
        "ensemble_vs_best_individual": {
            "total_return": {
                "ensemble": ensemble_return,
                "best_individual": best_individual_return,
                "improvement_pct": ((ensemble_return - best_individual_return) / abs(best_individual_return) * 100) if best_individual_return != 0 else 0
            },
            "sharpe_ratio": {
                "ensemble": ensemble_sharpe,
                "best_individual": best_individual_sharpe,
                "improvement": ensemble_sharpe - best_individual_sharpe
            },
            "win_rate": {
                "ensemble": ensemble_win_rate,
                "best_individual": best_individual_win_rate,
                "improvement_pct": (ensemble_win_rate - best_individual_win_rate) * 100
            }
        },
        "ensemble_vs_average": {
            "total_return": {
                "ensemble": ensemble_return,
                "average_individual": sum(metrics.get("total_return", 0) for metrics in individual_metrics.values()) / len(individual_metrics) if individual_metrics else 0
            },
            "sharpe_ratio": {
                "ensemble": ensemble_sharpe,
                "average_individual": sum(metrics.get("sharpe_ratio", 0) for metrics in individual_metrics.values()) / len(individual_metrics) if individual_metrics else 0
            },
            "win_rate": {
                "ensemble": ensemble_win_rate,
                "average_individual": sum(metrics.get("win_rate", 0) for metrics in individual_metrics.values()) / len(individual_metrics) if individual_metrics else 0
            }
        }
    }

    # Save comparison results
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)

    return comparison


def export_regime_aware_equity_progression(
    equity_progression: List[Dict[str, Any]], out_path: str = "results/regime_equity_curve.csv"
) -> str:
    """
    Export regime-aware equity progression records to CSV.

    Args:
        equity_progression: list of dicts with regime information
        out_path: destination CSV path

    Returns:
        The path to the written CSV file.
    """
    # Normalize input
    rows = equity_progression or []
    _ensure_results_dir(out_path)

    fieldnames = ["trade_id", "timestamp", "equity", "pnl", "cumulative_return",
                 "regime_name", "confidence_score", "regime_features"]

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
                "regime_name": rec.get("regime_name", ""),
                "confidence_score": rec.get("confidence_score", ""),
                "regime_features": json.dumps(rec.get("regime_features", {}))
            }
            writer.writerow(out)

    return out_path


def compute_regime_aware_metrics(
    equity_progression: List[Dict[str, Any]],
    regime_data: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Compute backtest metrics with regime-aware breakdown.

    Args:
        equity_progression: List of equity progression records
        regime_data: Optional list of regime detection results

    Returns:
        Dict with overall and per-regime metrics
    """
    # First compute overall metrics
    overall_metrics = compute_backtest_metrics(equity_progression)

    if not regime_data or not equity_progression:
        return {
            "overall": overall_metrics,
            "per_regime": {},
            "regime_summary": {}
        }

    # Group equity progression by regime
    regime_groups: Dict[str, List[Dict[str, Any]]] = {}
    regime_confidences: Dict[str, List[float]] = {}

    # Match equity records with regime data
    for i, record in enumerate(equity_progression):
        regime_name = "unknown"
        confidence = 0.0

        # Find corresponding regime data (by timestamp or index)
        if i < len(regime_data):
            regime_info = regime_data[i]
            regime_name = regime_info.get("regime_name", "unknown")
            confidence = regime_info.get("confidence_score", 0.0)

        # Add regime info to record
        record["regime_name"] = regime_name
        record["confidence_score"] = confidence

        # Group by regime
        if regime_name not in regime_groups:
            regime_groups[regime_name] = []
            regime_confidences[regime_name] = []

        regime_groups[regime_name].append(record)
        regime_confidences[regime_name].append(confidence)

    # Compute per-regime metrics
    per_regime_metrics = {}
    for regime_name, records in regime_groups.items():
        if records:  # Only compute if we have records
            regime_metrics = compute_backtest_metrics(records)
            regime_metrics["avg_confidence"] = sum(regime_confidences[regime_name]) / len(regime_confidences[regime_name])
            regime_metrics["trade_count"] = len(records)
            per_regime_metrics[regime_name] = regime_metrics

    # Create regime summary
    regime_summary = {
        "total_regimes": len(per_regime_metrics),
        "regime_distribution": {regime: len(records) for regime, records in regime_groups.items()},
        "best_performing_regime": max(per_regime_metrics.items(),
                                    key=lambda x: x[1].get("total_return", 0))[0] if per_regime_metrics else None,
        "worst_performing_regime": min(per_regime_metrics.items(),
                                     key=lambda x: x[1].get("total_return", 0))[0] if per_regime_metrics else None,
    }

    # Calculate regime performance comparison
    if per_regime_metrics:
        returns = [metrics.get("total_return", 0) for metrics in per_regime_metrics.values()]
        regime_summary["regime_return_range"] = max(returns) - min(returns)
        regime_summary["regime_return_std"] = stdev(returns) if len(returns) > 1 else 0.0

    return {
        "overall": overall_metrics,
        "per_regime": per_regime_metrics,
        "regime_summary": regime_summary
    }


def export_regime_aware_report(
    metrics: Dict[str, Any], out_path: str = "results/regime_aware_report.json"
) -> str:
    """
    Export comprehensive regime-aware backtest report.

    Args:
        metrics: Metrics dict from compute_regime_aware_metrics
        out_path: Destination path for the report

    Returns:
        Path to the written report file.
    """
    _ensure_results_dir(out_path)

    # Create comprehensive report
    report = {
        "report_type": "regime_aware_backtest_report",
        "timestamp": str(pd.Timestamp.now()),
        "summary": {
            "total_trades": metrics.get("overall", {}).get("total_trades", 0),
            "total_regimes": metrics.get("regime_summary", {}).get("total_regimes", 0),
            "best_regime": metrics.get("regime_summary", {}).get("best_performing_regime"),
            "worst_regime": metrics.get("regime_summary", {}).get("worst_performing_regime"),
        },
        "overall_performance": metrics.get("overall", {}),
        "regime_performance": metrics.get("per_regime", {}),
        "regime_analysis": metrics.get("regime_summary", {}),
        "recommendations": _generate_regime_recommendations(metrics)
    }

    # Save JSON report
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # Create CSV summary
    csv_path = os.path.splitext(out_path)[0] + "_summary.csv"
    _export_regime_csv_summary(metrics, csv_path)

    return out_path


def _generate_regime_recommendations(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Generate trading recommendations based on regime performance."""
    per_regime = metrics.get("per_regime", {})
    if not per_regime:
        return {"general": "Insufficient regime data for recommendations"}

    recommendations = {}

    # Find best and worst performing regimes
    regime_returns = {regime: data.get("total_return", 0)
                     for regime, data in per_regime.items()}

    best_regime = max(regime_returns.items(), key=lambda x: x[1])
    worst_regime = min(regime_returns.items(), key=lambda x: x[1])

    recommendations["best_regime"] = {
        "regime": best_regime[0],
        "return": best_regime[1],
        "recommendation": f"Strategy performs best in {best_regime[0]} conditions"
    }

    recommendations["worst_regime"] = {
        "regime": worst_regime[0],
        "return": worst_regime[1],
        "recommendation": f"Avoid or adjust strategy in {worst_regime[0]} conditions"
    }

    # Risk analysis
    regime_sharpes = {regime: data.get("sharpe_ratio", 0)
                     for regime, data in per_regime.items()}
    most_volatile = min(regime_sharpes.items(), key=lambda x: x[1])

    recommendations["risk_analysis"] = {
        "most_volatile_regime": most_volatile[0],
        "sharpe_ratio": most_volatile[1],
        "recommendation": f"Exercise caution in {most_volatile[0]} regime due to higher volatility"
    }

    return recommendations


def _export_regime_csv_summary(metrics: Dict[str, Any], csv_path: str) -> None:
    """Export regime metrics to CSV format."""
    per_regime = metrics.get("per_regime", {})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["Regime", "Total Return", "Sharpe Ratio", "Win Rate",
                        "Max Drawdown", "Total Trades", "Avg Confidence"])

        # Overall row
        overall = metrics.get("overall", {})
        writer.writerow([
            "OVERALL",
            f"{overall.get('total_return', 0):.4f}",
            f"{overall.get('sharpe_ratio', 0):.4f}",
            f"{overall.get('win_rate', 0):.4f}",
            f"{overall.get('max_drawdown', 0):.4f}",
            overall.get('total_trades', 0),
            "N/A"
        ])

        # Per-regime rows
        for regime_name, regime_data in per_regime.items():
            writer.writerow([
                regime_name.upper(),
                f"{regime_data.get('total_return', 0):.4f}",
                f"{regime_data.get('sharpe_ratio', 0):.4f}",
                f"{regime_data.get('win_rate', 0):.4f}",
                f"{regime_data.get('max_drawdown', 0):.4f}",
                regime_data.get('total_trades', 0),
                f"{regime_data.get('avg_confidence', 0):.4f}"
            ])


def export_regime_aware_equity_from_botengine(
    bot_engine: Any,
    regime_detector: Any,
    data: pd.DataFrame,
    out_path: str = "results/regime_aware_equity_curve.csv"
) -> str:
    """
    Enhanced version that includes regime information in equity progression.

    Args:
        bot_engine: BotEngine-like object
        regime_detector: Regime detector instance
        data: Historical data for regime detection
        out_path: destination CSV path

    Returns:
        The path to the written CSV file.
    """
    try:
        equity_progression = bot_engine.performance_stats.get("equity_progression", [])
    except Exception:
        equity_progression = getattr(bot_engine, "equity_progression", []) or []

    if not equity_progression:
        return ""

    # Detect regimes for each data point
    regime_data = []
    for i, row in data.iterrows():
        try:
            # Create a small window of data for regime detection
            window_data = data.loc[:i].tail(50)  # Last 50 periods
            if len(window_data) >= 20:  # Minimum required for regime detection
                regime_result = regime_detector.detect_enhanced_regime(window_data)
                regime_data.append({
                    "regime_name": regime_result.regime_name,
                    "confidence_score": regime_result.confidence_score,
                    "regime_features": regime_result.reasons
                })
            else:
                regime_data.append({
                    "regime_name": "insufficient_data",
                    "confidence_score": 0.0,
                    "regime_features": {}
                })
        except Exception as e:
            regime_data.append({
                "regime_name": "error",
                "confidence_score": 0.0,
                "regime_features": {"error": str(e)}
            })

    # Add regime information to equity progression
    for i, record in enumerate(equity_progression):
        if i < len(regime_data):
            record.update(regime_data[i])

    # Export regime-aware equity progression
    equity_csv = export_regime_aware_equity_progression(equity_progression, out_path=out_path)

    # Compute regime-aware metrics
    regime_metrics = compute_regime_aware_metrics(equity_progression, regime_data)

    # Export comprehensive report
    report_path = os.path.join(
        os.path.dirname(out_path) or "results", "regime_aware_report.json"
    )
    export_regime_aware_report(regime_metrics, out_path=report_path)

    return equity_csv
