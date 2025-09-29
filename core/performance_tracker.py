"""
PerformanceTracker - Performance metrics and equity tracking component.

Handles performance calculation, equity progression tracking,
and performance statistics management.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from utils.time import now_ms

from .interfaces import PerformanceTrackerInterface
from .logging_utils import LogSensitivity, get_structured_logger
from .utils.error_utils import ErrorHandler

logger = get_structured_logger("core.performance_tracker", LogSensitivity.SECURE)
error_handler = ErrorHandler("performance_tracker")


class PerformanceTracker(PerformanceTrackerInterface):
    """
    Tracks and calculates trading performance metrics.

    Responsibilities:
    - Performance statistics calculation
    - Equity progression tracking
    - Win/loss ratio calculation
    - Sharpe ratio and risk metrics
    - Performance reporting
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the PerformanceTracker.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Import configuration from centralized system
        from .config_manager import get_config_manager

        config_manager = get_config_manager()
        pt_config = config_manager.get_performance_tracker_config()

        # Starting balance from configuration (with fallback to config file)
        config_balance = float(
            self.config.get("trading", {}).get(
                "initial_balance", pt_config.starting_balance
            )
        )
        self.starting_balance: float = config_balance

        # Performance statistics
        self.performance_stats: Dict[str, Any] = {
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0,
            "equity_history": [],
            "returns_history": [],
            "equity_progression": [],
        }

        # Trading mode for context
        self.mode = config.get("environment", {}).get("mode", "paper")

    def update_performance_metrics(
        self, pnl: float, current_equity: Optional[float] = None
    ):
        """Update performance tracking metrics after a trade.

        Args:
            pnl: Profit/loss from the trade
            current_equity: Current equity value (optional)
        """
        try:
            # Update totals
            self.performance_stats["total_pnl"] += float(pnl)

            # Track returns for Sharpe ratio calculation
            if current_equity is not None:
                equity_history = self.performance_stats["equity_history"]
                returns_history = self.performance_stats["returns_history"]

                if equity_history:
                    prev_equity = equity_history[-1]
                    if prev_equity > 0:
                        daily_return = (current_equity - prev_equity) / prev_equity
                        returns_history.append(daily_return)

                equity_history.append(current_equity)

            # Calculate win/loss counts
            if pnl > 0:
                self.performance_stats["wins"] += 1
            elif pnl < 0:
                self.performance_stats["losses"] += 1

            total_trades = (
                self.performance_stats["wins"] + self.performance_stats["losses"]
            )
            if total_trades > 0:
                self.performance_stats["win_rate"] = (
                    self.performance_stats["wins"] / total_trades
                )

            # Calculate max drawdown
            if self.performance_stats["equity_history"]:
                equity_history = self.performance_stats["equity_history"]
                peak = max(equity_history)
                trough = min(equity_history)
                if peak > 0:
                    max_dd = (peak - trough) / peak
                    self.performance_stats["max_drawdown"] = max(
                        max_dd, float(self.performance_stats.get("max_drawdown", 0.0))
                    )

            # Calculate Sharpe ratio (annualized)
            returns_history = self.performance_stats["returns_history"]
            if len(returns_history) > 1:
                returns = np.array(returns_history)
                risk_free_rate = 0.0  # Can be configured
                excess_returns = returns - risk_free_rate
                std_returns = np.std(excess_returns)

                # Safe division: handle zero or very small standard deviation (constant returns)
                if std_returns > 0.001:  # Consider small std as constant
                    sharpe = float(np.mean(excess_returns) / std_returns)
                    self.performance_stats["sharpe_ratio"] = sharpe * np.sqrt(
                        252
                    )  # Annualize
                else:
                    # For constant returns, Sharpe ratio is undefined, use 0
                    self.performance_stats["sharpe_ratio"] = 0.0

        except Exception as e:
            logger.exception(f"Failed to update performance metrics: {e}")

    async def record_trade_equity(self, order_result: Dict[str, Any]) -> None:
        """
        Record equity progression after a trade execution.

        Args:
            order_result: Dictionary returned from OrderManager.execute_order containing at least:
              - id (optional): trade identifier
              - timestamp (optional): epoch ms or ISO timestamp
              - pnl (optional): profit/loss for the trade
        """
        try:
            if not order_result:
                return

            # Ensure equity_progression exists
            equity_prog: list[Dict[str, Any]] = self.performance_stats.setdefault(
                "equity_progression", []
            )

            # Get current equity - this will need to be passed in or retrieved from order manager
            # For now, we'll calculate it from total_pnl + starting_balance
            current_equity = self.starting_balance + self.performance_stats.get(
                "total_pnl", 0.0
            )

            # For backtest/paper modes, use the calculated equity
            if self.mode in ("backtest", "paper"):
                if current_equity == 0.0:
                    current_equity = float(self.starting_balance) + float(
                        self.performance_stats.get("total_pnl", 0.0)
                    )

            # Normalize values
            trade_id = order_result.get("id", f"trade_{now_ms()}")
            timestamp = order_result.get("timestamp", now_ms())
            pnl = order_result.get("pnl", None)

            # Calculate cumulative return relative to starting balance
            try:
                cumulative_return = 0.0
                if self.starting_balance and float(self.starting_balance) > 0:
                    cumulative_return = (
                        current_equity - float(self.starting_balance)
                    ) / float(self.starting_balance)
            except Exception:
                cumulative_return = 0.0

            # Normalize trade_id and timestamp.
            trade_id = order_result.get("id", f"trade_{now_ms()}")
            ts_raw = order_result.get("timestamp", now_ms())

            record: Dict[str, Any] = {
                "trade_id": trade_id,
                "timestamp": ts_raw,
                "symbol": order_result.get("symbol")
                if isinstance(order_result, dict)
                else None,
                "equity": current_equity,
                "pnl": pnl,
                "cumulative_return": cumulative_return,
            }

            equity_prog.append(record)

        except Exception as e:
            logger.exception(f"Failed to record trade equity: {e}")

    async def record_safe_mode_activation(self, details: Dict[str, Any]):
        """Record safe mode activation event."""
        try:
            # This could be extended to log to a separate audit trail
            logger.info(f"Safe mode activated with details: {details}")
        except Exception as e:
            logger.exception(f"Failed to record safe mode activation: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.performance_stats.copy()

    def get_equity_progression(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent equity progression records."""
        equity_prog = self.performance_stats.get("equity_progression", [])
        if limit is None or limit <= 0:
            return equity_prog[-limit:] if limit and limit < 0 else equity_prog
        return equity_prog[-limit:]

    def calculate_additional_metrics(self) -> Dict[str, Any]:
        """Calculate additional performance metrics."""
        try:
            equity_prog = self.performance_stats.get("equity_progression", [])
            if not equity_prog:
                return {"profit_factor": 0.0}

            # Calculate additional metrics
            equity_values = [
                record.get("equity", 0)
                for record in equity_prog
                if record.get("equity")
            ]
            pnl_values = [
                record.get("pnl", 0)
                for record in equity_prog
                if record.get("pnl") is not None
            ]

            additional_metrics = {}

            if equity_values:
                additional_metrics["current_equity"] = equity_values[-1]
                additional_metrics["peak_equity"] = max(equity_values)
                additional_metrics["lowest_equity"] = min(equity_values)

            if pnl_values:
                additional_metrics["avg_win"] = (
                    np.mean([p for p in pnl_values if p > 0])
                    if any(p > 0 for p in pnl_values)
                    else 0
                )
                additional_metrics["avg_loss"] = (
                    np.mean([p for p in pnl_values if p < 0])
                    if any(p < 0 for p in pnl_values)
                    else 0
                )
                additional_metrics["largest_win"] = max(pnl_values) if pnl_values else 0
                additional_metrics["largest_loss"] = (
                    min(pnl_values) if pnl_values else 0
                )

                # Profit factor - safe division
                total_wins = sum(p for p in pnl_values if p > 0)
                total_losses = abs(sum(p for p in pnl_values if p < 0))

                if total_losses > 0:
                    additional_metrics["profit_factor"] = total_wins / total_losses
                elif total_wins > 0:
                    # Wins but no losses - infinite profit factor
                    additional_metrics["profit_factor"] = float("inf")
                else:
                    # No wins and no losses - undefined, use 0
                    additional_metrics["profit_factor"] = 0.0
            else:
                # No pnl values, profit factor is 0
                additional_metrics["profit_factor"] = 0.0

            return additional_metrics

        except Exception as e:
            logger.exception(f"Failed to calculate additional metrics: {e}")
            return {"profit_factor": 0.0}

    def reset_performance(self):
        """Reset all performance tracking data."""
        self.performance_stats = {
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0,
            "equity_history": [],
            "returns_history": [],
            "equity_progression": [],
        }
        logger.info("Performance tracking reset")

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance summary report."""
        try:
            base_stats = self.get_performance_stats()
            additional_metrics = self.calculate_additional_metrics()

            report = {
                "performance_stats": base_stats,
                "additional_metrics": additional_metrics,
                "trading_mode": self.mode,
                "starting_balance": self.starting_balance,
                "total_return_pct": (
                    (base_stats.get("total_pnl", 0) / self.starting_balance * 100)
                    if self.starting_balance and self.starting_balance > 0
                    else 0.0
                ),
                "total_trades": base_stats.get("wins", 0) + base_stats.get("losses", 0),
                "report_generated_at": now_ms(),
            }

            return report

        except Exception as e:
            logger.exception(f"Failed to generate performance summary: {e}")
            return {"error": str(e)}
