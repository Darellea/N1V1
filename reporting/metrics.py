"""
Metrics Engine

Automatic performance and risk reporting system for the trading framework.
Computes comprehensive risk-adjusted performance metrics and persists them
for monitoring and analysis.
"""

import logging
import json
import csv
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
import numpy as np
import pandas as pd

from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


@dataclass
class MetricsResult:
    """Comprehensive metrics result container."""
    timestamp: datetime
    strategy_id: str
    portfolio_id: str
    period_start: datetime
    period_end: datetime

    # Performance Metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk Metrics
    max_drawdown: float
    max_drawdown_duration: int  # days
    value_at_risk_95: float
    expected_shortfall_95: float
    beta: Optional[float]
    alpha: Optional[float]

    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Risk-Adjusted Metrics
    kelly_criterion: float
    ulcer_index: float
    downside_deviation: float

    # Metadata
    benchmark_return: Optional[float] = None
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        data['timestamp'] = self.timestamp.isoformat()
        data['period_start'] = self.period_start.isoformat()
        data['period_end'] = self.period_end.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricsResult':
        """Create instance from dictionary."""
        # Convert ISO strings back to datetime
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['period_start'] = datetime.fromisoformat(data['period_start'])
        data['period_end'] = datetime.fromisoformat(data['period_end'])
        return cls(**data)


class MetricsEngine:
    """
    Comprehensive metrics calculation engine for trading performance analysis.

    Computes risk-adjusted performance metrics, risk metrics, and trade statistics
    from trading logs, PnL series, or equity curves.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the metrics engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Configuration
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.annual_trading_days = self.config.get('annual_trading_days', 252)
        self.min_periods = self.config.get('min_periods', 30)

        # Output directory
        self.output_dir = self.config.get('output_dir', 'reports/metrics')
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info("MetricsEngine initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'confidence_level': 0.95,  # 95% confidence for VaR
            'annual_trading_days': 252,  # Trading days per year
            'min_periods': 30,  # Minimum periods for reliable calculations
            'output_dir': 'reports/metrics'
        }

    def calculate_metrics(
        self,
        returns: Union[List[float], np.ndarray, pd.Series],
        strategy_id: str = "default",
        portfolio_id: str = "main",
        benchmark_returns: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
        trade_log: Optional[List[Dict[str, Any]]] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> MetricsResult:
        """
        Calculate comprehensive performance and risk metrics.

        Args:
            returns: Array of periodic returns (daily or periodic)
            strategy_id: Identifier for the strategy
            portfolio_id: Identifier for the portfolio
            benchmark_returns: Benchmark returns for comparison
            trade_log: Detailed trade log for trade statistics
            period_start: Start of the measurement period
            period_end: End of the measurement period

        Returns:
            MetricsResult with comprehensive metrics
        """
        try:
            self.logger.info(f"Calculating metrics for strategy {strategy_id}")

            # Convert inputs to numpy arrays
            returns = self._prepare_returns(returns)
            benchmark_returns = self._prepare_returns(benchmark_returns) if benchmark_returns is not None else None

            # Set period boundaries
            if period_start is None:
                period_start = datetime.now() - timedelta(days=len(returns))
            if period_end is None:
                period_end = datetime.now()

            # Validate minimum data requirements
            if len(returns) < self.min_periods:
                self.logger.warning(f"Insufficient data: {len(returns)} < {self.min_periods} periods")
                return self._create_empty_result(strategy_id, portfolio_id, period_start, period_end)

            # Calculate performance metrics
            perf_metrics = self._calculate_performance_metrics(returns)

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(returns)

            # Calculate trade statistics if trade log provided
            trade_stats = self._calculate_trade_statistics(trade_log) if trade_log else self._get_empty_trade_stats()

            # Calculate benchmark-relative metrics
            benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_returns)

            # Combine all metrics
            result = MetricsResult(
                timestamp=datetime.now(),
                strategy_id=strategy_id,
                portfolio_id=strategy_id,  # Using strategy_id as portfolio_id for now
                period_start=period_start,
                period_end=period_end,
                risk_free_rate=self.risk_free_rate,
                confidence_level=self.confidence_level,
                benchmark_return=benchmark_metrics.get('benchmark_return'),
                **perf_metrics,
                **risk_metrics,
                **trade_stats
            )

            self.logger.info(f"Metrics calculated successfully for {strategy_id}")
            return result

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}", exc_info=True)
            return self._create_empty_result(strategy_id, portfolio_id, period_start or datetime.now(), period_end or datetime.now())

    def _prepare_returns(self, returns: Union[List[float], np.ndarray, pd.Series, None]) -> np.ndarray:
        """Prepare returns data for calculations."""
        if returns is None:
            return np.array([])

        if isinstance(returns, pd.Series):
            returns = returns.values
        elif isinstance(returns, list):
            returns = np.array(returns)

        # Remove NaN values
        returns = returns[~np.isnan(returns)]

        return returns

    def _calculate_performance_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate performance-related metrics."""
        # Total return
        total_return = float(np.prod(1 + returns) - 1) if len(returns) > 0 else 0.0

        # Annualized return
        periods_per_year = self.annual_trading_days
        annualized_return = float((1 + total_return) ** (periods_per_year / len(returns)) - 1) if len(returns) > 0 else 0.0

        # Volatility (annualized)
        volatility = float(np.std(returns) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0

        # Sharpe Ratio
        excess_returns = returns - self.risk_free_rate / periods_per_year
        sharpe_ratio = float(np.mean(excess_returns) / np.std(excess_returns)) if len(excess_returns) > 1 and np.std(excess_returns) > 0 else 0.0

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = float(np.std(downside_returns) * np.sqrt(periods_per_year)) if len(downside_returns) > 0 else 0.0
        sortino_ratio = float(np.mean(excess_returns) / downside_deviation) if downside_deviation > 0 else 0.0

        # Ulcer Index
        ulcer_index = self._calculate_ulcer_index(returns)

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'ulcer_index': ulcer_index,
            'downside_deviation': downside_deviation
        }

    def _calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk-related metrics."""
        # Max Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = float(-np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Max Drawdown Duration
        drawdown_mask = drawdowns < 0
        if np.any(drawdown_mask):
            drawdown_durations = []
            current_duration = 0
            for i in range(len(drawdown_mask)):
                if drawdown_mask[i]:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        drawdown_durations.append(current_duration)
                        current_duration = 0
            if current_duration > 0:
                drawdown_durations.append(current_duration)
            max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0
        else:
            max_drawdown_duration = 0

        # Calmar Ratio
        calmar_ratio = float(self._calculate_performance_metrics(returns)['annualized_return'] / max_drawdown) if max_drawdown > 0 else 0.0

        # Value at Risk (95%)
        value_at_risk_95 = float(np.percentile(returns, (1 - self.confidence_level) * 100))

        # Expected Shortfall (95%)
        tail_returns = returns[returns <= value_at_risk_95]
        expected_shortfall_95 = float(np.mean(tail_returns)) if len(tail_returns) > 0 else value_at_risk_95

        # Kelly Criterion
        kelly_criterion = self._calculate_kelly_criterion(returns)

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'calmar_ratio': calmar_ratio,
            'value_at_risk_95': value_at_risk_95,
            'expected_shortfall_95': expected_shortfall_95,
            'kelly_criterion': kelly_criterion
        }

    def _calculate_trade_statistics(self, trade_log: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trade-level statistics."""
        if not trade_log or len(trade_log) == 0:
            return self._get_empty_trade_stats()

        # Extract PnL from trades
        pnl_values = []
        for trade in trade_log:
            if 'pnl' in trade:
                pnl_values.append(float(trade['pnl']))
            elif 'profit' in trade:
                pnl_values.append(float(trade['profit']))

        if not pnl_values:
            return self._get_empty_trade_stats()

        pnl_array = np.array(pnl_values)

        # Basic trade counts
        total_trades = len(pnl_values)
        winning_trades = int(np.sum(pnl_array > 0))
        losing_trades = int(np.sum(pnl_array < 0))
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Profit metrics
        winning_pnl = pnl_array[pnl_array > 0]
        losing_pnl = pnl_array[pnl_array < 0]

        avg_win = float(np.mean(winning_pnl)) if len(winning_pnl) > 0 else 0.0
        avg_loss = float(np.mean(losing_pnl)) if len(losing_pnl) > 0 else 0.0
        largest_win = float(np.max(winning_pnl)) if len(winning_pnl) > 0 else 0.0
        largest_loss = float(np.min(losing_pnl)) if len(losing_pnl) > 0 else 0.0

        # Profit factor
        total_wins = np.sum(winning_pnl) if len(winning_pnl) > 0 else 0
        total_losses = abs(np.sum(losing_pnl)) if len(losing_pnl) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }

    def _calculate_benchmark_metrics(self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculate benchmark-relative metrics."""
        if benchmark_returns is None or len(benchmark_returns) != len(returns):
            return {'benchmark_return': None, 'beta': None, 'alpha': None}

        # Benchmark return
        benchmark_return = float(np.prod(1 + benchmark_returns) - 1)

        # Beta and Alpha (simplified CAPM)
        try:
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)

            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
                # Alpha (annualized excess return)
                strategy_return = self._calculate_performance_metrics(returns)['annualized_return']
                benchmark_annualized = (1 + benchmark_return) ** (self.annual_trading_days / len(benchmark_returns)) - 1
                alpha = strategy_return - (self.risk_free_rate + beta * (benchmark_annualized - self.risk_free_rate))
            else:
                beta = 0.0
                alpha = 0.0
        except:
            beta = 0.0
            alpha = 0.0

        return {
            'benchmark_return': benchmark_return,
            'beta': beta,
            'alpha': alpha
        }

    def _calculate_ulcer_index(self, returns: np.ndarray) -> float:
        """Calculate Ulcer Index."""
        if len(returns) < 2:
            return 0.0

        # Calculate drawdowns
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max

        # Ulcer Index is square root of mean of squared drawdowns
        ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
        return float(ulcer_index)

    def _calculate_kelly_criterion(self, returns: np.ndarray) -> float:
        """Calculate Kelly Criterion."""
        if len(returns) < 2:
            return 0.0

        # Simplified Kelly calculation
        win_rate = np.mean(returns > 0)
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 0

        if avg_loss == 0:
            return 0.0

        kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        return max(0.0, float(kelly))  # Kelly can be negative, but we cap at 0

    def _get_empty_trade_stats(self) -> Dict[str, float]:
        """Get empty trade statistics."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }

    def _create_empty_result(self, strategy_id: str, portfolio_id: str, period_start: datetime, period_end: datetime) -> MetricsResult:
        """Create empty metrics result for error cases."""
        return MetricsResult(
            timestamp=datetime.now(),
            strategy_id=strategy_id,
            portfolio_id=portfolio_id,
            period_start=period_start,
            period_end=period_end,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            value_at_risk_95=0.0,
            expected_shortfall_95=0.0,
            beta=None,
            alpha=None,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            kelly_criterion=0.0,
            ulcer_index=0.0,
            downside_deviation=0.0
        )

    def save_to_json(self, result: MetricsResult, filename: Optional[str] = None) -> str:
        """
        Save metrics result to JSON file.

        Args:
            result: MetricsResult to save
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{result.strategy_id}_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

            self.logger.info(f"Metrics saved to JSON: {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Failed to save metrics to JSON: {e}")
            raise

    def save_to_csv(self, result: MetricsResult, filename: Optional[str] = None) -> str:
        """
        Save metrics result to CSV file.

        Args:
            result: MetricsResult to save
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{result.strategy_id}_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)

        try:
            # Convert to flat dictionary for CSV
            data = result.to_dict()

            # Flatten nested structures if any
            flat_data = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_data[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_data[key] = value

            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=flat_data.keys())
                writer.writeheader()
                writer.writerow(flat_data)

            self.logger.info(f"Metrics saved to CSV: {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Failed to save metrics to CSV: {e}")
            raise

    def load_from_json(self, filepath: str) -> MetricsResult:
        """
        Load metrics result from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Loaded MetricsResult
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            result = MetricsResult.from_dict(data)
            self.logger.info(f"Metrics loaded from JSON: {filepath}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to load metrics from JSON: {e}")
            raise

    def generate_session_report(self, returns: Union[List[float], np.ndarray, pd.Series],
                              strategy_id: str, trade_log: Optional[List[Dict[str, Any]]] = None) -> MetricsResult:
        """
        Generate metrics report for a trading session.

        Args:
            returns: Session returns
            strategy_id: Strategy identifier
            trade_log: Optional trade log

        Returns:
            MetricsResult for the session
        """
        self.logger.info(f"Generating session report for {strategy_id}")

        result = self.calculate_metrics(
            returns=returns,
            strategy_id=strategy_id,
            trade_log=trade_log,
            period_start=datetime.now() - timedelta(hours=24),  # Assume daily session
            period_end=datetime.now()
        )

        # Save results
        try:
            json_path = self.save_to_json(result)
            csv_path = self.save_to_csv(result)

            # Log summary
            self._log_metrics_summary(result)

            # Publish event (would integrate with event bus)
            self._publish_metrics_event(result)

        except Exception as e:
            self.logger.error(f"Failed to save session report: {e}")

        return result

    def generate_periodic_report(self, returns: Union[List[float], np.ndarray, pd.Series],
                               strategy_id: str, period_days: int = 30,
                               trade_log: Optional[List[Dict[str, Any]]] = None) -> MetricsResult:
        """
        Generate metrics report for a periodic interval.

        Args:
            returns: Period returns
            strategy_id: Strategy identifier
            period_days: Number of days in the period
            trade_log: Optional trade log

        Returns:
            MetricsResult for the period
        """
        self.logger.info(f"Generating {period_days}-day report for {strategy_id}")

        result = self.calculate_metrics(
            returns=returns,
            strategy_id=strategy_id,
            trade_log=trade_log,
            period_start=datetime.now() - timedelta(days=period_days),
            period_end=datetime.now()
        )

        # Save results
        try:
            json_path = self.save_to_json(result, f"periodic_{period_days}d_{strategy_id}_{datetime.now().strftime('%Y%m%d')}.json")
            csv_path = self.save_to_csv(result, f"periodic_{period_days}d_{strategy_id}_{datetime.now().strftime('%Y%m%d')}.csv")

            # Log summary
            self._log_metrics_summary(result)

        except Exception as e:
            self.logger.error(f"Failed to save periodic report: {e}")

        return result

    def _log_metrics_summary(self, result: MetricsResult) -> None:
        """Log a summary of metrics."""
        self.logger.info(f"Metrics Summary for {result.strategy_id}:")
        self.logger.info(f"  Total Return: {result.total_return:.2%}")
        self.logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        self.logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
        self.logger.info(f"  Win Rate: {result.win_rate:.1%}")
        self.logger.info(f"  Total Trades: {result.total_trades}")

        # Log to trade logger
        trade_logger.performance("Metrics Calculated", {
            'strategy_id': result.strategy_id,
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades
        })

    def _publish_metrics_event(self, result: MetricsResult) -> None:
        """Publish metrics event to event bus."""
        # This would integrate with the event bus system
        event_data = {
            'event_type': 'METRICS_REPORTED',
            'strategy_id': result.strategy_id,
            'timestamp': result.timestamp.isoformat(),
            'metrics': result.to_dict()
        }

        self.logger.debug(f"Published metrics event: {event_data}")

    def get_metrics_history(self, strategy_id: str, limit: int = 10) -> List[MetricsResult]:
        """
        Get historical metrics for a strategy.

        Args:
            strategy_id: Strategy identifier
            limit: Maximum number of records to return

        Returns:
            List of historical MetricsResult objects
        """
        try:
            # Find all JSON files for this strategy
            pattern = f"metrics_{strategy_id}_*.json"
            files = [f for f in os.listdir(self.output_dir) if f.startswith(f"metrics_{strategy_id}_") and f.endswith('.json')]

            # Sort by timestamp (newest first)
            files.sort(reverse=True)

            results = []
            for filename in files[:limit]:
                try:
                    filepath = os.path.join(self.output_dir, filename)
                    result = self.load_from_json(filepath)
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to load metrics file {filename}: {e}")

            return results

        except Exception as e:
            self.logger.error(f"Failed to get metrics history: {e}")
            return []


# Global instance
_metrics_engine: Optional[MetricsEngine] = None


def get_metrics_engine() -> MetricsEngine:
    """Get the global metrics engine instance."""
    global _metrics_engine
    if _metrics_engine is None:
        from utils.config_loader import get_config
        config = get_config('metrics', {})
        _metrics_engine = MetricsEngine(config)
    return _metrics_engine


def calculate_session_metrics(returns: Union[List[float], np.ndarray, pd.Series],
                            strategy_id: str, trade_log: Optional[List[Dict[str, Any]]] = None) -> MetricsResult:
    """
    Convenience function to calculate session metrics.

    Args:
        returns: Session returns
        strategy_id: Strategy identifier
        trade_log: Optional trade log

    Returns:
        MetricsResult for the session
    """
    engine = get_metrics_engine()
    return engine.generate_session_report(returns, strategy_id, trade_log)
