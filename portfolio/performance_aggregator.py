"""
Performance Aggregator for multi-strategy portfolio analysis.

Aggregates performance metrics across strategies, calculates portfolio-level statistics,
and provides comprehensive reporting capabilities.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
import statistics
import json
import os

# Import metrics functions or implement locally
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio from returns."""
    if len(returns) < 2:
        return 0.0

    try:
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        if std_return == 0:
            return 0.0

        # Annualized Sharpe ratio (assuming daily returns)
        sharpe_ratio = (avg_return - risk_free_rate) / std_return * (252 ** 0.5)
        return sharpe_ratio
    except:
        return 0.0


def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Calculate Sortino ratio from returns."""
    if len(returns) < 2:
        return 0.0

    try:
        # Calculate downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return 0.0

        avg_return = statistics.mean(returns)
        downside_std = statistics.stdev(negative_returns)

        if downside_std == 0:
            return 0.0

        # Annualized Sortino ratio
        sortino_ratio = (avg_return - risk_free_rate) / downside_std * (252 ** 0.5)
        return sortino_ratio
    except:
        return 0.0


def calculate_max_drawdown(returns):
    """Calculate maximum drawdown from returns."""
    if len(returns) < 2:
        return 0.0

    try:
        cumulative = [1.0]
        for r in returns:
            cumulative.append(cumulative[-1] * (1 + r))

        running_max = [cumulative[0]]
        for i in range(1, len(cumulative)):
            running_max.append(max(running_max[-1], cumulative[i]))

        drawdowns = []
        for i in range(len(cumulative)):
            drawdown = (cumulative[i] - running_max[i]) / running_max[i]
            drawdowns.append(drawdown)

        max_drawdown = min(drawdowns)
        return abs(max_drawdown)
    except:
        return 0.0


def calculate_calmar_ratio(returns, risk_free_rate=0.02):
    """Calculate Calmar ratio from returns."""
    if len(returns) < 2:
        return 0.0

    try:
        max_dd = calculate_max_drawdown(returns)
        if max_dd == 0:
            return 0.0

        total_return = sum(returns)
        annualized_return = total_return / len(returns) * 252  # Assuming daily returns

        calmar_ratio = annualized_return / max_dd
        return calmar_ratio
    except:
        return 0.0


class PerformanceAggregator:
    """
    Aggregates and analyzes performance across multiple trading strategies.

    Provides portfolio-level metrics, contribution analysis, and comprehensive reporting.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance aggregator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_trade_logger()

        # Configuration with defaults
        self.output_dir = self.config.get('output_dir', 'reports/portfolio')
        self.save_interval_hours = self.config.get('save_interval_hours', 24)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.benchmark_symbol = self.config.get('benchmark_symbol', 'SPY')

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Performance storage
        self.portfolio_history: List[Dict[str, Any]] = []
        self.strategy_contributions: Dict[str, Dict[str, Any]] = {}
        self.correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None

        # Risk metrics
        self.portfolio_volatility = 0.0
        self.value_at_risk = 0.0
        self.expected_shortfall = 0.0

        logger.info("PerformanceAggregator initialized")

    def aggregate_performance(self, strategy_performance: Dict[str, List[Dict[str, Any]]],
                            allocations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate performance across all strategies.

        Args:
            strategy_performance: Performance data for each strategy
            allocations: Current capital allocations

        Returns:
            Aggregated portfolio performance metrics
        """
        if not strategy_performance:
            return self._get_empty_portfolio_performance()

        try:
            # Calculate individual strategy metrics
            strategy_metrics = {}
            for strategy_id, performance_history in strategy_performance.items():
                strategy_metrics[strategy_id] = self._calculate_strategy_metrics(
                    strategy_id, performance_history, allocations.get(strategy_id, {})
                )

            # Calculate portfolio-level metrics
            portfolio_metrics = self._calculate_portfolio_metrics(strategy_metrics, allocations)

            # Calculate contribution analysis
            contribution_analysis = self._calculate_contribution_analysis(strategy_metrics, allocations)

            # Update correlation matrix
            self._update_correlation_matrix(strategy_performance)

            # Store portfolio history
            self._store_portfolio_snapshot(portfolio_metrics, strategy_metrics, allocations)

            # Compile final result
            result = {
                'portfolio_metrics': portfolio_metrics,
                'strategy_metrics': strategy_metrics,
                'contribution_analysis': contribution_analysis,
                'allocations': allocations,
                'correlation_matrix': self.correlation_matrix,
                'risk_metrics': self._calculate_risk_metrics(strategy_performance),
                'timestamp': datetime.now()
            }

            logger.info(f"Aggregated performance for {len(strategy_performance)} strategies")
            return result

        except Exception as e:
            logger.exception(f"Error aggregating performance: {e}")
            return self._get_empty_portfolio_performance()

    def _calculate_strategy_metrics(self, strategy_id: str,
                                  performance_history: List[Dict[str, Any]],
                                  allocation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a single strategy."""
        if not performance_history:
            return self._get_empty_strategy_metrics(strategy_id)

        try:
            # Extract key metrics
            returns = [p.get('daily_return', 0.0) for p in performance_history if 'daily_return' in p]
            pnl_values = [p.get('pnl', 0.0) for p in performance_history if 'pnl' in p]
            trade_counts = [p.get('trades', 0) for p in performance_history if 'trades' in p]

            if not returns:
                return self._get_empty_strategy_metrics(strategy_id)

            # Basic metrics
            total_return = sum(returns)
            total_pnl = sum(pnl_values)
            total_trades = sum(trade_counts)

            # Risk metrics
            if len(returns) > 1:
                volatility = statistics.stdev(returns)
                sharpe_ratio = calculate_sharpe_ratio(returns, self.risk_free_rate)
                sortino_ratio = calculate_sortino_ratio(returns, self.risk_free_rate)
                max_dd = calculate_max_drawdown(returns)
                calmar_ratio = calculate_calmar_ratio(returns, self.risk_free_rate)
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
                max_dd = 0.0
                calmar_ratio = 0.0

            # Win rate
            winning_trades = sum(1 for pnl in pnl_values if pnl > 0)
            win_rate = winning_trades / len(pnl_values) if pnl_values else 0.0

            # Profit factor
            gross_profit = sum(pnl for pnl in pnl_values if pnl > 0)
            gross_loss = abs(sum(pnl for pnl in pnl_values if pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            return {
                'strategy_id': strategy_id,
                'total_return': total_return,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_dd,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade_pnl': total_pnl / total_trades if total_trades > 0 else 0.0,
                'allocation_weight': allocation.get('weight', 0.0),
                'capital_allocated': float(allocation.get('capital_allocated', 0.0)),
                'performance_score': allocation.get('performance_score', 0.5),
                'data_points': len(performance_history)
            }

        except Exception as e:
            logger.exception(f"Error calculating metrics for strategy {strategy_id}: {e}")
            return self._get_empty_strategy_metrics(strategy_id)

    def _calculate_portfolio_metrics(self, strategy_metrics: Dict[str, Dict[str, Any]],
                                   allocations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio-level aggregated metrics."""
        if not strategy_metrics:
            return self._get_empty_portfolio_metrics()

        try:
            # Weighted aggregation of returns
            total_weighted_return = 0.0
            total_weighted_volatility = 0.0
            total_pnl = 0.0
            total_trades = 0
            total_wins = 0

            for strategy_id, metrics in strategy_metrics.items():
                weight = allocations.get(strategy_id, {}).get('weight', 0.0)

                total_weighted_return += metrics['total_return'] * weight
                total_weighted_volatility += metrics['volatility'] * weight
                total_pnl += metrics['total_pnl']
                total_trades += metrics['total_trades']
                total_wins += metrics['win_rate'] * metrics['total_trades']

            # Portfolio-level ratios
            portfolio_sharpe = total_weighted_return / total_weighted_volatility if total_weighted_volatility > 0 else 0.0
            portfolio_win_rate = total_wins / total_trades if total_trades > 0 else 0.0

            # Calculate portfolio drawdown (simplified)
            portfolio_max_dd = max((metrics['max_drawdown'] for metrics in strategy_metrics.values()), default=0.0)

            return {
                'total_return': total_weighted_return,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'portfolio_volatility': total_weighted_volatility,
                'portfolio_sharpe_ratio': portfolio_sharpe,
                'portfolio_win_rate': portfolio_win_rate,
                'portfolio_max_drawdown': portfolio_max_dd,
                'active_strategies': len(strategy_metrics),
                'total_allocated_capital': sum(float(a.get('capital_allocated', 0.0)) for a in allocations.values()),
                'diversification_ratio': self._calculate_diversification_ratio(strategy_metrics)
            }

        except Exception as e:
            logger.exception(f"Error calculating portfolio metrics: {e}")
            return self._get_empty_portfolio_metrics()

    def _calculate_contribution_analysis(self, strategy_metrics: Dict[str, Dict[str, Any]],
                                       allocations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate contribution analysis for each strategy."""
        try:
            contributions = {}

            for strategy_id, metrics in strategy_metrics.items():
                weight = allocations.get(strategy_id, {}).get('weight', 0.0)

                contributions[strategy_id] = {
                    'pnl_contribution': metrics['total_pnl'] * weight,
                    'return_contribution': metrics['total_return'] * weight,
                    'volatility_contribution': metrics['volatility'] * weight,
                    'sharpe_contribution': metrics['sharpe_ratio'] * weight,
                    'weight': weight,
                    'relative_performance': self._calculate_relative_performance(strategy_id, strategy_metrics)
                }

            # Sort by contribution
            sorted_contributions = dict(sorted(
                contributions.items(),
                key=lambda x: x[1]['pnl_contribution'],
                reverse=True
            ))

            return {
                'strategy_contributions': sorted_contributions,
                'top_performer': max(contributions.items(), key=lambda x: x[1]['pnl_contribution'])[0],
                'worst_performer': min(contributions.items(), key=lambda x: x[1]['pnl_contribution'])[0],
                'contribution_diversity': self._calculate_contribution_diversity(contributions)
            }

        except Exception as e:
            logger.exception(f"Error calculating contribution analysis: {e}")
            return {'strategy_contributions': {}, 'error': str(e)}

    def _calculate_relative_performance(self, strategy_id: str,
                                      strategy_metrics: Dict[str, Dict[str, Any]]) -> float:
        """Calculate relative performance compared to portfolio average."""
        if strategy_id not in strategy_metrics:
            return 0.0

        strategy_sharpe = strategy_metrics[strategy_id]['sharpe_ratio']
        avg_sharpe = statistics.mean(m['sharpe_ratio'] for m in strategy_metrics.values())

        if avg_sharpe == 0:
            return 0.0

        return (strategy_sharpe - avg_sharpe) / abs(avg_sharpe)

    def _calculate_diversification_ratio(self, strategy_metrics: Dict[str, Dict[str, Any]]) -> float:
        """Calculate portfolio diversification ratio."""
        if not strategy_metrics:
            return 0.0

        try:
            # Diversification ratio = portfolio volatility / weighted average of individual volatilities
            individual_volatilities = [m['volatility'] for m in strategy_metrics.values()]
            avg_individual_vol = statistics.mean(individual_volatilities)

            # Simplified portfolio volatility (would need correlation in full implementation)
            portfolio_vol = statistics.mean(individual_volatilities)  # Placeholder

            if avg_individual_vol == 0:
                return 1.0

            return portfolio_vol / avg_individual_vol

        except Exception:
            return 1.0

    def _calculate_contribution_diversity(self, contributions: Dict[str, Dict[str, Any]]) -> float:
        """Calculate diversity of strategy contributions."""
        if not contributions:
            return 0.0

        try:
            pnl_contributions = [c['pnl_contribution'] for c in contributions.values()]

            if len(pnl_contributions) < 2:
                return 0.0

            # Coefficient of variation as diversity measure
            mean_contrib = statistics.mean(pnl_contributions)
            std_contrib = statistics.stdev(pnl_contributions)

            if mean_contrib == 0:
                return 0.0

            return std_contrib / abs(mean_contrib)

        except Exception:
            return 0.0

    def _update_correlation_matrix(self, strategy_performance: Dict[str, List[Dict[str, Any]]]) -> None:
        """Update correlation matrix between strategies."""
        try:
            if len(strategy_performance) < 2:
                self.correlation_matrix = None
                return

            # Extract returns for correlation calculation
            strategy_returns = {}
            for strategy_id, performance_history in strategy_performance.items():
                returns = [p.get('daily_return', 0.0) for p in performance_history if 'daily_return' in p]
                if len(returns) >= 5:
                    strategy_returns[strategy_id] = returns

            if len(strategy_returns) < 2:
                self.correlation_matrix = None
                return

            # Calculate correlations
            correlation_matrix = {}
            strategy_ids = list(strategy_returns.keys())

            for i, sid1 in enumerate(strategy_ids):
                correlation_matrix[sid1] = {}
                for j, sid2 in enumerate(strategy_ids):
                    if i == j:
                        correlation_matrix[sid1][sid2] = 1.0
                    else:
                        returns1 = strategy_returns[sid1]
                        returns2 = strategy_returns[sid2]

                        # Ensure same length
                        min_len = min(len(returns1), len(returns2))
                        returns1 = returns1[-min_len:]
                        returns2 = returns2[-min_len:]

                        if len(returns1) > 1:
                            correlation = statistics.correlation(returns1, returns2)
                            correlation_matrix[sid1][sid2] = correlation
                        else:
                            correlation_matrix[sid1][sid2] = 0.0

            self.correlation_matrix = correlation_matrix

        except Exception as e:
            logger.exception(f"Error updating correlation matrix: {e}")
            self.correlation_matrix = None

    def _calculate_risk_metrics(self, strategy_performance: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics."""
        try:
            # Extract all returns for VaR calculation
            all_returns = []
            for performance_history in strategy_performance.values():
                returns = [p.get('daily_return', 0.0) for p in performance_history if 'daily_return' in p]
                all_returns.extend(returns)

            if len(all_returns) < 10:
                return {
                    'portfolio_volatility': 0.0,
                    'value_at_risk_95': 0.0,
                    'expected_shortfall_95': 0.0
                }

            # Calculate volatility
            portfolio_volatility = statistics.stdev(all_returns)

            # Calculate VaR (95% confidence)
            sorted_returns = sorted(all_returns)
            var_index = int(len(sorted_returns) * 0.05)
            value_at_risk = abs(sorted_returns[var_index])

            # Calculate Expected Shortfall (CVaR)
            tail_returns = sorted_returns[:var_index + 1]
            expected_shortfall = abs(statistics.mean(tail_returns))

            return {
                'portfolio_volatility': portfolio_volatility,
                'value_at_risk_95': value_at_risk,
                'expected_shortfall_95': expected_shortfall
            }

        except Exception as e:
            logger.exception(f"Error calculating risk metrics: {e}")
            return {
                'portfolio_volatility': 0.0,
                'value_at_risk_95': 0.0,
                'expected_shortfall_95': 0.0
            }

    def _store_portfolio_snapshot(self, portfolio_metrics: Dict[str, Any],
                                strategy_metrics: Dict[str, Dict[str, Any]],
                                allocations: Dict[str, Dict[str, Any]]) -> None:
        """Store portfolio performance snapshot."""
        try:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_metrics': portfolio_metrics,
                'strategy_metrics': strategy_metrics,
                'allocations': allocations,
                'correlation_matrix': self.correlation_matrix
            }

            self.portfolio_history.append(snapshot)

            # Keep only recent history (last 1000 snapshots)
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]

        except Exception as e:
            logger.exception(f"Error storing portfolio snapshot: {e}")

    def save_performance_report(self, aggregated_performance: Dict[str, Any]) -> str:
        """Save comprehensive performance report to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_performance_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)

            # Add metadata
            report = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_type': 'portfolio_performance',
                    'strategies_count': len(aggregated_performance.get('strategy_metrics', {})),
                    'time_period': self.config.get('performance_window_days', 30)
                },
                'data': aggregated_performance
            }

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Performance report saved to {filepath}")
            return filepath

        except Exception as e:
            logger.exception(f"Error saving performance report: {e}")
            return ""

    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for the specified period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_history = [
                h for h in self.portfolio_history
                if datetime.fromisoformat(h['timestamp']) > cutoff_date
            ]

            if not recent_history:
                return {'error': 'No performance data available for the specified period'}

            # Calculate summary statistics
            pnl_values = [h['portfolio_metrics'].get('total_pnl', 0) for h in recent_history]
            return_values = [h['portfolio_metrics'].get('total_return', 0) for h in recent_history]

            summary = {
                'period_days': days,
                'data_points': len(recent_history),
                'total_pnl': sum(pnl_values),
                'avg_daily_return': statistics.mean(return_values) if return_values else 0.0,
                'volatility': statistics.stdev(return_values) if len(return_values) > 1 else 0.0,
                'max_pnl': max(pnl_values) if pnl_values else 0.0,
                'min_pnl': min(pnl_values) if pnl_values else 0.0,
                'best_day': max(return_values) if return_values else 0.0,
                'worst_day': min(return_values) if return_values else 0.0
            }

            return summary

        except Exception as e:
            logger.exception(f"Error generating performance summary: {e}")
            return {'error': str(e)}

    def _get_empty_portfolio_performance(self) -> Dict[str, Any]:
        """Get empty portfolio performance structure."""
        return {
            'portfolio_metrics': self._get_empty_portfolio_metrics(),
            'strategy_metrics': {},
            'contribution_analysis': {'strategy_contributions': {}},
            'allocations': {},
            'correlation_matrix': None,
            'risk_metrics': {
                'portfolio_volatility': 0.0,
                'value_at_risk_95': 0.0,
                'expected_shortfall_95': 0.0
            },
            'timestamp': datetime.now()
        }

    def _get_empty_portfolio_metrics(self) -> Dict[str, Any]:
        """Get empty portfolio metrics structure."""
        return {
            'total_return': 0.0,
            'total_pnl': 0.0,
            'total_trades': 0,
            'portfolio_volatility': 0.0,
            'portfolio_sharpe_ratio': 0.0,
            'portfolio_win_rate': 0.0,
            'portfolio_max_drawdown': 0.0,
            'active_strategies': 0,
            'total_allocated_capital': 0.0,
            'diversification_ratio': 0.0
        }

    def _get_empty_strategy_metrics(self, strategy_id: str) -> Dict[str, Any]:
        """Get empty strategy metrics structure."""
        return {
            'strategy_id': strategy_id,
            'total_return': 0.0,
            'total_pnl': 0.0,
            'total_trades': 0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_trade_pnl': 0.0,
            'allocation_weight': 0.0,
            'capital_allocated': 0.0,
            'performance_score': 0.5,
            'data_points': 0
        }


# Factory function for creating performance aggregators
def create_performance_aggregator(config: Optional[Dict[str, Any]] = None) -> PerformanceAggregator:
    """Create a performance aggregator instance."""
    return PerformanceAggregator(config)
