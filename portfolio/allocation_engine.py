"""
Allocation Engine for dynamic capital allocation across strategies.

Provides sophisticated weighting algorithms based on performance metrics,
risk-adjusted returns, and portfolio optimization principles.
"""

import logging
import statistics
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)


class AllocationEngine:
    """
    Engine for calculating optimal capital allocations across trading strategies.

    Supports multiple allocation methods:
    - Sharpe-weighted: Based on risk-adjusted returns
    - Sortino-weighted: Based on downside risk-adjusted returns
    - Kelly-weighted: Based on Kelly criterion
    - Equal-weighted: Simple equal allocation
    - Custom-weighted: User-defined weights
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the allocation engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_trade_logger()

        # Configuration with defaults
        self.min_weight = self.config.get("min_weight", 0.05)  # 5%
        self.max_weight = self.config.get("max_weight", 0.4)  # 40%
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)  # 2%
        self.performance_window_days = self.config.get("performance_window_days", 30)
        self.rebalance_threshold = self.config.get("rebalance_threshold", 0.05)  # 5%

        # Risk parameters
        self.max_correlation = self.config.get("max_correlation", 0.7)
        self.target_volatility = self.config.get("target_volatility", 0.15)  # 15%

        logger.info("AllocationEngine initialized")

    def calculate_allocations(
        self,
        strategy_performance: Dict[str, List[Dict[str, Any]]],
        allocation_method: str = "sharpe_weighted",
        total_capital: Decimal = Decimal("10000"),
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate capital allocations for strategies.

        Args:
            strategy_performance: Performance history for each strategy
            allocation_method: Method to use for allocation calculation
            total_capital: Total capital to allocate

        Returns:
            Dictionary mapping strategy_id to allocation details
        """
        if not strategy_performance:
            logger.warning("No strategy performance data provided")
            return {}

        try:
            if allocation_method == "sharpe_weighted":
                weights = self._calculate_sharpe_weights(strategy_performance)
            elif allocation_method == "sortino_weighted":
                weights = self._calculate_sortino_weights(strategy_performance)
            elif allocation_method == "kelly_weighted":
                weights = self._calculate_kelly_weights(strategy_performance)
            elif allocation_method == "equal_weighted":
                weights = self._calculate_equal_weights(strategy_performance)
            elif allocation_method == "volatility_targeted":
                weights = self._calculate_volatility_targeted_weights(
                    strategy_performance
                )
            else:
                logger.warning(f"Unknown allocation method: {allocation_method}")
                weights = self._calculate_equal_weights(strategy_performance)

            # Apply constraints and normalize
            constrained_weights = self._apply_constraints(weights)

            # Calculate capital allocations
            allocations = {}
            for strategy_id, weight in constrained_weights.items():
                capital_allocated = total_capital * Decimal(str(weight))
                allocations[strategy_id] = {
                    "weight": weight,
                    "capital_allocated": capital_allocated,
                    "method": allocation_method,
                    "calculated_at": datetime.now(),
                }

            logger.info(
                f"Calculated allocations for {len(allocations)} strategies using {allocation_method}"
            )
            return allocations

        except Exception as e:
            logger.exception(f"Error calculating allocations: {e}")
            # Fallback to equal weights
            return self._calculate_fallback_allocations(
                strategy_performance, total_capital
            )

    def _calculate_sharpe_weights(
        self, strategy_performance: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Calculate weights based on Sharpe ratios."""
        weights = {}
        total_score = 0.0

        for strategy_id, performance_history in strategy_performance.items():
            sharpe_ratio = self._calculate_sharpe_ratio(performance_history)
            if sharpe_ratio is not None:
                # Convert Sharpe ratio to weight (higher Sharpe = higher weight)
                # Normalize around risk-free rate
                score = max(0.0, sharpe_ratio - self.risk_free_rate)
                weights[strategy_id] = score
                total_score += score
            else:
                weights[
                    strategy_id
                ] = 0.1  # Minimum weight for strategies with insufficient data
                total_score += 0.1

        # Normalize weights
        if total_score > 0:
            for strategy_id in weights:
                weights[strategy_id] /= total_score

        return weights

    def _calculate_sortino_weights(
        self, strategy_performance: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Calculate weights based on Sortino ratios."""
        weights = {}
        total_score = 0.0

        for strategy_id, performance_history in strategy_performance.items():
            sortino_ratio = self._calculate_sortino_ratio(performance_history)
            if sortino_ratio is not None:
                # Sortino ratio focuses on downside risk
                score = max(0.0, sortino_ratio)
                weights[strategy_id] = score
                total_score += score
            else:
                weights[strategy_id] = 0.1
                total_score += 0.1

        # Normalize weights
        if total_score > 0:
            for strategy_id in weights:
                weights[strategy_id] /= total_score

        return weights

    def _calculate_kelly_weights(
        self, strategy_performance: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Calculate weights based on Kelly criterion."""
        weights = {}

        for strategy_id, performance_history in strategy_performance.items():
            kelly_fraction = self._calculate_kelly_fraction(performance_history)
            if kelly_fraction is not None:
                # Kelly fraction gives optimal bet size
                # Scale it down for conservatism (half-Kelly)
                weights[strategy_id] = kelly_fraction * 0.5
            else:
                weights[strategy_id] = 0.1

        return weights

    def _calculate_equal_weights(
        self, strategy_performance: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Calculate equal weights for all strategies."""
        if not strategy_performance:
            return {}

        equal_weight = 1.0 / len(strategy_performance)
        return {
            strategy_id: equal_weight for strategy_id in strategy_performance.keys()
        }

    def _calculate_volatility_targeted_weights(
        self, strategy_performance: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Calculate weights to achieve target portfolio volatility."""
        try:
            # Extract returns for each strategy
            strategy_returns = {}
            for strategy_id, performance_history in strategy_performance.items():
                returns = [
                    p.get("daily_return", 0.0)
                    for p in performance_history
                    if "daily_return" in p
                ]
                if len(returns) >= 5:
                    strategy_returns[strategy_id] = returns

            if not strategy_returns:
                return self._calculate_equal_weights(strategy_performance)

            # Calculate volatilities
            volatilities = {}
            for strategy_id, returns in strategy_returns.items():
                if len(returns) > 1:
                    volatilities[strategy_id] = statistics.stdev(returns)

            # Calculate weights inversely proportional to volatility
            weights = {}
            total_inverse_vol = 0.0

            for strategy_id, vol in volatilities.items():
                if vol > 0:
                    inverse_vol = 1.0 / vol
                    weights[strategy_id] = inverse_vol
                    total_inverse_vol += inverse_vol

            # Normalize weights
            if total_inverse_vol > 0:
                for strategy_id in weights:
                    weights[strategy_id] /= total_inverse_vol

            # Fill in missing strategies with minimum weight
            for strategy_id in strategy_performance.keys():
                if strategy_id not in weights:
                    weights[strategy_id] = 0.05

            return weights

        except Exception as e:
            logger.exception(f"Error in volatility targeting: {e}")
            return self._calculate_equal_weights(strategy_performance)

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max weight constraints."""
        constrained_weights = {}

        # Apply min/max constraints
        for strategy_id, weight in weights.items():
            constrained_weights[strategy_id] = max(
                self.min_weight, min(self.max_weight, weight)
            )

        # Renormalize to ensure sum = 1.0
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            for strategy_id in constrained_weights:
                constrained_weights[strategy_id] /= total_weight

        return constrained_weights

    def _calculate_fallback_allocations(
        self,
        strategy_performance: Dict[str, List[Dict[str, Any]]],
        total_capital: Decimal,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate fallback allocations when main calculation fails."""
        equal_weight = 1.0 / len(strategy_performance) if strategy_performance else 0.0

        allocations = {}
        for strategy_id in strategy_performance.keys():
            allocations[strategy_id] = {
                "weight": equal_weight,
                "capital_allocated": total_capital * Decimal(str(equal_weight)),
                "method": "fallback_equal",
                "calculated_at": datetime.now(),
            }

        return allocations

    def _calculate_sharpe_ratio(
        self, performance_history: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Calculate Sharpe ratio from performance history."""
        if len(performance_history) < 5:
            return None

        try:
            returns = [
                p.get("daily_return", 0.0)
                for p in performance_history
                if "daily_return" in p
            ]

            if len(returns) < 2:
                return None

            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)

            if std_return == 0:
                return None

            # Annualized Sharpe ratio (assuming daily returns)
            sharpe_ratio = (
                (avg_return - self.risk_free_rate) / std_return * (252**0.5)
            )
            return sharpe_ratio

        except Exception as e:
            logger.warning(f"Error calculating Sharpe ratio: {e}")
            return None

    def _calculate_sortino_ratio(
        self, performance_history: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Calculate Sortino ratio from performance history."""
        if len(performance_history) < 5:
            return None

        try:
            returns = [
                p.get("daily_return", 0.0)
                for p in performance_history
                if "daily_return" in p
            ]

            if len(returns) < 2:
                return None

            # Calculate downside deviation (only negative returns)
            negative_returns = [r for r in returns if r < 0]
            if not negative_returns:
                return None

            avg_return = statistics.mean(returns)
            downside_std = statistics.stdev(negative_returns)

            if downside_std == 0:
                return None

            # Annualized Sortino ratio
            sortino_ratio = (
                (avg_return - self.risk_free_rate) / downside_std * (252**0.5)
            )
            return sortino_ratio

        except Exception as e:
            logger.warning(f"Error calculating Sortino ratio: {e}")
            return None

    def _calculate_kelly_fraction(
        self, performance_history: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Calculate Kelly fraction from performance history."""
        if len(performance_history) < 10:
            return None

        try:
            # Extract win/loss data
            wins = []
            losses = []

            for record in performance_history:
                pnl = record.get("pnl", 0)
                if pnl > 0:
                    wins.append(pnl)
                elif pnl < 0:
                    losses.append(abs(pnl))

            if not wins or not losses:
                return None

            # Calculate win/loss statistics
            win_rate = len(wins) / len(performance_history)
            avg_win = statistics.mean(wins)
            avg_loss = statistics.mean(losses)

            if avg_loss == 0:
                return None

            # Kelly fraction: (bp - q) / b
            # where b = odds (avg_win/avg_loss), p = win_rate, q = loss_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p

            kelly_fraction = (b * p - q) / b

            # Ensure non-negative
            return max(0.0, kelly_fraction)

        except Exception as e:
            logger.warning(f"Error calculating Kelly fraction: {e}")
            return None

    def should_rebalance(
        self,
        current_allocations: Dict[str, Dict[str, Any]],
        new_allocations: Dict[str, Dict[str, Any]],
    ) -> bool:
        """
        Determine if rebalancing is needed based on allocation changes.

        Args:
            current_allocations: Current strategy allocations
            new_allocations: Proposed new allocations

        Returns:
            True if rebalancing is recommended
        """
        if not current_allocations or not new_allocations:
            return False

        total_deviation = 0.0

        for strategy_id in set(current_allocations.keys()) | set(
            new_allocations.keys()
        ):
            current_weight = current_allocations.get(strategy_id, {}).get("weight", 0.0)
            new_weight = new_allocations.get(strategy_id, {}).get("weight", 0.0)

            deviation = abs(current_weight - new_weight)
            total_deviation += deviation

        # Check if total deviation exceeds threshold
        return total_deviation > self.rebalance_threshold

    def calculate_correlation_matrix(
        self, strategy_performance: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[np.ndarray]:
        """
        Calculate correlation matrix between strategies.

        Args:
            strategy_performance: Performance history for each strategy

        Returns:
            Correlation matrix as numpy array, or None if calculation fails
        """
        try:
            # Extract returns for each strategy
            strategy_returns = {}
            for strategy_id, performance_history in strategy_performance.items():
                returns = [
                    p.get("daily_return", 0.0)
                    for p in performance_history
                    if "daily_return" in p
                ]
                if len(returns) >= 5:
                    strategy_returns[strategy_id] = returns

            if len(strategy_returns) < 2:
                return None

            # Create returns matrix
            strategy_ids = list(strategy_returns.keys())
            returns_matrix = np.array([strategy_returns[sid] for sid in strategy_ids])

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(returns_matrix)

            return correlation_matrix

        except Exception as e:
            logger.exception(f"Error calculating correlation matrix: {e}")
            return None

    def optimize_for_risk_parity(
        self,
        strategy_performance: Dict[str, List[Dict[str, Any]]],
        target_volatility: float = None,
    ) -> Dict[str, float]:
        """
        Optimize allocations for risk parity (equal risk contribution).

        Args:
            strategy_performance: Performance history for each strategy
            target_volatility: Target portfolio volatility

        Returns:
            Optimized weights dictionary
        """
        if target_volatility is None:
            target_volatility = self.target_volatility

        try:
            # Calculate volatilities
            volatilities = {}
            for strategy_id, performance_history in strategy_performance.items():
                returns = [
                    p.get("daily_return", 0.0)
                    for p in performance_history
                    if "daily_return" in p
                ]
                if len(returns) >= 5:
                    volatilities[strategy_id] = statistics.stdev(returns)

            if not volatilities:
                return self._calculate_equal_weights(strategy_performance)

            # Risk parity: weights inversely proportional to volatility
            weights = {}
            total_inverse_vol = 0.0

            for strategy_id, vol in volatilities.items():
                if vol > 0:
                    inverse_vol = 1.0 / vol
                    weights[strategy_id] = inverse_vol
                    total_inverse_vol += inverse_vol

            # Normalize
            if total_inverse_vol > 0:
                for strategy_id in weights:
                    weights[strategy_id] /= total_inverse_vol

            return weights

        except Exception as e:
            logger.exception(f"Error in risk parity optimization: {e}")
            return self._calculate_equal_weights(strategy_performance)


# Factory function for creating allocation engines
def create_allocation_engine(
    config: Optional[Dict[str, Any]] = None
) -> AllocationEngine:
    """Create an allocation engine instance."""
    return AllocationEngine(config)
