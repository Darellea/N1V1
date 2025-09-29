"""
portfolio/allocator.py

Capital Allocation Strategies for Portfolio Management.

This module provides different strategies for allocating capital across assets:
- Equal Weight: Equal allocation to all assets
- Risk Parity: Allocation based on risk contribution
- Momentum Weighted: Higher allocation to stronger performing assets
"""

import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class CapitalAllocator(ABC):
    """
    Abstract base class for capital allocation strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize allocator.

        Args:
            config: Configuration dictionary for the allocator
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def allocate(
        self,
        assets: List[str],
        market_data: Optional[pd.DataFrame] = None,
        current_prices: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, float]:
        """
        Allocate capital across assets.

        Args:
            assets: List of asset symbols to allocate to
            market_data: Historical market data for allocation decisions
            current_prices: Current prices for all assets

        Returns:
            Dictionary mapping symbols to allocation percentages (0.0 to 1.0)
        """
        pass

    def validate_allocations(self, allocations: Dict[str, float]) -> bool:
        """
        Validate that allocations sum to 1.0 and are within valid ranges.

        Args:
            allocations: Allocation dictionary to validate

        Returns:
            True if allocations are valid
        """
        if not allocations:
            return False

        total_allocation = sum(allocations.values())

        # Check if allocations sum to approximately 1.0 (within tolerance)
        if not (0.99 <= total_allocation <= 1.01):
            self.logger.warning(
                f"Allocations sum to {total_allocation}, should sum to 1.0"
            )
            return False

        # Check individual allocations are within valid range
        for symbol, allocation in allocations.items():
            if not (0.0 <= allocation <= 1.0):
                self.logger.warning(
                    f"Allocation for {symbol} is {allocation}, should be between 0.0 and 1.0"
                )
                return False

        return True

    def normalize_allocations(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize allocations to ensure they sum to 1.0.

        Args:
            allocations: Allocation dictionary to normalize

        Returns:
            Normalized allocation dictionary
        """
        total = sum(allocations.values())
        if total == 0:
            return allocations

        normalized = {
            symbol: allocation / total for symbol, allocation in allocations.items()
        }
        return normalized


class EqualWeightAllocator(CapitalAllocator):
    """
    Equal Weight Allocation Strategy.

    Allocates capital equally across all selected assets.
    This is the simplest allocation strategy and provides diversification
    without considering asset-specific characteristics.
    """

    def allocate(
        self,
        assets: List[str],
        market_data: Optional[pd.DataFrame] = None,
        current_prices: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, float]:
        """
        Allocate capital equally across all assets.

        Args:
            assets: List of asset symbols to allocate to
            market_data: Not used in equal weight allocation
            current_prices: Not used in equal weight allocation

        Returns:
            Dictionary mapping symbols to equal allocation percentages
        """
        if not assets:
            return {}

        # Equal allocation to all assets
        equal_allocation = 1.0 / len(assets)
        allocations = {asset: equal_allocation for asset in assets}

        self.logger.info(f"Equal weight allocation: {allocations}")

        return allocations


class RiskParityAllocator(CapitalAllocator):
    """
    Risk Parity Allocation Strategy.

    Allocates capital based on risk contribution rather than capital.
    Assets with higher volatility receive smaller allocations to ensure
    each asset contributes equally to portfolio risk.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Risk Parity allocator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.lookback_period = self.config.get("lookback_period", 30)
        self.risk_measure = self.config.get(
            "risk_measure", "volatility"
        )  # 'volatility' or 'var'

    def allocate(
        self,
        assets: List[str],
        market_data: Optional[pd.DataFrame] = None,
        current_prices: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, float]:
        """
        Allocate capital using risk parity approach.

        Args:
            assets: List of asset symbols to allocate to
            market_data: Historical market data for risk calculation
            current_prices: Not used in risk parity allocation

        Returns:
            Dictionary mapping symbols to risk-parity allocation percentages
        """
        if not assets:
            return {}

        if market_data is None or market_data.empty:
            # Fallback to equal weight if no market data
            self.logger.warning(
                "No market data available, falling back to equal weight allocation"
            )
            return EqualWeightAllocator().allocate(assets)

        # Calculate risk measures for each asset
        risk_measures = {}
        for asset in assets:
            if asset in market_data.columns:
                try:
                    risk = self._calculate_risk_measure(market_data[asset])
                    risk_measures[asset] = risk
                except Exception as e:
                    self.logger.warning(
                        f"Could not calculate risk for {asset}: {str(e)}"
                    )
                    risk_measures[asset] = 1.0  # Default risk measure
            else:
                self.logger.warning(f"Asset {asset} not found in market data")
                risk_measures[asset] = 1.0

        # Risk parity allocation: inverse of risk
        total_inverse_risk = sum(
            1.0 / risk for risk in risk_measures.values() if risk > 0
        )

        if total_inverse_risk == 0:
            # Fallback to equal weight
            return EqualWeightAllocator().allocate(assets)

        allocations = {}
        for asset, risk in risk_measures.items():
            if risk > 0:
                allocation = (1.0 / risk) / total_inverse_risk
                allocations[asset] = allocation
            else:
                allocations[asset] = 0.0

        # Normalize to ensure sum = 1.0
        allocations = self.normalize_allocations(allocations)

        self.logger.info(f"Risk parity allocation: {allocations}")

        return allocations

    def _calculate_risk_measure(self, price_series: pd.Series) -> float:
        """
        Calculate risk measure for an asset.

        Args:
            price_series: Price series for the asset

        Returns:
            Risk measure value
        """
        if len(price_series) < self.lookback_period:
            return 1.0  # Default risk measure

        if self.risk_measure == "volatility":
            # Use annualized absolute volatility
            volatility = price_series.std() * np.sqrt(252)  # Assuming daily data
            return max(
                volatility, 0.001
            )  # Minimum volatility to avoid division by zero

        elif self.risk_measure == "var":
            # Use Value at Risk (95% confidence)
            returns = price_series.pct_change().dropna()
            if len(returns) < 10:
                return 1.0
            var = np.percentile(returns, 5)  # 5th percentile = 95% VaR
            return max(abs(var), 0.001)  # Use absolute value and minimum

        else:
            # Default to absolute volatility
            volatility = price_series.std() * np.sqrt(252)
            return max(volatility, 0.001)


class MomentumWeightAllocator(CapitalAllocator):
    """
    Momentum Weighted Allocation Strategy.

    Allocates more capital to assets with stronger recent performance
    (momentum). This follows the principle that assets that have performed
    well recently are likely to continue performing well.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Momentum Weighted allocator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.lookback_period = self.config.get("lookback_period", 30)
        self.momentum_type = self.config.get(
            "momentum_type", "returns"
        )  # 'returns', 'sharpe', 'vol_adjusted'

    def allocate(
        self,
        assets: List[str],
        market_data: Optional[pd.DataFrame] = None,
        current_prices: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, float]:
        """
        Allocate capital based on momentum scores.

        Args:
            assets: List of asset symbols to allocate to
            market_data: Historical market data for momentum calculation
            current_prices: Not used in momentum allocation

        Returns:
            Dictionary mapping symbols to momentum-weighted allocation percentages
        """
        if not assets:
            return {}

        if market_data is None or market_data.empty:
            # Fallback to equal weight if no market data
            self.logger.warning(
                "No market data available, falling back to equal weight allocation"
            )
            return EqualWeightAllocator().allocate(assets)

        # Calculate momentum scores for each asset
        momentum_scores = {}
        for asset in assets:
            if asset in market_data.columns:
                try:
                    score = self._calculate_momentum_score(market_data[asset])
                    momentum_scores[asset] = score
                except Exception as e:
                    self.logger.warning(
                        f"Could not calculate momentum for {asset}: {str(e)}"
                    )
                    momentum_scores[asset] = 0.0
            else:
                self.logger.warning(f"Asset {asset} not found in market data")
                momentum_scores[asset] = 0.0

        # Convert momentum scores to allocation weights
        allocations = self._momentum_to_allocations(momentum_scores)

        # Normalize to ensure sum = 1.0
        allocations = self.normalize_allocations(allocations)

        self.logger.info(f"Momentum weighted allocation: {allocations}")

        return allocations

    def _calculate_momentum_score(self, price_series: pd.Series) -> float:
        """
        Calculate momentum score for an asset.

        Args:
            price_series: Price series for the asset

        Returns:
            Momentum score
        """
        if len(price_series) < self.lookback_period:
            return 0.0

        try:
            if self.momentum_type == "returns":
                # Simple return over lookback period
                start_price = price_series.iloc[-self.lookback_period]
                end_price = price_series.iloc[-1]
                return (end_price - start_price) / start_price

            elif self.momentum_type == "sharpe":
                # Risk-adjusted return (simplified Sharpe)
                returns = price_series.pct_change().dropna()
                if len(returns) < 10:
                    return 0.0

                avg_return = returns.mean()
                volatility = returns.std()

                if volatility > 0:
                    return avg_return / volatility
                else:
                    return avg_return

            elif self.momentum_type == "vol_adjusted":
                # Volatility-adjusted momentum
                returns = price_series.pct_change().dropna()
                if len(returns) < 10:
                    return 0.0

                total_return = (
                    price_series.iloc[-1] - price_series.iloc[-self.lookback_period]
                ) / price_series.iloc[-self.lookback_period]
                volatility = returns.std()

                if volatility > 0:
                    return total_return / volatility
                else:
                    return total_return

            else:
                # Default to simple returns
                start_price = price_series.iloc[-self.lookback_period]
                end_price = price_series.iloc[-1]
                return (end_price - start_price) / start_price

        except Exception as e:
            self.logger.debug(f"Error calculating momentum score: {str(e)}")
            return 0.0

    def _momentum_to_allocations(
        self, momentum_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Convert momentum scores to allocation weights.

        Args:
            momentum_scores: Dictionary of momentum scores

        Returns:
            Dictionary of allocation weights
        """
        # Apply softmax to convert scores to probabilities
        scores = np.array(list(momentum_scores.values()))

        # Shift scores to be positive (avoid negative allocations)
        if len(scores) > 0:
            min_score = np.min(scores)
            if min_score < 0:
                scores = scores - min_score + 0.01  # Small positive offset

        # Apply softmax
        if len(scores) > 0 and np.sum(scores) > 0:
            exp_scores = np.exp(
                scores / np.max(scores)
            )  # Normalize by max to avoid overflow
            softmax_weights = exp_scores / np.sum(exp_scores)
        else:
            # Equal weights if all scores are zero or invalid
            softmax_weights = np.ones(len(scores)) / len(scores)

        # Convert back to dictionary
        allocations = {}
        for i, asset in enumerate(momentum_scores.keys()):
            allocations[asset] = float(softmax_weights[i])

        return allocations


class MinimumVarianceAllocator(CapitalAllocator):
    """
    Minimum Variance Allocation Strategy.

    Allocates capital to minimize portfolio variance.
    This is a classic mean-variance optimization approach.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Minimum Variance allocator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.lookback_period = self.config.get("lookback_period", 60)

    def allocate(
        self,
        assets: List[str],
        market_data: Optional[pd.DataFrame] = None,
        current_prices: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, float]:
        """
        Allocate capital using minimum variance optimization.

        Args:
            assets: List of asset symbols to allocate to
            market_data: Historical market data for covariance calculation
            current_prices: Not used in minimum variance allocation

        Returns:
            Dictionary mapping symbols to minimum variance allocation percentages
        """
        if not assets:
            return {}

        if market_data is None or market_data.empty or len(assets) < 2:
            # Fallback to equal weight if insufficient data
            self.logger.warning(
                "Insufficient data for minimum variance optimization, falling back to equal weight"
            )
            return EqualWeightAllocator().allocate(assets)

        try:
            # Calculate covariance matrix
            returns_data = market_data[assets].pct_change().dropna()

            if len(returns_data) < 10:
                return EqualWeightAllocator().allocate(assets)

            # Use only recent data
            recent_returns = returns_data.tail(
                min(self.lookback_period, len(returns_data))
            )

            # Calculate covariance matrix
            cov_matrix = recent_returns.cov().values

            # Minimize variance with constraint that weights sum to 1
            n_assets = len(assets)
            ones = np.ones(n_assets)

            # Solve: min w^T * Σ * w subject to w^T * 1 = 1
            # Using Lagrange multipliers
            try:
                inv_cov = np.linalg.inv(cov_matrix)
                weights = inv_cov @ ones
                weights = weights / np.sum(weights)

                # Ensure non-negative weights (no short selling)
                weights = np.maximum(weights, 0)
                weights = weights / np.sum(weights)

                allocations = {assets[i]: float(weights[i]) for i in range(n_assets)}

            except np.linalg.LinAlgError:
                # Fallback to equal weight if matrix is singular
                self.logger.warning(
                    "Covariance matrix is singular, falling back to equal weight"
                )
                return EqualWeightAllocator().allocate(assets)

        except Exception as e:
            self.logger.error(f"Error in minimum variance optimization: {str(e)}")
            return EqualWeightAllocator().allocate(assets)

        # Normalize to ensure sum = 1.0
        allocations = self.normalize_allocations(allocations)

        self.logger.info(f"Minimum variance allocation: {allocations}")

        return allocations


# Factory function for creating allocators
def create_allocator(
    allocator_type: str, config: Optional[Dict[str, Any]] = None
) -> CapitalAllocator:
    """
    Factory function to create capital allocators.

    Args:
        allocator_type: Type of allocator ('equal_weight', 'risk_parity', 'momentum_weighted', 'min_variance')
        config: Configuration dictionary for the allocator

    Returns:
        Configured allocator instance

    Raises:
        ValueError: If allocator type is not supported
    """
    allocators = {
        "equal_weight": EqualWeightAllocator,
        "risk_parity": RiskParityAllocator,
        "momentum_weighted": MomentumWeightAllocator,
        "min_variance": MinimumVarianceAllocator,
    }

    if allocator_type not in allocators:
        available = list(allocators.keys())
        raise ValueError(
            f"Unsupported allocator type: {allocator_type}. Available: {available}"
        )

    return allocators[allocator_type](config)
