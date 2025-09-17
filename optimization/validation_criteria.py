"""
Validation Criteria Module

This module defines the criteria and thresholds used for cross-asset validation.
It provides a centralized way to configure and evaluate strategy performance
against statistical thresholds and consistency requirements.

Key Features:
- Configurable performance thresholds (Sharpe ratio, drawdown, win rate, etc.)
- Consistency evaluation between primary and validation assets
- Pass/fail criteria assessment
- Overall validation scoring
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from .config import get_cross_asset_validation_config


class ValidationCriteria:
    """
    Defines pass/fail criteria for cross-asset validation.

    This class encapsulates all the logic for evaluating whether a strategy
    performs adequately across multiple validation assets. It uses configurable
    thresholds to determine if individual assets pass validation and calculates
    overall validation results.

    Attributes:
        min_sharpe_ratio: Minimum Sharpe ratio threshold
        max_drawdown_limit: Maximum drawdown limit (as decimal)
        min_win_rate: Minimum win rate threshold
        min_profit_factor: Minimum profit factor threshold
        consistency_threshold: Minimum consistency score between assets
        required_pass_rate: Minimum fraction of assets that must pass
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize validation criteria.

        Args:
            config: Configuration dictionary containing validation thresholds.
                   If None, uses default configuration from config module.
        """
        if config is None:
            validation_config = get_cross_asset_validation_config()
            config = {
                'min_sharpe_ratio': validation_config.validation_criteria.min_sharpe_ratio,
                'max_drawdown_limit': validation_config.validation_criteria.max_drawdown_limit,
                'min_win_rate': validation_config.validation_criteria.min_win_rate,
                'min_profit_factor': validation_config.validation_criteria.min_profit_factor,
                'consistency_threshold': validation_config.validation_criteria.consistency_threshold,
                'required_pass_rate': validation_config.validation_criteria.required_pass_rate,
                'min_calmar_ratio': validation_config.validation_criteria.min_calmar_ratio,
                'max_volatility': validation_config.validation_criteria.max_volatility,
                'min_sortino_ratio': validation_config.validation_criteria.min_sortino_ratio
            }

        # Core performance thresholds
        self.min_sharpe_ratio = config.get('min_sharpe_ratio', 0.5)
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.15)  # 15%
        self.min_win_rate = config.get('min_win_rate', 0.45)
        self.min_profit_factor = config.get('min_profit_factor', 1.2)

        # Consistency and robustness thresholds
        self.consistency_threshold = config.get('consistency_threshold', 0.7)
        self.required_pass_rate = config.get('required_pass_rate', 0.6)  # 60%

        # Advanced risk-adjusted metrics
        self.min_calmar_ratio = config.get('min_calmar_ratio', 0.3)
        self.max_volatility = config.get('max_volatility', 0.25)
        self.min_sortino_ratio = config.get('min_sortino_ratio', 0.4)

    def evaluate_asset(self, primary_metrics: Dict[str, Any],
                      validation_metrics: Dict[str, Any]) -> Tuple[Dict[str, bool], bool]:
        """
        Evaluate if an asset passes validation criteria.

        This method compares the performance metrics of a validation asset
        against the primary asset and determines whether it meets all
        required thresholds.

        Args:
            primary_metrics: Performance metrics from primary asset optimization
            validation_metrics: Performance metrics from validation asset

        Returns:
            Tuple of (pass_criteria_dict, overall_pass) where:
            - pass_criteria_dict: Dictionary mapping criterion names to pass/fail status
            - overall_pass: Boolean indicating if all criteria passed
        """
        pass_criteria = {}

        # Core performance criteria
        pass_criteria['sharpe_ratio'] = self._evaluate_sharpe_ratio(validation_metrics)
        pass_criteria['max_drawdown'] = self._evaluate_max_drawdown(validation_metrics)
        pass_criteria['win_rate'] = self._evaluate_win_rate(validation_metrics)
        pass_criteria['profit_factor'] = self._evaluate_profit_factor(validation_metrics)

        # Consistency criterion
        consistency_score = self._calculate_consistency_score(primary_metrics, validation_metrics)
        pass_criteria['consistency'] = consistency_score >= self.consistency_threshold

        # Advanced risk-adjusted criteria
        pass_criteria['calmar_ratio'] = self._evaluate_calmar_ratio(validation_metrics)
        pass_criteria['volatility'] = self._evaluate_volatility(validation_metrics)
        pass_criteria['sortino_ratio'] = self._evaluate_sortino_ratio(validation_metrics)

        # Overall pass (all criteria must pass)
        overall_pass = all(pass_criteria.values())

        return pass_criteria, overall_pass

    def evaluate_overall(self, asset_results: List[Any]) -> Tuple[float, bool]:
        """
        Evaluate overall validation results across all assets.

        This method aggregates results from multiple validation assets and
        determines if the strategy passes overall validation based on the
        required pass rate.

        Args:
            asset_results: List of individual asset validation results

        Returns:
            Tuple of (pass_rate, overall_pass) where:
            - pass_rate: Fraction of assets that passed validation
            - overall_pass: Boolean indicating overall validation success
        """
        if not asset_results:
            return 0.0, False

        # Calculate pass rate
        passed_assets = sum(1 for result in asset_results if result.overall_pass)
        pass_rate = passed_assets / len(asset_results)

        # Overall pass based on required pass rate
        overall_pass = pass_rate >= self.required_pass_rate

        return pass_rate, overall_pass

    def _evaluate_sharpe_ratio(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate Sharpe ratio criterion."""
        sharpe = metrics.get('sharpe_ratio', 0)
        return sharpe >= self.min_sharpe_ratio

    def _evaluate_max_drawdown(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate maximum drawdown criterion."""
        max_dd = metrics.get('max_drawdown', 1.0)
        return max_dd <= self.max_drawdown_limit

    def _evaluate_win_rate(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate win rate criterion."""
        win_rate = metrics.get('win_rate', 0)
        return win_rate >= self.min_win_rate

    def _evaluate_profit_factor(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate profit factor criterion."""
        profit_factor = metrics.get('profit_factor', 0)
        return profit_factor >= self.min_profit_factor

    def _evaluate_calmar_ratio(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate Calmar ratio criterion."""
        calmar = metrics.get('calmar_ratio', 0)
        return calmar >= self.min_calmar_ratio

    def _evaluate_volatility(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate volatility criterion."""
        volatility = metrics.get('volatility', 0)
        return volatility <= self.max_volatility

    def _evaluate_sortino_ratio(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate Sortino ratio criterion."""
        sortino = metrics.get('sortino_ratio', 0)
        return sortino >= self.min_sortino_ratio

    def _calculate_consistency_score(self, primary: Dict[str, Any],
                                   validation: Dict[str, Any]) -> float:
        """
        Calculate consistency score between primary and validation metrics.

        This method evaluates how consistent the strategy's performance is
        between the primary asset (used for optimization) and the validation
        asset. It handles division by zero safely and provides a normalized
        consistency score.

        Args:
            primary: Primary asset performance metrics
            validation: Validation asset performance metrics

        Returns:
            Consistency score between 0.0 (inconsistent) and 1.0 (perfectly consistent)
        """
        metrics_to_compare = [
            'sharpe_ratio', 'total_return', 'win_rate', 'profit_factor',
            'calmar_ratio', 'sortino_ratio'
        ]

        consistency_scores = []

        for metric in metrics_to_compare:
            primary_val = primary.get(metric, 0)
            validation_val = validation.get(metric, 0)

            # Safe division to prevent ZeroDivisionError
            if primary_val != 0:
                # Calculate relative difference
                relative_diff = abs(validation_val - primary_val) / abs(primary_val)
                # Convert to consistency score (lower difference = higher consistency)
                consistency = max(0, 1 - relative_diff)
            else:
                # Handle zero primary value case
                # If both values are effectively zero, they are perfectly consistent
                if abs(validation_val) < 1e-6:
                    consistency = 1.0
                else:
                    # If primary is zero but validation is not, they are inconsistent
                    consistency = 0.0

            consistency_scores.append(consistency)

        # Return average consistency score
        return np.mean(consistency_scores) if consistency_scores else 0.0

    def get_criteria_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all validation criteria.

        Returns:
            Dictionary containing all criteria thresholds and descriptions
        """
        return {
            'performance_thresholds': {
                'min_sharpe_ratio': self.min_sharpe_ratio,
                'max_drawdown_limit': self.max_drawdown_limit,
                'min_win_rate': self.min_win_rate,
                'min_profit_factor': self.min_profit_factor
            },
            'consistency_thresholds': {
                'consistency_threshold': self.consistency_threshold,
                'required_pass_rate': self.required_pass_rate
            },
            'advanced_metrics': {
                'min_calmar_ratio': self.min_calmar_ratio,
                'max_volatility': self.max_volatility,
                'min_sortino_ratio': self.min_sortino_ratio
            }
        }

    def calculate_robustness_score(self, asset_results: List[Any]) -> float:
        """
        Calculate overall robustness score for the validation.

        This method provides a comprehensive robustness assessment based on
        multiple factors including pass rate, consistency, and performance
        stability across validation assets.

        Args:
            asset_results: List of asset validation results

        Returns:
            Robustness score between 0.0 and 1.0
        """
        if not asset_results:
            return 0.0

        # Component scores
        pass_rate, _ = self.evaluate_overall(asset_results)
        pass_rate_score = pass_rate

        # Consistency score (average consistency across all assets)
        consistency_scores = []
        for result in asset_results:
            if hasattr(result, 'validation_metrics') and hasattr(result, 'primary_metrics'):
                consistency = self._calculate_consistency_score(
                    result.primary_metrics, result.validation_metrics
                )
                consistency_scores.append(consistency)

        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0

        # Performance stability (coefficient of variation of key metrics)
        sharpe_ratios = [r.validation_metrics.get('sharpe_ratio', 0)
                        for r in asset_results if hasattr(r, 'validation_metrics')]
        sharpe_stability = 1.0 / (1.0 + np.std(sharpe_ratios)) if len(sharpe_ratios) > 1 else 1.0

        # Weighted robustness score
        robustness = (
            0.4 * pass_rate_score +
            0.3 * avg_consistency +
            0.3 * sharpe_stability
        )

        return min(1.0, max(0.0, robustness))  # Clamp to [0, 1]
