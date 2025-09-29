"""
Validation Results Module

This module defines the data structures and result classes used for
cross-asset validation. It provides comprehensive result tracking,
serialization, and analysis capabilities for validation outcomes.

Key Features:
- Structured result classes for individual asset validation
- Complete cross-asset validation result aggregation
- JSON serialization and deserialization
- Result analysis and reporting utilities
- Performance metrics calculation and comparison
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .asset_selector import ValidationAsset


@dataclass
class AssetValidationResult:
    """
    Result of validating a strategy on a single asset.

    This class encapsulates all the information from validating a trading
    strategy on a specific validation asset, including performance metrics,
    pass/fail criteria evaluation, and timing information.

    Attributes:
        asset: The validation asset that was tested
        optimized_params: Parameters used for the strategy
        primary_metrics: Performance metrics from primary asset optimization
        validation_metrics: Performance metrics from validation asset
        pass_criteria: Dictionary of pass/fail results for each criterion
        overall_pass: Whether the asset passed overall validation
        validation_time: Time taken for validation in seconds
        error_message: Error message if validation failed
    """

    asset: ValidationAsset
    optimized_params: Dict[str, Any]
    primary_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    pass_criteria: Dict[str, bool]
    overall_pass: bool
    validation_time: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the result
        """
        return {
            "asset": self.asset.to_dict(),
            "optimized_params": self.optimized_params,
            "primary_metrics": self.primary_metrics,
            "validation_metrics": self.validation_metrics,
            "pass_criteria": self.pass_criteria,
            "overall_pass": self.overall_pass,
            "validation_time": self.validation_time,
            "error_message": self.error_message,
            "timestamp": datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetValidationResult":
        """
        Create from dictionary.

        Args:
            data: Dictionary representation of the result

        Returns:
            AssetValidationResult instance
        """
        from .asset_selector import ValidationAsset

        return cls(
            asset=ValidationAsset.from_dict(data["asset"]),
            optimized_params=data.get("optimized_params", {}),
            primary_metrics=data.get("primary_metrics", {}),
            validation_metrics=data.get("validation_metrics", {}),
            pass_criteria=data.get("pass_criteria", {}),
            overall_pass=data.get("overall_pass", False),
            validation_time=data.get("validation_time", 0.0),
            error_message=data.get("error_message"),
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.

        Returns:
            Dictionary with key performance indicators
        """
        return {
            "asset_symbol": self.asset.symbol,
            "asset_name": self.asset.name,
            "sharpe_ratio": self.validation_metrics.get("sharpe_ratio", 0),
            "total_return": self.validation_metrics.get("total_return", 0),
            "win_rate": self.validation_metrics.get("win_rate", 0),
            "max_drawdown": self.validation_metrics.get("max_drawdown", 0),
            "profit_factor": self.validation_metrics.get("profit_factor", 0),
            "overall_pass": self.overall_pass,
            "validation_time": self.validation_time,
        }

    def get_criteria_details(self) -> Dict[str, Any]:
        """
        Get detailed breakdown of pass/fail criteria.

        Returns:
            Dictionary with criteria evaluation details
        """
        return {
            "asset_symbol": self.asset.symbol,
            "pass_criteria": self.pass_criteria,
            "overall_pass": self.overall_pass,
            "failed_criteria": [k for k, v in self.pass_criteria.items() if not v],
            "passed_criteria": [k for k, v in self.pass_criteria.items() if v],
        }


@dataclass
class CrossAssetValidationResult:
    """
    Complete cross-asset validation results.

    This class aggregates results from validating a strategy across multiple
    assets, providing comprehensive analysis and reporting capabilities.

    Attributes:
        strategy_name: Name of the strategy being validated
        primary_asset: Primary asset symbol used for optimization
        validation_assets: List of validation assets used
        asset_results: Individual results for each validation asset
        aggregate_metrics: Aggregated performance metrics across all assets
        pass_rate: Fraction of assets that passed validation
        overall_pass: Whether the strategy passed overall validation
        robustness_score: Overall robustness score (0-1)
        timestamp: When validation was performed
        total_time: Total time taken for validation
    """

    strategy_name: str
    primary_asset: str
    validation_assets: List[ValidationAsset]
    asset_results: List[AssetValidationResult]
    aggregate_metrics: Dict[str, Any]
    pass_rate: float
    overall_pass: bool
    robustness_score: float
    timestamp: datetime
    total_time: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the complete validation result
        """
        return {
            "strategy_name": self.strategy_name,
            "primary_asset": self.primary_asset,
            "validation_assets": [asset.to_dict() for asset in self.validation_assets],
            "asset_results": [result.to_dict() for result in self.asset_results],
            "aggregate_metrics": self.aggregate_metrics,
            "pass_rate": self.pass_rate,
            "overall_pass": self.overall_pass,
            "robustness_score": self.robustness_score,
            "timestamp": self.timestamp.isoformat(),
            "total_time": self.total_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossAssetValidationResult":
        """
        Create from dictionary.

        Args:
            data: Dictionary representation of the result

        Returns:
            CrossAssetValidationResult instance
        """
        from .asset_selector import ValidationAsset

        return cls(
            strategy_name=data.get("strategy_name", "unknown"),
            primary_asset=data.get("primary_asset", "unknown"),
            validation_assets=[
                ValidationAsset.from_dict(asset_data)
                for asset_data in data.get("validation_assets", [])
            ],
            asset_results=[
                AssetValidationResult.from_dict(result_data)
                for result_data in data.get("asset_results", [])
            ],
            aggregate_metrics=data.get("aggregate_metrics", {}),
            pass_rate=data.get("pass_rate", 0.0),
            overall_pass=data.get("overall_pass", False),
            robustness_score=data.get("robustness_score", 0.0),
            timestamp=datetime.fromisoformat(
                data.get("timestamp", datetime.now().isoformat())
            ),
            total_time=data.get("total_time", 0.0),
        )

    def get_summary_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary report of the validation.

        Returns:
            Dictionary containing key validation metrics and insights
        """
        successful_validations = len(
            [r for r in self.asset_results if not r.error_message]
        )
        failed_validations = len([r for r in self.asset_results if r.error_message])

        return {
            "strategy_name": self.strategy_name,
            "primary_asset": self.primary_asset,
            "total_validation_assets": len(self.validation_assets),
            "successful_validations": successful_validations,
            "failed_validations": failed_validations,
            "pass_rate": self.pass_rate,
            "overall_pass": self.overall_pass,
            "robustness_score": self.robustness_score,
            "total_time": self.total_time,
            "timestamp": self.timestamp.isoformat(),
            "aggregate_metrics": self.aggregate_metrics,
        }

    def get_asset_performance_comparison(self) -> pd.DataFrame:
        """
        Get a DataFrame comparing performance across all validation assets.

        Returns:
            DataFrame with performance metrics for each asset
        """
        performance_data = []

        for result in self.asset_results:
            if result.error_message:
                # Include failed validations with NaN values
                performance_data.append(
                    {
                        "asset_symbol": result.asset.symbol,
                        "asset_name": result.asset.name,
                        "sharpe_ratio": np.nan,
                        "total_return": np.nan,
                        "win_rate": np.nan,
                        "max_drawdown": np.nan,
                        "profit_factor": np.nan,
                        "overall_pass": False,
                        "validation_time": result.validation_time,
                        "error": result.error_message,
                    }
                )
            else:
                performance_data.append(
                    {
                        "asset_symbol": result.asset.symbol,
                        "asset_name": result.asset.name,
                        "sharpe_ratio": result.validation_metrics.get(
                            "sharpe_ratio", 0
                        ),
                        "total_return": result.validation_metrics.get(
                            "total_return", 0
                        ),
                        "win_rate": result.validation_metrics.get("win_rate", 0),
                        "max_drawdown": result.validation_metrics.get(
                            "max_drawdown", 0
                        ),
                        "profit_factor": result.validation_metrics.get(
                            "profit_factor", 0
                        ),
                        "overall_pass": result.overall_pass,
                        "validation_time": result.validation_time,
                        "error": None,
                    }
                )

        return pd.DataFrame(performance_data)

    def get_criteria_pass_rates(self) -> Dict[str, float]:
        """
        Get pass rates for each validation criterion across all assets.

        Returns:
            Dictionary mapping criteria names to pass rates
        """
        if not self.asset_results:
            return {}

        criteria_counts = {}
        total_valid_results = 0

        for result in self.asset_results:
            if result.error_message:
                continue

            total_valid_results += 1
            for criterion, passed in result.pass_criteria.items():
                if criterion not in criteria_counts:
                    criteria_counts[criterion] = 0
                if passed:
                    criteria_counts[criterion] += 1

        # Calculate pass rates
        pass_rates = {}
        for criterion, count in criteria_counts.items():
            pass_rates[criterion] = (
                count / total_valid_results if total_valid_results > 0 else 0.0
            )

        return pass_rates

    def get_failed_assets(self) -> List[Dict[str, Any]]:
        """
        Get list of assets that failed validation with failure reasons.

        Returns:
            List of dictionaries with failure details
        """
        failed_assets = []

        for result in self.asset_results:
            if not result.overall_pass:
                failed_criteria = [k for k, v in result.pass_criteria.items() if not v]

                failed_assets.append(
                    {
                        "asset_symbol": result.asset.symbol,
                        "asset_name": result.asset.name,
                        "error_message": result.error_message,
                        "failed_criteria": failed_criteria,
                        "validation_time": result.validation_time,
                    }
                )

        return failed_assets

    def get_best_performing_asset(self) -> Optional[Dict[str, Any]]:
        """
        Get the best performing validation asset.

        Returns:
            Dictionary with best asset details, or None if no valid results
        """
        valid_results = [r for r in self.asset_results if not r.error_message]

        if not valid_results:
            return None

        # Find asset with highest Sharpe ratio
        best_result = max(
            valid_results,
            key=lambda r: r.validation_metrics.get("sharpe_ratio", float("-inf")),
        )

        return {
            "asset_symbol": best_result.asset.symbol,
            "asset_name": best_result.asset.name,
            "sharpe_ratio": best_result.validation_metrics.get("sharpe_ratio", 0),
            "total_return": best_result.validation_metrics.get("total_return", 0),
            "win_rate": best_result.validation_metrics.get("win_rate", 0),
            "overall_pass": best_result.overall_pass,
        }

    def get_worst_performing_asset(self) -> Optional[Dict[str, Any]]:
        """
        Get the worst performing validation asset.

        Returns:
            Dictionary with worst asset details, or None if no valid results
        """
        valid_results = [r for r in self.asset_results if not r.error_message]

        if not valid_results:
            return None

        # Find asset with lowest Sharpe ratio
        worst_result = min(
            valid_results,
            key=lambda r: r.validation_metrics.get("sharpe_ratio", float("inf")),
        )

        return {
            "asset_symbol": worst_result.asset.symbol,
            "asset_name": worst_result.asset.name,
            "sharpe_ratio": worst_result.validation_metrics.get("sharpe_ratio", 0),
            "total_return": worst_result.validation_metrics.get("total_return", 0),
            "win_rate": worst_result.validation_metrics.get("win_rate", 0),
            "overall_pass": worst_result.overall_pass,
        }

    def save_to_file(self, file_path: str) -> None:
        """
        Save validation results to a JSON file.

        Args:
            file_path: Path to save the results
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, file_path: str) -> "CrossAssetValidationResult":
        """
        Load validation results from a JSON file.

        Args:
            file_path: Path to load results from

        Returns:
            CrossAssetValidationResult instance
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Results file not found: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def generate_csv_report(self, file_path: str) -> None:
        """
        Generate a CSV report of validation results.

        Args:
            file_path: Path to save the CSV report
        """
        df = self.get_asset_performance_comparison()
        df.to_csv(file_path, index=False)

    def __str__(self) -> str:
        """
        String representation of the validation result.

        Returns:
            Human-readable summary
        """
        return (
            f"CrossAssetValidationResult(strategy={self.strategy_name}, "
            f"primary_asset={self.primary_asset}, "
            f"assets_tested={len(self.validation_assets)}, "
            f"pass_rate={self.pass_rate:.1%}, "
            f"overall_pass={self.overall_pass}, "
            f"robustness={self.robustness_score:.3f})"
        )


class ValidationResultAnalyzer:
    """
    Utility class for analyzing and comparing validation results.

    This class provides methods for comparing multiple validation results,
    generating comparative reports, and identifying trends in strategy
    performance across different assets and time periods.
    """

    @staticmethod
    def compare_results(results: List[CrossAssetValidationResult]) -> Dict[str, Any]:
        """
        Compare multiple validation results.

        Args:
            results: List of validation results to compare

        Returns:
            Dictionary with comparative analysis
        """
        if not results:
            return {}

        # Extract key metrics
        pass_rates = [r.pass_rate for r in results]
        robustness_scores = [r.robustness_score for r in results]
        total_times = [r.total_time for r in results]

        return {
            "num_results": len(results),
            "avg_pass_rate": np.mean(pass_rates),
            "std_pass_rate": np.std(pass_rates),
            "best_pass_rate": max(pass_rates),
            "worst_pass_rate": min(pass_rates),
            "avg_robustness": np.mean(robustness_scores),
            "std_robustness": np.std(robustness_scores),
            "best_robustness": max(robustness_scores),
            "worst_robustness": min(robustness_scores),
            "avg_time": np.mean(total_times),
            "total_time": sum(total_times),
        }

    @staticmethod
    def find_common_failures(
        results: List[CrossAssetValidationResult],
    ) -> Dict[str, int]:
        """
        Find assets that commonly fail validation across multiple results.

        Args:
            results: List of validation results

        Returns:
            Dictionary mapping asset symbols to failure counts
        """
        failure_counts = {}

        for result in results:
            for asset_result in result.asset_results:
                if not asset_result.overall_pass:
                    symbol = asset_result.asset.symbol
                    failure_counts[symbol] = failure_counts.get(symbol, 0) + 1

        return failure_counts

    @staticmethod
    def generate_performance_matrix(
        results: List[CrossAssetValidationResult],
    ) -> pd.DataFrame:
        """
        Generate a performance matrix across strategies and assets.

        Args:
            results: List of validation results

        Returns:
            DataFrame with strategies as rows and assets as columns
        """
        if not results:
            return pd.DataFrame()

        # Collect all unique assets
        all_assets = set()
        for result in results:
            for asset in result.validation_assets:
                all_assets.add(asset.symbol)

        all_assets = sorted(list(all_assets))

        # Build performance matrix
        matrix_data = []
        for result in results:
            row = {"strategy": result.strategy_name}
            asset_performance = {
                asset.symbol: False for asset in result.validation_assets
            }

            for asset_result in result.asset_results:
                asset_performance[asset_result.asset.symbol] = asset_result.overall_pass

            row.update(asset_performance)
            matrix_data.append(row)

        return pd.DataFrame(matrix_data)
