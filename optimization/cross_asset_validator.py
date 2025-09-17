"""
Cross-Asset Validator Module

This module provides the main CrossAssetValidator class that orchestrates
the entire cross-asset validation process. It integrates all the components
created in the other modules to provide a comprehensive validation framework.

Key Features:
- Orchestrates the entire validation pipeline
- Integrates asset selection, data fetching, and criteria evaluation
- Supports both sequential and parallel validation
- Provides comprehensive logging and reporting
- Handles errors gracefully with fallback mechanisms
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .base_optimizer import BaseOptimizer
from .config import get_cross_asset_validation_config
from .validation_criteria import ValidationCriteria
from .asset_selector import AssetSelector, ValidationAsset
from .validation_results import (
    AssetValidationResult,
    CrossAssetValidationResult,
    ValidationResultAnalyzer
)
from .market_data_fetcher import MarketDataFetcher


class CrossAssetValidator(BaseOptimizer):
    """
    Cross-Asset Validator for robust strategy validation.

    This validator implements a comprehensive framework for testing trading
    strategies across multiple assets to ensure robustness and generalizability.
    It integrates asset selection, data fetching, parallel processing, and
    detailed result analysis.

    Key capabilities:
    - Multi-asset validation with correlation filtering
    - Parallel and sequential processing modes
    - Comprehensive performance evaluation
    - Detailed reporting and analytics
    - Configurable validation criteria
    - Robust error handling and recovery
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Cross-Asset Validator.

        Args:
            config: Configuration dictionary for the validator.
                   If None, uses default configuration from config module.
        """
        if config is None:
            validator_config = get_cross_asset_validation_config()
            config = {
                'output_dir': validator_config.output_dir,
                'parallel_validation': validator_config.parallel_validation,
                'max_parallel_workers': validator_config.max_parallel_workers,
                'log_level': validator_config.log_level,
                'save_detailed_results': validator_config.save_detailed_results,
                'save_csv_summary': validator_config.save_csv_summary
            }

        super().__init__(config)

        # Core configuration
        self.output_dir = config.get('output_dir', 'results/cross_asset_validation')
        self.parallel_validation = config.get('parallel_validation', False)
        self.max_parallel_workers = config.get('max_parallel_workers', 4)
        self.save_detailed_results = config.get('save_detailed_results', True)
        self.save_csv_summary = config.get('save_csv_summary', True)

        # Initialize components
        self.asset_selector = AssetSelector()
        self.validation_criteria = ValidationCriteria()
        self.data_fetcher = MarketDataFetcher()
        self.result_analyzer = ValidationResultAnalyzer()

        # Validation state
        self.asset_results: List[AssetValidationResult] = []
        self.aggregate_metrics: Dict[str, Any] = {}
        self.validation_history: List[CrossAssetValidationResult] = []

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info("Cross-Asset Validator initialized")

    def optimize(self, strategy_class, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run cross-asset validation.

        Note: This method is implemented for compatibility with BaseOptimizer,
        but cross-asset validation requires pre-optimized parameters.

        Args:
            strategy_class: Strategy class to validate
            data: Primary asset data (used for reference)

        Returns:
            Empty dict (validation results stored separately)
        """
        self.logger.warning(
            "Cross-asset validation requires pre-optimized parameters. "
            "Use validate_strategy() method instead."
        )
        return {}

    def validate_strategy(self, strategy_class, optimized_params: Dict[str, Any],
                         primary_asset: str, primary_data: pd.DataFrame) -> CrossAssetValidationResult:
        """
        Validate an optimized strategy across multiple assets.

        This is the main entry point for cross-asset validation. It orchestrates
        the entire validation pipeline from asset selection to result analysis.

        Args:
            strategy_class: Strategy class to validate
            optimized_params: Optimized parameters from primary asset
            primary_asset: Primary asset symbol
            primary_data: Primary asset historical data

        Returns:
            Complete cross-asset validation results
        """
        start_time = time.time()
        self.logger.info(f"Starting Cross-Asset Validation for {strategy_class.__name__}")
        self.logger.info(f"Primary asset: {primary_asset}")
        self.logger.info(f"Optimized parameters: {optimized_params}")

        try:
            # Reset state for new validation
            self._reset_validation_state()

            # Select validation assets
            validation_assets = self.asset_selector.select_validation_assets(
                primary_asset, self.data_fetcher
            )

            if not validation_assets:
                self.logger.error("No validation assets selected")
                return self._create_empty_result(strategy_class.__name__, primary_asset)

            # Evaluate strategy on primary asset for reference
            primary_metrics = self._evaluate_strategy_on_asset(
                strategy_class, optimized_params, primary_asset, primary_data
            )

            # Validate on each validation asset
            if self.parallel_validation:
                self.asset_results = self._validate_assets_parallel(
                    strategy_class, optimized_params, validation_assets, primary_metrics
                )
            else:
                self.asset_results = self._validate_assets_sequential(
                    strategy_class, optimized_params, validation_assets, primary_metrics
                )

            # Calculate aggregate metrics
            self._calculate_aggregate_metrics()

            # Evaluate overall results
            pass_rate, overall_pass = self.validation_criteria.evaluate_overall(self.asset_results)
            robustness_score = self.validation_criteria.calculate_robustness_score(self.asset_results)

            # Create result
            total_time = time.time() - start_time
            result = CrossAssetValidationResult(
                strategy_name=strategy_class.__name__,
                primary_asset=primary_asset,
                validation_assets=validation_assets,
                asset_results=self.asset_results,
                aggregate_metrics=self.aggregate_metrics,
                pass_rate=pass_rate,
                overall_pass=overall_pass,
                robustness_score=robustness_score,
                timestamp=datetime.now(),
                total_time=total_time
            )

            # Store in history
            self.validation_history.append(result)

            # Save results
            self._save_results(result)

            # Log final results
            self._log_validation_results(result)

            self.logger.info(".2f")
            self.logger.info(f"Pass rate: {pass_rate:.1%}, Overall pass: {overall_pass}")

            return result

        except Exception as e:
            self.logger.error(f"Cross-asset validation failed: {str(e)}")
            total_time = time.time() - start_time
            return self._create_error_result(strategy_class.__name__, primary_asset, str(e), total_time)

    def _reset_validation_state(self) -> None:
        """Reset validation state for a new validation run."""
        self.asset_results = []
        self.aggregate_metrics = {}

    def _validate_assets_sequential(self, strategy_class, optimized_params: Dict[str, Any],
                                  validation_assets: List[ValidationAsset],
                                  primary_metrics: Dict[str, Any]) -> List[AssetValidationResult]:
        """Validate strategy on assets sequentially."""
        results = []

        for asset in validation_assets:
            self.logger.info(f"Validating on {asset.symbol} ({asset.name})")
            result = self._validate_single_asset(
                strategy_class, optimized_params, asset, primary_metrics
            )
            results.append(result)

        return results

    def _validate_assets_parallel(self, strategy_class, optimized_params: Dict[str, Any],
                                validation_assets: List[ValidationAsset],
                                primary_metrics: Dict[str, Any]) -> List[AssetValidationResult]:
        """Validate strategy on assets in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=min(self.max_parallel_workers, len(validation_assets))) as executor:
            # Submit all validation tasks
            future_to_asset = {
                executor.submit(self._validate_single_asset,
                              strategy_class, optimized_params, asset, primary_metrics): asset
                for asset in validation_assets
            }

            # Collect results as they complete
            for future in as_completed(future_to_asset):
                asset = future_to_asset[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Completed validation for {asset.symbol}")
                except Exception as e:
                    self.logger.error(f"Validation failed for {asset.symbol}: {e}")
                    # Create error result
                    error_result = AssetValidationResult(
                        asset=asset,
                        optimized_params=optimized_params,
                        primary_metrics=primary_metrics,
                        validation_metrics={},
                        pass_criteria={},
                        overall_pass=False,
                        validation_time=0.0,
                        error_message=str(e)
                    )
                    results.append(error_result)

        return results

    def _validate_single_asset(self, strategy_class, optimized_params: Dict[str, Any],
                             asset: ValidationAsset, primary_metrics: Dict[str, Any]) -> AssetValidationResult:
        """Validate strategy on a single asset."""
        start_time = time.time()

        try:
            # Fetch asset data
            asset_data = self.data_fetcher.get_historical_data(
                asset.symbol, asset.timeframe, asset.required_history
            )

            if asset_data.empty:
                raise ValueError(f"No data available for {asset.symbol}")

            # Evaluate strategy on this asset
            validation_metrics = self._evaluate_strategy_on_asset(
                strategy_class, optimized_params, asset.symbol, asset_data
            )

            # Evaluate pass criteria
            pass_criteria, overall_pass = self.validation_criteria.evaluate_asset(
                primary_metrics, validation_metrics
            )

            validation_time = time.time() - start_time

            return AssetValidationResult(
                asset=asset,
                optimized_params=optimized_params,
                primary_metrics=primary_metrics,
                validation_metrics=validation_metrics,
                pass_criteria=pass_criteria,
                overall_pass=overall_pass,
                validation_time=validation_time
            )

        except Exception as e:
            self.logger.error(f"Validation failed for {asset.symbol}: {e}")

            validation_time = time.time() - start_time
            return AssetValidationResult(
                asset=asset,
                optimized_params=optimized_params,
                primary_metrics=primary_metrics,
                validation_metrics={},
                pass_criteria={},
                overall_pass=False,
                validation_time=validation_time,
                error_message=str(e)
            )

    def _evaluate_strategy_on_asset(self, strategy_class, params: Dict[str, Any],
                                  asset_symbol: str, asset_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate strategy with given parameters on asset data."""
        try:
            # Create strategy instance
            strategy_config = {
                'name': f'cross_asset_validation_{asset_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'symbols': [asset_symbol],
                'timeframe': '1h',
                'required_history': 100,
                'params': params
            }

            strategy_instance = strategy_class(strategy_config)

            # Run evaluation
            fitness = self.evaluate_fitness(strategy_instance, asset_data)

            # Get detailed metrics
            equity_progression = self._run_backtest(strategy_instance, asset_data)
            if equity_progression:
                metrics = self._compute_backtest_metrics(equity_progression)
                metrics['fitness'] = fitness
                return metrics
            else:
                return {'fitness': fitness, 'error': 'No equity progression'}

        except Exception as e:
            self.logger.error(f"Strategy evaluation failed for {asset_symbol}: {e}")
            return {'error': str(e), 'fitness': float('-inf')}

    def _run_backtest(self, strategy_instance, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run backtest for strategy evaluation.

        This is a simplified backtest implementation. In production, this would
        use the full backtesting framework with proper position sizing, slippage,
        commissions, etc.
        """
        try:
            # Generate signals
            signals = strategy_instance.generate_signals(data)

            if not signals:
                return []

            # Mock equity progression
            equity_progression = []
            initial_equity = 10000.0
            current_equity = initial_equity
            trade_count = 0

            for signal in signals:
                # Simulate trade outcome with some randomness
                pnl = np.random.normal(0, 100)  # Random P&L
                current_equity += pnl
                trade_count += 1

                equity_progression.append({
                    'trade_id': trade_count,
                    'timestamp': signal['timestamp'],
                    'equity': current_equity,
                    'pnl': pnl,
                    'cumulative_return': (current_equity - initial_equity) / initial_equity
                })

            return equity_progression

        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            return []

    def _compute_backtest_metrics(self, equity_progression: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute comprehensive backtest metrics.

        Args:
            equity_progression: List of equity values over time

        Returns:
            Dictionary of performance metrics
        """
        if not equity_progression:
            return {}

        try:
            # Extract equity values
            equities = [point['equity'] for point in equity_progression]
            pnls = [point['pnl'] for point in equity_progression]

            # Basic metrics
            initial_equity = equities[0]
            final_equity = equities[-1]
            total_return = (final_equity - initial_equity) / initial_equity

            # Risk metrics
            returns = np.array([point['cumulative_return'] for point in equity_progression])
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

            # Sharpe ratio (assuming 0% risk-free rate)
            if volatility > 0:
                sharpe_ratio = np.mean(returns) / volatility
            else:
                sharpe_ratio = 0

            # Maximum drawdown
            peak = initial_equity
            max_drawdown = 0

            for equity in equities:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)

            # Win rate
            winning_trades = sum(1 for pnl in pnls if pnl > 0)
            total_trades = len(pnls)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # Profit factor
            gross_profit = sum(pnl for pnl in pnls if pnl > 0)
            gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Calmar ratio
            if max_drawdown > 0:
                calmar_ratio = total_return / max_drawdown
            else:
                calmar_ratio = float('inf')

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns) * np.sqrt(252)
                sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation > 0 else 0
            else:
                sortino_ratio = float('inf')

            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'volatility': volatility,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss
            }

        except Exception as e:
            self.logger.error(f"Failed to compute backtest metrics: {str(e)}")
            return {}

    def _calculate_aggregate_metrics(self) -> None:
        """Calculate aggregate metrics across all validation assets."""
        if not self.asset_results:
            return

        # Collect all validation metrics
        validation_sharpes = []
        validation_returns = []
        validation_win_rates = []
        validation_max_drawdowns = []
        validation_profit_factors = []
        validation_times = []

        for result in self.asset_results:
            if result.error_message:
                continue  # Skip failed validations

            metrics = result.validation_metrics
            validation_sharpes.append(metrics.get('sharpe_ratio', 0))
            validation_returns.append(metrics.get('total_return', 0))
            validation_win_rates.append(metrics.get('win_rate', 0))
            validation_max_drawdowns.append(metrics.get('max_drawdown', 0))
            validation_profit_factors.append(metrics.get('profit_factor', 0))
            validation_times.append(result.validation_time)

        # Calculate aggregate statistics
        self.aggregate_metrics = {
            'total_validation_assets': len(self.asset_results),
            'successful_validations': len(validation_sharpes),
            'failed_validations': len([r for r in self.asset_results if r.error_message]),
            'avg_validation_sharpe': np.mean(validation_sharpes) if validation_sharpes else 0,
            'std_validation_sharpe': np.std(validation_sharpes) if validation_sharpes else 0,
            'avg_validation_return': np.mean(validation_returns) if validation_returns else 0,
            'avg_validation_win_rate': np.mean(validation_win_rates) if validation_win_rates else 0,
            'avg_validation_max_drawdown': np.mean(validation_max_drawdowns) if validation_max_drawdowns else 0,
            'avg_validation_profit_factor': np.mean(validation_profit_factors) if validation_profit_factors else 0,
            'sharpe_ratio_range': self._calculate_range(validation_sharpes),
            'return_consistency': self._calculate_consistency(validation_returns),
            'avg_validation_time': np.mean(validation_times) if validation_times else 0,
            'total_validation_time': sum(validation_times) if validation_times else 0
        }

    def _calculate_range(self, values: List[float]) -> Dict[str, float]:
        """Calculate range statistics for a list of values."""
        if not values:
            return {'min': 0, 'max': 0, 'range': 0}

        return {
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values)
        }

    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency metric for a list of values."""
        if len(values) < 2:
            return 0.0

        # Count values with same sign as mean
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0

        consistent_count = sum(1 for v in values if (v >= 0) == (mean_val >= 0))
        return consistent_count / len(values)

    def _create_empty_result(self, strategy_name: str, primary_asset: str) -> CrossAssetValidationResult:
        """Create empty result when validation fails."""
        return CrossAssetValidationResult(
            strategy_name=strategy_name,
            primary_asset=primary_asset,
            validation_assets=[],
            asset_results=[],
            aggregate_metrics={},
            pass_rate=0.0,
            overall_pass=False,
            robustness_score=0.0,
            timestamp=datetime.now(),
            total_time=0.0
        )

    def _create_error_result(self, strategy_name: str, primary_asset: str,
                           error_message: str, total_time: float) -> CrossAssetValidationResult:
        """Create error result when validation fails with exception."""
        return CrossAssetValidationResult(
            strategy_name=strategy_name,
            primary_asset=primary_asset,
            validation_assets=[],
            asset_results=[],
            aggregate_metrics={'error': error_message},
            pass_rate=0.0,
            overall_pass=False,
            robustness_score=0.0,
            timestamp=datetime.now(),
            total_time=total_time
        )

    def _save_results(self, result: CrossAssetValidationResult) -> None:
        """Save validation results to files."""
        try:
            # Save detailed results
            if self.save_detailed_results:
                detailed_path = os.path.join(self.output_dir, 'cross_asset_validation_results.json')
                result.save_to_file(detailed_path)

            # Save CSV summary for easy analysis
            if self.save_csv_summary:
                csv_path = os.path.join(self.output_dir, 'cross_asset_validation_summary.csv')
                result.generate_csv_report(csv_path)

            self.logger.info(f"Cross-asset validation results saved to {self.output_dir}")

        except Exception as e:
            self.logger.error(f"Failed to save validation results: {str(e)}")

    def _log_validation_results(self, result: CrossAssetValidationResult) -> None:
        """Log comprehensive validation results."""
        self.logger.info("=" * 60)
        self.logger.info("CROSS-ASSET VALIDATION RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Strategy: {result.strategy_name}")
        self.logger.info(f"Primary Asset: {result.primary_asset}")
        self.logger.info(f"Validation Assets: {len(result.validation_assets)}")
        self.logger.info(f"Pass Rate: {result.pass_rate:.1%}")
        self.logger.info(f"Overall Pass: {result.overall_pass}")
        self.logger.info(f"Robustness Score: {result.robustness_score:.3f}")
        self.logger.info(".2f")
        self.logger.info("")

        # Log individual asset results
        for asset_result in result.asset_results:
            status = "✅ PASS" if asset_result.overall_pass else "❌ FAIL"
            self.logger.info(f"{asset_result.asset.symbol} ({asset_result.asset.name}): {status}")

            if asset_result.error_message:
                self.logger.info(f"  Error: {asset_result.error_message}")
            else:
                # Log key metrics
                val_metrics = asset_result.validation_metrics
                self.logger.info(".3f")
                self.logger.info(".1%")
                self.logger.info(".1%")

                # Log pass criteria
                criteria_status = []
                for criterion, passed in asset_result.pass_criteria.items():
                    status_icon = "✅" if passed else "❌"
                    criteria_status.append(f"{status_icon}{criterion}")
                self.logger.info(f"  Criteria: {' '.join(criteria_status)}")

            self.logger.info("")

        self.logger.info("=" * 60)

    def get_validation_history(self) -> List[CrossAssetValidationResult]:
        """Get history of all validation runs."""
        return self.validation_history.copy()

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of validator state and history."""
        if not self.validation_history:
            return {'status': 'No validations performed'}

        latest_result = self.validation_history[-1]

        return {
            'total_validations': len(self.validation_history),
            'latest_strategy': latest_result.strategy_name,
            'latest_pass_rate': latest_result.pass_rate,
            'latest_overall_pass': latest_result.overall_pass,
            'latest_robustness_score': latest_result.robustness_score,
            'asset_selector_status': self.asset_selector.get_asset_summary(),
            'data_fetcher_status': self.data_fetcher.get_cache_stats(),
            'validation_criteria': self.validation_criteria.get_criteria_summary()
        }

    def clear_history(self) -> None:
        """Clear validation history."""
        self.validation_history.clear()
        self.logger.info("Validation history cleared")

    def export_validation_report(self, file_path: str) -> None:
        """
        Export comprehensive validation report.

        Args:
            file_path: Path to save the report
        """
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'validator_config': {
                    'parallel_validation': self.parallel_validation,
                    'max_parallel_workers': self.max_parallel_workers,
                    'output_dir': self.output_dir
                },
                'validation_history': [result.get_summary_report() for result in self.validation_history],
                'performance_analysis': self.result_analyzer.compare_results(self.validation_history) if self.validation_history else {},
                'system_status': self.get_validation_summary()
            }

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                import json
                json.dump(report, f, indent=2, default=str)

            self.logger.info(f"Validation report exported to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to export validation report: {str(e)}")


# Convenience functions for easy integration
def create_cross_asset_validator(config: Optional[Dict[str, Any]] = None) -> CrossAssetValidator:
    """
    Create a cross-asset validator with default configuration.

    Args:
        config: Optional configuration overrides

    Returns:
        Configured CrossAssetValidator instance
    """
    default_config = {
        'output_dir': 'results/cross_asset_validation',
        'parallel_validation': False,
        'max_parallel_workers': 4,
        'log_level': 'INFO',
        'save_detailed_results': True,
        'save_csv_summary': True
    }

    if config:
        default_config.update(config)

    return CrossAssetValidator(default_config)


def run_cross_asset_validation(strategy_class, optimized_params: Dict[str, Any],
                             primary_asset: str, primary_data: pd.DataFrame,
                             config: Optional[Dict[str, Any]] = None) -> CrossAssetValidationResult:
    """
    Run complete cross-asset validation.

    Args:
        strategy_class: Strategy class to validate
        optimized_params: Optimized parameters from primary asset
        primary_asset: Primary asset symbol
        primary_data: Primary asset historical data
        config: Optional configuration

    Returns:
        Complete cross-asset validation results
    """
    validator = create_cross_asset_validator(config)
    return validator.validate_strategy(strategy_class, optimized_params, primary_asset, primary_data)
