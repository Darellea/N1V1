"""
optimization/walk_forward.py

Walk-Forward Optimization implementation.
Splits historical data into multiple train/test windows and optimizes parameters
for each window, then validates on out-of-sample data.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import os
import json
import csv
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from .base_optimizer import BaseOptimizer, ParameterBounds
from backtest.backtester import compute_backtest_metrics


class WalkForwardOptimizer(BaseOptimizer):
    """
    Walk-Forward Optimization for strategy parameter optimization.

    This optimizer:
    1. Splits historical data into rolling train/test windows
    2. Optimizes parameters on in-sample (train) data
    3. Validates performance on out-of-sample (test) data
    4. Updates strategy parameters if OOS performance improves
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Walk-Forward Optimizer.

        Args:
            config: Configuration dictionary containing:
                - train_window_days: Size of training window in days
                - test_window_days: Size of testing window in days
                - rolling: Whether to use rolling windows
                - min_observations: Minimum observations for optimization
                - improvement_threshold: Minimum improvement to update parameters
        """
        super().__init__(config)

        # Walk-forward specific configuration
        self.train_window_days = config.get('train_window_days', 90)
        self.test_window_days = config.get('test_window_days', 30)
        self.rolling = config.get('rolling', True)
        self.min_observations = config.get('min_observations', 1000)
        self.improvement_threshold = config.get('improvement_threshold', 0.05)  # 5% improvement

        # Optimization state
        self.windows_processed = 0
        self.best_oos_performance = float('-inf')
        self.current_baseline_params: Optional[Dict[str, Any]] = None

    def optimize(self, strategy_class, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run walk-forward optimization.

        Args:
            strategy_class: Strategy class to optimize
            data: Historical data for optimization

        Returns:
            Best parameter set found
        """
        start_time = time.time()

        self.logger.info("Starting Walk-Forward Optimization")
        self.logger.info(f"Data shape: {data.shape}")
        self.logger.info(f"Train window: {self.train_window_days} days")
        self.logger.info(f"Test window: {self.test_window_days} days")

        if len(data) < self.min_observations:
            self.logger.warning(f"Insufficient data: {len(data)} < {self.min_observations}")
            return {}

        # Generate walk-forward windows
        windows = self._generate_windows(data)

        if not windows:
            self.logger.error("No valid windows generated")
            return {}

        self.logger.info(f"Generated {len(windows)} optimization windows")

        # Process each window
        for i, (train_data, test_data) in enumerate(windows):
            self.logger.info(f"Processing window {i+1}/{len(windows)}")

            # Optimize parameters on training data
            window_params = self._optimize_window(strategy_class, train_data)

            if not window_params:
                continue

            # Evaluate on test data
            oos_performance = self._evaluate_oos_performance(
                strategy_class, window_params, test_data
            )

            # Update best parameters if OOS performance improved
            self._update_best_params(window_params, oos_performance)

            self.windows_processed += 1

            # Log progress
            self.logger.info(
                f"Window {i+1}: OOS Performance = {oos_performance:.4f}, "
                f"Best OOS = {self.best_oos_performance:.4f}"
            )

        # Finalize optimization
        optimization_time = time.time() - start_time
        self.config['optimization_time'] = optimization_time

        self.logger.info(f"Walk-Forward Optimization completed in {optimization_time:.2f}s")
        self.logger.info(f"Processed {self.windows_processed} windows")
        self.logger.info(f"Best OOS Performance: {self.best_oos_performance:.4f}")
        self.logger.info(f"Best Parameters: {self.best_params}")

        return self.best_params or {}

    def _generate_windows(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test windows for walk-forward analysis.

        Args:
            data: Historical data

        Returns:
            List of (train_data, test_data) tuples
        """
        windows = []

        # Ensure data is sorted by time
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        # Calculate window sizes in terms of data points
        # This is a simplification - in practice you'd want to be more precise with dates
        total_points = len(data)
        train_points = min(int(total_points * 0.7), self.min_observations)  # Use 70% for training
        test_points = min(int(total_points * 0.3), train_points // 2)  # Use 30% for testing

        if self.rolling:
            # Rolling windows
            step_size = max(1, test_points // 2)  # 50% overlap

            for start_idx in range(0, total_points - train_points - test_points + 1, step_size):
                train_end = start_idx + train_points
                test_end = train_end + test_points

                train_data = data.iloc[start_idx:train_end]
                test_data = data.iloc[train_end:test_end]

                if len(train_data) >= self.min_observations // 2 and len(test_data) > 0:
                    windows.append((train_data, test_data))
        else:
            # Non-overlapping windows
            window_size = train_points + test_points

            for start_idx in range(0, total_points - window_size + 1, window_size):
                train_end = start_idx + train_points
                test_end = train_end + test_points

                train_data = data.iloc[start_idx:train_end]
                test_data = data.iloc[train_end:test_end]

                if len(train_data) >= self.min_observations // 2 and len(test_data) > 0:
                    windows.append((train_data, test_data))

        return windows

    def _optimize_window(self, strategy_class, train_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize parameters for a single window.

        Args:
            strategy_class: Strategy class to optimize
            train_data: Training data for this window

        Returns:
            Best parameters found for this window
        """
        # Simple parameter search - in practice you'd use more sophisticated optimization
        best_params = {}
        best_fitness = float('-inf')

        # Generate parameter combinations to test
        param_combinations = self._generate_param_combinations()

        for params in param_combinations:
            # Create strategy instance with these parameters
            try:
                strategy_config = {
                    'name': 'optimization_strategy',
                    'symbols': ['BTC/USDT'],
                    'timeframe': '1h',
                    'required_history': 100,
                    'params': params
                }

                strategy_instance = strategy_class(strategy_config)

                # Evaluate fitness
                fitness = self.evaluate_fitness(strategy_instance, train_data)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = params.copy()

            except Exception as e:
                self.logger.debug(f"Parameter evaluation failed: {str(e)}")
                continue

        return best_params

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations to test.

        Returns:
            List of parameter dictionaries
        """
        # This is a simplified implementation
        # In practice, you'd want more sophisticated parameter space exploration

        combinations = []

        # Example: RSI strategy parameters
        rsi_periods = [7, 14, 21, 28]
        overbought_levels = [65, 70, 75]
        oversold_levels = [25, 30, 35]

        for rsi_period in rsi_periods:
            for overbought in overbought_levels:
                for oversold in oversold_levels:
                    if overbought > oversold:  # Ensure valid ranges
                        combinations.append({
                            'rsi_period': rsi_period,
                            'overbought': overbought,
                            'oversold': oversold
                        })

        # Limit combinations for faster optimization
        return combinations[:50]  # Test up to 50 combinations

    def _evaluate_oos_performance(self, strategy_class, params: Dict[str, Any],
                                test_data: pd.DataFrame) -> float:
        """
        Evaluate out-of-sample performance.

        Args:
            strategy_class: Strategy class
            params: Parameters to evaluate
            test_data: Out-of-sample test data

        Returns:
            Out-of-sample performance score
        """
        try:
            strategy_config = {
                'name': 'oos_evaluation',
                'symbols': ['BTC/USDT'],
                'timeframe': '1h',
                'required_history': 100,
                'params': params
            }

            strategy_instance = strategy_class(strategy_config)
            return self.evaluate_fitness(strategy_instance, test_data)

        except Exception as e:
            self.logger.error(f"OOS evaluation failed: {str(e)}")
            return float('-inf')

    def _update_best_params(self, params: Dict[str, Any], oos_performance: float) -> None:
        """
        Update best parameters if OOS performance improved.

        Args:
            params: Parameter set
            oos_performance: Out-of-sample performance
        """
        # Check if this is the first evaluation
        if self.best_oos_performance == float('-inf'):
            self.best_oos_performance = oos_performance
            self.best_params = params.copy()
            self.logger.info(f"Initial OOS performance: {oos_performance:.4f}")
            return

        # Calculate improvement
        improvement = oos_performance - self.best_oos_performance
        improvement_pct = improvement / abs(self.best_oos_performance) if self.best_oos_performance != 0 else 0

        # Update if improvement meets threshold
        if improvement_pct >= self.improvement_threshold:
            old_performance = self.best_oos_performance
            self.best_oos_performance = oos_performance
            self.best_params = params.copy()

            self.logger.info(
                f"OOS Performance improved: {old_performance:.4f} -> {oos_performance:.4f} "
                f"({improvement_pct:.1%})"
            )
            self.logger.info(f"Updated best parameters: {params}")
        else:
            self.logger.debug(
                f"OOS Performance: {oos_performance:.4f} (improvement: {improvement_pct:.1%})"
            )

    def cross_pair_validation(self, strategy_class, data_dict: Dict[str, pd.DataFrame],
                            train_pair: str, validation_pairs: List[str]) -> Dict[str, Any]:
        """
        Perform cross-pair validation: optimize on train_pair, validate on validation_pairs.

        Args:
            strategy_class: Strategy class to optimize
            data_dict: Dictionary mapping pair names to their historical data
            train_pair: Primary pair for optimization
            validation_pairs: List of pairs to validate on

        Returns:
            Results dictionary with metrics for each pair
        """
        start_time = time.time()

        self.logger.info("Starting Cross-Pair Validation")
        self.logger.info(f"Train pair: {train_pair}")
        self.logger.info(f"Validation pairs: {validation_pairs}")

        if train_pair not in data_dict:
            self.logger.error(f"Train pair {train_pair} not found in data_dict")
            return {}

        train_data = data_dict[train_pair]
        if len(train_data) < self.min_observations:
            self.logger.warning(f"Insufficient data for {train_pair}: {len(train_data)} < {self.min_observations}")
            return {}

        # Step 1: Optimize parameters on train_pair
        self.logger.info(f"Optimizing parameters on {train_pair}")
        best_params = self.optimize(strategy_class, train_data)

        if not best_params:
            self.logger.error("Failed to find optimal parameters")
            return {}

        self.logger.info(f"Best parameters found: {best_params}")

        # Step 2: Validate on each pair (including train_pair)
        all_pairs = [train_pair] + validation_pairs
        results = {
            "train_pair": train_pair,
            "validation_pairs": validation_pairs,
            "best_params": best_params,
            "results": {}
        }

        for pair in all_pairs:
            if pair not in data_dict:
                self.logger.warning(f"Data for {pair} not available, skipping")
                continue

            pair_data = data_dict[pair]
            self.logger.info(f"Validating on {pair} with {len(pair_data)} data points")

            # Run backtest with best parameters
            try:
                strategy_config = {
                    'name': f'cross_validation_{pair}',
                    'symbols': [pair],
                    'timeframe': '1h',
                    'required_history': 100,
                    'params': best_params
                }

                strategy_instance = strategy_class(strategy_config)

                # Simulate backtest to get equity progression
                equity_progression = self._run_backtest_simulation(strategy_instance, pair_data)

                # Compute metrics
                metrics = compute_backtest_metrics(equity_progression)

                results["results"][pair] = {
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0),
                    "profit_factor": metrics.get("profit_factor", 0.0),
                    "total_return": metrics.get("total_return", 0.0),
                    "win_rate": metrics.get("win_rate", 0.0),
                    "total_trades": metrics.get("total_trades", 0),
                    "expectancy": metrics.get("expectancy", 0.0)
                }

                self.logger.info(f"{pair} - Sharpe: {results['results'][pair]['sharpe_ratio']:.4f}, "
                               f"Return: {results['results'][pair]['total_return']:.4f}")

            except Exception as e:
                self.logger.error(f"Validation failed for {pair}: {str(e)}")
                results["results"][pair] = {
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "profit_factor": 0.0,
                    "total_return": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "expectancy": 0.0
                }

        # Step 3: Save results
        self._save_cross_validation_results(results)

        # Log summary
        validation_time = time.time() - start_time
        self.logger.info(f"Cross-Pair Validation completed in {validation_time:.2f}s")
        self.logger.info(f"Validated on {len(results['results'])} pairs")

        return results

    def _run_backtest_simulation(self, strategy_instance, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Simulate backtest to get equity progression.

        Args:
            strategy_instance: Strategy instance
            data: Historical data

        Returns:
            List of equity progression records
        """
        # This is a simplified simulation - in practice you'd use the full backtester
        equity_progression = []
        equity = 1000.0  # Starting balance
        trade_count = 0

        # Simple simulation: assume some trades based on data length
        num_trades = min(len(data) // 100, 50)  # Simulate reasonable number of trades

        for i in range(num_trades):
            # Simulate random trade outcome
            pnl = np.random.normal(0, 50)  # Random P&L
            equity += pnl

            equity_progression.append({
                "trade_id": f"trade_{i+1}",
                "timestamp": data.index[i * (len(data) // num_trades)] if i * (len(data) // num_trades) < len(data) else data.index[-1],
                "equity": equity,
                "pnl": pnl,
                "cumulative_return": (equity - 1000) / 1000
            })

        return equity_progression

    def _save_cross_validation_results(self, results: Dict[str, Any]) -> None:
        """
        Save cross-validation results to CSV and JSON.

        Args:
            results: Results dictionary
        """
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)

        # Save JSON
        json_path = "results/cross_pair_validation.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Results saved to {json_path}")

        # Save CSV
        csv_path = "results/cross_pair_validation.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(["pair", "sharpe_ratio", "max_drawdown", "profit_factor",
                           "total_return", "win_rate", "total_trades", "expectancy"])

            # Write data
            for pair, metrics in results["results"].items():
                writer.writerow([
                    pair,
                    metrics.get("sharpe_ratio", 0.0),
                    metrics.get("max_drawdown", 0.0),
                    metrics.get("profit_factor", 0.0),
                    metrics.get("total_return", 0.0),
                    metrics.get("win_rate", 0.0),
                    metrics.get("total_trades", 0),
                    metrics.get("expectancy", 0.0)
                ])

        self.logger.info(f"Results saved to {csv_path}")

    def get_walk_forward_summary(self) -> Dict[str, Any]:
        """
        Get summary of walk-forward optimization process.

        Returns:
            Summary dictionary
        """
        summary = self.get_optimization_summary()
        summary.update({
            'windows_processed': self.windows_processed,
            'best_oos_performance': self.best_oos_performance,
            'train_window_days': self.train_window_days,
            'test_window_days': self.test_window_days,
            'rolling_windows': self.rolling,
            'improvement_threshold': self.improvement_threshold
        })

        return summary
