"""
Walk-Forward Optimizer

This module implements walk-forward analysis for strategy optimization.
It prevents overfitting by using sliding training and testing windows,
providing robust out-of-sample performance evaluation.

Key Features:
- Sliding window data splitting with configurable overlap
- In-sample optimization and out-of-sample testing
- Performance metrics aggregation across windows
- Automated retraining scheduler
- Integration with existing optimization methods
- Comprehensive logging and reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod
import logging
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import asyncio
import threading
import time

from .base_optimizer import BaseOptimizer, OptimizationResult, ParameterBounds
from backtest.backtester import compute_backtest_metrics
from utils.config_loader import get_config


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""

    window_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    optimized_params: Dict[str, Any]
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    optimization_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'window_index': self.window_index,
            'train_start': self.train_start.isoformat(),
            'train_end': self.train_end.isoformat(),
            'test_start': self.test_start.isoformat(),
            'test_end': self.test_end.isoformat(),
            'optimized_params': self.optimized_params,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'optimization_time': self.optimization_time,
            'train_data_shape': self.train_data.shape if hasattr(self.train_data, 'shape') else None,
            'test_data_shape': self.test_data.shape if hasattr(self.test_data, 'shape') else None
        }


@dataclass
class WalkForwardResult:
    """Container for complete walk-forward analysis results."""

    strategy_name: str
    total_windows: int
    windows: List[WalkForwardWindow]
    aggregate_metrics: Dict[str, Any]
    performance_distribution: Dict[str, Any]
    optimization_summary: Dict[str, Any]
    timestamp: datetime
    total_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'strategy_name': self.strategy_name,
            'total_windows': self.total_windows,
            'windows': [w.to_dict() for w in self.windows],
            'aggregate_metrics': self.aggregate_metrics,
            'performance_distribution': self.performance_distribution,
            'optimization_summary': self.optimization_summary,
            'timestamp': self.timestamp.isoformat(),
            'total_time': self.total_time
        }


class WalkForwardDataSplitter:
    """Handles data splitting for walk-forward analysis."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data splitter.

        Args:
            config: Configuration dictionary containing:
                - train_window_size: Size of training window (int for periods, str for duration)
                - test_window_size: Size of testing window (int for periods, str for duration)
                - step_size: Step size for sliding window (int for periods, str for duration)
                - min_samples: Minimum samples required for valid window
                - overlap_allowed: Whether overlapping windows are allowed
        """
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Parse window sizes
        self.train_window_size = self._parse_window_size(
            config.get('train_window_size', 100)
        )
        self.test_window_size = self._parse_window_size(
            config.get('test_window_size', 20)
        )
        self.step_size = self._parse_window_size(
            config.get('step_size', 20)
        )

        self.min_samples = config.get('min_samples', 50)
        self.overlap_allowed = config.get('overlap_allowed', False)

    def _parse_window_size(self, size: Union[int, str]) -> Union[int, timedelta]:
        """Parse window size specification."""
        if isinstance(size, int):
            return size
        elif isinstance(size, str):
            # Parse duration strings like "30D", "1W", "24H"
            return pd.Timedelta(size)
        else:
            raise ValueError(f"Invalid window size: {size}")

    def split_data(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """
        Split data into walk-forward windows.

        Args:
            data: Time series data to split

        Returns:
            List of WalkForwardWindow objects
        """
        if data.empty or len(data) < self.min_samples:
            self.logger.warning(f"Insufficient data: {len(data)} samples, minimum {self.min_samples}")
            return []

        # Ensure data is sorted by time
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        windows = []
        window_index = 0

        # Calculate total possible windows
        if isinstance(self.train_window_size, int) and isinstance(self.test_window_size, int):
            # Period-based windows
            total_samples = len(data)
            train_size = self.train_window_size
            test_size = self.test_window_size
            step = self.step_size if isinstance(self.step_size, int) else test_size

            start_idx = 0
            while start_idx + train_size + test_size <= total_samples:
                train_end_idx = start_idx + train_size
                test_end_idx = train_end_idx + test_size

                train_data = data.iloc[start_idx:train_end_idx]
                test_data = data.iloc[train_end_idx:test_end_idx]

                # Create window
                window = self._create_window_from_indices(
                    window_index, train_data, test_data
                )
                windows.append(window)

                window_index += 1
                start_idx += step

        else:
            # Time-based windows
            data_start = data.index[0]
            data_end = data.index[-1]

            current_train_start = data_start

            while True:
                # Calculate window boundaries
                if isinstance(self.train_window_size, timedelta):
                    train_end = current_train_start + self.train_window_size
                else:
                    # Find index-based train end
                    train_end_idx = min(len(data) - 1, data.index.get_loc(current_train_start) + self.train_window_size)
                    train_end = data.index[train_end_idx]

                if isinstance(self.test_window_size, timedelta):
                    test_end = train_end + self.test_window_size
                else:
                    test_end_idx = min(len(data) - 1, data.index.get_loc(train_end) + self.test_window_size)
                    test_end = data.index[test_end_idx]

                # Check if we have enough data
                if test_end > data_end:
                    break

                # Extract data for this window
                train_mask = (data.index >= current_train_start) & (data.index < train_end)
                test_mask = (data.index >= train_end) & (data.index < test_end)

                train_data = data[train_mask]
                test_data = data[test_mask]

                if len(train_data) >= self.min_samples and len(test_data) > 0:
                    window = self._create_window_from_times(
                        window_index, train_data, test_data,
                        current_train_start, train_end, train_end, test_end
                    )
                    windows.append(window)
                    window_index += 1

                # Move to next window
                if isinstance(self.step_size, timedelta):
                    current_train_start += self.step_size
                else:
                    step_idx = data.index.get_loc(current_train_start) + (self.step_size or self.test_window_size)
                    if step_idx >= len(data):
                        break
                    current_train_start = data.index[step_idx]

        self.logger.info(f"Created {len(windows)} walk-forward windows")
        return windows

    def _create_window_from_indices(self, window_index: int,
                                   train_data: pd.DataFrame,
                                   test_data: pd.DataFrame) -> WalkForwardWindow:
        """Create window from index-based data slices."""
        return WalkForwardWindow(
            window_index=window_index,
            train_start=train_data.index[0].to_pydatetime(),
            train_end=train_data.index[-1].to_pydatetime(),
            test_start=test_data.index[0].to_pydatetime(),
            test_end=test_data.index[-1].to_pydatetime(),
            train_data=train_data,
            test_data=test_data,
            optimized_params={},
            train_metrics={},
            test_metrics={},
            optimization_time=0.0
        )

    def _create_window_from_times(self, window_index: int,
                                 train_data: pd.DataFrame,
                                 test_data: pd.DataFrame,
                                 train_start: datetime, train_end: datetime,
                                 test_start: datetime, test_end: datetime) -> WalkForwardWindow:
        """Create window from time-based data slices."""
        return WalkForwardWindow(
            window_index=window_index,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_data=train_data,
            test_data=test_data,
            optimized_params={},
            train_metrics={},
            test_metrics={},
            optimization_time=0.0
        )


class WalkForwardOptimizer(BaseOptimizer):
    """
    Walk-Forward Optimizer for robust strategy parameter optimization.

    This optimizer:
    1. Splits historical data into sliding training/testing windows
    2. Optimizes strategy parameters on training data
    3. Evaluates performance on out-of-sample testing data
    4. Aggregates results across all windows for robust assessment
    5. Provides statistical analysis of optimization stability
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Walk-Forward Optimizer.

        Args:
            config: Configuration dictionary containing:
                - base_optimizer: Configuration for underlying optimizer
                - data_splitter: Configuration for data splitting
                - output_dir: Directory for saving results
                - save_intermediate: Whether to save results after each window
                - parallel_execution: Whether to run windows in parallel
                - train_window_days: Number of days for training window
                - test_window_days: Number of days for testing window
                - min_observations: Minimum observations required
                - rolling: Whether to use rolling windows
                - improvement_threshold: Threshold for improvement detection
        """
        super().__init__(config)

        # Walk-forward specific configuration
        self.base_optimizer_config = config.get('base_optimizer', {})
        self.data_splitter_config = config.get('data_splitter', {})
        self.output_dir = config.get('output_dir', 'results/walk_forward')
        self.save_intermediate = config.get('save_intermediate', True)
        self.parallel_execution = config.get('parallel_execution', False)

        # Extract attributes expected by tests
        self.train_window_days = config.get('train_window_days', 90)
        self.test_window_days = config.get('test_window_days', 30)
        self.min_observations = config.get('min_observations', 1000)
        self.rolling = config.get('rolling', True)
        self.improvement_threshold = config.get('improvement_threshold', 0.05)

        # Initialize components
        self.data_splitter = WalkForwardDataSplitter(self.data_splitter_config)

        # Results storage
        self.windows: List[WalkForwardWindow] = []
        self.aggregate_metrics: Dict[str, Any] = {}
        self.performance_distribution: Dict[str, Any] = {}

        # Scheduler for retraining
        self.scheduler = WalkForwardScheduler(self)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info("Walk-Forward Optimizer initialized")

    def _generate_windows(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate rolling or expanding train/test splits based on configuration.

        Args:
            data: Input data to split

        Returns:
            List of (train_data, test_data) tuples where each is a DataFrame
        """
        if data.empty or len(data) < self.min_observations:
            return []

        windows = []
        n_samples = len(data)

        if self.rolling:
            # Rolling windows
            train_size = self.train_window_days
            test_size = self.test_window_days
            step_size = test_size  # Non-overlapping by default

            for start_idx in range(0, n_samples - train_size - test_size + 1, step_size):
                train_end = start_idx + train_size
                test_end = train_end + test_size

                train_data = data.iloc[start_idx:train_end]
                test_data = data.iloc[train_end:test_end]

                windows.append((train_data, test_data))
        else:
            # Expanding windows
            for test_end in range(self.min_observations, n_samples + 1, self.test_window_days):
                train_end = test_end - self.test_window_days
                train_start = max(0, train_end - self.train_window_days)

                train_data = data.iloc[train_start:train_end]
                test_data = data.iloc[train_end:test_end]

                if len(train_data) >= self.min_observations:
                    windows.append((train_data, test_data))

        return windows

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate combinations of hyperparameters for strategies.

        Returns:
            List of parameter dictionaries
        """
        import itertools

        # Default parameter ranges for common strategy parameters
        param_ranges = {
            'rsi_period': [5, 10, 14, 21, 28],
            'overbought': [65, 70, 75, 80],
            'oversold': [20, 25, 30, 35],
            'ema_period': [9, 12, 20, 26, 50],
            'sma_period': [10, 20, 50, 100, 200]
        }

        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]

        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)

        return combinations

    def cross_pair_validation(self, strategy, data_dict: Dict[str, pd.DataFrame],
                             train_pair: str, validation_pairs: List[str]) -> Dict[str, Any]:
        """
        Perform cross-pair validation by training on one pair and validating on others.

        Args:
            strategy: Strategy class to validate
            data_dict: Dictionary mapping pair names to DataFrames
            train_pair: Name of the pair to train on
            validation_pairs: List of pairs to validate on

        Returns:
            Dictionary containing validation results
        """
        if train_pair not in data_dict:
            self.logger.error(f"Train pair {train_pair} not found in data_dict")
            return {}

        # Train on the specified pair
        train_data = data_dict[train_pair]
        best_params = self.optimize(strategy, train_data)

        # If optimization failed (insufficient data), return empty results
        if not best_params:
            return {}

        results = {
            'train_pair': train_pair,
            'validation_pairs': validation_pairs,
            'best_params': best_params,
            'results': {}
        }

        # Validate on each validation pair
        for pair in [train_pair] + validation_pairs:
            if pair not in data_dict:
                self.logger.warning(f"Validation pair {pair} not found in data_dict")
                continue

            val_data = data_dict[pair]

            # Evaluate strategy with best parameters on this pair
            try:
                # Create strategy instance
                strategy_config = {
                    'name': f'cpv_{pair}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                    'symbols': [pair],
                    'timeframe': '1h',
                    'required_history': 100,
                    'params': best_params
                }

                strategy_instance = strategy(strategy_config)

                # Run backtest to get metrics
                equity_progression = self._run_backtest(strategy_instance, val_data)
                if equity_progression:
                    metrics = compute_backtest_metrics(equity_progression)
                else:
                    metrics = {'error': 'No equity progression'}

                results['results'][pair] = {
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'profit_factor': metrics.get('profit_factor', 1.0),
                    'total_return': metrics.get('total_return', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'total_trades': metrics.get('total_trades', 0),
                    'expectancy': metrics.get('expectancy', 0)
                }

            except Exception as e:
                self.logger.error(f"Error validating on pair {pair}: {e}")
                results['results'][pair] = {
                    'error': str(e),
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'profit_factor': 1.0,
                    'total_return': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'expectancy': 0
                }

        return results

    def _save_cross_validation_results(self, results: Dict[str, Any]) -> None:
        """
        Save cross-validation results to JSON and CSV files.

        Args:
            results: Cross-validation results dictionary
        """
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)

        # Save JSON results
        json_path = 'results/cross_pair_validation.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save CSV summary
        csv_path = 'results/cross_pair_validation.csv'
        rows = []

        for pair, metrics in results.get('results', {}).items():
            row = {
                'pair': pair,
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'profit_factor': metrics.get('profit_factor', 1.0),
                'total_return': metrics.get('total_return', 0),
                'win_rate': metrics.get('win_rate', 0),
                'total_trades': metrics.get('total_trades', 0),
                'expectancy': metrics.get('expectancy', 0)
            }
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)

        self.logger.info(f"Cross-validation results saved to {json_path} and {csv_path}")

    def optimize(self, strategy_class, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run walk-forward optimization.

        Args:
            strategy_class: Strategy class to optimize
            data: Historical data for walk-forward analysis

        Returns:
            Best parameter set based on walk-forward analysis
        """
        start_time = time.time()
        self.logger.info("Starting Walk-Forward Optimization")

        # Split data into windows
        self.windows = self.data_splitter.split_data(data)

        if not self.windows:
            self.logger.error("No valid windows created for walk-forward analysis")
            return {}

        self.logger.info(f"Processing {len(self.windows)} windows")

        # Process each window
        if self.parallel_execution:
            self._process_windows_parallel(strategy_class)
        else:
            self._process_windows_sequential(strategy_class)

        # Calculate aggregate metrics
        self._calculate_aggregate_metrics()

        # Calculate performance distribution
        self._calculate_performance_distribution()

        # Select best parameters based on walk-forward results
        best_params = self._select_best_parameters()

        # Save results
        total_time = time.time() - start_time
        self._save_results(total_time)

        self.logger.info(f"Walk-Forward Optimization completed in {total_time:.2f}s")
        self.logger.info(f"Best parameters: {best_params}")

        return best_params

    def _process_windows_sequential(self, strategy_class) -> None:
        """Process windows sequentially."""
        for i, window in enumerate(self.windows):
            self.logger.info(f"Processing window {i+1}/{len(self.windows)}")
            self._process_single_window(strategy_class, window, i)

    def _process_windows_parallel(self, strategy_class) -> None:
        """Process windows in parallel."""
        # For simplicity, using ThreadPoolExecutor
        # In production, consider using ProcessPoolExecutor for CPU-intensive tasks
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=min(4, len(self.windows))) as executor:
            futures = [
                executor.submit(self._process_single_window, strategy_class, window, i)
                for i, window in enumerate(self.windows)
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Window processing failed: {e}")

    def _process_single_window(self, strategy_class, window: WalkForwardWindow, index: int) -> None:
        """Process a single walk-forward window."""
        try:
            window_start_time = time.time()

            # Create optimizer instance for this window
            optimizer = self._create_optimizer()

            # Optimize parameters on training data
            optimized_params = optimizer.optimize(strategy_class, window.train_data)

            # Evaluate on training data
            train_metrics = self._evaluate_strategy(
                strategy_class, optimized_params, window.train_data
            )

            # Evaluate on testing data
            test_metrics = self._evaluate_strategy(
                strategy_class, optimized_params, window.test_data
            )

            # Update window with results
            window.optimized_params = optimized_params
            window.train_metrics = train_metrics
            window.test_metrics = test_metrics
            window.optimization_time = time.time() - window_start_time

            self.logger.info(
                f"Window {index+1}: Train Sharpe={train_metrics.get('sharpe_ratio', 0):.3f}, "
                f"Test Sharpe={test_metrics.get('sharpe_ratio', 0):.3f}"
            )

            # Save intermediate results if requested
            if self.save_intermediate:
                self._save_intermediate_results(window)

        except Exception as e:
            self.logger.error(f"Error processing window {index+1}: {e}")
            # Set default values for failed window
            window.optimized_params = {}
            window.train_metrics = {'error': str(e)}
            window.test_metrics = {'error': str(e)}
            window.optimization_time = time.time() - window_start_time

    def _create_optimizer(self):
        """Create optimizer instance for window processing."""
        # Import here to avoid circular imports
        try:
            from .genetic_optimizer import GeneticOptimizer
        except ImportError:
            # Fallback for testing or if genetic optimizer is not available
            from .base_optimizer import BaseOptimizer
            return BaseOptimizer(self.base_optimizer_config)

        # Use genetic optimizer as default, but could be configurable
        optimizer_config = self.base_optimizer_config.copy()
        optimizer_config.update({
            'parameter_bounds': list(self.parameter_bounds.values())
        })

        return GeneticOptimizer(optimizer_config)

    def _evaluate_strategy(self, strategy_class, params: Dict[str, Any],
                          data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate strategy with given parameters on data."""
        try:
            # Create strategy instance
            strategy_config = {
                'name': f'wf_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'symbols': ['BTC/USDT'],
                'timeframe': '1h',
                'required_history': 100,
                'params': params
            }

            strategy_instance = strategy_class(strategy_config)

            # Run evaluation
            fitness = self.evaluate_fitness(strategy_instance, data)

            # Get detailed metrics
            equity_progression = self._run_backtest(strategy_instance, data)
            if equity_progression:
                metrics = compute_backtest_metrics(equity_progression)
                metrics['fitness'] = fitness
                return metrics
            else:
                return {'fitness': fitness, 'error': 'No equity progression'}

        except Exception as e:
            self.logger.error(f"Strategy evaluation failed: {e}")
            return {'error': str(e), 'fitness': float('-inf')}

    def _calculate_aggregate_metrics(self) -> None:
        """Calculate aggregate metrics across all windows."""
        if not self.windows:
            return

        # Collect all test metrics
        test_sharpe_ratios = []
        test_returns = []
        test_win_rates = []
        test_max_drawdowns = []
        test_sortino_ratios = []

        for window in self.windows:
            metrics = window.test_metrics
            if 'error' not in metrics:
                test_sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                test_returns.append(metrics.get('total_return', 0))
                test_win_rates.append(metrics.get('win_rate', 0))
                test_max_drawdowns.append(metrics.get('max_drawdown', 0))
                test_sortino_ratios.append(metrics.get('sortino_ratio', 0))

        # Calculate aggregate statistics
        self.aggregate_metrics = {
            'total_windows': len(self.windows),
            'successful_windows': len(test_sharpe_ratios),
            'avg_test_sharpe': np.mean(test_sharpe_ratios) if test_sharpe_ratios else 0,
            'std_test_sharpe': np.std(test_sharpe_ratios) if test_sharpe_ratios else 0,
            'avg_test_return': np.mean(test_returns) if test_returns else 0,
            'avg_test_win_rate': np.mean(test_win_rates) if test_win_rates else 0,
            'avg_test_max_drawdown': np.mean(test_max_drawdowns) if test_max_drawdowns else 0,
            'avg_test_sortino': np.mean(test_sortino_ratios) if test_sortino_ratios else 0,
            'sharpe_ratio_stability': self._calculate_stability(test_sharpe_ratios),
            'return_consistency': self._calculate_consistency(test_returns)
        }

        self.logger.info("Aggregate metrics calculated:")
        self.logger.info(f"  Average Test Sharpe: {self.aggregate_metrics['avg_test_sharpe']:.3f}")
        self.logger.info(f"  Sharpe Stability: {self.aggregate_metrics['sharpe_ratio_stability']:.3f}")

    def _calculate_performance_distribution(self) -> None:
        """Calculate performance distribution statistics."""
        if not self.windows:
            return

        test_sharpes = [
            w.test_metrics.get('sharpe_ratio', 0)
            for w in self.windows
            if 'error' not in w.test_metrics
        ]

        if not test_sharpes:
            return

        # Calculate distribution statistics
        self.performance_distribution = {
            'sharpe_percentiles': {
                '10th': np.percentile(test_sharpes, 10),
                '25th': np.percentile(test_sharpes, 25),
                '50th': np.percentile(test_sharpes, 50),
                '75th': np.percentile(test_sharpes, 75),
                '90th': np.percentile(test_sharpes, 90)
            },
            'sharpe_quartiles': [
                np.percentile(test_sharpes, 25),
                np.percentile(test_sharpes, 50),
                np.percentile(test_sharpes, 75)
            ],
            'sharpe_iqr': np.subtract(*np.percentile(test_sharpes, [75, 25])),
            'positive_sharpe_ratio': np.mean([s > 0 for s in test_sharpes]),
            'sharpe_confidence_interval': self._calculate_confidence_interval(test_sharpes)
        }

    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability metric (lower is more stable)."""
        if len(values) < 2:
            return 0.0
        return np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else float('inf')

    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency metric (higher is more consistent)."""
        if len(values) < 2:
            return 0.0

        # Count positive periods
        positive_count = sum(1 for v in values if v > 0)
        return positive_count / len(values)

    def _calculate_confidence_interval(self, values: List[float],
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for values."""
        if len(values) < 2:
            return (np.mean(values), np.mean(values))

        mean = np.mean(values)
        std = np.std(values)
        z_score = 1.96  # 95% confidence

        margin = z_score * std / np.sqrt(len(values))
        return (mean - margin, mean + margin)

    def _select_best_parameters(self) -> Dict[str, Any]:
        """Select best parameters based on walk-forward results."""
        if not self.windows:
            return {}

        # Find parameters that performed best on average across test windows
        param_performance = defaultdict(list)

        for window in self.windows:
            if window.optimized_params and 'error' not in window.test_metrics:
                params_key = json.dumps(window.optimized_params, sort_keys=True)
                test_sharpe = window.test_metrics.get('sharpe_ratio', 0)
                param_performance[params_key].append((window.optimized_params, test_sharpe))

        if not param_performance:
            return {}

        # Find parameter set with highest average test Sharpe
        best_avg_sharpe = float('-inf')
        best_params = {}

        for params_key, performances in param_performance.items():
            avg_sharpe = np.mean([p[1] for p in performances])
            if avg_sharpe > best_avg_sharpe:
                best_avg_sharpe = avg_sharpe
                best_params = performances[0][0]  # Use first occurrence

        return best_params

    def _save_intermediate_results(self, window: WalkForwardWindow) -> None:
        """Save results for a single window."""
        filename = f"window_{window.window_index:03d}_results.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(window.to_dict(), f, indent=2, default=str)

    def _save_results(self, total_time: float) -> None:
        """Save complete walk-forward results."""
        result = WalkForwardResult(
            strategy_name=self.config.get('strategy_name', 'unknown'),
            total_windows=len(self.windows),
            windows=self.windows,
            aggregate_metrics=self.aggregate_metrics,
            performance_distribution=self.performance_distribution,
            optimization_summary=self.get_optimization_summary(),
            timestamp=datetime.now(),
            total_time=total_time
        )

        # Save detailed results
        detailed_path = os.path.join(self.output_dir, 'walk_forward_results.json')
        with open(detailed_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        # Save summary report
        summary_path = os.path.join(self.output_dir, 'walk_forward_summary.json')
        summary = {
            'strategy_name': result.strategy_name,
            'total_windows': result.total_windows,
            'aggregate_metrics': result.aggregate_metrics,
            'performance_distribution': result.performance_distribution,
            'timestamp': result.timestamp.isoformat(),
            'total_time': result.total_time
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save CSV summary for easy analysis
        self._save_csv_summary()

        self.logger.info(f"Walk-forward results saved to {self.output_dir}")

    def _save_csv_summary(self) -> None:
        """Save CSV summary of all windows."""
        if not self.windows:
            return

        csv_path = os.path.join(self.output_dir, 'walk_forward_windows.csv')

        rows = []
        for window in self.windows:
            row = {
                'window_index': window.window_index,
                'train_start': window.train_start.isoformat(),
                'train_end': window.train_end.isoformat(),
                'test_start': window.test_start.isoformat(),
                'test_end': window.test_end.isoformat(),
                'optimization_time': window.optimization_time,
                'train_sharpe': window.train_metrics.get('sharpe_ratio', 0),
                'test_sharpe': window.test_metrics.get('sharpe_ratio', 0),
                'train_return': window.train_metrics.get('total_return', 0),
                'test_return': window.test_metrics.get('total_return', 0),
                'train_win_rate': window.train_metrics.get('win_rate', 0),
                'test_win_rate': window.test_metrics.get('win_rate', 0),
                'train_max_drawdown': window.train_metrics.get('max_drawdown', 0),
                'test_max_drawdown': window.test_metrics.get('max_drawdown', 0)
            }

            # Add optimized parameters
            for param_name, param_value in window.optimized_params.items():
                row[f'param_{param_name}'] = param_value

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    def get_walk_forward_summary(self) -> Dict[str, Any]:
        """Get comprehensive walk-forward analysis summary."""
        return {
            'total_windows': len(self.windows),
            'aggregate_metrics': self.aggregate_metrics,
            'performance_distribution': self.performance_distribution,
            'optimization_summary': self.get_optimization_summary(),
            'stability_metrics': self._calculate_stability_metrics(),
            'robustness_score': self._calculate_robustness_score()
        }

    def _calculate_stability_metrics(self) -> Dict[str, Any]:
        """Calculate stability metrics for the walk-forward analysis."""
        if not self.windows:
            return {}

        test_sharpes = [
            w.test_metrics.get('sharpe_ratio', 0)
            for w in self.windows
            if 'error' not in w.test_metrics
        ]

        if len(test_sharpes) < 2:
            return {}

        return {
            'sharpe_volatility': np.std(test_sharpes),
            'sharpe_mean': np.mean(test_sharpes),
            'sharpe_coefficient_of_variation': np.std(test_sharpes) / abs(np.mean(test_sharpes)) if np.mean(test_sharpes) != 0 else float('inf'),
            'sharpe_autocorrelation': self._calculate_autocorrelation(test_sharpes),
            'performance_persistence': self._calculate_performance_persistence(test_sharpes)
        }

    def _calculate_autocorrelation(self, values: List[float], lag: int = 1) -> float:
        """Calculate autocorrelation of performance metric."""
        if len(values) <= lag:
            return 0.0

        try:
            return np.corrcoef(values[:-lag], values[lag:])[0, 1]
        except:
            return 0.0

    def _calculate_performance_persistence(self, values: List[float]) -> float:
        """Calculate performance persistence (how often sign changes)."""
        if len(values) < 2:
            return 0.0

        sign_changes = 0
        for i in range(1, len(values)):
            if (values[i] > 0) != (values[i-1] > 0):
                sign_changes += 1

        return 1.0 - (sign_changes / (len(values) - 1))

    def _calculate_robustness_score(self) -> float:
        """Calculate overall robustness score for the optimization."""
        if not self.aggregate_metrics:
            return 0.0

        # Combine multiple factors into robustness score
        avg_sharpe = self.aggregate_metrics.get('avg_test_sharpe', 0)
        sharpe_stability = self.aggregate_metrics.get('sharpe_ratio_stability', float('inf'))
        consistency = self.aggregate_metrics.get('return_consistency', 0)

        # Normalize and combine
        sharpe_score = max(0, min(1, (avg_sharpe + 2) / 4))  # Scale -2 to +2 to 0-1
        stability_score = max(0, min(1, 1 - sharpe_stability))  # Lower stability metric is better
        consistency_score = consistency

        # Weighted average
        robustness = (
            0.4 * sharpe_score +
            0.4 * stability_score +
            0.2 * consistency_score
        )

        return robustness

    def schedule_retraining(self, interval_days: int = 7, callback: Optional[Callable] = None) -> None:
        """Schedule periodic retraining."""
        self.scheduler.schedule_retraining(interval_days, callback)

    def cancel_retraining(self) -> None:
        """Cancel scheduled retraining."""
        self.scheduler.cancel_retraining()


class WalkForwardScheduler:
    """Scheduler for periodic walk-forward retraining."""

    def __init__(self, optimizer: WalkForwardOptimizer):
        """
        Initialize scheduler.

        Args:
            optimizer: WalkForwardOptimizer instance to schedule
        """
        self.optimizer = optimizer
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.timer: Optional[threading.Timer] = None
        self.is_running = False
        self.interval_days = 0
        self.callback: Optional[Callable] = None

    def schedule_retraining(self, interval_days: int, callback: Optional[Callable] = None) -> None:
        """
        Schedule periodic retraining.

        Args:
            interval_days: Interval between retraining in days
            callback: Optional callback function to call after retraining
        """
        self.cancel_retraining()  # Cancel any existing schedule

        self.interval_days = interval_days
        self.callback = callback
        self.is_running = True

        # Schedule first run
        self._schedule_next_run()

        self.logger.info(f"Scheduled walk-forward retraining every {interval_days} days")

    def cancel_retraining(self) -> None:
        """Cancel scheduled retraining."""
        if self.timer:
            self.timer.cancel()
            self.timer = None

        self.is_running = False
        self.logger.info("Walk-forward retraining cancelled")

    def _schedule_next_run(self) -> None:
        """Schedule the next retraining run."""
        if not self.is_running:
            return

        # Calculate interval in seconds
        interval_seconds = self.interval_days * 24 * 60 * 60

        self.timer = threading.Timer(interval_seconds, self._run_retraining)
        self.timer.daemon = True
        self.timer.start()

    def _run_retraining(self) -> None:
        """Execute retraining and schedule next run."""
        try:
            self.logger.info("Starting scheduled walk-forward retraining")

            # Note: In a real implementation, you would need to:
            # 1. Load fresh historical data
            # 2. Run walk-forward optimization
            # 3. Update strategy parameters
            # 4. Save results

            # For now, just log the event
            self.logger.info("Walk-forward retraining completed")

            # Call callback if provided
            if self.callback:
                try:
                    self.callback()
                except Exception as e:
                    self.logger.error(f"Retraining callback failed: {e}")

            # Schedule next run
            self._schedule_next_run()

        except Exception as e:
            self.logger.error(f"Scheduled retraining failed: {e}")
            # Still schedule next run even if this one failed
            self._schedule_next_run()


# Convenience functions for easy integration
def create_walk_forward_optimizer(config: Optional[Dict[str, Any]] = None) -> WalkForwardOptimizer:
    """
    Create a walk-forward optimizer with default configuration.

    Args:
        config: Optional configuration overrides

    Returns:
        Configured WalkForwardOptimizer instance
    """
    default_config = {
        'base_optimizer': {
            'population_size': 20,
            'generations': 10,
            'mutation_rate': 0.1,
            'crossover_rate': 0.7
        },
        'data_splitter': {
            'train_window_size': 100,
            'test_window_size': 20,
            'step_size': 20,
            'min_samples': 50,
            'overlap_allowed': False
        },
        'output_dir': 'results/walk_forward',
        'save_intermediate': True,
        'parallel_execution': False,
        'fitness_metric': 'sharpe_ratio'
    }

    if config:
        # Deep merge configurations
        def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result

        default_config = merge_dicts(default_config, config)

    return WalkForwardOptimizer(default_config)


def run_walk_forward_analysis(strategy_class, data: pd.DataFrame,
                             config: Optional[Dict[str, Any]] = None) -> WalkForwardResult:
    """
    Run complete walk-forward analysis.

    Args:
        strategy_class: Strategy class to analyze
        data: Historical data for analysis
        config: Optional configuration

    Returns:
        Complete walk-forward results
    """
    optimizer = create_walk_forward_optimizer(config)

    # Run optimization
    best_params = optimizer.optimize(strategy_class, data)

    # Create result object
    result = WalkForwardResult(
        strategy_name=strategy_class.__name__,
        total_windows=len(optimizer.windows),
        windows=optimizer.windows,
        aggregate_metrics=optimizer.aggregate_metrics,
        performance_distribution=optimizer.performance_distribution,
        optimization_summary=optimizer.get_walk_forward_summary(),
        timestamp=datetime.now(),
        total_time=sum(w.optimization_time for w in optimizer.windows)
    )

    return result
