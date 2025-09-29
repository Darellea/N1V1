"""
Unit and Integration Tests for Walk-Forward Optimizer

This module contains comprehensive tests for the walk-forward optimization system,
ensuring robustness, correctness, and proper integration with existing components.
"""

import logging

import pytest

logger = logging.getLogger(__name__)
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from optimization.base_optimizer import ParameterBounds
from optimization.walk_forward import (
    WalkForwardDataSplitter,
    WalkForwardOptimizer,
    WalkForwardResult,
    WalkForwardScheduler,
    WalkForwardWindow,
    create_walk_forward_optimizer,
    run_walk_forward_analysis,
)
from strategies.base_strategy import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy for testing purposes."""

    def __init__(self, config):
        super().__init__(config)
        self.default_params = {"fast_period": 9, "slow_period": 21, "signal_period": 9}
        config_params = (
            config.params if hasattr(config, "params") else config.get("params", {})
        )
        self.params = {**self.default_params, **(config_params or {})}

    async def calculate_indicators(self, data):
        return data

    async def generate_signals(self, data):
        return []

    def create_signal(self, **kwargs):
        return Mock()


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="h")

    # Generate realistic price data
    price = 100
    prices = []
    for _ in range(200):
        price *= 1 + np.random.normal(0, 0.01)  # 1% volatility
        prices.append(price)

    data = pd.DataFrame(
        {
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            "close": prices,
            "volume": np.random.randint(1000, 10000, 200),
        },
        index=dates,
    )

    return data


@pytest.fixture
def basic_config():
    """Basic configuration for walk-forward optimizer."""
    return {
        "base_optimizer": {
            "population_size": 10,
            "generations": 5,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
        },
        "data_splitter": {
            "train_window_size": 50,
            "test_window_size": 20,
            "step_size": 20,
            "min_samples": 30,
            "overlap_allowed": False,
        },
        "output_dir": tempfile.mkdtemp(),
        "save_intermediate": False,
        "parallel_execution": False,
        "fitness_metric": "sharpe_ratio",
    }


@pytest.fixture
def optimizer_with_bounds(basic_config):
    """Create optimizer with parameter bounds."""
    optimizer = WalkForwardOptimizer(basic_config)

    # Add parameter bounds
    optimizer.add_parameter_bounds(
        ParameterBounds(name="fast_period", min_value=5, max_value=20, param_type="int")
    )
    optimizer.add_parameter_bounds(
        ParameterBounds(
            name="slow_period", min_value=15, max_value=40, param_type="int"
        )
    )
    optimizer.add_parameter_bounds(
        ParameterBounds(
            name="signal_period", min_value=5, max_value=15, param_type="int"
        )
    )

    return optimizer


class TestWalkForwardDataSplitter:
    """Test cases for data splitting functionality."""

    def test_init_with_valid_config(self, basic_config):
        """Test initialization with valid configuration."""
        splitter = WalkForwardDataSplitter(basic_config["data_splitter"])
        assert splitter.train_window_size == 50
        assert splitter.test_window_size == 20
        assert splitter.step_size == 20
        assert splitter.min_samples == 30
        assert not splitter.overlap_allowed

    def test_parse_window_size_int(self):
        """Test parsing integer window sizes."""
        config = {"train_window_size": 100, "test_window_size": 20}
        splitter = WalkForwardDataSplitter(config)
        assert splitter.train_window_size == 100
        assert splitter.test_window_size == 20

    def test_parse_window_size_duration(self):
        """Test parsing duration string window sizes."""
        config = {
            "train_window_size": "30D",
            "test_window_size": "7D",
            "step_size": "7D",
        }
        splitter = WalkForwardDataSplitter(config)
        assert isinstance(splitter.train_window_size, timedelta)
        assert splitter.train_window_size == timedelta(days=30)
        assert isinstance(splitter.test_window_size, timedelta)
        assert splitter.test_window_size == timedelta(days=7)

    def test_split_data_insufficient_samples(self, sample_data):
        """Test handling of insufficient data samples."""
        config = {"train_window_size": 150, "test_window_size": 100, "min_samples": 200}
        splitter = WalkForwardDataSplitter(config)

        windows = splitter.split_data(sample_data)
        assert len(windows) == 0

    def test_split_data_valid_windows(self, sample_data):
        """Test creation of valid windows from sufficient data."""
        config = {
            "train_window_size": 50,
            "test_window_size": 20,
            "step_size": 20,
            "min_samples": 30,
        }
        splitter = WalkForwardDataSplitter(config)

        windows = splitter.split_data(sample_data)
        assert len(windows) > 0

        # Check first window structure
        first_window = windows[0]
        assert isinstance(first_window, WalkForwardWindow)
        assert first_window.window_index == 0
        assert len(first_window.train_data) == 50
        assert len(first_window.test_data) == 20

    def test_split_data_no_overlap(self, sample_data):
        """Test that windows don't overlap when step_size ensures no overlap."""
        config = {
            "train_window_size": 40,
            "test_window_size": 20,
            "step_size": 60,  # Large step to ensure no overlap (train + test = 60)
            "min_samples": 30,
            "overlap_allowed": False,
        }
        splitter = WalkForwardDataSplitter(config)

        windows = splitter.split_data(sample_data)

        # Check that windows don't overlap (allowing for small timing differences)
        for i in range(1, len(windows)):
            prev_end = windows[i - 1].test_end
            curr_start = windows[i].train_start
            # Allow for small timing differences due to data indexing
            time_diff = (curr_start - prev_end).total_seconds()
            assert (
                time_diff >= -3600
            ), f"Windows {i-1} and {i} overlap significantly"  # Allow 1 hour tolerance

    def test_split_data_time_based(self):
        """Test time-based window splitting."""
        # Create data with datetime index
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "close": np.random.randn(100),
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        config = {
            "train_window_size": "30D",
            "test_window_size": "7D",
            "step_size": "7D",
            "min_samples": 10,
        }
        splitter = WalkForwardDataSplitter(config)

        windows = splitter.split_data(data)
        assert len(windows) > 0

        # Check time spans
        first_window = windows[0]
        time_span = first_window.train_end - first_window.train_start
        assert time_span >= timedelta(days=25)  # Allow some tolerance


class TestWalkForwardOptimizer:
    """Test cases for the main walk-forward optimizer."""

    def test_init_with_config(self, basic_config):
        """Test initialization with configuration."""
        optimizer = WalkForwardOptimizer(basic_config)
        assert optimizer.output_dir == basic_config["output_dir"]
        assert not optimizer.parallel_execution
        assert optimizer.save_intermediate == basic_config["save_intermediate"]

    def test_add_parameter_bounds(self, basic_config):
        """Test adding parameter bounds."""
        optimizer = WalkForwardOptimizer(basic_config)

        bounds = ParameterBounds(
            name="test_param", min_value=0, max_value=100, param_type="int"
        )
        optimizer.add_parameter_bounds(bounds)

        assert "test_param" in optimizer.parameter_bounds
        assert optimizer.parameter_bounds["test_param"] == bounds

    def test_optimize_no_windows(self, basic_config):
        """Test optimization with no valid windows."""
        optimizer = WalkForwardOptimizer(basic_config)

        # Mock data splitter to return no windows
        with patch.object(optimizer.data_splitter, "split_data", return_value=[]):
            result = optimizer.optimize(MockStrategy, pd.DataFrame())
            assert result == {}

    def test_optimize_with_windows(self, basic_config, sample_data):
        """Test optimization with valid windows."""
        optimizer = WalkForwardOptimizer(basic_config)

        # Add parameter bounds
        optimizer.add_parameter_bounds(
            ParameterBounds(
                name="fast_period", min_value=5, max_value=20, param_type="int"
            )
        )
        optimizer.add_parameter_bounds(
            ParameterBounds(
                name="slow_period", min_value=15, max_value=40, param_type="int"
            )
        )

        # Mock the evaluation methods
        with patch.object(optimizer, "_evaluate_strategy") as mock_eval, patch.object(
            optimizer, "_create_optimizer"
        ) as mock_create:
            # Mock optimizer instance
            mock_optimizer_instance = Mock()
            mock_optimizer_instance.optimize.return_value = {
                "fast_period": 10,
                "slow_period": 25,
            }
            mock_create.return_value = mock_optimizer_instance

            mock_eval.return_value = {
                "sharpe_ratio": 1.5,
                "total_return": 0.15,
                "win_rate": 0.55,
                "max_drawdown": 0.08,
            }

            result = optimizer.optimize(MockStrategy, sample_data)

            # Verify optimization was called
            assert mock_create.called
            assert isinstance(result, dict)
            assert len(optimizer.windows) > 0

    def test_calculate_aggregate_metrics(self, basic_config):
        """Test calculation of aggregate metrics."""
        optimizer = WalkForwardOptimizer(basic_config)

        # Create mock windows with test metrics
        windows = []
        for i in range(3):
            window = WalkForwardWindow(
                window_index=i,
                train_start=datetime.now(),
                train_end=datetime.now(),
                test_start=datetime.now(),
                test_end=datetime.now(),
                train_data=pd.DataFrame(),
                test_data=pd.DataFrame(),
                optimized_params={"param1": i},
                train_metrics={"sharpe_ratio": 1.0 + i * 0.5},
                test_metrics={
                    "sharpe_ratio": 1.2 + i * 0.3,
                    "total_return": 0.1 + i * 0.05,
                },
                optimization_time=10.0,
            )
            windows.append(window)

        optimizer.windows = windows
        optimizer._calculate_aggregate_metrics()

        # Verify aggregate metrics
        assert "avg_test_sharpe" in optimizer.aggregate_metrics
        assert "std_test_sharpe" in optimizer.aggregate_metrics
        assert optimizer.aggregate_metrics["total_windows"] == 3
        assert optimizer.aggregate_metrics["successful_windows"] == 3

    def test_calculate_performance_distribution(self, basic_config):
        """Test calculation of performance distribution."""
        optimizer = WalkForwardOptimizer(basic_config)

        # Create mock windows
        windows = []
        for i in range(5):
            window = WalkForwardWindow(
                window_index=i,
                train_start=datetime.now(),
                train_end=datetime.now(),
                test_start=datetime.now(),
                test_end=datetime.now(),
                train_data=pd.DataFrame(),
                test_data=pd.DataFrame(),
                optimized_params={},
                train_metrics={},
                test_metrics={"sharpe_ratio": 0.5 + i * 0.2},
                optimization_time=10.0,
            )
            windows.append(window)

        optimizer.windows = windows
        optimizer._calculate_performance_distribution()

        # Verify distribution metrics
        assert "sharpe_percentiles" in optimizer.performance_distribution
        assert "sharpe_quartiles" in optimizer.performance_distribution
        assert "positive_sharpe_ratio" in optimizer.performance_distribution

    def test_select_best_parameters(self, basic_config):
        """Test selection of best parameters."""
        optimizer = WalkForwardOptimizer(basic_config)

        # Create mock windows with different parameters
        windows = []
        params_options = [
            {"fast_period": 10, "slow_period": 20},
            {"fast_period": 12, "slow_period": 25},
            {"fast_period": 10, "slow_period": 20},  # Same as first
        ]

        for i, params in enumerate(params_options):
            window = WalkForwardWindow(
                window_index=i,
                train_start=datetime.now(),
                train_end=datetime.now(),
                test_start=datetime.now(),
                test_end=datetime.now(),
                train_data=pd.DataFrame(),
                test_data=pd.DataFrame(),
                optimized_params=params,
                train_metrics={},
                test_metrics={"sharpe_ratio": 1.0 + i * 0.2},
                optimization_time=10.0,
            )
            windows.append(window)

        optimizer.windows = windows
        best_params = optimizer._select_best_parameters()

        # Should select parameters that appeared in best performing windows
        assert isinstance(best_params, dict)
        assert "fast_period" in best_params
        assert "slow_period" in best_params

    def test_save_results(self, basic_config, tmp_path):
        """Test saving of results."""
        basic_config["output_dir"] = str(tmp_path)
        optimizer = WalkForwardOptimizer(basic_config)

        # Create mock windows
        optimizer.windows = [
            WalkForwardWindow(
                window_index=0,
                train_start=datetime.now(),
                train_end=datetime.now(),
                test_start=datetime.now(),
                test_end=datetime.now(),
                train_data=pd.DataFrame(),
                test_data=pd.DataFrame(),
                optimized_params={"param1": 10},
                train_metrics={"sharpe_ratio": 1.5},
                test_metrics={"sharpe_ratio": 1.2},
                optimization_time=10.0,
            )
        ]

        optimizer.aggregate_metrics = {"avg_test_sharpe": 1.2}
        optimizer.performance_distribution = {"positive_sharpe_ratio": 0.8}

        # Save results
        optimizer._save_results(30.0)

        # Verify files were created
        assert (tmp_path / "walk_forward_results.json").exists()
        assert (tmp_path / "walk_forward_summary.json").exists()
        assert (tmp_path / "walk_forward_windows.csv").exists()

        # Verify JSON content
        with open(tmp_path / "walk_forward_summary.json", "r") as f:
            summary = json.load(f)
            assert "aggregate_metrics" in summary
            assert "performance_distribution" in summary

    def test_get_walk_forward_summary(self, basic_config):
        """Test getting walk-forward summary."""
        optimizer = WalkForwardOptimizer(basic_config)

        # Set up mock data
        optimizer.windows = [
            WalkForwardWindow(
                window_index=0,
                train_start=datetime.now(),
                train_end=datetime.now(),
                test_start=datetime.now(),
                test_end=datetime.now(),
                train_data=pd.DataFrame(),
                test_data=pd.DataFrame(),
                optimized_params={},
                train_metrics={},
                test_metrics={"sharpe_ratio": 1.2},
                optimization_time=10.0,
            )
        ]
        optimizer.aggregate_metrics = {"avg_test_sharpe": 1.2}
        optimizer.performance_distribution = {"positive_sharpe_ratio": 0.8}

        summary = optimizer.get_walk_forward_summary()

        assert "total_windows" in summary
        assert "aggregate_metrics" in summary
        assert "performance_distribution" in summary
        assert "stability_metrics" in summary
        assert "robustness_score" in summary


class TestWalkForwardScheduler:
    """Test cases for the walk-forward scheduler."""

    def test_init(self, basic_config):
        """Test scheduler initialization."""
        optimizer = WalkForwardOptimizer(basic_config)
        scheduler = WalkForwardScheduler(optimizer)

        assert not scheduler.is_running
        assert scheduler.interval_days == 0
        assert scheduler.timer is None

    def test_schedule_retraining(self, basic_config):
        """Test scheduling retraining."""
        optimizer = WalkForwardOptimizer(basic_config)
        scheduler = WalkForwardScheduler(optimizer)

        callback_called = False

        def test_callback():
            nonlocal callback_called
            callback_called = True

        scheduler.schedule_retraining(interval_days=1, callback=test_callback)

        assert scheduler.is_running
        assert scheduler.interval_days == 1
        assert scheduler.callback == test_callback
        assert scheduler.timer is not None

    def test_cancel_retraining(self, basic_config):
        """Test cancelling retraining."""
        optimizer = WalkForwardOptimizer(basic_config)
        scheduler = WalkForwardScheduler(optimizer)

        scheduler.schedule_retraining(interval_days=1)
        assert scheduler.is_running

        scheduler.cancel_retraining()
        assert not scheduler.is_running
        assert scheduler.timer is None


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_create_walk_forward_optimizer(self):
        """Test creating optimizer with convenience function."""
        config = {"data_splitter": {"train_window_size": 60, "test_window_size": 20}}

        optimizer = create_walk_forward_optimizer(config)

        assert isinstance(optimizer, WalkForwardOptimizer)
        assert optimizer.data_splitter.train_window_size == 60
        assert optimizer.data_splitter.test_window_size == 20

    def test_run_walk_forward_analysis(self, sample_data):
        """Test running complete walk-forward analysis."""
        config = {
            "data_splitter": {
                "train_window_size": 50,
                "test_window_size": 20,
                "min_samples": 30,
            },
            "save_intermediate": False,
        }

        with patch(
            "optimization.walk_forward.create_walk_forward_optimizer"
        ) as mock_create:
            # Mock the optimizer
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = {"fast_period": 10}
            mock_window = Mock()
            mock_window.optimization_time = 10.0
            mock_optimizer.windows = [mock_window]
            mock_create.return_value = mock_optimizer

            result = run_walk_forward_analysis(MockStrategy, sample_data, config)

            assert isinstance(result, WalkForwardResult)
            assert result.strategy_name == "MockStrategy"
            assert result.total_windows > 0


class TestIntegration:
    """Integration tests for walk-forward optimizer."""

    def test_full_walk_forward_workflow(self, sample_data, tmp_path):
        """Test complete walk-forward workflow."""
        config = {
            "base_optimizer": {"population_size": 5, "generations": 2},
            "data_splitter": {
                "train_window_size": 40,
                "test_window_size": 15,
                "step_size": 15,
                "min_samples": 25,
            },
            "output_dir": str(tmp_path),
            "save_intermediate": True,
            "parallel_execution": False,
        }

        optimizer = WalkForwardOptimizer(config)

        # Add parameter bounds
        optimizer.add_parameter_bounds(
            ParameterBounds(
                name="fast_period", min_value=5, max_value=15, param_type="int"
            )
        )

        with patch.object(optimizer, "_create_optimizer") as mock_create, patch.object(
            optimizer, "_evaluate_strategy"
        ) as mock_eval:
            # Mock optimizer instance
            mock_optimizer_instance = Mock()
            mock_optimizer_instance.optimize.return_value = {"fast_period": 10}
            mock_create.return_value = mock_optimizer_instance

            mock_eval.return_value = {
                "sharpe_ratio": 1.2,
                "total_return": 0.12,
                "win_rate": 0.55,
                "max_drawdown": 0.08,
            }

            # Run optimization
            best_params = optimizer.optimize(MockStrategy, sample_data)

            # Verify results
            assert isinstance(best_params, dict)
            assert len(optimizer.windows) > 0
            assert optimizer.aggregate_metrics["total_windows"] == len(
                optimizer.windows
            )

            # Verify files were created
            assert (tmp_path / "walk_forward_results.json").exists()
            assert (tmp_path / "walk_forward_summary.json").exists()

    def test_error_handling_insufficient_data(self):
        """Test error handling with insufficient data."""
        config = {
            "data_splitter": {
                "train_window_size": 1000,
                "test_window_size": 500,
                "min_samples": 100,
            }
        }

        optimizer = WalkForwardOptimizer(config)
        small_data = pd.DataFrame({"close": [1, 2, 3, 4, 5]})

        result = optimizer.optimize(MockStrategy, small_data)
        assert result == {}

    def test_error_handling_optimizer_failure(self, sample_data):
        """Test error handling when optimizer fails."""
        config = {
            "data_splitter": {
                "train_window_size": 30,
                "test_window_size": 10,
                "min_samples": 20,
            },
            "save_intermediate": False,
        }

        optimizer = WalkForwardOptimizer(config)

        with patch.object(optimizer, "_create_optimizer") as mock_create:
            # Mock optimizer to raise exception
            mock_instance = Mock()
            mock_instance.optimize.side_effect = Exception("Optimizer failed")
            mock_create.return_value = mock_instance

            # Should handle error gracefully
            result = optimizer.optimize(MockStrategy, sample_data)
            assert result == {}  # Should return empty dict on failure


if __name__ == "__main__":
    pytest.main([__file__])
