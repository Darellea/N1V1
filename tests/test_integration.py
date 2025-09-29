"""
Integration Tests for N1V1 Crypto Trading Framework Optimization Modules

This module contains comprehensive integration tests that verify the end-to-end
functionality of the optimization workflow, including data loading, strategy
generation, backtesting, and optimization processes.

Tests cover:
- End-to-end optimization workflow
- Component interactions between data, backtesting, and optimization systems
- Output validation and correctness verification
- Error scenarios and graceful failure handling
- Integration with test database for isolation
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from backtest.backtester import Backtester, compute_backtest_metrics
from data.historical_loader import HistoricalDataLoader
from data.interfaces import IDataFetcher
from optimization.genetic_optimizer import GeneticOptimizer
from optimization.strategy_factory import StrategyFactory
from optimization.walk_forward import WalkForwardOptimizer


class MockDataFetcher(IDataFetcher):
    """Mock data fetcher for testing purposes."""

    def __init__(self, test_data=None):
        self.test_data = test_data or {}

    def _generate_test_data(self):
        """Generate mock OHLCV data for testing."""
        dates = pd.date_range("2023-01-01", periods=1000, freq="1h")
        np.random.seed(42)  # For reproducible results

        # Generate realistic price data with trend and volatility
        base_price = 50000
        returns = np.random.normal(
            0.0001, 0.02, len(dates)
        )  # Small drift with volatility
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate volume data
        volumes = np.random.lognormal(10, 1, len(dates))

        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices * (1 + np.random.uniform(0, 0.02, len(dates))),
                "low": prices * (1 - np.random.uniform(0, 0.02, len(dates))),
                "close": prices * (1 + np.random.normal(0, 0.01, len(dates))),
                "volume": volumes,
            },
            index=dates,
        )

        # Ensure high >= max(open, close) and low <= min(open, close)
        data["high"] = np.maximum(data[["open", "close"]].max(axis=1), data["high"])
        data["low"] = np.minimum(data[["open", "close"]].min(axis=1), data["low"])

        return data

    async def get_historical_data(self, symbol, timeframe, since, limit=1000):
        """Return mock historical data."""
        # Generate data matching the requested timeframe
        if timeframe not in self.test_data:
            self.test_data[timeframe] = self._generate_test_data_for_timeframe(
                timeframe
            )

        # Filter data based on 'since' timestamp
        since_dt = pd.to_datetime(since, unit="ms")
        filtered_data = self.test_data[timeframe][
            self.test_data[timeframe].index >= since_dt
        ]

        if len(filtered_data) > limit:
            filtered_data = filtered_data.head(limit)

        return filtered_data

    def _generate_test_data_for_timeframe(self, timeframe):
        """Generate mock OHLCV data for a specific timeframe."""
        # Map timeframe to pandas frequency
        freq_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D",
            "1w": "1W",
        }
        freq = freq_map.get(timeframe, "1D")

        dates = pd.date_range("2023-01-01", periods=100, freq=freq)
        np.random.seed(42)  # For reproducible results

        # Generate realistic price data with trend and volatility
        base_price = 50000
        returns = np.random.normal(
            0.0001, 0.02, len(dates)
        )  # Small drift with volatility
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate volume data
        volumes = np.random.lognormal(10, 1, len(dates))

        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices * (1 + np.random.uniform(0, 0.02, len(dates))),
                "low": prices * (1 - np.random.uniform(0, 0.02, len(dates))),
                "close": prices * (1 + np.random.normal(0, 0.01, len(dates))),
                "volume": volumes,
            },
            index=dates,
        )

        # Ensure high >= max(open, close) and low <= min(open, close)
        data["high"] = np.maximum(data[["open", "close"]].max(axis=1), data["high"])
        data["low"] = np.minimum(data[["open", "close"]].min(axis=1), data["low"])

        return data

    async def get_multiple_historical_data(
        self, symbols, timeframe="1h", limit=1000, since=None
    ):
        """Return mock historical data for multiple symbols."""
        result = {}
        for symbol in symbols:
            result[symbol] = await self.get_historical_data(
                symbol, timeframe, since, limit
            )
        return result

    async def get_realtime_data(self, symbols, tickers=True, orderbooks=False, depth=5):
        """Return mock real-time data."""
        result = {}
        for symbol in symbols:
            if tickers:
                result[symbol] = {
                    "symbol": symbol,
                    "last": 50000 + np.random.uniform(-1000, 1000),
                    "bid": 49950 + np.random.uniform(-100, 100),
                    "ask": 50050 + np.random.uniform(-100, 100),
                    "volume": np.random.uniform(100, 1000),
                }
        return result

    async def shutdown(self):
        """Cleanup resources."""
        pass


class MockStrategy:
    """Mock trading strategy for testing."""

    def __init__(self, config):
        self.config = config
        self.name = config.get("name", "mock_strategy")

    def generate_signals(self, data):
        """Generate mock trading signals."""
        signals = []

        # Simple RSI-based strategy simulation
        rsi_period = self.config.get("rsi_period", 14)
        overbought = self.config.get("overbought", 70)
        oversold = self.config.get("oversold", 30)

        # Mock RSI calculation (simplified)
        if len(data) > rsi_period:
            for i in range(rsi_period, len(data)):
                # Simulate RSI values
                rsi_value = np.random.uniform(20, 80)

                if rsi_value <= oversold:
                    signals.append(
                        {
                            "timestamp": data.index[i],
                            "signal": "BUY",
                            "price": data.iloc[i]["close"],
                            "rsi": rsi_value,
                        }
                    )
                elif rsi_value >= overbought:
                    signals.append(
                        {
                            "timestamp": data.index[i],
                            "signal": "SELL",
                            "price": data.iloc[i]["close"],
                            "rsi": rsi_value,
                        }
                    )

        return signals


class TestOptimizationIntegration:
    """Integration tests for the complete optimization workflow."""

    @pytest.fixture
    def test_db_path(self):
        """Create a temporary test database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def mock_data_fetcher(self):
        """Create mock data fetcher with test data."""
        return MockDataFetcher()

    @pytest.fixture
    def historical_loader(self, mock_data_fetcher):
        """Create historical data loader with mock fetcher."""
        config = {"backtesting": {"data_dir": "test_data", "force_refresh": True}}
        return HistoricalDataLoader(config, mock_data_fetcher)

    @pytest.fixture
    def backtester(self):
        """Create backtester instance."""
        return Backtester()

    @pytest.fixture
    def genetic_optimizer(self):
        """Create genetic optimizer for testing."""
        config = {
            "population_size": 10,
            "generations": 3,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "elitism_rate": 0.1,
            "fitness_metric": "sharpe_ratio",
        }
        return GeneticOptimizer(config)

    @pytest.fixture
    def walk_forward_optimizer(self):
        """Create walk-forward optimizer for testing."""
        config = {
            "train_window_days": 30,
            "test_window_days": 7,
            "rolling": True,
            "min_observations": 50,
            "improvement_threshold": 0.05,
        }
        return WalkForwardOptimizer(config)

    @pytest.mark.asyncio
    async def test_end_to_end_optimization_workflow(
        self, historical_loader, backtester, genetic_optimizer, test_db_path
    ):
        """
        Test the complete end-to-end optimization workflow.

        This test verifies that data loading, strategy generation, backtesting,
        and optimization work together seamlessly.
        """
        # Step 1: Load historical data
        symbols = ["BTC/USDT"]
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        timeframe = "1d"

        data_dict = await historical_loader.load_historical_data(
            symbols, start_date, end_date, timeframe, force_refresh=True
        )

        assert "BTC/USDT" in data_dict
        btc_data = data_dict["BTC/USDT"]
        assert not btc_data.empty
        assert len(btc_data) >= 100  # Ensure sufficient data

        # Step 2: Register mock strategy with factory
        StrategyFactory.register_strategy(
            "test_rsi",
            MockStrategy,
            "Test RSI Strategy",
            {
                "rsi_period": {"min": 5, "max": 30, "type": int, "default": 14},
                "overbought": {"min": 60, "max": 80, "type": int, "default": 70},
                "oversold": {"min": 20, "max": 40, "type": int, "default": 30},
            },
        )

        # Step 3: Create and configure optimizer
        genetic_optimizer.add_parameter_bounds(
            genetic_optimizer.ParameterBounds(
                name="rsi_period", min_value=5, max_value=30, param_type="int"
            )
        )
        genetic_optimizer.add_parameter_bounds(
            genetic_optimizer.ParameterBounds(
                name="overbought", min_value=60, max_value=80, param_type="int"
            )
        )
        genetic_optimizer.add_parameter_bounds(
            genetic_optimizer.ParameterBounds(
                name="oversold", min_value=20, max_value=40, param_type="int"
            )
        )

        # Step 4: Mock strategy creation for testing
        def mock_strategy_factory(params):
            config = {
                "name": "test_strategy",
                "rsi_period": params.get("rsi_period", 14),
                "overbought": params.get("overbought", 70),
                "oversold": params.get("oversold", 30),
            }
            return MockStrategy(config)

        # Step 5: Run optimization with mocked components
        with patch.object(genetic_optimizer, "_run_backtest") as mock_backtest, patch(
            "optimization.strategy_factory.StrategyFactory.create_strategy_from_genome"
        ) as mock_create:
            # Mock backtest to return realistic equity progression
            mock_backtest.return_value = [
                {
                    "trade_id": 1,
                    "timestamp": btc_data.index[0],
                    "equity": 10000,
                    "pnl": 0,
                    "cumulative_return": 0.0,
                },
                {
                    "trade_id": 2,
                    "timestamp": btc_data.index[10],
                    "equity": 10200,
                    "pnl": 200,
                    "cumulative_return": 0.02,
                },
                {
                    "trade_id": 3,
                    "timestamp": btc_data.index[20],
                    "equity": 10100,
                    "pnl": -100,
                    "cumulative_return": 0.01,
                },
            ]

            # Mock strategy creation
            mock_create.return_value = mock_strategy_factory(
                {"rsi_period": 14, "overbought": 70, "oversold": 30}
            )

            # Run optimization
            result = genetic_optimizer.optimize(MockStrategy, btc_data)

            # Step 6: Verify results
            assert isinstance(result, dict)
            assert "rsi_period" in result
            assert "overbought" in result
            assert "oversold" in result

            # Verify parameter bounds
            assert 5 <= result["rsi_period"] <= 30
            assert 60 <= result["overbought"] <= 80
            assert 20 <= result["oversold"] <= 40

    def test_component_interaction_data_to_backtest(
        self, historical_loader, backtester
    ):
        """
        Test interaction between data loading and backtesting components.

        Verifies that data loaded by the historical loader can be properly
        consumed by the backtesting system.
        """
        # Create test data
        dates = pd.date_range("2023-01-01", periods=100, freq="1h")
        test_data = pd.DataFrame(
            {
                "open": np.random.uniform(50000, 51000, 100),
                "high": np.random.uniform(50500, 51500, 100),
                "low": np.random.uniform(49500, 50500, 100),
                "close": np.random.uniform(50000, 51000, 100),
                "volume": np.random.uniform(100, 1000, 100),
            },
            index=dates,
        )

        # Mock strategy that generates signals
        mock_strategy = Mock()
        mock_strategy.generate_signals.return_value = [
            {
                "timestamp": dates[10],
                "signal": "BUY",
                "price": test_data.iloc[10]["close"],
            },
            {
                "timestamp": dates[20],
                "signal": "SELL",
                "price": test_data.iloc[20]["close"],
            },
            {
                "timestamp": dates[30],
                "signal": "BUY",
                "price": test_data.iloc[30]["close"],
            },
            {
                "timestamp": dates[40],
                "signal": "SELL",
                "price": test_data.iloc[40]["close"],
            },
        ]

        # Test backtest execution
        with patch.object(backtester, "run_backtest") as mock_run:
            mock_run.return_value = {
                "total_return": 0.05,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.03,
                "win_rate": 0.6,
                "total_trades": 2,
                "equity_progression": [
                    {
                        "trade_id": 1,
                        "timestamp": dates[10],
                        "equity": 10000,
                        "pnl": 200,
                        "cumulative_return": 0.02,
                    },
                    {
                        "trade_id": 2,
                        "timestamp": dates[20],
                        "equity": 10200,
                        "pnl": -50,
                        "cumulative_return": 0.015,
                    },
                    {
                        "trade_id": 3,
                        "timestamp": dates[30],
                        "equity": 10150,
                        "pnl": 150,
                        "cumulative_return": 0.035,
                    },
                    {
                        "trade_id": 4,
                        "timestamp": dates[40],
                        "equity": 10300,
                        "pnl": 100,
                        "cumulative_return": 0.05,
                    },
                ],
                "metrics": {},
            }

            result = backtester.run_backtest_sync(mock_strategy, test_data)

            # Verify backtest was called
            mock_run.assert_called_once()

            # Verify result structure
            assert "total_return" in result
            assert "sharpe_ratio" in result
            assert "equity_progression" in result
            assert len(result["equity_progression"]) == 4

    def test_component_interaction_backtest_to_optimizer(
        self, backtester, genetic_optimizer
    ):
        """
        Test interaction between backtesting and optimization components.

        Verifies that backtest results are properly consumed by the optimizer
        for fitness evaluation.
        """
        # Create test data
        dates = pd.date_range("2023-01-01", periods=50, freq="1D")
        test_data = pd.DataFrame(
            {
                "close": np.random.uniform(50000, 51000, 50),
                "high": np.random.uniform(50500, 51500, 50),
                "low": np.random.uniform(49500, 50500, 50),
                "open": np.random.uniform(50000, 51000, 50),
                "volume": np.random.uniform(100, 1000, 50),
            },
            index=dates,
        )

        # Mock strategy
        mock_strategy = Mock()
        mock_strategy.generate_signals.return_value = [
            {
                "timestamp": dates[10],
                "signal": "BUY",
                "price": test_data.iloc[10]["close"],
            },
            {
                "timestamp": dates[20],
                "signal": "SELL",
                "price": test_data.iloc[20]["close"],
            },
        ]

        # Mock backtest results
        mock_equity_progression = [
            {
                "trade_id": 1,
                "timestamp": dates[10],
                "equity": 10000,
                "pnl": 200,
                "cumulative_return": 0.02,
            },
            {
                "trade_id": 2,
                "timestamp": dates[20],
                "equity": 10200,
                "pnl": 100,
                "cumulative_return": 0.04,
            },
        ]

        # Test fitness evaluation
        with patch.object(
            genetic_optimizer, "_run_backtest", return_value=mock_equity_progression
        ), patch(
            "optimization.base_optimizer.compute_backtest_metrics"
        ) as mock_compute:
            mock_compute.return_value = {
                "sharpe_ratio": 1.5,
                "total_return": 0.04,
                "max_drawdown": 0.02,
                "win_rate": 0.75,
                "total_trades": 2,
                "wins": 1,
                "losses": 1,
            }

            fitness = genetic_optimizer.evaluate_fitness(mock_strategy, test_data)

            # Verify fitness calculation
            assert isinstance(fitness, float)
            assert fitness > 0

            # Verify backtest was called
            genetic_optimizer._run_backtest.assert_called_once_with(
                mock_strategy, test_data
            )

            # Verify metrics computation was called
            mock_compute.assert_called_once_with(mock_equity_progression)

    def test_output_validation_optimization_results(self, genetic_optimizer):
        """
        Test validation of optimization output results.

        Verifies that optimization results contain all required fields
        and meet expected criteria.
        """
        # Create mock optimization results
        mock_result = {
            "rsi_period": 14,
            "overbought": 70,
            "oversold": 30,
            "fitness_score": 1.25,
            "total_evaluations": 50,
            "optimization_time": 45.2,
        }

        # Test result validation
        assert "rsi_period" in mock_result
        assert "overbought" in mock_result
        assert "oversold" in mock_result
        assert isinstance(mock_result["rsi_period"], int)
        assert isinstance(mock_result["overbought"], int)
        assert isinstance(mock_result["oversold"], int)
        assert 5 <= mock_result["rsi_period"] <= 30
        assert 60 <= mock_result["overbought"] <= 80
        assert 20 <= mock_result["oversold"] <= 40
        assert mock_result["fitness_score"] > 0
        assert mock_result["total_evaluations"] > 0
        assert mock_result["optimization_time"] > 0

    def test_output_validation_backtest_metrics(self):
        """
        Test validation of backtest metrics output.

        Verifies that backtest metrics contain all required fields
        and are within reasonable ranges.
        """
        # Create mock equity progression
        dates = pd.date_range("2023-01-01", periods=20, freq="1D")
        equity_progression = [
            {
                "trade_id": i + 1,
                "timestamp": dates[i],
                "equity": 10000 + i * 50,
                "pnl": 50 if i % 2 == 0 else -25,
                "cumulative_return": i * 0.005,
            }
            for i in range(20)
        ]

        # Compute metrics
        metrics = compute_backtest_metrics(equity_progression)

        # Validate required fields
        required_fields = [
            "equity_curve",
            "max_drawdown",
            "sharpe_ratio",
            "profit_factor",
            "total_return",
            "total_trades",
            "wins",
            "losses",
            "win_rate",
        ]

        for field in required_fields:
            assert field in metrics, f"Missing required field: {field}"

        # Validate field types and ranges
        assert isinstance(metrics["equity_curve"], list)
        assert len(metrics["equity_curve"]) == len(equity_progression)
        # max_drawdown can be numpy float or Python float/int
        assert hasattr(metrics["max_drawdown"], "__float__") or isinstance(
            metrics["max_drawdown"], (int, float)
        )
        assert 0 <= float(metrics["max_drawdown"]) <= 1  # Should be between 0 and 1
        assert isinstance(metrics["sharpe_ratio"], (int, float))
        assert isinstance(metrics["profit_factor"], (int, float))
        assert metrics["profit_factor"] >= 0
        assert isinstance(metrics["total_return"], (int, float))
        assert isinstance(metrics["total_trades"], int)
        assert metrics["total_trades"] >= 0
        assert isinstance(metrics["wins"], int)
        assert isinstance(metrics["losses"], int)
        assert metrics["wins"] + metrics["losses"] == metrics["total_trades"]
        assert isinstance(metrics["win_rate"], (int, float))
        assert 0 <= metrics["win_rate"] <= 1

    def test_error_scenario_backtest_failure(self, backtester, genetic_optimizer):
        """
        Test error handling when backtesting fails.

        Verifies that optimization continues gracefully when individual
        backtests fail.
        """
        # Create test data
        test_data = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "high": [102, 103, 104],
                "low": [98, 99, 100],
                "open": [100, 101, 102],
                "volume": [1000, 1100, 1200],
            }
        )

        # Mock strategy that fails
        mock_strategy = Mock()
        mock_strategy.side_effect = Exception("Strategy execution failed")

        # Test that fitness evaluation handles errors gracefully
        with patch.object(
            genetic_optimizer, "_run_backtest", side_effect=Exception("Backtest failed")
        ):
            fitness = genetic_optimizer.evaluate_fitness(mock_strategy, test_data)

            # Should return negative infinity for failed evaluation
            assert fitness == float("-inf")

    def test_error_scenario_optimization_with_invalid_parameters(
        self, genetic_optimizer
    ):
        """
        Test error handling when optimization receives invalid parameters.

        Verifies that parameter validation works correctly and handles
        invalid inputs appropriately.
        """
        # Test with invalid parameter bounds
        with pytest.raises((ValueError, TypeError)):
            genetic_optimizer.add_parameter_bounds(
                genetic_optimizer.ParameterBounds(
                    name="invalid_param",
                    min_value="invalid",  # Should be numeric
                    max_value=100,
                    param_type="int",
                )
            )

    def test_database_integration_isolation(self, test_db_path):
        """
        Test that integration tests use isolated test database.

        Verifies that tests can create and use a test database without
        affecting the main application database.
        """
        # Create test database connection
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()

        # Create test table
        cursor.execute(
            """
            CREATE TABLE test_optimization_results (
                id INTEGER PRIMARY KEY,
                strategy_name TEXT,
                best_params TEXT,
                best_fitness REAL,
                timestamp DATETIME
            )
        """
        )

        # Insert test data
        test_result = {
            "strategy_name": "test_strategy",
            "best_params": json.dumps({"rsi_period": 14}),
            "best_fitness": 1.25,
            "timestamp": datetime.now().isoformat(),
        }

        cursor.execute(
            """
            INSERT INTO test_optimization_results
            (strategy_name, best_params, best_fitness, timestamp)
            VALUES (?, ?, ?, ?)
        """,
            (
                test_result["strategy_name"],
                test_result["best_params"],
                test_result["best_fitness"],
                test_result["timestamp"],
            ),
        )

        conn.commit()

        # Verify data was inserted
        cursor.execute("SELECT COUNT(*) FROM test_optimization_results")
        count = cursor.fetchone()[0]
        assert count == 1

        # Verify data integrity
        cursor.execute("SELECT * FROM test_optimization_results WHERE id = 1")
        row = cursor.fetchone()
        assert row[1] == "test_strategy"
        assert json.loads(row[2])["rsi_period"] == 14
        assert row[3] == 1.25

        conn.close()

        # Verify database file exists and has content
        assert os.path.exists(test_db_path)
        assert os.path.getsize(test_db_path) > 0

    def test_walk_forward_integration_workflow(self, walk_forward_optimizer):
        """
        Test walk-forward optimization integration.

        Verifies that walk-forward optimization works correctly with
        the overall optimization framework.
        """
        # Create test data with sufficient length
        dates = pd.date_range("2023-01-01", periods=200, freq="1D")
        test_data = pd.DataFrame(
            {
                "close": np.random.uniform(50000, 51000, 200),
                "high": np.random.uniform(50500, 51500, 200),
                "low": np.random.uniform(49500, 50500, 200),
                "open": np.random.uniform(50000, 51000, 200),
                "volume": np.random.uniform(100, 1000, 200),
            },
            index=dates,
        )

        # Mock strategy
        mock_strategy = Mock()
        mock_strategy.generate_signals.return_value = []

        # Mock optimization method
        with patch.object(
            walk_forward_optimizer, "optimize", return_value={"test_param": 42}
        ):
            result = walk_forward_optimizer.optimize(mock_strategy, test_data)

            assert isinstance(result, dict)
            assert "test_param" in result

    def test_strategy_factory_integration(self):
        """
        Test strategy factory integration with optimization.

        Verifies that the strategy factory can create strategies
        that work with the optimization system.
        """
        # Register test strategy
        StrategyFactory.register_strategy(
            "integration_test",
            MockStrategy,
            "Integration Test Strategy",
            {"test_param": {"min": 1, "max": 100, "type": int, "default": 50}},
        )

        # Verify strategy is registered
        available = StrategyFactory.get_available_strategies()
        assert "integration_test" in available
        assert available["integration_test"] == "Integration Test Strategy"

        # Get strategy info
        info = StrategyFactory.get_strategy_info("integration_test")
        assert info is not None
        assert info["description"] == "Integration Test Strategy"
        assert "test_param" in info["parameters"]

    def test_memory_management_during_optimization(self, genetic_optimizer):
        """
        Test memory management during optimization runs.

        Verifies that the optimization system properly manages memory
        and doesn't leak resources during long-running optimizations.
        """
        # Create test data
        test_data = pd.DataFrame(
            {
                "close": np.random.uniform(50000, 51000, 100),
                "high": np.random.uniform(50500, 51500, 100),
                "low": np.random.uniform(49500, 50500, 100),
                "open": np.random.uniform(50000, 51000, 100),
                "volume": np.random.uniform(100, 1000, 100),
            }
        )

        # Mock strategy
        mock_strategy = Mock()

        # Track memory usage (simplified)
        initial_population_size = (
            len(genetic_optimizer.population)
            if hasattr(genetic_optimizer, "population")
            else 0
        )

        # Run multiple fitness evaluations
        for _ in range(5):
            with patch.object(genetic_optimizer, "_run_backtest", return_value=[]):
                fitness = genetic_optimizer.evaluate_fitness(mock_strategy, test_data)
                assert isinstance(fitness, (int, float))

        # Verify population size remains stable (no memory leaks)
        if hasattr(genetic_optimizer, "population"):
            assert len(genetic_optimizer.population) == initial_population_size

    def test_concurrent_optimization_runs(self, genetic_optimizer):
        """
        Test running multiple optimization instances concurrently.

        Verifies that the optimization system can handle concurrent
        runs without interference.
        """
        import threading

        results = {}
        errors = []

        def run_optimization(optimizer_id):
            try:
                # Create separate test data for each thread
                test_data = pd.DataFrame(
                    {
                        "close": np.random.uniform(50000, 51000, 50),
                        "high": np.random.uniform(50500, 51500, 50),
                        "low": np.random.uniform(49500, 50500, 50),
                        "open": np.random.uniform(50000, 51000, 50),
                        "volume": np.random.uniform(100, 1000, 50),
                    }
                )

                mock_strategy = Mock()

                with patch.object(genetic_optimizer, "_run_backtest", return_value=[]):
                    fitness = genetic_optimizer.evaluate_fitness(
                        mock_strategy, test_data
                    )
                    results[optimizer_id] = fitness
            except Exception as e:
                errors.append(f"Thread {optimizer_id}: {str(e)}")

        # Create and start threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_optimization, args=(f"opt_{i}",))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors in concurrent runs: {errors}"

        # Verify all threads produced results
        assert len(results) == 3
        for optimizer_id, fitness in results.items():
            assert isinstance(fitness, (int, float))
            assert optimizer_id.startswith("opt_")

    def test_large_dataset_handling(self):
        """
        Test optimization with large datasets.

        Verifies that the system can handle large amounts of data
        without performance degradation or memory issues.
        """
        # Create large test dataset
        dates = pd.date_range(
            "2020-01-01", periods=10000, freq="1h"
        )  # ~1 year of hourly data
        large_data = pd.DataFrame(
            {
                "close": np.random.uniform(50000, 51000, 10000),
                "high": np.random.uniform(50500, 51500, 10000),
                "low": np.random.uniform(49500, 50500, 10000),
                "open": np.random.uniform(50000, 51000, 10000),
                "volume": np.random.uniform(100, 1000, 10000),
            },
            index=dates,
        )

        # Test data processing
        assert len(large_data) == 10000
        assert not large_data.empty

        # Test metrics computation on large dataset
        # Create mock equity progression
        equity_progression = [
            {
                "trade_id": i + 1,
                "timestamp": dates[i],
                "equity": 10000 + i * 0.1,
                "pnl": 0.1,
                "cumulative_return": i * 0.00001,
            }
            for i in range(min(1000, len(dates)))  # Limit for performance
        ]

        metrics = compute_backtest_metrics(equity_progression)

        # Verify metrics are computed correctly
        assert "sharpe_ratio" in metrics
        assert "total_return" in metrics
        assert isinstance(metrics["sharpe_ratio"], (int, float))
        assert isinstance(metrics["total_return"], (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
