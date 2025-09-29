#!/usr/bin/env python3
"""
Demo script for Cross-Pair Validation functionality.

This script demonstrates how to use the cross-pair validation feature
to optimize parameters on one asset and validate on multiple others.
"""

from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd

from optimization.walk_forward import WalkForwardOptimizer


def create_mock_strategy():
    """Create a mock strategy class for demonstration."""

    class MockStrategy:
        def __init__(self, config):
            self.name = config.get("name", "mock_strategy")
            self.symbols = config.get("symbols", ["BTC/USDT"])
            self.params = config.get("params", {})

        def generate_signals(self, data):
            """Mock signal generation."""
            return []

    return MockStrategy


def create_sample_data():
    """Create sample price data for multiple pairs."""
    np.random.seed(42)  # For reproducible results

    # Create 2 years of daily data
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")

    # Generate correlated price series
    n_points = len(dates)

    # Base trend
    base_trend = np.cumsum(np.random.randn(n_points) * 0.5)

    data_dict = {}

    # BTC/USDT - Primary pair
    btc_noise = np.random.randn(n_points) * 2
    data_dict["BTC/USDT"] = pd.DataFrame(
        {
            "close": 50000 + base_trend * 1000 + btc_noise,
            "high": 50000
            + base_trend * 1000
            + btc_noise
            + abs(np.random.randn(n_points)) * 500,
            "low": 50000
            + base_trend * 1000
            + btc_noise
            - abs(np.random.randn(n_points)) * 500,
            "volume": np.random.randint(1000000, 10000000, n_points),
        },
        index=dates,
    )

    # ETH/USDT - Correlated with BTC
    eth_noise = np.random.randn(n_points) * 1.5
    data_dict["ETH/USDT"] = pd.DataFrame(
        {
            "close": 3000 + base_trend * 100 + eth_noise,
            "high": 3000
            + base_trend * 100
            + eth_noise
            + abs(np.random.randn(n_points)) * 100,
            "low": 3000
            + base_trend * 100
            + eth_noise
            - abs(np.random.randn(n_points)) * 100,
            "volume": np.random.randint(500000, 5000000, n_points),
        },
        index=dates,
    )

    # SOL/USDT - Less correlated
    sol_trend = np.cumsum(np.random.randn(n_points) * 0.3)
    sol_noise = np.random.randn(n_points) * 1
    data_dict["SOL/USDT"] = pd.DataFrame(
        {
            "close": 100 + sol_trend * 20 + sol_noise,
            "high": 100
            + sol_trend * 20
            + sol_noise
            + abs(np.random.randn(n_points)) * 5,
            "low": 100
            + sol_trend * 20
            + sol_noise
            - abs(np.random.randn(n_points)) * 5,
            "volume": np.random.randint(100000, 1000000, n_points),
        },
        index=dates,
    )

    return data_dict


def main():
    """Run the cross-pair validation demo."""
    print("🚀 Cross-Pair Validation Demo")
    print("=" * 50)

    # Create sample data
    print("\n📊 Creating sample data for BTC/USDT, ETH/USDT, SOL/USDT...")
    data_dict = create_sample_data()

    for pair, data in data_dict.items():
        print(f"  {pair}: {len(data)} data points")

    # Create optimizer
    config = {"min_observations": 100, "train_window_days": 90, "test_window_days": 30}
    optimizer = WalkForwardOptimizer(config)

    # Create mock strategy
    strategy_class = create_mock_strategy()

    # Define validation pairs
    train_pair = "BTC/USDT"
    validation_pairs = ["ETH/USDT", "SOL/USDT"]

    print(f"\n🎯 Training on: {train_pair}")
    print(f"🔍 Validating on: {validation_pairs}")

    # Mock the optimize method to return sample parameters
    optimizer.optimize = Mock(
        return_value={"rsi_period": 14, "overbought": 70, "oversold": 30}
    )

    # Run cross-pair validation
    print("\n⚡ Running cross-pair validation...")
    start_time = datetime.now()

    results = optimizer.cross_pair_validation(
        strategy_class, data_dict, train_pair, validation_pairs
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Display results
    print("\n📈 Results:")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Best Parameters: {results.get('best_params', {})}")

    print("\n📊 Performance Metrics:")
    print("<15")
    print("-" * 80)

    for pair, metrics in results.get("results", {}).items():
        print(
            "<15" "<10.3f" "<12.3f" "<12.3f" "<12.3f" "<8.1%" "<8d" "<10.4f",
            pair,
            metrics.get("sharpe_ratio", 0),
            metrics.get("max_drawdown", 0),
            metrics.get("profit_factor", 0),
            metrics.get("total_return", 0),
            metrics.get("win_rate", 0),
            metrics.get("total_trades", 0),
            metrics.get("expectancy", 0),
        )

    # Summary
    print("\n🎯 Summary:")
    train_metrics = results.get("results", {}).get(train_pair, {})
    validation_metrics = [
        results.get("results", {}).get(pair, {}) for pair in validation_pairs
    ]

    avg_sharpe = np.mean([m.get("sharpe_ratio", 0) for m in validation_metrics])
    avg_return = np.mean([m.get("total_return", 0) for m in validation_metrics])

    print(
        f"  Train Pair ({train_pair}) Sharpe: {train_metrics.get('sharpe_ratio', 0):.3f}"
    )
    print(f"  Avg Validation Sharpe: {avg_sharpe:.3f}")
    print(
        f"  Sharpe Ratio Difference: {avg_sharpe - train_metrics.get('sharpe_ratio', 0):.3f}"
    )

    print(
        f"  Train Pair ({train_pair}) Return: {train_metrics.get('total_return', 0):.3f}"
    )
    print(f"  Avg Validation Return: {avg_return:.3f}")
    print(
        f"  Return Difference: {avg_return - train_metrics.get('total_return', 0):.3f}"
    )

    # Check if results were saved
    import os

    if os.path.exists("results/cross_pair_validation.json"):
        print("\n💾 Results saved to:")
        print("  - results/cross_pair_validation.json")
        print("  - results/cross_pair_validation.csv")

    print("\n✅ Cross-pair validation demo completed!")
    print("\n💡 Key Benefits:")
    print("  • Optimize on one asset, validate on multiple others")
    print("  • Detect overfitting to specific market conditions")
    print("  • Ensure robustness across different pairs")
    print("  • Automatic results saving and reporting")


if __name__ == "__main__":
    main()
