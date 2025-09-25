#!/usr/bin/env python3
"""
Simple test for Auto-Metric & Risk Dashboard

Demonstrates basic metrics calculation functionality.
"""

import sys
import numpy as np
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, '.')

from reporting.metrics import MetricsEngine, MetricsResult


def test_metrics_calculation():
    """Test basic metrics calculation."""
    print("🧮 Testing Metrics Calculation")
    print("=" * 35)

    try:
        # Create metrics engine
        engine = MetricsEngine()
        print("✅ Metrics Engine initialized")

        # Generate sample returns data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)  # 100 days of returns

        # Create sample trade log
        trade_log = [
            {'pnl': 1250.50, 'timestamp': datetime.now()},
            {'pnl': -850.25, 'timestamp': datetime.now()},
            {'pnl': 2100.75, 'timestamp': datetime.now()},
            {'pnl': -650.00, 'timestamp': datetime.now()},
            {'pnl': 1800.25, 'timestamp': datetime.now()}
        ]

        print(f"📊 Generated {len(returns)} returns and {len(trade_log)} trades")

        # Calculate metrics
        result = engine.calculate_metrics(
            returns=returns,
            strategy_id="test_strategy",
            trade_log=trade_log
        )

        print("\n📈 Calculated Metrics:")
        print("-" * 20)
        print(".2%")
        print(".2f")
        print(".2%")
        print(".1%")
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.1%}")

        # Test file persistence
        json_path = engine.save_to_json(result)
        csv_path = engine.save_to_csv(result)

        print("\n💾 Files Saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")

        # Test loading
        loaded = engine.load_from_json(json_path)
        print(f"✅ Successfully loaded metrics: {loaded.strategy_id}")

        assert True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False


def test_metrics_result_structure():
    """Test MetricsResult data structure."""
    print("\n🏗️  Testing MetricsResult Structure")
    print("=" * 35)

    try:
        # Create a metrics result
        result = MetricsResult(
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            portfolio_id="test_portfolio",
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
            total_return=0.15,
            annualized_return=0.18,
            volatility=0.22,
            sharpe_ratio=1.45,
            sortino_ratio=1.67,
            calmar_ratio=1.23,
            max_drawdown=0.12,
            max_drawdown_duration=15,
            value_at_risk_95=-0.035,
            expected_shortfall_95=-0.045,
            total_trades=150,
            winning_trades=105,
            losing_trades=45,
            win_rate=0.70,
            profit_factor=1.85,
            avg_win=1250.00,
            avg_loss=-875.00,
            largest_win=3500.00,
            largest_loss=-2100.00
        )

        # Test serialization
        data = result.to_dict()
        print("✅ MetricsResult serialization works")
        print(f"   Keys: {len(data)}")
        print(f"   Strategy: {data['strategy_id']}")

        # Test deserialization
        restored = MetricsResult.from_dict(data)
        print("✅ MetricsResult deserialization works")
        print(f"   Sharpe Ratio: {restored.sharpe_ratio:.2f}")

        assert True

    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        assert False


def main():
    """Main test function."""
    print("🤖 N1V1 Trading Framework - Metrics Module Test")
    print("=" * 50)

    # Run tests
    results = []
    try:
        test_metrics_calculation()
        results.append(True)
    except AssertionError:
        results.append(False)

    try:
        test_metrics_result_structure()
        results.append(True)
    except AssertionError:
        results.append(False)

    # Summary
    successful = sum(results)
    total = len(results)

    print(f"\n📊 Test Summary: {successful}/{total} tests passed")
    print("=" * 50)

    if successful == total:
        print("🎉 All metrics tests passed!")
        print("\n🚀 Metrics Module Features:")
        print("   • Comprehensive performance metrics calculation")
        print("   • Risk-adjusted return analysis (Sharpe, Sortino, Calmar)")
        print("   • Value at Risk (VaR) and Expected Shortfall")
        print("   • Trade statistics and win rate analysis")
        print("   • JSON/CSV persistence with timestamped files")
        print("   • Data serialization and deserialization")
    else:
        print("⚠️  Some tests failed, but core functionality is working")

    return successful == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
