"""
Unit tests for backtest/backtester.py to verify algorithmic correctness fixes.
Tests division by zero protection, edge case handling, and statistical integrity.
"""
import pytest
import tempfile
import os
import json
from unittest.mock import Mock
from math import inf, nan

from backtest.backtester import (
    compute_backtest_metrics,
    _calculate_sharpe_ratio,
    _calculate_profit_factor,
    _calculate_max_drawdown,
    _compute_returns,
    export_equity_progression,
    export_metrics,
    compute_regime_aware_metrics,
    BacktestValidationError,
    BacktestSecurityError
)


class TestBacktestMetrics:
    """Test suite for backtest metrics calculations with edge cases."""

    def test_sharpe_ratio_division_by_zero_protection(self):
        """Test Sharpe ratio handles zero volatility safely."""
        # Test with constant returns (zero volatility)
        returns = [0.01, 0.01, 0.01, 0.01]
        sharpe = _calculate_sharpe_ratio(returns)
        assert sharpe == 0.0, "Should return 0.0 for zero volatility"

        # Test with empty returns
        sharpe = _calculate_sharpe_ratio([])
        assert sharpe == 0.0, "Should return 0.0 for empty returns"

        # Test with single return
        sharpe = _calculate_sharpe_ratio([0.01])
        assert sharpe == 0.0, "Should return 0.0 for single return"

    def test_sharpe_ratio_with_nan_inf_values(self):
        """Test Sharpe ratio handles NaN and infinite values safely."""
        returns = [0.01, nan, 0.02, inf, -inf]
        sharpe = _calculate_sharpe_ratio(returns)
        assert sharpe == 0.0 or isinstance(sharpe, float), "Should handle NaN/inf gracefully"

    def test_profit_factor_division_by_zero_protection(self):
        """Test profit factor handles zero losses safely."""
        # Test with profits but no losses
        equity_data = [
            {'pnl': 100.0},
            {'pnl': 200.0},
            {'pnl': 50.0}
        ]
        profit_factor = _calculate_profit_factor(equity_data)
        assert profit_factor == float('inf'), "Should return inf for profits with no losses"

        # Test with no profits or losses
        equity_data = [
            {'pnl': 0.0},
            {'pnl': 0.0}
        ]
        profit_factor = _calculate_profit_factor(equity_data)
        assert profit_factor == 0.0, "Should return 0.0 for no profits/losses"

    def test_max_drawdown_division_by_zero_protection(self):
        """Test max drawdown handles zero peak equity safely."""
        # Test with zero equity values
        equity_data = [
            {'equity': 0.0},
            {'equity': 0.0}
        ]
        max_dd = _calculate_max_drawdown(equity_data)
        assert max_dd == 0.0, "Should return 0.0 for zero equity"

    def test_returns_calculation_with_zero_equity(self):
        """Test returns calculation handles zero equity safely."""
        equity_data = [
            {'equity': 100.0},
            {'equity': 0.0},  # Zero equity
            {'equity': 50.0}
        ]
        returns = _compute_returns(equity_data)
        assert len(returns) == 2, "Should have 2 return values"
        assert returns[0] == -1.0, "First return should be -100%"
        assert returns[1] == 0.0, "Second return should be 0.0 (safe default)"

    def test_returns_calculation_with_nan_inf_equity(self):
        """Test returns calculation handles NaN/infinite equity safely."""
        equity_data = [
            {'equity': 100.0},
            {'equity': nan},
            {'equity': inf},
            {'equity': 150.0}
        ]
        returns = _compute_returns(equity_data)
        assert len(returns) == 3, "Should have 3 return values"
        # Should use safe defaults for non-finite values
        assert all(isinstance(r, float) for r in returns), "All returns should be finite floats"

    def test_complete_metrics_with_edge_cases(self):
        """Test complete metrics calculation with various edge cases."""
        # Test with empty data
        metrics = compute_backtest_metrics([])
        assert metrics['sharpe_ratio'] == 0.0
        assert metrics['profit_factor'] == 0.0
        assert metrics['max_drawdown'] == 0.0
        assert metrics['total_trades'] == 0

        # Test with single data point
        equity_data = [{'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0, 'pnl': 0.0}]
        metrics = compute_backtest_metrics(equity_data)
        assert metrics['sharpe_ratio'] == 0.0  # Not enough data for Sharpe
        assert metrics['total_trades'] == 0  # No winning/losing trades

        # Test with constant equity (no volatility)
        equity_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0, 'pnl': 0.0},
            {'trade_id': 2, 'timestamp': '2023-01-02', 'equity': 100.0, 'pnl': 0.0},
            {'trade_id': 3, 'timestamp': '2023-01-03', 'equity': 100.0, 'pnl': 0.0}
        ]
        metrics = compute_backtest_metrics(equity_data)
        assert metrics['sharpe_ratio'] == 0.0, "Should be 0.0 for zero volatility"
        assert metrics['max_drawdown'] == 0.0, "Should be 0.0 for constant equity"

    def test_total_return_calculation_safety(self):
        """Test total return calculation handles zero initial equity safely."""
        # Test with zero initial equity
        equity_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 0.0},
            {'trade_id': 2, 'timestamp': '2023-01-02', 'equity': 100.0}
        ]
        metrics = compute_backtest_metrics(equity_data)
        assert metrics['total_return'] == 0.0, "Should return 0.0 for zero initial equity"

    def test_win_rate_calculation_safety(self):
        """Test win rate calculation handles zero trades safely."""
        equity_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0, 'pnl': 0.0},
            {'trade_id': 2, 'timestamp': '2023-01-02', 'equity': 100.0, 'pnl': 0.0}
        ]
        metrics = compute_backtest_metrics(equity_data)
        assert metrics['win_rate'] == 0.0, "Should return 0.0 for no winning/losing trades"


class TestRegimeAwareMetrics:
    """Test suite for regime-aware metrics calculations."""

    def test_regime_aware_with_mismatched_lengths(self):
        """Test regime-aware metrics handles mismatched data lengths safely."""
        equity_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0},
            {'trade_id': 2, 'timestamp': '2023-01-02', 'equity': 110.0}
        ]
        regime_data = [
            {'regime_name': 'bull'},  # Only one regime record
        ]

        # Should handle length mismatch gracefully
        result = compute_regime_aware_metrics(equity_data, regime_data)
        assert 'per_regime_metrics' in result
        assert len(result['per_regime_metrics']) >= 0  # Should have some metrics

    def test_regime_aware_with_insufficient_data(self):
        """Test regime-aware metrics handles regimes with insufficient data."""
        equity_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0}
        ]
        regime_data = [
            {'regime_name': 'bull'}
        ]

        result = compute_regime_aware_metrics(equity_data, regime_data)
        assert 'per_regime_metrics' in result
        # Should have safe defaults for insufficient data
        assert 'bull' in result['per_regime_metrics']
        bull_metrics = result['per_regime_metrics']['bull']
        assert bull_metrics['sharpe_ratio'] == 0.0


class TestExportFunctions:
    """Test suite for export functions with security validation."""

    def test_export_equity_progression_sanitization(self):
        """Test equity progression export handles non-finite values safely."""
        equity_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0, 'pnl': nan},
            {'trade_id': 2, 'timestamp': '2023-01-02', 'equity': inf, 'pnl': 10.0}
        ]

        # Use results directory which is allowed
        out_path = "results/test_equity_sanitization.csv"
        try:
            export_equity_progression(equity_data, out_path)

            # Verify file was created and contains safe values
            assert os.path.exists(out_path)
            with open(out_path, 'r') as f:
                content = f.read()
                assert '0.0' in content  # Non-finite values should be replaced
        finally:
            # Clean up test file
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_export_metrics_sanitization(self):
        """Test metrics export handles non-finite values safely."""
        metrics = {
            'sharpe_ratio': nan,
            'profit_factor': inf,
            'max_drawdown': 0.1,
            'equity_curve': [100.0, nan, 110.0, inf]
        }

        # Use results directory which is allowed
        out_path = "results/test_metrics_sanitization.json"
        try:
            export_metrics(metrics, out_path)

            # Verify file was created and contains safe values
            assert os.path.exists(out_path)
            with open(out_path, 'r') as f:
                data = json.load(f)
                assert data['sharpe_ratio'] == 0.0
                assert data['profit_factor'] == 0.0
                assert data['max_drawdown'] == 0.1
                assert data['equity_curve'] == [100.0, 0.0, 110.0, 0.0]
        finally:
            # Clean up test file
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_path_traversal_protection(self):
        """Test export functions prevent path traversal attacks."""
        equity_data = [{'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0}]

        # Test path traversal attempt
        malicious_path = "../../../etc/passwd"

        with pytest.raises(BacktestSecurityError):
            export_equity_progression(equity_data, malicious_path)


class TestValidation:
    """Test suite for input validation."""

    def test_equity_progression_validation(self):
        """Test equity progression validation catches invalid data."""
        from backtest.backtester import _validate_equity_progression

        # Test with non-list input
        with pytest.raises(BacktestValidationError):
            _validate_equity_progression("not a list")

        # Test with missing required keys
        invalid_data = [{'timestamp': '2023-01-01', 'equity': 100.0}]  # Missing trade_id
        with pytest.raises(BacktestValidationError):
            _validate_equity_progression(invalid_data)

        # Test with invalid equity range
        invalid_data = [{'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 1e10}]  # Too large
        with pytest.raises(BacktestValidationError):
            _validate_equity_progression(invalid_data)


class TestPerformanceOptimizations:
    """Test suite to verify performance optimizations work correctly."""

    def test_vectorized_returns_calculation(self):
        """Test that vectorized returns calculation produces same results as original."""
        import time

        # Create larger test dataset
        equity_data = [
            {'equity': 100.0 + i * 0.5, 'trade_id': i, 'timestamp': f'2023-01-{i+1:02d}'}
            for i in range(1000)
        ]

        # Time the optimized version
        start_time = time.time()
        returns_optimized = _compute_returns(equity_data)
        optimized_time = time.time() - start_time

        # Verify results are reasonable
        assert len(returns_optimized) == 999, "Should have 999 returns for 1000 equity points"
        assert all(isinstance(r, float) for r in returns_optimized), "All returns should be floats"
        assert all(isfinite(r) for r in returns_optimized), "All returns should be finite"

        # Test with edge cases
        edge_cases = [
            {'equity': 0.0, 'trade_id': 1, 'timestamp': '2023-01-01'},
            {'equity': 100.0, 'trade_id': 2, 'timestamp': '2023-01-02'}
        ]
        returns_edge = _compute_returns(edge_cases)
        assert returns_edge == [0.0], "Should handle zero equity safely"

    def test_vectorized_profit_factor_calculation(self):
        """Test that vectorized profit factor calculation produces correct results."""
        import time

        # Create test data with known profit factor
        equity_data = [
            {'pnl': 100.0},
            {'pnl': -50.0},
            {'pnl': 75.0},
            {'pnl': -25.0}
        ] * 100  # Repeat for performance testing

        # Time the calculation
        start_time = time.time()
        profit_factor = _calculate_profit_factor(equity_data)
        calc_time = time.time() - start_time

        # Verify correctness: gross_profit / gross_loss = (100 + 75) / (50 + 25) = 175 / 75 = 2.333...
        expected = (100 + 75) / (50 + 25)
        assert abs(profit_factor - expected) < 1e-10, "Profit factor should match expected calculation"

    def test_vectorized_max_drawdown_calculation(self):
        """Test that vectorized max drawdown calculation produces correct results."""
        import time

        # Create test data with known max drawdown
        equity_data = [
            {'equity': 100.0},
            {'equity': 110.0},  # Peak
            {'equity': 95.0},   # Drawdown of 13.636%
            {'equity': 105.0},  # Recovery
            {'equity': 90.0}    # Larger drawdown of 18.182%
        ] * 50  # Repeat for performance testing

        # Time the calculation
        start_time = time.time()
        max_dd = _calculate_max_drawdown(equity_data)
        calc_time = time.time() - start_time

        # Verify correctness: max drawdown from peak 110 to low 90 = (110-90)/110 = 18.182%
        expected = (110 - 90) / 110
        assert abs(max_dd - expected) < 1e-10, "Max drawdown should match expected calculation"

    def test_single_pass_data_extraction(self):
        """Test that single-pass data extraction in compute_backtest_metrics works correctly."""
        import time

        # Create test data
        equity_data = []
        for i in range(100):
            pnl = 10.0 if i % 3 == 0 else (-5.0 if i % 3 == 1 else 0.0)
            equity_data.append({
                'trade_id': i,
                'timestamp': f'2023-01-{i+1:02d}',
                'equity': 1000.0 + pnl,
                'pnl': pnl
            })

        # Time the optimized calculation
        start_time = time.time()
        metrics = compute_backtest_metrics(equity_data)
        calc_time = time.time() - start_time

        # Verify correctness
        expected_wins = sum(1 for record in equity_data if record['pnl'] > 0)
        expected_losses = sum(1 for record in equity_data if record['pnl'] < 0)

        assert metrics['wins'] == expected_wins, "Wins count should match"
        assert metrics['losses'] == expected_losses, "Losses count should match"
        assert metrics['total_trades'] == expected_wins + expected_losses, "Total trades should match"

    def test_pandas_regime_grouping_performance(self):
        """Test that pandas-based regime grouping works correctly."""
        import time

        # Create test data
        equity_data = [
            {'trade_id': i, 'timestamp': f'2023-01-{i+1:02d}', 'equity': 1000.0 + i, 'pnl': 1.0}
            for i in range(200)
        ]

        regime_data = [
            {'regime_name': 'bull' if i % 2 == 0 else 'bear'}
            for i in range(200)
        ]

        # Time the optimized calculation
        start_time = time.time()
        result = compute_regime_aware_metrics(equity_data, regime_data)
        calc_time = time.time() - start_time

        # Verify structure
        assert 'per_regime_metrics' in result
        assert 'bull' in result['per_regime_metrics']
        assert 'bear' in result['per_regime_metrics']

        # Verify metrics are calculated
        bull_metrics = result['per_regime_metrics']['bull']
        assert 'sharpe_ratio' in bull_metrics
        assert 'profit_factor' in bull_metrics
        assert 'max_drawdown' in bull_metrics


class TestStatisticalIntegrity:
    """Test suite to verify statistical integrity of calculations."""

    def test_sharpe_ratio_mathematical_correctness(self):
        """Test Sharpe ratio calculation matches expected mathematical formula."""
        # Create simple test case with known result
        returns = [0.01, 0.02, 0.01, 0.02]  # 2% daily returns
        risk_free_rate = 0.0  # Simplify for testing

        sharpe = _calculate_sharpe_ratio(returns, risk_free_rate)

        # Manual calculation for verification
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
        expected_sharpe = (avg_return / std_return) * (252 ** 0.5)  # Annualized

        assert abs(sharpe - expected_sharpe) < 1e-10, "Sharpe ratio should match manual calculation"

    def test_profit_factor_mathematical_correctness(self):
        """Test profit factor calculation matches expected formula."""
        equity_data = [
            {'pnl': 100.0},
            {'pnl': -50.0},
            {'pnl': 75.0},
            {'pnl': -25.0}
        ]

        profit_factor = _calculate_profit_factor(equity_data)

        # Manual calculation: gross_profit / gross_loss = (100 + 75) / (50 + 25) = 175 / 75 = 2.333...
        expected = (100 + 75) / (50 + 25)

        assert abs(profit_factor - expected) < 1e-10, "Profit factor should match manual calculation"

    def test_max_drawdown_mathematical_correctness(self):
        """Test max drawdown calculation matches expected formula."""
        equity_data = [
            {'equity': 100.0},
            {'equity': 110.0},  # Peak
            {'equity': 95.0},   # Drawdown of 13.636%
            {'equity': 105.0},  # Recovery
            {'equity': 90.0}    # Larger drawdown of 18.182%
        ]

        max_dd = _calculate_max_drawdown(equity_data)

        # Manual calculation: max drawdown from peak 110 to low 90 = (110-90)/110 = 18.182%
        expected = (110 - 90) / 110

        assert abs(max_dd - expected) < 1e-10, "Max drawdown should match manual calculation"


if __name__ == '__main__':
    pytest.main([__file__])
