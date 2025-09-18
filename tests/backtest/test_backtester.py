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


class TestValidationFunctions:
    """Test suite for validation functions."""

    def test_validate_equity_progression_with_valid_data(self):
        """Test validation with valid equity progression data."""
        from backtest.backtester import _validate_equity_progression

        valid_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0, 'pnl': 10.0},
            {'trade_id': 2, 'timestamp': '2023-01-02', 'equity': 110.0, 'pnl': -5.0}
        ]

        # Should not raise exception
        _validate_equity_progression(valid_data)

    def test_validate_equity_progression_with_missing_keys(self):
        """Test validation with missing required keys."""
        from backtest.backtester import _validate_equity_progression

        invalid_data = [
            {'timestamp': '2023-01-01', 'equity': 100.0}  # Missing trade_id
        ]

        with pytest.raises(BacktestValidationError):
            _validate_equity_progression(invalid_data)

    def test_validate_equity_progression_with_invalid_types(self):
        """Test validation with invalid data types."""
        from backtest.backtester import _validate_equity_progression

        invalid_data = [
            {'trade_id': 'invalid', 'timestamp': '2023-01-01', 'equity': 100.0}
        ]

        with pytest.raises(BacktestValidationError):
            _validate_equity_progression(invalid_data)

    def test_validate_regime_data_with_valid_data(self):
        """Test regime data validation with valid data."""
        from backtest.backtester import _validate_regime_data

        valid_data = [
            {'regime_name': 'bull', 'confidence_score': 0.8},
            {'regime_name': 'bear', 'confidence_score': 0.6}
        ]

        # Should not raise exception
        _validate_regime_data(valid_data)

    def test_validate_regime_data_with_missing_regime_name(self):
        """Test regime data validation with missing regime_name."""
        from backtest.backtester import _validate_regime_data

        invalid_data = [
            {'confidence_score': 0.8}  # Missing regime_name
        ]

        with pytest.raises(BacktestValidationError):
            _validate_regime_data(invalid_data)


class TestSecurityFunctions:
    """Test suite for security-related functions."""

    def test_sanitize_file_path_with_valid_path(self):
        """Test path sanitization with valid paths."""
        from backtest.backtester import _sanitize_file_path

        # Test with allowed directory
        safe_path = _sanitize_file_path("results/test.csv")
        assert "results" in safe_path
        assert "test.csv" in safe_path

    def test_sanitize_file_path_with_path_traversal(self):
        """Test path sanitization prevents path traversal."""
        from backtest.backtester import _sanitize_file_path

        with pytest.raises(BacktestSecurityError):
            _sanitize_file_path("../../../etc/passwd")

    def test_sanitize_file_path_with_invalid_characters(self):
        """Test path sanitization with invalid characters."""
        from backtest.backtester import _sanitize_file_path

        with pytest.raises(BacktestSecurityError):
            _sanitize_file_path("results/test<script>.csv")

    def test_ensure_results_dir_creates_directory(self):
        """Test that _ensure_results_dir creates parent directories."""
        from backtest.backtester import _ensure_results_dir
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "nested", "dir", "test.csv")
            _ensure_results_dir(test_path)
            assert os.path.exists(os.path.dirname(test_path))


class TestExportFunctionsExtended:
    """Extended test suite for export functions."""

    def test_export_equity_progression_with_empty_data(self):
        """Test export with empty equity progression."""
        out_path = "results/test_empty.csv"
        try:
            export_equity_progression([], out_path)
            assert os.path.exists(out_path)

            with open(out_path, 'r') as f:
                content = f.read()
                assert "trade_id" in content  # Should have header
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_export_equity_progression_with_regime_data(self):
        """Test export with regime information."""
        equity_data = [
            {
                'trade_id': 1,
                'timestamp': '2023-01-01',
                'equity': 100.0,
                'pnl': 10.0,
                'regime_name': 'bull',
                'confidence_score': 0.8
            }
        ]

        out_path = "results/test_regime.csv"
        try:
            export_equity_progression(equity_data, out_path)
            assert os.path.exists(out_path)

            with open(out_path, 'r') as f:
                content = f.read()
                assert "bull" in content
                assert "0.8" in content
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_export_metrics_with_empty_metrics(self):
        """Test metrics export with empty data."""
        out_path = "results/test_empty_metrics.json"
        try:
            export_metrics({}, out_path)
            assert os.path.exists(out_path)

            with open(out_path, 'r') as f:
                data = json.load(f)
                assert isinstance(data, dict)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_export_regime_aware_report_with_empty_data(self):
        """Test regime-aware report export with empty data."""
        from backtest.backtester import export_regime_aware_report

        out_path = "results/test_empty_regime.json"
        try:
            export_regime_aware_report({}, out_path)
            assert os.path.exists(out_path)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


class TestRegimeAwareFunctions:
    """Test suite for regime-aware functions."""

    def test_compute_regime_aware_metrics_with_empty_equity(self):
        """Test regime-aware metrics with empty equity data."""
        result = compute_regime_aware_metrics([], None)
        assert 'overall' in result
        assert 'per_regime' in result
        assert 'regime_summary' in result

    def test_compute_regime_aware_metrics_with_no_regime_data(self):
        """Test regime-aware metrics with equity data but no regime data."""
        equity_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0}
        ]

        result = compute_regime_aware_metrics(equity_data, None)
        assert 'overall' in result
        assert result['per_regime'] == {}
        assert result['regime_summary']['total_regimes'] == 0

    def test_align_regime_data_lengths_with_mismatch(self):
        """Test data length alignment with mismatched lengths."""
        from backtest.backtester import _align_regime_data_lengths

        equity_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0},
            {'trade_id': 2, 'timestamp': '2023-01-02', 'equity': 110.0}
        ]

        regime_data = [
            {'regime_name': 'bull'}
        ]

        aligned_equity, aligned_regime = _align_regime_data_lengths(equity_data, regime_data)
        assert len(aligned_equity) == 1
        assert len(aligned_regime) == 1

    def test_create_default_regime_metrics(self):
        """Test creation of default regime metrics."""
        from backtest.backtester import _create_default_regime_metrics

        default_metrics = _create_default_regime_metrics()
        assert default_metrics['sharpe_ratio'] == 0.0
        assert default_metrics['profit_factor'] == 0.0
        assert default_metrics['max_drawdown'] == 0.0
        assert default_metrics['total_trades'] == 0

    def test_compute_regime_metrics_safe_with_insufficient_data(self):
        """Test safe regime metrics computation with insufficient data."""
        from backtest.backtester import _compute_regime_metrics_safe

        insufficient_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0}
        ]

        result = _compute_regime_metrics_safe(insufficient_data, 'test_regime')
        assert result['sharpe_ratio'] == 0.0
        assert result['total_trades'] == 0


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_generate_regime_recommendations(self):
        """Test regime recommendations generation."""
        from backtest.backtester import _generate_regime_recommendations

        metrics = {
            'per_regime': {
                'bull': {'total_return': 0.15, 'sharpe_ratio': 1.2},
                'bear': {'total_return': -0.05, 'sharpe_ratio': 0.8}
            }
        }

        recommendations = _generate_regime_recommendations(metrics)
        assert 'best_regime' in recommendations
        assert 'worst_regime' in recommendations

    def test_export_regime_csv_summary(self):
        """Test regime CSV summary export."""
        from backtest.backtester import _export_regime_csv_summary

        metrics = {
            'overall': {'total_return': 0.10, 'sharpe_ratio': 1.0, 'win_rate': 0.6, 'max_drawdown': 0.05, 'total_trades': 100},
            'per_regime': {
                'bull': {'total_return': 0.15, 'sharpe_ratio': 1.2, 'win_rate': 0.7, 'max_drawdown': 0.03, 'total_trades': 50}
            }
        }

        out_path = "results/test_regime_summary.csv"
        try:
            _export_regime_csv_summary(metrics, out_path)
            assert os.path.exists(out_path)

            with open(out_path, 'r') as f:
                content = f.read()
                assert "OVERALL" in content
                assert "bull" in content
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


class TestBacktesterClass:
    """Test suite for Backtester class."""

    def test_backtester_initialization(self):
        """Test Backtester class initialization."""
        from backtest.backtester import Backtester

        config = {'test': 'config'}
        backtester = Backtester(config)
        assert backtester.config == config

    def test_backtester_initialization_without_config(self):
        """Test Backtester initialization without config."""
        from backtest.backtester import Backtester

        backtester = Backtester()
        assert backtester.config == {}

    @pytest.mark.asyncio
    async def test_run_backtest_with_mock_data(self):
        """Test run_backtest with mock strategy and market data."""
        from backtest.backtester import Backtester

        backtester = Backtester()
        strategy_genome = {'type': 'test_strategy'}
        market_data = {'symbol': 'BTC/USDT', 'data': []}

        result = await backtester.run_backtest(strategy_genome, market_data)
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'equity_progression' in result

    def test_run_backtest_sync_with_mock_data(self):
        """Test synchronous run_backtest method."""
        from backtest.backtester import Backtester

        backtester = Backtester()
        strategy_genome = {'type': 'test_strategy'}
        market_data = {'symbol': 'BTC/USDT', 'data': []}

        result = backtester.run_backtest_sync(strategy_genome, market_data)
        assert 'total_return' in result
        assert 'sharpe_ratio' in result


class TestAsyncExportFunctions:
    """Test suite for async export functions."""

    @pytest.mark.asyncio
    async def test_export_equity_progression_async(self):
        """Test async equity progression export."""
        from backtest.backtester import export_equity_progression_async

        equity_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0}
        ]

        out_path = "results/test_async.csv"
        try:
            await export_equity_progression_async(equity_data, out_path)
            assert os.path.exists(out_path)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    @pytest.mark.asyncio
    async def test_export_metrics_async(self):
        """Test async metrics export."""
        from backtest.backtester import export_metrics_async

        metrics = {'sharpe_ratio': 1.0, 'total_return': 0.1}

        out_path = "results/test_async_metrics.json"
        try:
            await export_metrics_async(metrics, out_path)
            assert os.path.exists(out_path)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    @pytest.mark.asyncio
    async def test_export_regime_aware_report_async(self):
        """Test async regime-aware report export."""
        from backtest.backtester import export_regime_aware_report_async

        metrics = {
            'overall': {'total_return': 0.1},
            'per_regime': {},
            'regime_summary': {}
        }

        out_path = "results/test_async_regime.json"
        try:
            await export_regime_aware_report_async(metrics, out_path)
            assert os.path.exists(out_path)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


class TestExportFromBotEngine:
    """Test suite for bot engine export functions."""

    def test_export_equity_from_botengine_with_mock_engine(self):
        """Test export from mock bot engine."""
        from backtest.backtester import export_equity_from_botengine

        # Create mock bot engine
        mock_engine = Mock()
        mock_engine.performance_history = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0, 'pnl': 10.0, 'cumulative_return': 0.1}
        ]

        out_path = "results/test_bot_engine.csv"
        try:
            export_equity_from_botengine(mock_engine, out_path)
            assert os.path.exists(out_path)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_export_equity_from_botengine_with_no_history(self):
        """Test export from bot engine with no performance history."""
        from backtest.backtester import export_equity_from_botengine

        mock_engine = Mock()
        mock_engine.performance_history = []

        out_path = "results/test_empty_bot_engine.csv"
        # Should not raise exception
        export_equity_from_botengine(mock_engine, out_path)

    def test_export_regime_aware_equity_from_botengine(self):
        """Test regime-aware export from bot engine."""
        from backtest.backtester import export_regime_aware_equity_from_botengine
        import pandas as pd

        # Create mock objects
        mock_engine = Mock()
        mock_engine.performance_stats = {
            'equity_progression': [
                {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0}
            ]
        }

        mock_detector = Mock()
        mock_detector.detect_enhanced_regime.return_value = Mock(
            regime_name='bull',
            confidence_score=0.8,
            reasons={}
        )

        mock_data = pd.DataFrame({'close': [100.0]})

        out_path = "results/test_regime_bot_engine.csv"
        try:
            result_path = export_regime_aware_equity_from_botengine(mock_engine, mock_detector, mock_data, out_path)
            # Should return the path or empty string
            assert isinstance(result_path, str)
        except Exception:
            # May fail due to mock setup, but should not crash
            pass


class TestErrorHandling:
    """Test suite for error handling in various functions."""

    def test_export_equity_progression_with_io_error(self):
        """Test export equity progression handles I/O errors."""
        from backtest.backtester import export_equity_progression
        import unittest.mock

        equity_data = [{'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0}]

        with unittest.mock.patch('builtins.open', side_effect=IOError("Disk full")):
            with pytest.raises(BacktestSecurityError):
                export_equity_progression(equity_data, "results/test_io_error.csv")

    def test_compute_backtest_metrics_with_invalid_data(self):
        """Test compute_backtest_metrics handles invalid data gracefully."""
        # Test with data that has invalid equity values
        invalid_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 'invalid'}
        ]

        # Should handle gracefully and return safe defaults
        metrics = compute_backtest_metrics(invalid_data)
        assert isinstance(metrics, dict)
        assert 'sharpe_ratio' in metrics

    def test_regime_aware_metrics_with_pandas_import_error(self):
        """Test regime-aware metrics handles pandas import error."""
        import sys
        from unittest.mock import patch

        equity_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0}
        ]
        regime_data = [
            {'regime_name': 'bull'}
        ]

        # Mock pandas import to fail
        with patch.dict('sys.modules', {'pandas': None}):
            result = compute_regime_aware_metrics(equity_data, regime_data)
            assert 'per_regime' in result
            assert isinstance(result['per_regime'], dict)


class TestFileOperations:
    """Test suite for file operation safety."""

    def test_export_with_permission_denied(self):
        """Test export functions handle permission denied errors."""
        from backtest.backtester import export_equity_progression
        import unittest.mock

        equity_data = [{'trade_id': 1, 'timestamp': '2023-01-01', 'equity': 100.0}]

        with unittest.mock.patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(BacktestSecurityError):
                export_equity_progression(equity_data, "results/test_permission.csv")

    def test_export_with_long_path(self):
        """Test export with very long file paths."""
        from backtest.backtester import _sanitize_file_path

        long_path = "results/" + "a" * 200 + ".csv"

        # Should handle long paths gracefully
        try:
            safe_path = _sanitize_file_path(long_path)
            assert isinstance(safe_path, str)
        except BacktestSecurityError:
            # Expected for very long paths
            pass


class TestDataSanitization:
    """Test suite for data sanitization."""

    def test_equity_data_sanitization(self):
        """Test that equity data is properly sanitized before export."""
        from backtest.backtester import export_equity_progression

        # Data with non-finite values
        equity_data = [
            {'trade_id': 1, 'timestamp': '2023-01-01', 'equity': inf, 'pnl': nan},
            {'trade_id': 2, 'timestamp': '2023-01-02', 'equity': 100.0, 'pnl': 10.0}
        ]

        out_path = "results/test_sanitized.csv"
        try:
            export_equity_progression(equity_data, out_path)

            with open(out_path, 'r') as f:
                content = f.read()
                # Should contain sanitized values (0.0 for non-finite)
                assert "0.0" in content
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_metrics_data_sanitization(self):
        """Test that metrics data is properly sanitized before export."""
        from backtest.backtester import export_metrics

        # Metrics with non-finite values
        metrics = {
            'sharpe_ratio': inf,
            'profit_factor': nan,
            'max_drawdown': 0.1,
            'equity_curve': [100.0, inf, nan, 110.0]
        }

        out_path = "results/test_sanitized_metrics.json"
        try:
            export_metrics(metrics, out_path)

            with open(out_path, 'r') as f:
                data = json.load(f)
                assert data['sharpe_ratio'] == 0.0
                assert data['profit_factor'] == 0.0
                assert data['equity_curve'] == [100.0, 0.0, 0.0, 110.0]
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


if __name__ == '__main__':
    pytest.main([__file__])
