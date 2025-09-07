"""
Tests for regime-aware backtester functionality.
"""
import pytest
import pandas as pd
import json
import csv
import os
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from backtest.backtester import (
    compute_regime_aware_metrics,
    export_regime_aware_equity_progression,
    export_regime_aware_report,
    export_regime_aware_equity_from_botengine,
    _generate_regime_recommendations,
    _export_regime_csv_summary
)


class TestRegimeAwareMetrics:
    """Test regime-aware metrics computation."""

    def test_compute_regime_aware_metrics_empty_data(self):
        """Test handling of empty data."""
        result = compute_regime_aware_metrics([], [])

        assert "overall" in result
        assert "per_regime" in result
        assert "regime_summary" in result
        assert result["per_regime"] == {}
        assert result["regime_summary"] == {}

    def test_compute_regime_aware_metrics_no_regime_data(self):
        """Test handling when no regime data is provided."""
        equity_data = [
            {"trade_id": "1", "equity": 1000.0, "pnl": 0.0},
            {"trade_id": "2", "equity": 1100.0, "pnl": 100.0}
        ]

        result = compute_regime_aware_metrics(equity_data, None)

        assert "overall" in result
        assert result["overall"]["total_return"] == 0.1  # (1100-1000)/1000
        assert result["per_regime"] == {}
        assert result["regime_summary"] == {}

    def test_compute_regime_aware_metrics_with_regime_data(self):
        """Test metrics computation with regime data."""
        equity_data = [
            {"trade_id": "1", "equity": 1000.0, "pnl": 0.0},
            {"trade_id": "2", "equity": 1100.0, "pnl": 100.0},
            {"trade_id": "3", "equity": 1050.0, "pnl": -50.0},
            {"trade_id": "4", "equity": 1150.0, "pnl": 100.0}
        ]

        regime_data = [
            {"regime_name": "trend_up", "confidence_score": 0.8},
            {"regime_name": "trend_up", "confidence_score": 0.9},
            {"regime_name": "range_tight", "confidence_score": 0.7},
            {"regime_name": "range_tight", "confidence_score": 0.6}
        ]

        result = compute_regime_aware_metrics(equity_data, regime_data)

        # Check overall metrics
        assert "overall" in result
        assert result["overall"]["total_trades"] == 3  # 3 trades with pnl

        # Check per-regime metrics
        assert "per_regime" in result
        assert "trend_up" in result["per_regime"]
        assert "range_tight" in result["per_regime"]

        # Check trend_up regime
        trend_up = result["per_regime"]["trend_up"]
        assert trend_up["trade_count"] == 2
        assert trend_up["avg_confidence"] == pytest.approx(0.85, rel=1e-6)  # (0.8 + 0.9) / 2

        # Check range_tight regime
        range_tight = result["per_regime"]["range_tight"]
        assert range_tight["trade_count"] == 2  # Records 2 and 3
        assert range_tight["avg_confidence"] == pytest.approx(0.65, rel=1e-6)  # (0.7 + 0.6) / 2

        # Check regime summary
        assert "regime_summary" in result
        summary = result["regime_summary"]
        assert summary["total_regimes"] == 2
        assert summary["regime_distribution"]["trend_up"] == 2
        assert summary["regime_distribution"]["range_tight"] == 2

    def test_compute_regime_aware_metrics_single_regime(self):
        """Test metrics with only one regime."""
        equity_data = [
            {"trade_id": "1", "equity": 1000.0, "pnl": 100.0},
            {"trade_id": "2", "equity": 1100.0, "pnl": 200.0}
        ]

        regime_data = [
            {"regime_name": "trend_up", "confidence_score": 0.8},
            {"regime_name": "trend_up", "confidence_score": 0.9}
        ]

        result = compute_regime_aware_metrics(equity_data, regime_data)

        assert result["regime_summary"]["total_regimes"] == 1
        assert result["regime_summary"]["best_performing_regime"] == "trend_up"
        assert result["regime_summary"]["worst_performing_regime"] == "trend_up"

    def test_compute_regime_aware_metrics_missing_regime_data(self):
        """Test handling when regime data is shorter than equity data."""
        equity_data = [
            {"trade_id": "1", "equity": 1000.0, "pnl": 0.0},
            {"trade_id": "2", "equity": 1100.0, "pnl": 100.0},
            {"trade_id": "3", "equity": 1050.0, "pnl": -50.0}
        ]

        regime_data = [
            {"regime_name": "trend_up", "confidence_score": 0.8}
            # Missing regime data for last two records
        ]

        result = compute_regime_aware_metrics(equity_data, regime_data)

        # Should handle gracefully with "unknown" regime for missing data
        assert "unknown" in result["per_regime"]


class TestRegimeAwareExport:
    """Test regime-aware export functions."""

    def test_export_regime_aware_equity_progression(self):
        """Test CSV export with regime information."""
        equity_data = [
            {
                "trade_id": "1",
                "timestamp": "2023-01-01",
                "equity": 1000.0,
                "pnl": 0.0,
                "cumulative_return": 0.0,
                "regime_name": "trend_up",
                "confidence_score": 0.8,
                "regime_features": {"adx": 25.0}
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = os.path.join(temp_dir, "test_regime_equity.csv")
            result_path = export_regime_aware_equity_progression(equity_data, out_path)

            assert os.path.exists(result_path)

            # Verify CSV content
            with open(result_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert rows[0]["trade_id"] == "1"
            assert rows[0]["regime_name"] == "trend_up"
            assert rows[0]["confidence_score"] == "0.8"
            assert "adx" in rows[0]["regime_features"]

    def test_export_regime_aware_report(self):
        """Test comprehensive report export."""
        metrics = {
            "overall": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "win_rate": 0.6,
                "total_trades": 10
            },
            "per_regime": {
                "trend_up": {
                    "total_return": 0.25,
                    "sharpe_ratio": 1.8,
                    "win_rate": 0.8,
                    "total_trades": 5,
                    "avg_confidence": 0.85
                },
                "range_tight": {
                    "total_return": 0.05,
                    "sharpe_ratio": 0.5,
                    "win_rate": 0.4,
                    "total_trades": 3,
                    "avg_confidence": 0.7
                }
            },
            "regime_summary": {
                "total_regimes": 2,
                "best_performing_regime": "trend_up",
                "worst_performing_regime": "range_tight"
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = os.path.join(temp_dir, "test_report.json")
            result_path = export_regime_aware_report(metrics, out_path)

            assert os.path.exists(result_path)

            # Verify JSON content
            with open(result_path, 'r') as f:
                report = json.load(f)

            assert report["report_type"] == "regime_aware_backtest_report"
            assert "summary" in report
            assert "overall_performance" in report
            assert "regime_performance" in report
            assert "recommendations" in report

            # Check recommendations
            recommendations = report["recommendations"]
            assert "best_regime" in recommendations
            assert recommendations["best_regime"]["regime"] == "trend_up"
            assert "worst_regime" in recommendations
            assert recommendations["worst_regime"]["regime"] == "range_tight"

            # Check CSV summary was also created
            csv_path = os.path.splitext(result_path)[0] + "_summary.csv"
            assert os.path.exists(csv_path)


class TestRegimeRecommendations:
    """Test regime recommendation generation."""

    def test_generate_regime_recommendations_empty(self):
        """Test recommendations with no regime data."""
        metrics = {"per_regime": {}}
        recommendations = _generate_regime_recommendations(metrics)

        assert "general" in recommendations
        assert "Insufficient regime data" in recommendations["general"]

    def test_generate_regime_recommendations_full(self):
        """Test recommendations with complete regime data."""
        metrics = {
            "per_regime": {
                "trend_up": {
                    "total_return": 0.25,
                    "sharpe_ratio": 1.8
                },
                "range_tight": {
                    "total_return": 0.05,
                    "sharpe_ratio": 0.5
                },
                "volatile_spike": {
                    "total_return": -0.1,
                    "sharpe_ratio": -0.3
                }
            }
        }

        recommendations = _generate_regime_recommendations(metrics)

        assert "best_regime" in recommendations
        assert recommendations["best_regime"]["regime"] == "trend_up"
        assert recommendations["best_regime"]["return"] == 0.25

        assert "worst_regime" in recommendations
        assert recommendations["worst_regime"]["regime"] == "volatile_spike"
        assert recommendations["worst_regime"]["return"] == -0.1

        assert "risk_analysis" in recommendations
        assert recommendations["risk_analysis"]["most_volatile_regime"] == "volatile_spike"


class TestCSVSummaryExport:
    """Test CSV summary export functionality."""

    def test_export_regime_csv_summary(self):
        """Test CSV summary export."""
        metrics = {
            "overall": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "win_rate": 0.6,
                "max_drawdown": 0.05,
                "total_trades": 10
            },
            "per_regime": {
                "trend_up": {
                    "total_return": 0.25,
                    "sharpe_ratio": 1.8,
                    "win_rate": 0.8,
                    "max_drawdown": 0.03,
                    "total_trades": 5,
                    "avg_confidence": 0.85
                }
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "test_summary.csv")
            _export_regime_csv_summary(metrics, csv_path)

            assert os.path.exists(csv_path)

            # Verify CSV content
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Check header
            assert rows[0] == ["Regime", "Total Return", "Sharpe Ratio", "Win Rate",
                              "Max Drawdown", "Total Trades", "Avg Confidence"]

            # Check overall row
            assert rows[1][0] == "OVERALL"
            assert rows[1][1] == "0.1500"
            assert rows[1][2] == "1.2000"

            # Check regime row
            assert rows[2][0] == "TREND_UP"
            assert rows[2][1] == "0.2500"
            assert rows[2][6] == "0.8500"


class TestBotEngineIntegration:
    """Test integration with BotEngine."""

    @patch('backtest.backtester.export_regime_aware_equity_progression')
    @patch('backtest.backtester.compute_regime_aware_metrics')
    @patch('backtest.backtester.export_regime_aware_report')
    def test_export_regime_aware_equity_from_botengine(self, mock_export_report,
                                                       mock_compute_metrics,
                                                       mock_export_csv):
        """Test BotEngine integration function."""
        # Mock bot engine
        mock_bot = Mock()
        mock_bot.performance_stats = {
            "equity_progression": [
                {"trade_id": "1", "equity": 1000.0, "pnl": 0.0}
            ]
        }

        # Mock regime detector
        mock_detector = Mock()
        mock_regime_result = Mock()
        mock_regime_result.regime_name = "trend_up"
        mock_regime_result.confidence_score = 0.8
        mock_regime_result.reasons = {"adx": 25.0}
        mock_detector.detect_enhanced_regime.return_value = mock_regime_result

        # Mock data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=50, freq='D'),
            'open': [100] * 50,
            'high': [105] * 50,
            'low': [95] * 50,
            'close': [102] * 50,
            'volume': [1000] * 50
        })

        # Mock return values
        mock_export_csv.return_value = "/path/to/csv"
        mock_compute_metrics.return_value = {"overall": {}, "per_regime": {}, "regime_summary": {}}

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = os.path.join(temp_dir, "test_output.csv")

            result = export_regime_aware_equity_from_botengine(
                mock_bot, mock_detector, data, out_path
            )

            # Verify calls were made
            mock_export_csv.assert_called_once()
            mock_compute_metrics.assert_called_once()
            mock_export_report.assert_called_once()

    def test_export_regime_aware_equity_from_botengine_empty_equity(self):
        """Test handling of empty equity progression."""
        mock_bot = Mock()
        mock_bot.performance_stats = {"equity_progression": []}

        mock_detector = Mock()
        data = pd.DataFrame()

        result = export_regime_aware_equity_from_botengine(mock_bot, mock_detector, data)

        assert result == ""


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_regime_data_longer_than_equity(self):
        """Test when regime data is longer than equity progression."""
        equity_data = [
            {"trade_id": "1", "equity": 1000.0, "pnl": 0.0}
        ]

        regime_data = [
            {"regime_name": "trend_up", "confidence_score": 0.8},
            {"regime_name": "range_tight", "confidence_score": 0.7}  # Extra regime data
        ]

        result = compute_regime_aware_metrics(equity_data, regime_data)

        # Should only use regime data that matches equity records
        assert result["per_regime"]["trend_up"]["trade_count"] == 1

    def test_regime_data_with_errors(self):
        """Test handling of regime detection errors."""
        equity_data = [
            {"trade_id": "1", "equity": 1000.0, "pnl": 0.0}
        ]

        regime_data = [
            {"regime_name": "error", "confidence_score": 0.0, "regime_features": {"error": "detection failed"}}
        ]

        result = compute_regime_aware_metrics(equity_data, regime_data)

        assert "error" in result["per_regime"]
        assert result["per_regime"]["error"]["trade_count"] == 1

    def test_insufficient_data_regime(self):
        """Test handling of insufficient data regime."""
        equity_data = [
            {"trade_id": "1", "equity": 1000.0, "pnl": 0.0}
        ]

        regime_data = [
            {"regime_name": "insufficient_data", "confidence_score": 0.0, "regime_features": {}}
        ]

        result = compute_regime_aware_metrics(equity_data, regime_data)

        assert "insufficient_data" in result["per_regime"]
        assert result["per_regime"]["insufficient_data"]["avg_confidence"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
