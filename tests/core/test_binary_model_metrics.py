"""
Comprehensive tests for BinaryModelMetricsCollector - binary model monitoring and metrics.

Tests metrics collection, performance tracking, calibration monitoring, drift detection,
alert generation, and health monitoring for the binary entry model.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.binary_model_metrics import (
    BinaryModelMetricsCollector,
    create_binary_model_metrics_collector,
    get_binary_model_metrics_collector,
)
from core.metrics_collector import MetricsCollector


class TestBinaryModelMetricsCollectorInitialization:
    """Test BinaryModelMetricsCollector initialization and configuration."""

    def test_default_initialization(self):
        """Test BinaryModelMetricsCollector with default parameters."""
        collector = BinaryModelMetricsCollector({})

        assert collector.enabled == True
        assert collector.metrics_window_hours == 24
        assert collector.max_history_size == 10000
        assert collector.alert_cooldown_minutes == 15
        assert len(collector.prediction_history) == 0
        assert len(collector.decision_history) == 0
        assert isinstance(collector.regime_performance, dict)
        assert isinstance(collector.calibration_stats, dict)

    def test_custom_initialization(self):
        """Test BinaryModelMetricsCollector with custom parameters."""
        config = {
            "binary_model_metrics": {"enabled": False},
            "drift_detection": {
                "trade_frequency_change": 0.3,
                "accuracy_drop": 0.15,
                "calibration_error": 0.25,
            },
            "metrics_window_hours": 48,
            "max_history_size": 5000,
            "alert_cooldown_minutes": 30,
        }

        collector = BinaryModelMetricsCollector(config)

        assert collector.enabled == False
        assert collector.metrics_window_hours == 48
        assert collector.max_history_size == 5000
        assert collector.alert_cooldown_minutes == 30
        assert collector.drift_thresholds["trade_frequency_change"] == 0.3
        assert collector.drift_thresholds["accuracy_drop"] == 0.15
        assert collector.drift_thresholds["calibration_error"] == 0.25

    def test_drift_thresholds_initialization(self):
        """Test drift thresholds are properly initialized."""
        collector = BinaryModelMetricsCollector({})

        expected_thresholds = {
            "trade_frequency_change": 0.5,
            "accuracy_drop": 0.1,
            "calibration_error": 0.2,
        }

        assert collector.drift_thresholds == expected_thresholds


class TestPredictionRecording:
    """Test prediction recording functionality."""

    @pytest.fixture
    def collector(self):
        """Create a test collector."""
        return BinaryModelMetricsCollector({})

    def test_record_prediction_enabled(self, collector):
        """Test recording prediction when enabled."""
        features = {"RSI": 65.0, "MACD": 0.5}

        collector.record_prediction(
            symbol="BTC/USDT",
            probability=0.75,
            threshold=0.6,
            regime="bullish",
            features=features,
        )

        assert len(collector.prediction_history) == 1
        prediction = collector.prediction_history[0]

        assert prediction["symbol"] == "BTC/USDT"
        assert prediction["probability"] == 0.75
        assert prediction["threshold"] == 0.6
        assert prediction["regime"] == "bullish"
        assert prediction["features"] == features
        assert prediction["decision"] == "trade"
        assert isinstance(prediction["timestamp"], datetime)

    def test_record_prediction_disabled(self):
        """Test recording prediction when disabled."""
        config = {"binary_model_metrics": {"enabled": False}}
        collector = BinaryModelMetricsCollector(config)

        collector.record_prediction(
            symbol="BTC/USDT",
            probability=0.75,
            threshold=0.6,
            regime="bullish",
            features={"RSI": 65.0},
        )

        assert len(collector.prediction_history) == 0

    def test_record_prediction_skip_decision(self, collector):
        """Test recording prediction that results in skip decision."""
        collector.record_prediction(
            symbol="BTC/USDT",
            probability=0.45,
            threshold=0.6,
            regime="bearish",
            features={"RSI": 30.0},
        )

        assert len(collector.prediction_history) == 1
        prediction = collector.prediction_history[0]

        assert prediction["decision"] == "skip"
        assert prediction["probability"] == 0.45

    def test_prediction_history_size_limit(self, collector):
        """Test prediction history respects size limit."""
        collector.max_history_size = 3

        # Add more predictions than limit
        for i in range(5):
            collector.record_prediction(
                symbol=f"BTC/USDT_{i}",
                probability=0.6 + i * 0.1,
                threshold=0.6,
                regime="bullish",
                features={"RSI": 50.0 + i},
            )

        assert len(collector.prediction_history) == 3
        # Should keep most recent
        assert collector.prediction_history[0]["symbol"] == "BTC/USDT_2"
        assert collector.prediction_history[2]["symbol"] == "BTC/USDT_4"


class TestDecisionOutcomeRecording:
    """Test decision outcome recording functionality."""

    @pytest.fixture
    def collector(self):
        """Create a test collector."""
        return BinaryModelMetricsCollector({})

    def test_record_decision_outcome_enabled(self, collector):
        """Test recording decision outcome when enabled."""
        collector.record_decision_outcome(
            symbol="BTC/USDT",
            decision="trade",
            outcome="profit",
            pnl=150.0,
            regime="bullish",
            strategy="RSIStrategy",
        )

        assert len(collector.decision_history) == 1
        decision = collector.decision_history[0]

        assert decision["symbol"] == "BTC/USDT"
        assert decision["decision"] == "trade"
        assert decision["outcome"] == "profit"
        assert decision["pnl"] == 150.0
        assert decision["regime"] == "bullish"
        assert decision["strategy"] == "RSIStrategy"
        assert decision["was_correct"] == True
        assert isinstance(decision["timestamp"], datetime)

    def test_record_decision_outcome_disabled(self):
        """Test recording decision outcome when disabled."""
        config = {"binary_model_metrics": {"enabled": False}}
        collector = BinaryModelMetricsCollector(config)

        collector.record_decision_outcome(
            symbol="BTC/USDT",
            decision="trade",
            outcome="profit",
            pnl=150.0,
            regime="bullish",
            strategy="RSIStrategy",
        )

        assert len(collector.decision_history) == 0

    def test_record_decision_outcome_incorrect(self, collector):
        """Test recording incorrect decision outcome."""
        collector.record_decision_outcome(
            symbol="BTC/USDT",
            decision="trade",
            outcome="loss",
            pnl=-100.0,
            regime="bullish",
            strategy="RSIStrategy",
        )

        assert len(collector.decision_history) == 1
        decision = collector.decision_history[0]

        assert decision["was_correct"] == False
        assert decision["pnl"] == -100.0

    def test_record_decision_outcome_skip_correct(self, collector):
        """Test recording skip decision that was correct."""
        collector.record_decision_outcome(
            symbol="BTC/USDT",
            decision="skip",
            outcome="loss",
            pnl=0.0,
            regime="bearish",
            strategy="none",
        )

        assert len(collector.decision_history) == 1
        decision = collector.decision_history[0]

        assert (
            decision["was_correct"] == True
        )  # Skip was correct since outcome was loss

    def test_decision_history_size_limit(self, collector):
        """Test decision history respects size limit."""
        collector.max_history_size = 2

        # Add more decisions than limit
        for i in range(4):
            collector.record_decision_outcome(
                symbol=f"BTC/USDT_{i}",
                decision="trade",
                outcome="profit",
                pnl=100.0,
                regime="bullish",
                strategy="RSIStrategy",
            )

        assert len(collector.decision_history) == 2
        # Should keep most recent
        assert collector.decision_history[0]["symbol"] == "BTC/USDT_2"
        assert collector.decision_history[1]["symbol"] == "BTC/USDT_3"

    def test_regime_performance_tracking(self, collector):
        """Test regime performance tracking."""
        # Add decisions for different regimes
        collector.record_decision_outcome(
            symbol="BTC/USDT",
            decision="trade",
            outcome="profit",
            pnl=100.0,
            regime="bullish",
            strategy="RSIStrategy",
        )

        collector.record_decision_outcome(
            symbol="ETH/USDT",
            decision="trade",
            outcome="loss",
            pnl=-50.0,
            regime="bearish",
            strategy="MACDStrategy",
        )

        collector.record_decision_outcome(
            symbol="ADA/USDT",
            decision="skip",
            outcome="neutral",
            pnl=0.0,
            regime="bullish",
            strategy="none",
        )

        # Check bullish regime stats
        bullish_stats = collector.regime_performance["bullish"]
        assert bullish_stats["total_decisions"] == 2
        assert bullish_stats["correct_decisions"] == 2  # Both were correct
        assert bullish_stats["total_pnl"] == 100.0
        assert bullish_stats["winning_trades"] == 1
        assert bullish_stats["total_trades"] == 1

        # Check bearish regime stats
        bearish_stats = collector.regime_performance["bearish"]
        assert bearish_stats["total_decisions"] == 1
        assert bearish_stats["correct_decisions"] == 0  # Trade was incorrect
        assert bearish_stats["total_pnl"] == -50.0
        assert bearish_stats["winning_trades"] == 0
        assert bearish_stats["total_trades"] == 1


class TestMetricsCalculation:
    """Test metrics calculation methods."""

    @pytest.fixture
    def collector(self):
        """Create a test collector with sample data."""
        collector = BinaryModelMetricsCollector({})
        collector.metrics_window_hours = 1  # Short window for testing

        # Add sample predictions
        base_time = datetime.now()
        for i in range(10):
            prediction = {
                "timestamp": base_time - timedelta(minutes=i * 5),
                "symbol": f"BTC/USDT_{i}",
                "probability": 0.5 + i * 0.05,
                "threshold": 0.6,
                "regime": "bullish" if i % 2 == 0 else "bearish",
                "features": {"RSI": 50.0 + i},
                "decision": "trade" if (0.5 + i * 0.05) >= 0.6 else "skip",
            }
            collector.prediction_history.append(prediction)

        # Add sample decisions
        for i in range(8):
            decision = {
                "timestamp": base_time - timedelta(minutes=i * 5),
                "symbol": f"BTC/USDT_{i}",
                "decision": "trade" if i % 2 == 0 else "skip",
                "outcome": "profit" if i % 3 == 0 else "loss",
                "pnl": 100.0 if i % 3 == 0 else -50.0,
                "regime": "bullish" if i % 2 == 0 else "bearish",
                "strategy": "RSIStrategy" if i % 2 == 0 else "none",
                "was_correct": (i % 2 == 0 and i % 3 == 0)
                or (i % 2 == 1 and i % 3 != 0),
            }
            collector.decision_history.append(decision)

        return collector

    def test_calculate_average_ptrade(self, collector):
        """Test average p_trade calculation."""
        avg_ptrade = collector._calculate_average_ptrade()

        assert avg_ptrade is not None
        assert isinstance(avg_ptrade, float)
        assert 0.5 <= avg_ptrade <= 1.0  # Should be in reasonable range

    def test_calculate_average_ptrade_empty_history(self):
        """Test average p_trade calculation with empty history."""
        collector = BinaryModelMetricsCollector({})

        avg_ptrade = collector._calculate_average_ptrade()

        assert avg_ptrade is None

    def test_calculate_regime_trade_counts(self, collector):
        """Test regime trade counts calculation."""
        regime_counts = collector._calculate_regime_trade_counts()

        assert isinstance(regime_counts, dict)
        # Should have counts for regimes that had trades
        assert "bullish" in regime_counts or "bearish" in regime_counts

    def test_calculate_hit_rate_metrics(self, collector):
        """Test hit rate metrics calculation."""
        metrics = collector._calculate_hit_rate_metrics()

        assert isinstance(metrics, dict)

        # Should have basic metrics
        expected_keys = ["accuracy", "precision", "recall", "f1_score"]
        for key in expected_keys:
            if key in metrics:
                assert isinstance(metrics[key], float)
                assert 0.0 <= metrics[key] <= 1.0

    def test_calculate_hit_rate_metrics_empty_history(self):
        """Test hit rate metrics calculation with empty history."""
        collector = BinaryModelMetricsCollector({})

        metrics = collector._calculate_hit_rate_metrics()

        assert metrics == {}

    def test_calculate_model_health_metrics(self, collector):
        """Test model health metrics calculation."""
        metrics = collector._calculate_model_health_metrics()

        assert isinstance(metrics, dict)

        # Should have health metrics
        expected_keys = [
            "prediction_stability",
            "trade_decision_ratio",
            "feature_count",
        ]
        for key in expected_keys:
            if key in metrics:
                assert isinstance(metrics[key], (int, float))

    def test_calculate_model_health_metrics_empty_history(self):
        """Test model health metrics calculation with empty history."""
        collector = BinaryModelMetricsCollector({})

        metrics = collector._calculate_model_health_metrics()

        assert metrics == {}

    def test_calculate_calibration_metrics(self, collector):
        """Test calibration metrics calculation."""
        metrics = collector._calculate_calibration_metrics()

        assert isinstance(metrics, dict)

        # May have calibration metrics if data allows
        if metrics:
            for key, value in metrics.items():
                assert isinstance(value, (int, float))

    def test_calculate_drift_indicators(self, collector):
        """Test drift indicators calculation."""
        indicators = collector._calculate_drift_indicators()

        assert isinstance(indicators, dict)

        # May have drift indicators if enough data
        for key, value in indicators.items():
            assert isinstance(value, (int, float))

    def test_calculate_drift_indicators_insufficient_data(self):
        """Test drift indicators calculation with insufficient data."""
        collector = BinaryModelMetricsCollector({})

        # Add minimal data
        for i in range(50):
            collector.prediction_history.append(
                {"timestamp": datetime.now(), "probability": 0.6, "decision": "trade"}
            )

        indicators = collector._calculate_drift_indicators()

        # Should return empty dict for insufficient data
        assert indicators == {}


class TestMetricsCollection:
    """Test metrics collection functionality."""

    @pytest.fixture
    def collector(self):
        """Create a test collector."""
        return BinaryModelMetricsCollector({})

    @pytest.fixture
    def mock_metrics_collector(self):
        """Create a mock metrics collector."""
        collector = Mock(spec=MetricsCollector)
        collector.record_metric = AsyncMock()
        return collector

    @pytest.mark.asyncio
    async def test_collect_binary_model_metrics_enabled(
        self, collector, mock_metrics_collector
    ):
        """Test collecting binary model metrics when enabled."""
        # Add some sample data
        collector.prediction_history = [
            {
                "timestamp": datetime.now(),
                "probability": 0.7,
                "threshold": 0.6,
                "regime": "bullish",
                "decision": "trade",
            }
        ]

        collector.decision_history = [
            {
                "timestamp": datetime.now(),
                "decision": "trade",
                "outcome": "profit",
                "regime": "bullish",
            }
        ]

        await collector.collect_binary_model_metrics(mock_metrics_collector)

        # Verify metrics were recorded
        assert mock_metrics_collector.record_metric.called

    @pytest.mark.asyncio
    async def test_collect_binary_model_metrics_disabled(self, mock_metrics_collector):
        """Test collecting binary model metrics when disabled."""
        config = {"binary_model_metrics": {"enabled": False}}
        collector = BinaryModelMetricsCollector(config)

        await collector.collect_binary_model_metrics(mock_metrics_collector)

        # Verify no metrics were recorded
        assert not mock_metrics_collector.record_metric.called

    @pytest.mark.asyncio
    async def test_collect_binary_model_metrics_with_exception(
        self, collector, mock_metrics_collector
    ):
        """Test collecting metrics handles exceptions gracefully."""
        # Make record_metric raise an exception
        mock_metrics_collector.record_metric.side_effect = Exception("Test error")

        # Should not raise exception
        await collector.collect_binary_model_metrics(mock_metrics_collector)

        # Should have attempted to record metrics
        assert mock_metrics_collector.record_metric.called


class TestAlertGeneration:
    """Test alert generation functionality."""

    @pytest.fixture
    def collector(self):
        """Create a test collector with sufficient data for alerts."""
        collector = BinaryModelMetricsCollector({})

        # Add historical data for comparison
        base_time = datetime.now() - timedelta(hours=2)

        # Add historical predictions with high trade frequency
        for i in range(100):
            collector.prediction_history.append(
                {
                    "timestamp": base_time - timedelta(hours=1) + timedelta(minutes=i),
                    "probability": 0.8,
                    "decision": "trade",
                    "regime": "bullish",
                }
            )

        # Add recent predictions with low trade frequency (drift)
        for i in range(100):
            collector.prediction_history.append(
                {
                    "timestamp": base_time + timedelta(minutes=i),
                    "probability": 0.3,
                    "decision": "skip",
                    "regime": "bullish",
                }
            )

        # Add decisions for accuracy comparison
        for i in range(50):
            collector.decision_history.append(
                {
                    "timestamp": base_time
                    - timedelta(hours=1)
                    + timedelta(minutes=i * 2),
                    "decision": "trade",
                    "outcome": "profit",
                    "was_correct": True,
                }
            )

        for i in range(50):
            collector.decision_history.append(
                {
                    "timestamp": base_time + timedelta(minutes=i * 2),
                    "decision": "trade",
                    "outcome": "loss",
                    "was_correct": False,
                }
            )

        return collector

    @pytest.mark.asyncio
    async def test_check_for_alerts_trade_frequency_drift(self, collector):
        """Test alert generation for trade frequency drift."""
        alerts = await collector.check_for_alerts()

        # Should generate trade frequency drift alert
        trade_freq_alerts = [
            a for a in alerts if a["alert_name"] == "BinaryModelTradeFrequencyDrift"
        ]
        assert len(trade_freq_alerts) == 1

        alert = trade_freq_alerts[0]
        assert alert["severity"] == "warning"
        assert "trade frequency changed" in alert["description"]
        assert alert["value"] > collector.drift_thresholds["trade_frequency_change"]

    @pytest.mark.asyncio
    async def test_check_for_alerts_accuracy_drop(self, collector):
        """Test alert generation for accuracy drop."""
        alerts = await collector.check_for_alerts()

        # Should generate accuracy drop alert
        accuracy_alerts = [
            a for a in alerts if a["alert_name"] == "BinaryModelAccuracyDrop"
        ]
        assert len(accuracy_alerts) == 1

        alert = accuracy_alerts[0]
        assert alert["severity"] == "critical"
        assert "Accuracy dropped" in alert["description"]

    @pytest.mark.asyncio
    async def test_check_for_alerts_disabled(self):
        """Test alert generation when disabled."""
        config = {"binary_model_metrics": {"enabled": False}}
        collector = BinaryModelMetricsCollector(config)

        alerts = await collector.check_for_alerts()

        assert alerts == []

    @pytest.mark.asyncio
    async def test_check_for_alerts_no_drift(self):
        """Test no alerts generated when no drift detected."""
        collector = BinaryModelMetricsCollector({})

        # Add consistent data with no drift
        base_time = datetime.now()
        for i in range(50):
            collector.prediction_history.append(
                {
                    "timestamp": base_time + timedelta(minutes=i),
                    "probability": 0.65,
                    "decision": "trade",
                    "regime": "bullish",
                }
            )

            collector.decision_history.append(
                {
                    "timestamp": base_time + timedelta(minutes=i),
                    "decision": "trade",
                    "outcome": "profit",
                    "was_correct": True,
                }
            )

        alerts = await collector.check_for_alerts()

        # Should not generate alerts for consistent data
        assert len(alerts) == 0

    def test_alert_cooldown(self, collector):
        """Test alert cooldown functionality."""
        # First alert should be generated
        current_time = datetime.now()
        should_alert = collector._should_alert("test_alert", current_time)
        assert should_alert == True

        # Immediate second alert should be blocked by cooldown
        immediate_time = current_time + timedelta(seconds=10)
        should_alert = collector._should_alert("test_alert", immediate_time)
        assert should_alert == False

        # Alert after cooldown should be allowed
        cooldown_time = current_time + timedelta(minutes=20)
        should_alert = collector._should_alert("test_alert", cooldown_time)
        assert should_alert == True


class TestPerformanceReporting:
    """Test performance reporting functionality."""

    @pytest.fixture
    def collector(self):
        """Create a test collector with sample data."""
        collector = BinaryModelMetricsCollector({})

        # Add sample data
        base_time = datetime.now()
        for i in range(10):
            collector.prediction_history.append(
                {
                    "timestamp": base_time,
                    "probability": 0.6 + i * 0.05,
                    "decision": "trade",
                }
            )

            collector.decision_history.append(
                {
                    "timestamp": base_time,
                    "decision": "trade",
                    "outcome": "profit",
                    "regime": "bullish",
                }
            )

        collector.regime_performance = {
            "bullish": {
                "total_decisions": 10,
                "correct_decisions": 8,
                "total_pnl": 500.0,
            }
        }

        return collector

    def test_get_performance_report(self, collector):
        """Test performance report generation."""
        report = collector.get_performance_report()

        assert isinstance(report, dict)
        assert "timestamp" in report
        assert "metrics_window_hours" in report
        assert "total_predictions" in report
        assert "total_decisions" in report
        assert "regime_performance" in report
        assert "current_metrics" in report

        # Verify data
        assert report["total_predictions"] == 10
        assert report["total_decisions"] == 10
        assert report["regime_performance"]["bullish"]["total_decisions"] == 10

    def test_get_performance_report_empty_collector(self):
        """Test performance report generation with empty collector."""
        collector = BinaryModelMetricsCollector({})

        report = collector.get_performance_report()

        assert isinstance(report, dict)
        assert report["total_predictions"] == 0
        assert report["total_decisions"] == 0
        assert report["regime_performance"] == {}


class TestGlobalFunctions:
    """Test global utility functions."""

    def test_get_binary_model_metrics_collector_singleton(self):
        """Test that get_binary_model_metrics_collector returns singleton."""
        collector1 = get_binary_model_metrics_collector()
        collector2 = get_binary_model_metrics_collector()

        assert collector1 is collector2
        assert isinstance(collector1, BinaryModelMetricsCollector)

    def test_create_binary_model_metrics_collector(self):
        """Test creating new collector instance."""
        config = {"test": "config"}
        collector = create_binary_model_metrics_collector(config)

        assert isinstance(collector, BinaryModelMetricsCollector)
        assert collector.config == config

    def test_create_binary_model_metrics_collector_default_config(self):
        """Test creating collector with default config."""
        collector = create_binary_model_metrics_collector()

        assert isinstance(collector, BinaryModelMetricsCollector)
        assert collector.config == {}


class TestErrorHandling:
    """Test error handling in various scenarios."""

    @pytest.mark.asyncio
    async def test_collect_metrics_with_binary_integration_import_error(self):
        """Test metrics collection handles binary integration import error."""
        collector = BinaryModelMetricsCollector({})

        mock_metrics_collector = Mock()
        mock_metrics_collector.record_metric = AsyncMock()

        with patch(
            "core.binary_model_metrics.get_binary_integration", side_effect=ImportError
        ):
            # Should not raise exception
            await collector.collect_binary_model_metrics(mock_metrics_collector)

    def test_record_prediction_with_invalid_data(self):
        """Test recording prediction with invalid data types."""
        collector = BinaryModelMetricsCollector({})

        # Should handle gracefully
        collector.record_prediction(
            symbol=None,  # Invalid
            probability="invalid",  # Invalid
            threshold=0.6,
            regime="test",
            features=None,  # Invalid
        )

        # Should still record (or handle gracefully)
        assert len(collector.prediction_history) == 1

    def test_record_decision_outcome_with_invalid_data(self):
        """Test recording decision outcome with invalid data."""
        collector = BinaryModelMetricsCollector({})

        # Should handle gracefully
        collector.record_decision_outcome(
            symbol=None,  # Invalid
            decision="invalid_decision",  # Invalid
            outcome="invalid_outcome",  # Invalid
            pnl="invalid_pnl",  # Invalid
            regime="test",
            strategy=None,  # Invalid
        )

        # Should still record (or handle gracefully)
        assert len(collector.decision_history) == 1

    def test_calculate_metrics_with_corrupted_data(self):
        """Test metrics calculation with corrupted data."""
        collector = BinaryModelMetricsCollector({})

        # Add corrupted prediction data
        collector.prediction_history = [
            {
                "timestamp": "invalid_timestamp",
                "probability": "invalid_probability",
                "decision": "invalid_decision",
            }
        ]

        # Should handle gracefully without crashing
        avg_ptrade = collector._calculate_average_ptrade()
        assert avg_ptrade is None or isinstance(avg_ptrade, (int, float))

    @pytest.mark.asyncio
    async def test_check_for_alerts_with_exception(self):
        """Test alert checking handles exceptions gracefully."""
        collector = BinaryModelMetricsCollector({})

        # Mock a method to raise exception
        with patch.object(
            collector,
            "_calculate_drift_indicators",
            side_effect=Exception("Test error"),
        ):
            alerts = await collector.check_for_alerts()

            # Should return empty list on exception
            assert alerts == []


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_prediction_history_data_integrity(self):
        """Test prediction history maintains data integrity."""
        collector = BinaryModelMetricsCollector({})

        # Add predictions with various data types
        predictions = [
            {
                "symbol": "BTC/USDT",
                "probability": 0.75,
                "threshold": 0.6,
                "regime": "bullish",
                "features": {"RSI": 65.0, "MACD": 0.5},
                "decision": "trade",
            },
            {
                "symbol": "ETH/USDT",
                "probability": 0.45,
                "threshold": 0.6,
                "regime": "bearish",
                "features": {"RSI": 35.0, "MACD": -0.3},
                "decision": "skip",
            },
        ]

        for prediction in predictions:
            collector.record_prediction(**prediction)

        # Verify data integrity
        assert len(collector.prediction_history) == 2
        for i, prediction in enumerate(collector.prediction_history):
            assert "timestamp" in prediction
            assert isinstance(prediction["timestamp"], datetime)
            assert prediction["symbol"] == predictions[i]["symbol"]
            assert prediction["probability"] == predictions[i]["probability"]

    def test_decision_history_data_integrity(self):
        """Test decision history maintains data integrity."""
        collector = BinaryModelMetricsCollector({})

        # Add decisions
        decisions = [
            {
                "symbol": "BTC/USDT",
                "decision": "trade",
                "outcome": "profit",
                "pnl": 150.0,
                "regime": "bullish",
                "strategy": "RSIStrategy",
            },
            {
                "symbol": "ETH/USDT",
                "decision": "skip",
                "outcome": "loss",
                "pnl": 0.0,
                "regime": "bearish",
                "strategy": "none",
            },
        ]

        for decision in decisions:
            collector.record_decision_outcome(**decision)

        # Verify data integrity
        assert len(collector.decision_history) == 2
        for i, decision in enumerate(collector.decision_history):
            assert "timestamp" in decision
            assert isinstance(decision["timestamp"], datetime)
            assert "was_correct" in decision
            assert isinstance(decision["was_correct"], bool)


class TestConcurrency:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_prediction_recording(self):
        """Test concurrent prediction recording."""
        collector = BinaryModelMetricsCollector({})

        async def record_prediction(i):
            await asyncio.sleep(0.001)  # Small delay to simulate concurrency
            collector.record_prediction(
                symbol=f"BTC/USDT_{i}",
                probability=0.5 + i * 0.01,
                threshold=0.6,
                regime="bullish",
                features={"RSI": 50.0 + i},
            )

        # Record predictions concurrently
        tasks = [record_prediction(i) for i in range(20)]
        await asyncio.gather(*tasks)

        # Verify all predictions were recorded
        assert len(collector.prediction_history) == 20

        # Verify data integrity
        symbols = [p["symbol"] for p in collector.prediction_history]
        assert len(set(symbols)) == 20  # All unique

    @pytest.mark.asyncio
    async def test_concurrent_metrics_collection(self):
        """Test concurrent metrics collection."""
        collector = BinaryModelMetricsCollector({})

        # Add some test data
        collector.prediction_history = [
            {"timestamp": datetime.now(), "probability": 0.7, "decision": "trade"}
        ]

        mock_metrics_collector = Mock()
        mock_metrics_collector.record_metric = AsyncMock()

        # Run multiple concurrent collections
        tasks = [
            collector.collect_binary_model_metrics(mock_metrics_collector)
            for _ in range(5)
        ]

        await asyncio.gather(*tasks)

        # Should have attempted to record metrics multiple times
        assert mock_metrics_collector.record_metric.call_count >= 5


if __name__ == "__main__":
    pytest.main([__file__])
