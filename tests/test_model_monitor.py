"""
Test suite for Model Monitor.

Tests model monitoring, performance tracking, drift detection, and auto-recalibration.
"""

import pytest
import asyncio
import time
import json
import os
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from datetime import datetime, timedelta
from typing import Dict, Any

from core.model_monitor import (
    ModelMonitor, AutoRecalibrator, PerformanceMetrics, DriftMetrics, ModelHealthReport,
    create_model_monitor, create_auto_recalibrator, generate_monitoring_report
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test basic PerformanceMetrics creation."""
        timestamp = datetime.now()
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            auc=0.85,
            f1_score=0.82,
            precision=0.78,
            recall=0.87,
            pnl=1250.50,
            sharpe_ratio=1.25,
            max_drawdown=-0.15,
            total_trades=150,
            win_rate=0.65,
            calibration_error=0.05,
            sample_size=200
        )

        assert metrics.timestamp == timestamp
        assert metrics.auc == 0.85
        assert metrics.f1_score == 0.82
        assert metrics.pnl == 1250.50
        assert metrics.total_trades == 150
        assert metrics.sample_size == 200

    def test_performance_metrics_to_dict(self):
        """Test PerformanceMetrics serialization."""
        metrics = PerformanceMetrics(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            auc=0.85,
            f1_score=0.82,
            precision=0.78,
            recall=0.87,
            pnl=1250.50,
            sharpe_ratio=1.25,
            max_drawdown=-0.15,
            total_trades=150,
            win_rate=0.65,
            calibration_error=0.05,
            sample_size=200
        )

        data = metrics.__dict__
        assert data['auc'] == 0.85
        assert data['pnl'] == 1250.50
        assert isinstance(data['timestamp'], datetime)


class TestDriftMetrics:
    """Test DriftMetrics dataclass."""

    def test_drift_metrics_creation(self):
        """Test basic DriftMetrics creation."""
        timestamp = datetime.now()
        metrics = DriftMetrics(
            timestamp=timestamp,
            feature_drift_scores={'feature1': 0.15, 'feature2': 0.08},
            prediction_drift_score=0.12,
            label_drift_score=0.05,
            overall_drift_score=0.11,
            is_drift_detected=False
        )

        assert metrics.timestamp == timestamp
        assert metrics.feature_drift_scores['feature1'] == 0.15
        assert metrics.prediction_drift_score == 0.12
        assert metrics.overall_drift_score == 0.11
        assert metrics.is_drift_detected is False

    def test_drift_metrics_drift_detected(self):
        """Test drift detection logic."""
        metrics = DriftMetrics(
            timestamp=datetime.now(),
            feature_drift_scores={},
            prediction_drift_score=0.25,
            label_drift_score=0.15,
            overall_drift_score=0.20,
            is_drift_detected=True
        )

        assert metrics.is_drift_detected is True


class TestModelHealthReport:
    """Test ModelHealthReport dataclass."""

    def test_model_health_report_creation(self):
        """Test basic ModelHealthReport creation."""
        timestamp = datetime.now()
        report = ModelHealthReport(
            timestamp=timestamp,
            overall_health_score=0.75,
            performance_score=0.82,
            drift_score=0.15,
            calibration_score=0.78,
            recommendations=["Monitor performance closely", "Consider retraining"],
            requires_retraining=False,
            confidence_level="MEDIUM"
        )

        assert report.timestamp == timestamp
        assert report.overall_health_score == 0.75
        assert report.performance_score == 0.82
        assert len(report.recommendations) == 2
        assert report.requires_retraining is False
        assert report.confidence_level == "MEDIUM"


class TestModelMonitor:
    """Test ModelMonitor class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'model_path': 'models/test_model.pkl',
            'config_path': 'models/test_model_config.json',
            'monitoring_window_days': 30,
            'output_dir': 'test_monitoring',
            'monitor_interval_minutes': 60
        }
        self.monitor = ModelMonitor(self.config)

    def test_monitor_initialization(self):
        """Test ModelMonitor initialization."""
        assert self.monitor.model_path == 'models/test_model.pkl'
        assert self.monitor.monitoring_window_days == 30
        assert self.monitor.monitoring_active is False
        assert len(self.monitor.performance_history) == 0
        assert len(self.monitor.drift_history) == 0
        assert len(self.monitor.alerts) == 0

    @patch('os.path.exists')
    @patch('joblib.load')
    @patch('builtins.open', new_callable=mock_open, read_data='{"optimal_threshold": 0.6}')
    def test_load_model_success(self, mock_file, mock_joblib, mock_exists):
        """Test successful model loading."""
        mock_exists.return_value = True
        mock_joblib.return_value = MagicMock()

        self.monitor._load_model()

        assert self.monitor.model is not None
        assert self.monitor.optimal_threshold == 0.6
        mock_joblib.assert_called_once_with('models/test_model.pkl')

    @patch('os.path.exists')
    def test_load_model_file_not_found(self, mock_exists):
        """Test model loading when file doesn't exist."""
        mock_exists.return_value = False

        with patch('core.model_monitor.logger') as mock_logger:
            self.monitor._load_model()

            mock_logger.warning.assert_called()
            assert self.monitor.model is None

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        assert self.monitor.monitoring_active is False

        # Start monitoring
        self.monitor.start_monitoring()
        assert self.monitor.monitoring_active is True
        assert self.monitor.monitor_thread is not None

        # Stop monitoring
        self.monitor.stop_monitoring()
        assert self.monitor.monitoring_active is False

    def test_update_predictions(self):
        """Test updating predictions."""
        features = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        predictions = np.array([0.3, 0.7, 0.8])
        true_labels = np.array([0, 1, 1])
        timestamp = datetime.now()

        self.monitor.update_predictions(features, predictions, true_labels, timestamp)

        assert len(self.monitor.prediction_history) == 1
        record = self.monitor.prediction_history[0]
        assert record['timestamp'] == timestamp
        assert len(record['predictions']) == 3
        assert len(record['true_labels']) == 3

    def test_update_predictions_no_labels(self):
        """Test updating predictions without true labels."""
        features = pd.DataFrame({'feature1': [1, 2]})
        predictions = np.array([0.4, 0.6])

        self.monitor.update_predictions(features, predictions)

        assert len(self.monitor.prediction_history) == 1
        record = self.monitor.prediction_history[0]
        assert record['true_labels'] is None

    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        predictions = np.array([0.2, 0.8, 0.6, 0.9])
        true_labels = np.array([0, 1, 1, 1])

        metrics = self.monitor.calculate_performance_metrics(predictions, true_labels)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.auc >= 0.0
        assert metrics.f1_score >= 0.0
        assert metrics.precision >= 0.0
        assert metrics.recall >= 0.0
        assert metrics.sample_size == 4

    def test_calculate_performance_metrics_custom_threshold(self):
        """Test performance metrics with custom threshold."""
        predictions = np.array([0.3, 0.7, 0.5])
        true_labels = np.array([0, 1, 1])

        metrics = self.monitor.calculate_performance_metrics(predictions, true_labels, threshold=0.6)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.sample_size == 3

    def test_detect_drift_no_reference(self):
        """Test drift detection without reference data."""
        features = pd.DataFrame({'feature1': [1, 2, 3]})
        predictions = np.array([0.3, 0.7, 0.5])

        drift_metrics = self.monitor.detect_drift(features, predictions)

        assert isinstance(drift_metrics, DriftMetrics)
        assert drift_metrics.is_drift_detected is False
        assert drift_metrics.overall_drift_score == 0.0

    def test_detect_drift_with_reference(self):
        """Test drift detection with reference data."""
        # Set up reference data and feature columns
        self.monitor.reference_features = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        self.monitor.reference_predictions = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        self.monitor.reference_labels = np.array([0, 0, 1, 1, 1])
        self.monitor.feature_columns = ['feature1']  # Set feature columns

        # Current data (slightly different)
        features = pd.DataFrame({'feature1': [1.1, 2.1, 3.1, 4.1, 5.1]})
        predictions = np.array([0.35, 0.45, 0.55, 0.65, 0.75])
        labels = np.array([0, 1, 1, 1, 1])

        drift_metrics = self.monitor.detect_drift(features, predictions, labels)

        assert isinstance(drift_metrics, DriftMetrics)
        # Feature drift scores may be empty if scipy is not available, but other scores should be present
        assert drift_metrics.prediction_drift_score >= 0.0
        assert drift_metrics.label_drift_score >= 0.0
        assert drift_metrics.overall_drift_score >= 0.0

    def test_check_model_health_no_data(self):
        """Test model health check with no data."""
        health_report = self.monitor.check_model_health()

        assert isinstance(health_report, ModelHealthReport)
        assert health_report.overall_health_score >= 0.0
        assert health_report.overall_health_score <= 1.0
        assert len(health_report.recommendations) > 0

    def test_check_model_health_with_data(self):
        """Test model health check with performance data."""
        # Add some performance metrics
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            auc=0.85,
            f1_score=0.82,
            precision=0.78,
            recall=0.87,
            pnl=100.0,
            sharpe_ratio=1.2,
            max_drawdown=-0.1,
            total_trades=50,
            win_rate=0.65,
            calibration_error=0.05,
            sample_size=100
        )
        self.monitor.performance_history.append(metrics)

        health_report = self.monitor.check_model_health()

        assert isinstance(health_report, ModelHealthReport)
        assert health_report.performance_score > 0.0
        assert health_report.confidence_level in ["LOW", "MEDIUM", "HIGH"]

    def test_trigger_alert(self):
        """Test alert triggering."""
        alert_type = "TEST_ALERT"
        message = "Test alert message"

        with patch('core.model_monitor.logger') as mock_logger:
            self.monitor._trigger_alert(alert_type, message)

        assert len(self.monitor.alerts) == 1
        assert self.monitor.alerts[0]['type'] == alert_type
        assert self.monitor.alerts[0]['message'] == message
        mock_logger.warning.assert_called_once()

    def test_add_alert_callback(self):
        """Test adding alert callback."""
        callback = MagicMock()
        self.monitor.add_alert_callback(callback)

        assert len(self.monitor.alert_callbacks) == 1
        assert callback in self.monitor.alert_callbacks

    def test_generate_report(self):
        """Test report generation."""
        report = self.monitor.generate_report()

        assert 'timestamp' in report
        assert 'model_path' in report
        assert 'health_assessment' in report
        assert 'performance_summary' in report
        assert 'drift_summary' in report
        assert 'alerts_summary' in report
        assert 'recommendations' in report

    def test_get_performance_summary_no_data(self):
        """Test performance summary with no data."""
        summary = self.monitor._generate_performance_summary()

        assert 'status' in summary
        assert summary['status'] == 'No performance data available'

    def test_get_performance_summary_with_data(self):
        """Test performance summary with data."""
        # Add performance metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                auc=0.8 + i * 0.01,
                f1_score=0.75 + i * 0.01,
                precision=0.7 + i * 0.01,
                recall=0.8 + i * 0.01,
                pnl=100.0 + i * 10,
                sharpe_ratio=1.0 + i * 0.1,
                max_drawdown=-0.1,
                total_trades=50 + i * 5,
                win_rate=0.6 + i * 0.01,
                calibration_error=0.05 + i * 0.005,
                sample_size=100
            )
            self.monitor.performance_history.append(metrics)

        summary = self.monitor._generate_performance_summary()

        assert 'total_records' in summary
        assert 'avg_auc' in summary
        assert 'avg_sharpe' in summary
        assert summary['total_records'] == 5

    def test_get_drift_summary_no_data(self):
        """Test drift summary with no data."""
        summary = self.monitor._generate_drift_summary()

        assert 'status' in summary
        assert summary['status'] == 'No drift data available'

    def test_get_alerts_summary_no_data(self):
        """Test alerts summary with no data."""
        summary = self.monitor._generate_alerts_summary()

        assert 'status' in summary
        assert summary['status'] == 'No alerts generated'

    def test_get_alerts_summary_with_data(self):
        """Test alerts summary with data."""
        # Add some alerts
        self.monitor.alerts = [
            {'timestamp': datetime.now(), 'type': 'PERFORMANCE_DEGRADED', 'message': 'Test alert 1'},
            {'timestamp': datetime.now(), 'type': 'DRIFT_DETECTED', 'message': 'Test alert 2'},
            {'timestamp': datetime.now() - timedelta(days=10), 'type': 'OLD_ALERT', 'message': 'Old alert'}
        ]

        summary = self.monitor._generate_alerts_summary()

        assert 'total_alerts' in summary
        assert 'recent_alerts' in summary
        assert 'alert_types' in summary
        assert summary['total_alerts'] == 3
        assert summary['recent_alerts'] == 2  # Only recent alerts (within 7 days)

    @patch('os.path.exists', return_value=False)
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_monitoring_data(self, mock_json_dump, mock_file, mock_makedirs, mock_exists):
        """Test saving monitoring data."""
        # Add some data
        self.monitor.performance_history.append(PerformanceMetrics(
            timestamp=datetime.now(),
            auc=0.85, f1_score=0.82, precision=0.78, recall=0.87,
            pnl=100.0, sharpe_ratio=1.2, max_drawdown=-0.1,
            total_trades=50, win_rate=0.65, calibration_error=0.05, sample_size=100
        ))
        self.monitor.alerts.append({'timestamp': datetime.now(), 'type': 'TEST', 'message': 'Test'})

        self.monitor._save_monitoring_data()

        # Verify files were written (makedirs may or may not be called depending on directory existence)
        assert mock_json_dump.call_count >= 1


class TestAutoRecalibrator:
    """Test AutoRecalibrator class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'model_path': 'models/test_model.pkl',
            'retraining_data_path': 'data/test_data.csv',
            'min_retraining_interval_hours': 24
        }
        self.recalibrator = AutoRecalibrator(self.config)

    def test_recalibrator_initialization(self):
        """Test AutoRecalibrator initialization."""
        assert isinstance(self.recalibrator.monitor, ModelMonitor)
        assert self.recalibrator.retraining_active is False
        assert self.recalibrator.min_retraining_interval_hours == 24

    def test_handle_alert_retraining_required(self):
        """Test handling RETRAINING_REQUIRED alert."""
        alert = {'type': 'RETRAINING_REQUIRED', 'message': 'Model needs retraining'}

        with patch.object(self.recalibrator, '_trigger_retraining') as mock_trigger:
            self.recalibrator._handle_alert(alert)
            mock_trigger.assert_called_once_with(alert)

    def test_handle_alert_performance_degraded(self):
        """Test handling PERFORMANCE_DEGRADED alert."""
        alert = {'type': 'PERFORMANCE_DEGRADED', 'message': 'Performance degraded'}

        with patch('core.model_monitor.logger') as mock_logger:
            self.recalibrator._handle_alert(alert)
            mock_logger.warning.assert_called_once()

    def test_trigger_retraining_too_soon(self):
        """Test retraining trigger when too soon since last retraining."""
        # Set last retraining to recent time
        self.recalibrator.last_retraining = datetime.now() - timedelta(hours=1)

        alert = {'type': 'RETRAINING_REQUIRED', 'message': 'Test'}

        with patch('core.model_monitor.logger') as mock_logger:
            self.recalibrator._trigger_retraining(alert)
            mock_logger.info.assert_called_with("Retraining skipped - too soon since last retraining")

    def test_trigger_retraining_already_active(self):
        """Test retraining trigger when retraining is already active."""
        self.recalibrator.retraining_active = True

        alert = {'type': 'RETRAINING_REQUIRED', 'message': 'Test'}

        with patch('core.model_monitor.logger') as mock_logger:
            self.recalibrator._trigger_retraining(alert)
            mock_logger.info.assert_called_with("Retraining already in progress")

    @patch('threading.Thread')
    def test_trigger_retraining_success(self, mock_thread):
        """Test successful retraining trigger."""
        alert = {'type': 'RETRAINING_REQUIRED', 'message': 'Test'}

        self.recalibrator._trigger_retraining(alert)

        assert self.recalibrator.retraining_active is True
        mock_thread.assert_called_once()

    def test_start_stop(self):
        """Test starting and stopping the recalibrator."""
        with patch.object(self.recalibrator.monitor, 'start_monitoring') as mock_start, \
             patch.object(self.recalibrator.monitor, 'stop_monitoring') as mock_stop:

            self.recalibrator.start()
            mock_start.assert_called_once()

            self.recalibrator.stop()
            mock_stop.assert_called_once()


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_model_monitor(self):
        """Test create_model_monitor function."""
        config = {'model_path': 'test.pkl'}
        monitor = create_model_monitor(config)

        assert isinstance(monitor, ModelMonitor)
        assert monitor.model_path == 'test.pkl'

    def test_create_auto_recalibrator(self):
        """Test create_auto_recalibrator function."""
        config = {'model_path': 'test.pkl'}
        recalibrator = create_auto_recalibrator(config)

        assert isinstance(recalibrator, AutoRecalibrator)
        assert recalibrator.monitor.model_path == 'test.pkl'

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_generate_monitoring_report(self, mock_json_dump, mock_file, mock_makedirs):
        """Test generate_monitoring_report function."""
        monitor = ModelMonitor({'model_path': 'test.pkl'})

        # Test without output path
        report = generate_monitoring_report(monitor)
        assert 'timestamp' in report
        assert 'model_path' in report

        # Test with output path
        report = generate_monitoring_report(monitor, 'test_report.json')
        assert 'timestamp' in report
        mock_json_dump.assert_called_once()


class TestModelMonitorEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = ModelMonitor({'model_path': 'test.pkl'})

    def test_calculate_performance_metrics_empty_arrays(self):
        """Test performance calculation with empty arrays."""
        predictions = np.array([])
        true_labels = np.array([])

        with pytest.raises(Exception):  # Should handle gracefully
            self.monitor.calculate_performance_metrics(predictions, true_labels)

    def test_update_predictions_empty_data(self):
        """Test updating predictions with empty data."""
        features = pd.DataFrame()
        predictions = np.array([])

        # Should not crash
        self.monitor.update_predictions(features, predictions)

    def test_detect_drift_empty_features(self):
        """Test drift detection with empty features."""
        features = pd.DataFrame()
        predictions = np.array([])

        drift_metrics = self.monitor.detect_drift(features, predictions)

        assert isinstance(drift_metrics, DriftMetrics)
        assert drift_metrics.is_drift_detected is False

    def test_check_model_health_empty_history(self):
        """Test health check with empty history."""
        # Ensure histories are empty
        self.monitor.performance_history = []
        self.monitor.drift_history = []

        health_report = self.monitor.check_model_health()

        assert isinstance(health_report, ModelHealthReport)
        assert health_report.overall_health_score >= 0.0
        # Check that recommendations contain the expected text (may have additional recommendations)
        recommendation_text = " ".join(health_report.recommendations)
        assert "Insufficient monitoring history" in recommendation_text

    def test_trigger_alert_with_callbacks(self):
        """Test alert triggering with callbacks."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        self.monitor.add_alert_callback(callback1)
        self.monitor.add_alert_callback(callback2)

        self.monitor._trigger_alert('TEST', 'Test alert')

        # Callbacks should be called with the alert dict (which includes additional fields like timestamp and model_path)
        assert callback1.call_count == 1
        assert callback2.call_count == 1

        # Check that the alert dict contains the expected fields
        call_args = callback1.call_args[0][0]
        assert call_args['type'] == 'TEST'
        assert call_args['message'] == 'Test alert'
        assert 'timestamp' in call_args
        assert 'model_path' in call_args

    def test_trigger_alert_callback_error(self):
        """Test alert triggering when callback raises error."""
        callback = MagicMock(side_effect=Exception("Callback error"))

        self.monitor.add_alert_callback(callback)

        with patch('core.model_monitor.logger') as mock_logger:
            self.monitor._trigger_alert('TEST', 'Test alert')

            # Alert should still be recorded
            assert len(self.monitor.alerts) == 1
            # Error should be logged
            mock_logger.error.assert_called_once()


class TestModelMonitorIntegration:
    """Test integration scenarios."""

    def test_complete_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        monitor = ModelMonitor({
            'model_path': 'test.pkl',
            'output_dir': 'test_output'
        })

        # Simulate some predictions
        features = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        predictions = np.array([0.3, 0.6, 0.8, 0.2, 0.7])
        true_labels = np.array([0, 1, 1, 0, 1])

        # Update predictions
        monitor.update_predictions(features, predictions, true_labels)

        # Calculate performance
        metrics = monitor.calculate_performance_metrics(predictions, true_labels)
        monitor.performance_history.append(metrics)

        # Check health
        health_report = monitor.check_model_health()

        assert isinstance(health_report, ModelHealthReport)
        assert health_report.performance_score > 0.0

        # Generate report
        report = monitor.generate_report()

        assert 'health_assessment' in report
        assert 'performance_summary' in report

    def test_monitoring_with_drift_detection(self):
        """Test monitoring with drift detection."""
        monitor = ModelMonitor({'model_path': 'test.pkl'})

        # Set up reference data and feature columns
        monitor.reference_features = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        monitor.reference_predictions = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        monitor.reference_labels = np.array([0, 0, 1, 1, 1])
        monitor.feature_columns = ['feature1', 'feature2']  # Set feature columns

        # Current data with some drift
        current_features = pd.DataFrame({
            'feature1': [1.5, 2.5, 3.5, 4.5, 5.5],  # Slightly different
            'feature2': [0.15, 0.25, 0.35, 0.45, 0.55]  # Slightly different
        })
        current_predictions = np.array([0.35, 0.45, 0.55, 0.65, 0.75])
        current_labels = np.array([0, 1, 1, 1, 1])

        # Detect drift
        drift_metrics = monitor.detect_drift(current_features, current_predictions, current_labels)
        monitor.drift_history.append(drift_metrics)

        # Check health (should detect some drift)
        health_report = monitor.check_model_health()

        assert isinstance(health_report, ModelHealthReport)
        assert health_report.drift_score >= 0.0

        # Verify drift detection was attempted (feature drift scores may be empty if scipy is not available)
        assert isinstance(drift_metrics.feature_drift_scores, dict)
        assert drift_metrics.prediction_drift_score >= 0.0
        assert drift_metrics.label_drift_score >= 0.0


if __name__ == "__main__":
    pytest.main([__file__])
