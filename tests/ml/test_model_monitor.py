"""
Unit tests for model monitoring and drift detection functionality.

Tests cover:
- Model monitor initialization and configuration
- Performance metrics calculation
- Drift detection algorithms
- Health assessment and alerting
- Background monitoring functionality
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, call
import sys
import os
import tempfile
import json
from datetime import datetime, timedelta

# Add the ml directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.model_monitor import (
    ModelMonitor,
    PerformanceMetrics,
    DriftMetrics,
    ModelHealthReport,
    create_model_monitor,
    generate_monitoring_report,
    _safe_joblib_dump
)


class TestModelMonitor:
    """Test model monitor functionality."""

    def setup_method(self):
        """Set up test data and mock model."""
        np.random.seed(42)

        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.predict_proba.return_value = np.random.rand(100, 2)
        self.mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])

        # Create temporary files for model and config
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_model.pkl')
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')

        # Save mock model using safe dump
        _safe_joblib_dump(self.mock_model, self.model_path)

        # Create mock config
        self.config = {
            'model_path': self.model_path,
            'config_path': self.config_path,
            'monitoring_window_days': 30,
            'drift_thresholds': {'overall_threshold': 0.1},
            'alerts': {
                'performance_threshold': 0.6,
                'drift_threshold': 0.7
            },
            'monitor_interval_minutes': 60
        }

        # Save config
        with open(self.config_path, 'w') as f:
            json.dump({
                'optimal_threshold': 0.5,
                'expected_performance': {'pnl': 0.005}
            }, f)

        # Create test data
        self.feature_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.random.normal(-2, 0.5, 100),
            'feature4': np.random.normal(1, 0.8, 100)
        })
        self.predictions = np.random.rand(100, 2)
        self.true_labels = np.random.choice([0, 1], 100)

    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_monitor_initialization(self):
        """Test model monitor initialization."""
        monitor = ModelMonitor(self.config)

        assert monitor.model_path == self.model_path
        assert monitor.config_path == self.config_path
        assert monitor.monitoring_window_days == 30
        assert not monitor.monitoring_active
        assert monitor.optimal_threshold == 0.5

    def test_update_predictions(self):
        """Test updating predictions for monitoring."""
        monitor = ModelMonitor(self.config)

        # Update predictions
        timestamp = datetime.now()
        monitor.update_predictions(
            self.feature_data,
            self.predictions,
            self.true_labels,
            timestamp
        )

        assert len(monitor.prediction_history) == 1
        assert monitor.prediction_history[0]['timestamp'] == timestamp
        assert len(monitor.prediction_history[0]['features']) == 100

    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        monitor = ModelMonitor(self.config)

        metrics = monitor.calculate_performance_metrics(
            self.predictions[:, 1],  # Probabilities for positive class
            self.true_labels,
            threshold=0.5
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, 'auc')
        assert hasattr(metrics, 'f1_score')
        assert hasattr(metrics, 'precision')
        assert hasattr(metrics, 'recall')
        assert hasattr(metrics, 'pnl')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'calibration_error')
        assert metrics.sample_size == 100

        # Check reasonable value ranges
        assert 0 <= metrics.auc <= 1
        assert 0 <= metrics.f1_score <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1

    def test_detect_drift_no_reference(self):
        """Test drift detection without reference data."""
        monitor = ModelMonitor(self.config)

        drift_metrics = monitor.detect_drift(
            self.feature_data,
            self.predictions[:, 1],
            self.true_labels
        )

        assert isinstance(drift_metrics, DriftMetrics)
        assert not drift_metrics.is_drift_detected
        assert drift_metrics.overall_drift_score == 0.0
        assert len(drift_metrics.feature_drift_scores) == 0

    def test_detect_drift_with_reference(self):
        """Test drift detection with reference data."""
        monitor = ModelMonitor(self.config)

        # Set up reference data
        monitor.reference_features = self.feature_data.copy()
        monitor.reference_predictions = self.predictions[:, 1].copy()
        monitor.reference_labels = self.true_labels.copy()

        # Create slightly different current data (simulating drift)
        current_features = self.feature_data + np.random.normal(0, 0.1, self.feature_data.shape)
        current_predictions = self.predictions[:, 1] + np.random.normal(0, 0.05, 100)
        current_labels = self.true_labels.copy()

        drift_metrics = monitor.detect_drift(
            current_features,
            current_predictions,
            current_labels
        )

        assert isinstance(drift_metrics, DriftMetrics)
        assert isinstance(drift_metrics.is_drift_detected, bool)
        assert isinstance(drift_metrics.overall_drift_score, float)
        assert len(drift_metrics.feature_drift_scores) == 4  # 4 features

    def test_check_model_health(self):
        """Test model health assessment."""
        monitor = ModelMonitor(self.config)

        # Add some performance history
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=datetime.now() - timedelta(hours=i),
                auc=0.8 - i * 0.05,
                f1_score=0.75 - i * 0.04,
                precision=0.7 - i * 0.03,
                recall=0.8 - i * 0.04,
                pnl=0.01 - i * 0.002,
                sharpe_ratio=1.5 - i * 0.2,
                max_drawdown=-0.05 + i * 0.01,
                total_trades=50 - i * 5,
                win_rate=0.65 - i * 0.05,
                calibration_error=0.1 + i * 0.02,
                sample_size=100
            )
            monitor.performance_history.append(metrics)

        health_report = monitor.check_model_health()

        assert isinstance(health_report, ModelHealthReport)
        assert isinstance(health_report.overall_health_score, float)
        assert isinstance(health_report.performance_score, float)
        assert isinstance(health_report.drift_score, float)
        assert isinstance(health_report.calibration_score, float)
        assert isinstance(health_report.recommendations, list)
        assert isinstance(health_report.requires_retraining, bool)
        assert health_report.confidence_level in ['HIGH', 'MEDIUM', 'LOW']

    def test_trigger_alert(self):
        """Test alert triggering."""
        monitor = ModelMonitor(self.config)

        # Mock alert callback
        alert_callback = MagicMock()
        monitor.add_alert_callback(alert_callback)

        # Trigger alert
        monitor._trigger_alert("TEST_ALERT", "Test alert message")

        assert len(monitor.alerts) == 1
        assert monitor.alerts[0]['type'] == "TEST_ALERT"
        assert monitor.alerts[0]['message'] == "Test alert message"

        # Check callback was called
        alert_callback.assert_called_once()
        call_args = alert_callback.call_args[0][0]
        assert call_args['type'] == "TEST_ALERT"
        assert call_args['message'] == "Test alert message"

    def test_save_monitoring_data(self):
        """Test saving monitoring data."""
        monitor = ModelMonitor(self.config)

        # Add some data
        monitor.performance_history.append(PerformanceMetrics(
            timestamp=datetime.now(),
            auc=0.8, f1_score=0.75, precision=0.7, recall=0.8,
            pnl=0.01, sharpe_ratio=1.5, max_drawdown=-0.05,
            total_trades=50, win_rate=0.65, calibration_error=0.1,
            sample_size=100
        ))

        monitor.drift_history.append(DriftMetrics(
            timestamp=datetime.now(),
            feature_drift_scores={'feature1': 0.1, 'feature2': 0.05},
            prediction_drift_score=0.08,
            label_drift_score=0.03,
            overall_drift_score=0.07,
            is_drift_detected=False
        ))

        monitor.alerts.append({
            'timestamp': datetime.now(),
            'type': 'TEST_ALERT',
            'message': 'Test message',
            'model_path': monitor.model_path
        })

        # Save data
        monitor._save_monitoring_data()

        # Check files were created
        perf_file = os.path.join(monitor.output_dir, 'performance_history.json')
        drift_file = os.path.join(monitor.output_dir, 'drift_history.json')
        alerts_file = os.path.join(monitor.output_dir, 'alerts.json')

        assert os.path.exists(perf_file)
        assert os.path.exists(drift_file)
        assert os.path.exists(alerts_file)

        # Check file contents
        with open(perf_file, 'r') as f:
            perf_data = json.load(f)
            assert len(perf_data) == 1
            assert 'auc' in perf_data[0]

        with open(drift_file, 'r') as f:
            drift_data = json.load(f)
            assert len(drift_data) == 1
            assert 'overall_drift_score' in drift_data[0]

        with open(alerts_file, 'r') as f:
            alerts_data = json.load(f)
            assert len(alerts_data) == 1
            assert alerts_data[0]['type'] == 'TEST_ALERT'

    def test_generate_report(self):
        """Test monitoring report generation."""
        monitor = ModelMonitor(self.config)

        # Add some performance history
        monitor.performance_history.append(PerformanceMetrics(
            timestamp=datetime.now(),
            auc=0.8, f1_score=0.75, precision=0.7, recall=0.8,
            pnl=0.01, sharpe_ratio=1.5, max_drawdown=-0.05,
            total_trades=50, win_rate=0.65, calibration_error=0.1,
            sample_size=100
        ))

        report = monitor.generate_report()

        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'model_path' in report
        assert 'health_assessment' in report
        assert 'performance_summary' in report
        assert 'drift_summary' in report
        assert 'alerts_summary' in report
        assert 'recommendations' in report

    def test_start_stop_monitoring(self):
        """Test starting and stopping background monitoring."""
        monitor = ModelMonitor(self.config)

        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring_active
        assert monitor.monitor_thread is not None
        assert monitor.monitor_thread.is_alive()

        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring_active

    @patch('time.sleep')
    def test_monitoring_loop(self, mock_sleep):
        """Test the monitoring loop."""
        monitor = ModelMonitor(self.config)

        # Mock the health check to avoid actual processing
        with patch.object(monitor, 'check_model_health') as mock_health:
            mock_health.return_value = ModelHealthReport(
                timestamp=datetime.now(),
                overall_health_score=0.8,
                performance_score=0.75,
                drift_score=0.1,
                calibration_score=0.85,
                recommendations=["Model is healthy"],
                requires_retraining=False,
                confidence_level="HIGH"
            )

            # Start monitoring in a separate thread
            monitor.monitoring_active = True
            monitor._monitoring_loop()

            # Should have called health check
            mock_health.assert_called()

            # Should have tried to sleep
            mock_sleep.assert_called()


class TestMonitoringUtilities:
    """Test monitoring utility functions."""

    def test_create_model_monitor(self):
        """Test creating model monitor instance."""
        config = {
            'model_path': '/path/to/model.pkl',
            'config_path': '/path/to/config.json',
            'monitoring_window_days': 30
        }

        monitor = create_model_monitor(config)

        assert isinstance(monitor, ModelMonitor)
        assert monitor.model_path == '/path/to/model.pkl'
        assert monitor.monitoring_window_days == 30

    @patch('ml.model_monitor.os.makedirs')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.dump')
    def test_generate_monitoring_report_with_save(self, mock_json_dump, mock_open, mock_makedirs):
        """Test generating and saving monitoring report."""
        # Create mock monitor
        mock_monitor = MagicMock()
        mock_monitor.generate_report.return_value = {
            'timestamp': '2023-01-01T00:00:00',
            'model_path': '/path/to/model.pkl',
            'health_assessment': {'overall_health_score': 0.8}
        }

        # Generate report with save path
        report = generate_monitoring_report(mock_monitor, '/path/to/report.json')

        assert isinstance(report, dict)
        assert 'timestamp' in report

        # Check that file operations were called
        mock_makedirs.assert_called_once_with('/path/to', exist_ok=True)
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics instance."""
        timestamp = datetime.now()

        metrics = PerformanceMetrics(
            timestamp=timestamp,
            auc=0.85,
            f1_score=0.82,
            precision=0.78,
            recall=0.87,
            pnl=0.015,
            sharpe_ratio=1.8,
            max_drawdown=-0.08,
            total_trades=45,
            win_rate=0.67,
            calibration_error=0.12,
            sample_size=100
        )

        assert metrics.timestamp == timestamp
        assert metrics.auc == 0.85
        assert metrics.f1_score == 0.82
        assert metrics.precision == 0.78
        assert metrics.recall == 0.87
        assert metrics.pnl == 0.015
        assert metrics.sharpe_ratio == 1.8
        assert metrics.max_drawdown == -0.08
        assert metrics.total_trades == 45
        assert metrics.win_rate == 0.67
        assert metrics.calibration_error == 0.12
        assert metrics.sample_size == 100

    def test_performance_metrics_asdict(self):
        """Test converting PerformanceMetrics to dictionary."""
        timestamp = datetime.now()

        metrics = PerformanceMetrics(
            timestamp=timestamp,
            auc=0.85,
            f1_score=0.82,
            precision=0.78,
            recall=0.87,
            pnl=0.015,
            sharpe_ratio=1.8,
            max_drawdown=-0.08,
            total_trades=45,
            win_rate=0.67,
            calibration_error=0.12,
            sample_size=100
        )

        metrics_dict = metrics.__dict__

        assert isinstance(metrics_dict, dict)
        assert 'timestamp' in metrics_dict
        assert 'auc' in metrics_dict
        assert metrics_dict['auc'] == 0.85


class TestDriftMetrics:
    """Test DriftMetrics dataclass."""

    def test_drift_metrics_creation(self):
        """Test creating DriftMetrics instance."""
        timestamp = datetime.now()

        metrics = DriftMetrics(
            timestamp=timestamp,
            feature_drift_scores={'feature1': 0.1, 'feature2': 0.05},
            prediction_drift_score=0.08,
            label_drift_score=0.03,
            overall_drift_score=0.07,
            is_drift_detected=False
        )

        assert metrics.timestamp == timestamp
        assert metrics.feature_drift_scores == {'feature1': 0.1, 'feature2': 0.05}
        assert metrics.prediction_drift_score == 0.08
        assert metrics.label_drift_score == 0.03
        assert metrics.overall_drift_score == 0.07
        assert not metrics.is_drift_detected


class TestModelHealthReport:
    """Test ModelHealthReport dataclass."""

    def test_health_report_creation(self):
        """Test creating ModelHealthReport instance."""
        timestamp = datetime.now()

        report = ModelHealthReport(
            timestamp=timestamp,
            overall_health_score=0.85,
            performance_score=0.82,
            drift_score=0.1,
            calibration_score=0.88,
            recommendations=["Model performing well", "Monitor drift closely"],
            requires_retraining=False,
            confidence_level="HIGH"
        )

        assert report.timestamp == timestamp
        assert report.overall_health_score == 0.85
        assert report.performance_score == 0.82
        assert report.drift_score == 0.1
        assert report.calibration_score == 0.88
        assert report.recommendations == ["Model performing well", "Monitor drift closely"]
        assert not report.requires_retraining
        assert report.confidence_level == "HIGH"


if __name__ == '__main__':
    pytest.main([__file__])
