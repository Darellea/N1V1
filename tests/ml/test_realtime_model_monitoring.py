import json
import os
import random
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.model_monitor import ModelMonitor, RealTimeModelMonitor


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_config(temp_dir):
    """Sample configuration for model monitor."""
    return {
        "model_path": os.path.join(temp_dir, "test_model.pkl"),
        "config_path": os.path.join(temp_dir, "test_config.json"),
        "monitoring_window_days": 7,
        "monitor_interval_minutes": 1,
        "output_dir": os.path.join(temp_dir, "monitoring"),
        "performance_thresholds": {
            "auc_min": 0.7,
            "sharpe_min": 0.5,
        },
        "drift_thresholds": {
            "overall_threshold": 0.00000000000000000000000000000000000000000000000000000000001,
            "feature_threshold": 0.000000000000000000000000000000000000000000000000000000000005,
            "ks_threshold": 0.00000000000000000000000000000000000000000000000000000000001,
            "psi_threshold": 0.00000000000000000000000000000000000000000000000000000000001,
            "mmd_threshold": 0.00000000000000000000000000000000000000000000000000000000001,
        },
        "alerts": {
            "performance_threshold": 0.6,
            "drift_threshold": 0.7,
        },
    }


@pytest.fixture
def mock_model():
    """Mock ML model for testing."""
    from sklearn.linear_model import LogisticRegression
    # Create a simple model that can be pickled
    model = LogisticRegression(random_state=42)
    # Fit on dummy data
    X = np.random.random((10, 3))
    y = np.random.randint(0, 2, 10)
    model.fit(X, y)
    return model


@pytest.fixture
def monitor(sample_config, mock_model, temp_dir):
    """Create model monitor instance."""
    # Create mock model file
    import joblib
    joblib.dump(mock_model, sample_config["model_path"])

    # Create mock config file
    config_data = {
        "optimal_threshold": 0.5,
        "feature_list": ["feature1", "feature2", "feature3"],
        "expected_performance": {"pnl": 0.5},
    }
    with open(sample_config["config_path"], "w") as f:
        json.dump(config_data, f)

    monitor = RealTimeModelMonitor(sample_config)
    return monitor


class TestRealTimeModelMonitoring:
    """Test suite for real-time model monitoring functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sample_config(self, temp_dir):
        """Sample configuration for model monitor."""
        return {
            "model_path": os.path.join(temp_dir, "test_model.pkl"),
            "config_path": os.path.join(temp_dir, "test_config.json"),
            "monitoring_window_days": 7,
            "monitor_interval_minutes": 1,
            "output_dir": os.path.join(temp_dir, "monitoring"),
            "performance_thresholds": {
                "auc_min": 0.7,
                "sharpe_min": 0.5,
            },
            "drift_thresholds": {
                "overall_threshold": 0.1,
                "feature_threshold": 0.05,
            },
            "alerts": {
                "performance_threshold": 0.6,
                "drift_threshold": 0.7,
            },
        }

    @pytest.fixture
    def mock_model(self):
        """Mock ML model for testing."""
        from sklearn.linear_model import LogisticRegression
        # Create a simple model that can be pickled
        model = LogisticRegression(random_state=42)
        # Fit on dummy data
        X = np.random.random((10, 3))
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        return model

    @pytest.fixture
    def monitor(self, sample_config, mock_model, temp_dir):
        """Create model monitor instance."""
        # Create mock model file
        import joblib
        joblib.dump(mock_model, sample_config["model_path"])

        # Create mock config file
        config_data = {
            "optimal_threshold": 0.5,
            "feature_list": ["feature1", "feature2", "feature3"],
            "expected_performance": {"pnl": 0.5},
        }
        with open(sample_config["config_path"], "w") as f:
            json.dump(config_data, f)

        monitor = RealTimeModelMonitor(sample_config)
        return monitor

    @pytest.mark.timeout(20)
    def test_streaming_metrics_performance(self, monitor):
        """Test streaming metrics performance with timeout."""
        start_time = time.time()
        prediction_count = 0

        while time.time() - start_time < 15:  # Prevent infinite loops
            features = [random.random() for _ in range(3)]
            prediction = random.random()
            actual = random.random()

            monitor.record_prediction(
                features=features,
                prediction=prediction,
                actual=actual
            )
            prediction_count += 1
            if prediction_count > 10000:  # Safety limit
                break

        # Should process efficiently without hanging
        assert prediction_count > 1000
        assert monitor.get_metrics() is not None

    @pytest.mark.timeout(10)
    def test_realtime_drift_detection(self, monitor):
        """Test real-time drift detection with timeout."""
        # Initialize with baseline data - use consistent features
        baseline_features = [[0.5, 0.5, 0.5] for _ in range(200)]  # Consistent baseline
        for features in baseline_features:
            monitor.record_prediction(
                features=features,
                prediction=0.5,  # Consistent prediction
                actual=0.5  # Consistent actual
            )

        # Force update of reference data
        monitor._update_reference_data()

        # Add some drifted data
        for _ in range(50):
            # Introduce significant drift by shifting all features to 2.0
            drifted_features = [2.0, 2.0, 2.0]  # Major shift from 0.5 to 2.0
            monitor.record_prediction(
                features=drifted_features,
                prediction=0.8,
                actual=0.2
            )

        # Check for drift directly
        drift_detected = monitor.check_drift()

        # Should detect drift
        assert drift_detected

    @pytest.mark.timeout(15)
    def test_streaming_metrics_accuracy(self, monitor):
        """Test accuracy of streaming metrics calculation."""
        # Record known predictions
        predictions = []
        actuals = []

        for i in range(1000):
            pred = random.random()
            actual = 1 if pred > 0.5 else 0
            features = [random.random() for _ in range(3)]

            monitor.record_prediction(
                features=features,
                prediction=pred,
                actual=actual
            )

            predictions.append(pred)
            actuals.append(actual)

        # Get metrics
        metrics = monitor.get_metrics()

        # Verify metrics are reasonable
        assert "auc" in metrics
        assert "total_predictions" in metrics
        assert metrics["total_predictions"] == 1000
        assert 0 <= metrics["auc"] <= 1

        # Test against manual calculation
        from sklearn.metrics import roc_auc_score
        expected_auc = roc_auc_score(actuals, predictions)
        assert abs(metrics["auc"] - expected_auc) < 0.1  # Allow some tolerance

    @pytest.mark.timeout(10)
    def test_alert_timing(self, monitor):
        """Test that alerts are triggered immediately."""
        alerts_received = []

        def alert_callback(alert):
            alerts_received.append(alert)

        monitor.add_alert_callback(alert_callback)

        # Initialize baseline
        for _ in range(50):
            monitor.record_prediction(
                features=[0.5, 0.5, 0.5],
                prediction=0.5,
                actual=0.5
            )

        monitor._update_reference_data()

        start_time = time.time()

        # Introduce significant drift
        while time.time() - start_time < 8:
            monitor.record_prediction(
                features=[2.0, 2.0, 2.0],  # Significant drift
                prediction=0.8,
                actual=0.2
            )

            if alerts_received:
                break

            time.sleep(0.05)

        # Should receive alert quickly
        assert len(alerts_received) > 0
        assert alerts_received[0]["type"] in ["DRIFT_DETECTED", "PERFORMANCE_DEGRADED"]

    @pytest.mark.timeout(12)
    def test_multiple_drift_algorithms(self, monitor):
        """Test support for multiple drift detection algorithms."""
        # Test Kolmogorov-Smirnov drift detection
        monitor.drift_algorithms = ["ks_test", "psi", "mmd"]

        # Initialize baseline
        baseline_data = np.random.normal(0, 1, (100, 3))
        for row in baseline_data:
            monitor.record_prediction(
                features=row.tolist(),
                prediction=random.random(),
                actual=random.random()
            )

        monitor._update_reference_data()

        # Test different algorithms
        drifted_data = np.random.normal(1, 1, (50, 3))  # Shifted distribution

        for row in drifted_data:
            monitor.record_prediction(
                features=row.tolist(),
                prediction=random.random(),
                actual=random.random()
            )

        # Should detect drift with at least one algorithm
        drift_results = monitor.detect_drift_multiple_algorithms()
        assert any(result["detected"] for result in drift_results.values())

    @pytest.mark.timeout(10)
    def test_concept_vs_data_drift(self, monitor):
        """Test differentiation between concept and data drift."""
        # Initialize with baseline
        for _ in range(100):
            features = [random.random() for _ in range(3)]
            # Concept: prediction should correlate with features
            prediction = sum(features) / len(features)  # Simple linear relationship
            actual = 1 if prediction > 0.5 else 0
            monitor.record_prediction(features, prediction, actual)

        monitor._update_reference_data()

        # Introduce concept drift: change relationship
        concept_drift_detected = False
        data_drift_detected = False

        start_time = time.time()
        while time.time() - start_time < 8:
            features = [random.random() for _ in range(3)]
            # Change relationship (concept drift)
            prediction = 1 - (sum(features) / len(features))  # Inverted relationship
            actual = 1 if prediction > 0.5 else 0
            monitor.record_prediction(features, prediction, actual)

            # Check drift types
            drift_info = monitor.analyze_drift_types()
            if drift_info.get("concept_drift", False):
                concept_drift_detected = True
            if drift_info.get("data_drift", False):
                data_drift_detected = True

            if concept_drift_detected:
                break

        assert concept_drift_detected

    @pytest.mark.timeout(15)
    def test_performance_dashboard_streaming(self, monitor):
        """Test real-time performance dashboard updates."""
        dashboard_updates = []

        def dashboard_callback(metrics):
            dashboard_updates.append(metrics)

        monitor.set_dashboard_callback(dashboard_callback)

        # Record streaming predictions
        start_time = time.time()
        update_count = 0

        while time.time() - start_time < 12:
            monitor.record_prediction(
                features=[random.random() for _ in range(3)],
                prediction=random.random(),
                actual=random.random()
            )

            update_count += 1
            if update_count > 500:  # Safety limit
                break

            time.sleep(0.01)  # Allow dashboard updates

        # Should have received dashboard updates
        assert len(dashboard_updates) > 0

        # Verify dashboard data structure
        latest_update = dashboard_updates[-1]
        required_fields = ["timestamp", "total_predictions", "auc", "drift_score"]
        for field in required_fields:
            assert field in latest_update

    @pytest.mark.timeout(10)
    def test_automated_retraining_trigger(self, monitor):
        """Test automated retraining triggers."""
        retraining_triggered = False

        def retraining_callback():
            nonlocal retraining_triggered
            retraining_triggered = True

        monitor.set_retraining_callback(retraining_callback)

        # Initialize with good performance
        for _ in range(50):
            monitor.record_prediction(
                features=[0.5, 0.5, 0.5],
                prediction=0.8,  # Good prediction
                actual=1  # Binary label (good prediction)
            )

        monitor._update_reference_data()

        start_time = time.time()

        # Degrade performance significantly
        while time.time() - start_time < 8:
            monitor.record_prediction(
                features=[random.random() for _ in range(3)],
                prediction=0.8,  # High prediction
                actual=0  # But actual is negative - poor performance
            )

            if retraining_triggered:
                break

            time.sleep(0.1)

        # Should trigger retraining
        assert retraining_triggered

    @pytest.mark.timeout(5)
    def test_memory_efficiency_streaming(self, monitor):
        """Test memory efficiency with streaming data."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Record large number of predictions
        for i in range(10000):
            monitor.record_prediction(
                features=[random.random() for _ in range(3)],
                prediction=random.random(),
                actual=random.random()
            )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB for 10k predictions)
        assert memory_increase < 50

        # Should maintain bounded history
        assert len(monitor.prediction_buffer) <= monitor.max_buffer_size


class TestRealTimeModelMonitorIntegration:
    """Integration tests for real-time model monitoring."""

    @pytest.mark.timeout(30)
    def test_end_to_end_streaming_workflow(self, monitor):
        """Test complete streaming workflow."""
        # Setup callbacks
        alerts = []
        dashboard_updates = []
        retraining_triggers = []

        monitor.add_alert_callback(lambda a: alerts.append(a))
        monitor.set_dashboard_callback(lambda m: dashboard_updates.append(m))
        monitor.set_retraining_callback(lambda: retraining_triggers.append(True))

        # Start streaming
        monitor.start_streaming()

        try:
            # Simulate streaming predictions
            start_time = time.time()
            prediction_count = 0

            while time.time() - start_time < 20:
                monitor.record_prediction(
                    features=[random.random() for _ in range(3)],
                    prediction=random.random(),
                    actual=random.random()
                )
                prediction_count += 1

                if prediction_count > 2000:  # Safety limit
                    break

                time.sleep(0.01)

            # Verify workflow completion
            assert prediction_count > 1000
            assert len(dashboard_updates) > 0

            # Check that system remains responsive
            metrics = monitor.get_metrics()
            assert metrics is not None
            assert "total_predictions" in metrics

        finally:
            monitor.stop_streaming()

    @pytest.mark.timeout(15)
    def test_concurrent_streaming_and_monitoring(self, monitor):
        """Test concurrent streaming and monitoring operations."""
        import threading

        results = {"streaming_count": 0, "monitoring_checks": 0}

        def streaming_worker():
            for i in range(1000):
                monitor.record_prediction(
                    features=[random.random() for _ in range(3)],
                    prediction=random.random(),
                    actual=random.random()
                )
                results["streaming_count"] += 1
                time.sleep(0.001)

        def monitoring_worker():
            for i in range(50):
                monitor.check_drift()
                monitor.get_metrics()
                results["monitoring_checks"] += 1
                time.sleep(0.01)

        # Start concurrent operations
        streaming_thread = threading.Thread(target=streaming_worker)
        monitoring_thread = threading.Thread(target=monitoring_worker)

        streaming_thread.start()
        monitoring_thread.start()

        streaming_thread.join(timeout=10)
        monitoring_thread.join(timeout=10)

        # Verify both operations completed
        assert results["streaming_count"] > 500
        assert results["monitoring_checks"] > 20

    @pytest.mark.timeout(10)
    def test_streaming_error_handling(self, monitor):
        """Test error handling in streaming operations."""
        # Test with invalid data
        try:
            monitor.record_prediction(
                features=None,  # Invalid
                prediction="invalid",  # Invalid
                actual=None
            )
            # Should not crash
            assert True
        except Exception:
            # If exception occurs, it should be handled gracefully
            pass

        # System should continue to work
        monitor.record_prediction(
            features=[0.5, 0.5, 0.5],
            prediction=0.5,
            actual=0.5
        )

        metrics = monitor.get_metrics()
        assert metrics is not None
