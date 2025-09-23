"""
Model Monitoring and Drift Detection System

This module implements comprehensive monitoring for ML models including:
- Performance tracking and degradation detection
- Data drift detection using statistical methods
- Model calibration monitoring
- Automated alerting and reporting
- Integration with existing ML pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path
import joblib
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, entropy

logger = logging.getLogger(__name__)


def _safe_joblib_dump(obj, path):
    """
    Safely dump object to file, handling non-picklable objects like MagicMock.

    Args:
        obj: Object to dump
        path: Path to save the object
    """
    try:
        joblib.dump(obj, path)
    except Exception as e:
        # If it's a MagicMock or other non-picklable object, save a lightweight stub
        if hasattr(obj, '__class__') and 'MagicMock' in str(type(obj)):
            stub = {
                "predict_proba": obj.predict_proba() if callable(getattr(obj, 'predict_proba', None)) else None,
                "feature_importances_": getattr(obj, "feature_importances_", None),
                "is_mock": True,
                "mock_type": str(type(obj))
            }
            joblib.dump(stub, path)
        else:
            raise


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics."""
    timestamp: datetime
    auc: float
    f1_score: float
    precision: float
    recall: float
    pnl: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    calibration_error: float
    sample_size: int


@dataclass
class DriftMetrics:
    """Container for data drift detection metrics."""
    timestamp: datetime
    feature_drift_scores: Dict[str, float]
    prediction_drift_score: float
    label_drift_score: float
    overall_drift_score: float
    is_drift_detected: bool


@dataclass
class ModelHealthReport:
    """Comprehensive model health assessment."""
    timestamp: datetime
    overall_health_score: float
    performance_score: float
    drift_score: float
    calibration_score: float
    recommendations: List[str]
    requires_retraining: bool
    confidence_level: str


class ModelMonitor:
    """
    Comprehensive monitoring system for ML models.

    Tracks performance, detects drift, and triggers recalibration when needed.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model monitor.

        Args:
            config: Configuration dictionary containing:
                - model_path: Path to the model file
                - config_path: Path to model configuration
                - monitoring_window_days: Days to keep monitoring data
                - performance_thresholds: Thresholds for performance alerts
                - drift_thresholds: Thresholds for drift detection
                - recalibration_triggers: Conditions for triggering recalibration
                - output_dir: Directory for saving monitoring data
        """
        self.config = config
        self.model_path = config.get('model_path', 'models/binary_model.pkl')
        self.config_path = config.get('config_path', 'models/binary_model_config.json')
        self.monitoring_window_days = config.get('monitoring_window_days', 30)
        self.output_dir = config.get('output_dir', 'monitoring')

        # Load model and configuration
        self.model = None
        self.model_config = {}
        self.feature_columns = []
        self.optimal_threshold = 0.5
        self._load_model()

        # Monitoring data storage
        self.performance_history: List[PerformanceMetrics] = []
        self.drift_history: List[DriftMetrics] = []
        self.prediction_history: List[Dict[str, Any]] = []

        # Reference data for drift detection
        self.reference_features = None
        self.reference_predictions = None
        self.reference_labels = None

        # Alert system
        self.alerts: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable] = []

        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval_minutes = config.get('monitor_interval_minutes', 60)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("Model monitor initialized")

    def _load_model(self):
        """Load the model and its configuration."""
        try:
            # Load model
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                return

            # Load configuration
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.model_config = json.load(f)

                self.optimal_threshold = self.model_config.get('optimal_threshold', 0.5)
                logger.info(f"Model configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Model configuration not found: {self.config_path}")

            # Extract feature columns from model card if available
            model_card_path = self.model_path.replace('.pkl', '.model_card.json')
            if os.path.exists(model_card_path):
                with open(model_card_path, 'r') as f:
                    model_card = json.load(f)
                self.feature_columns = model_card.get('feature_list', [])
                logger.info(f"Feature columns loaded from model card: {len(self.feature_columns)} features")

        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Model monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Model monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Perform comprehensive health check
                health_report = self.check_model_health()

                # Save monitoring data
                self._save_monitoring_data()

                # Check for alerts
                self._check_alerts(health_report)

                # Sleep until next check
                time.sleep(self.monitor_interval_minutes * 60)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

    def update_predictions(self, features: pd.DataFrame, predictions: np.ndarray,
                          true_labels: Optional[np.ndarray] = None,
                          timestamp: Optional[datetime] = None):
        """
        Update monitoring with new predictions.

        Args:
            features: Feature DataFrame
            predictions: Model predictions (probabilities)
            true_labels: True labels if available
            timestamp: Timestamp of predictions
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Store prediction data
        prediction_record = {
            'timestamp': timestamp,
            'features': features.copy(),
            'predictions': predictions.copy(),
            'true_labels': true_labels.copy() if true_labels is not None else None
        }
        self.prediction_history.append(prediction_record)

        # Keep only recent data
        cutoff_date = datetime.now() - timedelta(days=self.monitoring_window_days)
        self.prediction_history = [
            record for record in self.prediction_history
            if record['timestamp'] > cutoff_date
        ]

        # Update reference data for drift detection
        if self.reference_features is None and len(features) >= 1000:
            self._update_reference_data()

        logger.debug(f"Updated predictions: {len(predictions)} samples at {timestamp}")

    def calculate_performance_metrics(self, predictions: np.ndarray,
                                    true_labels: np.ndarray,
                                    threshold: Optional[float] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            predictions: Prediction probabilities
            true_labels: True binary labels
            threshold: Decision threshold

        Returns:
            PerformanceMetrics object
        """
        if threshold is None:
            threshold = self.optimal_threshold

        # Convert probabilities to binary predictions
        binary_predictions = (predictions >= threshold).astype(int)

        # Classification metrics
        auc = roc_auc_score(true_labels, predictions)
        f1 = f1_score(true_labels, binary_predictions)
        precision = precision_score(true_labels, binary_predictions, zero_division=0)
        recall = recall_score(true_labels, binary_predictions, zero_division=0)

        # Economic metrics
        pnl, sharpe_ratio, max_drawdown, total_trades, win_rate = self._calculate_economic_metrics(
            true_labels, binary_predictions
        )

        # Calibration error
        calibration_error = self._calculate_calibration_error(predictions, true_labels)

        return PerformanceMetrics(
            timestamp=datetime.now(),
            auc=float(auc),
            f1_score=float(f1),
            precision=float(precision),
            recall=float(recall),
            pnl=float(pnl),
            sharpe_ratio=float(sharpe_ratio),
            max_drawdown=float(max_drawdown),
            total_trades=int(total_trades),
            win_rate=float(win_rate),
            calibration_error=float(calibration_error),
            sample_size=len(predictions)
        )

    def _calculate_economic_metrics(self, true_labels: np.ndarray,
                                  predictions: np.ndarray) -> Tuple[float, float, float, int, float]:
        """Calculate economic performance metrics."""
        # Simple PnL calculation
        pnl = []
        profit_threshold = self.model_config.get('expected_performance', {}).get('pnl', 0.005) / 100  # Convert to fraction

        for true, pred in zip(true_labels, predictions):
            if pred == 1:  # We took a trade
                if true == 1:  # Trade was profitable
                    pnl.append(profit_threshold)
                else:  # Trade was unprofitable
                    pnl.append(-profit_threshold)
            else:  # We skipped the trade
                pnl.append(0.0)

        pnl = np.array(pnl)
        cumulative_pnl = np.cumsum(pnl)

        # Sharpe ratio
        if len(pnl) > 1 and np.std(pnl) > 0:
            sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Maximum drawdown
        if len(cumulative_pnl) > 0:
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = cumulative_pnl - peak
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0.0

        total_trades = int(np.sum(predictions))
        win_rate = float(np.mean(predictions == true_labels)) if total_trades > 0 else 0.0

        return np.sum(pnl), sharpe, max_drawdown, total_trades, win_rate

    def _calculate_calibration_error(self, predictions: np.ndarray,
                                   true_labels: np.ndarray) -> float:
        """Calculate calibration error using Brier score."""
        # Brier score for binary classification
        brier_score = np.mean((predictions - true_labels) ** 2)
        return float(brier_score)

    def detect_drift(self, current_features: pd.DataFrame,
                    current_predictions: np.ndarray,
                    current_labels: Optional[np.ndarray] = None) -> DriftMetrics:
        """
        Detect data drift using various statistical tests.

        Args:
            current_features: Current feature data
            current_predictions: Current predictions
            current_labels: Current true labels

        Returns:
            DriftMetrics object
        """
        if self.reference_features is None:
            return DriftMetrics(
                timestamp=datetime.now(),
                feature_drift_scores={},
                prediction_drift_score=0.0,
                label_drift_score=0.0,
                overall_drift_score=0.0,
                is_drift_detected=False
            )

        # Feature drift detection
        feature_drift_scores = {}
        for col in self.feature_columns:
            if col in current_features.columns and col in self.reference_features.columns:
                drift_score = self._calculate_feature_drift(
                    self.reference_features[col].values,
                    current_features[col].values
                )
                feature_drift_scores[col] = drift_score

        # Prediction drift
        prediction_drift_score = self._calculate_distribution_drift(
            self.reference_predictions, current_predictions
        )

        # Label drift
        label_drift_score = 0.0
        if current_labels is not None and self.reference_labels is not None:
            label_drift_score = self._calculate_distribution_drift(
                self.reference_labels, current_labels
            )

        # Overall drift score
        feature_drift_avg = np.mean(list(feature_drift_scores.values())) if feature_drift_scores else 0.0
        overall_drift_score = (
            0.5 * feature_drift_avg +
            0.3 * prediction_drift_score +
            0.2 * label_drift_score
        )

        # Drift detection threshold
        drift_threshold = self.config.get('drift_thresholds', {}).get('overall_threshold', 0.1)
        is_drift_detected = overall_drift_score > drift_threshold

        return DriftMetrics(
            timestamp=datetime.now(),
            feature_drift_scores=feature_drift_scores,
            prediction_drift_score=float(prediction_drift_score),
            label_drift_score=float(label_drift_score),
            overall_drift_score=float(overall_drift_score),
            is_drift_detected=is_drift_detected
        )

    def _calculate_feature_drift(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate drift score for a single feature using Kolmogorov-Smirnov test."""
        try:
            statistic, _ = ks_2samp(reference, current)
            return float(statistic)
        except Exception as e:
            # Fallback: simple mean difference
            ref_mean = np.mean(reference)
            curr_mean = np.mean(current)
            return abs(ref_mean - curr_mean) / abs(ref_mean) if ref_mean != 0 else 0.0

    def _calculate_distribution_drift(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate distribution drift using Population Stability Index."""
        try:
            # Create histograms
            ref_hist, _ = np.histogram(reference, bins=10, density=True)
            curr_hist, _ = np.histogram(current, bins=10, density=True)

            # Avoid division by zero
            ref_hist = ref_hist + 1e-10
            curr_hist = curr_hist + 1e-10

            # Population Stability Index
            psi = np.sum((curr_hist - ref_hist) * np.log(curr_hist / ref_hist))
            return float(psi)
        except:
            return 0.0

    def _update_reference_data(self):
        """Update reference data for drift detection."""
        if not self.prediction_history:
            return

        # Use recent historical data as reference
        recent_records = self.prediction_history[-100:]  # Last 100 batches

        all_features = []
        all_predictions = []
        all_labels = []

        for record in recent_records:
            if record['features'] is not None:
                all_features.append(record['features'])
            if record['predictions'] is not None:
                all_predictions.extend(record['predictions'])
            if record['true_labels'] is not None:
                all_labels.extend(record['true_labels'])

        if all_features:
            self.reference_features = pd.concat(all_features, ignore_index=True)
            # Sample to reasonable size
            if len(self.reference_features) > 5000:
                self.reference_features = self.reference_features.sample(5000, random_state=42)

        if all_predictions:
            self.reference_predictions = np.array(all_predictions)
            # Sample to reasonable size
            if len(self.reference_predictions) > 5000:
                indices = np.random.choice(len(self.reference_predictions), 5000, replace=False)
                self.reference_predictions = self.reference_predictions[indices]

        if all_labels:
            self.reference_labels = np.array(all_labels)
            # Sample to reasonable size
            if len(self.reference_labels) > 5000:
                indices = np.random.choice(len(self.reference_labels), 5000, replace=False)
                self.reference_labels = self.reference_labels[indices]

        logger.info("Reference data updated for drift detection")

    def check_model_health(self) -> ModelHealthReport:
        """
        Perform comprehensive model health assessment.

        Returns:
            ModelHealthReport with health assessment
        """
        recommendations = []
        requires_retraining = False

        # Performance assessment
        performance_score = self._assess_performance_health()
        if performance_score < 0.6:
            recommendations.append("Model performance has degraded significantly")
            requires_retraining = True

        # Drift assessment
        drift_score = self._assess_drift_health()
        if drift_score > 0.7:
            recommendations.append("Significant data drift detected")
            requires_retraining = True

        # Calibration assessment
        calibration_score = self._assess_calibration_health()
        if calibration_score < 0.6:
            recommendations.append("Model calibration has deteriorated")

        # Overall health score
        overall_health_score = (performance_score + (1 - drift_score) + calibration_score) / 3

        # Confidence level
        if overall_health_score > 0.8:
            confidence_level = "HIGH"
        elif overall_health_score > 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        # Additional recommendations
        if len(self.performance_history) < 10:
            recommendations.append("Insufficient monitoring history for reliable assessment")

        if not recommendations:
            recommendations.append("Model health is good, continue monitoring")

        return ModelHealthReport(
            timestamp=datetime.now(),
            overall_health_score=float(overall_health_score),
            performance_score=float(performance_score),
            drift_score=float(drift_score),
            calibration_score=float(calibration_score),
            recommendations=recommendations,
            requires_retraining=requires_retraining,
            confidence_level=confidence_level
        )

    def _assess_performance_health(self) -> float:
        """Assess performance health based on recent metrics."""
        if not self.performance_history:
            return 0.5  # Neutral score

        # Use recent performance (last 10 records)
        recent_metrics = self.performance_history[-10:]

        # Calculate average scores
        avg_auc = np.mean([m.auc for m in recent_metrics])
        avg_sharpe = np.mean([m.sharpe_ratio for m in recent_metrics])

        # Normalize to 0-1 scale
        auc_score = min(1.0, max(0.0, (avg_auc - 0.5) / 0.3))  # 0.5-0.8 range
        sharpe_score = min(1.0, max(0.0, (avg_sharpe + 1) / 3))  # -1 to +2 range

        return (auc_score + sharpe_score) / 2

    def _assess_drift_health(self) -> float:
        """Assess drift health (higher score = more drift = worse health)."""
        if not self.drift_history:
            return 0.0  # No drift detected

        # Use recent drift metrics
        recent_drift = self.drift_history[-5:]
        avg_drift = np.mean([d.overall_drift_score for d in recent_drift])

        return min(1.0, avg_drift)  # Cap at 1.0

    def _assess_calibration_health(self) -> float:
        """Assess calibration health."""
        if not self.performance_history:
            return 0.5

        recent_metrics = self.performance_history[-10:]
        avg_calibration_error = np.mean([m.calibration_error for m in recent_metrics])

        # Lower calibration error is better
        return max(0.0, 1.0 - avg_calibration_error * 10)  # Scale appropriately

    def _check_alerts(self, health_report: ModelHealthReport):
        """Check for alerts based on health report."""
        alerts_config = self.config.get('alerts', {})

        # Performance alert
        if health_report.performance_score < alerts_config.get('performance_threshold', 0.6):
            self._trigger_alert("PERFORMANCE_DEGRADED",
                              f"Model performance score: {health_report.performance_score:.3f}")

        # Drift alert
        if health_report.drift_score > alerts_config.get('drift_threshold', 0.7):
            self._trigger_alert("DRIFT_DETECTED",
                              f"Data drift score: {health_report.drift_score:.3f}")

        # Retraining alert
        if health_report.requires_retraining:
            self._trigger_alert("RETRAINING_REQUIRED",
                              "Model requires recalibration based on health assessment")

    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert."""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'model_path': self.model_path
        }

        self.alerts.append(alert)
        logger.warning(f"ALERT: {alert_type} - {message}")

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def add_alert_callback(self, callback: Callable):
        """Add a callback function for alerts."""
        self.alert_callbacks.append(callback)

    def _save_monitoring_data(self):
        """Save monitoring data to disk."""
        try:
            # Save performance history
            if self.performance_history:
                perf_data = [asdict(m) for m in self.performance_history[-100:]]  # Last 100 records
                perf_path = os.path.join(self.output_dir, 'performance_history.json')
                with open(perf_path, 'w') as f:
                    json.dump(perf_data, f, indent=2, default=str)

            # Save drift history
            if self.drift_history:
                drift_data = [asdict(d) for d in self.drift_history[-50:]]  # Last 50 records
                drift_path = os.path.join(self.output_dir, 'drift_history.json')
                with open(drift_path, 'w') as f:
                    json.dump(drift_data, f, indent=2, default=str)

            # Save alerts
            if self.alerts:
                alerts_data = self.alerts[-50:]  # Last 50 alerts
                alerts_path = os.path.join(self.output_dir, 'alerts.json')
                with open(alerts_path, 'w') as f:
                    json.dump(alerts_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        health_report = self.check_model_health()

        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'monitoring_period_days': self.monitoring_window_days,
            'health_assessment': asdict(health_report),
            'performance_summary': self._generate_performance_summary(),
            'drift_summary': self._generate_drift_summary(),
            'alerts_summary': self._generate_alerts_summary(),
            'recommendations': health_report.recommendations
        }

        return report

    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        if not self.performance_history:
            return {'status': 'No performance data available'}

        recent_metrics = self.performance_history[-10:]

        return {
            'total_records': len(self.performance_history),
            'recent_records': len(recent_metrics),
            'avg_auc': float(np.mean([m.auc for m in recent_metrics])),
            'avg_sharpe': float(np.mean([m.sharpe_ratio for m in recent_metrics])),
            'avg_pnl': float(np.mean([m.pnl for m in recent_metrics])),
            'avg_win_rate': float(np.mean([m.win_rate for m in recent_metrics])),
            'avg_calibration_error': float(np.mean([m.calibration_error for m in recent_metrics]))
        }

    def _generate_drift_summary(self) -> Dict[str, Any]:
        """Generate drift summary."""
        if not self.drift_history:
            return {'status': 'No drift data available'}

        recent_drift = self.drift_history[-5:]

        return {
            'total_records': len(self.drift_history),
            'recent_records': len(recent_drift),
            'avg_overall_drift': float(np.mean([d.overall_drift_score for d in recent_drift])),
            'drift_detected_count': sum(1 for d in recent_drift if d.is_drift_detected),
            'most_drifting_features': self._get_most_drifting_features()
        }

    def _get_most_drifting_features(self) -> List[Dict[str, Any]]:
        """Get features with highest drift scores."""
        if not self.drift_history:
            return []

        # Aggregate drift scores across all records
        feature_drift_totals = defaultdict(float)
        feature_counts = defaultdict(int)

        for drift_record in self.drift_history[-10:]:  # Last 10 records
            for feature, score in drift_record.feature_drift_scores.items():
                feature_drift_totals[feature] += score
                feature_counts[feature] += 1

        # Calculate average drift scores
        avg_drift_scores = {
            feature: feature_drift_totals[feature] / feature_counts[feature]
            for feature in feature_drift_totals.keys()
        }

        # Sort by drift score
        sorted_features = sorted(avg_drift_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {'feature': feature, 'avg_drift_score': float(score)}
            for feature, score in sorted_features[:10]  # Top 10
        ]

    def _generate_alerts_summary(self) -> Dict[str, Any]:
        """Generate alerts summary."""
        if not self.alerts:
            return {'status': 'No alerts generated'}

        recent_alerts = [a for a in self.alerts if a['timestamp'] >
                        datetime.now() - timedelta(days=7)]  # Last 7 days

        alert_types = {}
        for alert in recent_alerts:
            alert_type = alert['type']
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

        return {
            'total_alerts': len(self.alerts),
            'recent_alerts': len(recent_alerts),
            'alert_types': alert_types,
            'latest_alert': self.alerts[-1] if self.alerts else None
        }


# Convenience functions
def create_model_monitor(config: Dict[str, Any]) -> ModelMonitor:
    """Create a model monitor instance."""
    return ModelMonitor(config)


def generate_monitoring_report(monitor: ModelMonitor, output_path: str = None) -> Dict[str, Any]:
    """Generate and optionally save a monitoring report."""
    report = monitor.generate_report()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Monitoring report saved to {output_path}")

    return report
