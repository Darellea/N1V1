"""
Model Monitoring and Auto-Recalibration System

This module implements comprehensive monitoring for binary trading models,
including performance tracking, drift detection, and automatic model updates.

Key Features:
- Real-time performance monitoring
- Data drift detection
- Model performance degradation alerts
- Automatic model recalibration
- Performance reporting and visualization
- Integration with existing trading infrastructure
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


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
    Comprehensive monitoring system for binary trading models.

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
        self.model_path = config.get("model_path", "models/binary_model.pkl")
        self.config_path = config.get("config_path", "models/binary_model_config.json")
        self.monitoring_window_days = config.get("monitoring_window_days", 30)
        self.output_dir = config.get("output_dir", "monitoring")

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
        self.monitor_interval_minutes = config.get("monitor_interval_minutes", 60)

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
                with open(self.config_path, "r") as f:
                    self.model_config = json.load(f)

                self.optimal_threshold = self.model_config.get("optimal_threshold", 0.5)
                logger.info(f"Model configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Model configuration not found: {self.config_path}")

            # Extract feature columns from model card if available
            model_card_path = self.model_path.replace(".pkl", ".model_card.json")
            if os.path.exists(model_card_path):
                with open(model_card_path, "r") as f:
                    model_card = json.load(f)
                self.feature_columns = model_card.get("feature_list", [])
                logger.info(
                    f"Feature columns loaded from model card: {len(self.feature_columns)} features"
                )

        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return

        # Skip monitoring in testing mode to avoid thread conflicts
        if os.environ.get("TESTING"):
            logger.info("Skipping model monitoring in testing mode")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
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

    def update_predictions(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
    ):
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
            "timestamp": timestamp,
            "features": features.copy(),
            "predictions": predictions.copy(),
            "true_labels": true_labels.copy() if true_labels is not None else None,
        }
        self.prediction_history.append(prediction_record)

        # Keep only recent data
        cutoff_date = datetime.now() - timedelta(days=self.monitoring_window_days)
        self.prediction_history = [
            record
            for record in self.prediction_history
            if record["timestamp"] > cutoff_date
        ]

        # Update reference data for drift detection
        if self.reference_features is None and len(features) >= 1000:
            self._update_reference_data()

        logger.debug(f"Updated predictions: {len(predictions)} samples at {timestamp}")

    def calculate_performance_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        threshold: Optional[float] = None,
    ) -> PerformanceMetrics:
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
        (
            pnl,
            sharpe_ratio,
            max_drawdown,
            total_trades,
            win_rate,
        ) = self._calculate_economic_metrics(true_labels, binary_predictions)

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
            sample_size=len(predictions),
        )

    def _calculate_economic_metrics(
        self, true_labels: np.ndarray, predictions: np.ndarray
    ) -> Tuple[float, float, float, int, float]:
        """Calculate economic performance metrics."""
        # Simple PnL calculation
        pnl = []
        profit_threshold = (
            self.model_config.get("expected_performance", {}).get("pnl", 0.005) / 100
        )  # Convert to fraction

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
        win_rate = (
            float(np.mean(predictions == true_labels)) if total_trades > 0 else 0.0
        )

        return np.sum(pnl), sharpe, max_drawdown, total_trades, win_rate

    def _calculate_calibration_error(
        self, predictions: np.ndarray, true_labels: np.ndarray
    ) -> float:
        """Calculate calibration error using Brier score."""
        # Brier score for binary classification
        brier_score = np.mean((predictions - true_labels) ** 2)
        return float(brier_score)

    def detect_drift(
        self,
        current_features: pd.DataFrame,
        current_predictions: np.ndarray,
        current_labels: Optional[np.ndarray] = None,
    ) -> DriftMetrics:
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
                is_drift_detected=False,
            )

        # Feature drift detection
        feature_drift_scores = {}
        for col in self.feature_columns:
            if (
                col in current_features.columns
                and col in self.reference_features.columns
            ):
                drift_score = self._calculate_feature_drift(
                    self.reference_features[col].values, current_features[col].values
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
        feature_drift_avg = (
            np.mean(list(feature_drift_scores.values()))
            if feature_drift_scores
            else 0.0
        )
        overall_drift_score = (
            0.5 * feature_drift_avg
            + 0.3 * prediction_drift_score
            + 0.2 * label_drift_score
        )

        # Drift detection threshold
        drift_threshold = self.config.get("drift_thresholds", {}).get(
            "overall_threshold", 0.1
        )
        is_drift_detected = overall_drift_score > drift_threshold

        return DriftMetrics(
            timestamp=datetime.now(),
            feature_drift_scores=feature_drift_scores,
            prediction_drift_score=float(prediction_drift_score),
            label_drift_score=float(label_drift_score),
            overall_drift_score=float(overall_drift_score),
            is_drift_detected=is_drift_detected,
        )

    def _calculate_feature_drift(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """Calculate drift score for a single feature using Kolmogorov-Smirnov test."""
        try:
            from scipy.stats import ks_2samp

            statistic, _ = ks_2samp(reference, current)
            return float(statistic)
        except ImportError:
            # Fallback: simple mean difference
            ref_mean = np.mean(reference)
            curr_mean = np.mean(current)
            return abs(ref_mean - curr_mean) / abs(ref_mean) if ref_mean != 0 else 0.0

    def _calculate_distribution_drift(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
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
        # For streaming monitor, use prediction_buffer; for batch monitor, use prediction_history
        if hasattr(self, "prediction_buffer") and self.prediction_buffer:
            # Streaming monitor
            recent_records = self.prediction_buffer[-500:]  # Last 500 predictions

            all_features = []
            all_predictions = []
            all_labels = []

            for record in recent_records:
                if record["features"]:
                    all_features.append(record["features"])
                all_predictions.append(record["prediction"])
                if record["actual"] is not None:
                    all_labels.append(record["actual"])

            if all_features:
                # Create DataFrame with proper column names
                column_names = (
                    [f"feature{i}" for i in range(len(all_features[0]))]
                    if all_features
                    else []
                )
                self.reference_features = pd.DataFrame(
                    all_features, columns=column_names
                )
                # Sample to reasonable size
                if len(self.reference_features) > 1000:
                    self.reference_features = self.reference_features.sample(
                        1000, random_state=42
                    )

            if all_predictions:
                self.reference_predictions = np.array(all_predictions)
                # Sample to reasonable size
                if len(self.reference_predictions) > 1000:
                    indices = np.random.choice(
                        len(self.reference_predictions), 1000, replace=False
                    )
                    self.reference_predictions = self.reference_predictions[indices]

            if all_labels:
                self.reference_labels = np.array(all_labels)
                # Sample to reasonable size
                if len(self.reference_labels) > 1000:
                    indices = np.random.choice(
                        len(self.reference_labels), 1000, replace=False
                    )
                    self.reference_labels = self.reference_labels[indices]

        elif self.prediction_history:
            # Batch monitor
            recent_records = self.prediction_history[-100:]  # Last 100 batches

            all_features = []
            all_predictions = []
            all_labels = []

            for record in recent_records:
                if record["features"] is not None:
                    all_features.append(record["features"])
                if record["predictions"] is not None:
                    all_predictions.extend(record["predictions"])
                if record["true_labels"] is not None:
                    all_labels.extend(record["true_labels"])

            if all_features:
                self.reference_features = pd.concat(all_features, ignore_index=True)
                # Sample to reasonable size
                if len(self.reference_features) > 5000:
                    self.reference_features = self.reference_features.sample(
                        5000, random_state=42
                    )

            if all_predictions:
                self.reference_predictions = np.array(all_predictions)
                # Sample to reasonable size
                if len(self.reference_predictions) > 5000:
                    indices = np.random.choice(
                        len(self.reference_predictions), 5000, replace=False
                    )
                    self.reference_predictions = self.reference_predictions[indices]

            if all_labels:
                self.reference_labels = np.array(all_labels)
                # Sample to reasonable size
                if len(self.reference_labels) > 5000:
                    indices = np.random.choice(
                        len(self.reference_labels), 5000, replace=False
                    )
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
        overall_health_score = (
            performance_score + (1 - drift_score) + calibration_score
        ) / 3

        # Confidence level
        if overall_health_score > 0.8:
            confidence_level = "HIGH"
        elif overall_health_score > 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        # Additional recommendations
        if len(self.performance_history) < 10:
            recommendations.append(
                "Insufficient monitoring history for reliable assessment"
            )

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
            confidence_level=confidence_level,
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
        alerts_config = self.config.get("alerts", {})

        # Performance alert
        if health_report.performance_score < alerts_config.get(
            "performance_threshold", 0.6
        ):
            self._trigger_alert(
                "PERFORMANCE_DEGRADED",
                f"Model performance score: {health_report.performance_score:.3f}",
            )

        # Drift alert
        if health_report.drift_score > alerts_config.get("drift_threshold", 0.7):
            self._trigger_alert(
                "DRIFT_DETECTED", f"Data drift score: {health_report.drift_score:.3f}"
            )

        # Retraining alert
        if health_report.requires_retraining:
            self._trigger_alert(
                "RETRAINING_REQUIRED",
                "Model requires recalibration based on health assessment",
            )

    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert."""
        alert = {
            "timestamp": datetime.now(),
            "type": alert_type,
            "message": message,
            "model_path": self.model_path,
        }

        self.alerts.append(alert)
        logger.warning(f"ALERT: {alert_type} - {message}")

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _trigger_drift_alert(self):
        """Trigger drift alert to all registered callbacks."""
        alert = {"type": "DRIFT_DETECTED", "timestamp": time.time()}
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _trigger_retraining(self):
        """Trigger retraining callback."""
        if self.retraining_callbacks:
            for callback in self.retraining_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Retraining callback failed: {e}")

    def _calculate_recent_accuracy(self) -> float:
        """Calculate accuracy of recent predictions."""
        recent = self.prediction_buffer[-self.min_samples_for_metrics :]
        correct = sum(
            1 for p in recent if (p["prediction"] > 0.5) == (p["actual"] > 0.5)
        )
        return correct / len(recent)

    def add_alert_callback(self, callback: Callable):
        """Add a callback function for alerts."""
        self.alert_callbacks.append(callback)

    def _save_monitoring_data(self):
        """Save monitoring data to disk."""
        try:
            # Save performance history
            if self.performance_history:
                perf_data = [
                    asdict(m) for m in self.performance_history[-100:]
                ]  # Last 100 records
                perf_path = os.path.join(self.output_dir, "performance_history.json")
                with open(perf_path, "w") as f:
                    json.dump(perf_data, f, indent=2, default=str)

            # Save drift history
            if self.drift_history:
                drift_data = [
                    asdict(d) for d in self.drift_history[-50:]
                ]  # Last 50 records
                drift_path = os.path.join(self.output_dir, "drift_history.json")
                with open(drift_path, "w") as f:
                    json.dump(drift_data, f, indent=2, default=str)

            # Save alerts
            if self.alerts:
                alerts_data = self.alerts[-50:]  # Last 50 alerts
                alerts_path = os.path.join(self.output_dir, "alerts.json")
                with open(alerts_path, "w") as f:
                    json.dump(alerts_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        health_report = self.check_model_health()

        report = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "monitoring_period_days": self.monitoring_window_days,
            "health_assessment": asdict(health_report),
            "performance_summary": self._generate_performance_summary(),
            "drift_summary": self._generate_drift_summary(),
            "alerts_summary": self._generate_alerts_summary(),
            "recommendations": health_report.recommendations,
        }

        return report

    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        if not self.performance_history:
            return {"status": "No performance data available"}

        recent_metrics = self.performance_history[-10:]

        return {
            "total_records": len(self.performance_history),
            "recent_records": len(recent_metrics),
            "avg_auc": float(np.mean([m.auc for m in recent_metrics])),
            "avg_sharpe": float(np.mean([m.sharpe_ratio for m in recent_metrics])),
            "avg_pnl": float(np.mean([m.pnl for m in recent_metrics])),
            "avg_win_rate": float(np.mean([m.win_rate for m in recent_metrics])),
            "avg_calibration_error": float(
                np.mean([m.calibration_error for m in recent_metrics])
            ),
        }

    def _generate_drift_summary(self) -> Dict[str, Any]:
        """Generate drift summary."""
        if not self.drift_history:
            return {"status": "No drift data available"}

        recent_drift = self.drift_history[-5:]

        return {
            "total_records": len(self.drift_history),
            "recent_records": len(recent_drift),
            "avg_overall_drift": float(
                np.mean([d.overall_drift_score for d in recent_drift])
            ),
            "drift_detected_count": sum(1 for d in recent_drift if d.is_drift_detected),
            "most_drifting_features": self._get_most_drifting_features(),
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
        sorted_features = sorted(
            avg_drift_scores.items(), key=lambda x: x[1], reverse=True
        )

        return [
            {"feature": feature, "avg_drift_score": float(score)}
            for feature, score in sorted_features[:10]  # Top 10
        ]

    def _generate_alerts_summary(self) -> Dict[str, Any]:
        """Generate alerts summary."""
        if not self.alerts:
            return {"status": "No alerts generated"}

        recent_alerts = [
            a
            for a in self.alerts
            if a["timestamp"] > datetime.now() - timedelta(days=7)
        ]  # Last 7 days

        alert_types = {}
        for alert in recent_alerts:
            alert_type = alert["type"]
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

        return {
            "total_alerts": len(self.alerts),
            "recent_alerts": len(recent_alerts),
            "alert_types": alert_types,
            "latest_alert": self.alerts[-1] if self.alerts else None,
        }


class AutoRecalibrator:
    """
    Automatic model recalibration system.

    Monitors model performance and triggers retraining when needed.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the auto-recalibrator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.monitor = ModelMonitor(config)
        self.retraining_active = False
        self.last_retraining = None
        self.min_retraining_interval_hours = config.get(
            "min_retraining_interval_hours", 24
        )

        # Setup alert callbacks
        self.monitor.add_alert_callback(self._handle_alert)

        logger.info("Auto-recalibrator initialized")

    def _handle_alert(self, alert: Dict[str, Any]):
        """Handle alerts from the monitor."""
        alert_type = alert["type"]

        if alert_type == "RETRAINING_REQUIRED":
            self._trigger_retraining(alert)
        elif alert_type in ["PERFORMANCE_DEGRADED", "DRIFT_DETECTED"]:
            logger.warning(f"Alert received: {alert_type} - {alert['message']}")

    def _trigger_retraining(self, alert: Dict[str, Any]):
        """Trigger model retraining."""
        # Check if enough time has passed since last retraining
        if self.last_retraining:
            time_since_last = datetime.now() - self.last_retraining
            if (
                time_since_last.total_seconds()
                < self.min_retraining_interval_hours * 3600
            ):
                logger.info("Retraining skipped - too soon since last retraining")
                return

        if self.retraining_active:
            logger.info("Retraining already in progress")
            return

        logger.info("Triggering automatic model retraining")

        try:
            self.retraining_active = True

            # Run retraining in background
            retraining_thread = threading.Thread(
                target=self._run_retraining, daemon=True
            )
            retraining_thread.start()

        except Exception as e:
            logger.error(f"Error triggering retraining: {e}")
            self.retraining_active = False

    def _run_retraining(self):
        """Run the retraining process."""
        try:
            logger.info("Starting automatic model retraining")

            # Import here to avoid circular imports
            from ml.trainer import (
                create_binary_labels,
                generate_enhanced_features,
                load_data,
                train_model_binary,
            )

            # Load recent data for retraining
            data_path = self.config.get("retraining_data_path")
            if not data_path or not os.path.exists(data_path):
                logger.error("Retraining data path not found")
                return

            # Load and prepare data
            df = load_data(data_path)

            # Generate enhanced features
            df = generate_enhanced_features(
                df,
                include_multi_horizon=True,
                include_regime_features=True,
                include_interaction_features=True,
            )

            # Create labels
            df = create_binary_labels(df, horizon=5, profit_threshold=0.005)

            # Prepare for training
            feature_columns = [
                col
                for col in df.columns
                if col not in ["Open", "High", "Low", "Close", "Volume", "label_binary"]
            ]

            # Generate new model path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_model_path = f"models/binary_model_retrained_{timestamp}.pkl"

            # Train new model
            results = train_model_binary(
                df=df,
                save_path=new_model_path,
                results_path=f"models/training_results_retrained_{timestamp}.json",
                feature_columns=feature_columns,
                tune=True,  # Use hyperparameter tuning
                n_trials=15,  # Fewer trials for automatic retraining
                eval_economic=True,
            )

            # Update monitor with new model
            self.monitor.model_path = new_model_path
            self.monitor.config_path = new_model_path.replace(".pkl", "_config.json")
            self.monitor._load_model()

            self.last_retraining = datetime.now()
            logger.info("Automatic model retraining completed successfully")

        except Exception as e:
            logger.error(f"Error during automatic retraining: {e}")
        finally:
            self.retraining_active = False

    def start(self):
        """Start the auto-recalibration system."""
        self.monitor.start_monitoring()
        logger.info("Auto-recalibration system started")

    def stop(self):
        """Stop the auto-recalibration system."""
        self.monitor.stop_monitoring()
        logger.info("Auto-recalibration system stopped")


class RealTimeModelMonitor(ModelMonitor):
    """
    Real-time streaming model monitoring system.

    Extends ModelMonitor with streaming capabilities for immediate drift detection
    and real-time performance tracking.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the real-time model monitor.

        Args:
            config: Configuration dictionary with additional streaming parameters:
                - max_buffer_size: Maximum predictions to keep in memory
                - streaming_update_interval: Seconds between metric updates
                - drift_check_interval: Predictions between drift checks
                - dashboard_update_interval: Seconds between dashboard updates
        """
        super().__init__(config)

        # Streaming configuration
        self.max_buffer_size = config.get("max_buffer_size", 10000)
        self.streaming_update_interval = config.get("streaming_update_interval", 1.0)
        self.drift_check_interval = config.get("drift_check_interval", 100)
        self.dashboard_update_interval = config.get("dashboard_update_interval", 5.0)

        # Streaming data structures
        self.prediction_buffer: List[Dict[str, Any]] = []
        self.metrics_cache: Dict[str, Any] = {}
        self.last_metrics_update = time.time()
        self.last_drift_check = 0
        self.last_dashboard_update = time.time()

        # Callbacks
        self.dashboard_callbacks: List[Callable] = []
        self.retraining_callbacks: List[Callable] = []

        # Streaming state
        self.streaming_active = False
        self.streaming_thread = None

        # Drift detection algorithms
        self.drift_algorithms = config.get("drift_algorithms", ["ks_test", "psi"])

        # Performance tracking
        self.prediction_count = 0
        self.consecutive_drift_checks = 0
        self.performance_window = config.get("performance_window", 1000)
        self.min_samples_for_drift = config.get("min_samples_for_drift", 100)
        self.min_samples_for_metrics = config.get("min_samples_for_metrics", 50)

        logger.info("Real-time model monitor initialized")

    def record_prediction(
        self,
        features: List[float],
        prediction: float,
        actual: float,
        timestamp: Optional[datetime] = None,
    ):
        """
        Record a single prediction in real-time with automatic drift checking.

        Args:
            features: Feature values
            prediction: Model prediction (probability)
            actual: Actual outcome
            timestamp: Prediction timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Create prediction record
        record = {
            "timestamp": timestamp,
            "features": features.copy() if features else [],
            "prediction": float(prediction),
            "actual": float(actual) if actual is not None else None,
            "prediction_id": self.prediction_count,
        }

        # Add to buffer
        self.prediction_buffer.append(record)
        self.prediction_count += 1

        # Maintain buffer size
        if len(self.prediction_buffer) > self.max_buffer_size:
            self.prediction_buffer.pop(0)

        # Auto-update reference data when we have enough samples
        if (
            len(self.prediction_buffer) >= self.min_samples_for_drift
            and self.reference_features is None
        ):
            self._update_reference_data()

        # Check for drift periodically
        if (
            self.prediction_count % self.drift_check_interval == 0
            and self.reference_features is not None
        ):
            if self.check_drift():
                self._trigger_drift_alert()

        # Check for performance degradation
        if (
            len(self.prediction_buffer) >= self.min_samples_for_metrics
            and self.retraining_callbacks is not None
        ):
            recent_accuracy = self._calculate_recent_accuracy()
            if recent_accuracy < 0.6:  # Threshold for retraining
                self._trigger_retraining()

        # Periodic updates
        current_time = time.time()

        # Update metrics periodically
        if current_time - self.last_metrics_update >= self.streaming_update_interval:
            self._update_streaming_metrics()
            self.last_metrics_update = current_time

        # Update dashboard periodically
        if current_time - self.last_dashboard_update >= self.dashboard_update_interval:
            self._update_dashboard()
            self.last_dashboard_update = current_time

    def get_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive streaming metrics.

        Returns:
            Dictionary with current performance metrics
        """
        if len(self.prediction_buffer) < self.min_samples_for_metrics:
            return {}

        predictions = [p["prediction"] for p in self.prediction_buffer]
        actuals = [p["actual"] for p in self.prediction_buffer]

        try:
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            # Binary classification metrics - binarize both predictions and actuals
            pred_binary = [1 if p > 0.5 else 0 for p in predictions]
            actual_binary = [1 if a > 0.5 else 0 for a in actuals]

            metrics = {
                "total_predictions": len(self.prediction_buffer),
                "accuracy": accuracy_score(actual_binary, pred_binary),
                "precision": precision_score(
                    actual_binary, pred_binary, zero_division=0
                ),
                "recall": recall_score(actual_binary, pred_binary, zero_division=0),
                "timestamp": datetime.now(),
            }

            # AUC requires both classes to be present
            if len(set(actual_binary)) == 2:
                metrics["auc"] = roc_auc_score(actual_binary, predictions)

            return metrics
        except Exception as e:
            logger.warning(f"Metrics calculation failed: {e}")
            return {}

    def _calculate_streaming_metrics(self) -> Dict[str, Any]:
        """Calculate metrics from current prediction buffer."""
        if not self.prediction_buffer:
            return {
                "total_predictions": 0,
                "auc": 0.5,
                "drift_score": 0.0,
                "timestamp": datetime.now(),
            }

        # Get recent predictions (last performance_window)
        recent_predictions = self.prediction_buffer[-self.performance_window :]

        predictions = [r["prediction"] for r in recent_predictions]
        actuals = [r["actual"] for r in recent_predictions if r["actual"] is not None]

        metrics = {
            "total_predictions": len(self.prediction_buffer),
            "recent_predictions": len(recent_predictions),
            "timestamp": datetime.now(),
        }

        # Calculate AUC if we have actuals
        if len(actuals) >= 2:  # Need at least 2 samples for AUC
            try:
                # Convert actuals to binary if they're continuous
                binary_actuals = [1 if a >= 0.5 else 0 for a in actuals]
                auc = roc_auc_score(binary_actuals, predictions[: len(actuals)])
                metrics["auc"] = float(auc)
            except Exception:
                # Fallback: try with original actuals in case they're already binary
                try:
                    auc = roc_auc_score(actuals, predictions[: len(actuals)])
                    metrics["auc"] = float(auc)
                except Exception as e2:
                    logger.warning(f"Error calculating AUC: {e2}")
                    metrics["auc"] = 0.5  # Default neutral score
        else:
            metrics["auc"] = 0.5

        # Calculate drift score
        drift_score = self._calculate_current_drift_score()
        metrics["drift_score"] = drift_score

        # Additional metrics
        metrics["avg_prediction"] = float(np.mean(predictions))
        metrics["prediction_std"] = float(np.std(predictions))

        if actuals:
            metrics["avg_actual"] = float(np.mean(actuals))
            metrics["actual_std"] = float(np.std(actuals))

        # Cache metrics
        self.metrics_cache = metrics.copy()

        return metrics

    def _update_streaming_metrics(self):
        """Update streaming metrics cache."""
        self.metrics_cache = self._calculate_streaming_metrics()

    def check_drift(self) -> bool:
        """
        Check for drift in real-time.

        Returns:
            True if drift detected
        """
        print(
            f"DEBUG: check_drift called, ref_features={self.reference_features is not None if self.reference_features is not None else 'None'}, buffer_size={len(self.prediction_buffer)}"
        )
        if (
            self.reference_features is None
            or self.reference_features.empty
            or len(self.prediction_buffer) < 50
        ):
            print("DEBUG: Cannot check drift: ref_features empty or buffer too small")
            return False

        # Use multiple algorithms
        drift_results = self.detect_drift_multiple_algorithms()

        # Check if any algorithm detects drift
        detected = any(
            result.get("detected", False) for result in drift_results.values()
        )
        logger.debug(
            f"DEBUG: Drift check: detected={detected}, results={drift_results}"
        )
        return detected

    def detect_drift_multiple_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect drift using multiple algorithms.

        Returns:
            Dictionary with results from each algorithm
        """
        results = {}

        if (
            self.reference_features is None
            or self.reference_features.empty
            or not self.prediction_buffer
        ):
            return results

        # Get current data
        current_data = [r["features"] for r in self.prediction_buffer[-200:]]
        if current_data:
            column_names = [f"feature{i}" for i in range(len(current_data[0]))]
            current_features = pd.DataFrame(current_data, columns=column_names)
        else:
            current_features = pd.DataFrame()

        for algorithm in self.drift_algorithms:
            try:
                if algorithm == "ks_test":
                    results["ks_test"] = self._detect_drift_ks_test(current_features)
                elif algorithm == "psi":
                    results["psi"] = self._detect_drift_psi(current_features)
                elif algorithm == "mmd":
                    results["mmd"] = self._detect_drift_mmd(current_features)
            except Exception as e:
                logger.warning(f"Error in drift detection algorithm {algorithm}: {e}")
                results[algorithm] = {"detected": False, "score": 0.0, "error": str(e)}

        return results

    def _detect_drift_ks_test(self, current_features: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift using Kolmogorov-Smirnov test."""
        drift_scores = []

        # Use feature_columns if available, otherwise use all common columns
        columns_to_check = (
            self.feature_columns
            if self.feature_columns
            else list(
                set(current_features.columns) & set(self.reference_features.columns)
            )
        )

        for col in columns_to_check:
            if (
                col in current_features.columns
                and col in self.reference_features.columns
            ):
                try:
                    from scipy.stats import ks_2samp

                    stat, _ = ks_2samp(
                        self.reference_features[col].values,
                        current_features[col].values,
                    )
                    drift_scores.append(stat)
                except:
                    pass

        avg_drift = np.mean(drift_scores) if drift_scores else 0.0
        threshold = self.config.get("drift_thresholds", {}).get(
            "ks_threshold", 0.05
        )  # Lower threshold

        return {
            "detected": avg_drift > threshold,
            "score": float(avg_drift),
            "algorithm": "ks_test",
        }

    def _detect_drift_psi(self, current_features: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift using Population Stability Index."""
        drift_scores = []

        # Use feature_columns if available, otherwise use all common columns
        columns_to_check = (
            self.feature_columns
            if self.feature_columns
            else list(
                set(current_features.columns) & set(self.reference_features.columns)
            )
        )

        for col in columns_to_check:
            if (
                col in current_features.columns
                and col in self.reference_features.columns
            ):
                try:
                    # Create histograms
                    ref_hist, _ = np.histogram(
                        self.reference_features[col].values, bins=10, density=True
                    )
                    curr_hist, _ = np.histogram(
                        current_features[col].values, bins=10, density=True
                    )

                    # Avoid division by zero
                    ref_hist = ref_hist + 1e-10
                    curr_hist = curr_hist + 1e-10

                    # PSI calculation
                    psi = np.sum((curr_hist - ref_hist) * np.log(curr_hist / ref_hist))
                    drift_scores.append(psi)
                except:
                    pass

        avg_drift = np.mean(drift_scores) if drift_scores else 0.0
        threshold = self.config.get("drift_thresholds", {}).get(
            "psi_threshold", 0.05
        )  # Lower threshold

        return {
            "detected": avg_drift > threshold,
            "score": float(avg_drift),
            "algorithm": "psi",
        }

    def _detect_drift_mmd(self, current_features: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift using Maximum Mean Discrepancy (simplified)."""
        # Simplified MMD implementation
        drift_scores = []

        # Use feature_columns if available, otherwise use all common columns
        columns_to_check = (
            self.feature_columns
            if self.feature_columns
            else list(
                set(current_features.columns) & set(self.reference_features.columns)
            )
        )

        for col in columns_to_check:
            if (
                col in current_features.columns
                and col in self.reference_features.columns
            ):
                try:
                    ref_mean = np.mean(self.reference_features[col].values)
                    curr_mean = np.mean(current_features[col].values)
                    ref_std = np.std(self.reference_features[col].values)
                    curr_std = np.std(current_features[col].values)

                    # Simple distance-based drift score
                    mean_diff = abs(ref_mean - curr_mean)
                    std_diff = abs(ref_std - curr_std)

                    drift_score = (mean_diff / (abs(ref_mean) + 1e-10)) + (
                        std_diff / (ref_std + 1e-10)
                    )
                    drift_scores.append(drift_score)
                except:
                    pass

        avg_drift = np.mean(drift_scores) if drift_scores else 0.0
        threshold = self.config.get("drift_thresholds", {}).get("mmd_threshold", 0.2)

        return {
            "detected": avg_drift > threshold,
            "score": float(avg_drift),
            "algorithm": "mmd",
        }

    def analyze_drift_types(self) -> Dict[str, Any]:
        """
        Analyze whether drift is concept drift or data drift.

        Returns:
            Dictionary with drift type analysis
        """
        if not self.prediction_buffer or (
            self.reference_features is None or self.reference_features.empty
        ):
            return {"concept_drift": False, "data_drift": False}

        # Get recent predictions
        recent_records = self.prediction_buffer[-200:]
        current_features = pd.DataFrame([r["features"] for r in recent_records])
        current_predictions = np.array([r["prediction"] for r in recent_records])

        # Data drift: Check feature distribution changes
        data_drift_detected = False
        feature_drift_scores = {}

        for col in self.feature_columns:
            if (
                col in current_features.columns
                and col in self.reference_features.columns
            ):
                try:
                    from scipy.stats import ks_2samp

                    stat, _ = ks_2samp(
                        self.reference_features[col].values,
                        current_features[col].values,
                    )
                    feature_drift_scores[col] = stat
                    if stat > 0.1:  # Threshold for data drift
                        data_drift_detected = True
                except:
                    pass

        # Concept drift: Check if prediction distribution changed while features didn't
        concept_drift_detected = False

        if self.reference_predictions is not None and len(current_predictions) > 10:
            try:
                # Compare prediction distributions
                pred_drift = self._calculate_distribution_drift(
                    self.reference_predictions, current_predictions
                )

                # If predictions drifted but features didn't significantly, it's concept drift
                avg_feature_drift = (
                    np.mean(list(feature_drift_scores.values()))
                    if feature_drift_scores
                    else 0.0
                )

                if pred_drift > 0.15 and avg_feature_drift < 0.05:
                    concept_drift_detected = True
            except:
                pass

        return {
            "concept_drift": concept_drift_detected,
            "data_drift": data_drift_detected,
            "feature_drift_scores": feature_drift_scores,
            "prediction_drift_score": pred_drift if "pred_drift" in locals() else 0.0,
        }

    def _calculate_current_drift_score(self) -> float:
        """Calculate current drift score from buffer."""
        if (
            self.reference_features is None
            or self.reference_features.empty
            or not self.prediction_buffer
        ):
            return 0.0

        try:
            current_features = pd.DataFrame(
                [r["features"] for r in self.prediction_buffer[-100:]]
            )
            drift_results = self.detect_drift_multiple_algorithms()
            scores = [r["score"] for r in drift_results.values() if "score" in r]
            return float(np.mean(scores)) if scores else 0.0
        except:
            return 0.0

    def _check_streaming_drift(self):
        """Check for drift in streaming context."""
        if self.check_drift():
            self.consecutive_drift_checks += 1

            # Trigger alert if drift persists
            if self.consecutive_drift_checks >= 3:
                self._trigger_alert(
                    "DRIFT_DETECTED",
                    f"Persistent drift detected over {self.consecutive_drift_checks} checks",
                )
                self.consecutive_drift_checks = 0
        else:
            self.consecutive_drift_checks = 0

        # Also check for performance degradation
        self._check_performance_degradation()

    def _check_performance_degradation(self):
        """Check for significant performance degradation."""
        if len(self.prediction_buffer) < 50:  # Need minimum data
            return

        # Get recent metrics
        metrics = self.get_metrics()
        current_auc = metrics.get("auc", 0.5)

        # Simple threshold-based retraining trigger
        # If AUC drops below 0.3 (significant degradation), trigger retraining
        if current_auc < 0.3 and self.retraining_callbacks:
            logger.warning(
                f"Performance degraded significantly (AUC: {current_auc:.3f}), triggering retraining"
            )

            for callback in self.retraining_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in retraining callback: {e}")

            # Trigger alert as well
            self._trigger_alert(
                "PERFORMANCE_DEGRADED",
                f"Model performance degraded significantly (AUC: {current_auc:.3f})",
            )

    def set_dashboard_callback(self, callback: Callable):
        """Set callback for dashboard updates."""
        self.dashboard_callbacks.append(callback)

    def set_retraining_callback(self, callback: Callable):
        """Set callback for retraining triggers."""
        self.retraining_callbacks.append(callback)

    def _update_dashboard(self):
        """Update dashboard with current metrics."""
        metrics = self.get_metrics()

        # Add drift score if available
        if metrics:
            drift_score = self._calculate_current_drift_score()
            metrics["drift_score"] = drift_score

        for callback in self.dashboard_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Error in dashboard callback: {e}")

    def start_streaming(self):
        """Start streaming monitoring."""
        if self.streaming_active:
            logger.warning("Streaming is already active")
            return

        self.streaming_active = True
        self.streaming_thread = threading.Thread(
            target=self._streaming_worker, daemon=True
        )
        self.streaming_thread.start()
        logger.info("Real-time streaming monitoring started")

    def stop_streaming(self):
        """Stop streaming monitoring."""
        self.streaming_active = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=5)
        logger.info("Real-time streaming monitoring stopped")

    def _streaming_worker(self):
        """Background streaming worker."""
        while self.streaming_active:
            try:
                # Periodic maintenance
                self._cleanup_old_data()
                time.sleep(10)  # Maintenance every 10 seconds
            except Exception as e:
                logger.error(f"Error in streaming worker: {e}")
                time.sleep(30)

    def _cleanup_old_data(self):
        """Clean up old data to maintain memory efficiency."""
        if len(self.prediction_buffer) > self.max_buffer_size:
            # Keep only recent data
            keep_count = self.max_buffer_size // 2
            self.prediction_buffer = self.prediction_buffer[-keep_count:]

        # Clean up old alerts
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [
            alert for alert in self.alerts if alert["timestamp"] > cutoff_time
        ]


# Convenience functions
def create_model_monitor(config: Dict[str, Any]) -> ModelMonitor:
    """Create a model monitor instance."""
    return ModelMonitor(config)


def create_realtime_model_monitor(config: Dict[str, Any]) -> RealTimeModelMonitor:
    """Create a real-time model monitor instance."""
    return RealTimeModelMonitor(config)


def create_auto_recalibrator(config: Dict[str, Any]) -> AutoRecalibrator:
    """Create an auto-recalibrator instance."""
    return AutoRecalibrator(config)


def generate_monitoring_report(
    monitor: ModelMonitor, output_path: str = None
) -> Dict[str, Any]:
    """Generate and optionally save a monitoring report."""
    report = monitor.generate_report()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Monitoring report saved to {output_path}")

    return report
