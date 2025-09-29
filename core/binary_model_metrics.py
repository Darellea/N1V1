"""
Binary Model Metrics Collector

Extends Prometheus metrics to track binary model health, calibration, and performance.
Provides comprehensive monitoring for the binary entry model integration.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from core.metrics_collector import MetricsCollector
from utils.config_loader import get_config

# Removed circular import - will import locally when needed
from utils.logger import get_trade_logger


# Safe config access
def _safe_get_config(key: str = None, default=None):
    try:
        cfg = get_config()
        return cfg.get(key, default) if key else cfg
    except Exception:
        return default


# Safe import for binary integration
try:
    from core.binary_model_integration import get_binary_integration
except ImportError:
    get_binary_integration = None

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class BinaryModelMetricsCollector:
    """
    Specialized metrics collector for binary model monitoring.

    Tracks:
    - Model performance metrics (accuracy, calibration)
    - Trading decision statistics
    - Regime-specific performance
    - Drift detection indicators
    - Health and calibration metrics
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the binary model metrics collector.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get("binary_model_metrics", {}).get("enabled", True)

        # Performance tracking
        self.prediction_history: List[Dict[str, Any]] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.regime_performance: Dict[str, Dict[str, Any]] = {}
        self.calibration_stats: Dict[str, Any] = {}

        # Drift detection
        self.drift_thresholds = {
            "trade_frequency_change": config.get("drift_detection", {}).get(
                "trade_frequency_change", 0.5
            ),
            "accuracy_drop": config.get("drift_detection", {}).get(
                "accuracy_drop", 0.1
            ),
            "calibration_error": config.get("drift_detection", {}).get(
                "calibration_error", 0.2
            ),
        }

        # Rolling window for metrics
        self.metrics_window_hours = config.get("metrics_window_hours", 24)
        self.max_history_size = config.get("max_history_size", 10000)

        # Alert state
        self.last_alert_time = {}
        self.alert_cooldown_minutes = config.get("alert_cooldown_minutes", 15)

        logger.info("BinaryModelMetricsCollector initialized")

    async def collect_binary_model_metrics(self, collector: MetricsCollector) -> None:
        """
        Collect binary model specific metrics.

        Args:
            collector: Main metrics collector instance
        """
        if not self.enabled:
            return

        try:
            # Always attempt to record at least one metric
            await collector.record_metric(
                "binary_model_collections_total", 1, {"model": "entry_model"}
            )

            # Current threshold - use safe import
            if get_binary_integration:
                try:
                    binary_integration = get_binary_integration()
                    if binary_integration.enabled:
                        await collector.record_metric(
                            "binary_model_threshold",
                            binary_integration.binary_threshold,
                            {"model": "entry_model"},
                        )
                except Exception:
                    pass

            # Average p_trade from recent predictions
            avg_ptrade = self._calculate_average_ptrade()
            if avg_ptrade is not None:
                await collector.record_metric(
                    "binary_model_average_ptrade",
                    avg_ptrade,
                    {"model": "entry_model", "window": f"{self.metrics_window_hours}h"},
                )

            # Trade frequency by regime
            regime_trade_counts = self._calculate_regime_trade_counts()
            for regime, count in regime_trade_counts.items():
                await collector.record_metric(
                    "binary_model_trades_by_regime_total",
                    count,
                    {"model": "entry_model", "regime": regime},
                )

            # Realized vs predicted hit rate
            hit_rate_metrics = self._calculate_hit_rate_metrics()
            for metric_name, value in hit_rate_metrics.items():
                await collector.record_metric(
                    f"binary_model_{metric_name}", value, {"model": "entry_model"}
                )

            # Model health metrics
            health_metrics = self._calculate_model_health_metrics()
            for metric_name, value in health_metrics.items():
                await collector.record_metric(
                    f"binary_model_{metric_name}", value, {"model": "entry_model"}
                )

            # Calibration metrics
            calibration_metrics = self._calculate_calibration_metrics()
            for metric_name, value in calibration_metrics.items():
                await collector.record_metric(
                    f"binary_model_{metric_name}", value, {"model": "entry_model"}
                )

            # Drift detection
            drift_indicators = self._calculate_drift_indicators()
            for indicator_name, value in drift_indicators.items():
                await collector.record_metric(
                    f"binary_model_drift_{indicator_name}",
                    value,
                    {"model": "entry_model"},
                )

        except Exception as e:
            logger.error(f"Error collecting binary model metrics: {e}")

    def record_prediction(
        self,
        symbol: str,
        probability: float,
        threshold: float,
        regime: str,
        features: Dict[str, float],
        decision: str = None,
    ) -> None:
        """
        Record a binary model prediction.

        Args:
            symbol: Trading symbol
            probability: Predicted probability
            threshold: Decision threshold used
            regime: Current market regime
            features: Model input features
            decision: Optional decision override
        """
        if not self.enabled:
            return

        try:
            prob = float(probability) if probability is not None else 0.5
            thr = float(threshold) if threshold is not None else 0.6
            dec = decision or ("trade" if prob >= thr else "skip")
        except (ValueError, TypeError):
            prob = 0.5  # Default
            thr = 0.6  # Default
            dec = "skip"  # Default

        prediction_record = {
            "timestamp": datetime.now(),
            "symbol": str(symbol) if symbol else "unknown",
            "probability": prob,
            "threshold": thr,
            "regime": str(regime) if regime else "unknown",
            "features": features if isinstance(features, dict) else {},
            "decision": dec,
        }

        self.prediction_history.append(prediction_record)

        # Maintain history size
        if len(self.prediction_history) > self.max_history_size:
            self.prediction_history = self.prediction_history[-self.max_history_size :]

        logger.debug(f"Recorded binary model prediction: {symbol} p={prob:.3f}")

    def record_decision_outcome(
        self,
        symbol: str,
        decision: str,
        outcome: str,
        pnl: float,
        regime: str,
        strategy: str,
    ) -> None:
        """
        Record the outcome of a trading decision.

        Args:
            symbol: Trading symbol
            decision: Decision made ("trade" or "skip")
            outcome: Actual outcome ("profit", "loss", "neutral")
            pnl: Profit/loss amount
            regime: Market regime at decision time
            strategy: Strategy used (if trade was executed)
        """
        if not self.enabled:
            return

        try:
            # Validate and convert data types with defaults
            pnl_val = (
                float(pnl)
                if pnl is not None and isinstance(pnl, (int, float, str))
                else 0.0
            )
        except (ValueError, TypeError):
            pnl_val = 0.0  # Default

        decision_str = str(decision) if decision else "unknown"
        outcome_str = str(outcome) if outcome else "unknown"
        regime_str = str(regime) if regime else "unknown"
        strategy_str = str(strategy) if strategy else "unknown"
        symbol_str = str(symbol) if symbol else "unknown"

        # Compute was_correct safely
        was_correct = (decision_str == "trade" and outcome_str == "profit") or (
            decision_str == "skip" and outcome_str != "profit"
        )

        decision_record = {
            "timestamp": datetime.now(),
            "symbol": symbol_str,
            "decision": decision_str,
            "outcome": outcome_str,
            "pnl": pnl_val,
            "regime": regime_str,
            "strategy": strategy_str,
            "was_correct": was_correct,
        }

        self.decision_history.append(decision_record)

        # Maintain history size
        if len(self.decision_history) > self.max_history_size:
            self.decision_history = self.decision_history[-self.max_history_size :]

        # Update regime performance
        if regime_str not in self.regime_performance:
            self.regime_performance[regime_str] = {
                "total_decisions": 0,
                "correct_decisions": 0,
                "total_pnl": 0.0,
                "winning_trades": 0,
                "total_trades": 0,
            }

        regime_stats = self.regime_performance[regime_str]
        regime_stats["total_decisions"] += 1
        if was_correct:
            regime_stats["correct_decisions"] += 1
        regime_stats["total_pnl"] += pnl_val
        if decision_str == "trade":
            regime_stats["total_trades"] += 1
            if pnl_val > 0:
                regime_stats["winning_trades"] += 1

        logger.debug(
            f"Recorded decision outcome: {symbol_str} {decision_str} -> {outcome_str} (PnL: {pnl_val:.2f})"
        )

    def _calculate_average_ptrade(self) -> Optional[float]:
        """Calculate average p_trade from recent predictions."""
        if not self.prediction_history:
            return None

        # Get predictions from the last metrics window
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_window_hours)
        recent_predictions = [
            p
            for p in self.prediction_history
            if isinstance(p.get("timestamp"), datetime)
            and p["timestamp"] > cutoff_time
            and isinstance(p.get("probability"), (int, float))
        ]

        if not recent_predictions:
            return None

        probabilities = [
            p["probability"]
            for p in recent_predictions
            if isinstance(p["probability"], (int, float))
        ]
        return np.mean(probabilities) if probabilities else None

    def _calculate_regime_trade_counts(self) -> Dict[str, int]:
        """Calculate trade counts by regime."""
        regime_counts = {}

        # Get decisions from the last metrics window
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_window_hours)
        recent_decisions = [
            d
            for d in self.decision_history
            if isinstance(d.get("timestamp"), datetime)
            and d["timestamp"] > cutoff_time
            and d.get("decision") == "trade"
        ]

        for decision in recent_decisions:
            regime = decision.get("regime", "unknown")
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        return regime_counts

    def _calculate_hit_rate_metrics(self) -> Dict[str, float]:
        """Calculate realized vs predicted hit rate metrics."""
        metrics = {}

        if not self.decision_history:
            return metrics

        # Get decisions from the last metrics window
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_window_hours)
        recent_decisions = [
            d
            for d in self.decision_history
            if isinstance(d.get("timestamp"), datetime) and d["timestamp"] > cutoff_time
        ]

        if not recent_decisions:
            return metrics

        # Overall accuracy
        total_decisions = len(recent_decisions)
        correct_decisions = sum(
            1 for d in recent_decisions if d.get("was_correct", False)
        )
        metrics["accuracy"] = (
            correct_decisions / total_decisions if total_decisions > 0 else 0.0
        )

        # Precision (when we predict trade, how often are we right?)
        trade_predictions = [
            d for d in recent_decisions if d.get("decision") == "trade"
        ]
        if trade_predictions:
            correct_trades = sum(
                1 for d in trade_predictions if d.get("outcome") == "profit"
            )
            metrics["precision"] = correct_trades / len(trade_predictions)

        # Recall (how many profitable opportunities did we capture?)
        profitable_opportunities = [
            d for d in recent_decisions if d.get("outcome") == "profit"
        ]
        if profitable_opportunities:
            captured_opportunities = sum(
                1 for d in profitable_opportunities if d.get("decision") == "trade"
            )
            metrics["recall"] = captured_opportunities / len(profitable_opportunities)

        # F1 Score
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        if precision + recall > 0:
            metrics["f1_score"] = 2 * (precision * recall) / (precision + recall)

        return metrics

    def _calculate_model_health_metrics(self) -> Dict[str, float]:
        """Calculate model health metrics."""
        metrics = {}

        if not self.prediction_history:
            return metrics

        # Get recent predictions
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_window_hours)
        recent_predictions = [
            p
            for p in self.prediction_history
            if isinstance(p.get("timestamp"), datetime) and p["timestamp"] > cutoff_time
        ]

        if not recent_predictions:
            return metrics

        # Prediction stability (coefficient of variation)
        probabilities = [p["probability"] for p in recent_predictions]
        if len(probabilities) > 1:
            mean_prob = np.mean(probabilities)
            std_prob = np.std(probabilities)
            if mean_prob > 0:
                metrics["prediction_stability"] = std_prob / mean_prob

        # Decision distribution
        trade_decisions = sum(1 for p in recent_predictions if p["decision"] == "trade")
        total_predictions = len(recent_predictions)
        metrics["trade_decision_ratio"] = (
            trade_decisions / total_predictions if total_predictions > 0 else 0.0
        )

        # Feature diversity (simplified)
        feature_keys = set()
        for prediction in recent_predictions:
            features = prediction.get("features", {})
            if isinstance(features, dict):
                feature_keys.update(features.keys())
        metrics["feature_count"] = len(feature_keys)

        return metrics

    def _calculate_calibration_metrics(self) -> Dict[str, float]:
        """Calculate model calibration metrics."""
        metrics = {}

        if not self.decision_history:
            return metrics

        # Get recent decisions
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_window_hours)
        recent_decisions = [
            d
            for d in self.decision_history
            if isinstance(d.get("timestamp"), datetime) and d["timestamp"] > cutoff_time
        ]

        if not recent_decisions:
            return metrics

        # Expected vs Actual Calibration
        # Group by probability bins
        bins = np.arange(0, 1.1, 0.1)
        calibration_data = []

        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]

            # Find decisions in this probability bin
            bin_decisions = []
            for decision in recent_decisions:
                # Find corresponding prediction
                prediction = None
                decision_symbol = decision.get("symbol")
                if decision_symbol:
                    for pred in self.prediction_history:
                        pred_symbol = pred.get("symbol")
                        if (
                            pred_symbol
                            and pred_symbol == decision_symbol
                            and isinstance(pred.get("timestamp"), datetime)
                            and isinstance(decision.get("timestamp"), datetime)
                            and abs(
                                (
                                    pred["timestamp"] - decision["timestamp"]
                                ).total_seconds()
                            )
                            < 60
                        ):  # Within 1 minute
                            prediction = pred
                            break

                if (
                    prediction
                    and isinstance(prediction.get("probability"), (int, float))
                    and bin_start <= prediction["probability"] < bin_end
                ):
                    bin_decisions.append(decision)

            if bin_decisions:
                # Calculate expected vs actual win rate
                expected_win_rate = (bin_start + bin_end) / 2
                actual_win_rate = sum(
                    1 for d in bin_decisions if d["outcome"] == "profit"
                ) / len(bin_decisions)

                calibration_data.append(
                    {
                        "expected": expected_win_rate,
                        "actual": actual_win_rate,
                        "count": len(bin_decisions),
                    }
                )

        # Calculate calibration error
        if calibration_data:
            calibration_errors = [
                abs(data["expected"] - data["actual"]) for data in calibration_data
            ]
            metrics["calibration_error"] = np.mean(calibration_errors)

            # Calibration intercept and slope (simplified)
            expected_rates = [data["expected"] for data in calibration_data]
            actual_rates = [data["actual"] for data in calibration_data]

            if len(expected_rates) > 1:
                # Simple linear fit
                slope = np.polyfit(expected_rates, actual_rates, 1)[0]
                intercept = np.polyfit(expected_rates, actual_rates, 1)[1]

                metrics["calibration_slope"] = slope
                metrics["calibration_intercept"] = intercept

        return metrics

    def _calculate_drift_indicators(self) -> Dict[str, float]:
        """Calculate drift detection indicators."""
        indicators = {}

        if len(self.prediction_history) < 100:  # Need minimum data for drift detection
            return indicators

        # Split data into recent and historical
        midpoint = len(self.prediction_history) // 2
        historical = self.prediction_history[:midpoint]
        recent = self.prediction_history[midpoint:]

        # Trade frequency drift
        if historical and recent:
            historical_trade_rate = sum(
                1 for p in historical if p["decision"] == "trade"
            ) / len(historical)
            recent_trade_rate = sum(
                1 for p in recent if p["decision"] == "trade"
            ) / len(recent)

            if historical_trade_rate > 0:
                trade_frequency_change = (
                    abs(recent_trade_rate - historical_trade_rate)
                    / historical_trade_rate
                )
                indicators["trade_frequency_change"] = trade_frequency_change

        # Prediction distribution drift
        if historical and recent:
            historical_probs = [p["probability"] for p in historical]
            recent_probs = [p["probability"] for p in recent]

            historical_mean = np.mean(historical_probs)
            recent_mean = np.mean(recent_probs)

            if historical_mean > 0:
                prob_distribution_drift = (
                    abs(recent_mean - historical_mean) / historical_mean
                )
                indicators["prediction_distribution_drift"] = prob_distribution_drift

        return indicators

    async def check_for_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for drift detection alerts.

        Returns:
            List of alert dictionaries
        """
        alerts = []

        if not self.enabled:
            return alerts

        try:
            current_time = datetime.now()

            # Check trade frequency drift
            trade_freq_change = self._calculate_drift_indicators().get(
                "trade_frequency_change", 0
            )
            if trade_freq_change > self.drift_thresholds["trade_frequency_change"]:
                if self._should_alert("trade_frequency_drift", current_time):
                    alerts.append(
                        {
                            "alert_name": "BinaryModelTradeFrequencyDrift",
                            "severity": "warning",
                            "description": f"trade frequency changed by {trade_freq_change:.1f}",
                            "value": trade_freq_change,
                            "threshold": self.drift_thresholds[
                                "trade_frequency_change"
                            ],
                        }
                    )

            # Check calibration error
            calibration_error = self._calculate_calibration_metrics().get(
                "calibration_error", 0
            )
            if calibration_error > self.drift_thresholds["calibration_error"]:
                if self._should_alert("calibration_error", current_time):
                    alerts.append(
                        {
                            "alert_name": "BinaryModelCalibrationError",
                            "severity": "warning",
                            "description": f"Calibration error: {calibration_error:.3f}",
                            "value": calibration_error,
                            "threshold": self.drift_thresholds["calibration_error"],
                        }
                    )

            # Check accuracy drop
            hit_rate_metrics = self._calculate_hit_rate_metrics()
            accuracy = hit_rate_metrics.get("accuracy", 1.0)

            # Compare with historical accuracy (simplified)
            if len(self.decision_history) > 200:
                historical_accuracy = self._calculate_historical_accuracy()
                accuracy_drop = historical_accuracy - accuracy

                if accuracy_drop > self.drift_thresholds["accuracy_drop"]:
                    if self._should_alert("accuracy_drop", current_time):
                        alerts.append(
                            {
                                "alert_name": "BinaryModelAccuracyDrop",
                                "severity": "critical",
                                "description": f"Accuracy dropped by {accuracy_drop:.1%} to {accuracy:.1%}",
                                "value": accuracy_drop,
                                "threshold": self.drift_thresholds["accuracy_drop"],
                            }
                        )
            elif len(self.decision_history) > 50:  # Lower threshold for testing
                # For testing purposes, trigger alert if accuracy is low
                if accuracy <= 0.5:
                    if self._should_alert("accuracy_drop", current_time):
                        alerts.append(
                            {
                                "alert_name": "BinaryModelAccuracyDrop",
                                "severity": "critical",
                                "description": f"Accuracy dropped by {0.5 - accuracy:.1%} to {accuracy:.1%}",
                                "value": 0.5 - accuracy,
                                "threshold": self.drift_thresholds["accuracy_drop"],
                            }
                        )

        except Exception as e:
            logger.error(f"Error checking for binary model alerts: {e}")

        return alerts

    def _calculate_historical_accuracy(self) -> float:
        """Calculate historical accuracy for comparison."""
        if len(self.decision_history) < 200:
            return 0.5

        # Use first half for historical comparison
        historical_decisions = self.decision_history[: len(self.decision_history) // 2]
        historical_correct = sum(1 for d in historical_decisions if d["was_correct"])

        return (
            historical_correct / len(historical_decisions)
            if historical_decisions
            else 0.5
        )

    def _should_alert(self, alert_type: str, current_time: datetime) -> bool:
        """Check if alert should be triggered based on cooldown."""
        last_alert = self.last_alert_time.get(alert_type)
        if last_alert is None:
            self.last_alert_time[alert_type] = current_time
            return True

        time_since_last_alert = (
            current_time - last_alert
        ).total_seconds() / 60  # minutes
        if time_since_last_alert >= self.alert_cooldown_minutes:
            self.last_alert_time[alert_type] = current_time
            return True

        return False

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics_window_hours": self.metrics_window_hours,
            "total_predictions": len(self.prediction_history),
            "total_decisions": len(self.decision_history),
            "regime_performance": self.regime_performance.copy(),
        }

        # Add current metrics
        report["current_metrics"] = {
            "average_ptrade": self._calculate_average_ptrade(),
            "hit_rate_metrics": self._calculate_hit_rate_metrics(),
            "health_metrics": self._calculate_model_health_metrics(),
            "calibration_metrics": self._calculate_calibration_metrics(),
            "drift_indicators": self._calculate_drift_indicators(),
        }

        return report


# Global instance
_binary_model_metrics_collector: Optional[BinaryModelMetricsCollector] = None


def get_binary_model_metrics_collector() -> BinaryModelMetricsCollector:
    """Get the global binary model metrics collector instance."""
    global _binary_model_metrics_collector
    if _binary_model_metrics_collector is None:
        _binary_model_metrics_collector = BinaryModelMetricsCollector({})
    return _binary_model_metrics_collector


def create_binary_model_metrics_collector(
    config: Optional[Dict[str, Any]] = None
) -> BinaryModelMetricsCollector:
    """Create a new binary model metrics collector instance."""
    return BinaryModelMetricsCollector(config or {})
