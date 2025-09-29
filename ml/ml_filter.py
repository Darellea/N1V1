"""
ML Filter Module

This module provides a wrapper around ML models for signal filtering in trading systems.
It supports multiple ML algorithms and provides a unified interface for training,
prediction, and confidence scoring.

Supported models:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Optional imports for different ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning(
        "scikit-learn not available. Basic ML functionality will be limited."
    )

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available.")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available.")

logger = logging.getLogger(__name__)

# Centralized configuration for ML model parameters
# This eliminates hard-coded values and allows easy configuration changes
ML_MODEL_CONFIG = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": "balanced",
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "xgboost": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "scale_pos_weight": 1,
    },
    "lightgbm": {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42,
        "is_unbalance": True,
    },
}


class MLModel(ABC):
    """Abstract base class for ML models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        # Don't reset is_trained if it's already set (during deserialization)
        if not hasattr(self, "is_trained"):
            self.is_trained = False
        self.feature_names = []
        self.metadata = {}

    def __reduce__(self):
        """Custom pickle reduction to preserve training state."""
        return (self.__class__, (), self.__getstate__())

    def __getstate__(self):
        """Custom serialization to preserve training state."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Custom deserialization to restore training state."""
        self.__dict__.update(state)
        # Ensure is_trained is preserved during deserialization
        if "is_trained" in state:
            self.is_trained = state["is_trained"]

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """Get model type identifier."""
        pass

    def save(self, path: str) -> None:
        """Save model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "model": self.model,
            "config": self.config,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "model_type": self.get_model_type(),
            "trained_at": datetime.now().isoformat(),
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        model_data = joblib.load(path)
        self.model = model_data["model"]
        self.config = model_data.get("config", {})
        self.feature_names = model_data.get("feature_names", [])
        self.metadata = model_data.get("metadata", {})
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


class LogisticRegressionModel(MLModel):
    """Logistic Regression model wrapper."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for LogisticRegressionModel")

        # Use centralized config as defaults, overridden by passed config
        # This centralizes configuration and eliminates hard-coded values
        self.config = {**ML_MODEL_CONFIG["logistic_regression"], **(config or {})}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the logistic regression model."""
        self.feature_names = list(X.columns)
        self.model = LogisticRegression(**self.config)
        self.model.fit(X.values, y.values)
        self.is_trained = True

        # Store training metadata
        self.metadata = {
            "n_features": len(self.feature_names),
            "n_samples": len(X),
            "classes": list(np.unique(y)),
        }

        logger.info(f"LogisticRegression model trained on {len(X)} samples")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with logistic regression."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Feature alignment check: Ensure all required features are present in input DataFrame
        # This prevents KeyError during prediction and ensures model robustness in production
        if not all(col in X.columns for col in self.feature_names):
            missing_features = [
                col for col in self.feature_names if col not in X.columns
            ]
            raise ValueError(
                f"Missing required features for prediction: {missing_features}. "
                f"Input DataFrame must contain all expected features: {self.feature_names}"
            )

        X_aligned = X[self.feature_names]

        # Convert to numpy array for sklearn models
        X_array = X_aligned.values

        predictions = self.model.predict(X_array)
        probabilities = self.model.predict_proba(X_array)

        # Get confidence scores (probability of predicted class)
        confidence_scores = np.max(probabilities, axis=1)

        return predictions, confidence_scores

    def get_model_type(self) -> str:
        return "logistic_regression"


class RandomForestModel(MLModel):
    """Random Forest model wrapper."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RandomForestModel")

        # Use centralized config as defaults, overridden by passed config
        # This centralizes configuration and eliminates hard-coded values
        self.config = {**ML_MODEL_CONFIG["random_forest"], **(config or {})}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the random forest model."""
        self.feature_names = list(X.columns)
        self.model = RandomForestClassifier(**self.config)
        self.model.fit(X.values, y.values)
        self.is_trained = True

        # Store training metadata
        self.metadata = {
            "n_features": len(self.feature_names),
            "n_samples": len(X),
            "classes": list(np.unique(y)),
            "n_estimators": self.config["n_estimators"],
        }

        logger.info(f"RandomForest model trained on {len(X)} samples")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with random forest."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Feature alignment check: Ensure all required features are present in input DataFrame
        # This prevents KeyError during prediction and ensures model robustness in production
        if not all(col in X.columns for col in self.feature_names):
            missing_features = [
                col for col in self.feature_names if col not in X.columns
            ]
            raise ValueError(
                f"Missing required features for prediction: {missing_features}. "
                f"Input DataFrame must contain all expected features: {self.feature_names}"
            )

        X_aligned = X[self.feature_names]

        predictions = self.model.predict(X_aligned.values)
        probabilities = self.model.predict_proba(X_aligned.values)

        # Get confidence scores (probability of predicted class)
        confidence_scores = np.max(probabilities, axis=1)

        return predictions, confidence_scores

    def get_model_type(self) -> str:
        return "random_forest"

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))


class XGBoostModel(MLModel):
    """XGBoost model wrapper."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required for XGBoostModel")

        # Use centralized config as defaults, overridden by passed config
        # This centralizes configuration and eliminates hard-coded values
        self.config = {**ML_MODEL_CONFIG["xgboost"], **(config or {})}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the XGBoost model."""
        self.feature_names = list(X.columns)
        self.model = xgb.XGBClassifier(**self.config)
        self.model.fit(X.values, y.values)
        self.is_trained = True

        # Store training metadata
        self.metadata = {
            "n_features": len(self.feature_names),
            "n_samples": len(X),
            "classes": list(np.unique(y)),
        }

        logger.info(f"XGBoost model trained on {len(X)} samples")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with XGBoost."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Feature alignment check: Ensure all required features are present in input DataFrame
        # This prevents KeyError during prediction and ensures model robustness in production
        if not all(col in X.columns for col in self.feature_names):
            missing_features = [
                col for col in self.feature_names if col not in X.columns
            ]
            raise ValueError(
                f"Missing required features for prediction: {missing_features}. "
                f"Input DataFrame must contain all expected features: {self.feature_names}"
            )

        X_aligned = X[self.feature_names]

        predictions = self.model.predict(X_aligned.values)
        probabilities = self.model.predict_proba(X_aligned.values)

        # Get confidence scores (probability of predicted class)
        confidence_scores = np.max(probabilities, axis=1)

        return predictions, confidence_scores

    def get_model_type(self) -> str:
        return "xgboost"

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))


class LightGBMModel(MLModel):
    """LightGBM model wrapper."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for LightGBMModel")

        # Use centralized config as defaults, overridden by passed config
        # This centralizes configuration and eliminates hard-coded values
        self.config = {**ML_MODEL_CONFIG["lightgbm"], **(config or {})}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the LightGBM model."""
        self.feature_names = list(X.columns)
        self.model = lgb.LGBMClassifier(**self.config)
        self.model.fit(X.values, y.values)
        self.is_trained = True

        # Store training metadata
        self.metadata = {
            "n_features": len(self.feature_names),
            "n_samples": len(X),
            "classes": list(np.unique(y)),
        }

        logger.info(f"LightGBM model trained on {len(X)} samples")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with LightGBM."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Feature alignment check: Ensure all required features are present in input DataFrame
        # This prevents KeyError during prediction and ensures model robustness in production
        if not all(col in X.columns for col in self.feature_names):
            missing_features = [
                col for col in self.feature_names if col not in X.columns
            ]
            raise ValueError(
                f"Missing required features for prediction: {missing_features}. "
                f"Input DataFrame must contain all expected features: {self.feature_names}"
            )

        X_aligned = X[self.feature_names]

        predictions = self.model.predict(X_aligned.values)
        probabilities = self.model.predict_proba(X_aligned.values)

        # Get confidence scores (probability of predicted class)
        confidence_scores = np.max(probabilities, axis=1)

        return predictions, confidence_scores

    def get_model_type(self) -> str:
        return "lightgbm"

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))


class MLFilter:
    """
    ML Filter wrapper for trading signal filtering.

    Provides a unified interface for different ML models and handles
    the complete pipeline from feature processing to signal filtering.
    Uses dependency injection for model loading functions to reduce coupling.
    """

    def __init__(
        self,
        model_type: str = "logistic_regression",
        config: Optional[Dict[str, Any]] = None,
        load_model_func: Optional[Callable] = None,
        predict_func: Optional[Callable] = None,
    ):
        """
        Initialize ML Filter.

        Args:
            model_type: Type of ML model ('logistic_regression', 'random_forest', 'xgboost', 'lightgbm')
            config: Model configuration
            load_model_func: Function to load models (dependency injection)
            predict_func: Function to make predictions (dependency injection)
        """
        self.model_type = model_type
        self.config = config or {}
        self.model = self._create_model()
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        self.feature_scaler = None

        # Dependency injection for model loading functions to reduce coupling
        # If not provided, import default implementations (for backward compatibility)
        if load_model_func is None:
            from ml.model_loader import load_model as load_model_from_file

            self._load_model_func = load_model_from_file
        else:
            self._load_model_func = load_model_func

        if predict_func is None:
            from ml.model_loader import predict as ml_predict

            self._predict_func = ml_predict
        else:
            self._predict_func = predict_func

    def _create_model(self) -> MLModel:
        """Create ML model instance based on type."""
        model_configs = {
            "logistic_regression": LogisticRegressionModel,
            "random_forest": RandomForestModel,
            "xgboost": XGBoostModel,
            "lightgbm": LightGBMModel,
        }

        if self.model_type not in model_configs:
            available = list(model_configs.keys())
            raise ValueError(
                f"Unknown model type '{self.model_type}'. Available: {available}"
            )

        model_class = model_configs[self.model_type]
        return model_class(self.config.get("model_config", {}))

    def fit(
        self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the ML model.

        Args:
            X: Feature DataFrame
            y: Target labels
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary with training metrics
        """
        if X.empty or y.empty:
            raise ValueError("Training data cannot be empty")

        # Split data for validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = (
                X,
                pd.Series(dtype=y.dtype),
                y,
                pd.Series(dtype=y.dtype),
            )

        # Train model
        self.model.fit(X_train, y_train)

        # Calculate validation metrics if validation data exists
        metrics = {}
        if not X_val.empty:
            val_predictions, val_confidence = self.model.predict(X_val)

            metrics = {
                "accuracy": accuracy_score(y_val, val_predictions),
                "precision": precision_score(
                    y_val, val_predictions, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_val, val_predictions, average="weighted", zero_division=0
                ),
                "f1": f1_score(
                    y_val, val_predictions, average="weighted", zero_division=0
                ),
                "mean_confidence": np.mean(val_confidence),
            }

            logger.info(f"Model validation metrics: {metrics}")

        return metrics

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (predictions, confidence_scores, decisions)
        """
        if not self.model.is_trained:
            raise ValueError("Model must be trained before prediction")

        predictions, confidence_scores = self.model.predict(X)

        # Make filtering decisions based on confidence threshold
        decisions = confidence_scores >= self.confidence_threshold

        return predictions, confidence_scores, decisions

    def filter_signal(self, features: pd.DataFrame, signal_type: str) -> Dict[str, Any]:
        """
        Filter a trading signal using the ML model.

        Args:
            features: Feature DataFrame for the current candle
            signal_type: Type of signal ('buy' or 'sell')

        Returns:
            Dictionary with filtering results
        """
        if features.empty:
            return {"approved": False, "confidence": 0.0, "reason": "no_features"}

        try:
            predictions, confidence_scores, decisions = self.predict(features)

            # Get the result for the latest data point
            latest_idx = -1
            prediction = predictions[latest_idx]
            confidence = confidence_scores[latest_idx]
            decision = decisions[latest_idx]

            # Convert prediction to signal direction
            # Assuming 1 = buy/up, 0/-1 = sell/down
            desired_direction = (
                1 if signal_type.lower() in ["buy", "entry_long"] else -1
            )
            predicted_direction = 1 if prediction == 1 else -1

            # Check if prediction matches desired signal direction
            direction_match = predicted_direction == desired_direction

            result = {
                "approved": decision and direction_match,
                "confidence": float(confidence),
                "prediction": int(prediction),
                "direction_match": direction_match,
                "threshold": self.confidence_threshold,
            }

            if not result["approved"]:
                if not decision:
                    result["reason"] = "low_confidence"
                elif not direction_match:
                    result["reason"] = "direction_mismatch"

            return result

        except Exception as e:
            logger.error(f"Error in signal filtering: {e}")
            return {"approved": False, "confidence": 0.0, "reason": "prediction_error"}

    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.model.is_trained:
            raise ValueError("Model must be trained before saving")

        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save model with training state
        model_data = {
            "model": self.model,
            "is_trained": getattr(self.model, "is_trained", False),
            "config": getattr(self, "config", {}),
        }
        joblib.dump(model_data, path)

        # Save filter metadata
        metadata = {
            "model_type": self.model_type,
            "config": self.config,
            "confidence_threshold": self.confidence_threshold,
            "created_at": datetime.now().isoformat(),
        }

        metadata_path = str(Path(path).with_suffix("")) + "_filter_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"ML Filter saved to {path}")

    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        # Load model data
        model_data = joblib.load(path)
        self.model = model_data["model"]
        # Restore training state explicitly - ensure it's marked as trained
        # The saved data should have is_trained=True, so restore it
        saved_is_trained = model_data.get("is_trained", False)
        if hasattr(self.model, "is_trained"):
            self.model.is_trained = saved_is_trained
        self.config = model_data.get("config", {})

        # Try to load filter metadata
        metadata_path = str(Path(path).with_suffix("")) + "_filter_metadata.json"
        if Path(metadata_path).exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.model_type = metadata.get("model_type", self.model_type)
                self.config = metadata.get("config", self.config)
                self.confidence_threshold = metadata.get(
                    "confidence_threshold", self.confidence_threshold
                )

        logger.info(
            f"ML Filter loaded from {path}, is_trained restored to: {saved_is_trained}"
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.model.is_trained:
            return {
                "status": "not_trained",
                "model_type": self.model_type,
                "confidence_threshold": self.confidence_threshold,
            }

        return {
            "model_type": self.model_type,
            "is_trained": self.model.is_trained,
            "feature_names": self.model.feature_names,
            "confidence_threshold": self.confidence_threshold,
            "metadata": self.model.metadata,
        }

    def update_confidence_threshold(self, threshold: float) -> None:
        """Update the confidence threshold for signal filtering."""
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")

        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold updated to {threshold}")


def create_ml_filter(
    model_type: str = "logistic_regression", config: Optional[Dict[str, Any]] = None
) -> MLFilter:
    """
    Create an ML filter with the specified model type.

    Args:
        model_type: Type of ML model
        config: Configuration dictionary

    Returns:
        Configured MLFilter instance
    """
    return MLFilter(model_type, config)


def load_ml_filter(path: str) -> MLFilter:
    """
    Load an ML filter from disk.

    Args:
        path: Path to the saved model file

    Returns:
        Loaded MLFilter instance
    """
    # Load metadata to determine model type
    metadata_path = str(Path(path).with_suffix("")) + "_filter_metadata.json"

    model_type = "logistic_regression"  # default
    config = {}

    if Path(metadata_path).exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            model_type = metadata.get("model_type", model_type)
            config = metadata.get("config", config)

    ml_filter = MLFilter(model_type, config)
    ml_filter.load_model(path)
    return ml_filter
