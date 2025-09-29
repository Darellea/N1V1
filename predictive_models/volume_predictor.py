"""
Volume Surge Predictor

Detects volume surges using statistical methods or ML classifiers.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class VolumePredictor:
    """
    Detects volume surges using statistical thresholds or ML models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the volume predictor.

        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.model_type = config.get("type", "zscore")
        self.threshold = config.get(
            "threshold", 2.5
        )  # Z-score threshold for statistical method
        self.lookback = config.get("lookback", 50)
        self.model_path = config.get(
            "model_path", f"models/volume_{self.model_type}.pkl"
        )
        self.scaler_path = config.get("scaler_path", "models/volume_scaler.pkl")

        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []

        # Create models directory if it doesn't exist
        Path("models").mkdir(exist_ok=True)

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for volume surge detection from OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features
        """
        if df.empty or "volume" not in df.columns:
            return pd.DataFrame()

        # Basic volume features
        df = df.copy()
        df["volume_ma_5"] = df["volume"].rolling(5).mean()
        df["volume_ma_10"] = df["volume"].rolling(10).mean()
        df["volume_ma_20"] = df["volume"].rolling(20).mean()
        df["volume_ma_50"] = df["volume"].rolling(50).mean()

        # Volume ratios
        df["volume_ratio_5"] = df["volume"] / df["volume_ma_5"]
        df["volume_ratio_10"] = df["volume"] / df["volume_ma_10"]
        df["volume_ratio_20"] = df["volume"] / df["volume_ma_20"]
        df["volume_ratio_50"] = df["volume"] / df["volume_ma_50"]

        # Volume changes
        df["volume_change"] = df["volume"].pct_change()
        df["volume_change_5"] = df["volume"].pct_change(5)
        df["volume_change_10"] = df["volume"].pct_change(10)

        # Volume volatility
        df["volume_volatility_10"] = df["volume_change"].rolling(10).std()
        df["volume_volatility_20"] = df["volume_change"].rolling(20).std()

        # Price-volume relationships
        df["returns"] = df["close"].pct_change()
        df["price_volume_corr_10"] = df["returns"].rolling(10).corr(df["volume"])
        df["price_volume_corr_20"] = df["returns"].rolling(20).corr(df["volume"])

        # Volume concentration (how much volume is in recent periods)
        df["volume_concentration_5"] = (
            df["volume"].rolling(5).sum() / df["volume"].rolling(20).sum()
        )
        df["volume_concentration_10"] = (
            df["volume"].rolling(10).sum() / df["volume"].rolling(40).sum()
        )

        # Statistical measures
        df["volume_zscore_20"] = (df["volume"] - df["volume_ma_20"]) / df[
            "volume"
        ].rolling(20).std()
        df["volume_zscore_50"] = (df["volume"] - df["volume_ma_50"]) / df[
            "volume"
        ].rolling(50).std()

        # Volume patterns
        df["volume_above_ma_20"] = (df["volume"] > df["volume_ma_20"]).astype(int)
        df["volume_above_ma_50"] = (df["volume"] > df["volume_ma_50"]).astype(int)

        # Price action during high volume
        df["high_volume_price_change"] = df["returns"].where(
            df["volume"] > df["volume_ma_20"], 0
        )
        df["high_volume_price_volatility"] = (
            df["returns"].rolling(5).std().where(df["volume"] > df["volume_ma_20"], 0)
        )

        # Drop NaN values
        df = df.dropna()

        # Select feature columns
        exclude_cols = ["open", "high", "low", "close", "volume", "timestamp"]
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]

        return df[self.feature_columns]

    def _create_labels(self, df: pd.DataFrame, threshold: float = 2.5) -> pd.Series:
        """
        Create labels for volume surge detection.

        Args:
            df: DataFrame with volume data
            threshold: Z-score threshold for surge detection

        Returns:
            Series with binary labels (1 for surge, 0 for normal)
        """
        # Calculate z-score of volume
        volume_ma = df["volume"].rolling(50).mean()
        volume_std = df["volume"].rolling(50).std()
        volume_zscore = (df["volume"] - volume_ma) / volume_std

        # Label surges
        labels = (volume_zscore > threshold).astype(int)

        return labels.dropna()

    def _get_ml_model(self):
        """Get the appropriate ML model based on configuration."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "xgboost":
            return xgb.XGBClassifier(
                objective="binary:logistic", random_state=42, n_estimators=100
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                objective="binary", random_state=42, n_estimators=100
            )
        else:
            raise ValueError(f"Unsupported ML model type: {self.model_type}")

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the volume surge detection model.

        Args:
            df: Training data with OHLCV columns

        Returns:
            Training metrics
        """
        if self.model_type == "zscore":
            # For statistical method, just save parameters
            metrics = {
                "model_type": "zscore",
                "threshold": self.threshold,
                "lookback": self.lookback,
            }
            logger.info(f"Z-score model configured with threshold {self.threshold}")
            return metrics

        logger.info(f"Training {self.model_type} model for volume surge detection")

        # Create features and labels
        features_df = self._create_features(df)
        labels = self._create_labels(df, self.threshold)

        # Align features and labels
        common_idx = features_df.index.intersection(labels.index)
        X = features_df.loc[common_idx]
        y = labels.loc[common_idx]

        if len(X) < 100:
            raise ValueError("Insufficient training data")

        # Split data (time series split)
        tscv = TimeSeriesSplit(n_splits=5)
        train_scores = []
        val_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train model
            model = self._get_ml_model()
            model.fit(X_train_scaled, y_train)

            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            val_score = model.score(X_val_scaled, y_val)

            train_scores.append(train_score)
            val_scores.append(val_score)

        # Train final model on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._get_ml_model()
        self.model.fit(X_scaled, y)

        # Save model and scaler
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        metrics = {
            "model_type": self.model_type,
            "train_accuracy": np.mean(train_scores),
            "val_accuracy": np.mean(val_scores),
            "final_accuracy": self.model.score(X_scaled, y),
            "n_samples": len(X),
            "n_features": len(self.feature_columns),
            "surge_ratio": y.mean(),  # Ratio of surge events
        }

        logger.info(
            f"ML model trained. Accuracy: {metrics['final_accuracy']:.3f}, Surge ratio: {metrics['surge_ratio']:.3f}"
        )
        return metrics

    def load_model(self) -> bool:
        """
        Load trained model from disk.

        Returns:
            True if model loaded successfully
        """
        try:
            if self.model_type == "zscore":
                # No model to load for statistical method
                return True
            elif os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded volume prediction model from {self.model_path}")
                return True
            else:
                logger.warning(f"Model files not found: {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _predict_zscore(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect volume surge using z-score method.

        Args:
            df: Recent market data

        Returns:
            Tuple of (is_surge, confidence)
        """
        try:
            if "volume" not in df.columns or df.empty:
                return False, 0.5

            # Calculate z-score
            recent_data = df.tail(self.lookback)
            volume_ma = recent_data["volume"].mean()
            volume_std = recent_data["volume"].std()

            if volume_std == 0:
                return False, 0.5

            current_volume = recent_data["volume"].iloc[-1]
            zscore = (current_volume - volume_ma) / volume_std

            # Detect surge - ensure native Python bool
            is_surge = bool(zscore > self.threshold)
            confidence = min(1.0, zscore / (self.threshold * 2))  # Normalize confidence

            return is_surge, float(confidence)

        except Exception as e:
            logger.error(f"Z-score prediction failed: {e}")
            return False, 0.5

    def _predict_ml(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect volume surge using ML model.

        Args:
            df: Recent market data

        Returns:
            Tuple of (is_surge, confidence)
        """
        try:
            # Create features from recent data
            features_df = self._create_features(df)
            if features_df.empty:
                return False, 0.5

            # Use most recent data point
            latest_features = features_df.iloc[-1:].values
            features_scaled = self.scaler.transform(latest_features)

            # Get predictions
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features_scaled)[0]
                predicted_class = np.argmax(probabilities)
                confidence = np.max(probabilities)
            else:
                predicted_class = self.model.predict(features_scaled)[0]
                confidence = 0.5

            is_surge = bool(predicted_class)
            return is_surge, float(confidence)

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return False, 0.5

    def predict(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect if there's a volume surge in the current data.

        Args:
            df: Recent market data

        Returns:
            Tuple of (is_surge, confidence)
        """
        if self.model_type == "zscore":
            return self._predict_zscore(df)
        else:
            if self.model is None:
                if not self.load_model():
                    return False, 0.5
            return self._predict_ml(df)
