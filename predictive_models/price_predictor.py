"""
Price Direction Predictor

Predicts next candle price direction using various ML models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class PricePredictor:
    """
    Predicts price direction (up, down, neutral) using ML models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the price predictor.

        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.model_type = config.get("type", "lightgbm")
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.lookback = config.get("lookback", 50)
        self.model_path = config.get("model_path", f"models/price_{self.model_type}.pkl")
        self.scaler_path = config.get("scaler_path", f"models/price_scaler.pkl")

        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []

        # Create models directory if it doesn't exist
        Path("models").mkdir(exist_ok=True)

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for price prediction from OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features
        """
        if df.empty:
            return pd.DataFrame()

        # Basic price features
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1

        # Volatility
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()

        # Volume features
        if 'volume' in df.columns:
            df['volume_ma_10'] = df['volume'].rolling(10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_10']

        # Technical indicators
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Drop NaN values
        df = df.dropna()

        # Select feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]

        return df[self.feature_columns]

    def _create_labels(self, df: pd.DataFrame, horizon: int = 5, profit_threshold: float = 0.005,
                      include_fees: bool = True, fee_rate: float = 0.001) -> pd.Series:
        """
        Create binary labels for trading decisions with strict prevention of look-ahead bias.

        Args:
            df: DataFrame with price data
            horizon: Number of periods ahead to look for forward return
            profit_threshold: Minimum profit threshold after fees (fractional)
            include_fees: Whether to account for trading fees
            fee_rate: Trading fee rate (fractional)

        Returns:
            Series with binary labels (1 for trade, 0 for skip)
        """
        # Calculate forward return over N bars
        future_price = df['close'].shift(-horizon)
        forward_return = (future_price - df['close']) / df['close']

        # Account for trading fees if requested
        if include_fees:
            # Assume round-trip fees (entry + exit)
            total_fee_rate = 2 * fee_rate
            # Adjust profit threshold to account for fees
            effective_threshold = profit_threshold + total_fee_rate
            labels = (forward_return > effective_threshold).astype(int)
        else:
            labels = (forward_return > profit_threshold).astype(int)

        return labels.dropna()

    def _get_model(self):
        """Get the appropriate ML model based on configuration."""
        if self.model_type == "logistic_regression":
            return LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "xgboost":
            return xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                n_estimators=100
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                objective='binary',
                random_state=42,
                n_estimators=100
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the price prediction model.

        Args:
            df: Training data with OHLCV columns

        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_type} model for price prediction")

        # Create features and labels
        features_df = self._create_features(df)
        labels = self._create_labels(df)

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
            model = self._get_model()
            model.fit(X_train_scaled, y_train)

            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            val_score = model.score(X_val_scaled, y_val)

            train_scores.append(train_score)
            val_scores.append(val_score)

        # Train final model on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._get_model()
        self.model.fit(X_scaled, y)

        # Save model and scaler
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        metrics = {
            "train_accuracy": np.mean(train_scores),
            "val_accuracy": np.mean(val_scores),
            "final_accuracy": self.model.score(X_scaled, y),
            "n_samples": len(X),
            "n_features": len(self.feature_columns)
        }

        logger.info(f"Model trained. Accuracy: {metrics['final_accuracy']:.3f}")
        return metrics

    def load_model(self) -> bool:
        """
        Load trained model from disk.

        Returns:
            True if model loaded successfully
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded price prediction model from {self.model_path}")
                return True
            else:
                logger.warning(f"Model files not found: {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict trading decision for the next candle.

        Args:
            df: Recent market data

        Returns:
            Tuple of (decision, confidence) where decision is "trade" or "skip"
        """
        if self.model is None:
            if not self.load_model():
                return "neutral", 0.5

        try:
            # Create features from recent data
            features_df = self._create_features(df)
            if features_df.empty:
                return "skip", 0.5

            # Use most recent data point
            latest_features = features_df.iloc[-1:].values
            features_scaled = self.scaler.transform(latest_features)

            # Get predictions
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                # For binary classification, probabilities[1] is probability of positive class (trade)
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            else:
                predicted_class = self.model.predict(features_scaled)[0]
                confidence = 0.5  # Default confidence for models without predict_proba

            # Map to decision
            decision_map = {0: "skip", 1: "trade"}
            decision = decision_map.get(predicted_class, "skip")

            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                decision = "skip"

            return decision, float(confidence)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return "skip", 0.5
