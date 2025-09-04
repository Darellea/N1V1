"""
Volatility Predictor

Predicts next N-candle volatility using GARCH models or ML regression.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
from pathlib import Path
from scipy import stats

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logging.warning("arch library not available. GARCH models will not work.")

logger = logging.getLogger(__name__)


class VolatilityPredictor:
    """
    Predicts volatility regime (high/low) using statistical or ML models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the volatility predictor.

        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.model_type = config.get("type", "garch")
        self.forecast_horizon = config.get("forecast_horizon", 5)
        self.threshold = config.get("threshold", 0.02)  # Threshold for high/low classification
        self.lookback = config.get("lookback", 100)
        self.model_path = config.get("model_path", f"models/volatility_{self.model_type}.pkl")
        self.scaler_path = config.get("scaler_path", f"models/volatility_scaler.pkl")

        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []

        # Create models directory if it doesn't exist
        Path("models").mkdir(exist_ok=True)

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for volatility prediction from OHLCV data.

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
        df['abs_returns'] = df['returns'].abs()

        # Historical volatility measures
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'high_low_range_{period}'] = (df['high'] - df['low']).rolling(period).mean() / df['close']

        # Realized volatility (sum of squared returns)
        df['realized_vol_5'] = (df['returns'] ** 2).rolling(5).sum()
        df['realized_vol_10'] = (df['returns'] ** 2).rolling(10).sum()
        df['realized_vol_20'] = (df['returns'] ** 2).rolling(20).sum()

        # Volume-based volatility
        if 'volume' in df.columns:
            df['volume_volatility_10'] = df['volume'].pct_change().rolling(10).std()
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']

        # Price range features
        df['daily_range'] = (df['high'] - df['low']) / df['close']
        df['range_volatility_10'] = df['daily_range'].rolling(10).std()

        # Technical indicators that correlate with volatility
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()

        # Bollinger Band width
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_width'] = (2 * std_20) / sma_20

        # RSI for volatility signals
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Drop NaN values
        df = df.dropna()

        # Select feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]

        return df[self.feature_columns]

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create labels for volatility regime classification.

        Args:
            df: DataFrame with price data

        Returns:
            Series with future volatility measures
        """
        # Calculate future realized volatility
        future_returns = df['returns'].shift(-self.forecast_horizon)
        future_volatility = future_returns.rolling(self.forecast_horizon).std()

        return future_volatility.dropna()

    def _get_ml_model(self):
        """Get the appropriate ML model based on configuration."""
        if self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == "xgboost":
            return xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_estimators=100
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMRegressor(
                objective='regression',
                random_state=42,
                n_estimators=100
            )
        else:
            raise ValueError(f"Unsupported ML model type: {self.model_type}")

    def _train_garch(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Train GARCH model for volatility forecasting.

        Args:
            returns: Time series of returns

        Returns:
            Training metrics
        """
        if not ARCH_AVAILABLE:
            raise ImportError("arch library is required for GARCH models")

        try:
            # Fit GARCH(1,1) model
            model = arch_model(returns, vol='Garch', p=1, q=1)
            result = model.fit(disp='off')

            # Save model
            joblib.dump(result, self.model_path)

            # Calculate metrics
            fitted_volatility = result.conditional_volatility
            actual_volatility = returns.rolling(20).std()

            # Align the series
            common_idx = fitted_volatility.index.intersection(actual_volatility.index)
            fitted_aligned = fitted_volatility.loc[common_idx]
            actual_aligned = actual_volatility.loc[common_idx]

            # Calculate correlation
            correlation = fitted_aligned.corr(actual_aligned)

            metrics = {
                "model_type": "garch",
                "aic": result.aic,
                "bic": result.bic,
                "log_likelihood": result.loglikelihood,
                "correlation": correlation,
                "n_samples": len(returns)
            }

            logger.info(f"GARCH model trained. AIC: {result.aic:.2f}, Correlation: {correlation:.3f}")
            return metrics

        except Exception as e:
            logger.error(f"GARCH training failed: {e}")
            raise

    def _train_ml_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train ML model for volatility prediction.

        Args:
            df: Training data with OHLCV columns

        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_type} model for volatility prediction")

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
            model = self._get_ml_model()
            model.fit(X_train_scaled, y_train)

            # Evaluate (R² score)
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
            "train_r2": np.mean(train_scores),
            "val_r2": np.mean(val_scores),
            "final_r2": self.model.score(X_scaled, y),
            "n_samples": len(X),
            "n_features": len(self.feature_columns)
        }

        logger.info(f"ML model trained. R²: {metrics['final_r2']:.3f}")
        return metrics

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the volatility prediction model.

        Args:
            df: Training data with OHLCV columns

        Returns:
            Training metrics
        """
        if self.model_type == "garch":
            returns = df['close'].pct_change().dropna()
            return self._train_garch(returns)
        else:
            return self._train_ml_model(df)

    def load_model(self) -> bool:
        """
        Load trained model from disk.

        Returns:
            True if model loaded successfully
        """
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                if self.model_type != "garch" and os.path.exists(self.scaler_path):
                    self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded volatility prediction model from {self.model_path}")
                return True
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _predict_garch(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict volatility using GARCH model.

        Args:
            df: Recent market data

        Returns:
            Tuple of (regime, confidence)
        """
        if not ARCH_AVAILABLE:
            return "low", 0.5

        try:
            returns = df['close'].pct_change().dropna()
            if len(returns) < 20:
                return "low", 0.5

            # Forecast volatility
            forecasts = self.model.forecast(horizon=self.forecast_horizon)
            predicted_volatility = np.sqrt(forecasts.variance.iloc[-1].mean())

            # Get current realized volatility for comparison
            current_volatility = returns.tail(20).std()

            # Classify regime
            if predicted_volatility > current_volatility * (1 + self.threshold):
                regime = "high"
                confidence = min(1.0, (predicted_volatility / current_volatility - 1) / self.threshold)
            else:
                regime = "low"
                confidence = 0.5

            return regime, float(confidence)

        except Exception as e:
            logger.error(f"GARCH prediction failed: {e}")
            return "low", 0.5

    def _predict_ml(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict volatility using ML model.

        Args:
            df: Recent market data

        Returns:
            Tuple of (regime, confidence)
        """
        try:
            # Create features from recent data
            features_df = self._create_features(df)
            if features_df.empty:
                return "low", 0.5

            # Use most recent data point
            latest_features = features_df.iloc[-1:].values
            features_scaled = self.scaler.transform(latest_features)

            # Get prediction
            predicted_volatility = self.model.predict(features_scaled)[0]

            # Get current realized volatility for comparison
            returns = df['close'].pct_change().dropna()
            current_volatility = returns.tail(20).std() if len(returns) >= 20 else returns.std()

            # Classify regime
            if predicted_volatility > current_volatility * (1 + self.threshold):
                regime = "high"
                confidence = min(1.0, (predicted_volatility / current_volatility - 1) / self.threshold)
            else:
                regime = "low"
                confidence = 0.5

            return regime, float(confidence)

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return "low", 0.5

    def predict(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict volatility regime for the next N candles.

        Args:
            df: Recent market data

        Returns:
            Tuple of (regime, confidence)
        """
        if self.model is None:
            if not self.load_model():
                return "low", 0.5

        if self.model_type == "garch":
            return self._predict_garch(df)
        else:
            return self._predict_ml(df)
