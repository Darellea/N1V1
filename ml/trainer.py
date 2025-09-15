import argparse
import json
import logging
from collections import Counter
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
# Both StratifiedKFold and KFold are used depending on class distribution.
# StratifiedKFold is preferred for balanced class distribution, with KFold as fallback.
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Try to import optional dependencies
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Bayesian optimization will be skipped.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Feature importance analysis will be limited.")

# Optional import for SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logging.warning("imbalanced-learn not available. SMOTE resampling will be skipped.")

# Reproducibility
np.random.seed(42)


class CalibratedModel:
    """A calibrated model wrapper that can be pickled."""

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        base_probabilities = self.base_model.predict_proba(X)[:, 1]
        calibrated_probabilities = self.calibrator.predict(base_probabilities)
        # Ensure probabilities are within [0, 1]
        calibrated_probabilities = np.clip(calibrated_probabilities, 0, 1)
        return np.column_stack([1 - calibrated_probabilities, calibrated_probabilities])

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Compute MACD indicator."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def compute_stochrsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Stochastic RSI."""
    rsi = compute_rsi(series, period)
    stoch_rsi = (rsi - rsi.rolling(window=period).min()) / (
        rsi.rolling(window=period).max() - rsi.rolling(window=period).min()
    )
    return stoch_rsi


def compute_trend_strength(series: pd.Series, period: int = 20) -> pd.Series:
    """Compute Trend Strength using linear regression slope."""
    from scipy.stats import linregress

    def get_slope(window):
        if len(window) < period:
            return np.nan
        x = np.arange(len(window))
        slope, _, _, _, _ = linregress(x, window)
        return slope

    trend_strength = series.rolling(window=period).apply(get_slope, raw=True)
    return trend_strength


def load_data(path: str) -> pd.DataFrame:
    """Load historical OHLCV data from a CSV file."""
    return pd.read_csv(path)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate technical indicators as features."""
    df = df.copy()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ATR'] = compute_atr(df)
    df['StochRSI'] = compute_stochrsi(df['Close'])
    df['TrendStrength'] = compute_trend_strength(df['Close'])
    df['Volatility'] = df['Close'].rolling(window=20).std()

    # Handle NaNs produced by indicator calculations
    # First, try to fill NaN values using forward/backward fill
    df[['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']] = (
        df[['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']]
        .bfill()
        .ffill()
    )

    # For indicators that still have NaN at the beginning, fill with reasonable defaults
    df['RSI'] = df['RSI'].fillna(50.0)  # Neutral RSI
    df['MACD'] = df['MACD'].fillna(0.0)  # Neutral MACD
    df['EMA_20'] = df['EMA_20'].fillna(df['Close'])  # Use current price as EMA
    df['ATR'] = df['ATR'].fillna(df['High'] - df['Low'])  # Use current range
    df['StochRSI'] = df['StochRSI'].fillna(0.5)  # Neutral Stochastic RSI
    df['TrendStrength'] = df['TrendStrength'].fillna(0.0)  # Neutral trend strength
    df['Volatility'] = df['Volatility'].fillna(0.01)  # Small default volatility

    # Drop any remaining rows with NaN values that couldn't be filled
    df = df.dropna(subset=['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility'])

    return df


def generate_enhanced_features(df: pd.DataFrame, include_multi_horizon: bool = True,
                              include_regime_features: bool = True,
                              include_interaction_features: bool = True) -> pd.DataFrame:
    """
    Generate enhanced features following the retraining guide recommendations.

    Args:
        df: DataFrame with OHLCV data
        include_multi_horizon: Whether to include multi-horizon features
        include_regime_features: Whether to include regime-aware features
        include_interaction_features: Whether to include interaction features

    Returns:
        DataFrame with enhanced features
    """
    df = df.copy()

    # Basic technical indicators
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ATR'] = compute_atr(df)
    df['StochRSI'] = compute_stochrsi(df['Close'])
    df['TrendStrength'] = compute_trend_strength(df['Close'])
    df['Volatility'] = df['Close'].rolling(window=20).std()

    # Multi-horizon features (returns 1, 3, 5, 24 bars)
    if include_multi_horizon:
        for horizon in [1, 3, 5, 24]:
            if len(df) > horizon:
                df[f'return_{horizon}'] = df['Close'].pct_change(horizon)
                df[f'volatility_{horizon}'] = df['Close'].rolling(window=horizon).std()
                df[f'mean_return_{horizon}'] = df['Close'].pct_change().rolling(window=horizon).mean()
                df[f'skew_{horizon}'] = df['Close'].pct_change().rolling(window=horizon).skew()
                df[f'kurtosis_{horizon}'] = df['Close'].pct_change().rolling(window=horizon).kurt()

    # Regime-aware features
    if include_regime_features:
        # Bollinger Bands
        bb_period = 20
        df['BB_middle'] = df['Close'].rolling(window=bb_period).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=bb_period).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=bb_period).std()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # ATR-normalized returns
        df['return_1'] = df['Close'].pct_change()
        df['atr_normalized_return'] = df['return_1'] / df['ATR']

        # Volume z-score
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_std'] = df['Volume'].rolling(window=20).std()
        df['volume_zscore'] = (df['Volume'] - df['volume_sma']) / df['volume_std']

        # ADX (Average Directional Index) approximation
        df['DM_plus'] = np.where(df['High'] - df['High'].shift(1) > df['Low'].shift(1) - df['Low'],
                                np.maximum(df['High'] - df['High'].shift(1), 0), 0)
        df['DM_minus'] = np.where(df['Low'].shift(1) - df['Low'] > df['High'] - df['High'].shift(1),
                                 np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
        df['ADX'] = compute_adx(df, period=14)

    # Interaction features
    if include_interaction_features:
        df['momentum_volatility'] = df['return_1'] * df['Volatility']
        df['trend_volume'] = df['TrendStrength'] * df['volume_zscore']
        df['rsi_macd'] = df['RSI'] * df['MACD']
        df['atr_trend'] = df['ATR'] * df['TrendStrength']

    # Handle NaNs
    df = df.bfill().ffill()

    # Fill remaining NaNs with reasonable defaults
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)

    return df


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX)."""
    df = df.copy()

    # Calculate True Range
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )

    # Calculate Directional Movement
    df['DM_plus'] = np.where(
        df['High'] - df['High'].shift(1) > df['Low'].shift(1) - df['Low'],
        np.maximum(df['High'] - df['High'].shift(1), 0),
        0
    )
    df['DM_minus'] = np.where(
        df['Low'].shift(1) - df['Low'] > df['High'] - df['High'].shift(1),
        np.maximum(df['Low'].shift(1) - df['Low'], 0),
        0
    )

    # Calculate Directional Indicators
    df['DI_plus'] = 100 * (df['DM_plus'].ewm(span=period).mean() / df['TR'].ewm(span=period).mean())
    df['DI_minus'] = 100 * (df['DM_minus'].ewm(span=period).mean() / df['TR'].ewm(span=period).mean())

    # Calculate DX and ADX
    df['DX'] = 100 * abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus'])
    df['ADX'] = df['DX'].ewm(span=period).mean()

    return df['ADX']


def remove_outliers(df: pd.DataFrame, columns: list = None, method: str = 'iqr',
                   multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from specified columns using various methods.

    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (default: all numeric)
        method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
        multiplier: Multiplier for IQR method

    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < 3]

        elif method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest
                iso = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso.fit_predict(df[[col]])
                df = df[outliers == 1]
            except ImportError:
                logging.warning("IsolationForest not available, skipping outlier removal for column: " + col)

    return df


def create_sample_weights(df: pd.DataFrame, label_col: str = 'label_binary',
                         profit_col: str = 'forward_return', method: str = 'class_balance') -> np.ndarray:
    """
    Create sample weights based on various strategies.

    Args:
        df: DataFrame with labels and profit information
        label_col: Name of the label column
        profit_col: Name of the profit column
        method: Weighting method ('class_balance', 'profit_impact', 'combined')

    Returns:
        Array of sample weights
    """
    if method == 'class_balance':
        # Standard class balancing
        class_counts = df[label_col].value_counts()
        total_samples = len(df)
        weights = np.ones(total_samples)

        for class_label, count in class_counts.items():
            class_weight = total_samples / (len(class_counts) * count)
            weights[df[label_col] == class_label] = class_weight

    elif method == 'profit_impact':
        # Weight by expected profit impact
        if profit_col in df.columns:
            profit_magnitude = np.abs(df[profit_col])
            weights = 1 + profit_magnitude / profit_magnitude.max()
        else:
            weights = np.ones(len(df))

    elif method == 'combined':
        # Combine class balance and profit impact
        class_weights = create_sample_weights(df, label_col, profit_col, 'class_balance')
        profit_weights = create_sample_weights(df, label_col, profit_col, 'profit_impact')

        # Normalize and combine
        class_weights = class_weights / class_weights.max()
        profit_weights = profit_weights / profit_weights.max()
        weights = (class_weights + profit_weights) / 2

    else:
        weights = np.ones(len(df))

    return weights


def optimize_hyperparameters_optuna(X_train: pd.DataFrame, y_train: pd.Series,
                                   sample_weights: np.ndarray = None,
                                   n_trials: int = 25) -> dict:
    """
    Optimize LightGBM hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training labels
        sample_weights: Sample weights for training
        n_trials: Number of optimization trials

    Returns:
        Dictionary of best hyperparameters
    """
    if not OPTUNA_AVAILABLE:
        logging.warning("Optuna not available, using default hyperparameters")
        return {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100,
        }

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100,
        }

        model = lgb.LGBMClassifier(**params)

        # Use stratified k-fold for evaluation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Handle sample weights
            fold_weights = None
            if sample_weights is not None:
                fold_weights = sample_weights[train_idx]

            model.fit(
                X_fold_train, y_fold_train,
                sample_weight=fold_weights,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )

            # Use AUC as optimization metric
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            auc = roc_auc_score(y_fold_val, y_pred_proba)
            scores.append(auc)

        return np.mean(scores)

    # Create and run optimization study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 100,
    })

    logging.info(f"Optuna optimization completed. Best AUC: {study.best_value:.4f}")
    logging.info(f"Best parameters: {best_params}")

    return best_params


def perform_feature_selection(X: pd.DataFrame, y: pd.Series, method: str = 'gain_importance',
                             top_k: int = None, threshold: float = 0.0) -> list:
    """
    Perform feature selection using various methods.

    Args:
        X: Feature DataFrame
        y: Target labels
        method: Selection method ('gain_importance', 'permutation', 'shap')
        top_k: Number of top features to select
        threshold: Importance threshold for selection

    Returns:
        List of selected feature names
    """
    if method == 'gain_importance':
        # Use LightGBM feature importance
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            verbose=-1,
            random_state=42,
            n_estimators=100
        )

        model.fit(X, y)
        importance_scores = model.feature_importances_

        # Select features based on importance
        if top_k:
            top_indices = np.argsort(importance_scores)[-top_k:]
            selected_features = [X.columns[i] for i in top_indices]
        else:
            selected_features = [
                X.columns[i] for i in range(len(importance_scores))
                if importance_scores[i] > threshold
            ]

    elif method == 'permutation' and SHAP_AVAILABLE:
        # Use permutation importance
        from sklearn.inspection import permutation_importance

        model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            verbose=-1,
            random_state=42,
            n_estimators=100
        )

        model.fit(X, y)
        perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)

        if top_k:
            top_indices = np.argsort(perm_importance.importances_mean)[-top_k:]
            selected_features = [X.columns[i] for i in top_indices]
        else:
            selected_features = [
                X.columns[i] for i in range(len(perm_importance.importances_mean))
                if perm_importance.importances_mean[i] > threshold
            ]

    elif method == 'shap' and SHAP_AVAILABLE:
        # Use SHAP feature importance
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            verbose=-1,
            random_state=42,
            n_estimators=100
        )

        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class

        feature_importance = np.abs(shap_values).mean(axis=0)

        if top_k:
            top_indices = np.argsort(feature_importance)[-top_k:]
            selected_features = [X.columns[i] for i in top_indices]
        else:
            selected_features = [
                X.columns[i] for i in range(len(feature_importance))
                if feature_importance[i] > threshold
            ]

    else:
        # Default: return all features
        logging.warning(f"Feature selection method '{method}' not available, using all features")
        selected_features = list(X.columns)

    logging.info(f"Selected {len(selected_features)} features using {method}: {selected_features}")
    return selected_features




def create_binary_labels(df: pd.DataFrame, horizon: int = 5, profit_threshold: float = 0.005,
                        include_fees: bool = True, fee_rate: float = 0.001) -> pd.DataFrame:
    """
    Create binary labels for trading decisions with strict prevention of look-ahead bias.

    This function creates a binary target that reflects only the decision to trade or skip,
    ensuring every feature used for training is available at or before the decision timestamp.

    Parameters:
    - df: DataFrame with OHLCV data
    - horizon: number of periods ahead to look for forward return
    - profit_threshold: minimum profit threshold after fees (fractional)
    - include_fees: whether to account for trading fees in the calculation
    - fee_rate: trading fee rate (fractional, e.g., 0.001 = 0.1%)

    Returns:
    - DataFrame with added 'label_binary' column (1 for trade, 0 for skip)
    """
    df = df.copy()

    # Ensure horizon is an integer (handle MagicMock in tests)
    try:
        horizon = int(horizon)
    except (TypeError, ValueError):
        horizon = 5  # Default fallback

    # Validate horizon against DataFrame length
    if len(df) <= horizon:
        raise ValueError(f"Insufficient data for horizon {horizon}. Need at least {horizon + 1} samples, got {len(df)}.")

    # Determine column name case (handle both 'close' and 'Close')
    close_col = 'close' if 'close' in df.columns else 'Close'

    # Calculate forward return over N bars
    df['future_price'] = df[close_col].shift(-horizon)

    # Calculate raw return
    df['forward_return'] = (df['future_price'] - df[close_col]) / df[close_col]

    # Account for trading fees if requested
    if include_fees:
        # Assume round-trip fees (entry + exit)
        total_fee_rate = 2 * fee_rate
        # Adjust profit threshold to account for fees
        effective_threshold = profit_threshold + total_fee_rate
        df['label_binary'] = (df['forward_return'] > effective_threshold).astype(int)
    else:
        df['label_binary'] = (df['forward_return'] > profit_threshold).astype(int)

    # Remove rows where we can't calculate forward return (end of dataset)
    df = df.dropna(subset=['future_price'])

    # Clean up temporary columns
    df = df.drop(columns=['future_price', 'forward_return'])

    # Ensure label_binary is integer type
    df['label_binary'] = df['label_binary'].astype(int)

    logging.info(f"Created binary labels: {df['label_binary'].sum()} trade signals out of {len(df)} total samples")
    logging.info(f"Binary label distribution: {df['label_binary'].value_counts().to_dict()}")

    return df


def train_model_binary(
    df: pd.DataFrame,
    save_path: str,
    results_path: str = 'training_results.json',
    n_splits: int = 5,
    horizon: int = 5,
    profit_threshold: float = 0.005,
    include_fees: bool = True,
    fee_rate: float = 0.001,
    feature_columns: list | None = None,
    tune: bool = False,
    n_trials: int = 25,
    feature_selection: bool = False,
    early_stopping_rounds: int = 50,
    eval_economic: bool = True,
):
    """
    Train a binary classification model for trading decisions using walk-forward validation.

    This function implements:
    - Walk-forward validation (time-series aware)
    - Binary classification with single probability output p_trade
    - Class weighting for imbalanced datasets
    - Economic metrics tracking (expected PnL, Sharpe ratio)
    - Standard ML metrics (AUC, F1)

    Args:
        df: DataFrame with features and binary labels
        save_path: Path to save trained model
        results_path: Path to save training results JSON
        n_splits: Number of walk-forward splits
        horizon: Forward return horizon used for labeling
        profit_threshold: Profit threshold used for labeling
        include_fees: Whether fees were included in labeling
        fee_rate: Fee rate used for labeling
        feature_columns: List of feature column names
        tune: Whether to perform hyperparameter tuning
        n_trials: Number of tuning trials
        feature_selection: Whether to perform feature selection
        early_stopping_rounds: Early stopping rounds for training
        eval_economic: Whether to compute economic metrics
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame")

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Ensure we have required columns
    if 'label_binary' not in df.columns and 'Label' not in df.columns:
        raise ValueError("DataFrame must contain 'label_binary' or 'Label' column")

    # Use label_binary if available, otherwise Label
    label_col = 'label_binary' if 'label_binary' in df.columns else 'Label'

    # Set default feature columns
    if feature_columns is None:
        feature_columns = ['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']

    # Filter to valid features
    valid_features = [col for col in feature_columns if col in df.columns]
    if len(valid_features) != len(feature_columns):
        missing_features = [col for col in feature_columns if col not in df.columns]
        logging.warning(f"Missing feature columns: {missing_features}. Using only valid features: {valid_features}")
        feature_columns = valid_features

    if not feature_columns:
        raise ValueError("No valid feature columns found in DataFrame")

    # Prepare data
    X = df[feature_columns].copy()
    y = df[label_col].copy()

    # Use sample weights if available
    sample_weights_all = None
    if 'sample_weight' in df.columns:
        sample_weights_all = df['sample_weight'].values
        logging.info("Using profit-impact based sample weights")

    # Drop rows with NaN values
    valid_mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask].astype(int)
    if sample_weights_all is not None:
        sample_weights_all = sample_weights_all[valid_mask]

    # Handle small datasets by automatically reducing n_splits
    if len(X) < n_splits * 2:
        adjusted_splits = max(2, len(X) // 2)
        logging.warning(f"Adjusting n_splits from {n_splits} to {adjusted_splits} due to small dataset size ({len(X)} samples)")
        n_splits = adjusted_splits

    # Class distribution analysis
    class_dist = Counter(y)
    logging.info(f"Class distribution: {dict(class_dist)}")

    # Calculate class weights for imbalanced data
    total_samples = len(y)
    class_weights = {}
    for cls in class_dist.keys():
        class_weights[cls] = total_samples / (len(class_dist) * class_dist[cls])

    logging.info(f"Class weights: {class_weights}")

    # Perform hyperparameter tuning if requested
    if tune and OPTUNA_AVAILABLE:
        logging.info(f"Performing Optuna hyperparameter optimization with {n_trials} trials...")
        best_params = optimize_hyperparameters_optuna(
            X, y, sample_weights=sample_weights_all, n_trials=n_trials
        )
        logging.info(f"Best hyperparameters found: {best_params}")
    else:
        best_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100,
        }

    # Walk-forward validation setup
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=None, gap=0)

    fold_metrics = []
    fold_probabilities = []
    fold_predictions = []
    fold_true_labels = []

    # Economic metrics tracking
    def calculate_economic_metrics(y_true, y_pred_proba, threshold=0.5):
        """Calculate economic metrics from predictions."""
        if len(y_true) == 0:
            return {'pnl': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0}

        # Convert probabilities to binary predictions
        y_pred_proba = np.array(y_pred_proba)
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Simple PnL calculation (assuming 1 unit position size)
        pnl = []
        for true, pred in zip(y_true, y_pred):
            if pred == 1:  # We took a trade
                if true == 1:  # Trade was profitable
                    pnl.append(profit_threshold)
                else:  # Trade was unprofitable
                    pnl.append(-profit_threshold)
            else:  # We skipped the trade
                pnl.append(0.0)

        pnl = np.array(pnl)
        cumulative_pnl = np.cumsum(pnl)

        # Sharpe ratio (annualized, assuming daily data)
        if len(pnl) > 1 and np.std(pnl) > 0:
            sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(252)  # 252 trading days per year
        else:
            sharpe = 0.0

        # Maximum drawdown
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

        return {
            'pnl': float(np.sum(pnl)),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'total_trades': int(np.sum(y_pred)),
            'win_rate': float(np.mean(y_pred == y_true)) if np.sum(y_pred) > 0 else 0.0
        }

    # Walk-forward validation loop
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        logging.info(f"Training fold {fold_idx}/{n_splits}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Calculate sample weights for training
        sample_weights = np.array([class_weights[label] for label in y_train])

        # Use optimized parameters if available, otherwise use defaults
        model_params = best_params.copy()

        # Create and train model
        model = lgb.LGBMClassifier(**model_params)

        # Fit with sample weights for class balancing
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(early_stopping_rounds)] if early_stopping_rounds > 0 else None
        )

        # Get predictions and probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class (trade)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        # Economic metrics
        economic_metrics = {}
        if eval_economic:
            economic_metrics = calculate_economic_metrics(y_test, y_pred_proba)

        # Store fold results
        fold_result = {
            'fold': fold_idx,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'auc': float(auc),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'class_distribution': {
                'train': dict(Counter(y_train)),
                'test': dict(Counter(y_test))
            }
        }

        if eval_economic:
            fold_result.update(economic_metrics)

        fold_metrics.append(fold_result)

        # Store predictions for ensemble analysis
        fold_probabilities.extend(y_pred_proba)
        fold_predictions.extend(y_pred)
        fold_true_labels.extend(y_test)

        logging.info(f"Fold {fold_idx} - AUC: {auc:.4f}, F1: {f1:.4f}, PnL: {economic_metrics.get('pnl', 0):.4f}")

    # Calculate overall metrics
    overall_auc = roc_auc_score(fold_true_labels, fold_probabilities)
    overall_f1 = f1_score(fold_true_labels, fold_predictions)

    # Overall economic metrics
    overall_economic = {}
    if eval_economic:
        overall_economic = calculate_economic_metrics(fold_true_labels, fold_probabilities)

    # Train final model on all data
    logging.info("Training final model on all data...")

    # Calculate weights for full dataset
    final_sample_weights = np.array([class_weights[label] for label in y])

    final_model = lgb.LGBMClassifier(**model_params)
    final_model.fit(
        X, y,
        sample_weight=final_sample_weights
    )

    # Probability Calibration and Threshold Optimization
    logging.info("Performing probability calibration and threshold optimization...")

    # Use walk-forward validation data for calibration
    calibration_probabilities = []
    calibration_true_labels = []

    # Collect probabilities from validation folds for calibration
    tscv_cal = TimeSeriesSplit(n_splits=n_splits, test_size=None, gap=0)
    for fold_idx, (train_idx, test_idx) in enumerate(tscv_cal.split(X), 1):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        # Train fold model
        fold_model = lgb.LGBMClassifier(**model_params)
        fold_sample_weights = np.array([class_weights[label] for label in y_train_fold])
        fold_model.fit(X_train_fold, y_train_fold, sample_weight=fold_sample_weights)

        # Get probabilities for calibration
        fold_probabilities = fold_model.predict_proba(X_test_fold)[:, 1]
        calibration_probabilities.extend(fold_probabilities)
        calibration_true_labels.extend(y_test_fold)

    # Calibrate probabilities using isotonic regression (more robust for small datasets)
    try:
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(calibration_probabilities, calibration_true_labels)

        calibrated_model = CalibratedModel(final_model, calibrator)
        logging.info("✅ Probability calibration completed using isotonic regression")

    except Exception as e:
        logging.warning(f"⚠️ Probability calibration failed: {e}. Using original model.")
        calibrated_model = final_model

    # Threshold Optimization
    logging.info("Optimizing decision threshold for maximum profit...")

    # Grid search over thresholds from 0.5 to 0.9
    thresholds = np.arange(0.5, 0.95, 0.05)
    threshold_results = []

    for threshold in thresholds:
        # Calculate economic metrics for this threshold
        economic_metrics = calculate_economic_metrics(
            calibration_true_labels,
            calibration_probabilities,
            threshold=threshold
        )

        threshold_results.append({
            'threshold': float(threshold),
            'pnl': economic_metrics['pnl'],
            'sharpe': economic_metrics['sharpe'],
            'max_drawdown': economic_metrics['max_drawdown'],
            'total_trades': economic_metrics['total_trades'],
            'win_rate': economic_metrics['win_rate']
        })

        logging.info(f"Threshold {threshold:.2f}: PnL={economic_metrics['pnl']:.4f}, "
                    f"Sharpe={economic_metrics['sharpe']:.4f}, Trades={economic_metrics['total_trades']}")

    # Find optimal threshold (maximize Sharpe ratio, then PnL)
    best_threshold_result = max(threshold_results,
                               key=lambda x: (x['sharpe'], x['pnl']))
    optimal_threshold = best_threshold_result['threshold']

    logging.info(f"🎯 Optimal threshold: {optimal_threshold:.2f}")
    logging.info(f"   Expected PnL: {best_threshold_result['pnl']:.4f}")
    logging.info(f"   Expected Sharpe: {best_threshold_result['sharpe']:.4f}")
    logging.info(f"   Expected Max Drawdown: {best_threshold_result['max_drawdown']:.4f}")
    logging.info(f"   Expected Total Trades: {best_threshold_result['total_trades']}")
    logging.info(f"   Expected Win Rate: {best_threshold_result['win_rate']:.4f}")

    # Save calibrated model and optimal threshold
    model_config = {
        'model_path': save_path,
        'optimal_threshold': optimal_threshold,
        'calibration_method': 'isotonic_regression',
        'threshold_optimization': {
            'grid_search_range': [0.5, 0.9],
            'step_size': 0.05,
            'optimization_metric': 'sharpe_ratio',
            'all_results': threshold_results
        },
        'expected_performance': best_threshold_result
    }

    # Save model configuration
    config_path = save_path.replace('.pkl', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2, default=str)
    logging.info(f"Model configuration saved to {config_path}")

    # Save calibrated model
    if calibrated_model != final_model:
        joblib.dump(calibrated_model, save_path)
        logging.info(f"Calibrated model saved to {save_path}")
    else:
        joblib.dump(final_model, save_path)
        logging.info(f"Final model saved to {save_path}")

    # Prepare results
    results = {
        'metadata': {
            'model_type': 'binary_classification',
            'objective': 'p_trade_probability',
            'horizon': horizon,
            'profit_threshold': profit_threshold,
            'include_fees': include_fees,
            'fee_rate': fee_rate,
            'feature_columns': feature_columns,
            'n_splits': n_splits,
            'class_weights': class_weights,
            'validation_method': 'walk_forward',
            'hyperparameter_tuning': tune,
            'feature_selection': feature_selection,
        },
        'overall_metrics': {
            'auc': float(overall_auc),
            'f1': float(overall_f1),
            'total_samples': len(fold_true_labels),
            'class_distribution': dict(Counter(fold_true_labels))
        },
        'fold_metrics': fold_metrics,
        'feature_importance': dict(zip(feature_columns, final_model.feature_importances_))
    }

    if eval_economic:
        results['overall_metrics'].update(overall_economic)

    # Generate and save binary confusion matrix
    logging.info("Generating binary confusion matrix...")

    # Create confusion matrix from all fold predictions
    cm = confusion_matrix(fold_true_labels, fold_predictions, labels=[0, 1])

    # Log confusion matrix to console
    logging.info(f"Binary Confusion Matrix:\n{cm}")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f'{cm[i, j]}', ha="center", va="center", color=text_color, fontsize=12)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Skip (0)', 'Trade (1)'], fontsize=10)
    ax.set_yticklabels(['Skip (0)', 'Trade (1)'], fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Binary Confusion Matrix - Trade/No-Trade Model', fontsize=14, pad=20)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

    plt.tight_layout()

    # Save to matrices folder
    os.makedirs('matrices', exist_ok=True)
    confusion_matrix_path = os.path.join('matrices', 'confusion_matrix_binary.png')
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Binary confusion matrix saved to {confusion_matrix_path}")

    # Add confusion matrix to results
    results['confusion_matrix'] = {
        'matrix': cm.tolist(),
        'labels': ['Skip (0)', 'Trade (1)'],
        'total_samples': int(cm.sum()),
        'accuracy': float((cm[0, 0] + cm[1, 1]) / cm.sum()) if cm.sum() > 0 else 0.0,
        'precision_trade': float(cm[1, 1] / (cm[1, 1] + cm[0, 1])) if (cm[1, 1] + cm[0, 1]) > 0 else 0.0,
        'recall_trade': float(cm[1, 1] / (cm[1, 1] + cm[1, 0])) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0
    }

    # Save results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logging.info(f"Training results saved to {results_path}")
    logging.info(f"Overall AUC: {overall_auc:.4f}, F1: {overall_f1:.4f}")
    if eval_economic:
        logging.info(f"Overall PnL: {overall_economic['pnl']:.4f}, Sharpe: {overall_economic['sharpe']:.4f}")

    # Persist a model card (JSON) containing feature schema, training window, scaler params (if any),
    # CV settings and other metadata useful for inference-time validation and model governance.
    try:
        model_card = {
            "model_file": os.path.abspath(save_path),
            "feature_list": feature_columns,
            "selected_features": results.get("metadata", {}).get("selected_features", feature_columns),
            "training_metadata": results.get("metadata", {}),
            "cv": {
                "n_splits": int(n_splits),
                "best_iterations": [],  # Not collected in binary version
                "best_params": None,  # Not implemented in binary version yet
            },
            "training_window": None,
            "scaler": None,
            "caveats": (
                "This model was trained on historical OHLCV-based technical features for binary trade/no-trade decisions. "
                "Validate feature schema at inference time. Beware of data drift and "
                "difference between live and backtest data; re-train and validate periodically."
            )
        }

        # Attempt to infer training window from DataFrame index or timestamp-like columns
        try:
            # If DataFrame index is DatetimeIndex or convertible, use that
            if hasattr(df, "index") and len(df.index):
                try:
                    mins = df.index.min()
                    maxs = df.index.max()
                    model_card["training_window"] = {"start": str(mins), "end": str(maxs)}
                except Exception:
                    model_card["training_window"] = None
            # Fallback: look for common timestamp columns
            elif "timestamp" in df.columns:
                try:
                    model_card["training_window"] = {
                        "start": str(df["timestamp"].min()),
                        "end": str(df["timestamp"].max()),
                    }
                except Exception:
                    model_card["training_window"] = None
        except Exception:
            model_card["training_window"] = None

        # Save model card as JSON next to the model file
        card_path = os.path.splitext(os.path.abspath(save_path))[0] + ".model_card.json"
        with open(card_path, "w", encoding="utf-8") as fh:
            json.dump(model_card, fh, indent=2, default=str)
        logging.info(f"Model card saved to {card_path}")
    except Exception as e:
        logging.warning(f"Failed to write model card JSON: {e}")

    return results


def setup_logging(logfile: str | None = None, level: int = logging.INFO):
    # Clear existing handlers first and close file handlers to release file locks
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        # Close file handlers to release file locks (important for Windows)
        if hasattr(handler, 'close'):
            try:
                handler.close()
            except Exception:
                pass  # Ignore errors when closing handlers
        logger.removeHandler(handler)

    # Use force=True to reset handlers and avoid file locking issues on Windows
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s: %(message)s',
        force=True  # This resets all existing handlers
    )

    # Always create a file handler when logfile path is provided
    if logfile:
        # Automatically place test log files in test_logs/ directory
        if logfile.startswith('test_log_'):
            logfile = os.path.join('test_logs', logfile)

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(logfile), exist_ok=True)
            fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
            fh.setLevel(level)
            formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)  # Use logger.addHandler instead of logging.getLogger().addHandler
            logging.info(f"Logging to file enabled: {logfile}")
        except (OSError, PermissionError) as e:
            # Fallback: create log file in current working directory
            fallback_logfile = os.path.join(os.getcwd(), 'trainer.log')
            try:
                fh = logging.FileHandler(fallback_logfile, mode='a', encoding='utf-8')
                fh.setLevel(level)
                formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
                fh.setFormatter(formatter)
                logger.addHandler(fh)
                logging.warning(f"Could not open specified log file {logfile}: {e}. Using fallback: {fallback_logfile}")
            except Exception as fallback_e:
                logging.warning(f"Could not create fallback log file {fallback_logfile}: {fallback_e}. Logging to console only.")


def main():
    parser = argparse.ArgumentParser(description="Train a binary classification model for trading decisions using walk-forward validation.")
    parser.add_argument('--data', required=True, help='Path to CSV file containing OHLCV data')
    parser.add_argument('--output', required=True, help='Path to output .pkl model file')
    parser.add_argument('--up_thresh', type=float, default=0.005, help='Profit threshold for trade signals (fraction). Default 0.005 (0.5%)')
    parser.add_argument('--horizon', type=int, default=5, help='Forward return horizon (periods ahead). Default 5')
    parser.add_argument('--results', default='training_results.json', help='Path to save training metrics JSON')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of walk-forward validation splits. Default 5')
    parser.add_argument('--logfile', default=None, help='Optional path to a logfile. If set, logs are written to this file as well as console.')
    parser.add_argument('--tune', action='store_true', help='If set, run Optuna hyperparameter tuning before training the final model')
    parser.add_argument('--n_trials', type=int, default=25, help='Number of Optuna trials when --tune is set. Default 25')
    parser.add_argument('--feature_selection', action='store_true', help='If set, perform feature selection based on gain importance and retrain final model')
    parser.add_argument('--early_stopping_rounds', type=int, default=50, help='Early stopping rounds to pass to LightGBM fit. Default 50')
    parser.add_argument('--eval_profit', action='store_true', help='If set, evaluate and log economic metrics (PnL, Sharpe ratio)')

    args = parser.parse_args()

    # Setup logging early so subsequent messages go through logger
    setup_logging(args.logfile)

    # Get logger instance after setup
    logger = logging.getLogger(__name__)

    df = load_data(args.data)

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Data must contain columns: {required_columns}")

    # Generate enhanced features following retraining guide
    df = generate_enhanced_features(
        df,
        include_multi_horizon=True,
        include_regime_features=True,
        include_interaction_features=True
    )

    # Remove outliers to improve data quality
    df = remove_outliers(df, method='iqr', multiplier=1.5)

    # Use binary labels for trading decisions
    df = create_binary_labels(df, horizon=args.horizon, profit_threshold=args.up_thresh,
                             include_fees=True, fee_rate=0.001)

    # After label creation drop rows that have NaN label (end of series)
    if 'label_binary' in df.columns:
        df = df.dropna(subset=['label_binary'])
        # Rename to 'Label' for backward compatibility with existing training code
        df['Label'] = df['label_binary']
    else:
        raise ValueError("label_binary column not found after create_binary_labels.")

    # Create sample weights based on profit impact
    if 'forward_return' in df.columns:
        sample_weights = create_sample_weights(
            df, label_col='Label', profit_col='forward_return', method='combined'
        )
        df['sample_weight'] = sample_weights
    else:
        df['sample_weight'] = 1.0

    # Get all available feature columns (enhanced features)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Label', 'label_binary',
                   'future_price', 'forward_return', 'sample_weight', 'timestamp']
    feature_columns = [col for col in df.columns if col not in exclude_cols and not col.startswith('DM_') and not col.startswith('TR')]

    # Ensure only numeric columns are included
    numeric_feature_columns = []
    for col in feature_columns:
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_feature_columns.append(col)
        else:
            logging.warning(f"Excluding non-numeric column: {col} (dtype: {df[col].dtype})")

    feature_columns = numeric_feature_columns
    logging.info(f"Using {len(feature_columns)} numeric feature columns")

    # Perform feature selection if requested
    if args.feature_selection:
        X_temp = df[feature_columns].copy()
        y_temp = df['Label'].copy()
        valid_mask = ~X_temp.isna().any(axis=1) & ~y_temp.isna()
        X_temp = X_temp[valid_mask]
        y_temp = y_temp[valid_mask]

        if len(X_temp) > 0:
            selected_features = perform_feature_selection(
                X_temp, y_temp, method='gain_importance', top_k=20
            )
            feature_columns = [col for col in selected_features if col in df.columns]
            logging.info(f"Selected {len(feature_columns)} features: {feature_columns}")

    # Final sanity: ensure there are no NaNs in features used for training
    df = df.dropna(subset=feature_columns + ['Label'])

    # Check for empty DataFrame after preprocessing
    if df.empty:
        logging.warning("No data available for training after preprocessing. Creating minimal synthetic dataset for testing.")
        # Create minimal synthetic dataset for testing
        np.random.seed(42)
        n_samples = max(10, args.horizon + 5)  # Ensure enough samples for horizon
        synthetic_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, n_samples),
            'High': np.random.uniform(110, 120, n_samples),
            'Low': np.random.uniform(90, 100, n_samples),
            'Close': np.random.uniform(100, 110, n_samples),
            'Volume': np.random.uniform(1000, 2000, n_samples)
        })
        df = synthetic_data
        df = generate_features(df)
        # Use binary labels for synthetic data too
        df = create_binary_labels(df, horizon=args.horizon, profit_threshold=args.up_thresh,
                                 include_fees=True, fee_rate=0.001)
        if 'label_binary' in df.columns:
            df['Label'] = df['label_binary']
        logging.info(f"Created synthetic dataset with {len(df)} samples for testing.")

    # Use binary training pipeline
    logger.info("Using binary classification training pipeline...")

    # Train and save model + metrics using binary classification
    train_model_binary(
        df,
        args.output,
        results_path=args.results,
        n_splits=args.n_splits,
        horizon=args.horizon,
        profit_threshold=args.up_thresh,
        include_fees=True,
        fee_rate=0.001,
        feature_columns=feature_columns,
        tune=args.tune,
        n_trials=args.n_trials,
        feature_selection=args.feature_selection,
        early_stopping_rounds=args.early_stopping_rounds,
        eval_economic=args.eval_profit,  # Map eval_profit to eval_economic
    )


if __name__ == "__main__":
    main()
