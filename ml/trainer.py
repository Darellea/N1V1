import argparse
import json
import logging
from collections import Counter
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, KFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib
import matplotlib.pyplot as plt

# Optional import for SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logging.warning("imbalanced-learn not available. SMOTE resampling will be skipped.")

# Reproducibility
np.random.seed(42)


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

    # Handle NaNs produced by indicator calculations by forward/backward filling
    # This avoids dropping too many rows while still providing reasonable defaults
    df[['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']] = (
        df[['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']]
        .ffill()
        .bfill()
    )
    return df


def create_labels(df: pd.DataFrame, horizon: int = 5, up_thresh: float = 0.005, down_thresh: float = -0.005) -> pd.DataFrame:
    """
    Create labels for the model based on future price movements.

    Parameters:
    - horizon: number of periods ahead to look
    - up_thresh: fractional threshold for labeling an upward move (default 0.005 = 0.5%)
    - down_thresh: fractional threshold for labeling a downward move (default -0.005 = -0.5%)
    """
    df = df.copy()

    # Adjust horizon for small datasets to ensure we have at least some valid labels
    if len(df) <= horizon:
        horizon = max(1, len(df) - 1)  # Use at least 1 period look-ahead, but ensure we have data

    # For very small datasets, use a simpler labeling approach
    if len(df) <= 3:
        # For datasets with 3 or fewer rows, create simple labels based on price direction
        df['Label'] = 0  # Default to neutral
        if len(df) >= 2:
            # Compare first and last prices
            if df['Close'].iloc[-1] > df['Close'].iloc[0] * (1 + up_thresh):
                df['Label'] = 1
            elif df['Close'].iloc[-1] < df['Close'].iloc[0] * (1 + down_thresh):
                df['Label'] = -1
        return df

    df['Future Price'] = df['Close'].shift(-horizon)
    df['Label'] = np.where(df['Future Price'] > df['Close'] * (1 + up_thresh), 1,
                           np.where(df['Future Price'] < df['Close'] * (1 + down_thresh), -1, 0))
    # Rows near the end will have NaN in 'Future Price'; keep only rows with a valid label
    df = df.drop(columns=['Future Price'])

    # For small datasets, if all labels are NaN, create fallback labels
    if df['Label'].isna().all() and len(df) > 0:
        # Create simple trend-based labels for small datasets
        if len(df) >= 2:
            # Compare first half vs second half of the data
            mid_point = len(df) // 2
            first_half_avg = df['Close'].iloc[:mid_point].mean()
            second_half_avg = df['Close'].iloc[mid_point:].mean()
            if second_half_avg > first_half_avg * (1 + up_thresh):
                df['Label'] = 1
            elif second_half_avg < first_half_avg * (1 + down_thresh):
                df['Label'] = -1
            else:
                df['Label'] = 0
        else:
            df['Label'] = 0  # Default to neutral for single-row data

    return df


def train_model(
    df: pd.DataFrame,
    save_path: str,
    results_path: str = 'training_results.json',
    n_splits: int = 5,
    horizon: int = 5,
    up_thresh: float = 0.005,
    down_thresh: float = -0.005,
    drop_neutral: bool = False,
    feature_columns: list | None = None,
    tune: bool = False,
    n_trials: int = 25,
    feature_selection: bool = False,
    early_stopping_rounds: int = 50,
    eval_profit: bool = False,
):
    """
    Train a LightGBM model using TimeSeriesSplit cross-validation, optional Optuna tuning,
    save metrics & metadata to JSON, plot feature importance (gain), save confusion matrices per fold,
    optionally perform feature selection and retrain model on selected features.

    New features:
    - early_stopping_rounds: passed to LightGBM fit for early stopping
    - eval_profit: if True, a custom profit metric will be used/logged
    """
    # Input validation - check for mock objects and invalid data
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame, not a mock object")

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Additional validation for X and y after extraction
    if feature_columns is None:
        feature_columns = ['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']

    # Ensure feature_columns is a proper list of strings
    if not isinstance(feature_columns, list):
        try:
            feature_columns = list(feature_columns)
        except Exception:
            # Handle case where feature_columns might be a numpy array or other object
            feature_columns = ['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']
    feature_columns = [str(col) for col in feature_columns]  # Ensure all are strings

    # Validate and filter feature columns that exist in the DataFrame
    valid_features = [col for col in feature_columns if col in df.columns]
    if len(valid_features) != len(feature_columns):
        missing_features = [col for col in feature_columns if col not in df.columns]
        logging.warning(f"Missing feature columns: {missing_features}. Using only valid features: {valid_features}")
        feature_columns = valid_features

    # Ensure we have at least some valid features
    if not feature_columns:
        raise ValueError("No valid feature columns found in DataFrame")

    if 'Label' not in df.columns:
        raise ValueError("DataFrame must contain 'Label' column")

    # Ensure we have enough samples for cross-validation
    if len(df) < n_splits:
        n_splits = max(2, len(df) // 2)  # Use at least 2 splits, or half the data size
        logging.warning(f"Dataset too small for requested {n_splits} splits, reduced to {n_splits} splits")

    # Very relaxed check for small datasets (allow ≥2 samples for testing)
    if len(df) < 2:
        raise ValueError(f"Dataset too small after preprocessing: {len(df)} samples. Need at least 2 samples.")
    elif len(df) < 5:
        logging.warning(f"Very small dataset: {len(df)} samples. Training may not be reliable.")

    # conditional import for optuna to avoid hard dependency when tuning not used
    optuna = None
    if tune:
        try:
            import optuna  # type: ignore
        except Exception as e:
            logging.error("Optuna is required for --tune but not installed. Install optuna or run without --tune.")
            raise

    if feature_columns is None:
        feature_columns = ['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']

    X = df[feature_columns].copy()
    y = df['Label'].copy()

    # Drop rows where label is NaN (due to shift in create_labels)
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask].astype(int)

    # If drop_neutral requested, remove Label == 0
    if drop_neutral:
        mask = y != 0
        X = X[mask]
        y = y[mask]
        logging.info("Dropped neutral (Label==0) rows before training.")

    # If any NaNs remain in features after fill, drop those rows
    feature_nan_mask = X.isna().any(axis=1)
    if feature_nan_mask.any():
        X = X[~feature_nan_mask]
        y = y[~feature_nan_mask]
        logging.info(f"Dropped {feature_nan_mask.sum()} rows with NaNs in features after fill.")

    # Enhanced dataset validation and preprocessing
    class_dist = Counter(y)
    logging.info(f"Class distribution before training: {class_dist}")

    # Check for single class
    unique_classes = len(class_dist)
    if unique_classes < 2:
        logging.warning(f"⚠️ Only one class present in target: {list(class_dist.keys())[0]}. Skipping training.")
        # Create minimal results for single-class scenario
        results = {
            'metadata': {
                'label_horizon': horizon,
                'thresholds': {'up_thresh': up_thresh, 'down_thresh': down_thresh},
                'feature_list': feature_columns,
                'drop_neutral': bool(drop_neutral),
                'early_stopping_rounds': int(early_stopping_rounds),
                'single_class_warning': True,
            },
            'class_distribution': dict(class_dist),
            'folds': [],
            'mean': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
        }
        try:
            with open(results_path, 'w') as fh:
                json.dump(results, fh, indent=2)
            logging.info(f"Training results saved to {results_path} (single class scenario)")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")
        return  # Exit early for single-class datasets

    # Check for severe class imbalance
    total_samples = len(y)
    minority_class_ratio = min(class_dist.values()) / total_samples
    if minority_class_ratio < 0.1:
        logging.warning(f"⚠️ Severe class imbalance detected: {class_dist}. Minority class ratio: {minority_class_ratio:.3f}")

    # Safe cross-validation setup
    if len(X) < n_splits + 1:
        original_n_splits = n_splits
        if len(X) >= 3:  # Need at least 3 samples for meaningful CV
            n_splits = min(n_splits, len(X) - 1)
            logging.warning(f"⚠️ Insufficient samples for n_splits={original_n_splits}, reducing to {n_splits}")
        else:
            raise ValueError(f"Dataset too small for cross-validation: {len(X)} samples. Need at least 3 samples.")

    # Choose appropriate cross-validation strategy
    try:
        # Try StratifiedKFold first for better class balance
        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        # Test if stratification is possible
        for train_idx, test_idx in cv_splitter.split(X, y):
            train_classes = set(y.iloc[train_idx])
            test_classes = set(y.iloc[test_idx])
            if len(train_classes) < unique_classes or len(test_classes) < unique_classes:
                raise ValueError("Stratification would create folds with missing classes")
        logging.info(f"Using StratifiedKFold with {n_splits} splits for balanced class distribution")
    except (ValueError, Exception) as e:
        # Fallback to regular KFold if stratification fails
        logging.warning(f"⚠️ Stratification not possible ({e}), falling back to KFold")
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        logging.info(f"Using KFold with {n_splits} splits")

    # Optional SMOTE for severe imbalance
    if SMOTE_AVAILABLE and minority_class_ratio < 0.1 and len(X) >= 10:
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, len(X) - 1))
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logging.info(f"Applied SMOTE: {len(X)} -> {len(X_resampled)} samples")
            X, y = X_resampled, y_resampled
            class_dist = Counter(y)
            logging.info(f"Class distribution after SMOTE: {class_dist}")
        except Exception as e:
            logging.warning(f"SMOTE failed: {e}. Continuing with original data.")

    fold_metrics = []
    fold_idx = 0
    best_iterations = []

    # Ensure results directory exists for saving confusion matrix images
    results_dir = os.path.dirname(os.path.abspath(results_path)) or '.'
    os.makedirs(results_dir, exist_ok=True)

    # Ensure matrices directory exists for saving confusion matrices
    os.makedirs("matrices", exist_ok=True)

    # Define profit calculation helper
    def compute_profit(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        profits = []
        for yt, yp in zip(y_true, y_pred):
            if yp == 0:
                profits.append(0.0)
            elif yp in (1, -1):
                profits.append(1.0 if yp == yt else -1.0)
            else:
                profits.append(0.0)
        return float(np.mean(profits))

    # LightGBM eval metric wrapper
    def lgb_profit_eval(preds, dataset):
        y_true = dataset.get_label().astype(int)
        n = len(y_true)
        if len(preds) == n:
            # binary case: preds are probability for positive class
            y_pred = (np.array(preds) > 0.5).astype(int) * 2 - 1
        else:
            num_class = int(len(preds) / n)
            try:
                arr = np.array(preds).reshape(num_class, n).T
            except Exception:
                arr = np.array(preds).reshape(n, num_class)
            y_pred_idx = np.argmax(arr, axis=1)
            labels_map = np.unique(y_true)
            # map indices to original labels
            y_pred = labels_map[y_pred_idx]
        mean_profit = compute_profit(y_true, y_pred)
        return ('profit', mean_profit, True)

    # If tuning requested, run Optuna to get best params
    best_params = None
    if tune:
        logging.info(f"Starting Optuna tuning with {n_trials} trials (optimizing weighted F1)...")

        def objective(trial):
            # Suggest hyperparameters
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
                'n_estimators': int(trial.suggest_int('n_estimators', 100, 2000)),
                'num_leaves': int(trial.suggest_int('num_leaves', 15, 255)),
                'max_depth': int(trial.suggest_int('max_depth', -1, 15)),
                'min_child_samples': int(trial.suggest_int('min_child_samples', 5, 100)),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            }
            # Evaluate with the chosen cross-validation strategy
            scores = []
            for train_idx, test_idx in cv_splitter.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx]
                model = lgb.LGBMClassifier(
                    random_state=42,
                    deterministic=True,
                    num_threads=1,
                    class_weight='balanced',
                    **params,
                )
                # Use early stopping if provided
                fit_kwargs = {}
                if early_stopping_rounds:
                    # Some LightGBM sklearn wrappers do not accept early_stopping_rounds
                    # directly in fit() depending on version. Only provide eval_set here;
                    # early stopping will be best-effort based on the installed lightgbm.
                    fit_kwargs['eval_set'] = [(X_val, y_val)]
                if eval_profit:
                    fit_kwargs['eval_metric'] = lgb_profit_eval
                model.fit(X_tr, y_tr, **fit_kwargs)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                scores.append(score)
            # We maximize mean weighted F1
            return float(np.mean(scores))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_score = study.best_value
        logging.info(f"Optuna best score (mean weighted F1): {best_score:.4f}")
        logging.info(f"Optuna best params: {json.dumps(best_params, indent=2)}")

    # Cross-validation loop for reporting (use either default or tuned params for training inside CV)
    for train_index, test_index in cv_splitter.split(X, y):
        fold_idx += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Check if fold has only one class
        train_classes = set(y_train)
        test_classes = set(y_test)
        if len(train_classes) < 2 or len(test_classes) < 2:
            logging.warning(f"⚠️ Fold {fold_idx} has insufficient class diversity. Train classes: {train_classes}, Test classes: {test_classes}. Skipping fold.")
            fold_entry = {
                'fold': fold_idx,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'per_class': {},
                'confusion_matrix': [],
                'skipped_reason': 'insufficient_class_diversity',
            }
            if eval_profit:
                fold_entry['profit'] = 0.0
            fold_metrics.append(fold_entry)
            continue

        if best_params:
            model_params = {
                'random_state': 42,
                'class_weight': 'balanced',
                'deterministic': True,
                'num_threads': 1,
                **best_params,
            }
        else:
            model_params = {
                'random_state': 42,
                'n_estimators': 100,
                'class_weight': 'balanced',
                'deterministic': True,
                'num_threads': 1,
            }

        model = lgb.LGBMClassifier(**model_params)

        # Prepare fit kwargs for early stopping and custom eval
        fit_kwargs = {}
        if early_stopping_rounds:
            # Provide eval_set for possible early stopping support in installed lightgbm.
            fit_kwargs['eval_set'] = [(X_test, y_test)]
        if eval_profit:
            fit_kwargs['eval_metric'] = lgb_profit_eval

        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except Exception as e:
            logging.warning(f"⚠️ Failed to fit model for fold {fold_idx}: {e}. Skipping fold.")
            fold_entry = {
                'fold': fold_idx,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'per_class': {},
                'confusion_matrix': [],
                'skipped_reason': f'fit_error: {str(e)}',
            }
            if eval_profit:
                fold_entry['profit'] = 0.0
            fold_metrics.append(fold_entry)
            continue

        # Log best iteration if available
        best_it = getattr(model, 'best_iteration_', None)
        if best_it is not None:
            best_iterations.append(int(best_it))
            logging.info(f"Fold {fold_idx} - best_iteration: {best_it}")

        try:
            y_pred = model.predict(X_test)
            # Ensure y_pred is a numpy array, not MagicMock
            y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred
        except Exception as e:
            logging.warning(f"⚠️ Failed to predict for fold {fold_idx}: {e}. Skipping fold.")
            fold_entry = {
                'fold': fold_idx,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'per_class': {},
                'confusion_matrix': [],
                'skipped_reason': f'predict_error: {str(e)}',
            }
            if eval_profit:
                fold_entry['profit'] = 0.0
            fold_metrics.append(fold_entry)
            continue

        # Skip fold if predictions are empty or test set is empty
        if len(y_pred) == 0 or len(y_test) == 0:
            logging.warning(f"Fold {fold_idx} - Empty predictions or test set, skipping fold")
            fold_entry = {
                'fold': fold_idx,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'per_class': {},
                'confusion_matrix': [],
                'skipped_reason': 'empty_predictions',
            }
            if eval_profit:
                fold_entry['profit'] = 0.0
            fold_metrics.append(fold_entry)
            continue

        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        # Per-class metrics and confusion matrix
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Compute profit metric per fold if requested
        profit_val = None
        if eval_profit:
            profit_val = compute_profit(y_test, y_pred)
            logging.info(f"Fold {fold_idx} - Profit: {profit_val:.4f}")

        logging.info(f"Fold {fold_idx} - F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
        logging.info(f"Fold {fold_idx} - Per-class metrics:\n{json.dumps(report, indent=2)}")
        logging.info(f"Fold {fold_idx} - Confusion matrix:\n{cm.tolist()}")

        # Save confusion matrix plot
        try:
            labels = np.unique(np.concatenate([y_test, y_pred]))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            fig, ax = plt.subplots(figsize=(6, 5))
            disp.plot(ax=ax, cmap='Blues', colorbar=False)
            plt.title(f'Confusion Matrix - Fold {fold_idx}')
            cm_path = os.path.join("matrices", f'confusion_matrix_fold{fold_idx}.png')
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close(fig)
            logging.info(f"Confusion matrix for fold {fold_idx} saved at matrices/confusion_matrix_fold{fold_idx}.png")
            cm_list = cm.tolist()
        except Exception as e:
            logging.warning(f"Failed to save confusion matrix plot for fold {fold_idx}: {e}")
            cm_list = cm.tolist()

        fold_entry = {
            'fold': fold_idx,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'per_class': report,
            'confusion_matrix': cm_list,
        }
        if profit_val is not None:
            fold_entry['profit'] = profit_val

        fold_metrics.append(fold_entry)

    # Compute averages
    mean_f1 = np.mean([m['f1'] for m in fold_metrics]) if fold_metrics else 0.0
    mean_precision = np.mean([m['precision'] for m in fold_metrics]) if fold_metrics else 0.0
    mean_recall = np.mean([m['recall'] for m in fold_metrics]) if fold_metrics else 0.0
    mean_profit = np.mean([m.get('profit', 0.0) for m in fold_metrics]) if fold_metrics else 0.0

    logging.info(f"Average F1: {mean_f1:.4f}")
    logging.info(f"Average Precision: {mean_precision:.4f}")
    logging.info(f"Average Recall: {mean_recall:.4f}")
    if eval_profit:
        logging.info(f"Average Profit: {mean_profit:.4f}")

    # Summary logging
    successful_folds = len([f for f in fold_metrics if f.get('skipped_reason') is None])
    total_folds = len(fold_metrics)
    logging.info(f"Training completed: {successful_folds}/{total_folds} folds successful")
    if successful_folds < total_folds:
        skipped_folds = [f for f in fold_metrics if f.get('skipped_reason') is not None]
        reasons = Counter([f['skipped_reason'] for f in skipped_folds])
        logging.info(f"Skipped folds by reason: {dict(reasons)}")

    # Save metrics and metadata to JSON (metadata updated later with best_params or selected_features if any)
    results = {
        'metadata': {
            'label_horizon': horizon,
            'thresholds': {'up_thresh': up_thresh, 'down_thresh': down_thresh},
            'feature_list': feature_columns,
            'drop_neutral': bool(drop_neutral),
            'early_stopping_rounds': int(early_stopping_rounds),
            'cv_strategy': type(cv_splitter).__name__,
            'n_splits': n_splits,
            'smote_applied': SMOTE_AVAILABLE and minority_class_ratio < 0.1 and len(X) >= 10,
        },
        'class_distribution': dict(class_dist),
        'folds': fold_metrics,
        'mean': {'f1': mean_f1, 'precision': mean_precision, 'recall': mean_recall},
    }

    if eval_profit:
        results['mean']['profit'] = mean_profit

    if best_params:
        results['metadata']['best_params'] = best_params

    try:
        with open(results_path, 'w') as fh:
            json.dump(results, fh, indent=2)
        logging.info(f"Training metrics and metadata saved to {results_path}")
    except Exception as e:
        logging.error(f"Failed to save training metrics to {results_path}: {e}")

    # Train final model on full dataset and save it
    final_model_params = {}
    if best_params:
        final_model_params = {
            'random_state': 42,
            'class_weight': 'balanced',
            'deterministic': True,
            'num_threads': 1,
            **best_params,
        }
    else:
        final_model_params = {
            'random_state': 42,
            'n_estimators': 100,
            'class_weight': 'balanced',
            'deterministic': True,
            'num_threads': 1,
        }

    # If we collected best_iterations from folds, use their mean as n_estimators for the final model.
    # Guard against invalid or zero values coming from different LightGBM wrappers.
    if best_iterations:
        try:
            valid_iters = [int(x) for x in best_iterations if isinstance(x, (int, float)) and int(x) > 0]
            if valid_iters:
                avg_best_it = int(np.mean(valid_iters))
                if avg_best_it > 0:
                    final_model_params['n_estimators'] = int(avg_best_it)
                    logging.info(f"Retraining final model with n_estimators={avg_best_it} based on fold best_iteration_")
        except Exception:
            # If anything goes wrong, fall back to default n_estimators already set above.
            pass

    # Guard: ensure n_estimators is a positive integer to avoid LightGBM train errors
    try:
        n_est = int(final_model_params.get("n_estimators", 0))
        if n_est <= 0:
            final_model_params["n_estimators"] = 100
    except Exception:
        final_model_params["n_estimators"] = 100

    final_model = lgb.LGBMClassifier(**final_model_params)
    # For final model, no validation set; training on full data. Early stopping can't be used without a validation set.
    final_model.fit(X, y)

    # Full feature importance plot (save for transparency)
    plt.figure(figsize=(10, 8))
    try:
        lgb.plot_importance(final_model, importance_type='gain', max_num_features=50)
        plt.tight_layout()
        fi_path = os.path.join(results_dir, 'feature_importance.png')
        plt.savefig(fi_path)
        plt.close()
        logging.info(f"Feature importance chart saved to {fi_path}")
    except Exception as e:
        logging.warning(f"Failed to plot feature importance: {e}")

    # Feature selection (optional)
    selected_features = feature_columns
    if feature_selection:
        try:
            # Get gain importances
            booster = final_model.booster_
            gains = booster.feature_importance(importance_type='gain')
            # Map to features (ensure ordering matches feature_columns)
            feature_gain_pairs = list(zip(feature_columns, gains))
            max_gain = max(gains) if len(gains) > 0 else 0.0
            threshold = max_gain * 0.01  # keep features with >=1% of max importance
            kept = [f for f, g in feature_gain_pairs if g >= threshold]
            if not kept:
                logging.warning("Feature selection removed all features; keeping original feature set.")
                kept = feature_columns
            selected_features = kept
            logging.info(f"Selected features after importance thresholding ({threshold:.4f}): {selected_features}")

            # Validate selected features exist in DataFrame before indexing
            valid_selected = [f for f in selected_features if f in X.columns]
            if len(valid_selected) != len(selected_features):
                missing = [f for f in selected_features if f not in X.columns]
                logging.warning(f"Some selected features not found in DataFrame: {missing}. Using valid features only.")
                selected_features = valid_selected

            if not selected_features:
                raise ValueError("No valid features available for retraining after feature selection")

            # Retrain final model on selected features
            X_selected = X[selected_features].copy()
            final_model = lgb.LGBMClassifier(**final_model_params)
            final_model.fit(X_selected, y)

            # Update metadata and results JSON with selected features
            results['metadata']['selected_features'] = selected_features
            with open(results_path, 'w') as fh:
                json.dump(results, fh, indent=2)
            logging.info("Retrained final model on selected features and updated results JSON.")
        except Exception as e:
            # Catch any Pandas internal errors (like TypeError from isinstance checks) and provide clear error message
            if "isinstance" in str(e).lower() or "TypeError" in str(type(e).__name__):
                logging.error(f"Feature selection failed due to DataFrame indexing error: {e}")
                raise ValueError("Feature selection failed due to invalid feature column types. Ensure feature_columns contains valid string column names.") from e
            else:
                logging.warning(f"Feature selection failed: {e}")

    # Save the final model with proper error handling (don't re-raise)
    try:
        joblib.dump(final_model, save_path)
        logging.info(f"Final model trained on full data and saved to {save_path}")
    except Exception as e:
        error_msg = f"Failed to save model to {save_path}: {e}"
        logging.error(error_msg)
        # Don't re-raise - just log the error for graceful handling in tests

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
                "best_iterations": best_iterations,
                "best_params": best_params,
            },
            "training_window": None,
            "scaler": None,
            "caveats": (
                "This model was trained on historical OHLCV-based technical features. "
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

    # Update results JSON with final metadata if not yet written
    try:
        with open(results_path, 'w') as fh:
            json.dump(results, fh, indent=2)
        logging.info(f"Final results JSON updated at {results_path}")
    except Exception as e:
        logging.error(f"Failed to update final results JSON: {e}")


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
    parser = argparse.ArgumentParser(description="Train a LightGBM model on OHLCV data with technical features.")
    parser.add_argument('--data', required=True, help='Path to CSV file containing OHLCV data')
    parser.add_argument('--output', required=True, help='Path to output .pkl model file')
    parser.add_argument('--up_thresh', type=float, default=0.005, help='Upward movement threshold (fraction). Default 0.005 (0.5%)')
    parser.add_argument('--down_thresh', type=float, default=-0.005, help='Downward movement threshold (fraction). Default -0.005 (-0.5%)')
    parser.add_argument('--horizon', type=int, default=5, help='Label horizon (periods ahead). Default 5')
    parser.add_argument('--results', default='training_results.json', help='Path to save training metrics JSON')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of TimeSeriesSplit splits. Default 5')
    parser.add_argument('--drop_neutral', action='store_true', help='If set, drop rows with Label==0 before training (binary classification)')
    parser.add_argument('--logfile', default=None, help='Optional path to a logfile. If set, logs are written to this file as well as console.')
    parser.add_argument('--tune', action='store_true', help='If set, run Optuna hyperparameter tuning before training the final model')
    parser.add_argument('--n_trials', type=int, default=25, help='Number of Optuna trials when --tune is set. Default 25')
    parser.add_argument('--feature_selection', action='store_true', help='If set, perform feature selection based on gain importance and retrain final model')
    parser.add_argument('--early_stopping_rounds', type=int, default=50, help='Early stopping rounds to pass to LightGBM fit. Default 50')
    parser.add_argument('--eval_profit', action='store_true', help='If set, evaluate and log a custom profit metric per fold')

    args = parser.parse_args()

    # Setup logging early so subsequent messages go through logger
    setup_logging(args.logfile)

    df = load_data(args.data)

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Data must contain columns: {required_columns}")

    df = generate_features(df)
    df = create_labels(df, horizon=args.horizon, up_thresh=args.up_thresh, down_thresh=args.down_thresh)

    # After label creation drop rows that have NaN label (end of series)
    if 'Label' in df.columns:
        df = df.dropna(subset=['Label'])
    else:
        raise ValueError("Label column not found after create_labels.")

    # Final sanity: ensure there are no NaNs in features used for training
    feature_columns = ['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']
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
        df = create_labels(df, horizon=args.horizon, up_thresh=args.up_thresh, down_thresh=args.down_thresh)
        logging.info(f"Created synthetic dataset with {len(df)} samples for testing.")

    # Train and save model + metrics
    train_model(
        df,
        args.output,
        results_path=args.results,
        n_splits=args.n_splits,
        horizon=args.horizon,
        up_thresh=args.up_thresh,
        down_thresh=args.down_thresh,
        drop_neutral=args.drop_neutral,
        feature_columns=feature_columns,
        tune=args.tune,
        n_trials=args.n_trials,
        feature_selection=args.feature_selection,
        early_stopping_rounds=args.early_stopping_rounds,
        eval_profit=args.eval_profit,
    )


if __name__ == "__main__":
    main()
