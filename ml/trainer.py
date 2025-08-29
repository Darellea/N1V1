import argparse
import json
import logging
from collections import Counter
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
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
    df['Future Price'] = df['Close'].shift(-horizon)
    df['Label'] = np.where(df['Future Price'] > df['Close'] * (1 + up_thresh), 1,
                           np.where(df['Future Price'] < df['Close'] * (1 + down_thresh), -1, 0))
    # Rows near the end will have NaN in 'Future Price'; keep only rows with a valid label
    df = df.drop(columns=['Future Price'])
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
        logging.info("Dropped rows with NaNs in features after fill.")

    # Print class distribution
    class_dist = Counter(y)
    logging.info(f"Class distribution before training: {class_dist}")

    # Prepare cross-validation
    tss = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []
    fold_idx = 0
    best_iterations = []

    # Ensure results directory exists for saving confusion matrix images
    results_dir = os.path.dirname(os.path.abspath(results_path)) or '.'
    os.makedirs(results_dir, exist_ok=True)

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
            # Evaluate with TimeSeriesSplit
            scores = []
            for train_idx, test_idx in tss.split(X):
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
                    fit_kwargs['eval_set'] = [(X_val, y_val)]
                    fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
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
    for train_index, test_index in tss.split(X):
        fold_idx += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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
            fit_kwargs['eval_set'] = [(X_test, y_test)]
            fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
        if eval_profit:
            fit_kwargs['eval_metric'] = lgb_profit_eval

        model.fit(X_train, y_train, **fit_kwargs)

        # Log best iteration if available
        best_it = getattr(model, 'best_iteration_', None)
        if best_it is not None:
            best_iterations.append(int(best_it))
            logging.info(f"Fold {fold_idx} - best_iteration: {best_it}")

        y_pred = model.predict(X_test)
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
            cm_path = os.path.join(results_dir, f'confusion_matrix_fold{fold_idx}.png')
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close(fig)
            logging.info(f"Confusion matrix plot saved to {cm_path}")
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

    # Save metrics and metadata to JSON (metadata updated later with best_params or selected_features if any)
    results = {
        'metadata': {
            'label_horizon': horizon,
            'thresholds': {'up_thresh': up_thresh, 'down_thresh': down_thresh},
            'feature_list': feature_columns,
            'drop_neutral': bool(drop_neutral),
            'early_stopping_rounds': int(early_stopping_rounds),
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

    # If we collected best_iterations from folds, use their mean as n_estimators for the final model
    if best_iterations:
        try:
            avg_best_it = int(np.mean(best_iterations))
            final_model_params['n_estimators'] = int(avg_best_it)
            logging.info(f"Retraining final model with n_estimators={avg_best_it} based on fold best_iteration_")
        except Exception:
            pass

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
            logging.warning(f"Feature selection failed: {e}")

    # Save the final model
    joblib.dump(final_model, save_path)
    logging.info(f"Final model trained on full data and saved to {save_path}")

    # Update results JSON with final metadata if not yet written
    try:
        with open(results_path, 'w') as fh:
            json.dump(results, fh, indent=2)
        logging.info(f"Final results JSON updated at {results_path}")
    except Exception as e:
        logging.error(f"Failed to update final results JSON: {e}")


def setup_logging(logfile: str | None = None, level: int = logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    # Clear existing handlers
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logging.info(f"Logging to file enabled: {logfile}")


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
