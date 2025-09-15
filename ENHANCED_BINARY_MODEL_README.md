# Enhanced Binary Model Training System

This document describes the comprehensive improvements implemented to enhance binary trading model performance following the retraining guide principles.

## Overview

The enhanced binary model training system implements all key recommendations from the retraining guide:

1. **Data Quality & Labeling** - Outlier removal, enhanced labeling strategies
2. **Feature Engineering** - Multi-horizon, regime-aware, and interaction features
3. **Sample Weighting** - Profit-impact based weighting strategies
4. **Hyperparameter Optimization** - Optuna-based Bayesian optimization
5. **Calibration & Thresholding** - Isotonic regression calibration with optimal threshold selection
6. **Walk-Forward Validation** - Robust out-of-sample testing
7. **Monitoring & Auto-Recalibration** - Continuous performance monitoring with automatic retraining

## Key Improvements

### 1. Enhanced Feature Engineering

#### Multi-Horizon Features
```python
# Returns over multiple time horizons
df['return_1'] = df['Close'].pct_change(1)
df['return_3'] = df['Close'].pct_change(3)
df['return_5'] = df['Close'].pct_change(5)
df['return_24'] = df['Close'].pct_change(24)

# Rolling statistics
df['volatility_5'] = df['Close'].rolling(window=5).std()
df['mean_return_5'] = df['Close'].pct_change().rolling(window=5).mean()
df['skew_5'] = df['Close'].pct_change().rolling(window=5).skew()
df['kurtosis_5'] = df['Close'].pct_change().rolling(window=5).kurt()
```

#### Regime-Aware Features
```python
# Bollinger Bands
df['BB_middle'] = df['Close'].rolling(window=20).mean()
df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

# ATR-normalized returns
df['atr_normalized_return'] = df['return_1'] / df['ATR']

# Volume z-score
df['volume_zscore'] = (df['Volume'] - df['volume_sma']) / df['volume_std']

# ADX (Average Directional Index)
df['ADX'] = compute_adx(df, period=14)
```

#### Interaction Features
```python
# Momentum-volatility interaction
df['momentum_volatility'] = df['return_1'] * df['Volatility']

# Trend-volume interaction
df['trend_volume'] = df['TrendStrength'] * df['volume_zscore']

# Technical indicator interactions
df['rsi_macd'] = df['RSI'] * df['MACD']
df['atr_trend'] = df['ATR'] * df['TrendStrength']
```

### 2. Advanced Sample Weighting

#### Profit-Impact Based Weighting
```python
def create_sample_weights(df, method='combined'):
    if method == 'profit_impact':
        # Weight by expected profit impact
        profit_magnitude = np.abs(df['forward_return'])
        weights = 1 + profit_magnitude / profit_magnitude.max()

    elif method == 'combined':
        # Combine class balance and profit impact
        class_weights = create_class_weights(df)
        profit_weights = create_profit_weights(df)
        weights = (class_weights + profit_weights) / 2
```

### 3. Optuna Hyperparameter Optimization

```python
def optimize_hyperparameters_optuna(X_train, y_train, n_trials=25):
    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        }
        # Optimization logic with cross-validation
        return auc_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
```

### 4. Feature Selection

```python
def perform_feature_selection(X, y, method='gain_importance', top_k=20):
    if method == 'gain_importance':
        model = lgb.LGBMClassifier()
        model.fit(X, y)
        importance_scores = model.feature_importances_
        top_indices = np.argsort(importance_scores)[-top_k:]
        return [X.columns[i] for i in top_indices]

    elif method == 'shap':
        # SHAP-based feature selection
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(feature_importance)[-top_k:]
        return [X.columns[i] for i in top_indices]
```

### 5. Probability Calibration & Optimal Thresholding

```python
# Calibrate probabilities
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(calibration_probabilities, calibration_true_labels)
calibrated_model = CalibratedModel(final_model, calibrator)

# Optimize threshold for maximum Sharpe ratio
thresholds = np.arange(0.5, 0.95, 0.05)
best_threshold = max(thresholds, key=lambda t: calculate_sharpe_at_threshold(t))
```

### 6. Model Monitoring & Auto-Recalibration

```python
# Setup monitoring
monitor_config = {
    'model_path': 'models/enhanced_model.pkl',
    'monitoring_window_days': 30,
    'alerts': {
        'performance_threshold': 0.6,
        'drift_threshold': 0.7
    }
}

monitor = create_model_monitor(monitor_config)
recalibrator = create_auto_recalibrator(monitor_config)

# Start monitoring
recalibrator.start()
```

## Usage Examples

### Basic Enhanced Training

```bash
python examples/enhanced_binary_training_example.py \
    --data historical_data.csv \
    --output models/enhanced_model.pkl
```

### Training with Monitoring

```bash
python examples/enhanced_binary_training_example.py \
    --data historical_data.csv \
    --output models/enhanced_model.pkl \
    --enable-monitoring \
    --retraining-data recent_data.csv
```

### Custom Configuration

```bash
python examples/enhanced_binary_training_example.py \
    --data historical_data.csv \
    --output models/enhanced_model.pkl \
    --config config/custom_training_config.json
```

## Configuration Options

### Training Configuration

```json
{
  "data_quality": {
    "remove_outliers": true,
    "outlier_method": "iqr",
    "outlier_multiplier": 1.5
  },
  "feature_engineering": {
    "include_multi_horizon": true,
    "include_regime_features": true,
    "include_interaction_features": true,
    "feature_selection": true,
    "top_features": 25
  },
  "sample_weighting": {
    "method": "combined",
    "profit_col": "forward_return"
  },
  "hyperparameter_tuning": {
    "enabled": true,
    "n_trials": 25,
    "optimization_metric": "auc"
  },
  "model_training": {
    "n_splits": 5,
    "early_stopping_rounds": 50,
    "eval_economic": true
  },
  "monitoring": {
    "enabled": true,
    "monitoring_window_days": 30,
    "monitor_interval_minutes": 60,
    "alerts": {
      "performance_threshold": 0.6,
      "drift_threshold": 0.7
    }
  }
}
```

## Performance Metrics

The enhanced system tracks comprehensive performance metrics:

### Classification Metrics
- AUC (Area Under ROC Curve)
- F1 Score
- Precision/Recall
- Confusion Matrix

### Economic Metrics
- Sharpe Ratio (annualized)
- Maximum Drawdown
- Total Return
- Win Rate
- Profit Factor
- Total Trades

### Calibration Metrics
- Brier Score
- Calibration Error
- Reliability Diagrams

## Output Files

The enhanced training system generates:

1. **Model File**: `models/enhanced_model.pkl`
2. **Configuration**: `models/enhanced_model_config.json`
3. **Training Results**: `models/enhanced_model_results.json`
4. **Training Report**: `models/enhanced_model_report.json`
5. **Visualizations**: `models/enhanced_model_plots/`
   - Feature importance plots
   - Fold performance plots
   - Confusion matrix
6. **Monitoring Data**: `monitoring/`
   - Performance history
   - Drift detection results
   - Alert logs

## Monitoring Dashboard

When monitoring is enabled, the system provides:

- **Real-time Performance Tracking**: AUC, Sharpe ratio, win rate
- **Drift Detection**: Feature drift, prediction drift, label drift
- **Health Assessment**: Overall model health score
- **Automated Alerts**: Performance degradation, drift detection
- **Auto-Recalibration**: Automatic model retraining when needed

## Best Practices

### Data Preparation
1. Ensure high-quality OHLCV data with no missing values
2. Remove extreme outliers that could skew training
3. Validate timestamp alignment and forward-looking bias prevention

### Feature Engineering
1. Start with basic technical indicators
2. Add multi-horizon features for temporal patterns
3. Include regime-aware features for market condition adaptation
4. Use interaction features sparingly to avoid overfitting

### Model Training
1. Always use walk-forward validation for realistic performance estimation
2. Enable hyperparameter tuning for optimal parameter selection
3. Use feature selection to reduce dimensionality and improve stability
4. Apply appropriate sample weighting for imbalanced datasets

### Monitoring & Maintenance
1. Enable monitoring from day one of deployment
2. Set appropriate alert thresholds based on your risk tolerance
3. Regularly review monitoring reports and adjust thresholds as needed
4. Keep retraining data current and representative of live market conditions

## Troubleshooting

### Common Issues

1. **Poor Performance**: Check data quality, feature engineering, and hyperparameter tuning
2. **Overfitting**: Reduce model complexity, increase regularization, use more data
3. **Drift Detection**: False positives may indicate noisy data or inappropriate thresholds
4. **Calibration Issues**: Check probability distributions and consider different calibration methods

### Performance Optimization

1. **Speed**: Reduce Optuna trials, use feature selection, optimize data preprocessing
2. **Memory**: Process data in batches, use appropriate data types, clear intermediate results
3. **Accuracy**: Increase training data, improve feature engineering, fine-tune hyperparameters

## Integration with Existing Systems

The enhanced training system is designed to integrate seamlessly with existing trading infrastructure:

```python
# Load enhanced model
model = joblib.load('models/enhanced_model.pkl')
config = json.load(open('models/enhanced_model_config.json'))

# Make predictions with optimal threshold
features = prepare_features(new_data)
probabilities = model.predict_proba(features)[:, 1]
predictions = (probabilities >= config['optimal_threshold']).astype(int)

# Update monitoring
monitor.update_predictions(features, probabilities, true_labels)
```

## Future Enhancements

Planned improvements include:

1. **Ensemble Methods**: Combine multiple models for improved stability
2. **Meta-Labeling**: Two-stage prediction for execution decisions
3. **Online Learning**: Incremental model updates without full retraining
4. **Advanced Drift Detection**: More sophisticated statistical tests
5. **Automated Feature Discovery**: ML-based feature generation
6. **Multi-Asset Support**: Cross-asset validation and transfer learning

## Support

For questions or issues with the enhanced binary model training system:

1. Check the troubleshooting section above
2. Review the example scripts and configuration files
3. Examine the monitoring reports for insights
4. Ensure all dependencies are properly installed

## License

This enhanced training system is part of the broader trading framework and follows the same licensing terms.
