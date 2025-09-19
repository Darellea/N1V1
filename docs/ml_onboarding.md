# ML Pipeline Onboarding Guide

This guide covers the production-grade ML pipeline for N1V1, including advanced ensemble methods, comprehensive monitoring, drift detection, and full reproducibility guarantees.

## Overview

The ML pipeline delivers institutional-grade reliability with advanced features:

### Core Capabilities
- **Ensemble Methods**: LightGBM + XGBoost + Neural Networks with stacking
- **Advanced Features**: Cross-asset correlations, time-anchored features, drift detectors
- **Production Monitoring**: Real-time drift detection, performance tracking, automated alerts
- **Full Reproducibility**: Deterministic seeding, environment capture, Git versioning
- **Dataset Versioning**: Immutable dataset management with hashing
- **Calibration & Uncertainty**: Platt scaling, isotonic regression, conformal prediction

### Key Features
- Deterministic seeding across all libraries (NumPy, pandas, scikit-learn, PyTorch, TensorFlow, LightGBM, XGBoost, CatBoost)
- Environment snapshot capture (packages, system info, Git state, hardware specs)
- Systematic benchmarking with regression detection (F1 ≥0.70 target)
- MLflow model registry integration with local fallbacks
- Comprehensive experiment tracking and metadata storage
- Automated drift detection with Kolmogorov-Smirnov and Population Stability Index
- Ensemble model uncertainty estimation and calibration

## Training Pipeline

### Running Training

```bash
# Basic training
python ml/train.py --data data.csv --config config.json --output results.json

# Training with specific symbol and verbose logging
python ml/train.py --data data.csv --symbol BTC --config config.json --output results.json --verbose

# Training with custom minimum samples
python ml/train.py --data data.csv --min-samples 2000 --config config.json --output results.json
```

### Reproducibility Features

The training pipeline automatically:

1. **Sets deterministic seeds** for all ML libraries using seed `42`
2. **Captures environment snapshot** including:
   - Python version and platform details
   - Package versions for ML libraries
   - System hardware information
   - Git repository state (commit, branch, remote)
   - Environment variables (filtered for safety)

3. **Logs comprehensive metadata** in `experiments/train_*/metadata.json`

### Experiment Tracking

Each training run creates an experiment directory with:

```
experiments/train_all_20250919_093000/
├── metadata.json          # Complete experiment metadata
├── parameters.json        # Training parameters
├── artifacts/
│   ├── data.csv          # Copy of input data
│   └── config.json       # Configuration used
└── logs/                 # Training logs
```

## Benchmarking Pipeline

### Running Benchmarks

```bash
# Benchmark all models in a directory
python scripts/run_model_benchmarks.py --model-dir models/ --validation-data validation.csv --output benchmark_results.json

# Benchmark with symbol filtering
python scripts/run_model_benchmarks.py --model-dir models/ --validation-data validation.csv --symbol BTC --output benchmark_results.json

# Compare against previous benchmarks
python scripts/run_model_benchmarks.py --model-dir models/ --validation-data validation.csv --history previous_benchmarks.json --output benchmark_results.json
```

### Benchmarking Features

The benchmarking script provides:

1. **Comprehensive metrics**:
   - F1 score, precision, recall, accuracy
   - AUC (Area Under Curve) for probability-based models
   - Confusion matrix components
   - Prediction latency (ms per sample)

2. **Regression detection**:
   - Compares current vs previous benchmark results
   - Flags performance degradation >5% threshold
   - Reports on latency increases

3. **Target validation**:
   - F1 score target: ≥0.70 for buy signals
   - Latency target: ≤100ms per sample

### Benchmark Report Structure

```json
{
  "timestamp": "2025-09-19T09:30:00",
  "summary": {
    "total_models": 3,
    "successful_benchmarks": 3,
    "failed_benchmarks": 0,
    "models_with_regressions": 0
  },
  "benchmarks": [...],
  "regressions": {...},
  "performance_summary": {
    "avg_f1_score": 0.78,
    "max_f1_score": 0.82,
    "target_achievement": {
      "f1_buy_target_met": true,
      "latency_target_met": true
    }
  }
}
```

## Model Registry

### MLflow Integration

Models are automatically logged to MLflow during training:

```python
# Automatic logging during training
log_to_mlflow(experiment_name, parameters, metrics, models, artifacts)
```

### Loading Models

The system supports flexible model loading with fallbacks:

```python
from ml.model_loader import load_model_with_fallback

# Try registry first, fallback to local
model, card = load_model_with_fallback("my_model")

# Load specific version from registry
model, card = load_model_from_registry("my_model", version="1")

# Load from local file
model, card = load_model_with_card("models/my_model.pkl")
```

### Registry Features

1. **Automatic model versioning** during training
2. **Environment snapshot storage** with each model
3. **Fallback to local files** if registry unavailable
4. **Model card support** for metadata and documentation

## Reproducing Experiments

### Same Environment Setup

To reproduce any experiment:

1. **Check environment compatibility**:
   ```bash
   # Compare Python/package versions
   python -c "import sys; print(sys.version)"
   pip list | grep -E "(numpy|pandas|scikit-learn|torch|tensorflow)"
   ```

2. **Use same random seed** (automatically set to 42)

3. **Verify Git state**:
   ```bash
   git checkout <commit_hash>
   git status  # Should show clean working directory
   ```

### Reproducing from Metadata

```python
import json
from ml.train import set_deterministic_seeds, capture_environment_snapshot

# Load experiment metadata
with open('experiments/train_20250919_093000/metadata.json', 'r') as f:
    metadata = json.load(f)

# Verify environment matches
current_env = capture_environment_snapshot()
stored_env = metadata['parameters']['environment_snapshot']

# Check critical differences
if current_env['python_version'] != stored_env['python_version']:
    print("WARNING: Python version mismatch")

# Reproduce with same parameters
set_deterministic_seeds(metadata['parameters']['random_seed'])
```

## Testing

### Reproducibility Tests

```bash
# Test deterministic seeding
pytest tests/ml/test_train.py::TestReproducibility -v

# Test environment capture
pytest tests/ml/test_train.py::TestReproducibility::test_capture_environment_snapshot -v

# Test model registry fallback
pytest tests/ml/test_train.py::TestModelLoaderReproducibility -v
```

### Benchmarking Tests

```bash
# Test benchmark computation
pytest tests/ml/test_train.py::TestConfusionMatrixGeneration -v
```

## Configuration

### Training Configuration

Key configuration options in `config.json`:

```json
{
  "predictive_models": {
    "enabled": true,
    "models": ["price_predictor", "signal_classifier"]
  },
  "mlflow": {
    "tracking_uri": "file:./mlruns",
    "experiment_name": "n1v1_models"
  }
}
```

### Benchmarking Configuration

Benchmarks use built-in thresholds (configurable in `scripts/run_model_benchmarks.py`):

```python
BENCHMARK_CONFIG = {
    'thresholds': {
        'f1_buy_target': 0.70,
        'max_latency_ms': 100,
        'regression_threshold': 0.05
    }
}
```

## Troubleshooting

### Common Issues

1. **Non-deterministic results**:
   - Ensure all seeds are set (check `set_deterministic_seeds`)
   - Verify no external randomness sources
   - Check for threading/multiprocessing issues

2. **MLflow connection failures**:
   - Falls back to local storage automatically
   - Check MLflow server status if using remote registry

3. **Environment mismatches**:
   - Compare package versions between training and deployment
   - Use virtual environments with pinned dependencies

4. **Benchmarking failures**:
   - Ensure validation data has required columns
   - Check for missing true labels in validation data

### Performance Optimization

1. **Training speed**:
   - Use parallel processing for multi-symbol data
   - Enable GPU acceleration if available
   - Optimize data loading with chunked reading

2. **Benchmarking speed**:
   - Use batch prediction for large datasets
   - Parallelize model benchmarking
   - Cache preprocessed validation data

## Best Practices

1. **Always use version control** for code and data
2. **Document environment changes** that affect model performance
3. **Regular benchmarking** against validation sets
4. **Monitor for regressions** in CI/CD pipelines
5. **Backup model artifacts** and experiment metadata
6. **Use consistent random seeds** across experiments
7. **Validate environment compatibility** before deployment

## Advanced Features

### Ensemble Methods

The system supports multiple ensemble strategies for improved performance and uncertainty estimation:

#### Ensemble Training

```python
from ml.trainer import train_ensemble_model, create_ensemble_model

# Create ensemble with default models
ensemble = create_ensemble_model()

# Train ensemble
results = train_ensemble_model(
    df=df,
    save_path='models/ensemble_model.pkl',
    feature_columns=['RSI', 'MACD', 'volume_ma_7d'],
    n_splits=5
)
```

#### Ensemble Configuration

```json
{
  "ensemble": {
    "enabled": true,
    "models": {
      "lightgbm": {
        "weight": 1.0,
        "params": {"num_leaves": 31, "learning_rate": 0.05}
      },
      "xgboost": {
        "weight": 0.9,
        "params": {"max_depth": 6, "learning_rate": 0.1}
      },
      "neural_net": {
        "weight": 0.8,
        "params": {"hidden_layers": [64, 32], "dropout": 0.2}
      }
    },
    "meta_model": "logistic_regression",
    "calibration": "isotonic"
  }
}
```

#### Uncertainty Estimation

```python
# Get predictions with uncertainty
predictions = ensemble.predict_proba(X_test)
uncertainty = ensemble.estimate_uncertainty(X_test)

print(f"Mean prediction: {uncertainty['mean_proba']:.3f}")
print(f"Uncertainty: {uncertainty['uncertainty']:.3f}")
print(f"Confidence: {uncertainty['confidence']:.3f}")
```

### Advanced Feature Engineering

#### Cross-Asset Features

```python
from ml.features import generate_cross_asset_features

# Generate cross-asset features
cross_features = generate_cross_asset_features(multi_asset_df)

# Features include:
# - BTC_ETH_correlation_7d, BTC_ETH_correlation_14d
# - BTC_ETH_spread, BTC_ETH_ratio
# - Rolling correlations and spreads
```

#### Time-Anchored Features

```python
from ml.features import generate_time_anchored_features

# Generate time-anchored features
time_features = generate_time_anchored_features(price_df)

# Features include:
# - volatility_7d, volatility_14d, volatility_30d
# - momentum_7d, momentum_14d, momentum_30d
# - volume_ma_7d, volume_zscore_7d
```

#### Drift Detection

```python
from ml.features import detect_feature_drift, FeatureDriftDetector

# Detect drift between reference and current data
drift_detected, drift_scores = detect_feature_drift(
    reference_data, current_data, method='ks'
)

# Use drift detector class
detector = FeatureDriftDetector(method='psi', threshold=0.1)
drift_scores = detector.detect_drift(reference_data, current_data)
```

### Model Monitoring

#### Setting Up Monitoring

```python
from ml.model_monitor import ModelMonitor, create_model_monitor

# Create monitor configuration
config = {
    'model_path': 'models/binary_model.pkl',
    'config_path': 'models/binary_model_config.json',
    'monitoring_window_days': 30,
    'drift_thresholds': {'overall_threshold': 0.1},
    'alerts': {
        'performance_threshold': 0.6,
        'drift_threshold': 0.7
    }
}

# Create and start monitoring
monitor = create_model_monitor(config)
monitor.start_monitoring()
```

#### Updating Predictions

```python
# Update monitor with new predictions
monitor.update_predictions(
    features=X_batch,
    predictions=y_pred_proba,
    true_labels=y_true,
    timestamp=datetime.now()
)
```

#### Health Assessment

```python
# Get comprehensive health report
health_report = monitor.check_model_health()

print(f"Overall health: {health_report.overall_health_score:.3f}")
print(f"Performance score: {health_report.performance_score:.3f}")
print(f"Drift score: {health_report.drift_score:.3f}")
print(f"Recommendations: {health_report.recommendations}")

if health_report.requires_retraining:
    print("Model requires retraining!")
```

#### Generating Reports

```python
# Generate monitoring report
report = monitor.generate_report()

# Save report to file
monitor.generate_report(output_path='monitoring/report.json')
```

### Dataset Versioning

#### Creating Dataset Versions

```python
from data.dataset_versioning import DatasetVersionManager

# Initialize version manager
version_manager = DatasetVersionManager(storage_dir='data/versions/')

# Create new dataset version
version_id = version_manager.create_version(
    data=df,
    metadata={
        'source': 'historical_data.csv',
        'date_range': '2023-01-01 to 2023-12-31',
        'features': list(df.columns)
    }
)

print(f"Created dataset version: {version_id}")
```

#### Loading Dataset Versions

```python
# Load specific version
data, metadata = version_manager.load_version(version_id)

# List all versions
versions = version_manager.list_versions()

# Get version info
info = version_manager.get_version_info(version_id)
```

### Calibration Methods

#### Probability Calibration

```python
from ml.trainer import CalibratedModel
from sklearn.calibration import CalibratedClassifierCV

# Train base model
base_model = lgb.LGBMClassifier()
base_model.fit(X_train, y_train)

# Calibrate probabilities
calibrator = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model = CalibratedModel(base_model, calibrator)

# Get calibrated predictions
calibrated_proba = calibrated_model.predict_proba(X_test)
```

#### Threshold Optimization

```python
from ml.trainer import optimize_threshold

# Optimize decision threshold
calibrated_model, optimal_threshold, results = optimize_threshold(
    X=X_train,
    y=y_train,
    model_params=lgb_params,
    class_weights=class_weights,
    n_splits=5,
    profit_threshold=0.005
)

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Expected Sharpe: {results['sharpe']:.3f}")
```

## Monitoring and Alerting

### Alert Rules

The system includes comprehensive alerting for:

- **Model Drift**: PSI > 0.2, triggers warning
- **Feature Drift**: KS statistic > 0.15, triggers warning
- **Performance Degradation**: Score < 0.6, triggers warning
- **Calibration Error**: Brier score > 0.3, triggers warning
- **Retraining Required**: Health assessment indicates retraining needed

### Prometheus Metrics

```yaml
# Key metrics exposed
model_drift_score{feature="overall"}  # Overall drift score
model_performance_score  # Current performance score
model_calibration_error  # Calibration error
model_prediction_latency_seconds  # Prediction latency
ensemble_model_disagreement  # Ensemble disagreement score
```

### Alert Configuration

```yaml
# Example alert rule
- alert: ModelDriftDetected
  expr: model_drift_score > 0.2
  for: 30m
  labels:
    severity: warning
    category: model
  annotations:
    summary: "Model drift detected"
    description: "Model drift score: {{ $value }} (threshold: 0.2)"
```

## Reproducibility Checklist

### Pre-Training Setup

- [ ] Git repository is clean (`git status` shows no uncommitted changes)
- [ ] All dependencies are installed with pinned versions
- [ ] Random seed is set to 42 (automatic)
- [ ] Environment snapshot captured (automatic)
- [ ] Hardware configuration documented

### Training Execution

- [ ] Deterministic seed setting verified across all libraries
- [ ] Environment snapshot includes all relevant packages
- [ ] Git commit hash recorded in metadata
- [ ] Dataset version ID stored with experiment
- [ ] Model artifacts include calibration metadata

### Post-Training Validation

- [ ] F1 score meets target (≥0.70)
- [ ] Reproducibility checklist generated
- [ ] Experiment metadata includes all required fields
- [ ] Model card created with feature schema
- [ ] Performance metrics logged comprehensively

### Reproduction Testing

- [ ] Same random seed produces identical results
- [ ] Same Git commit produces identical results
- [ ] Same environment produces identical results
- [ ] Model calibration is consistent
- [ ] Feature importance scores are stable

## CI/CD Integration

### Reproducibility Tests

```yaml
# .github/workflows/ci.yml
- name: Run Reproducibility Tests
  run: |
    pytest tests/ml/test_reproducibility.py -v
    pytest tests/ml/test_features.py::TestIntegrationFeatures -v

- name: Validate F1 Score Target
  run: |
    python scripts/validate_f1_target.py --threshold 0.70

- name: Check Deterministic Reproduction
  run: |
    python scripts/test_deterministic_reproduction.py
```

### Regression Testing

```yaml
- name: Benchmark Against Previous Models
  run: |
    python scripts/run_model_benchmarks.py \
      --model-dir models/ \
      --validation-data validation.csv \
      --history benchmark_history.json \
      --regression-threshold 0.05
```

## API Reference

### Training Functions

- `set_deterministic_seeds(seed: int)` - Set seeds for reproducibility
- `capture_environment_snapshot()` - Capture environment metadata
- `initialize_experiment_tracking(args, config, seed)` - Setup experiment tracking
- `train_ensemble_model(df, save_path, **kwargs)` - Train ensemble model
- `optimize_threshold(X, y, model_params, **kwargs)` - Optimize decision threshold

### Feature Engineering Functions

- `generate_cross_asset_features(df)` - Generate cross-asset features
- `generate_time_anchored_features(df)` - Generate time-anchored features
- `detect_feature_drift(ref_data, curr_data, method)` - Detect feature drift
- `validate_features(df, **checks)` - Validate feature quality

### Monitoring Functions

- `create_model_monitor(config)` - Create model monitor instance
- `ModelMonitor.update_predictions(features, predictions, labels)` - Update monitoring
- `ModelMonitor.check_model_health()` - Assess model health
- `ModelMonitor.generate_report()` - Generate monitoring report

### Dataset Versioning Functions

- `DatasetVersionManager.create_version(data, metadata)` - Create dataset version
- `DatasetVersionManager.load_version(version_id)` - Load dataset version
- `DatasetVersionManager.list_versions()` - List available versions

### Benchmarking Functions

- `benchmark_model(model_path, validation_data, model_name)` - Benchmark single model
- `detect_regressions(current_results, previous_results)` - Find performance regressions
- `validate_f1_target(results, threshold)` - Validate F1 score target achievement
