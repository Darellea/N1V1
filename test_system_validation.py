#!/usr/bin/env python3
"""
System Validation Script

This script validates that the ML pipeline meets all requirements:
- F1 score ≥0.70 for buy signals
- Deterministic reproduction of results
- Ensemble model performance
- Model monitoring functionality
- Feature engineering capabilities
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset_versioning import DatasetVersionManager
from ml.features import detect_feature_drift
from ml.model_monitor import ModelMonitor
from ml.train import set_deterministic_seeds
from ml.trainer import (
    generate_enhanced_features,
    train_ensemble_model,
    train_model_binary,
)


def generate_test_data(n_samples=5000, seed=42):
    """Generate synthetic test data for validation."""
    print("Generating synthetic test data...")

    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="D")

    # Generate realistic price data with trends and volatility
    base_price = 50000
    trend = np.cumsum(np.random.randn(n_samples) * 50)
    noise = np.random.randn(n_samples) * 200
    close_prices = base_price + trend + noise

    # Ensure prices stay positive
    close_prices = np.maximum(close_prices, 1000)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close_prices + np.random.randn(n_samples) * 100,
            "high": close_prices + np.abs(np.random.randn(n_samples)) * 200,
            "low": close_prices - np.abs(np.random.randn(n_samples)) * 200,
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, n_samples),
        }
    )

    # Create target variable based on future returns
    df["future_return"] = df["close"].shift(-5) / df["close"] - 1
    df["target"] = (df["future_return"] > 0.005).astype(int)

    # Remove rows with NaN targets
    df = df.dropna(subset=["target"])

    print(f"Generated {len(df)} samples of test data")
    return df


def validate_f1_score_target():
    """Validate that the model achieves F1 score ≥0.70."""
    print("\n" + "=" * 50)
    print("VALIDATING F1 SCORE TARGET (≥0.70)")
    print("=" * 50)

    # Generate test data
    df = generate_test_data(n_samples=3000)

    # Set deterministic seed
    set_deterministic_seeds(42)

    # Generate enhanced features
    print("Generating enhanced features...")
    df_features = generate_enhanced_features(df)

    # Create binary labels
    from ml.trainer import create_binary_labels

    df_labeled = create_binary_labels(df_features, horizon=5, profit_threshold=0.005)

    # Select numeric features
    exclude_cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "target",
        "future_price",
        "forward_return",
        "sample_weight",
    ]
    feature_cols = [
        col
        for col in df_labeled.columns
        if col not in exclude_cols
        and not col.startswith("DM_")
        and not col.startswith("TR")
    ]

    numeric_features = []
    for col in feature_cols:
        if df_labeled[col].dtype in ["int64", "float64", "int32", "float32"]:
            numeric_features.append(col)

    print(f"Selected {len(numeric_features)} numeric features")

    # Train model
    print("Training binary classification model...")
    results = train_model_binary(
        df=df_labeled,
        save_path="validation_model.pkl",
        results_path="validation_results.json",
        n_splits=5,
        feature_columns=numeric_features[:25],  # Limit features for stability
        tune=False,
        eval_economic=True,
    )

    # Extract metrics
    overall_f1 = results["overall_metrics"]["f1"]
    overall_auc = results["overall_metrics"]["auc"]
    overall_pnl = results["overall_metrics"].get("pnl", 0)

    print(f"F1 Score: {overall_f1:.4f}")
    print(f"AUC Score: {overall_auc:.4f}")
    print(f"PNL: {overall_pnl:.4f}")
    # Validate F1 score target
    if overall_f1 >= 0.70:
        print("✅ F1 SCORE TARGET MET (≥0.70)")
        return True, overall_f1
    else:
        print("❌ F1 SCORE TARGET NOT MET")
        return False, overall_f1


def validate_deterministic_reproduction():
    """Validate that results are deterministically reproducible."""
    print("\n" + "=" * 50)
    print("VALIDATING DETERMINISTIC REPRODUCTION")
    print("=" * 50)

    # Generate test data
    df = generate_test_data(n_samples=2000)

    results_1 = None
    results_2 = None

    # Run training twice with same seed
    for run in [1, 2]:
        print(f"Training run {run}...")

        # Reset seed
        set_deterministic_seeds(42)

        # Generate features
        df_features = generate_enhanced_features(df)

        # Create labels
        from ml.trainer import create_binary_labels

        df_labeled = create_binary_labels(
            df_features, horizon=5, profit_threshold=0.005
        )

        # Select features
        exclude_cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "target",
            "future_price",
            "forward_return",
        ]
        feature_cols = [
            col
            for col in df_labeled.columns
            if col not in exclude_cols
            and not col.startswith("DM_")
            and not col.startswith("TR")
        ]

        numeric_features = []
        for col in feature_cols:
            if df_labeled[col].dtype in ["int64", "float64", "int32", "float32"]:
                numeric_features.append(col)

        # Train model
        results = train_model_binary(
            df=df_labeled,
            save_path=f"validation_model_run{run}.pkl",
            results_path=f"validation_results_run{run}.json",
            n_splits=3,
            feature_columns=numeric_features[:20],
            tune=False,
        )

        if run == 1:
            results_1 = results
        else:
            results_2 = results

    # Compare results
    f1_1 = results_1["overall_metrics"]["f1"]
    f1_2 = results_2["overall_metrics"]["f1"]
    auc_1 = results_1["overall_metrics"]["auc"]
    auc_2 = results_2["overall_metrics"]["auc"]

    print(f"F1 Run 1: {f1_1:.4f}")
    print(f"F1 Run 2: {f1_2:.4f}")
    print(f"AUC Run 1: {auc_1:.4f}")
    print(f"AUC Run 2: {auc_2:.4f}")
    # Check if results are identical (within small tolerance)
    f1_diff = abs(f1_1 - f1_2)
    auc_diff = abs(auc_1 - auc_2)

    if f1_diff < 0.001 and auc_diff < 0.001:
        print("✅ DETERMINISTIC REPRODUCTION VALIDATED")
        return True
    else:
        print("❌ DETERMINISTIC REPRODUCTION FAILED")
        print(f"F1 difference: {f1_diff:.6f}")
        print(f"AUC difference: {auc_diff:.6f}")
        return False


def validate_ensemble_performance():
    """Validate ensemble model performance."""
    print("\n" + "=" * 50)
    print("VALIDATING ENSEMBLE MODEL PERFORMANCE")
    print("=" * 50)

    # Generate test data
    df = generate_test_data(n_samples=2500)

    # Set seed
    set_deterministic_seeds(42)

    # Generate features
    df_features = generate_enhanced_features(df)

    # Create labels
    from ml.trainer import create_binary_labels

    df_labeled = create_binary_labels(df_features, horizon=5, profit_threshold=0.005)

    # Select features
    exclude_cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "target",
        "future_price",
        "forward_return",
    ]
    feature_cols = [
        col
        for col in df_labeled.columns
        if col not in exclude_cols
        and not col.startswith("DM_")
        and not col.startswith("TR")
    ]

    numeric_features = []
    for col in feature_cols:
        if df_labeled[col].dtype in ["int64", "float64", "int32", "float32"]:
            numeric_features.append(col)

    # Train ensemble
    print("Training ensemble model...")
    results = train_ensemble_model(
        df=df_labeled,
        save_path="validation_ensemble.pkl",
        results_path="validation_ensemble_results.json",
        feature_columns=numeric_features[:20],
        n_splits=3,
    )

    # Extract metrics
    overall_auc = results["overall_metrics"]["auc"]
    n_base_models = results["n_base_models"]
    base_models = results["base_models"]

    print(f"AUC Score: {overall_auc:.4f}")
    print(f"Number of base models: {n_base_models}")
    print(f"Base models: {base_models}")

    if overall_auc >= 0.75:
        print("✅ ENSEMBLE PERFORMANCE TARGET MET (AUC ≥0.75)")
        return True, overall_auc
    else:
        print("❌ ENSEMBLE PERFORMANCE TARGET NOT MET")
        return False, overall_auc


def validate_model_monitoring():
    """Validate model monitoring functionality."""
    print("\n" + "=" * 50)
    print("VALIDATING MODEL MONITORING")
    print("=" * 50)

    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {"optimal_threshold": 0.5, "expected_performance": {"pnl": 0.005}}, f
            )
            config_path = f.name

        # Create monitor
        config = {
            "model_path": "validation_model.pkl",
            "config_path": config_path,
            "monitoring_window_days": 30,
        }

        monitor = ModelMonitor(config)

        # Generate test predictions
        np.random.seed(42)
        n_samples = 200
        features = pd.DataFrame(
            np.random.randn(n_samples, 10), columns=[f"feature_{i}" for i in range(10)]
        )
        predictions = np.random.rand(n_samples)
        true_labels = np.random.choice([0, 1], n_samples)

        # Update monitor
        monitor.update_predictions(features, predictions, true_labels)

        # Check health
        health_report = monitor.check_model_health()

        print(f"Health score: {health_report.health_score:.4f}")
        print(f"Performance score: {health_report.performance_score:.3f}")
        print(f"Drift score: {health_report.drift_score:.3f}")
        print(f"Requires retraining: {health_report.requires_retraining}")

        # Cleanup
        os.unlink(config_path)

        print("✅ MODEL MONITORING VALIDATED")
        return True

    except Exception as e:
        print(f"❌ MODEL MONITORING FAILED: {e}")
        return False


def validate_dataset_versioning():
    """Validate dataset versioning functionality."""
    print("\n" + "=" * 50)
    print("VALIDATING DATASET VERSIONING")
    print("=" * 50)

    try:
        # Create test data
        df = pd.DataFrame(
            {
                "feature1": np.random.rand(100),
                "feature2": np.random.rand(100),
                "target": np.random.choice([0, 1], 100),
            }
        )

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize version manager
            version_manager = DatasetVersionManager(storage_dir=temp_dir)

            # Create version
            version_id = version_manager.create_version(
                data=df,
                metadata={
                    "description": "Validation test dataset",
                    "created_by": "system_validation",
                    "feature_count": len(df.columns),
                },
            )

            print(f"Created dataset version: {version_id}")

            # Load version
            loaded_data, loaded_metadata = version_manager.load_version(version_id)

            # Validate
            assert loaded_data.equals(df), "Data mismatch"
            assert (
                loaded_metadata["description"] == "Validation test dataset"
            ), "Metadata mismatch"

            print("✅ DATASET VERSIONING VALIDATED")
            return True

    except Exception as e:
        print(f"❌ DATASET VERSIONING FAILED: {e}")
        return False


def validate_feature_drift_detection():
    """Validate feature drift detection."""
    print("\n" + "=" * 50)
    print("VALIDATING FEATURE DRIFT DETECTION")
    print("=" * 50)

    try:
        # Create reference data
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(5, 2, 1000),
            }
        )

        # Test 1: No drift
        current_no_drift = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(5, 2, 1000),
            }
        )

        drift_detected, drift_scores = detect_feature_drift(
            reference, current_no_drift, method="ks"
        )

        print("No drift test - Drift detected:", drift_detected)
        print("No drift test - Drift scores:", drift_scores)

        # Test 2: With drift
        current_with_drift = pd.DataFrame(
            {
                "feature1": np.random.normal(0.5, 1.5, 1000),  # Mean and std shift
                "feature2": np.random.normal(6, 2.5, 1000),  # Mean and std shift
            }
        )

        drift_detected, drift_scores = detect_feature_drift(
            reference, current_with_drift, method="ks"
        )

        print("With drift test - Drift detected:", drift_detected)
        print("With drift test - Drift scores:", drift_scores)

        if not drift_detected:
            print("❌ DRIFT DETECTION FAILED: Should have detected drift")
            return False

        print("✅ FEATURE DRIFT DETECTION VALIDATED")
        return True

    except Exception as e:
        print(f"❌ FEATURE DRIFT DETECTION FAILED: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🚀 STARTING ML PIPELINE SYSTEM VALIDATION")
    print("=" * 60)

    results = {}

    # Test 1: F1 Score Target
    f1_passed, f1_score = validate_f1_score_target()
    results["f1_target"] = {"passed": f1_passed, "score": f1_score}

    # Test 2: Deterministic Reproduction
    deterministic_passed = validate_deterministic_reproduction()
    results["deterministic"] = {"passed": deterministic_passed}

    # Test 3: Ensemble Performance
    ensemble_passed, ensemble_auc = validate_ensemble_performance()
    results["ensemble"] = {"passed": ensemble_passed, "auc": ensemble_auc}

    # Test 4: Model Monitoring
    monitoring_passed = validate_model_monitoring()
    results["monitoring"] = {"passed": monitoring_passed}

    # Test 5: Dataset Versioning
    versioning_passed = validate_dataset_versioning()
    results["versioning"] = {"passed": versioning_passed}

    # Test 6: Feature Drift Detection
    drift_passed = validate_feature_drift_detection()
    results["drift_detection"] = {"passed": drift_passed}

    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        print(f"{test_name:<20} {status}")

        if not result["passed"]:
            all_passed = False

        # Print additional metrics
        if "score" in result:
            print(f"{'':<25}F1 Score: {result['score']:.4f}")
        if "auc" in result:
            print(f"{'':<25}AUC Score: {result['auc']:.4f}")

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ ML Pipeline meets institutional reliability standards")
        return 0
    else:
        print("⚠️  SOME VALIDATION TESTS FAILED")
        print("❌ ML Pipeline requires attention before production deployment")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
