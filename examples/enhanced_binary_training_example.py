#!/usr/bin/env python3
"""
Enhanced Binary Model Training Example

This script demonstrates the complete implementation of the retraining guide
for improving binary trading model sharpness. It showcases:

1. Enhanced feature engineering with multi-horizon and regime-aware features
2. Data quality improvements with outlier removal
3. Optuna hyperparameter optimization
4. Advanced sample weighting strategies
5. Comprehensive model monitoring and auto-recalibration
6. Economic metrics evaluation and threshold optimization

Usage:
    python examples/enhanced_binary_training_example.py --data path/to/data.csv --output models/enhanced_model.pkl
"""

import argparse
import json
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import monitoring system
from core.model_monitor import create_auto_recalibrator, create_model_monitor

# Import enhanced training functions
from ml.trainer import (
    create_binary_labels,
    create_sample_weights,
    generate_enhanced_features,
    load_data,
    perform_feature_selection,
    remove_outliers,
    train_model_binary,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_enhanced_training_config() -> dict:
    """Create configuration for enhanced training."""
    return {
        "data_quality": {
            "remove_outliers": True,
            "outlier_method": "iqr",
            "outlier_multiplier": 1.5,
        },
        "feature_engineering": {
            "include_multi_horizon": True,
            "include_regime_features": True,
            "include_interaction_features": True,
            "feature_selection": True,
            "top_features": 25,
        },
        "sample_weighting": {
            "method": "combined",  # 'class_balance', 'profit_impact', 'combined'
            "profit_col": "forward_return",
        },
        "hyperparameter_tuning": {
            "enabled": True,
            "n_trials": 25,
            "optimization_metric": "auc",
        },
        "model_training": {
            "n_splits": 5,
            "early_stopping_rounds": 50,
            "eval_economic": True,
        },
        "monitoring": {
            "enabled": True,
            "monitoring_window_days": 30,
            "monitor_interval_minutes": 60,
            "alerts": {"performance_threshold": 0.6, "drift_threshold": 0.7},
        },
    }


def prepare_enhanced_data(data_path: str, config: dict) -> pd.DataFrame:
    """
    Prepare data with enhanced preprocessing following the retraining guide.

    Args:
        data_path: Path to raw OHLCV data
        config: Training configuration

    Returns:
        Prepared DataFrame with enhanced features
    """
    logger.info("Loading and preparing enhanced training data...")

    # Load raw data
    df = load_data(data_path)

    # Validate required columns
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Loaded {len(df)} raw data points")

    # Step 1: Generate enhanced features
    logger.info("Generating enhanced features...")
    df = generate_enhanced_features(
        df,
        include_multi_horizon=config["feature_engineering"]["include_multi_horizon"],
        include_regime_features=config["feature_engineering"][
            "include_regime_features"
        ],
        include_interaction_features=config["feature_engineering"][
            "include_interaction_features"
        ],
    )

    logger.info(f"Generated {len(df.columns)} features")

    # Step 2: Remove outliers for data quality
    if config["data_quality"]["remove_outliers"]:
        logger.info("Removing outliers...")
        before_count = len(df)
        df = remove_outliers(
            df,
            method=config["data_quality"]["outlier_method"],
            multiplier=config["data_quality"]["outlier_multiplier"],
        )
        after_count = len(df)
        logger.info(f"Removed {before_count - after_count} outlier samples")

    # Step 3: Create binary labels
    logger.info("Creating binary labels...")
    df = create_binary_labels(
        df,
        horizon=5,  # 5-period forward return
        profit_threshold=0.005,  # 0.5% profit threshold
        include_fees=True,
        fee_rate=0.001,  # 0.1% trading fees
    )

    # Step 4: Create sample weights
    logger.info("Creating sample weights...")
    sample_weights = create_sample_weights(
        df,
        label_col="label_binary",
        profit_col=config["sample_weighting"]["profit_col"],
        method=config["sample_weighting"]["method"],
    )
    df["sample_weight"] = sample_weights

    # Step 5: Feature selection (optional, done later in training)
    if config["feature_engineering"]["feature_selection"]:
        logger.info("Feature selection will be performed during training...")

    # Final cleanup
    df = df.dropna(subset=["label_binary"])
    logger.info(f"Final dataset: {len(df)} samples, {len(df.columns)} features")

    return df


def perform_enhanced_training(df: pd.DataFrame, output_path: str, config: dict) -> dict:
    """
    Perform enhanced training with all improvements from the retraining guide.

    Args:
        df: Prepared DataFrame
        output_path: Path to save trained model
        config: Training configuration

    Returns:
        Training results dictionary
    """
    logger.info("Starting enhanced model training...")

    # Prepare feature columns
    exclude_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "label_binary",
        "future_price",
        "forward_return",
        "sample_weight",
    ]
    all_feature_columns = [col for col in df.columns if col not in exclude_cols]

    # Perform feature selection if enabled
    if config["feature_engineering"]["feature_selection"]:
        logger.info("Performing feature selection...")
        X_temp = df[all_feature_columns].copy()
        y_temp = df["label_binary"].copy()

        # Remove NaN values for feature selection
        valid_mask = ~X_temp.isna().any(axis=1) & ~y_temp.isna()
        X_temp = X_temp[valid_mask]
        y_temp = y_temp[valid_mask]

        if len(X_temp) > 0:
            selected_features = perform_feature_selection(
                X_temp,
                y_temp,
                method="gain_importance",
                top_k=config["feature_engineering"]["top_features"],
            )
            feature_columns = [col for col in selected_features if col in df.columns]
            logger.info(f"Selected {len(feature_columns)} features")
        else:
            feature_columns = all_feature_columns
    else:
        feature_columns = all_feature_columns

    # Perform hyperparameter tuning if enabled
    tune = config["hyperparameter_tuning"]["enabled"]
    n_trials = config["hyperparameter_tuning"]["n_trials"] if tune else 0

    # Train the enhanced model
    logger.info("Training enhanced binary model...")
    results = train_model_binary(
        df=df,
        save_path=output_path,
        results_path=output_path.replace(".pkl", "_results.json"),
        n_splits=config["model_training"]["n_splits"],
        horizon=5,
        profit_threshold=0.005,
        include_fees=True,
        fee_rate=0.001,
        feature_columns=feature_columns,
        tune=tune,
        n_trials=n_trials,
        feature_selection=False,  # Already done above
        early_stopping_rounds=config["model_training"]["early_stopping_rounds"],
        eval_economic=config["model_training"]["eval_economic"],
    )

    logger.info("Enhanced training completed successfully!")
    return results


def setup_monitoring_system(model_path: str, config: dict) -> tuple:
    """
    Setup comprehensive monitoring and auto-recalibration system.

    Args:
        model_path: Path to trained model
        config: Monitoring configuration

    Returns:
        Tuple of (monitor, recalibrator) instances
    """
    logger.info("Setting up monitoring and auto-recalibration system...")

    # Create monitoring configuration
    monitor_config = {
        "model_path": model_path,
        "config_path": model_path.replace(".pkl", "_config.json"),
        "monitoring_window_days": config["monitoring"]["monitoring_window_days"],
        "monitor_interval_minutes": config["monitoring"]["monitor_interval_minutes"],
        "output_dir": "monitoring",
        "alerts": config["monitoring"]["alerts"],
        "drift_thresholds": {
            "overall_threshold": config["monitoring"]["alerts"]["drift_threshold"]
        },
    }

    # Create monitor and auto-recalibrator
    monitor = create_model_monitor(monitor_config)

    recalibrator_config = monitor_config.copy()
    recalibrator_config.update(
        {
            "min_retraining_interval_hours": 24,
            "retraining_data_path": config.get("retraining_data_path"),
        }
    )

    recalibrator = create_auto_recalibrator(recalibrator_config)

    logger.info("Monitoring system setup complete")
    return monitor, recalibrator


def generate_training_report(results: dict, output_path: str):
    """Generate comprehensive training report."""
    logger.info("Generating training report...")

    report = {
        "timestamp": datetime.now().isoformat(),
        "training_type": "enhanced_binary_model",
        "summary": {
            "total_samples": results.get("overall_metrics", {}).get("total_samples", 0),
            "auc_score": results.get("overall_metrics", {}).get("auc", 0),
            "f1_score": results.get("overall_metrics", {}).get("f1", 0),
            "sharpe_ratio": results.get("overall_metrics", {}).get("sharpe", 0),
            "max_drawdown": results.get("overall_metrics", {}).get("max_drawdown", 0),
            "total_trades": results.get("overall_metrics", {}).get("total_trades", 0),
            "win_rate": results.get("overall_metrics", {}).get("win_rate", 0),
        },
        "fold_performance": results.get("fold_metrics", []),
        "feature_importance": results.get("feature_importance", {}),
        "model_configuration": results.get("metadata", {}),
        "improvements_applied": [
            "Enhanced feature engineering (multi-horizon, regime-aware, interaction features)",
            "Data quality improvements (outlier removal)",
            "Advanced sample weighting (profit-impact based)",
            "Optuna hyperparameter optimization",
            "Feature selection using gain importance",
            "Probability calibration with isotonic regression",
            "Optimal threshold selection for maximum Sharpe ratio",
            "Comprehensive economic metrics evaluation",
            "Walk-forward validation for robustness",
            "Model monitoring and auto-recalibration system",
        ],
    }

    # Save report
    report_path = output_path.replace(".pkl", "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Training report saved to {report_path}")
    return report


def create_visualizations(results: dict, output_path: str):
    """Create training performance visualizations."""
    logger.info("Creating performance visualizations...")

    # Create output directory for plots
    plots_dir = output_path.replace(".pkl", "_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Feature Importance Plot
    if "feature_importance" in results:
        plt.figure(figsize=(12, 8))
        features = list(results["feature_importance"].keys())
        importance = list(results["feature_importance"].values())

        # Sort by importance
        sorted_idx = np.argsort(importance)[-20:]  # Top 20 features
        features = [features[i] for i in sorted_idx]
        importance = [importance[i] for i in sorted_idx]

        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel("Feature Importance")
        plt.title("Top 20 Feature Importance Scores")
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, "feature_importance.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # 2. Fold Performance Plot
    if "fold_metrics" in results:
        fold_metrics = results["fold_metrics"]
        folds = range(1, len(fold_metrics) + 1)
        auc_scores = [m.get("auc", 0) for m in fold_metrics]
        sharpe_scores = [m.get("sharpe", 0) for m in fold_metrics]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(folds, auc_scores, "o-", label="AUC")
        plt.xlabel("Fold")
        plt.ylabel("AUC Score")
        plt.title("AUC Score by Fold")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(folds, sharpe_scores, "o-", color="orange", label="Sharpe")
        plt.xlabel("Fold")
        plt.ylabel("Sharpe Ratio")
        plt.title("Sharpe Ratio by Fold")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, "fold_performance.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # 3. Confusion Matrix Plot
    if "confusion_matrix" in results:
        cm = results["confusion_matrix"]["matrix"]
        labels = results["confusion_matrix"]["labels"]

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar()

        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        # Add text annotations
        thresh = np.array(cm).max() / 2.0
        for i, j in np.ndindex(cm):
            plt.text(
                j,
                i,
                format(cm[i][j], "d"),
                horizontalalignment="center",
                color="white" if cm[i][j] > thresh else "black",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, "confusion_matrix.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    logger.info(f"Visualizations saved to {plots_dir}")


def main():
    """Main function for enhanced binary model training."""
    parser = argparse.ArgumentParser(
        description="Enhanced Binary Model Training Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training with enhanced features
    python examples/enhanced_binary_training_example.py --data data.csv --output models/enhanced_model.pkl

    # Training with monitoring enabled
    python examples/enhanced_binary_training_example.py --data data.csv --output models/enhanced_model.pkl --enable-monitoring

    # Full pipeline with custom configuration
    python examples/enhanced_binary_training_example.py --data data.csv --output models/enhanced_model.pkl --config config/custom_config.json
        """,
    )

    parser.add_argument("--data", required=True, help="Path to OHLCV data CSV file")
    parser.add_argument(
        "--output", required=True, help="Output path for trained model (.pkl)"
    )
    parser.add_argument("--config", help="Path to custom configuration JSON file")
    parser.add_argument(
        "--enable-monitoring",
        action="store_true",
        help="Enable model monitoring and auto-recalibration",
    )
    parser.add_argument("--retraining-data", help="Path to data for auto-retraining")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting Enhanced Binary Model Training")
    logger.info("=" * 50)

    try:
        # Load or create configuration
        if args.config and os.path.exists(args.config):
            with open(args.config, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = create_enhanced_training_config()
            logger.info("Using default enhanced training configuration")

        # Add retraining data path if provided
        if args.retraining_data:
            config["retraining_data_path"] = args.retraining_data

        # Step 1: Prepare enhanced data
        logger.info("Step 1: Data Preparation")
        df = prepare_enhanced_data(args.data, config)

        # Step 2: Perform enhanced training
        logger.info("Step 2: Model Training")
        results = perform_enhanced_training(df, args.output, config)

        # Step 3: Generate reports and visualizations
        logger.info("Step 3: Generating Reports")
        report = generate_training_report(results, args.output)
        create_visualizations(results, args.output)

        # Step 4: Setup monitoring (optional)
        if args.enable_monitoring:
            logger.info("Step 4: Setting up Monitoring System")
            monitor, recalibrator = setup_monitoring_system(args.output, config)

            # Start monitoring
            recalibrator.start()
            logger.info("Model monitoring and auto-recalibration started")

            # Generate initial monitoring report
            monitoring_report = monitor.generate_report()
            monitoring_path = args.output.replace(".pkl", "_monitoring_report.json")
            with open(monitoring_path, "w") as f:
                json.dump(monitoring_report, f, indent=2, default=str)
            logger.info(f"Initial monitoring report saved to {monitoring_path}")

        # Summary
        logger.info("=" * 50)
        logger.info("ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"Model saved to: {args.output}")
        logger.info(f"Training results: {args.output.replace('.pkl', '_results.json')}")
        logger.info(f"Training report: {args.output.replace('.pkl', '_report.json')}")
        logger.info(f"Visualizations: {args.output.replace('.pkl', '_plots/')}")

        if args.enable_monitoring:
            logger.info("Monitoring data: monitoring/")
            logger.info("Auto-recalibration system is ACTIVE")

        logger.info("=" * 50)

        # Print key metrics
        overall = results.get("overall_metrics", {})
        logger.info("KEY PERFORMANCE METRICS:")
        logger.info(f"  AUC Score: {overall.get('auc', 0):.4f}")
        logger.info(f"  F1 Score: {overall.get('f1', 0):.4f}")
        logger.info(f"  Sharpe Ratio: {overall.get('sharpe', 0):.4f}")
        logger.info(f"  Max Drawdown: {overall.get('max_drawdown', 0):.4f}")
        logger.info(f"  Win Rate: {overall.get('win_rate', 0):.4f}")
        logger.info(f"  Total Trades: {overall.get('total_trades', 0)}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
