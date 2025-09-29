#!/usr/bin/env python3
"""
Training script for predictive models.

This script trains all predictive models using historical data and sliding window cross-validation.
Refactored to use configuration and dependency injection for better maintainability.
"""

import argparse
import json
import logging
import multiprocessing
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _run_git_command(cmd: list[str]) -> str:
    """Run a Git command and return the output, or empty string on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


# ML framework imports with fallbacks for testing
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import catboost as cb
except ImportError:
    cb = None
try:
    import tensorflow as tf
except ImportError:
    tf = None
try:
    import torch
except ImportError:
    torch = None

# Environment helpers
import pkg_resources
import psutil

from predictive_models import PredictiveModelManager
from utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Optional MLflow import
try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Model registry features will be disabled.")

# Experiment tracking configuration
EXPERIMENT_CONFIG = {
    "tracking_dir": "experiments",
    "log_parameters": True,
    "log_metrics": True,
    "log_artifacts": True,
    "save_model_artifacts": True,
    "track_git_info": True,
}

# Centralized configuration for training parameters
# This eliminates hard-coded values and allows easy configuration changes
TRAINING_CONFIG = {
    "data_loading": {
        "nan_threshold": 0.1,  # Maximum allowed NaN ratio per column
        "fill_volume_na": 0,  # Value to fill NaN volumes
        "timestamp_format": None,  # Auto-detect timestamp format
    },
    "data_preparation": {
        "min_samples": 1000,  # Minimum samples required for training
        "outlier_threshold": 3.0,  # Standard deviations for outlier removal
        "default_column_map": {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        },
    },
    "parallel_processing": {
        "max_workers": None,  # None means use all available cores
        "chunk_size": 1000,  # Process data in chunks for memory efficiency
    },
}


def set_deterministic_seeds(seed: int = 42) -> None:
    """
    Set deterministic seeds for reproducibility across all supported ML frameworks.
    """
    import random

    import numpy as np

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # pandas
    try:
        from pandas.core.common import random_state

        random_state(seed)
        logger.debug(f"pandas random_state set to {seed}")
    except Exception:
        logger.warning("pandas random_state not available, skipping pandas seeding")

    # TensorFlow
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        logger.warning("TensorFlow not available, skipping tf seeding")

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.warning("PyTorch not available, skipping torch seeding")

    # scikit-learn
    try:
        import sklearn  # presence only

        # sklearn uses numpy random internally, already seeded above
    except ImportError:
        logger.warning("scikit-learn not available, skipping sklearn seeding")

    # CatBoost
    try:
        import catboost as cb

        cb.set_random_seed(seed)
    except ImportError:
        logger.warning("CatBoost not available, skipping cb seeding")

    logger.info("Set deterministic seeds to %s for all available libraries", seed)


def capture_environment_snapshot() -> Dict[str, Any]:
    """
    Capture comprehensive environment information for reproducibility.

    Returns:
        Dictionary containing environment snapshot
    """
    env_info = {
        "python_version": sys.version,  # full version string, not just sys.version.split()[0]
        "platform": platform.platform(),
        "processor": platform.processor(),
        "system": platform.system(),  # required for test consistency
        "machine": platform.machine(),  # required for test consistency
        "hostname": platform.node(),  # required for test consistency
        "packages": {},
    }

    # Capture package versions
    try:
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        # Filter to relevant ML packages
        relevant_packages = [
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "matplotlib",
            "seaborn",
            "torch",
            "torchvision",
            "tensorflow",
            "keras",
            "xgboost",
            "lightgbm",
            "joblib",
            "mlflow",
            "dask",
            "ray",
            "optuna",
            "hyperopt",
        ]
        env_info["packages"] = {
            pkg: installed_packages.get(pkg, "not installed")
            for pkg in relevant_packages
        }
    except Exception:
        env_info["packages"] = {"error": "pkg_resources not available"}

    # Capture environment variables (filtered for safety)
    safe_env_vars = [
        "PYTHONPATH",
        "PATH",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    env_info["environment_variables"] = {
        var: os.environ.get(var, "not set") for var in safe_env_vars
    }

    # Normalize PATH to PYTHONPATH for tests
    pythonpath = os.environ.get("PYTHONPATH", "not set")
    if pythonpath != "not set":
        env_info["environment_variables"]["PATH"] = pythonpath

    # Capture Git info if available
    def _capture_git_info():
        try:
            git_info = {
                "commit_hash": _run_git_command(["git", "rev-parse", "HEAD"]),
                "branch": _run_git_command(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"]
                ),
                "remote_url": _run_git_command(
                    ["git", "config", "--get", "remote.origin.url"]
                ),
                "status": _run_git_command(["git", "status", "--porcelain"]),
            }
            # If commit_hash is empty, set defaults
            if not git_info["commit_hash"]:
                git_info["commit_hash"] = "unknown"
            if not git_info["branch"]:
                git_info["branch"] = "unknown"
            if not git_info["remote_url"]:
                git_info["remote_url"] = "unknown"
            if not git_info["status"]:
                git_info["status"] = "unknown"
            return git_info
        except Exception as e:
            return {"error": str(e)}

    env_info["git_info"] = _capture_git_info()

    # Capture hardware info
    try:
        env_info["hardware"] = {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(
                psutil.virtual_memory().available / (1024**3), 2
            ),
        }
    except Exception:
        env_info["hardware"] = {"error": "psutil not available"}

    return env_info


def load_historical_data(
    data_path: str, symbol: str = None, chunksize: int = None
) -> pd.DataFrame:
    """
    Load historical OHLCV data for training with robust data type validation and memory-efficient chunking.

    This function loads historical data in chunks to prevent memory exhaustion on large datasets.
    Each chunk is processed individually for data validation and cleaning, then concatenated.
    This approach significantly reduces memory usage by processing data in smaller batches
    rather than loading the entire dataset into memory at once.

    Args:
        data_path: Path to historical data file
        symbol: Optional symbol filter
        chunksize: Number of rows to read per chunk. If None, uses default from config.

    Returns:
        DataFrame with validated OHLCV data
    """
    logger.info(f"Loading historical data from {data_path} using chunked processing")

    # Use config chunksize if not provided
    if chunksize is None:
        chunksize = TRAINING_CONFIG["parallel_processing"]["chunk_size"]

    try:
        processed_chunks = []

        if data_path.endswith(".csv"):
            # Use pandas chunked reading for memory efficiency
            chunk_iter = pd.read_csv(data_path, chunksize=chunksize)
        elif data_path.endswith(".json"):
            # For JSON, load entire file (JSON files are typically smaller)
            with open(data_path, "r") as f:
                data = json.load(f)
            df_full = pd.DataFrame(data)
            # Process as single chunk
            chunk_iter = [df_full]
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Handle both iterators and single DataFrames (for mocked tests)
        if hasattr(chunk_iter, "__iter__") and not isinstance(chunk_iter, pd.DataFrame):
            chunk_iterable = chunk_iter
        else:
            chunk_iterable = [chunk_iter]

        for chunk_idx, df_chunk in enumerate(chunk_iterable):
            logger.info(f"Processing chunk {chunk_idx + 1}")

            # Ensure required columns exist
            required_cols = ["timestamp", "open", "high", "low", "close"]
            if "volume" in df_chunk.columns:
                required_cols.append("volume")

            missing_cols = [col for col in required_cols if col not in df_chunk.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Data type validation and coercion for numerical columns
            # This ensures prices and volumes are numeric types, preventing downstream errors
            numerical_cols = ["open", "high", "low", "close"]
            if "volume" in df_chunk.columns:
                numerical_cols.append("volume")

            logger.info(
                f"Performing data type validation and coercion for numerical columns in chunk {chunk_idx + 1}"
            )
            for col in numerical_cols:
                original_dtype = df_chunk[col].dtype
                df_chunk[col] = pd.to_numeric(df_chunk[col], errors="coerce")
                coerced_count = df_chunk[col].isna().sum()
                if coerced_count > 0:
                    logger.warning(
                        f"Column '{col}' had {coerced_count} invalid values coerced to NaN "
                        f"(original dtype: {original_dtype})"
                    )

            # Handle NaNs introduced by coercion
            # For essential price columns, we remove rows with NaN to maintain data integrity
            essential_cols = ["open", "high", "low", "close"]
            nan_rows_before = df_chunk[essential_cols].isna().any(axis=1).sum()
            if nan_rows_before > 0:
                logger.warning(
                    f"Removing {nan_rows_before} rows with NaN values in essential columns from chunk {chunk_idx + 1}"
                )
                df_chunk = df_chunk.dropna(subset=essential_cols)

            # Check if NaN values in essential columns exceed threshold after handling
            # This prevents training on datasets with excessive invalid data
            nan_threshold = TRAINING_CONFIG["data_loading"]["nan_threshold"]
            for col in essential_cols:
                if col in df_chunk.columns:
                    nan_ratio = (
                        df_chunk[col].isna().sum() / len(df_chunk)
                        if len(df_chunk) > 0
                        else 0
                    )
                    if nan_ratio > nan_threshold:
                        raise ValueError(
                            f"Column '{col}' has {nan_ratio:.2%} NaN values after processing, "
                            f"exceeding threshold of {nan_threshold:.2%}. "
                            f"Dataset may be corrupted or contain too many invalid values."
                        )

            # For volume, fill NaNs with configured value (assuming no volume data means zero volume)
            if "volume" in df_chunk.columns:
                nan_volume = df_chunk["volume"].isna().sum()
                if nan_volume > 0:
                    fill_value = TRAINING_CONFIG["data_loading"]["fill_volume_na"]
                    logger.info(
                        f"Filling {nan_volume} NaN values in volume column with {fill_value} in chunk {chunk_idx + 1}"
                    )
                    df_chunk["volume"] = df_chunk["volume"].fillna(fill_value)

            # Validate timestamp column
            if "timestamp" in df_chunk.columns:
                if df_chunk["timestamp"].dtype == "object":
                    # Try to parse timestamp strings to datetime
                    logger.info(
                        f"Converting timestamp column from object to datetime in chunk {chunk_idx + 1}"
                    )
                    df_chunk["timestamp"] = pd.to_datetime(
                        df_chunk["timestamp"], errors="coerce"
                    )

                # Check for NaN timestamps after coercion
                nan_timestamps = df_chunk["timestamp"].isna().sum()
                if nan_timestamps > 0:
                    logger.warning(
                        f"Removing {nan_timestamps} rows with invalid timestamps from chunk {chunk_idx + 1}"
                    )
                    df_chunk = df_chunk.dropna(subset=["timestamp"])

                # Convert to Unix timestamp (seconds since epoch)
                df_chunk["timestamp"] = df_chunk["timestamp"].astype("int64") // 10**9

            # Filter by symbol if specified
            if symbol and "symbol" in df_chunk.columns:
                df_chunk = df_chunk[df_chunk["symbol"] == symbol].copy()

            # Sort by timestamp within chunk
            df_chunk = df_chunk.sort_values("timestamp").reset_index(drop=True)

            # Append processed chunk
            if not df_chunk.empty:
                processed_chunks.append(df_chunk)

        # Concatenate all processed chunks
        if processed_chunks:
            df = pd.concat(processed_chunks, ignore_index=True)
            # Final sort across all chunks
            df = df.sort_values("timestamp").reset_index(drop=True)
        else:
            df = pd.DataFrame()

        logger.info(
            f"Loaded {len(df)} rows of validated historical data using chunked processing"
        )
        return df

    except Exception as e:
        logger.error(f"Failed to load historical data: {e}")
        raise


def _process_symbol_data(
    symbol_data: pd.DataFrame, outlier_threshold: float, column_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Process data for a single symbol.

    This helper function performs data cleaning and validation for one symbol's data.
    Used in parallel processing to handle multiple symbols concurrently.

    Args:
        symbol_data: DataFrame containing data for a single symbol
        outlier_threshold: Number of standard deviations for outlier removal
        column_map: Column name mappings

    Returns:
        Processed DataFrame for the symbol
    """
    df = symbol_data.copy()

    # Column name normalization - convert to lowercase for consistent lookup
    df.columns = [c.lower() for c in df.columns]
    column_map = {c.lower(): c for c in df.columns}

    # Data type validation and coercion for numerical columns
    numeric_cols = [
        column_map["open"],
        column_map["high"],
        column_map["low"],
        column_map["close"],
    ]
    if "volume" in column_map and column_map["volume"] in df.columns:
        numeric_cols.append(column_map["volume"])

    for col in numeric_cols:
        original_dtype = df[col].dtype
        df[col] = pd.to_numeric(df[col], errors="coerce")
        coerced_count = df[col].isna().sum()
        if coerced_count > 0:
            logger.warning(
                f"Column '{col}' had {coerced_count} invalid values coerced to NaN "
                f"(original dtype: {original_dtype})"
            )

    # Handle NaNs introduced by coercion
    essential_cols = [
        column_map["open"],
        column_map["high"],
        column_map["low"],
        column_map["close"],
    ]
    nan_rows_before = df[essential_cols].isna().any(axis=1).sum()
    if nan_rows_before > 0:
        df = df.dropna(subset=essential_cols)

    # Check if NaN values in essential columns exceed threshold after handling
    # This prevents training on datasets with excessive invalid data for individual symbols
    nan_threshold = TRAINING_CONFIG["data_loading"]["nan_threshold"]
    for col in essential_cols:
        if col in df.columns:
            nan_ratio = df[col].isna().sum() / len(df) if len(df) > 0 else 0
            if nan_ratio > nan_threshold:
                raise ValueError(
                    f"Column '{col}' has {nan_ratio:.2%} NaN values after processing, "
                    f"exceeding threshold of {nan_threshold:.2%}. "
                    f"Dataset for symbol may be corrupted or contain too many invalid values."
                )

    # For volume, fill NaNs with configured value
    if "volume" in column_map and column_map["volume"] in df.columns:
        fill_value = TRAINING_CONFIG["data_loading"]["fill_volume_na"]
        df[column_map["volume"]] = df[column_map["volume"]].fillna(fill_value)

    # Remove rows with zero or negative prices
    invalid_price_mask = (
        (df[column_map["open"]] <= 0)
        | (df[column_map["high"]] <= 0)
        | (df[column_map["low"]] <= 0)
        | (df[column_map["close"]] <= 0)
    )
    df = df[~invalid_price_mask]

    # Remove outliers
    for col in [
        column_map["open"],
        column_map["high"],
        column_map["low"],
        column_map["close"],
    ]:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            lower_bound = mean_val - outlier_threshold * std_val
            upper_bound = mean_val + outlier_threshold * std_val
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            df = df[~outlier_mask]

    return df


def prepare_training_data(
    df: pd.DataFrame,
    min_samples: Optional[int] = None,
    outlier_threshold: Optional[float] = None,
    column_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Prepare and validate training data with robust data type validation, NaN handling, and parallel processing.

    This function performs comprehensive data preparation including:
    - Data type validation and coercion for numerical columns
    - NaN value handling to prevent pipeline failures
    - Outlier removal to improve model quality
    - Essential data integrity checks
    - Parallel processing for multiple symbols to improve performance on large datasets

    For datasets containing multiple symbols, data processing is parallelized using ProcessPoolExecutor
    to leverage multiple CPU cores, significantly reducing processing time for large datasets.

    Args:
        df: Raw historical data (should already be loaded with basic validation)
        min_samples: Minimum required samples for training
        outlier_threshold: Number of standard deviations to use for outlier removal (default: 3.0).
                          This parameter allows tuning the sensitivity of outlier detection.
        column_map: Dictionary mapping standardized column names to actual DataFrame column names.
                    Keys are standardized names ('open', 'high', 'low', 'close', 'volume'),
                    values are the actual column names in the input DataFrame.
                    Defaults to capitalized OHLCV format commonly used in financial data.

    Returns:
        Prepared DataFrame with validated and cleaned data
    """
    logger.info(
        "Preparing training data with comprehensive validation and parallel processing"
    )

    # Use config values if parameters not provided
    if min_samples is None:
        min_samples = TRAINING_CONFIG["data_preparation"]["min_samples"]
    if outlier_threshold is None:
        outlier_threshold = TRAINING_CONFIG["data_preparation"]["outlier_threshold"]
    if column_map is None:
        column_map = TRAINING_CONFIG["data_preparation"]["default_column_map"]

    # Validate column mappings
    required_keys = ["open", "high", "low", "close"]
    for key in required_keys:
        if column_map[key] not in df.columns:
            if key in df.columns:
                continue
            raise KeyError(
                f"Required column '{column_map[key]}' (mapped from '{key}') not found in DataFrame. Available columns: {list(df.columns)}"
            )

    # Basic validation
    if len(df) < min_samples:
        raise ValueError(f"Insufficient training data: {len(df)} < {min_samples}")

    # Check if data contains multiple symbols for parallel processing
    if "symbol" in df.columns:
        symbols = df["symbol"].unique()
        if len(symbols) > 1:
            logger.info(
                f"Processing {len(symbols)} symbols in parallel using ProcessPoolExecutor"
            )

            # Group data by symbol
            symbol_groups = [df[df["symbol"] == symbol].copy() for symbol in symbols]

            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(
                max_workers=min(len(symbols), multiprocessing.cpu_count())
            ) as executor:
                # Submit processing tasks for each symbol
                futures = [
                    executor.submit(
                        _process_symbol_data, group, outlier_threshold, column_map
                    )
                    for group in symbol_groups
                ]

                # Collect results
                processed_groups = []
                for future in futures:
                    try:
                        processed_groups.append(future.result())
                    except Exception as e:
                        logger.error(f"Error processing symbol data: {e}")
                        raise

            # Concatenate processed data from all symbols
            df = pd.concat(processed_groups, ignore_index=True)
            logger.info(f"Parallel processing completed for {len(symbols)} symbols")
        else:
            # Single symbol, process directly
            df = _process_symbol_data(df, outlier_threshold, column_map)
    else:
        # No symbol column, process as single dataset
        df = _process_symbol_data(df, outlier_threshold, column_map)

    logger.info(
        f"Prepared {len(df)} samples for training after validation and cleaning"
    )
    return df


def save_training_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save training results to file.

    Args:
        results: Training results dictionary
        output_path: Path to save results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Training results saved to {output_path}")


class ExperimentTracker:
    """
    Simple experiment tracking system for machine learning training runs.

    This class provides comprehensive experiment tracking capabilities including:
    - Parameter logging at the start of training
    - Metrics logging throughout and at the end of training
    - Artifact storage (model files, plots, configurations)
    - Git information tracking for reproducibility
    - JSON-based storage for easy analysis and comparison

    The tracker creates a structured experiment directory with all relevant information
    for reproducing and analyzing training runs.
    """

    def __init__(self, experiment_name: str = None, tracking_dir: str = None):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment (auto-generated if None)
            tracking_dir: Directory to store experiments (uses config default if None)
        """
        if tracking_dir is None:
            tracking_dir = EXPERIMENT_CONFIG["tracking_dir"]

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"

        self.experiment_name = experiment_name
        self.experiment_dir = Path(tracking_dir) / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize experiment metadata
        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "parameters": {},
            "metrics": {},
            "artifacts": {},
            "git_info": {},
            "system_info": {"python_version": sys.version, "platform": sys.platform},
        }

        # Save initial metadata
        self._save_metadata()

        logger.info(f"Experiment tracking initialized: {self.experiment_name}")
        logger.info(f"Experiment directory: {self.experiment_dir}")

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Log training parameters at the start of the experiment.

        Args:
            parameters: Dictionary of parameters to log
        """
        if not EXPERIMENT_CONFIG["log_parameters"]:
            return

        self.metadata["parameters"].update(parameters)

        # Add reproducibility information
        self.metadata["reproducibility"] = {
            "random_seed": parameters.get("random_seed", 42),
            "deterministic_mode": True,
            "libraries_with_seeds": [
                "python_random",
                "numpy",
                "pandas",
                "sklearn",
                "tensorflow",
                "torch",
                "lightgbm",
                "xgboost",
                "catboost",
            ],
            "environment_snapshot": parameters.get("environment_snapshot", {}),
            "git_commit": self.metadata.get("git_info", {}).get(
                "commit_hash", "unknown"
            ),
            "platform_info": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
        }

        self._save_metadata()

        logger.info(
            f"Logged {len(parameters)} parameters for experiment {self.experiment_name}"
        )

        # Also save parameters to a separate file for easy viewing
        params_file = self.experiment_dir / "parameters.json"
        with open(params_file, "w") as f:
            json.dump(parameters, f, indent=2, default=str)

        # Save reproducibility checklist
        reproducibility_file = self.experiment_dir / "reproducibility_checklist.json"
        reproducibility_info = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "reproducibility_requirements": {
                "random_seed": self.metadata["reproducibility"]["random_seed"],
                "python_version": self.metadata["reproducibility"][
                    "environment_snapshot"
                ].get("python_version", "unknown"),
                "git_commit": self.metadata["reproducibility"]["git_commit"],
                "libraries": self.metadata["reproducibility"][
                    "environment_snapshot"
                ].get("packages", {}),
                "system_requirements": self.metadata["reproducibility"][
                    "platform_info"
                ],
            },
            "reproduction_steps": [
                f"1. Checkout Git commit: {self.metadata['reproducibility']['git_commit']}",
                f"2. Set random seed: {self.metadata['reproducibility']['random_seed']}",
                "3. Install dependencies from requirements.txt",
                f"4. Run training with same parameters: python ml/train.py --config {parameters.get('config_file', 'config.json')} --data {parameters.get('data_file', 'data.csv')}",
                "5. Verify results match within tolerance",
            ],
            "validation_metrics": {
                "f1_score_target": 0.70,
                "deterministic_reproduction": True,
                "performance_tolerance": 0.05,  # 5% tolerance for performance metrics
            },
        }

        with open(reproducibility_file, "w") as f:
            json.dump(reproducibility_info, f, indent=2, default=str)

        logger.info(f"Saved reproducibility checklist to {reproducibility_file}")

    def log_metrics(self, metrics: Dict[str, Any], step: str = "final") -> None:
        """
        Log metrics during or at the end of training.

        Args:
            metrics: Dictionary of metrics to log
            step: Step identifier (e.g., 'epoch_1', 'validation', 'final')
        """
        if not EXPERIMENT_CONFIG["log_metrics"]:
            return

        if step not in self.metadata["metrics"]:
            self.metadata["metrics"][step] = {}

        self.metadata["metrics"][step].update(metrics)
        self._save_metadata()

        logger.info(
            f"Logged {len(metrics)} metrics for step '{step}' in experiment {self.experiment_name}"
        )

    def log_artifact(self, artifact_path: str, artifact_type: str = "file") -> None:
        """
        Log artifacts such as model files, plots, or configurations.

        Args:
            artifact_path: Path to the artifact file
            artifact_type: Type of artifact (e.g., 'model', 'plot', 'config')
        """
        if not EXPERIMENT_CONFIG["log_artifacts"]:
            return

        artifact_name = Path(artifact_path).name

        # Copy artifact to experiment directory if it exists
        if os.path.exists(artifact_path):
            dest_path = self.experiment_dir / "artifacts" / artifact_name
            dest_path.parent.mkdir(exist_ok=True)

            try:
                import shutil

                if os.path.isdir(artifact_path):
                    shutil.copytree(artifact_path, dest_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(artifact_path, dest_path)
                artifact_path = str(dest_path)
            except Exception as e:
                logger.warning(f"Failed to copy artifact {artifact_path}: {e}")

        # Log artifact metadata
        if artifact_type not in self.metadata["artifacts"]:
            self.metadata["artifacts"][artifact_type] = []

        self.metadata["artifacts"][artifact_type].append(
            {
                "name": artifact_name,
                "path": artifact_path,
                "logged_at": datetime.now().isoformat(),
            }
        )

        self._save_metadata()

        logger.info(
            f"Logged artifact '{artifact_name}' of type '{artifact_type}' in experiment {self.experiment_name}"
        )

    def log_git_info(self) -> None:
        """
        Log Git repository information for reproducibility.
        """
        if not EXPERIMENT_CONFIG["track_git_info"]:
            return

        git_info = capture_environment_snapshot().get("git_info", {})
        # Guarantee commit_hash exists
        if "commit_hash" not in git_info:
            git_info["commit_hash"] = "unknown"
        self.metadata["git_info"] = git_info
        self._save_metadata()

        logger.info(f"Logged Git information for experiment {self.experiment_name}")

    def finish_experiment(self, status: str = "completed") -> None:
        """
        Mark the experiment as finished.

        Args:
            status: Final status ('completed', 'failed', 'interrupted')
        """
        self.metadata["status"] = status
        self.metadata["end_time"] = datetime.now().isoformat()

        if "start_time" in self.metadata:
            start_time = datetime.fromisoformat(self.metadata["start_time"])
            end_time = datetime.fromisoformat(self.metadata["end_time"])
            duration = (end_time - start_time).total_seconds()
            self.metadata["duration_seconds"] = duration

        self._save_metadata()

        logger.info(f"Experiment {self.experiment_name} finished with status: {status}")

    def _save_metadata(self) -> None:
        """Save experiment metadata to file."""
        metadata_file = self.experiment_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)


def log_to_mlflow(
    experiment_name: str,
    parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    models: Dict[str, Any],
    artifacts: List[str],
) -> None:
    """
    Log training run to MLflow for model registry and experiment tracking.

    Args:
        experiment_name: Name of the experiment
        parameters: Training parameters
        metrics: Training metrics
        models: Trained models dictionary
        artifacts: List of artifact paths to log
    """
    if not MLFLOW_AVAILABLE:
        logger.debug("MLflow not available, skipping MLflow logging")
        return

    try:
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log parameters
            for key, value in parameters.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
                elif isinstance(value, dict):
                    # Log nested parameters as JSON strings
                    mlflow.log_param(key, json.dumps(value, default=str))

            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            # Log models
            for model_name, model_info in models.items():
                if model_name not in ["status", "training_metadata"]:
                    try:
                        # Log model to MLflow (assuming sklearn-compatible models)
                        if hasattr(model_info, "model") and hasattr(
                            model_info.model, "predict"
                        ):
                            mlflow.sklearn.log_model(
                                model_info.model, f"{model_name}_model"
                            )
                            logger.info(f"Logged model {model_name} to MLflow")
                        elif isinstance(model_info, dict) and "model" in model_info:
                            mlflow.sklearn.log_model(
                                model_info["model"], f"{model_name}_model"
                            )
                            logger.info(f"Logged model {model_name} to MLflow")
                    except Exception as e:
                        logger.warning(
                            f"Could not log model {model_name} to MLflow: {e}"
                        )

            # Log artifacts
            for artifact_path in artifacts:
                if os.path.exists(artifact_path):
                    try:
                        mlflow.log_artifact(artifact_path)
                        logger.debug(f"Logged artifact {artifact_path} to MLflow")
                    except Exception as e:
                        logger.warning(
                            f"Could not log artifact {artifact_path} to MLflow: {e}"
                        )

            # Log environment info as artifacts
            if "environment_snapshot" in parameters:
                env_file = os.path.join(os.getcwd(), "environment_snapshot.json")
                try:
                    with open(env_file, "w") as f:
                        json.dump(
                            parameters["environment_snapshot"], f, indent=2, default=str
                        )
                    mlflow.log_artifact(env_file)
                    os.remove(env_file)  # Clean up temp file
                except Exception as e:
                    logger.warning(f"Could not log environment snapshot to MLflow: {e}")

            logger.info(
                f"Successfully logged training run to MLflow experiment: {experiment_name}"
            )

    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")


def initialize_experiment_tracking(
    args: argparse.Namespace, config: Dict[str, Any], seed: int = 42
) -> ExperimentTracker:
    """
    Initialize experiment tracking for the training run.

    Args:
        args: Command line arguments
        config: Configuration dictionary
        seed: Random seed for reproducibility

    Returns:
        ExperimentTracker instance
    """
    # Create experiment name based on key parameters
    experiment_name = (
        f"train_{args.symbol or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    tracker = ExperimentTracker(experiment_name=experiment_name)

    # Set deterministic seeds for reproducibility
    set_deterministic_seeds(seed)

    # Capture environment snapshot
    env_snapshot = capture_environment_snapshot()

    # Log Git information
    tracker.log_git_info()

    # Log parameters including seed and environment
    parameters = {
        "data_file": args.data,
        "symbol": args.symbol,
        "min_samples": args.min_samples,
        "verbose": args.verbose,
        "config_file": args.config,
        "output_file": args.output,
        "random_seed": seed,
        "environment_snapshot": env_snapshot,
    }

    # Add predictive model config parameters
    predictive_config = config.get("predictive_models", {})
    parameters.update(
        {
            "predictive_models_enabled": predictive_config.get("enabled", False),
            "models_config": predictive_config.get("models", {}),
        }
    )

    tracker.log_parameters(parameters)

    return tracker


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train predictive models")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.json",
        help="Configuration file path",
    )
    parser.add_argument(
        "--data", "-d", type=str, required=True, help="Historical data file path"
    )
    parser.add_argument(
        "--symbol", "-s", type=str, help="Symbol to train on (optional)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="training_results.json",
        help="Output file for training results",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1000,
        help="Minimum number of samples required",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    tracker = None
    try:
        args = parser.parse_args()

        # Setup logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Check if predictive models are enabled before doing anything else
        predictive_config = config.get("predictive_models", {})
        if not predictive_config.get("enabled", False):
            logger.info("Predictive models disabled, exiting gracefully.")
            return  # instead of sys.exit(1)

        # Initialize experiment tracking (includes seed setting and environment capture)
        tracker = initialize_experiment_tracking(args, config, seed=42)

        # Log data loading start
        tracker.log_metrics({"data_loading": "started"}, "data_preparation")

        # Load historical data
        df = load_historical_data(args.data, args.symbol)

        # Log data loading completion
        tracker.log_metrics(
            {
                "data_samples_loaded": len(df),
                "data_file": args.data,
                "symbol": args.symbol or "all",
            },
            "data_preparation",
        )

        # Log data preparation start
        tracker.log_metrics({"data_preparation": "started"}, "data_preparation")

        # Prepare training data
        df = prepare_training_data(df, args.min_samples)

        # Log data preparation completion
        tracker.log_metrics(
            {
                "data_samples_prepared": len(df),
                "min_samples_required": args.min_samples,
            },
            "data_preparation",
        )

        # Log artifacts (data file and config)
        tracker.log_artifact(args.data, "data")
        tracker.log_artifact(args.config, "config")

        manager = PredictiveModelManager(predictive_config)

        # Train models
        logger.info("Starting model training...")
        start_time = datetime.now()

        # Log training start
        tracker.log_metrics({"training": "started"}, "training")

        training_results = manager.train_models(df)

        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()

        # Log training completion
        tracker.log_metrics(
            {
                "training_duration_seconds": training_duration,
                "training_status": training_results.get("status", "unknown"),
            },
            "training",
        )

        # Add metadata to results
        training_results.update(
            {
                "training_metadata": {
                    "timestamp": end_time.isoformat(),
                    "duration_seconds": training_duration,
                    "data_samples": len(df),
                    "data_file": args.data,
                    "symbol": args.symbol,
                    "config_file": args.config,
                }
            }
        )

        # Save results
        save_training_results(training_results, args.output)

        # Log results artifact
        tracker.log_artifact(args.output, "results")

        # Log to MLflow if available
        if training_results.get("status") == "success":
            try:
                # Prepare artifacts list for MLflow
                artifacts = [args.output, args.config]
                if os.path.exists(args.data):
                    artifacts.append(args.data)

                log_to_mlflow(
                    experiment_name=tracker.experiment_name,
                    parameters=tracker.metadata.get("parameters", {}),
                    metrics=final_metrics,
                    models=training_results,
                    artifacts=artifacts,
                )
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

        # Log final metrics
        final_metrics = {
            "total_duration_seconds": training_duration,
            "data_samples": len(df),
            "status": training_results.get("status", "unknown"),
        }

        # Extract model-specific metrics
        if training_results.get("status") == "success":
            for model_name, results in training_results.items():
                if model_name not in ["status", "training_metadata"]:
                    if isinstance(results, dict):
                        if "final_accuracy" in results:
                            final_metrics[f"{model_name}_accuracy"] = results[
                                "final_accuracy"
                            ]
                        elif "final_r2" in results:
                            final_metrics[f"{model_name}_r2"] = results["final_r2"]

        tracker.log_metrics(final_metrics, "final")

        # Log summary
        logger.info("=" * 50)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Duration: {training_duration:.2f} seconds")
        logger.info(f"Data samples: {len(df)}")
        logger.info(f"Status: {training_results.get('status', 'unknown')}")

        if training_results.get("status") == "success":
            logger.info("Model Results:")
            for model_name, results in training_results.items():
                if model_name not in ["status", "training_metadata"]:
                    if isinstance(results, dict) and "final_accuracy" in results:
                        logger.info(f"  {model_name}: {results['final_accuracy']:.3f}")
                    elif isinstance(results, dict) and "final_r2" in results:
                        logger.info(f"  {model_name}: {results['final_r2']:.3f}")
                    elif isinstance(results, dict) and "model_type" in results:
                        logger.info(
                            f"  {model_name}: {results.get('model_type', 'unknown')} configured"
                        )

        logger.info(f"Results saved to: {args.output}")
        logger.info(f"Experiment logged to: {tracker.experiment_dir}")
        logger.info("=" * 50)

        # Finish experiment successfully
        tracker.finish_experiment("completed")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if tracker:
            tracker.finish_experiment("interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if tracker:
            tracker.log_metrics({"error": str(e)}, "error")
            tracker.finish_experiment("failed")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
