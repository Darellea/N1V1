"""
Dataset Versioning System

Manages versioning of datasets to maintain clean history of changes
and ensure backward compatibility.
"""

import os
import json
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import hashlib
import numpy as np

logger = logging.getLogger(__name__)


class PathTraversalError(Exception):
    """Raised when path traversal is detected in version names."""

    pass


class DataValidationError(Exception):
    """Raised when DataFrame validation fails."""

    pass


class MetadataError(Exception):
    """Raised when metadata loading fails and cannot be recovered."""

    pass


def validate_dataframe(
    df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None
) -> None:
    """
    Validate DataFrame schema and content.

    Args:
        df: DataFrame to validate
        schema: Optional schema definition with column requirements

    Raises:
        DataValidationError: If validation fails
    """
    if df is None or df.empty:
        logger.error("DataFrame validation failed: DataFrame is None or empty")
        raise DataValidationError("DataFrame is None or empty")

    # Default schema for OHLCV data
    if schema is None:
        schema = {
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
            "column_types": {
                "timestamp": ["datetime64[ns]", "object"],
                "open": ["float64", "int64"],
                "high": ["float64", "int64"],
                "low": ["float64", "int64"],
                "close": ["float64", "int64"],
                "volume": ["float64", "int64"],
            },
            "constraints": {
                "no_nan_in_key_columns": ["timestamp", "open", "high", "low", "close"],
                "positive_values": ["volume"],
                "logical_price_order": True,  # high >= low, close/open reasonable
            },
        }

    # Check required columns
    required_cols = schema.get("required_columns", [])
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    # Check column types
    column_types = schema.get("column_types", {})
    for col, expected_types in column_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if actual_type not in expected_types:
                logger.warning(
                    f"Column '{col}' has type {actual_type}, expected one of {expected_types}"
                )

    # Check constraints
    constraints = schema.get("constraints", {})

    # No NaN in key columns
    no_nan_cols = constraints.get("no_nan_in_key_columns", [])
    for col in no_nan_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                raise DataValidationError(
                    f"Column '{col}' contains {nan_count} NaN values"
                )

    # Positive values constraint
    positive_cols = constraints.get("positive_values", [])
    for col in positive_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                raise DataValidationError(
                    f"Column '{col}' contains {negative_count} negative values"
                )

    # Logical price order constraint
    if constraints.get("logical_price_order", False):
        if all(col in df.columns for col in ["high", "low", "open", "close"]):
            invalid_high_low = (df["high"] < df["low"]).sum()
            if invalid_high_low > 0:
                raise DataValidationError(
                    f"Found {invalid_high_low} rows where high < low"
                )

            # Check for extreme price deviations (optional, can be made configurable)
            extreme_deviations = 0
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    # Flag prices that deviate more than 50% from median in a rolling window
                    median_price = (
                        df[col].rolling(window=min(100, len(df)), center=True).median()
                    )
                    deviation = abs(df[col] - median_price) / median_price
                    extreme_deviations += (deviation > 0.5).sum()

            if extreme_deviations > len(df) * 0.01:  # More than 1% extreme deviations
                logger.warning(
                    f"Found {extreme_deviations} extreme price deviations, may indicate data quality issues"
                )

    logger.debug(
        f"DataFrame validation passed for {len(df)} rows, {len(df.columns)} columns"
    )


class DatasetVersionManager:
    """
    Manages dataset versioning and maintains clean history of dataset changes.
    """

    def __init__(self, base_path="data/versions", legacy_mode: bool = False):
        """
        Initialize the dataset version manager.

        Args:
            base_path: Base directory for storing versioned datasets or config dict
            legacy_mode: If True, return True instead of version IDs for backward compatibility
        """
        if isinstance(base_path, str):
            self.base_path = Path(base_path)
        else:
            self.base_path = Path(base_path["versioning"]["base_dir"])
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_path / "version_metadata.json"
        self.legacy_mode = legacy_mode
        self._load_metadata()

    def _load_metadata(self):
        """
        Load version metadata from disk with proper error handling and recovery.

        Handles specific exceptions:
        - FileNotFoundError: Log warning and initialize with default metadata
        - JSONDecodeError: Log error, attempt backup recovery, raise MetadataError if failed
        - Other exceptions: Log error and re-raise as MetadataError
        """
        if not self.metadata_file.exists():
            logger.warning(
                f"Metadata file not found: {self.metadata_file}. Initializing with default metadata."
            )
            self.metadata = {}
            return

        try:
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
            logger.info(
                f"Successfully loaded metadata with {len(self.metadata)} versions from {self.metadata_file}"
            )
        except FileNotFoundError as e:
            logger.warning(
                f"Metadata file not found during loading: {self.metadata_file}. Initializing with default metadata."
            )
            self.metadata = {}
        except json.JSONDecodeError as e:
            logger.error(
                f"Corrupted metadata file detected: {self.metadata_file}. Error: {str(e)}"
            )
            # Attempt backup recovery
            if not self._attempt_backup_recovery():
                error_msg = f"Metadata corruption unrecoverable: {self.metadata_file}. Error: {str(e)}"
                logger.error(error_msg)
                raise MetadataError(error_msg)
        except Exception as e:
            logger.error(
                f"Unexpected error loading metadata from {self.metadata_file}: {str(e)}"
            )
            raise MetadataError(f"Failed to load metadata: {str(e)}")

    def _attempt_backup_recovery(self) -> bool:
        """
        Attempt to recover metadata from backup file.

        Returns:
            True if recovery successful, False otherwise
        """
        backup_file = self.metadata_file.with_suffix(".bak")
        if not backup_file.exists():
            logger.warning(f"No backup metadata file found: {backup_file}")
            return False

        try:
            with open(backup_file, "r") as f:
                self.metadata = json.load(f)
            logger.info(f"Successfully recovered metadata from backup: {backup_file}")
            # Save the recovered metadata as the main file
            self._save_metadata()
            return True
        except Exception as e:
            logger.error(f"Failed to recover from backup {backup_file}: {str(e)}")
            return False

    def _save_metadata(self):
        """Save version metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save version metadata: {e}")

    def _sanitize_version_name(self, name: str) -> str:
        """
        Sanitize version name to prevent path traversal attacks.

        Args:
            name: Version name to sanitize

        Returns:
            Sanitized version name

        Raises:
            PathTraversalError: If path traversal is detected
        """
        if not name or not isinstance(name, str):
            raise PathTraversalError("Version name must be a non-empty string")

        # Check for absolute path patterns (Windows and Unix)
        if (
            name.startswith("/")
            or name.startswith("\\")
            or (len(name) >= 3 and name[1:3] == ":\\")
        ):
            logger.error(f"Absolute path detected in version name: {name}")
            raise PathTraversalError("Absolute path detected")

        # Check for path traversal patterns
        if ".." in name:
            logger.error(f"Path traversal detected in version name: {name}")
            raise PathTraversalError(f"Path traversal detected in version name: {name}")

        if "/" in name or "\\" in name:
            logger.error(f"Path separators not allowed in version name: {name}")
            raise PathTraversalError(
                f"Path separators not allowed in version name: {name}"
            )

        # Allow only alphanumeric characters, underscores, and hyphens
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            logger.error(f"Invalid characters in version name: {name}")
            raise PathTraversalError("Invalid characters in version name")

        # Ensure name is not too long (prevent filesystem issues)
        if len(name) > 100:
            logger.error(f"Version name too long: {name}")
            raise PathTraversalError(
                f"Version name too long (max 100 characters): {name}"
            )

        logger.debug(f"Version name sanitized successfully: {name}")
        return name

    def create_version(
        self,
        df,
        version_name: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[str, bool]:
        logger.info(
            "Starting dataset version creation",
            extra={
                "version_name": version_name,
                "description": description,
                "data_shape": df.shape if hasattr(df, "shape") else None,
            },
        )

        # Validate DataFrame
        df_to_validate = df.reset_index()
        if "timestamp" not in df_to_validate.columns:
            raise DataValidationError("DataFrame must contain 'timestamp' column")

        # Sanitize version name to prevent path traversal
        sanitized_name = self._sanitize_version_name(version_name)

        version_id = f"{sanitized_name}_{uuid.uuid4().hex[:8]}"

        version_dir = self.base_path / sanitized_name
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset (reset index to ensure consistent structure)
        df.reset_index().to_json(
            version_dir / "data.json", orient="records", date_format="iso"
        )

        # Save metadata
        version_metadata = {
            "version_id": version_id,
            "name": sanitized_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "validation_passed": True,
            "metadata": metadata or {},
        }
        self.metadata[version_id] = version_metadata
        self._save_metadata()

        logger.info(
            "Dataset version created successfully",
            extra={
                "version_name": version_name,
                "version_id": version_id,
                "version_dir": str(version_dir),
            },
        )

        # Return True for legacy compatibility, version_id for new code
        return True if self.legacy_mode else version_id

    def load_version(self, version_id: str) -> Optional[pd.DataFrame]:
        """
        Load a specific version of the dataset.

        Args:
            version_id: Version identifier

        Returns:
            DataFrame if version exists, None otherwise
        """
        # Extract version name from version_id (format: "name_uuid")
        version_parts = version_id.rsplit("_", 1)
        if len(version_parts) != 2:
            logger.error(f"Invalid version_id format: {version_id}")
            return None

        version_name = version_parts[0]
        version_path = self.base_path / version_name
        dataset_file = version_path / "data.json"

        if not dataset_file.exists():
            logger.error(f"Version {version_id} not found at {dataset_file}")
            return None

        try:
            df = pd.read_json(dataset_file, orient="records")
            # Drop the index column if it exists (artifact from reset_index())
            if "index" in df.columns:
                df = df.drop("index", axis=1)
            # Convert timestamp strings back to datetime if needed
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df.set_index("timestamp", inplace=True)
            logger.info(f"Loaded dataset version: {version_id}")
            return df
        except Exception as e:
            logger.error(f"Failed to load version {version_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load version {version_id}: {e}")
            return None

    def get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific version.

        Args:
            version_id: Version identifier

        Returns:
            Version information dictionary
        """
        return self.metadata.get(version_id)

    def list_versions(self) -> List[str]:
        """
        List all available versions.

        Returns:
            List of version IDs
        """
        return list(self.metadata.keys())

    def get_latest_version(self, version_name: Optional[str] = None) -> Optional[str]:
        """
        Get the latest version, optionally filtered by version name.

        Args:
            version_name: Filter by version name

        Returns:
            Latest version ID
        """
        versions = self.list_versions()

        if version_name:
            versions = [v for v in versions if v.startswith(f"{version_name}_")]

        if not versions:
            return None

        # Sort by timestamp (embedded in version ID)
        versions.sort(reverse=True)
        return versions[0]

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        Compare two versions of the dataset.

        Args:
            version_id1: First version ID
            version_id2: Second version ID

        Returns:
            Comparison results
        """
        info1 = self.get_version_info(version_id1)
        info2 = self.get_version_info(version_id2)

        if not info1 or not info2:
            return {"error": "One or both versions not found"}

        comparison = {
            "version1": version_id1,
            "version2": version_id2,
            "shape_diff": {
                "rows": info2["shape"][0] - info1["shape"][0],
                "cols": info2["shape"][1] - info1["shape"][1],
            },
            "column_changes": {
                "added": list(set(info2["columns"]) - set(info1["columns"])),
                "removed": list(set(info1["columns"]) - set(info2["columns"])),
            },
            "hash_changed": info1["hash"] != info2["hash"],
        }

        return comparison

    def _calculate_dataframe_hash(
        self, df: pd.DataFrame, use_full_hash: bool = False
    ) -> str:
        """
        Calculate a hash of the DataFrame for change detection.

        Args:
            df: DataFrame to hash
            use_full_hash: If True, hash the entire dataset. If False, use deterministic sampling.

        Returns:
            Hash string
        """
        if use_full_hash or len(df) <= 10000:
            # Hash the entire dataset for small datasets or when explicitly requested
            sample = df
            logger.debug(f"Using full hash for dataset with {len(df)} rows")
        else:
            # Use deterministic sampling for large datasets
            # Sample every nth row to ensure deterministic and representative coverage
            sample_size = min(10000, len(df))
            step = max(1, len(df) // sample_size)

            # Get deterministic sample by taking every step-th row
            sample_indices = list(range(0, len(df), step))[:sample_size]
            sample = df.iloc[sample_indices]

            logger.debug(
                f"Using deterministic sampling: {len(sample)} rows from {len(df)} total rows (step={step})"
            )

        # Convert to string representation for hashing
        data_str = sample.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()

    def cleanup_old_versions(self, keep_recent: int = 10):
        """
        Clean up old versions, keeping only the most recent ones.

        Args:
            keep_recent: Number of recent versions to keep
        """
        versions = self.list_versions()

        if len(versions) <= keep_recent:
            return

        # Sort by timestamp (embedded in version ID)
        versions.sort(reverse=True)

        # Keep recent versions
        versions_to_keep = versions[:keep_recent]
        versions_to_remove = versions[keep_recent:]

        # Remove old versions
        for version_id in versions_to_remove:
            version_path = self.base_path / version_id
            try:
                import shutil

                shutil.rmtree(version_path)
                del self.metadata[version_id]
                logger.info(f"Removed old version: {version_id}")
            except Exception as e:
                logger.warning(f"Failed to remove version {version_id}: {e}")

        # Save updated metadata
        self._save_metadata()

    def migrate_legacy_dataset(
        self, legacy_path: str, new_name: str, description: str
    ) -> Union[str, bool]:
        logger.info(
            "Starting legacy dataset migration",
            extra={
                "legacy_path": legacy_path,
                "new_name": new_name,
                "description": description,
            },
        )

        try:
            with open(legacy_path, "r") as f:
                legacy_data = json.load(f)

            df = pd.DataFrame(legacy_data.get("data", []))
            if "timestamp" not in df.columns:
                # Attempt to find a timestamp column or create one if possible
                if "date" in df.columns:
                    df.rename(columns={"date": "timestamp"}, inplace=True)
                else:
                    # Fallback: if no obvious date column, this will fail validation in create_version
                    pass

            version_id = self.create_version(
                df=df,
                version_name=new_name,
                description=description,
                metadata={"migrated_from": legacy_path},
            )

            logger.info(
                "Legacy dataset migration completed", extra={"version_id": version_id}
            )

            # Return True for legacy compatibility, version_id for new code
            return True if self.legacy_mode else version_id
        except Exception as e:
            logger.error(f"Legacy dataset migration failed: {e}", exc_info=True)
            raise


def create_binary_target_dataset(
    df: pd.DataFrame,
    version_manager: DatasetVersionManager,
    horizon: int = 5,
    profit_threshold: float = 0.005,
    include_fees: bool = True,
    fee_rate: float = 0.001,
) -> str:
    """
    Create a new dataset version with binary labels for trading decisions.

    Args:
        df: Input DataFrame with OHLCV data
        version_manager: Dataset version manager
        horizon: Number of periods ahead to look for forward return
        profit_threshold: Minimum profit threshold after fees (fractional)
        include_fees: Whether to account for trading fees
        fee_rate: Trading fee rate (fractional)

    Returns:
        Version ID of the created dataset
    """
    from ml.trainer import create_binary_labels

    # Create binary labels
    df_with_labels = create_binary_labels(
        df=df,
        horizon=horizon,
        profit_threshold=profit_threshold,
        include_fees=include_fees,
        fee_rate=fee_rate,
    )

    # Create version metadata
    metadata = {
        "label_type": "binary",
        "horizon": horizon,
        "profit_threshold": profit_threshold,
        "include_fees": include_fees,
        "fee_rate": fee_rate,
        "label_distribution": df_with_labels["label_binary"].value_counts().to_dict(),
    }

    # Create version
    version_id = version_manager.create_version(
        df=df_with_labels,
        version_name="binary_labels_v2",
        description="Dataset with binary trading decision labels (1=trade, 0=skip)",
        metadata=metadata,
    )

    return version_id


def migrate_legacy_dataset(
    df: pd.DataFrame,
    version_manager: DatasetVersionManager,
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Migrate a legacy dataset to the new binary labeling system.

    Args:
        df: Legacy DataFrame with old labeling
        version_manager: Dataset version manager
        schema: Optional schema for DataFrame validation

    Returns:
        Version ID of the migrated dataset

    Raises:
        DataValidationError: If DataFrame validation fails
    """
    start_time = time.time()
    logger.info(
        f"Starting legacy dataset migration: shape={df.shape}, columns={list(df.columns)}"
    )

    try:
        # Ensure DataFrame has timestamp column for validation (reset index if necessary)
        df_for_validation = df.copy()
        if "timestamp" not in df_for_validation.columns:
            if isinstance(
                df_for_validation.index, pd.DatetimeIndex
            ) or pd.api.types.is_datetime64_any_dtype(df_for_validation.index):
                df_for_validation = df_for_validation.reset_index()
                if df_for_validation.columns[0] != "timestamp":
                    df_for_validation.rename(
                        columns={df_for_validation.columns[0]: "timestamp"},
                        inplace=True,
                    )
            else:
                raise DataValidationError(
                    "DataFrame must have a 'timestamp' column or DatetimeIndex"
                )

        # Validate input DataFrame
        try:
            validate_dataframe(df_for_validation, schema)
            logger.debug("Input DataFrame validation passed for migration")
        except DataValidationError as e:
            logger.error(
                f"DataFrame validation failed for legacy dataset migration: {str(e)}"
            )
            raise

        # Keep the original data but add binary labels
        df_migrated = df.copy()

        # If binary labels don't exist, create them
        if "label_binary" not in df_migrated.columns:
            logger.info("Adding binary labels to legacy dataset")
            from ml.trainer import create_binary_labels

            df_migrated = create_binary_labels(df_migrated)
            logger.debug(
                f"Binary labels added: {df_migrated['label_binary'].value_counts().to_dict()}"
            )
        else:
            logger.debug("Binary labels already present in legacy dataset")

        # Reset index so that timestamp is included as a column for schema validation
        df_migrated = (
            df_migrated.reset_index()
            if "timestamp" not in df_migrated.columns
            and df_migrated.index.name == "timestamp"
            else df_migrated
        )

        # Validate migrated DataFrame as well
        try:
            validate_dataframe(df_migrated, schema)
            logger.debug("Migrated DataFrame validation passed")
        except DataValidationError as e:
            logger.error(f"DataFrame validation failed for migrated dataset: {str(e)}")
            raise

        # Create version metadata
        metadata = {
            "migration_type": "legacy_to_binary",
            "original_columns": list(df.columns),
            "has_binary_labels": "label_binary" in df_migrated.columns,
            "validation_passed": True,
        }

        # Create version
        version_id = version_manager.create_version(
            df=df_migrated,
            version_name="migrated_v2",
            description="Migrated legacy dataset with binary labels",
            metadata=metadata,
        )

        duration = time.time() - start_time
        logger.info(
            f"Legacy dataset migration completed: {version_id}, duration={duration:.2f}s"
        )

        return version_id

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Legacy dataset migration failed: {str(e)}, duration={duration:.2f}s"
        )
        raise


class DatasetDriftDetector:
    """
    Detect drift in datasets using statistical methods and feature distribution comparison.
    """

    def __init__(self, version_manager: DatasetVersionManager):
        """
        Initialize the drift detector.

        Args:
            version_manager: Dataset version manager instance
        """
        self.version_manager = version_manager
        self.reference_datasets = {}  # Cache for loaded reference datasets

    def detect_drift(
        self,
        current_df: pd.DataFrame,
        reference_version_id: str,
        features_to_check: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Detect drift between current data and a reference dataset version.

        Args:
            current_df: Current dataset to check for drift
            reference_version_id: Version ID of reference dataset
            features_to_check: Specific features to check (None for all numeric features)

        Returns:
            Drift detection results
        """
        # Load reference dataset
        if reference_version_id not in self.reference_datasets:
            ref_df = self.version_manager.load_version(reference_version_id)
            if ref_df is None:
                return {"error": f"Reference version {reference_version_id} not found"}
            self.reference_datasets[reference_version_id] = ref_df

        ref_df = self.reference_datasets[reference_version_id]

        # Determine features to check
        if features_to_check is None:
            # Use common numeric features
            ref_numeric = ref_df.select_dtypes(include=[np.number]).columns
            curr_numeric = current_df.select_dtypes(include=[np.number]).columns
            features_to_check = list(set(ref_numeric) & set(curr_numeric))

        drift_results = {}

        for feature in features_to_check:
            if feature in ref_df.columns and feature in current_df.columns:
                ref_values = ref_df[feature].dropna().values
                curr_values = current_df[feature].dropna().values

                if len(ref_values) == 0 or len(curr_values) == 0:
                    drift_results[feature] = {"error": "Insufficient data"}
                    continue

                # Kolmogorov-Smirnov test
                try:
                    from scipy.stats import ks_2samp

                    ks_stat, ks_pvalue = ks_2samp(ref_values, curr_values)
                except ImportError:
                    ks_stat, ks_pvalue = None, None

                # Population Stability Index
                psi_score = self._calculate_psi(ref_values, curr_values)

                # Distribution statistics
                ref_stats = {
                    "mean": float(np.mean(ref_values)),
                    "std": float(np.std(ref_values)),
                    "min": float(np.min(ref_values)),
                    "max": float(np.max(ref_values)),
                }

                curr_stats = {
                    "mean": float(np.mean(curr_values)),
                    "std": float(np.std(curr_values)),
                    "min": float(np.min(curr_values)),
                    "max": float(np.max(curr_values)),
                }

                # Drift detection
                drift_detected = (
                    ks_pvalue is not None and ks_pvalue < 0.05
                ) or psi_score > 0.25

                drift_results[feature] = {
                    "drift_detected": drift_detected,
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "psi_score": float(psi_score),
                    "reference_stats": ref_stats,
                    "current_stats": curr_stats,
                    "distribution_change": self._analyze_distribution_change(
                        ref_stats, curr_stats
                    ),
                }

        # Overall drift assessment
        total_features = len(drift_results)
        drifted_features = sum(
            1
            for r in drift_results.values()
            if isinstance(r, dict) and r.get("drift_detected", False)
        )

        overall_assessment = {
            "total_features_checked": total_features,
            "drifted_features": drifted_features,
            "drift_percentage": drifted_features / total_features
            if total_features > 0
            else 0,
            "overall_drift_detected": drifted_features
            > total_features * 0.1,  # >10% features drifted
            "feature_results": drift_results,
        }

        return overall_assessment

    def _calculate_psi(
        self, reference: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        """Calculate Population Stability Index."""
        try:
            ref_hist, bin_edges = np.histogram(reference, bins=bins, density=True)
            curr_hist, _ = np.histogram(current, bins=bin_edges, density=True)

            # Avoid division by zero
            ref_hist = ref_hist + 1e-10
            curr_hist = curr_hist + 1e-10

            psi = np.sum((curr_hist - ref_hist) * np.log(curr_hist / ref_hist))
            return float(psi)
        except Exception:
            return 0.0

    def _analyze_distribution_change(self, ref_stats: Dict, curr_stats: Dict) -> str:
        """Analyze the type of distribution change."""
        mean_change = (
            abs(curr_stats["mean"] - ref_stats["mean"]) / abs(ref_stats["mean"])
            if ref_stats["mean"] != 0
            else 0
        )
        std_change = (
            abs(curr_stats["std"] - ref_stats["std"]) / abs(ref_stats["std"])
            if ref_stats["std"] != 0
            else 0
        )

        if mean_change > 0.1 and std_change < 0.1:
            return "mean_shift"
        elif std_change > 0.1 and mean_change < 0.1:
            return "variance_shift"
        elif mean_change > 0.1 and std_change > 0.1:
            return "mean_and_variance_shift"
        else:
            return "minor_change"


def create_binary_labels(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Stub implementation of create_binary_labels.

    This is a placeholder function that returns the input DataFrame unchanged.
    In a real implementation, this would add binary labels for trading decisions.

    Args:
        df: Input DataFrame
        **kwargs: Additional arguments (ignored in stub)

    Returns:
        DataFrame with binary labels (unchanged for now)
    """
    # Stub implementation: return input DataFrame unchanged
    return df


def create_dataset_drift_detector(
    version_manager: DatasetVersionManager,
) -> DatasetDriftDetector:
    """
    Create a dataset drift detector instance.

    Args:
        version_manager: Dataset version manager

    Returns:
        DatasetDriftDetector instance
    """
    return DatasetDriftDetector(version_manager)
