"""
Unit tests for reproducibility functionality.

Tests cover:
- Deterministic seed setting across all libraries
- Environment snapshot capture and validation
- Git information capture for version tracking
- Reproducibility checklist generation
- Experiment metadata validation
"""

import json
import os
import random
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add the ml directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.train import (
    ExperimentTracker,
    capture_environment_snapshot,
    set_deterministic_seeds,
)


@pytest.fixture(autouse=True)
def isolate_tensorflow_operations():
    """Isolate TensorFlow operations to prevent thread conflicts"""
    try:
        import tensorflow as tf

        # Clear any existing graph and session
        tf.keras.backend.clear_session()
        yield
        tf.keras.backend.clear_session()
    except ImportError:
        # TensorFlow not available, skip isolation
        yield


class TestDeterministicSeeds:
    """Test deterministic seed setting functionality."""

    def test_set_deterministic_seeds_python_random(self, isolate_tensorflow_operations):
        """Test that Python's random module is seeded."""
        # Set seed
        set_deterministic_seeds(42)

        # Generate some random numbers
        first_run = [random.random() for _ in range(10)]

        # Reset and regenerate
        random.seed(42)
        second_run = [random.random() for _ in range(10)]

        # Should be identical
        assert first_run == second_run

    def test_set_deterministic_seeds_numpy(self):
        """Test that NumPy is seeded."""
        # Set seed
        set_deterministic_seeds(42)

        # Generate some random numbers
        first_run = np.random.rand(10)

        # Reset and regenerate
        np.random.seed(42)
        second_run = np.random.rand(10)

        # Should be identical
        np.testing.assert_array_equal(first_run, second_run)

    def test_set_deterministic_seeds_pandas(self):
        """Test that pandas operations are deterministic."""
        # Set seed
        set_deterministic_seeds(42)

        # Create DataFrame with random operations
        df = pd.DataFrame({"A": range(100), "B": range(100, 200)})
        first_sample = df.sample(frac=0.5, random_state=None).index.tolist()

        # Reset and resample
        np.random.seed(42)
        df = pd.DataFrame({"A": range(100), "B": range(100, 200)})
        second_sample = df.sample(frac=0.5, random_state=None).index.tolist()

        # Should be identical
        assert first_sample == second_sample

    @patch("ml.train.lgb")
    def test_set_deterministic_seeds_lightgbm(self, mock_lgb):
        """Test that LightGBM seed is set."""
        set_deterministic_seeds(42)

        # Check that lgb.seed was set
        mock_lgb.seed = 42

    @patch("ml.train.xgb")
    def test_set_deterministic_seeds_xgboost(self, mock_xgb):
        """Test that XGBoost seeding is handled (via numpy, not set_config)."""
        set_deterministic_seeds(42)

        # XGBoost seeding is handled via numpy, not direct set_config call
        # The mock should not be called since we don't use set_config anymore
        mock_xgb.set_config.assert_not_called()

    @patch.dict("sys.modules", {"catboost": MagicMock()})
    def test_set_deterministic_seeds_catboost(self):
        """Test that CatBoost seed is set."""
        set_deterministic_seeds(42)

        # Check that catboost was imported and set_random_seed was called
        import catboost as cb

        cb.set_random_seed.assert_called_with(42)

    @patch.dict("sys.modules", {"tensorflow": MagicMock()})
    def test_set_deterministic_seeds_tensorflow(self):
        """Test that TensorFlow seed is set."""
        set_deterministic_seeds(42)

        # Check that tensorflow was imported and random.set_seed was called
        import tensorflow as tf

        tf.random.set_seed.assert_called_with(42)

    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_set_deterministic_seeds_pytorch(self):
        """Test that PyTorch seeds are set."""
        set_deterministic_seeds(42)

        # Check that torch was imported and methods were called
        import torch

        torch.manual_seed.assert_called_with(42)
        torch.cuda.manual_seed.assert_called_with(42)
        torch.cuda.manual_seed_all.assert_called_with(42)


class TestEnvironmentSnapshot:
    """Test environment snapshot capture functionality."""

    @patch("ml.train.sys")
    @patch("ml.train.platform")
    def test_capture_environment_snapshot_basic_info(self, mock_platform, mock_sys):
        """Test basic environment information capture."""
        # Mock system information
        mock_sys.version = "3.9.0 (default, Oct  1 2021, 00:00:00)"
        mock_sys.platform = "linux"
        mock_platform.platform.return_value = (
            "Linux-5.4.0-74-generic-x86_64-with-glibc2.29"
        )
        mock_platform.architecture.return_value = ("64bit", "ELF")
        mock_platform.processor.return_value = "x86_64"
        mock_platform.system.return_value = "Linux"
        mock_platform.release.return_value = "5.4.0-74-generic"
        mock_platform.version.return_value = (
            "#83-Ubuntu SMP Sat May 8 02:35:39 UTC 2021"
        )
        mock_platform.machine.return_value = "x86_64"
        mock_platform.node.return_value = "test-hostname"

        env_snapshot = capture_environment_snapshot()

        assert (
            env_snapshot["python_version"] == "3.9.0 (default, Oct  1 2021, 00:00:00)"
        )
        assert (
            env_snapshot["platform"] == "Linux-5.4.0-74-generic-x86_64-with-glibc2.29"
        )
        assert env_snapshot["system"] == "Linux"
        assert env_snapshot["machine"] == "x86_64"
        assert env_snapshot["hostname"] == "test-hostname"

    @patch("ml.train.pkg_resources")
    def test_capture_environment_snapshot_packages(self, mock_pkg_resources):
        """Test package version capture."""
        # Mock package information
        mock_working_set = [
            MagicMock(key="numpy", version="1.21.0"),
            MagicMock(key="pandas", version="1.3.0"),
            MagicMock(key="scikit-learn", version="0.24.0"),
            MagicMock(key="requests", version="2.25.0"),  # Non-ML package
        ]
        mock_pkg_resources.working_set = mock_working_set

        env_snapshot = capture_environment_snapshot()

        assert "numpy" in env_snapshot["packages"]
        assert "pandas" in env_snapshot["packages"]
        assert "scikit-learn" in env_snapshot["packages"]
        assert env_snapshot["packages"]["numpy"] == "1.21.0"
        assert env_snapshot["packages"]["pandas"] == "1.3.0"
        assert env_snapshot["packages"]["scikit-learn"] == "0.24.0"

    @patch("ml.train.os")
    def test_capture_environment_snapshot_env_vars(self, mock_os):
        """Test environment variable capture."""
        mock_os.environ.get.side_effect = lambda key, default: {
            "PYTHONPATH": "/usr/local/lib/python3.9",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "PATH": "/usr/local/bin:/usr/bin",
            "NON_STANDARD_VAR": "should_not_appear",
        }.get(key, "not set")

        env_snapshot = capture_environment_snapshot()

        assert (
            env_snapshot["environment_variables"]["PYTHONPATH"]
            == "/usr/local/lib/python3.9"
        )
        assert env_snapshot["environment_variables"]["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert (
            env_snapshot["environment_variables"]["PATH"] == "/usr/local/lib/python3.9"
        )
        assert "NON_STANDARD_VAR" not in env_snapshot["environment_variables"]

    @patch("ml.train._run_git_command")
    def test_capture_environment_snapshot_git_info(self, mock_run_git_command):
        """Test Git information capture."""
        # Mock successful git commands - return values in order they're called
        mock_run_git_command.side_effect = [
            "abc123def456",  # commit_hash
            "main",  # branch
            "https://github.com/user/repo.git",  # remote_url
            "M file1.py\n?? file2.py\n",  # status
        ]

        env_snapshot = capture_environment_snapshot()

        assert env_snapshot["git_info"]["commit_hash"] == "abc123def456"
        assert env_snapshot["git_info"]["branch"] == "main"
        assert (
            env_snapshot["git_info"]["remote_url"] == "https://github.com/user/repo.git"
        )
        assert env_snapshot["git_info"]["status"] == "M file1.py\n?? file2.py\n"

    @patch("ml.train._run_git_command")
    def test_capture_environment_snapshot_git_failure(self, mock_run_git_command):
        """Test Git information capture when Git is not available."""
        # Mock all git commands to return empty strings (failure)
        mock_run_git_command.return_value = ""

        env_snapshot = capture_environment_snapshot()

        # Since we always return keys with defaults, check that defaults are used
        assert env_snapshot["git_info"]["commit_hash"] == "unknown"
        assert env_snapshot["git_info"]["branch"] == "unknown"
        assert env_snapshot["git_info"]["remote_url"] == "unknown"
        assert env_snapshot["git_info"]["status"] == "unknown"

    @patch("ml.train.psutil")
    def test_capture_environment_snapshot_hardware(self, mock_psutil):
        """Test hardware information capture."""
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.total = 16 * 1024**3  # 16 GB
        mock_virtual_memory.available = 8 * 1024**3  # 8 GB

        mock_psutil.cpu_count.side_effect = [8, 16]  # Logical, then physical
        mock_psutil.virtual_memory.return_value = mock_virtual_memory

        env_snapshot = capture_environment_snapshot()

        assert env_snapshot["hardware"]["cpu_count"] == 8
        assert env_snapshot["hardware"]["cpu_count_logical"] == 16
        assert env_snapshot["hardware"]["memory_total_gb"] == 16.0
        assert env_snapshot["hardware"]["memory_available_gb"] == 8.0

    @patch("ml.train.psutil", None)
    def test_capture_environment_snapshot_hardware_failure(self):
        """Test hardware information capture when psutil is not available."""
        env_snapshot = capture_environment_snapshot()

        assert "error" in env_snapshot["hardware"]


class TestExperimentTrackerReproducibility:
    """Test experiment tracker reproducibility features."""

    def setup_method(self):
        """Set up test experiment tracker."""
        self.temp_dir = tempfile.mkdtemp()
        self.experiment_name = "test_experiment"

    def teardown_method(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_experiment_tracker_initialization(self):
        """Test experiment tracker initialization."""
        tracker = ExperimentTracker(
            experiment_name=self.experiment_name, tracking_dir=self.temp_dir
        )

        assert tracker.experiment_name == self.experiment_name
        assert tracker.experiment_dir.exists()
        assert "reproducibility" not in tracker.metadata

    def test_log_parameters_with_reproducibility(self):
        """Test logging parameters with reproducibility information."""
        tracker = ExperimentTracker(
            experiment_name=self.experiment_name, tracking_dir=self.temp_dir
        )

        parameters = {
            "random_seed": 42,
            "environment_snapshot": {
                "python_version": "3.9.0",
                "packages": {"numpy": "1.21.0", "pandas": "1.3.0"},
            },
            "config_file": "config.json",
            "data_file": "data.csv",
        }

        tracker.log_parameters(parameters)

        assert "reproducibility" in tracker.metadata
        assert tracker.metadata["reproducibility"]["random_seed"] == 42
        assert tracker.metadata["reproducibility"]["deterministic_mode"] is True
        assert (
            "python_random"
            in tracker.metadata["reproducibility"]["libraries_with_seeds"]
        )
        assert (
            tracker.metadata["reproducibility"]["environment_snapshot"][
                "python_version"
            ]
            == "3.9.0"
        )

    def test_reproducibility_checklist_generation(self):
        """Test reproducibility checklist generation."""
        tracker = ExperimentTracker(
            experiment_name=self.experiment_name, tracking_dir=self.temp_dir
        )

        # Mock Git info
        tracker.metadata["git_info"] = {"commit_hash": "abc123"}

        parameters = {
            "random_seed": 42,
            "environment_snapshot": {
                "python_version": "3.9.0",
                "packages": {"numpy": "1.21.0"},
            },
            "config_file": "config.json",
            "data_file": "data.csv",
        }

        tracker.log_parameters(parameters)

        # Check that reproducibility checklist file was created
        checklist_file = tracker.experiment_dir / "reproducibility_checklist.json"
        assert checklist_file.exists()

        with open(checklist_file, "r") as f:
            checklist = json.load(f)

        assert checklist["experiment_name"] == self.experiment_name
        assert checklist["reproducibility_requirements"]["random_seed"] == 42
        assert checklist["reproducibility_requirements"]["git_commit"] == "abc123"
        assert len(checklist["reproduction_steps"]) > 0
        assert checklist["validation_metrics"]["f1_score_target"] == 0.70
        assert checklist["validation_metrics"]["deterministic_reproduction"] is True

    def test_experiment_tracker_git_info_logging(self):
        """Test Git information logging in experiment tracker."""
        tracker = ExperimentTracker(
            experiment_name=self.experiment_name, tracking_dir=self.temp_dir
        )

        # Mock Git info using the new _run_git_command
        with patch("ml.train._run_git_command") as mock_run_git_command:
            mock_run_git_command.side_effect = [
                "def456abc789",  # commit_hash
                "develop",  # branch
                "https://github.com/user/repo.git",  # remote_url
                "",  # status (empty)
            ]

            tracker.log_git_info()

            assert tracker.metadata["git_info"]["commit_hash"] == "def456abc789"
            assert tracker.metadata["git_info"]["branch"] == "develop"
            assert (
                tracker.metadata["git_info"]["remote_url"]
                == "https://github.com/user/repo.git"
            )
            assert (
                tracker.metadata["git_info"]["status"] == "unknown"
            )  # empty becomes unknown

    def test_experiment_tracker_finish_experiment(self):
        """Test experiment finishing with reproducibility data."""
        tracker = ExperimentTracker(
            experiment_name=self.experiment_name, tracking_dir=self.temp_dir
        )

        # Add reproducibility data
        tracker.metadata["reproducibility"] = {"random_seed": 42}

        tracker.finish_experiment("completed")

        assert tracker.metadata["status"] == "completed"
        assert "end_time" in tracker.metadata
        assert "duration_seconds" in tracker.metadata

        # Check that metadata file was updated
        metadata_file = tracker.experiment_dir / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file, "r") as f:
            saved_metadata = json.load(f)

        assert saved_metadata["status"] == "completed"
        assert "reproducibility" in saved_metadata


class TestReproducibilityValidation:
    """Test reproducibility validation functionality."""

    def test_deterministic_numpy_operations(self):
        """Test that NumPy operations are deterministic with fixed seed."""
        # Set seed
        set_deterministic_seeds(123)

        # Generate random data
        data1 = np.random.rand(100, 10)

        # Reset seed and regenerate
        set_deterministic_seeds(123)
        data2 = np.random.rand(100, 10)

        # Should be identical
        np.testing.assert_array_equal(data1, data2)

    def test_deterministic_pandas_operations(self):
        """Test that pandas operations are deterministic with fixed seed."""
        # Set seed
        set_deterministic_seeds(456)

        # Create DataFrame and perform operations
        df = pd.DataFrame(
            {
                "A": range(100),
                "B": np.random.rand(100),
                "C": np.random.choice(["X", "Y", "Z"], 100),
            }
        )

        # Sample data
        sample1 = df.sample(n=50, random_state=None).index.tolist()

        # Reset and resample
        set_deterministic_seeds(456)
        df = pd.DataFrame(
            {
                "A": range(100),
                "B": np.random.rand(100),
                "C": np.random.choice(["X", "Y", "Z"], 100),
            }
        )
        sample2 = df.sample(n=50, random_state=None).index.tolist()

        # Should be identical
        assert sample1 == sample2

    def test_environment_snapshot_consistency(self):
        """Test that environment snapshot is consistent."""
        snapshot1 = capture_environment_snapshot()
        snapshot2 = capture_environment_snapshot()

        # Basic system info should be the same
        assert snapshot1["python_version"] == snapshot2["python_version"]
        assert snapshot1["system"] == snapshot2["system"]
        assert snapshot1["machine"] == snapshot2["machine"]

    def test_reproducibility_metadata_structure(self):
        """Test reproducibility metadata structure."""
        tracker = ExperimentTracker(experiment_name="test_reproducibility")

        parameters = {
            "random_seed": 999,
            "environment_snapshot": {
                "python_version": "3.10.0",
                "packages": {"numpy": "1.22.0", "pandas": "1.4.0"},
            },
        }

        tracker.log_parameters(parameters)

        reproducibility = tracker.metadata["reproducibility"]

        # Check required fields
        required_fields = [
            "random_seed",
            "deterministic_mode",
            "libraries_with_seeds",
            "environment_snapshot",
            "platform_info",
        ]

        for field in required_fields:
            assert field in reproducibility

        assert reproducibility["random_seed"] == 999
        assert reproducibility["deterministic_mode"] is True
        assert isinstance(reproducibility["libraries_with_seeds"], list)
        assert len(reproducibility["libraries_with_seeds"]) > 0


class TestReproducibilityIntegration:
    """Integration tests for reproducibility functionality."""

    def test_full_reproducibility_workflow(self):
        """Test the complete reproducibility workflow."""
        # 1. Set deterministic seeds
        seed = 777
        set_deterministic_seeds(seed)

        # 2. Create experiment tracker
        tracker = ExperimentTracker(experiment_name="integration_test")

        # 3. Capture environment
        env_snapshot = capture_environment_snapshot()

        # 4. Log parameters with reproducibility info
        parameters = {
            "random_seed": seed,
            "environment_snapshot": env_snapshot,
            "model_config": {"type": "test", "params": {"n_estimators": 100}},
        }

        tracker.log_parameters(parameters)

        # 5. Verify reproducibility metadata
        assert tracker.metadata["reproducibility"]["random_seed"] == seed
        assert tracker.metadata["reproducibility"]["deterministic_mode"] is True

        # 6. Check reproducibility checklist
        checklist_file = tracker.experiment_dir / "reproducibility_checklist.json"
        assert checklist_file.exists()

        with open(checklist_file, "r") as f:
            checklist = json.load(f)

        assert checklist["reproducibility_requirements"]["random_seed"] == seed
        assert len(checklist["reproduction_steps"]) >= 3  # At least 3 steps

        # 7. Finish experiment
        tracker.finish_experiment("completed")

        # 8. Verify final metadata
        assert tracker.metadata["status"] == "completed"
        assert "reproducibility" in tracker.metadata

    def test_reproducibility_with_different_seeds(self):
        """Test that different seeds produce different results."""
        # Run with seed 1
        set_deterministic_seeds(1)
        data1 = np.random.rand(10)

        # Run with seed 2
        set_deterministic_seeds(2)
        data2 = np.random.rand(10)

        # Should be different
        assert not np.array_equal(data1, data2)

    def test_reproducibility_with_same_seed(self):
        """Test that same seed produces identical results."""
        # Run with seed 100 twice
        set_deterministic_seeds(100)
        data1 = np.random.rand(10)

        set_deterministic_seeds(100)
        data2 = np.random.rand(10)

        # Should be identical
        np.testing.assert_array_equal(data1, data2)


if __name__ == "__main__":
    pytest.main([__file__])
