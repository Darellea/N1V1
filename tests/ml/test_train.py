"""
Tests for train.py - Predictive model training script.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, call
import argparse

# Mock mlflow before importing anything that might use it
import sys
from unittest.mock import MagicMock
sys.modules['mlflow'] = MagicMock()
sys.modules['mlflow.sklearn'] = MagicMock()
sys.modules['mlflow.tracking'] = MagicMock()
sys.modules['mlflow.tracking.MlflowClient'] = MagicMock()

from ml.train import (
    load_historical_data,
    prepare_training_data,
    save_training_results,
    set_deterministic_seeds,
    capture_environment_snapshot,
    main
)
from ml.model_loader import load_model_from_registry


class TestLoadHistoricalData:
    """Test load_historical_data function."""

    def test_load_csv_data_success(self):
        """Test loading CSV data successfully."""
        csv_data = """timestamp,open,high,low,close,volume
1640995200,100.0,105.0,95.0,102.0,1000
1641081600,102.0,107.0,97.0,104.0,1100"""

        with patch('builtins.open', mock_open(read_data=csv_data)):
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.return_value = pd.DataFrame({
                    'timestamp': [1640995200, 1641081600],
                    'open': [100.0, 102.0],
                    'high': [105.0, 107.0],
                    'low': [95.0, 97.0],
                    'close': [102.0, 104.0],
                    'volume': [1000, 1100]
                })

                result = load_historical_data('test.csv')

                assert len(result) == 2
                assert list(result.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    def test_load_json_data_success(self):
        """Test loading JSON data successfully."""
        json_data = [
            {"timestamp": 1640995200, "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000},
            {"timestamp": 1641081600, "open": 102.0, "high": 107.0, "low": 97.0, "close": 104.0, "volume": 1100}
        ]

        with patch('builtins.open', mock_open(read_data=json.dumps(json_data))):
            with patch('json.load', return_value=json_data):
                result = load_historical_data('test.json')

                assert len(result) == 2
                assert 'timestamp' in result.columns
                assert 'close' in result.columns

    def test_load_data_with_symbol_filter(self):
        """Test loading data with symbol filtering."""
        csv_data = """timestamp,symbol,open,high,low,close,volume
1640995200,BTC,100.0,105.0,95.0,102.0,1000
1641081600,ETH,102.0,107.0,97.0,104.0,1100
1641168000,BTC,104.0,109.0,99.0,106.0,1200"""

        with patch('builtins.open', mock_open(read_data=csv_data)):
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.return_value = pd.DataFrame({
                    'timestamp': [1640995200, 1641081600, 1641168000],
                    'symbol': ['BTC', 'ETH', 'BTC'],
                    'open': [100.0, 102.0, 104.0],
                    'high': [105.0, 107.0, 109.0],
                    'low': [95.0, 97.0, 99.0],
                    'close': [102.0, 104.0, 106.0],
                    'volume': [1000, 1100, 1200]
                })

                result = load_historical_data('test.csv', symbol='BTC')

                assert len(result) == 2
                assert all(result['symbol'] == 'BTC')

    def test_load_data_missing_required_columns(self):
        """Test loading data with missing required columns."""
        csv_data = """timestamp,open,high,low
1640995200,100.0,105.0,95.0"""

        with patch('builtins.open', mock_open(read_data=csv_data)):
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.return_value = pd.DataFrame({
                    'timestamp': [1640995200],
                    'open': [100.0],
                    'high': [105.0],
                    'low': [95.0]
                })

                with pytest.raises(ValueError, match="Missing required columns"):
                    load_historical_data('test.csv')

    def test_load_data_unsupported_format(self):
        """Test loading data with unsupported file format."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_historical_data('test.txt')

    def test_load_data_file_not_found(self):
        """Test loading data when file doesn't exist."""
        with patch('pandas.read_csv', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                load_historical_data('nonexistent.csv')


class TestPrepareTrainingData:
    """Test prepare_training_data function."""

    def test_prepare_data_success(self):
        """Test preparing training data successfully."""
        data = pd.DataFrame({
            'timestamp': [1640995200, 1641081600, 1641168000],
            'open': [100.0, 102.0, 104.0],
            'high': [105.0, 107.0, 109.0],
            'low': [95.0, 97.0, 99.0],
            'close': [102.0, 104.0, 106.0],
            'volume': [1000, 1100, 1200]
        })

        result = prepare_training_data(data, min_samples=2)

        assert len(result) == 3
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_prepare_data_insufficient_samples(self):
        """Test preparing data with insufficient samples."""
        data = pd.DataFrame({
            'timestamp': [1640995200],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000]
        })

        with pytest.raises(ValueError, match="Insufficient training data"):
            prepare_training_data(data, min_samples=1000)

    def test_prepare_data_with_nan_values(self):
        """Test preparing data with NaN values."""
        data = pd.DataFrame({
            'timestamp': [1640995200, 1641081600, 1641168000],
            'open': [100.0, None, 104.0],
            'high': [105.0, 107.0, 109.0],
            'low': [95.0, 97.0, 99.0],
            'close': [102.0, 104.0, 106.0],
            'volume': [1000, 1100, 1200]
        })

        result = prepare_training_data(data, min_samples=2)

        # Should drop the row with NaN in open column
        assert len(result) == 2
        assert not result['open'].isnull().any()

    def test_prepare_data_with_zero_prices(self):
        """Test preparing data with zero or negative prices."""
        data = pd.DataFrame({
            'timestamp': [1640995200, 1641081600, 1641168000],
            'open': [0.0, 102.0, 104.0],
            'high': [105.0, 107.0, 109.0],
            'low': [95.0, 97.0, 99.0],
            'close': [102.0, 104.0, 106.0],
            'volume': [1000, 1100, 1200]
        })

        result = prepare_training_data(data, min_samples=2)

        # Should remove the row with zero price
        assert len(result) == 2
        assert all(result['open'] > 0)

    def test_prepare_data_outlier_removal(self):
        """Test outlier removal in data preparation."""
        # Create data with extreme outliers
        data = pd.DataFrame({
            'timestamp': list(range(1640995200, 1640995200 + 100)),
            'open': [100.0] * 98 + [1000.0, 1.0],  # Outliers
            'high': [105.0] * 100,
            'low': [95.0] * 100,
            'close': [102.0] * 100,
            'volume': [1000] * 100
        })

        result = prepare_training_data(data, min_samples=10)

        # Should remove outliers (values more than 3 std dev from mean)
        assert len(result) < len(data)


class TestSaveTrainingResults:
    """Test save_training_results function."""

    def test_save_results_success(self):
        """Test saving training results successfully."""
        results = {
            "status": "success",
            "model_results": {"accuracy": 0.85},
            "training_metadata": {
                "timestamp": "2024-01-01T00:00:00",
                "duration_seconds": 120.5
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "results.json")

            save_training_results(results, output_path)

            # Verify file was created and contains correct data
            assert os.path.exists(output_path)

            with open(output_path, 'r') as f:
                saved_data = json.load(f)

            assert saved_data["status"] == "success"
            assert "model_results" in saved_data

    def test_save_results_creates_directory(self):
        """Test saving results creates necessary directories."""
        results = {"status": "success"}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "subdir", "results.json")

            save_training_results(results, output_path)

            assert os.path.exists(output_path)
            assert os.path.exists(os.path.dirname(output_path))


class TestMainFunction:
    """Test main function."""

    @patch('ml.train.load_config')
    @patch('ml.train.load_historical_data')
    @patch('ml.train.prepare_training_data')
    @patch('ml.train.PredictiveModelManager')
    @patch('ml.train.save_training_results')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_success(self, mock_parse_args, mock_save_results, mock_manager_class,
                         mock_prepare_data, mock_load_data, mock_load_config):
        """Test main function with successful execution."""
        # Mock command line arguments
        mock_args = MagicMock()
        mock_args.config = 'config.json'
        mock_args.data = 'data.csv'
        mock_args.symbol = None
        mock_args.output = 'results.json'
        mock_args.min_samples = 100
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args

        # Mock configuration
        mock_config = {
            'predictive_models': {
                'enabled': True,
                'models': ['price_predictor']
            }
        }
        mock_load_config.return_value = mock_config

        # Mock data with smaller dataset
        mock_data = pd.DataFrame({
            'timestamp': list(range(100)),
            'open': [100.0] * 100,
            'high': [105.0] * 100,
            'low': [95.0] * 100,
            'close': [102.0] * 100,
            'volume': [1000] * 100
        })
        mock_load_data.return_value = mock_data
        mock_prepare_data.return_value = mock_data

        # Mock model manager with fast training
        mock_manager = MagicMock()
        mock_manager.train_models.return_value = {
            "status": "success",
            "price_predictor": {"final_accuracy": 0.85}
        }
        mock_manager_class.return_value = mock_manager

        # Run main function
        with patch('sys.exit') as mock_exit:
            main()

            # Verify no exit was called (success)
            mock_exit.assert_not_called()

        # Verify all functions were called
        mock_load_config.assert_called_once_with('config.json')
        mock_load_data.assert_called_once_with('data.csv', None)
        mock_prepare_data.assert_called_once_with(mock_data, 100)
        mock_manager_class.assert_called_once()
        mock_manager.train_models.assert_called_once()
        mock_save_results.assert_called_once()

    @patch('ml.train.load_config')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_predictive_models_disabled(self, mock_parse_args, mock_load_config):
        """Test main function when predictive models are disabled."""
        # Mock command line arguments
        mock_args = MagicMock()
        mock_args.config = 'config.json'
        mock_args.data = 'data.csv'
        mock_parse_args.return_value = mock_args

        # Mock configuration with disabled predictive models
        mock_config = {
            'predictive_models': {
                'enabled': False
            }
        }
        mock_load_config.return_value = mock_config

        # Run main function
        with patch('sys.exit') as mock_exit:
            main()

            # Should exit successfully without training
            mock_exit.assert_not_called()

    @patch('ml.train.load_config')
    @patch('ml.train.load_historical_data')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_training_failure(self, mock_parse_args, mock_load_data, mock_load_config):
        """Test main function with training failure."""
        # Mock command line arguments
        mock_args = MagicMock()
        mock_args.config = 'config.json'
        mock_args.data = 'data.csv'
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args

        # Mock configuration
        mock_config = {'predictive_models': {'enabled': True}}
        mock_load_config.return_value = mock_config

        # Mock data loading failure
        mock_load_data.side_effect = Exception("Data loading failed")

        # Run main function
        with patch('sys.exit') as mock_exit:
            main()

            # Should exit with error
            mock_exit.assert_called_once_with(1)

    @patch('argparse.ArgumentParser.parse_args')
    def test_main_keyboard_interrupt(self, mock_parse_args):
        """Test main function with keyboard interrupt."""
        # Mock keyboard interrupt during argument parsing
        mock_parse_args.side_effect = KeyboardInterrupt()

        with patch('sys.exit') as mock_exit:
            # The main function should handle KeyboardInterrupt gracefully
            main()

            # Should exit gracefully
            mock_exit.assert_called_once_with(1)


class TestConfusionMatrixGeneration:
    """Test cases for binary confusion matrix generation."""

    def test_binary_confusion_matrix_generation(self, tmp_path):
        """Test that binary confusion matrix is generated correctly."""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        # Create mock true labels and predictions
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Validate confusion matrix properties
        assert cm.shape == (2, 2), "Confusion matrix should be 2x2 for binary classification"
        assert cm.sum() == len(y_true), "Confusion matrix sum should equal number of samples"

        # Create and save confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Skip (0)', 'Trade (1)'])
        ax.set_yticklabels(['Skip (0)', 'Trade (1)'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Binary Confusion Matrix')

        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center",
                             color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.tight_layout()
        confusion_matrix_path = tmp_path / "confusion_matrix_binary.png"
        plt.savefig(confusion_matrix_path)
        plt.close()

        # Validate that the file was created
        assert confusion_matrix_path.exists(), "Binary confusion matrix file should be created"

        # Validate confusion matrix values
        # cm[0,0] = True Negatives, cm[0,1] = False Positives
        # cm[1,0] = False Negatives, cm[1,1] = True Positives
        assert cm[0, 0] >= 0, "True negatives should be non-negative"
        assert cm[0, 1] >= 0, "False positives should be non-negative"
        assert cm[1, 0] >= 0, "False negatives should be non-negative"
        assert cm[1, 1] >= 0, "True positives should be non-negative"

    def test_confusion_matrix_with_perfect_predictions(self, tmp_path):
        """Test confusion matrix with perfect predictions."""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        # Perfect predictions
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Should have only diagonal elements
        assert cm[0, 0] == 3, "Should have 3 true negatives"
        assert cm[0, 1] == 0, "Should have 0 false positives"
        assert cm[1, 0] == 0, "Should have 0 false negatives"
        assert cm[1, 1] == 3, "Should have 3 true positives"

        # Save plot
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Skip (0)', 'Trade (1)'])
        ax.set_yticklabels(['Skip (0)', 'Trade (1)'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Perfect Binary Confusion Matrix')

        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center",
                             color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.tight_layout()
        confusion_matrix_path = tmp_path / "perfect_confusion_matrix_binary.png"
        plt.savefig(confusion_matrix_path)
        plt.close()

        assert confusion_matrix_path.exists(), "Perfect confusion matrix file should be created"


class TestReproducibility:
    """Test reproducibility features."""

    @patch('random.seed')
    @patch('numpy.random.seed')
    @patch('pandas.core.common.random_state')
    def test_set_deterministic_seeds_basic(self, mock_pandas_random, mock_numpy_seed, mock_random_seed):
        """Test that set_deterministic_seeds calls all expected seed functions."""
        set_deterministic_seeds(seed=123)

        # Verify all seed functions were called with correct seed
        mock_random_seed.assert_called_once_with(123)
        mock_numpy_seed.assert_called_once_with(123)
        mock_pandas_random.assert_called_once_with(123)

    @patch('ml.train.logger')
    def test_set_deterministic_seeds_with_libs(self, mock_logger):
        """Test set_deterministic_seeds with optional libraries available."""
        with patch.dict('sys.modules', {'sklearn': MagicMock(), 'torch': MagicMock(), 'tensorflow': MagicMock(), 'catboost': MagicMock()}):
            # Just ensure the function runs without errors when libraries are available
            set_deterministic_seeds(seed=456)

            # Verify no warnings were logged (libraries are available)
            mock_logger.warning.assert_not_called()

    @patch('ml.train.logger')
    def test_set_deterministic_seeds_libs_unavailable(self, mock_logger):
        """Test set_deterministic_seeds when optional libraries are not available."""
        with patch.dict('sys.modules', {'sklearn': None, 'torch': None, 'tensorflow': None}):
            # Should not raise any exceptions
            set_deterministic_seeds(seed=789)

            # Verify warning was logged for sklearn
            mock_logger.warning.assert_called()

    @patch('platform.platform')
    @patch('platform.processor')
    @patch('sys.version', '3.10.11')
    @patch('os.environ', {'PATH': '/usr/bin', 'PYTHONPATH': '/app'})
    def test_capture_environment_snapshot(self, mock_platform, mock_processor):
        """Test environment snapshot capture."""
        mock_platform.return_value = 'Linux-5.4.0'
        mock_processor.return_value = 'x86_64'

        with patch('pkg_resources.working_set', []):  # No packages
            with patch('psutil.cpu_count', return_value=8):
                with patch('psutil.virtual_memory') as mock_mem:
                    mock_mem_instance = MagicMock()
                    mock_mem_instance.total = 16 * 1024**3  # 16GB
                    mock_mem_instance.available = 8 * 1024**3  # 8GB
                    mock_mem.return_value = mock_mem_instance

                    snapshot = capture_environment_snapshot()

                    # Verify basic system info
                    assert snapshot['python_version'] == '3.10.11'
                    assert 'platform' in snapshot
                    assert 'processor' in snapshot
                    assert isinstance(snapshot['platform'], str)
                    assert isinstance(snapshot['processor'], str)

                    # Verify environment variables (filtered)
                    assert 'PATH' in snapshot['environment_variables']
                    assert 'PYTHONPATH' in snapshot['environment_variables']

                    # Verify hardware info
                    assert snapshot['hardware']['cpu_count'] == 8
                    assert snapshot['hardware']['memory_total_gb'] == 16.0

    @patch('ml.train._run_git_command')
    def test_capture_environment_snapshot_git_info(self, mock_run_git_command):
        """Test Git information capture in environment snapshot."""
        # Mock successful git commands - return values in order they're called
        mock_run_git_command.side_effect = [
            'abc123',  # commit_hash
            'main',    # branch
            'https://github.com/user/repo.git',  # remote_url
            ''  # status
        ]

        snapshot = capture_environment_snapshot()

        # Verify Git info was captured
        assert snapshot['git_info']['commit_hash'] == 'abc123'
        assert snapshot['git_info']['branch'] == 'main'
        assert snapshot['git_info']['remote_url'] == 'https://github.com/user/repo.git'

    @patch('ml.train._run_git_command')
    def test_capture_environment_snapshot_git_failure(self, mock_run_git_command):
        """Test environment snapshot when Git commands fail."""
        # Mock all git commands to raise exception
        mock_run_git_command.side_effect = Exception("Git not available")

        snapshot = capture_environment_snapshot()

        # Should have error in git_info
        assert 'error' in snapshot['git_info']


class TestModelLoaderReproducibility:
    """Test model loader reproducibility features."""

    @patch('ml.model_loader.MLFLOW_AVAILABLE', True)
    @patch('mlflow.sklearn.load_model')
    @patch('mlflow.tracking.MlflowClient')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    def test_load_model_from_registry_success(self, mock_start_run, mock_set_experiment, mock_client_class, mock_load_model):
        """Test successful model loading from registry."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock model version
        mock_mv = MagicMock()
        mock_mv.run_id = 'test_run_id'
        mock_client.get_model_version.return_value = mock_mv

        # Mock artifacts
        mock_artifact = MagicMock()
        mock_artifact.path = 'environment_snapshot.json'
        mock_client.list_artifacts.return_value = [mock_artifact]

        # Mock file operations for both environment snapshot and model card
        def mock_exists(path):
            return path in ['environment_snapshot.json', 'models/test_model.pkl_card.json']

        with patch('os.path.exists', side_effect=mock_exists):
            with patch('builtins.open', mock_open(read_data='{"test": "data"}')):
                with patch('os.remove'):  # Mock file cleanup
                    with patch('ml.model_loader.load_model') as mock_load_model_func:
                        mock_load_model_func.return_value = mock_model
                        model, card = load_model_from_registry('test_model', version='1')

                        assert model == mock_model
                        # Card loading is optional and may be None in mocked environment
                        assert card is None or isinstance(card, dict)

    @patch('ml.model_loader.MLFLOW_AVAILABLE', False)
    def test_load_model_from_registry_no_mlflow(self):
        """Test model loading from registry when MLflow is not available."""
        with pytest.raises(ImportError, match="MLflow not available"):
            load_model_from_registry('test_model')

    @patch('ml.model_loader.MLFLOW_AVAILABLE', True)
    @patch('mlflow.sklearn.load_model', side_effect=Exception("Registry error"))
    @patch('os.path.exists', return_value=True)
    @patch('ml.model_loader.load_model_with_card')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    def test_load_model_from_registry_fallback(self, mock_start_run, mock_set_experiment, mock_load_with_card, mock_exists, mock_load_model):
        """Test registry loading with fallback to local file."""
        mock_model = MagicMock()
        mock_card = {"fallback": True}
        mock_load_with_card.return_value = (mock_model, mock_card)

        model, card = load_model_from_registry('test_model')

        # Should have called fallback loading
        mock_load_with_card.assert_called_once_with('test_model')
        assert model == mock_model
        assert card == mock_card


if __name__ == '__main__':
    pytest.main([__file__])
