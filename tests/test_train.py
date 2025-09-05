"""
Tests for train.py - Predictive model training script.
"""

import pytest
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import argparse

from train import (
    load_historical_data,
    prepare_training_data,
    save_training_results,
    main
)


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

    @patch('train.load_config')
    @patch('train.load_historical_data')
    @patch('train.prepare_training_data')
    @patch('train.PredictiveModelManager')
    @patch('train.save_training_results')
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

    @patch('train.load_config')
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

    @patch('train.load_config')
    @patch('train.load_historical_data')
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


if __name__ == '__main__':
    pytest.main([__file__])
