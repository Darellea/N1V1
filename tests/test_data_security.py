"""
Unit tests for data module security features.

Tests path traversal prevention in data_fetcher.py and DataFrame validation
in dataset_versioning.py.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from data.data_fetcher import DataFetcher, PathTraversalError
from data.dataset_versioning import (
    DatasetVersionManager,
    validate_dataframe,
    DataValidationError,
    migrate_legacy_dataset,
    PathTraversalError as VersionPathTraversalError
)
from data.historical_loader import HistoricalDataLoader, ConfigurationError


class TestPathTraversalPrevention:
    """Test path traversal prevention in DataFetcher."""

    def test_sanitize_cache_path_valid_relative(self):
        """Test sanitizing valid relative cache paths."""
        config = {'cache_enabled': True, 'cache_dir': 'my_cache'}
        fetcher = DataFetcher(config)

        # Should resolve to data/cache/my_cache
        expected_base = os.path.join(os.getcwd(), 'data', 'cache')
        expected_path = os.path.join(expected_base, 'my_cache')

        assert fetcher.cache_dir == expected_path

    def test_sanitize_cache_path_traversal_dots(self):
        """Test blocking path traversal with .. patterns."""
        config = {'cache_enabled': True, 'cache_dir': '../../../etc/passwd'}
        fetcher = DataFetcher(config)

        with pytest.raises(PathTraversalError, match="Path traversal detected"):
            fetcher._initialize_exchange()

    def test_sanitize_cache_path_absolute_path(self):
        """Test blocking absolute paths."""
        config = {'cache_enabled': True, 'cache_dir': '/etc/passwd'}
        fetcher = DataFetcher(config)

        with pytest.raises(PathTraversalError, match="Invalid cache directory path"):
            fetcher._initialize_exchange()

    def test_sanitize_cache_path_backslash_traversal(self):
        """Test blocking backslash path traversal on Windows."""
        config = {'cache_enabled': True, 'cache_dir': '..\\..\\etc\\passwd'}
        fetcher = DataFetcher(config)

        with pytest.raises(PathTraversalError, match="Path traversal detected"):
            fetcher._initialize_exchange()

    def test_sanitize_cache_path_complex_traversal(self):
        """Test blocking complex traversal patterns."""
        config = {'cache_enabled': True, 'cache_dir': 'cache/../../../etc/passwd'}
        fetcher = DataFetcher(config)

        with pytest.raises(PathTraversalError, match="Path traversal pattern detected"):
            fetcher._initialize_exchange()

    def test_sanitize_cache_path_valid_nested(self):
        """Test allowing valid nested paths."""
        config = {'cache_enabled': True, 'cache_dir': 'nested/cache/dir'}
        fetcher = DataFetcher(config)

        expected_base = os.path.join(os.getcwd(), 'data', 'cache')
        expected_path = os.path.join(expected_base, 'nested', 'cache', 'dir')

        assert fetcher.cache_dir == expected_path

    def test_sanitize_cache_path_empty_string(self):
        """Test handling empty cache directory string."""
        config = {'cache_enabled': True, 'cache_dir': ''}
        fetcher = DataFetcher(config)

        expected_base = os.path.join(os.getcwd(), 'data', 'cache')
        expected_path = expected_base  # Should resolve to base cache dir

        assert fetcher.cache_dir == expected_path

    def test_cache_disabled_no_validation(self):
        """Test that path validation is skipped when cache is disabled."""
        config = {'cache_enabled': False, 'cache_dir': '../../../etc/passwd'}
        fetcher = DataFetcher(config)

        # Should not raise exception when cache is disabled
        assert fetcher.cache_dir == '../../../etc/passwd'


class TestDataFrameValidation:
    """Test DataFrame validation functionality."""

    def create_valid_ohlcv_df(self, rows=100):
        """Create a valid OHLCV DataFrame for testing."""
        np.random.seed(42)
        timestamps = pd.date_range('2023-01-01', periods=rows, freq='1H')

        # Generate realistic OHLCV data
        base_price = 50000
        prices = []
        for i in range(rows):
            change = np.random.normal(0, 0.02)  # 2% volatility
            base_price *= (1 + change)
            high = base_price * (1 + abs(np.random.normal(0, 0.005)))
            low = base_price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = base_price * (1 + np.random.normal(0, 0.002))
            close = base_price
            volume = np.random.uniform(100, 1000)

            prices.append({
                'timestamp': timestamps[i],
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': round(volume, 2)
            })

        return pd.DataFrame(prices)

    def test_validate_dataframe_valid_ohlcv(self):
        """Test validation of valid OHLCV DataFrame."""
        df = self.create_valid_ohlcv_df()

        # Should not raise exception
        validate_dataframe(df)

    def test_validate_dataframe_none_input(self):
        """Test validation with None input."""
        with pytest.raises(DataValidationError, match="DataFrame is None or empty"):
            validate_dataframe(None)

    def test_validate_dataframe_empty_df(self):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(DataValidationError, match="DataFrame is None or empty"):
            validate_dataframe(df)

    def test_validate_dataframe_missing_columns(self):
        """Test validation with missing required columns."""
        df = self.create_valid_ohlcv_df()
        df = df.drop('timestamp', axis=1)  # Remove required column

        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_dataframe(df)

    def test_validate_dataframe_nan_in_key_columns(self):
        """Test validation with NaN values in key columns."""
        df = self.create_valid_ohlcv_df()
        df.loc[0, 'timestamp'] = pd.NaT  # Add NaN to timestamp

        with pytest.raises(DataValidationError, match="contains.*NaN values"):
            validate_dataframe(df)

    def test_validate_dataframe_negative_volume(self):
        """Test validation with negative volume values."""
        df = self.create_valid_ohlcv_df()
        df.loc[0, 'volume'] = -100  # Negative volume

        with pytest.raises(DataValidationError, match="contains.*negative values"):
            validate_dataframe(df)

    def test_validate_dataframe_invalid_price_order(self):
        """Test validation with high < low (invalid price order)."""
        df = self.create_valid_ohlcv_df()
        # Swap high and low for first row
        temp = df.loc[0, 'high']
        df.loc[0, 'high'] = df.loc[0, 'low']
        df.loc[0, 'low'] = temp

        with pytest.raises(DataValidationError, match="high < low"):
            validate_dataframe(df)

    def test_validate_dataframe_custom_schema(self):
        """Test validation with custom schema."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': [95.5, 87.2, 91.8]
        })

        custom_schema = {
            'required_columns': ['id', 'name'],
            'column_types': {
                'id': ['int64'],
                'score': ['float64']
            },
            'constraints': {
                'positive_values': ['score']
            }
        }

        # Should not raise exception with valid custom schema
        validate_dataframe(df, custom_schema)

    def test_validate_dataframe_missing_custom_required_column(self):
        """Test validation with missing custom required column."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': [95.5, 87.2, 91.8]
        })

        custom_schema = {
            'required_columns': ['id', 'name'],  # 'id' is missing
        }

        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_dataframe(df, custom_schema)


class TestDatasetVersionManagerSecurity:
    """Test security features in DatasetVersionManager."""

    def test_create_version_with_validation(self):
        """Test create_version with DataFrame validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            # Create valid DataFrame
            df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='1H'),
                'open': [100 + i for i in range(10)],
                'high': [105 + i for i in range(10)],
                'low': [95 + i for i in range(10)],
                'close': [102 + i for i in range(10)],
                'volume': [1000 + i*10 for i in range(10)]
            })

            # Should succeed with valid DataFrame
            version_id = manager.create_version(df, "test_version", "Test version")
            assert version_id.startswith("test_version_")

            # Verify validation_passed flag in metadata
            info = manager.get_version_info(version_id)
            assert info['validation_passed'] is True

    def test_create_version_with_invalid_dataframe(self):
        """Test create_version with invalid DataFrame."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            # Create invalid DataFrame (missing required column)
            df = pd.DataFrame({
                'open': [100 + i for i in range(10)],
                'high': [105 + i for i in range(10)],
                'low': [95 + i for i in range(10)],
                'close': [102 + i for i in range(10)],
                'volume': [1000 + i*10 for i in range(10)]
            })

            # Should raise DataValidationError
            with pytest.raises(DataValidationError):
                manager.create_version(df, "test_version", "Test version")

    def test_migrate_legacy_dataset_validation(self):
        """Test migrate_legacy_dataset with validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            # Create legacy DataFrame
            df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='1H'),
                'open': [100 + i for i in range(10)],
                'high': [105 + i for i in range(10)],
                'low': [95 + i for i in range(10)],
                'close': [102 + i for i in range(10)],
                'volume': [1000 + i*10 for i in range(10)]
            })

            # Mock the create_binary_labels function
            with patch('data.dataset_versioning.create_binary_labels') as mock_create_labels:
                mock_create_labels.return_value = df.copy()  # Return same DataFrame

                # Should succeed with valid DataFrame
                version_id = migrate_legacy_dataset(df, manager)
                assert version_id.startswith("migrated_v2_")

                # Verify validation_passed flag in metadata
                info = manager.get_version_info(version_id)
                assert info['validation_passed'] is True

    def test_migrate_legacy_dataset_invalid_input(self):
        """Test migrate_legacy_dataset with invalid input DataFrame."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            # Create invalid DataFrame
            df = pd.DataFrame({
                'open': [100 + i for i in range(10)],  # Missing timestamp
                'high': [105 + i for i in range(10)],
                'low': [95 + i for i in range(10)],
                'close': [102 + i for i in range(10)],
                'volume': [1000 + i*10 for i in range(10)]
            })

            # Should raise DataValidationError
            with pytest.raises(DataValidationError):
                migrate_legacy_dataset(df, manager)


class TestVersionNameSanitization:
    """Test version name sanitization in DatasetVersionManager."""

    def test_sanitize_version_name_valid_alphanumeric(self):
        """Test sanitizing valid alphanumeric version names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            # Should accept valid names
            assert manager._sanitize_version_name("version1") == "version1"
            assert manager._sanitize_version_name("test_version") == "test_version"
            assert manager._sanitize_version_name("my-version-2") == "my-version-2"
            assert manager._sanitize_version_name("Version123") == "Version123"

    def test_sanitize_version_name_path_traversal_dots(self):
        """Test blocking path traversal with .. patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            with pytest.raises(VersionPathTraversalError, match="Path traversal detected"):
                manager._sanitize_version_name("../../etc/passwd")

            with pytest.raises(VersionPathTraversalError, match="Path traversal detected"):
                manager._sanitize_version_name("../../../hack")

            with pytest.raises(VersionPathTraversalError, match="Path traversal detected"):
                manager._sanitize_version_name("version/../escape")

    def test_sanitize_version_name_path_separators(self):
        """Test blocking path separators."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            with pytest.raises(VersionPathTraversalError, match="Path separators not allowed"):
                manager._sanitize_version_name("version/hack")

            with pytest.raises(VersionPathTraversalError, match="Path separators not allowed"):
                manager._sanitize_version_name("version\\hack")

            with pytest.raises(VersionPathTraversalError, match="Path separators not allowed"):
                manager._sanitize_version_name("path/to/file")

    def test_sanitize_version_name_absolute_paths(self):
        """Test blocking absolute paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            with pytest.raises(VersionPathTraversalError, match="Absolute path detected"):
                manager._sanitize_version_name("/etc/passwd")

            with pytest.raises(VersionPathTraversalError, match="Absolute path detected"):
                manager._sanitize_version_name("\\Windows\\System32")

            with pytest.raises(VersionPathTraversalError, match="Absolute path detected"):
                manager._sanitize_version_name("C:\\Windows\\System32")

    def test_sanitize_version_name_invalid_characters(self):
        """Test blocking invalid characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            with pytest.raises(VersionPathTraversalError, match="Invalid characters"):
                manager._sanitize_version_name("version@hack")

            with pytest.raises(VersionPathTraversalError, match="Invalid characters"):
                manager._sanitize_version_name("version hack")  # space

            with pytest.raises(VersionPathTraversalError, match="Invalid characters"):
                manager._sanitize_version_name("version.hack")  # dot

            with pytest.raises(VersionPathTraversalError, match="Invalid characters"):
                manager._sanitize_version_name("version:hack")  # colon

    def test_sanitize_version_name_empty_and_invalid_types(self):
        """Test handling empty strings and invalid types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            with pytest.raises(VersionPathTraversalError, match="must be a non-empty string"):
                manager._sanitize_version_name("")

            with pytest.raises(VersionPathTraversalError, match="must be a non-empty string"):
                manager._sanitize_version_name(None)

            with pytest.raises(VersionPathTraversalError, match="must be a non-empty string"):
                manager._sanitize_version_name(123)  # integer

    def test_sanitize_version_name_too_long(self):
        """Test blocking overly long version names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            long_name = "a" * 101  # 101 characters, exceeds limit
            with pytest.raises(VersionPathTraversalError, match="too long"):
                manager._sanitize_version_name(long_name)

    def test_create_version_with_malicious_name(self):
        """Test create_version rejects malicious version names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            # Create valid DataFrame
            df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=5, freq='1H'),
                'open': [100, 102, 104, 106, 108],
                'high': [105, 107, 109, 111, 113],
                'low': [95, 97, 99, 101, 103],
                'close': [102, 104, 106, 108, 110],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })

            # Test various malicious names
            malicious_names = [
                "../../etc/passwd",
                "../../../hack",
                "version/../escape",
                "/absolute/path",
                "C:\\Windows\\System32",
                "version@hack",
                "version hack",
                ""  # empty string
            ]

            for malicious_name in malicious_names:
                with pytest.raises(VersionPathTraversalError):
                    manager.create_version(df, malicious_name, "Test version")

    def test_create_version_with_valid_name(self):
        """Test create_version accepts valid version names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            # Create valid DataFrame
            df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=5, freq='1H'),
                'open': [100, 102, 104, 106, 108],
                'high': [105, 107, 109, 111, 113],
                'low': [95, 97, 99, 101, 103],
                'close': [102, 104, 106, 108, 110],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })

            # Test valid names
            valid_names = ["version1", "test_version", "my-version-2", "Version123"]

            for valid_name in valid_names:
                version_id = manager.create_version(df, valid_name, "Test version")
                assert version_id.startswith(valid_name + "_")
                assert valid_name in version_id


class TestConfigurationValidation:
    """Test configuration parameter validation in HistoricalDataLoader."""

    def test_validate_data_directory_valid_names(self):
        """Test validating valid data directory names."""
        config = {'backtesting': {'data_dir': 'historical_data'}}
        data_fetcher = MagicMock()
        loader = HistoricalDataLoader(config, data_fetcher)

        # Should accept valid names
        assert loader._validate_data_directory("historical_data") == "historical_data"
        assert loader._validate_data_directory("my_data_dir") == "my_data_dir"
        assert loader._validate_data_directory("data-2023") == "data-2023"
        assert loader._validate_data_directory("DataDir123") == "DataDir123"

    def test_validate_data_directory_path_traversal_dots(self):
        """Test blocking path traversal with .. patterns."""
        config = {'backtesting': {'data_dir': 'historical_data'}}
        data_fetcher = MagicMock()
        loader = HistoricalDataLoader(config, data_fetcher)

        with pytest.raises(ConfigurationError, match="Path traversal detected"):
            loader._validate_data_directory("../../etc/passwd")

        with pytest.raises(ConfigurationError, match="Path traversal detected"):
            loader._validate_data_directory("../../../hack")

        with pytest.raises(ConfigurationError, match="Path traversal detected"):
            loader._validate_data_directory("data/../escape")

    def test_validate_data_directory_path_separators(self):
        """Test blocking path separators."""
        config = {'backtesting': {'data_dir': 'historical_data'}}
        data_fetcher = MagicMock()
        loader = HistoricalDataLoader(config, data_fetcher)

        with pytest.raises(ConfigurationError, match="Path separators not allowed"):
            loader._validate_data_directory("data/hack")

        with pytest.raises(ConfigurationError, match="Path separators not allowed"):
            loader._validate_data_directory("data\\hack")

        with pytest.raises(ConfigurationError, match="Path separators not allowed"):
            loader._validate_data_directory("path/to/file")

    def test_validate_data_directory_absolute_paths(self):
        """Test blocking absolute paths."""
        config = {'backtesting': {'data_dir': 'historical_data'}}
        data_fetcher = MagicMock()
        loader = HistoricalDataLoader(config, data_fetcher)

        with pytest.raises(ConfigurationError, match="Absolute path detected"):
            loader._validate_data_directory("/etc/passwd")

        with pytest.raises(ConfigurationError, match="Absolute path detected"):
            loader._validate_data_directory("\\Windows\\System32")

        with pytest.raises(ConfigurationError, match="Absolute path detected"):
            loader._validate_data_directory("C:\\Windows\\System32")

    def test_validate_data_directory_invalid_characters(self):
        """Test blocking invalid characters."""
        config = {'backtesting': {'data_dir': 'historical_data'}}
        data_fetcher = MagicMock()
        loader = HistoricalDataLoader(config, data_fetcher)

        with pytest.raises(ConfigurationError, match="Invalid characters"):
            loader._validate_data_directory("data@hack")

        with pytest.raises(ConfigurationError, match="Invalid characters"):
            loader._validate_data_directory("data hack")  # space

        with pytest.raises(ConfigurationError, match="Invalid characters"):
            loader._validate_data_directory("data.hack")  # dot

        with pytest.raises(ConfigurationError, match="Invalid characters"):
            loader._validate_data_directory("data:hack")  # colon

    def test_validate_data_directory_empty_and_invalid_types(self):
        """Test handling empty strings and invalid types."""
        config = {'backtesting': {'data_dir': 'historical_data'}}
        data_fetcher = MagicMock()
        loader = HistoricalDataLoader(config, data_fetcher)

        with pytest.raises(ConfigurationError, match="must be a non-empty string"):
            loader._validate_data_directory("")

        with pytest.raises(ConfigurationError, match="must be a non-empty string"):
            loader._validate_data_directory(None)

        with pytest.raises(ConfigurationError, match="must be a non-empty string"):
            loader._validate_data_directory(123)  # integer

    def test_validate_data_directory_too_long(self):
        """Test blocking overly long data directory names."""
        config = {'backtesting': {'data_dir': 'historical_data'}}
        data_fetcher = MagicMock()
        loader = HistoricalDataLoader(config, data_fetcher)

        long_name = "a" * 101  # 101 characters, exceeds limit
        with pytest.raises(ConfigurationError, match="too long"):
            loader._validate_data_directory(long_name)

    def test_setup_data_directory_with_malicious_config(self):
        """Test _setup_data_directory rejects malicious data_dir values."""
        data_fetcher = MagicMock()

        # Test various malicious configurations
        malicious_configs = [
            {'backtesting': {'data_dir': '../../etc/passwd'}},
            {'backtesting': {'data_dir': '../../../hack'}},
            {'backtesting': {'data_dir': 'data/../escape'}},
            {'backtesting': {'data_dir': '/absolute/path'}},
            {'backtesting': {'data_dir': 'C:\\Windows\\System32'}},
            {'backtesting': {'data_dir': 'data@hack'}},
            {'backtesting': {'data_dir': 'data hack'}},
            {'backtesting': {'data_dir': ''}}  # empty string
        ]

        for config in malicious_configs:
            with pytest.raises(ConfigurationError):
                HistoricalDataLoader(config, data_fetcher)

    def test_setup_data_directory_with_valid_config(self):
        """Test _setup_data_directory accepts valid data_dir values."""
        data_fetcher = MagicMock()

        # Test valid configurations
        valid_configs = [
            {'backtesting': {'data_dir': 'historical_data'}},
            {'backtesting': {'data_dir': 'my_data_dir'}},
            {'backtesting': {'data_dir': 'data-2023'}},
            {'backtesting': {'data_dir': 'DataDir123'}}
        ]

        for config in valid_configs:
            loader = HistoricalDataLoader(config, data_fetcher)
            # Should create directory under data/historical/
            expected_base = os.path.join(os.getcwd(), 'data', 'historical')
            assert loader.data_dir.startswith(expected_base)
            assert config['backtesting']['data_dir'] in loader.data_dir


class TestIntegrationSecurity:
    """Integration tests for security features."""

    def test_full_data_pipeline_security(self):
        """Test the full data pipeline with security features."""
        # Test DataFetcher with safe cache path
        config = {
            'name': 'binance',
            'cache_enabled': True,
            'cache_dir': 'safe_cache_dir'
        }

        fetcher = DataFetcher(config)
        assert 'safe_cache_dir' in fetcher.cache_dir
        assert fetcher.cache_dir.startswith(os.path.join(os.getcwd(), 'data', 'cache'))

        # Test DatasetVersionManager with validation
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetVersionManager(temp_dir)

            # Create valid DataFrame
            df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=5, freq='1H'),
                'open': [100, 102, 104, 106, 108],
                'high': [105, 107, 109, 111, 113],
                'low': [95, 97, 99, 101, 103],
                'close': [102, 104, 106, 108, 110],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })

            # Test successful creation
            version_id = manager.create_version(df, "integration_test", "Integration test")
            assert version_id is not None

            # Test loading and validation
            loaded_df = manager.load_version(version_id)
            assert loaded_df is not None
            assert len(loaded_df) == 5

            # Verify data integrity
            pd.testing.assert_frame_equal(df.reset_index(drop=True), loaded_df.reset_index(drop=True))

    def test_error_handling_and_logging(self):
        """Test that errors are properly logged and handled."""
        with patch('data.data_fetcher.logger') as mock_logger:
            # Test path traversal error logging
            config = {'cache_enabled': True, 'cache_dir': '../../../etc/passwd'}
            fetcher = DataFetcher(config)

            with pytest.raises(PathTraversalError):
                fetcher._initialize_exchange()

            # Verify error was logged
            mock_logger.error.assert_called()

        with patch('data.dataset_versioning.logger') as mock_logger:
            # Test DataFrame validation error logging
            df = pd.DataFrame()  # Empty DataFrame

            with pytest.raises(DataValidationError):
                validate_dataframe(df)

            # Verify error was logged
            mock_logger.error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
