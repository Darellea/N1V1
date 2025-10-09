"""
tests/test_data_module_refactoring.py

Comprehensive unit tests for refactored data module functionality.
Tests data validation, caching, pagination, and versioning logic.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from data.data_fetcher import CacheLoadError, DataFetcher, PathTraversalError
from data.dataset_versioning import DatasetVersionManager
from data.historical_loader import ConfigurationError, HistoricalDataLoader


class TestDataValidation:
    """Test data validation functions across data modules."""

    def test_validate_fetch_parameters_valid(self):
        """Test _validate_fetch_parameters with valid inputs."""
        config = {"backtesting": {}}
        data_fetcher = Mock()
        loader = HistoricalDataLoader(config, data_fetcher)

        result = loader._validate_fetch_parameters(
            symbol="BTC/USDT",
            start_date="2023-01-01",
            end_date="2023-01-02",
            timeframe="1h",
        )

        assert result is True

    def test_validate_fetch_parameters_invalid_symbol(self):
        """Test _validate_fetch_parameters with invalid symbol."""
        config = {"backtesting": {}}
        data_fetcher = Mock()
        loader = HistoricalDataLoader(config, data_fetcher)

        result = loader._validate_fetch_parameters(
            symbol="", start_date="2023-01-01", end_date="2023-01-02", timeframe="1h"
        )

        assert result is False

    def test_validate_fetch_parameters_invalid_timeframe(self):
        """Test _validate_fetch_parameters with invalid timeframe."""
        config = {"backtesting": {}}
        data_fetcher = Mock()
        loader = HistoricalDataLoader(config, data_fetcher)

        result = loader._validate_fetch_parameters(
            symbol="BTC/USDT",
            start_date="2023-01-01",
            end_date="2023-01-02",
            timeframe="invalid",
        )

        assert result is False

    def test_validate_fetch_parameters_date_order(self):
        """Test _validate_fetch_parameters with invalid date order."""
        config = {"backtesting": {}}
        data_fetcher = Mock()
        loader = HistoricalDataLoader(config, data_fetcher)

        result = loader._validate_fetch_parameters(
            symbol="BTC/USDT",
            start_date="2023-01-02",
            end_date="2023-01-01",
            timeframe="1h",
        )

        assert result is False

    def test_validate_data_empty_dataframe(self):
        """Test _validate_data with empty DataFrame."""
        config = {"backtesting": {}}
        data_fetcher = Mock()
        loader = HistoricalDataLoader(config, data_fetcher)

        df = pd.DataFrame()
        result = loader._validate_data(df, "1h")

        assert result is False

    def test_validate_data_missing_columns(self):
        """Test _validate_data with missing required columns."""
        config = {"backtesting": {}}
        data_fetcher = Mock()
        loader = HistoricalDataLoader(config, data_fetcher)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5, freq="1h"),
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                # Missing 'low', 'close', 'volume'
            }
        )
        df.set_index("timestamp", inplace=True)

        result = loader._validate_data(df, "1h")

        assert result is False

    def test_validate_data_valid(self):
        """Test _validate_data with valid DataFrame."""
        config = {"backtesting": {}}
        data_fetcher = Mock()
        loader = HistoricalDataLoader(config, data_fetcher)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5, freq="1h"),
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [102, 103, 104, 105, 106],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )
        df.set_index("timestamp", inplace=True)

        result = loader._validate_data(df, "1h")

        assert result is True


class TestCaching:
    """Test caching functionality (_save_to_cache, _load_from_cache)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"cache_enabled": True, "cache_dir": "test_cache"}
        self.data_fetcher = DataFetcher(self.config)

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up cache directory
        if os.path.exists(self.data_fetcher.cache_dir):
            import shutil

            shutil.rmtree(self.data_fetcher.cache_dir)

    def test_save_to_cache_success(self):
        """Test successful cache save operation."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="1h"),
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )
        df.set_index("timestamp", inplace=True)

        cache_key = "test_key"

        # This should not raise an exception
        result = self.data_fetcher._save_to_cache(cache_key, df)

        # Verify cache file was created
        cache_path = os.path.join(
            self.data_fetcher._cache_dir_path, f"{cache_key}.json"
        )
        assert os.path.exists(cache_path)

        # Verify cache file contents
        with open(cache_path, "r") as f:
            cached_data = json.load(f)

        assert "data" in cached_data
        assert len(cached_data["data"]) == 3

    def test_load_from_cache_success(self):
        """Test successful cache load operation."""
        # First save some data
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="1h"),
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )
        df.set_index("timestamp", inplace=True)

        cache_key = "test_load_key"
        self.data_fetcher._save_to_cache(cache_key, df)

        # Now load the data
        loaded_df = self.data_fetcher._load_from_cache(cache_key)

        assert loaded_df is not None
        assert not loaded_df.empty
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["open", "high", "low", "close", "volume"]

    def test_load_from_cache_nonexistent(self):
        """Test cache load with nonexistent key."""
        result = self.data_fetcher._load_from_cache("nonexistent_key")

        assert result is None

    def test_load_from_cache_expired(self):
        """Test cache load with expired cache file."""
        # Create a cache file with old timestamp
        cache_key = "expired_key"
        cache_path = os.path.join(self.data_fetcher.cache_dir, f"{cache_key}.json")

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Create cache file with timestamp from 2 days ago
        old_timestamp = int((datetime.now() - timedelta(days=2)).timestamp() * 1000)
        cache_data = {
            "timestamp": old_timestamp,
            "data": [
                {
                    "timestamp": "2023-01-01T00:00:00",
                    "open": 100,
                    "high": 105,
                    "low": 95,
                    "close": 102,
                    "volume": 1000,
                }
            ],
        }

        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        # Try to load - should return None due to expiration
        result = self.data_fetcher._load_from_cache(cache_key)

        assert result is None

    def test_cache_key_generation(self):
        """Test cache key generation."""
        symbol = "BTC/USDT"
        timeframe = "1h"
        limit = 1000
        since = 1640995200000  # 2022-01-01 00:00:00 UTC

        key = self.data_fetcher._get_cache_key(symbol, timeframe, limit, since)

        # Key should be deterministic
        expected_key = self.data_fetcher._get_cache_key(symbol, timeframe, limit, since)
        assert key == expected_key

        # Different parameters should produce different keys
        different_key = self.data_fetcher._get_cache_key(symbol, "4h", limit, since)
        assert key != different_key


class TestPagination:
    """Test data fetching and pagination logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"backtesting": {}}
        self.data_fetcher = AsyncMock()
        self.loader = HistoricalDataLoader(self.config, self.data_fetcher)

    def test_calculate_pagination_params(self):
        """Test pagination parameter calculation."""
        start_date = "2023-01-01"
        end_date = "2023-01-08"  # 7 days
        timeframe = "1h"

        (
            start_dt,
            end_dt,
            delta,
            estimated_requests,
        ) = self.loader._calculate_pagination_params(start_date, end_date, timeframe)

        assert start_dt == pd.Timestamp("2023-01-01")
        assert end_dt == pd.Timestamp("2023-01-08")
        assert delta == timedelta(hours=1)
        assert estimated_requests > 0

    def test_execute_pagination_loop_empty_data(self):
        """Test pagination loop with no data returned."""
        # Mock data fetcher to return empty data
        self.data_fetcher.get_historical_data.return_value = pd.DataFrame()

        result = asyncio.run(
            self.loader._execute_pagination_loop(
                symbol="BTC/USDT",
                timeframe="1h",
                start_dt=pd.Timestamp("2023-01-01"),
                end_dt=pd.Timestamp("2023-01-02"),
                delta=timedelta(hours=1),
                estimated_requests=1,
            )
        )

        assert result == []

    def test_execute_pagination_loop_with_data(self):
        """Test pagination loop with data returned."""
        # Create mock data for first call
        mock_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=2, freq="1h"),
                "open": [100, 101],
                "high": [105, 106],
                "low": [95, 96],
                "close": [102, 103],
                "volume": [1000, 1100],
            }
        )
        mock_data.set_index("timestamp", inplace=True)

        # Mock to return data for first call, empty for subsequent calls
        self.data_fetcher.get_historical_data.side_effect = (
            lambda *args, **kwargs: mock_data
            if self.data_fetcher.get_historical_data.call_count == 1
            else pd.DataFrame()
        )

        result = asyncio.run(
            self.loader._execute_pagination_loop(
                symbol="BTC/USDT",
                timeframe="1h",
                start_dt=pd.Timestamp("2023-01-01"),
                end_dt=pd.Timestamp("2023-01-01 02:00:00"),
                delta=timedelta(hours=1),
                estimated_requests=1,
            )
        )

        assert len(result) == 1
        assert not result[0].empty

    def test_process_paginated_data_empty(self):
        """Test processing empty paginated data."""
        result = self.loader._process_paginated_data([], "BTC/USDT")

        assert result is None

    def test_process_paginated_data_success(self):
        """Test processing paginated data successfully."""
        # Create test data
        df1 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="1h"),
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )
        df1.set_index("timestamp", inplace=True)

        df2 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01 03:00:00", periods=3, freq="1h"),
                "open": [103, 104, 105],
                "high": [108, 109, 110],
                "low": [98, 99, 100],
                "close": [105, 106, 107],
                "volume": [1300, 1400, 1500],
            }
        )
        df2.set_index("timestamp", inplace=True)

        result = self.loader._process_paginated_data([df1, df2], "BTC/USDT")

        assert result is not None
        assert not result.empty
        assert len(result) == 6  # 3 + 3 rows

    def test_infinite_loop_detection(self):
        """Test infinite loop detection in pagination."""
        current_start = pd.Timestamp("2023-01-01")
        last_current_start = pd.Timestamp("2023-01-01")
        consecutive_same_start = 2  # One less than threshold

        should_break, new_count = self.loader._detect_infinite_loop(
            current_start, last_current_start, consecutive_same_start, "BTC/USDT"
        )

        assert should_break is False
        assert new_count == 3

        # Test with count at threshold
        should_break, new_count = self.loader._detect_infinite_loop(
            current_start, last_current_start, 3, "BTC/USDT"
        )

        assert should_break is True
        assert new_count == 4


class TestVersioning:
    """Test dataset versioning functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"versioning": {"base_dir": "test_versions", "max_versions": 10}}
        self.version_manager = DatasetVersionManager(self.config, legacy_mode=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up version directory
        if os.path.exists("test_versions"):
            import shutil

            shutil.rmtree("test_versions")

    def test_create_version_success(self):
        """Test successful version creation."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="1h"),
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )
        df.set_index("timestamp", inplace=True)

        version_name = "test_version_1"
        description = "Test version"

        # This should not raise an exception
        result = self.version_manager.create_version(df, version_name, description)

        assert result is True

        # Verify version directory was created
        version_dir = os.path.join("test_versions", version_name)
        assert os.path.exists(version_dir)

    def test_create_version_invalid_name(self):
        """Test version creation with invalid name."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="1h"),
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )
        df.set_index("timestamp", inplace=True)

        # Invalid version name with path traversal
        version_name = "../../../invalid_version"

        with pytest.raises(Exception):  # Should raise path traversal error
            self.version_manager.create_version(df, version_name, "Test")

    def test_migrate_legacy_dataset(self):
        """Test migration of legacy dataset."""
        # Create a legacy dataset file
        legacy_data = {
            "data": [
                {
                    "timestamp": "2023-01-01T00:00:00",
                    "open": 100,
                    "high": 105,
                    "low": 95,
                    "close": 102,
                    "volume": 1000,
                },
                {
                    "timestamp": "2023-01-01T01:00:00",
                    "open": 102,
                    "high": 107,
                    "low": 97,
                    "close": 104,
                    "volume": 1100,
                },
            ],
            "metadata": {"source": "legacy_system", "created": "2023-01-01"},
        }

        os.makedirs("test_versions", exist_ok=True)
        legacy_path = os.path.join("test_versions", "legacy_dataset.json")

        with open(legacy_path, "w") as f:
            json.dump(legacy_data, f)

        # This should not raise an exception
        result = self.version_manager.migrate_legacy_dataset(
            legacy_path, "migrated_version", "Migrated from legacy"
        )

        assert result is True

        # Verify migrated version exists
        migrated_dir = os.path.join("test_versions", "migrated_version")
        assert os.path.exists(migrated_dir)


class TestPathTraversal:
    """Test path traversal prevention."""

    def test_sanitize_cache_path_valid(self):
        """Test cache path sanitization with valid path."""
        config = {"cache_enabled": True, "cache_dir": "valid_cache"}
        data_fetcher = DataFetcher(config)

        result = data_fetcher._sanitize_cache_path("valid_cache")

        assert "valid_cache" in result
        assert ".." not in result

    def test_sanitize_cache_path_traversal(self):
        """Test cache path sanitization with path traversal attempt."""
        config = {"cache_enabled": True, "cache_dir": "../../../etc/passwd"}

        # Should raise PathTraversalError immediately
        with pytest.raises(PathTraversalError, match="Invalid cache directory path"):
            DataFetcher(config)

    def test_validate_data_directory_valid(self):
        """Test data directory validation with valid path."""
        config = {"backtesting": {}}
        data_fetcher = Mock()
        loader = HistoricalDataLoader(config, data_fetcher)

        result = loader._validate_data_directory("valid_directory")

        assert result == "valid_directory"

    def test_validate_data_directory_invalid_chars(self):
        """Test data directory validation with invalid characters."""
        config = {"backtesting": {}}
        data_fetcher = Mock()
        loader = HistoricalDataLoader(config, data_fetcher)

        with pytest.raises(ConfigurationError):
            loader._validate_data_directory("invalid/directory")

    def test_validate_data_directory_path_traversal(self):
        """Test data directory validation with path traversal."""
        config = {"backtesting": {}}
        data_fetcher = Mock()
        loader = HistoricalDataLoader(config, data_fetcher)

        with pytest.raises(ConfigurationError):
            loader._validate_data_directory("../../../etc")


class TestBackwardCompatibility:
    """Test that refactored code maintains backward compatibility."""

    def test_fetch_complete_history_still_works(self):
        """Test that _fetch_complete_history still works after refactoring."""
        config = {"backtesting": {}}
        data_fetcher = AsyncMock()
        loader = HistoricalDataLoader(config, data_fetcher)

        # Mock the data fetcher to return data once, then empty for all subsequent calls
        mock_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="1h"),
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )
        mock_df.set_index("timestamp", inplace=True)

        # Return data for first call, empty for all subsequent calls
        data_fetcher.get_historical_data.side_effect = (
            lambda *args, **kwargs: mock_df
            if data_fetcher.get_historical_data.call_count == 1
            else pd.DataFrame()
        )

        result = asyncio.run(
            loader._fetch_complete_history("BTC/USDT", "2023-01-01", "2023-01-02", "1h")
        )

        assert result is not None
        assert not result.empty
        assert len(result) == 3

        # Verify data fetcher was called
        data_fetcher.get_historical_data.assert_called()

    def test_all_public_methods_still_exist(self):
        """Test that all public methods still exist and work."""
        config = {"backtesting": {}}
        data_fetcher = AsyncMock()
        loader = HistoricalDataLoader(config, data_fetcher)

        # Test that all expected public methods exist
        assert hasattr(loader, "load_historical_data")
        assert hasattr(loader, "resample_data")
        assert hasattr(loader, "get_available_pairs")
        assert hasattr(loader, "shutdown")

        # Test that methods can be called without error
        result = asyncio.run(
            loader.load_historical_data([], "2023-01-01", "2023-01-02", "1h")
        )
        assert isinstance(result, dict)

        result = asyncio.run(loader.resample_data({}, "4h"))
        assert isinstance(result, dict)

        result = loader.get_available_pairs()
        assert isinstance(result, list)

        # Shutdown should not raise exception
        asyncio.run(loader.shutdown())


if __name__ == "__main__":
    pytest.main([__file__])
